from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from queue import Empty, Full, Queue
from time import perf_counter
import sys
import threading
import traceback

import cupy as cp
import cupyx
import numpy as np
from cupyx.profiler import time_range

from cupy_naive.config import ExecutionMode, Params
from cupy_naive.cupy_naive import (
    BenchmarkStats,
    DummyGilThread,
    PercentileClipDisplay2D,
    SlidingMean2D,
    clear_cupy_pools,
    doppler_bin_range,
    make_quadratic_phase,
)
from cupy_naive.io import InputInfo, bytes_per_frame, cycle_batches


@dataclass(frozen=True)
class ThreadedRuntimeConfig:
    """Host scheduling parameters for the threaded benchmark."""

    queue_depth: int = 2
    queue_put_policy: str = "timed_put"
    queue_put_timeout_s: float = 0.0005
    gil_switch_interval_s: float | None = None


@dataclass(frozen=True)
class InputBatch:
    sequence: int
    host_batch: np.ndarray


@dataclass(frozen=True)
class UploadedBatch:
    sequence: int
    raw_batch_device: cp.ndarray


@dataclass(frozen=True)
class PowerBatch:
    sequence: int
    batch_power: cp.ndarray


@dataclass(frozen=True)
class DisplayBatch:
    sequence: int
    display_image: cp.ndarray


@dataclass(frozen=True)
class OutputBatch:
    sequence: int
    image: np.ndarray


@dataclass(frozen=True)
class WorkerFailure:
    stage: str
    details: str


_STOP = object()


class ThreadedPowerDopplerWorkers:
    """Own the four long-lived worker threads and their queue handoffs.

    Every CUDA-producing stage synchronizes its own stream before enqueueing to
    the next stage. Queue output operations all pass through `_queue_put_once`,
    so `queue_put_policy` applies consistently to main-to-worker and
    worker-to-worker handoffs.
    """

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
        runtime: ThreadedRuntimeConfig,
    ) -> None:
        if runtime.queue_depth <= 0:
            raise ValueError(f"queue_depth must be positive, got {runtime.queue_depth}.")
        if runtime.queue_put_policy not in {"timed_put", "nowait"}:
            raise ValueError(
                "queue_put_policy must be 'timed_put' or 'nowait', got "
                f"{runtime.queue_put_policy!r}."
            )
        if runtime.queue_put_timeout_s <= 0.0:
            raise ValueError(
                "queue_put_timeout_s must be positive, got "
                f"{runtime.queue_put_timeout_s}."
            )

        self.info = info
        self.params = params
        self.mode = mode
        self.runtime = runtime

        self.height = info.height
        self.width = info.width
        self.batch_frames = params.batch_frames
        self.device_id = cp.cuda.Device().id

        self.h2d_input_queue: Queue[InputBatch | object] = Queue(
            maxsize=runtime.queue_depth
        )
        self.fft_input_queue: Queue[UploadedBatch | object] = Queue(
            maxsize=runtime.queue_depth
        )
        self.post_input_queue: Queue[PowerBatch | object] = Queue(
            maxsize=runtime.queue_depth
        )
        self.d2h_input_queue: Queue[DisplayBatch | object] = Queue(
            maxsize=runtime.queue_depth
        )
        # Terminal result delivery should not backpressure the GPU pipeline.
        # Pipeline memory is already bounded by the stage queues above.
        self.output_queue: Queue[OutputBatch] = Queue()
        self.failure_queue: Queue[WorkerFailure] = Queue()

        self.raw_buffer_pool: Queue[cp.ndarray] | None = None
        if mode.preallocate_work_buffers:
            raw_buffer_count = runtime.queue_depth + 2
            self.raw_buffer_pool = Queue(maxsize=raw_buffer_count)
            for _ in range(raw_buffer_count):
                self.raw_buffer_pool.put(
                    cp.empty(
                        (self.batch_frames, self.height, self.width),
                        dtype=params.acquisition_dtype,
                    )
                )

        self._threads = [
            threading.Thread(
                name="cupy-threaded-h2d",
                target=self._run_worker,
                args=("h2d", self._h2d_worker),
                daemon=True,
            ),
            threading.Thread(
                name="cupy-threaded-fft",
                target=self._run_worker,
                args=("fft", self._fft_worker),
                daemon=True,
            ),
            threading.Thread(
                name="cupy-threaded-post",
                target=self._run_worker,
                args=("post", self._post_worker),
                daemon=True,
            ),
            threading.Thread(
                name="cupy-threaded-d2h",
                target=self._run_worker,
                args=("d2h", self._d2h_worker),
                daemon=True,
            ),
        ]

    def start(self) -> None:
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        self._stop_h2d_worker()
        for thread in self._threads:
            while thread.is_alive():
                thread.join(timeout=0.05)
                self._drain_completion_queues()
        self._raise_worker_failure_if_any()

    def _stop_h2d_worker(self) -> None:
        while True:
            self._raise_worker_failure_if_any()
            if self._queue_put_once(
                self.h2d_input_queue,
                _STOP,
                "main -> H2D stop",
                color_id=80,
            ):
                return
            self._drain_completion_queues()

    def _drain_completion_queues(self) -> None:
        self._drain_queue(self.output_queue)

    @staticmethod
    def _drain_queue(queue: Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def submit(self, item: InputBatch) -> None:
        self._raise_worker_failure_if_any()
        self._queue_put(
            self.h2d_input_queue,
            item,
            "main -> H2D",
            color_id=81,
        )
        self._raise_worker_failure_if_any()

    def try_submit(self, item: InputBatch) -> bool:
        self._raise_worker_failure_if_any()
        submitted = self._queue_put_once(
            self.h2d_input_queue,
            item,
            "main -> H2D",
            color_id=101,
        )
        self._raise_worker_failure_if_any()
        return submitted

    def get_output_timeout(self, timeout_s: float) -> OutputBatch | None:
        self._raise_worker_failure_if_any()
        with time_range("wait sink output queue input", color_id=105):
            try:
                output = self.output_queue.get(timeout=timeout_s)
            except Empty:
                return None

        self._raise_worker_failure_if_any()
        return output

    def raise_worker_failure_if_any(self) -> None:
        self._raise_worker_failure_if_any()

    def _raise_worker_failure_if_any(self) -> None:
        try:
            failure = self.failure_queue.get_nowait()
        except Empty:
            return

        raise RuntimeError(f"{failure.stage} worker failed:\n{failure.details}")

    def _run_worker(self, stage: str, worker: Callable[[], None]) -> None:
        try:
            with cp.cuda.Device(self.device_id):
                worker()
        except BaseException:
            self.failure_queue.put(
                WorkerFailure(stage=stage, details=traceback.format_exc())
            )

    def _queue_get(self, queue: Queue, label: str, color_id: int) -> object:
        with time_range(f"wait {label} input", color_id=color_id):
            return queue.get()

    def _queue_put(
        self,
        queue: Queue,
        item: object,
        label: str,
        color_id: int,
    ) -> None:
        while not self._queue_put_once(queue, item, label, color_id):
            self._raise_worker_failure_if_any()

    def _queue_put_once(
        self,
        queue: Queue,
        item: object,
        label: str,
        color_id: int,
    ) -> bool:
        range_name = (
            f"try {label} output slot"
            if self.runtime.queue_put_policy == "nowait"
            else f"wait {label} output slot"
        )
        with time_range(range_name, color_id=color_id):
            try:
                if self.runtime.queue_put_policy == "nowait":
                    queue.put_nowait(item)
                else:
                    queue.put(item, timeout=self.runtime.queue_put_timeout_s)
            except Full:
                return False
            return True

    def _get_raw_buffer(self) -> cp.ndarray:
        if self.raw_buffer_pool is None:
            return cp.empty(
                (self.batch_frames, self.height, self.width),
                dtype=self.params.acquisition_dtype,
            )

        return self._queue_get(
            self.raw_buffer_pool,
            "H2D raw buffer pool",
            color_id=83,
        )

    def _release_raw_buffer(self, raw_batch_device: cp.ndarray) -> None:
        if self.raw_buffer_pool is not None:
            self._queue_put(
                self.raw_buffer_pool,
                raw_batch_device,
                "FFT raw buffer pool",
                color_id=84,
            )

    def _h2d_worker(self) -> None:
        stream = cp.cuda.Stream(non_blocking=True)

        while True:
            item = self._queue_get(self.h2d_input_queue, "H2D", color_id=85)
            if item is _STOP:
                self._queue_put(
                    self.fft_input_queue,
                    _STOP,
                    "H2D -> FFT stop",
                    color_id=86,
                )
                return
            if not isinstance(item, InputBatch):
                raise TypeError(f"Unexpected H2D item: {type(item)!r}")

            raw_batch_device = self._get_raw_buffer()
            with stream, time_range("threaded H2D upload", color_id=51):
                raw_batch_device.set(item.host_batch, stream=stream)

            with time_range("threaded H2D sync before enqueue", color_id=52):
                stream.synchronize()

            self._queue_put(
                self.fft_input_queue,
                UploadedBatch(
                    sequence=item.sequence,
                    raw_batch_device=raw_batch_device,
                ),
                "H2D -> FFT",
                color_id=87,
            )

    def _fft_worker(self) -> None:
        stream = cp.cuda.Stream(non_blocking=True)

        k0, k1 = doppler_bin_range(
            window_size=self.params.batch_frames,
            sample_rate_hz=self.params.sample_rate_hz,
            doppler_low_hz=self.params.doppler_low_hz,
            doppler_high_hz=self.params.doppler_high_hz,
        )

        with stream, time_range("threaded init FFT tensors", color_id=53):
            quadratic_phase = (
                make_quadratic_phase(self.height, self.width, self.params)
                if self.mode.precompute_static_tensors
                else None
            )
            real_batch_device = (
                cp.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=self.params.real_dtype,
                )
                if self.mode.preallocate_work_buffers
                else None
            )

        stream.synchronize()

        while True:
            item = self._queue_get(self.fft_input_queue, "FFT", color_id=88)
            if item is _STOP:
                self._queue_put(
                    self.post_input_queue,
                    _STOP,
                    "FFT -> post stop",
                    color_id=89,
                )
                return
            if not isinstance(item, UploadedBatch):
                raise TypeError(f"Unexpected FFT item: {type(item)!r}")

            with stream, time_range("threaded FFT batch", color_id=54):
                if real_batch_device is None:
                    real_batch = item.raw_batch_device.astype(
                        self.params.real_dtype,
                        copy=False,
                    )
                else:
                    cp.copyto(
                        real_batch_device,
                        item.raw_batch_device,
                        casting="unsafe",
                    )
                    real_batch = real_batch_device

                temporal_spectrum = cp.fft.rfft(real_batch, axis=0)[k0:k1]

                phase = quadratic_phase
                if phase is None:
                    phase = make_quadratic_phase(self.height, self.width, self.params)

                propagated = cp.fft.fft2(
                    temporal_spectrum * phase,
                    axes=(-2, -1),
                )
                batch_power = (cp.abs(propagated) ** 2).sum(axis=0)

            with time_range("threaded FFT sync before enqueue", color_id=55):
                stream.synchronize()

            self._release_raw_buffer(item.raw_batch_device)
            self._queue_put(
                self.post_input_queue,
                PowerBatch(
                    sequence=item.sequence,
                    batch_power=batch_power,
                ),
                "FFT -> post",
                color_id=90,
            )

    def _post_worker(self) -> None:
        stream = cp.cuda.Stream(non_blocking=True)

        with stream, time_range("threaded init post tensors", color_id=61):
            sliding_mean = SlidingMean2D(
                window_length=self.params.sliding_window_batches,
                height=self.height,
                width=self.width,
                dtype=self.params.real_dtype,
                reuse_mean_buffer=self.mode.preallocate_work_buffers,
            )
            display_clipper = PercentileClipDisplay2D(
                height=self.height,
                width=self.width,
                dtype=self.params.real_dtype,
                roi_radius=self.params.contrast_roi_radius,
                low_percentile=self.params.contrast_low_percentile,
                high_percentile=self.params.contrast_high_percentile,
                precompute_mask=self.mode.precompute_static_tensors,
                reuse_output_buffer=False,
            )

        stream.synchronize()

        while True:
            item = self._queue_get(self.post_input_queue, "post", color_id=91)
            if item is _STOP:
                self._queue_put(
                    self.d2h_input_queue,
                    _STOP,
                    "post -> D2H stop",
                    color_id=92,
                )
                return
            if not isinstance(item, PowerBatch):
                raise TypeError(f"Unexpected post item: {type(item)!r}")

            with stream, time_range("threaded postprocess batch", color_id=62):
                ready = sliding_mean.push(item.batch_power)

                display_image: cp.ndarray | None = None
                if ready:
                    averaged = sliding_mean.mean()
                    shifted = cp.fft.fftshift(averaged)
                    display_image = display_clipper.apply(shifted).astype(
                        self.params.real_dtype,
                        copy=False,
                    )

            with time_range("threaded post sync before enqueue", color_id=63):
                stream.synchronize()

            if display_image is not None:
                self._queue_put(
                    self.d2h_input_queue,
                    DisplayBatch(
                        sequence=item.sequence,
                        display_image=display_image,
                    ),
                    "post -> D2H",
                    color_id=94,
                )

    def _d2h_worker(self) -> None:
        stream = cp.cuda.Stream(non_blocking=True)
        output_host = (
            cupyx.empty_pinned(
                (self.height, self.width),
                dtype=self.params.real_dtype,
            )
            if self.mode.preallocate_work_buffers
            else None
        )

        while True:
            item = self._queue_get(self.d2h_input_queue, "D2H", color_id=95)
            if item is _STOP:
                return
            if not isinstance(item, DisplayBatch):
                raise TypeError(f"Unexpected D2H item: {type(item)!r}")

            if output_host is None:
                output_host = cupyx.empty_pinned(
                    (self.height, self.width),
                    dtype=self.params.real_dtype,
                )

            with stream, time_range("threaded D2H output", color_id=56):
                item.display_image.get(out=output_host, blocking=False)

            with time_range("threaded D2H sync before enqueue", color_id=57):
                stream.synchronize()

            self._queue_put(
                self.output_queue,
                OutputBatch(
                    sequence=item.sequence,
                    image=np.array(output_host, copy=True),
                ),
                "D2H -> output",
                color_id=96,
            )

            if not self.mode.preallocate_work_buffers:
                output_host = None


class ThreadedBenchmarkRunner:
    """Drive the threaded pipeline with queue-capacity backpressure.

    The main thread fills the H2D input queue, then spins on completed outputs
    while opportunistically refilling that input queue. A pending input batch is
    retained across failed queue-put attempts so cyclic host input is never
    skipped when the queue is full.
    """

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
        runtime: ThreadedRuntimeConfig,
        host_batch_iter: Iterator[np.ndarray],
    ) -> None:
        self.info = info
        self.params = params
        self.mode = mode
        self.runtime = runtime
        self.host_batch_iter = host_batch_iter
        self.workers = ThreadedPowerDopplerWorkers(info, params, mode, runtime)
        self._next_sequence = 0
        self._pending_input: InputBatch | None = None
        self._dummy_thread: DummyGilThread | None = None
        self._previous_switch_interval: float | None = None
        self._host_failure: str | None = None
        self._host_failure_lock = threading.Lock()

    def __enter__(self) -> ThreadedBenchmarkRunner:
        if self.runtime.gil_switch_interval_s is not None:
            with time_range("set threaded GIL switch interval", color_id=97):
                self._previous_switch_interval = sys.getswitchinterval()
                sys.setswitchinterval(self.runtime.gil_switch_interval_s)

        if self.mode.enable_dummy_gil_thread:
            with time_range("start threaded dummy GIL thread", color_id=99):
                self._dummy_thread = DummyGilThread(self.mode.dummy_gil_inner_loops)
                self._dummy_thread.start()
                self._dummy_thread.wait_until_ready()

        self.workers.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self.workers.close()
        finally:
            try:
                if self._dummy_thread is not None:
                    with time_range("stop threaded dummy GIL thread", color_id=100):
                        self._dummy_thread.stop()
                        self._dummy_thread.join()
            finally:
                if self._previous_switch_interval is not None:
                    with time_range(
                        "restore threaded GIL switch interval",
                        color_id=98,
                    ):
                        sys.setswitchinterval(self._previous_switch_interval)

    def _submit_next_batch(self) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1
        self.workers.submit(
            InputBatch(
                sequence=sequence,
                host_batch=next(self.host_batch_iter),
            )
        )
        return sequence

    def _try_submit_next_batch(self) -> bool:
        if self._pending_input is None:
            self._pending_input = InputBatch(
                sequence=self._next_sequence,
                host_batch=next(self.host_batch_iter),
            )

        submitted = self.workers.try_submit(self._pending_input)
        if submitted:
            self._next_sequence += 1
            self._pending_input = None
        return submitted

    def _wait_for_one_output(self) -> OutputBatch:
        while True:
            self._raise_host_failure_if_any()
            output = self.workers.get_output_timeout(0.05)
            if output is not None:
                return output

    def _set_host_failure(self, details: str) -> None:
        with self._host_failure_lock:
            if self._host_failure is None:
                self._host_failure = details

    def _raise_host_failure_if_any(self) -> None:
        with self._host_failure_lock:
            failure = self._host_failure
        if failure is not None:
            raise RuntimeError(f"Host driver thread failed:\n{failure}")

    def _source_loop(self, stop_event: threading.Event) -> None:
        try:
            with time_range("source thread loop", color_id=106):
                while not stop_event.is_set():
                    self._try_submit_next_batch()
        except BaseException:
            self._set_host_failure(traceback.format_exc())
            stop_event.set()

    def _sink_loop(
        self,
        done_event: threading.Event,
        result: dict[str, object],
        t0: float,
    ) -> None:
        completed = 0
        last_image: np.ndarray | None = None
        elapsed = 0.0

        try:
            with time_range("sink thread loop", color_id=107):
                while not done_event.is_set():
                    output = self.workers.get_output_timeout(0.05)
                    if output is None:
                        continue

                    completed += 1
                    last_image = output.image
                    elapsed = perf_counter() - t0
                    if elapsed >= self.params.benchmark_seconds:
                        done_event.set()

            result["completed"] = completed
            result["last_image"] = last_image
            result["elapsed"] = elapsed
        except BaseException:
            self._set_host_failure(traceback.format_exc())
            done_event.set()

    def prime(self) -> None:
        batches_to_prime = self.params.sliding_window_batches - 1

        print("Priming threaded sliding window...")
        with time_range("threaded prime sliding window", color_id=58):
            for _ in range(batches_to_prime):
                self._submit_next_batch()

    def warmup(self) -> None:
        print("Warming up threaded pipeline...")
        with time_range("threaded warmup outputs", color_id=59):
            for _ in range(self.params.warmup_outputs):
                self._submit_next_batch()
                self._wait_for_one_output()

    def run(self) -> tuple[np.ndarray, BenchmarkStats]:
        params = self.params
        mode = self.mode
        info = self.info

        completed = 0
        elapsed = 0.0
        last_image: np.ndarray | None = None

        print("Running threaded steady-state benchmark...")

        with time_range("threaded steady-state benchmark loop", color_id=60):
            t0 = perf_counter()
            done_event = threading.Event()
            result: dict[str, object] = {}
            source_thread = threading.Thread(
                name="cupy-threaded-source",
                target=self._source_loop,
                args=(done_event,),
                daemon=True,
            )
            sink_thread = threading.Thread(
                name="cupy-threaded-sink",
                target=self._sink_loop,
                args=(done_event, result, t0),
                daemon=True,
            )

            source_thread.start()
            sink_thread.start()

            while not done_event.wait(timeout=0.05):
                self._raise_host_failure_if_any()
                self.workers.raise_worker_failure_if_any()

            source_thread.join()
            sink_thread.join()
            self._raise_host_failure_if_any()
            self.workers.raise_worker_failure_if_any()

            completed = int(result.get("completed", 0))
            elapsed = float(result.get("elapsed", 0.0))
            maybe_image = result.get("last_image")
            if isinstance(maybe_image, np.ndarray):
                last_image = maybe_image

        if last_image is None:
            raise RuntimeError("Benchmark produced no output image.")

        dummy_iterations = (
            0 if self._dummy_thread is None else self._dummy_thread.iterations
        )
        total_frames = completed * params.batch_frames
        input_frame_bytes = bytes_per_frame(
            height=info.height,
            width=info.width,
            dtype=params.acquisition_dtype,
        )
        cast_frame_bytes = bytes_per_frame(
            height=info.height,
            width=info.width,
            dtype=params.real_dtype,
        )
        output_image_bytes = bytes_per_frame(
            height=info.height,
            width=info.width,
            dtype=params.real_dtype,
        )

        k0, k1 = doppler_bin_range(
            window_size=params.batch_frames,
            sample_rate_hz=params.sample_rate_hz,
            doppler_low_hz=params.doppler_low_hz,
            doppler_high_hz=params.doppler_high_hz,
        )

        return last_image, BenchmarkStats(
            mode_name=_threaded_mode_name(mode),
            precompute_static_tensors=mode.precompute_static_tensors,
            preallocate_work_buffers=mode.preallocate_work_buffers,
            seconds=elapsed,
            frames=total_frames,
            batches=completed,
            outputs=completed,
            input_fps=total_frames / elapsed,
            batches_per_second=completed / elapsed,
            outputs_per_second=completed / elapsed,
            wall_ms_per_output=1e3 * elapsed / completed,
            h2d_gbps=(total_frames * input_frame_bytes) / elapsed / 1e9,
            cast_effective_gbps=(total_frames * cast_frame_bytes) / elapsed / 1e9,
            d2h_output_mbps=(completed * output_image_bytes) / elapsed / 1e6,
            shape=(info.height, info.width),
            file_dtype=str(info.dtype),
            host_dtype=str(info.dtype),
            device_input_dtype=str(params.acquisition_dtype),
            real_dtype=str(params.real_dtype),
            complex_dtype=str(params.complex_dtype),
            doppler_bins=(k0, k1),
            doppler_bin_count=k1 - k0,
            batch_frames=params.batch_frames,
            batches_per_output=params.batches_per_output,
            frames_per_output=params.temporal_support_frames,
            output_stride_frames=params.output_stride_frames,
            temporal_support_ms=1e3 * params.temporal_support_seconds,
            dummy_gil_thread_enabled=mode.enable_dummy_gil_thread,
            dummy_gil_inner_loops=mode.dummy_gil_inner_loops,
            dummy_gil_iterations=dummy_iterations,
            dummy_gil_iterations_per_second=dummy_iterations / elapsed,
            dummy_gil_switch_interval_s=self.runtime.gil_switch_interval_s,
        )


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
    runtime: ThreadedRuntimeConfig,
) -> tuple[np.ndarray, BenchmarkStats]:
    expected_shape = (
        params.sliding_window_batches,
        params.batch_frames,
        info.height,
        info.width,
    )
    if host_batches.shape != expected_shape:
        raise ValueError(
            f"Unexpected preloaded host batch shape: got {host_batches.shape}, "
            f"expected {expected_shape}."
        )

    clear_cupy_pools()

    print()
    print("=" * 80)
    print(f"Mode: {_threaded_mode_name(mode)}")
    print(
        f"  precompute_static_tensors={mode.precompute_static_tensors} | "
        f"preallocate_work_buffers={mode.preallocate_work_buffers}"
    )
    print(
        f"  queue_depth={runtime.queue_depth} | "
        f"queue_put_policy={runtime.queue_put_policy} | "
        f"queue_put_timeout_s={runtime.queue_put_timeout_s} | "
        f"gil_switch_interval_s={runtime.gil_switch_interval_s}"
    )
    if mode.enable_dummy_gil_thread:
        print(f"  dummy_gil_inner_loops={mode.dummy_gil_inner_loops}")

    with ThreadedBenchmarkRunner(
        info=info,
        params=params,
        mode=mode,
        runtime=runtime,
        host_batch_iter=cycle_batches(host_batches),
    ) as runner:
        runner.prime()
        runner.warmup()
        image, stats = runner.run()

    clear_cupy_pools()
    return image, stats


def benchmark_suite(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    modes: Sequence[ExecutionMode],
    runtime: ThreadedRuntimeConfig,
) -> list[tuple[np.ndarray, BenchmarkStats]]:
    results: list[tuple[np.ndarray, BenchmarkStats]] = []

    for mode in modes:
        image, stats = benchmark_mode(
            host_batches=host_batches,
            info=info,
            params=params,
            mode=mode,
            runtime=runtime,
        )
        results.append((image, stats))

    return results


def _threaded_mode_name(mode: ExecutionMode) -> str:
    return mode.name.replace("cupy-naive", "cupy-threaded", 1)
