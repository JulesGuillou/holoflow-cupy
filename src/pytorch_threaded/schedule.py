from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from time import perf_counter
import sys
import threading
import traceback

import numpy as np
import torch

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo
from holoflow_benchmarks.stats import RunMeasurement

from .compute import (
    BatchPowerComputer,
    PercentileClipDisplay2D,
    SlidingMean2D,
    cuda_device,
    fftshift_image,
)
from .dtypes import torch_dtype
from .nvtx import time_range
from .runtime import DummyGilThread


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
    raw_batch_device: torch.Tensor


@dataclass(frozen=True)
class PowerBatch:
    sequence: int
    batch_power: torch.Tensor


@dataclass(frozen=True)
class DisplayBatch:
    sequence: int
    display_image: torch.Tensor


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
    """Own the four long-lived worker threads and their queue handoffs."""

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
        self.device = cuda_device()
        self.device_id = torch.cuda.current_device()
        self.acquisition_dtype = torch_dtype(params.acquisition_dtype)
        self.real_dtype = torch_dtype(params.real_dtype)

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
        self.output_queue: Queue[OutputBatch] = Queue()
        self.failure_queue: Queue[WorkerFailure] = Queue()

        self.raw_buffer_pool: Queue[torch.Tensor] | None = None
        if mode.preallocate_work_buffers:
            raw_buffer_count = runtime.queue_depth + 2
            self.raw_buffer_pool = Queue(maxsize=raw_buffer_count)
            for _ in range(raw_buffer_count):
                self.raw_buffer_pool.put(
                    torch.empty(
                        (self.batch_frames, self.height, self.width),
                        dtype=self.acquisition_dtype,
                        device=self.device,
                    )
                )

        self._threads = [
            threading.Thread(
                name="pytorch-threaded-h2d",
                target=self._run_worker,
                args=("h2d", self._h2d_worker),
                daemon=True,
            ),
            threading.Thread(
                name="pytorch-threaded-fft",
                target=self._run_worker,
                args=("fft", self._fft_worker),
                daemon=True,
            ),
            threading.Thread(
                name="pytorch-threaded-post",
                target=self._run_worker,
                args=("post", self._post_worker),
                daemon=True,
            ),
            threading.Thread(
                name="pytorch-threaded-d2h",
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
                color_id=280,
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
            color_id=281,
        )
        self._raise_worker_failure_if_any()

    def try_submit(self, item: InputBatch) -> bool:
        self._raise_worker_failure_if_any()
        submitted = self._queue_put_once(
            self.h2d_input_queue,
            item,
            "main -> H2D",
            color_id=301,
        )
        self._raise_worker_failure_if_any()
        return submitted

    def get_output_timeout(self, timeout_s: float) -> OutputBatch | None:
        self._raise_worker_failure_if_any()
        with time_range("pytorch-threaded wait output queue input", color_id=305):
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
            torch.cuda.set_device(self.device_id)
            worker()
        except BaseException:
            self.failure_queue.put(
                WorkerFailure(stage=stage, details=traceback.format_exc())
            )

    def _queue_get(self, queue: Queue, label: str, color_id: int) -> object:
        with time_range(f"pytorch-threaded wait {label} input", color_id=color_id):
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
            f"pytorch-threaded try {label} output slot"
            if self.runtime.queue_put_policy == "nowait"
            else f"pytorch-threaded wait {label} output slot"
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

    def _get_raw_buffer(self) -> torch.Tensor:
        if self.raw_buffer_pool is None:
            return torch.empty(
                (self.batch_frames, self.height, self.width),
                dtype=self.acquisition_dtype,
                device=self.device,
            )

        buffer = self._queue_get(
            self.raw_buffer_pool,
            "H2D raw buffer pool",
            color_id=283,
        )
        if not isinstance(buffer, torch.Tensor):
            raise TypeError(f"Unexpected raw buffer item: {type(buffer)!r}")
        return buffer

    def _release_raw_buffer(self, raw_batch_device: torch.Tensor) -> None:
        if self.raw_buffer_pool is not None:
            self._queue_put(
                self.raw_buffer_pool,
                raw_batch_device,
                "FFT raw buffer pool",
                color_id=284,
            )

    def _h2d_worker(self) -> None:
        stream = torch.cuda.Stream(device=self.device_id)

        while True:
            item = self._queue_get(self.h2d_input_queue, "H2D", color_id=285)
            if item is _STOP:
                self._queue_put(
                    self.fft_input_queue,
                    _STOP,
                    "H2D -> FFT stop",
                    color_id=286,
                )
                return
            if not isinstance(item, InputBatch):
                raise TypeError(f"Unexpected H2D item: {type(item)!r}")

            raw_batch_device = self._get_raw_buffer()
            host_tensor = torch.from_numpy(item.host_batch)
            with torch.cuda.stream(stream), time_range(
                "pytorch-threaded H2D upload",
                color_id=251,
            ):
                raw_batch_device.copy_(host_tensor, non_blocking=True)

            with time_range("pytorch-threaded H2D sync before enqueue", color_id=252):
                stream.synchronize()

            self._queue_put(
                self.fft_input_queue,
                UploadedBatch(
                    sequence=item.sequence,
                    raw_batch_device=raw_batch_device,
                ),
                "H2D -> FFT",
                color_id=287,
            )

    def _fft_worker(self) -> None:
        stream = torch.cuda.Stream(device=self.device_id)

        with torch.cuda.stream(stream):
            power_computer = BatchPowerComputer(
                info=self.info,
                params=self.params,
                mode=self.mode,
                device=self.device,
            )

        stream.synchronize()

        while True:
            item = self._queue_get(self.fft_input_queue, "FFT", color_id=288)
            if item is _STOP:
                self._queue_put(
                    self.post_input_queue,
                    _STOP,
                    "FFT -> post stop",
                    color_id=289,
                )
                return
            if not isinstance(item, UploadedBatch):
                raise TypeError(f"Unexpected FFT item: {type(item)!r}")

            with torch.cuda.stream(stream), time_range(
                "pytorch-threaded FFT batch",
                color_id=254,
            ):
                batch_power = power_computer.compute(item.raw_batch_device)

            with time_range("pytorch-threaded FFT sync before enqueue", color_id=255):
                stream.synchronize()

            self._release_raw_buffer(item.raw_batch_device)
            self._queue_put(
                self.post_input_queue,
                PowerBatch(
                    sequence=item.sequence,
                    batch_power=batch_power,
                ),
                "FFT -> post",
                color_id=290,
            )

    def _post_worker(self) -> None:
        stream = torch.cuda.Stream(device=self.device_id)

        with torch.cuda.stream(stream), time_range(
            "pytorch-threaded init post tensors",
            color_id=261,
        ):
            sliding_mean = SlidingMean2D(
                window_length=self.params.sliding_window_batches,
                height=self.height,
                width=self.width,
                dtype=self.params.real_dtype,
                device=self.device,
                reuse_mean_buffer=self.mode.preallocate_work_buffers,
            )
            display_clipper = PercentileClipDisplay2D(
                height=self.height,
                width=self.width,
                dtype=self.params.real_dtype,
                device=self.device,
                roi_radius=self.params.contrast_roi_radius,
                low_percentile=self.params.contrast_low_percentile,
                high_percentile=self.params.contrast_high_percentile,
                precompute_mask=self.mode.precompute_static_tensors,
                reuse_output_buffer=False,
            )

        stream.synchronize()

        while True:
            item = self._queue_get(self.post_input_queue, "post", color_id=291)
            if item is _STOP:
                self._queue_put(
                    self.d2h_input_queue,
                    _STOP,
                    "post -> D2H stop",
                    color_id=292,
                )
                return
            if not isinstance(item, PowerBatch):
                raise TypeError(f"Unexpected post item: {type(item)!r}")

            with torch.cuda.stream(stream), time_range(
                "pytorch-threaded postprocess batch",
                color_id=262,
            ):
                ready = sliding_mean.push(item.batch_power)

                display_image: torch.Tensor | None = None
                if ready:
                    averaged = sliding_mean.mean()
                    shifted = fftshift_image(averaged)
                    display_image = display_clipper.apply(shifted).to(
                        dtype=self.real_dtype,
                        copy=False,
                    )

            with time_range("pytorch-threaded post sync before enqueue", color_id=263):
                stream.synchronize()

            if display_image is not None:
                self._queue_put(
                    self.d2h_input_queue,
                    DisplayBatch(
                        sequence=item.sequence,
                        display_image=display_image,
                    ),
                    "post -> D2H",
                    color_id=294,
                )

    def _d2h_worker(self) -> None:
        stream = torch.cuda.Stream(device=self.device_id)
        output_host = (
            torch.empty(
                (self.height, self.width),
                dtype=self.real_dtype,
                pin_memory=True,
            )
            if self.mode.preallocate_work_buffers
            else None
        )

        while True:
            item = self._queue_get(self.d2h_input_queue, "D2H", color_id=295)
            if item is _STOP:
                return
            if not isinstance(item, DisplayBatch):
                raise TypeError(f"Unexpected D2H item: {type(item)!r}")

            if output_host is None:
                output_host = torch.empty(
                    (self.height, self.width),
                    dtype=self.real_dtype,
                    pin_memory=True,
                )

            with torch.cuda.stream(stream), time_range(
                "pytorch-threaded D2H output",
                color_id=256,
            ):
                output_host.copy_(item.display_image, non_blocking=True)

            with time_range("pytorch-threaded D2H sync before enqueue", color_id=257):
                stream.synchronize()

            self._queue_put(
                self.output_queue,
                OutputBatch(
                    sequence=item.sequence,
                    image=output_host.numpy().copy(),
                ),
                "D2H -> output",
                color_id=296,
            )

            if not self.mode.preallocate_work_buffers:
                output_host = None


class ThreadedBenchmarkRunner:
    """Drive the threaded pipeline with queue-capacity backpressure."""

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
            with time_range("set pytorch-threaded GIL switch interval", color_id=297):
                self._previous_switch_interval = sys.getswitchinterval()
                sys.setswitchinterval(self.runtime.gil_switch_interval_s)

        if self.mode.enable_dummy_gil_thread:
            with time_range("start pytorch-threaded dummy GIL thread", color_id=299):
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
                    with time_range(
                        "stop pytorch-threaded dummy GIL thread",
                        color_id=300,
                    ):
                        self._dummy_thread.stop()
                        self._dummy_thread.join()
            finally:
                if self._previous_switch_interval is not None:
                    with time_range(
                        "restore pytorch-threaded GIL switch interval",
                        color_id=298,
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
            with time_range("pytorch-threaded source thread loop", color_id=306):
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
            with time_range("pytorch-threaded sink thread loop", color_id=307):
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

        print("Priming PyTorch threaded sliding window...")
        with time_range("pytorch-threaded prime sliding window", color_id=258):
            for _ in range(batches_to_prime):
                self._submit_next_batch()

    def warmup(self) -> None:
        print("Warming up PyTorch threaded pipeline...")
        with time_range("pytorch-threaded warmup outputs", color_id=259):
            for _ in range(self.params.warmup_outputs):
                self._submit_next_batch()
                self._wait_for_one_output()

    def run(self) -> RunMeasurement:
        completed = 0
        elapsed = 0.0
        last_image: np.ndarray | None = None

        print("Running PyTorch threaded steady-state benchmark...")

        with time_range("pytorch-threaded steady-state benchmark loop", color_id=260):
            t0 = perf_counter()
            done_event = threading.Event()
            result: dict[str, object] = {}
            source_thread = threading.Thread(
                name="pytorch-threaded-source",
                target=self._source_loop,
                args=(done_event,),
                daemon=True,
            )
            sink_thread = threading.Thread(
                name="pytorch-threaded-sink",
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
        return RunMeasurement(
            image=last_image,
            seconds=elapsed,
            batches=completed,
            outputs=completed,
            dummy_gil_iterations=dummy_iterations,
            dummy_gil_switch_interval_s=self.runtime.gil_switch_interval_s,
        )
