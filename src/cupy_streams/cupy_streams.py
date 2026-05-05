from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import cupyx
import numpy as np
from cupyx.profiler import time_range

from cupy_naive.config import ExecutionMode, Params
from cupy_naive.cupy_naive import (
    BenchmarkStats,
    DummyGilThread,
    PowerDopplerPipeline,
    clear_cupy_pools,
    doppler_bin_range,
    start_dummy_gil_thread,
    stop_dummy_gil_thread,
)
from cupy_naive.io import InputInfo, bytes_per_frame, cycle_batches


@dataclass(frozen=True)
class SingleThreadStreamRuntimeConfig:
    """CUDA stream scheduler parameters.

    `num_slots` is the maximum number of batches admitted into the stream/event
    graph. Slot reuse is gated by the slot's previous D2H completion event.
    """

    num_slots: int = 2


@dataclass(frozen=True)
class StreamOutput:
    sequence: int
    image: np.ndarray


@dataclass
class StreamSlot:
    index: int
    raw_batch_device: cp.ndarray
    real_batch_device: cp.ndarray | None
    output_device: cp.ndarray
    output_host: np.ndarray
    h2d_done: cp.cuda.Event
    compute_done: cp.cuda.Event
    d2h_done: cp.cuda.Event
    host_input: np.ndarray | None = None
    sequence: int | None = None
    output_ready: bool = False
    output_collected: bool = True


class SingleThreadStreamPowerDopplerPipeline(PowerDopplerPipeline):
    """Submit the LDH pipeline from one host thread into three CUDA streams.

    CUDA streams are GPU-side command queues, not CPU threads. This scheduler
    uses one Python thread to submit H2D, compute, and D2H work into separate
    streams, and CUDA events express the same dependencies that queues and
    worker handoffs express in the threaded implementation:

        H_n -> C_n -> D_n, and H_n waits for D_{n-K} before reusing a slot.
    """

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
        runtime: SingleThreadStreamRuntimeConfig,
    ) -> None:
        if runtime.num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {runtime.num_slots}.")

        self.runtime = runtime
        self.h2d_stream = cp.cuda.Stream(non_blocking=True)
        self.compute_stream = cp.cuda.Stream(non_blocking=True)
        self.d2h_stream = cp.cuda.Stream(non_blocking=True)

        with self.compute_stream:
            super().__init__(
                info=info,
                params=params,
                mode=mode,
                allocate_batch_io_buffers=False,
                reuse_display_output=False,
            )

        # Static tensors created on the compute stream must be complete before
        # later non-default streams start waiting on per-batch events.
        self.compute_stream.synchronize()

        self.slots = [self._make_slot(index) for index in range(runtime.num_slots)]
        self._next_sequence = 0
        self._pending_outputs: deque[StreamSlot] = deque()
        self._completed_outputs: deque[StreamOutput] = deque()

    def _make_slot(self, index: int) -> StreamSlot:
        raw_batch_device = cp.empty(
            (self.batch_frames, self.height, self.width),
            dtype=self.params.acquisition_dtype,
        )
        real_batch_device = (
            cp.empty(
                (self.batch_frames, self.height, self.width),
                dtype=self.params.real_dtype,
            )
            if self.mode.preallocate_work_buffers
            else None
        )
        output_dtype = self._slot_output_dtype()
        output_device = cp.empty((self.height, self.width), dtype=output_dtype)
        output_host = cupyx.empty_pinned(
            (self.height, self.width),
            dtype=output_dtype,
        )

        return StreamSlot(
            index=index,
            raw_batch_device=raw_batch_device,
            real_batch_device=real_batch_device,
            output_device=output_device,
            output_host=output_host,
            h2d_done=cp.cuda.Event(disable_timing=True),
            compute_done=cp.cuda.Event(disable_timing=True),
            d2h_done=cp.cuda.Event(disable_timing=True),
        )

    def _slot_output_dtype(self) -> np.dtype:
        if self.mode.preallocate_work_buffers:
            return self.params.real_dtype

        # Match the naive non-preallocated display path: cp.percentile produces
        # float64 clip bounds, and cp.clip promotes the returned image.
        return np.dtype(np.float64)

    def submit_batch(self, host_batch: np.ndarray) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1

        slot = self.slots[sequence % len(self.slots)]
        self._wait_until_reusable(slot)

        slot.host_input = host_batch
        slot.sequence = sequence
        slot.output_ready = False
        slot.output_collected = True

        with self.h2d_stream, time_range("streams H2D upload", color_id=120):
            slot.raw_batch_device.set(host_batch, stream=self.h2d_stream)
            slot.h2d_done.record(self.h2d_stream)

        self.compute_stream.wait_event(slot.h2d_done)
        with self.compute_stream, time_range("streams compute batch", color_id=121):
            ready = self.process_batch_device(
                raw_batch_device=slot.raw_batch_device,
                real_batch_device=slot.real_batch_device,
            )
            if ready:
                self.finalize_display_image_device(out=slot.output_device)
            slot.compute_done.record(self.compute_stream)

        self.d2h_stream.wait_event(slot.compute_done)
        with self.d2h_stream, time_range("streams D2H output", color_id=122):
            if ready:
                slot.output_device.get(
                    out=slot.output_host,
                    stream=self.d2h_stream,
                    blocking=False,
                )
                slot.output_ready = True
                slot.output_collected = False
                self._pending_outputs.append(slot)

            slot.d2h_done.record(self.d2h_stream)

        return sequence

    def _wait_until_reusable(self, slot: StreamSlot) -> None:
        if slot.sequence is None:
            return

        # This is the H_n waits for D_{n-K} dependency. It blocks only this
        # host scheduler until the slot's prior D2H/no-op D stage is complete.
        with time_range("streams wait reusable slot", color_id=123):
            slot.d2h_done.synchronize()

        self._collect_slot_output(slot)
        slot.host_input = None
        slot.sequence = None
        slot.output_ready = False

    def poll_completed_outputs(self) -> list[StreamOutput]:
        while self._pending_outputs and self._pending_outputs[0].output_collected:
            self._pending_outputs.popleft()

        while self._pending_outputs:
            slot = self._pending_outputs[0]
            if not slot.d2h_done.done:
                break

            self._collect_slot_output(slot)
            self._pending_outputs.popleft()

        outputs = list(self._completed_outputs)
        self._completed_outputs.clear()
        return outputs

    def wait_for_one_output(self) -> StreamOutput:
        while True:
            outputs = self.poll_completed_outputs()
            if outputs:
                for output in outputs[1:]:
                    self._completed_outputs.append(output)
                return outputs[0]

            if not self._pending_outputs:
                raise RuntimeError("No stream output is pending.")

            with time_range("streams wait output", color_id=124):
                self._pending_outputs[0].d2h_done.synchronize()

    def finish(self) -> None:
        with time_range("streams final D2H sync", color_id=125):
            self.d2h_stream.synchronize()
        self.poll_completed_outputs()

    def _collect_slot_output(self, slot: StreamSlot) -> None:
        if not slot.output_ready or slot.output_collected:
            return
        if slot.sequence is None:
            raise RuntimeError("Cannot collect an output from an unassigned slot.")

        self._completed_outputs.append(
            StreamOutput(
                sequence=slot.sequence,
                image=np.array(slot.output_host, copy=True),
            )
        )
        slot.output_collected = True


class SingleThreadStreamBenchmarkRunner:
    """Prime, warm up, and measure the single-thread stream scheduler."""

    def __init__(
        self,
        pipeline: SingleThreadStreamPowerDopplerPipeline,
        host_batch_iter: Iterator[np.ndarray],
    ) -> None:
        self.pipeline = pipeline
        self.host_batch_iter = host_batch_iter

    def _submit_next_batch(self) -> int:
        return self.pipeline.submit_batch(next(self.host_batch_iter))

    def prime(self) -> None:
        batches_to_prime = self.pipeline.params.sliding_window_batches - 1

        print("Priming stream sliding window...")
        with time_range("streams prime sliding window", color_id=126):
            for _ in range(batches_to_prime):
                self._submit_next_batch()
            self.pipeline.finish()

    def warmup(self) -> None:
        print("Warming up stream pipeline...")
        with time_range("streams warmup outputs", color_id=127):
            for _ in range(self.pipeline.params.warmup_outputs):
                self._submit_next_batch()
                self.pipeline.wait_for_one_output()
            self.pipeline.finish()

    def run(self) -> tuple[np.ndarray, BenchmarkStats]:
        params = self.pipeline.params
        info = self.pipeline.info
        mode = self.pipeline.mode

        completed = 0
        elapsed = 0.0
        last_image: np.ndarray | None = None

        dummy_thread: DummyGilThread | None = None
        previous_switch_interval: float | None = None

        print("Running stream steady-state benchmark...")

        try:
            dummy_thread, previous_switch_interval = start_dummy_gil_thread(mode)

            with time_range("streams steady-state benchmark loop", color_id=128):
                t0 = perf_counter()
                while elapsed < params.benchmark_seconds:
                    self._submit_next_batch()

                    for output in self.pipeline.poll_completed_outputs():
                        completed += 1
                        last_image = output.image
                        elapsed = perf_counter() - t0

                # Keep the benchmark result tied to completed D2H outputs, but
                # leave any extra in-flight batch out of the measured counters.
                self.pipeline.finish()

        finally:
            stop_dummy_gil_thread(dummy_thread, previous_switch_interval)

        if last_image is None or completed == 0:
            raise RuntimeError("Benchmark produced no output image.")

        dummy_iterations = 0 if dummy_thread is None else dummy_thread.iterations
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
            mode_name=_stream_mode_name(mode, self.pipeline.runtime),
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
            dummy_gil_switch_interval_s=mode.dummy_gil_switch_interval_s,
        )


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
    runtime: SingleThreadStreamRuntimeConfig,
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
    print(f"Mode: {_stream_mode_name(mode, runtime)}")
    print(
        f"  precompute_static_tensors={mode.precompute_static_tensors} | "
        f"preallocate_work_buffers={mode.preallocate_work_buffers} | "
        f"num_slots={runtime.num_slots}"
    )
    if mode.enable_dummy_gil_thread:
        print(
            f"  dummy_gil_inner_loops={mode.dummy_gil_inner_loops} | "
            f"dummy_gil_switch_interval_s={mode.dummy_gil_switch_interval_s}"
        )

    with time_range(f"init stream mode {mode.name}", color_id=129):
        pipeline = SingleThreadStreamPowerDopplerPipeline(
            info=info,
            params=params,
            mode=mode,
            runtime=runtime,
        )
        runner = SingleThreadStreamBenchmarkRunner(
            pipeline=pipeline,
            host_batch_iter=cycle_batches(host_batches),
        )

    with time_range(f"run stream mode {mode.name}", color_id=130):
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
    runtime: SingleThreadStreamRuntimeConfig,
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


def _stream_mode_name(
    mode: ExecutionMode,
    runtime: SingleThreadStreamRuntimeConfig,
) -> str:
    name = mode.name
    for prefix in ("cupy-naive", "cupy-threaded", "cupy-streams"):
        if name.startswith(prefix):
            name = name.replace(prefix, "cupy-streams", 1)
            break
    else:
        name = f"cupy-streams | {name}"

    return f"{name} | slots={runtime.num_slots}"
