from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import cupyx
import numpy as np
from cupyx.profiler import time_range

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo
from holoflow_benchmarks.runtime import (
    DummyGilThread,
    start_dummy_gil_thread,
    stop_dummy_gil_thread,
)
from holoflow_benchmarks.stats import RunMeasurement

from .compute import PowerDopplerPipeline


@dataclass(frozen=True)
class SingleThreadStreamRuntimeConfig:
    """CUDA stream scheduler parameters.

    `num_slots` is the maximum number of batches admitted into the stream/event
    graph. Slot reuse is gated by the slot's previous D2H completion event.

    `h2d_prefetch_batches` keeps that many uploads queued ahead of the oldest
    batch whose compute/D2H work has not been submitted yet. This avoids copy
    engine head-of-line blocking on devices where D2H and later H2D work share
    one async copy engine.
    """

    num_slots: int = 3
    h2d_prefetch_batches: int = 1

    def __post_init__(self) -> None:
        if self.num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {self.num_slots}.")
        if self.h2d_prefetch_batches < 0:
            raise ValueError(
                "h2d_prefetch_batches must be non-negative, got "
                f"{self.h2d_prefetch_batches}."
            )
        if self.h2d_prefetch_batches >= self.num_slots - 1:
            raise ValueError(
                "h2d_prefetch_batches must leave at least two non-prefetch slots, got "
                f"{self.h2d_prefetch_batches} and {self.num_slots}."
            )


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
        self._prefetched_slots: deque[StreamSlot] = deque()
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

    @property
    def prefetched_batch_count(self) -> int:
        return len(self._prefetched_slots)

    def submit_batch(self, host_batch: np.ndarray) -> int:
        sequence = self.prefetch_batch(host_batch)
        self.schedule_oldest_prefetched_batch()
        return sequence

    def prefetch_batch(self, host_batch: np.ndarray) -> int:
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

        self._prefetched_slots.append(slot)
        return sequence

    def schedule_oldest_prefetched_batch(self) -> int:
        if not self._prefetched_slots:
            raise RuntimeError("No prefetched stream batch is ready to schedule.")

        slot = self._prefetched_slots.popleft()
        if slot.sequence is None:
            raise RuntimeError("Cannot schedule an unassigned stream slot.")

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

        return slot.sequence

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
        if self._prefetched_slots:
            self.discard_prefetched_batches()

        with time_range("streams final H2D sync", color_id=125):
            self.h2d_stream.synchronize()
        with time_range("streams final compute sync", color_id=125):
            self.compute_stream.synchronize()
        with time_range("streams final D2H sync", color_id=125):
            self.d2h_stream.synchronize()
        self.poll_completed_outputs()

    def discard_prefetched_batches(self) -> None:
        with time_range("streams discard prefetched H2D", color_id=125):
            self.h2d_stream.synchronize()

        while self._prefetched_slots:
            slot = self._prefetched_slots.popleft()
            slot.host_input = None
            slot.sequence = None
            slot.output_ready = False
            slot.output_collected = True

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

    def _prefetch_next_batch(self) -> int:
        return self.pipeline.prefetch_batch(next(self.host_batch_iter))

    def _submit_next_batch_with_h2d_lookahead(self) -> int:
        while (
            self.pipeline.prefetched_batch_count
            <= self.pipeline.runtime.h2d_prefetch_batches
        ):
            self._prefetch_next_batch()

        return self.pipeline.schedule_oldest_prefetched_batch()

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

    def run(self) -> RunMeasurement:
        params = self.pipeline.params
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
                    self._submit_next_batch_with_h2d_lookahead()

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
        return RunMeasurement(
            image=last_image,
            seconds=elapsed,
            batches=completed,
            outputs=completed,
            dummy_gil_iterations=dummy_iterations,
            dummy_gil_switch_interval_s=mode.dummy_gil_switch_interval_s,
        )
