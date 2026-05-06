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

    `pipeline_prefetch_batches` is the number of logical pipeline ticks kept
    submitted ahead of completed outputs during steady state. Each tick submits
    H2D for x_t, compute for x_{t-1}, and D2H for x_{t-2} when those items
    exist. CUDA events preserve dependencies while allowing API calls to be
    queued before the work is executable.
    """

    num_slots: int = 3
    pipeline_prefetch_batches: int = 1

    def __post_init__(self) -> None:
        if self.num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {self.num_slots}.")
        if self.pipeline_prefetch_batches <= 0:
            raise ValueError(
                "pipeline_prefetch_batches must be positive, got "
                f"{self.pipeline_prefetch_batches}."
            )
        if self.pipeline_prefetch_batches >= self.num_slots:
            raise ValueError(
                "pipeline_prefetch_batches must be smaller than num_slots, got "
                f"{self.pipeline_prefetch_batches} and {self.num_slots}."
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
    display_ready: bool = False
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
        self._h2d_queue: deque[StreamSlot] = deque()
        self._compute_queue: deque[StreamSlot] = deque()
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
    def h2d_queue_depth(self) -> int:
        return len(self._h2d_queue)

    @property
    def compute_queue_depth(self) -> int:
        return len(self._compute_queue)

    def submit_batch(self, host_batch: np.ndarray) -> int:
        sequence = self.prefetch_batch(host_batch, tick=None)
        self.schedule_oldest_h2d_batch(tick=None)
        self.schedule_oldest_compute_batch(tick=None)
        return sequence

    def submit_pipeline_tick(self, host_batch: np.ndarray) -> int:
        """Submit one formal pipeline tick in source-to-sink API order.

        At tick t this enqueues H2D for x_t, compute for x_{t-1}, and D2H for
        x_{t-2} when those older stage queues are non-empty. The pre-existing
        queue lengths are captured before H2D submission so a new item cannot
        advance through multiple stages in the same logical tick.
        """
        schedule_compute = bool(self._h2d_queue)
        schedule_d2h = bool(self._compute_queue)
        tick = self._next_sequence
        compute_sequence = self._h2d_queue[0].sequence if schedule_compute else None
        d2h_sequence = self._compute_queue[0].sequence if schedule_d2h else None

        tick_label = (
            f"streams tick {tick}: "
            f"H2D batch {tick}; "
            f"compute batch {self._optional_sequence_label(compute_sequence)}; "
            f"D2H batch {self._optional_sequence_label(d2h_sequence)}"
        )
        with time_range(tick_label, color_id=128):
            sequence = self.prefetch_batch(host_batch, tick=tick)

            if schedule_compute:
                self.schedule_oldest_h2d_batch(tick=tick)
            if schedule_d2h:
                self.schedule_oldest_compute_batch(tick=tick)

        return sequence

    @staticmethod
    def _optional_sequence_label(sequence: int | None) -> str:
        return "-" if sequence is None else str(sequence)

    def _stage_label(self, stage: str, sequence: int, tick: int | None) -> str:
        if tick is None:
            return f"streams {stage} batch {sequence}"

        return f"streams tick {tick}: {stage} batch {sequence}"

    def prefetch_batch(self, host_batch: np.ndarray, tick: int | None) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1

        slot = self.slots[sequence % len(self.slots)]
        self._wait_until_reusable(slot)

        slot.host_input = host_batch
        slot.sequence = sequence
        slot.display_ready = False
        slot.output_ready = False
        slot.output_collected = True

        label = self._stage_label("H2D", sequence, tick)
        with self.h2d_stream, time_range(label, color_id=120):
            slot.raw_batch_device.set(host_batch, stream=self.h2d_stream)
            slot.h2d_done.record(self.h2d_stream)

        self._h2d_queue.append(slot)
        return sequence

    def schedule_oldest_h2d_batch(self, tick: int | None) -> int:
        if not self._h2d_queue:
            raise RuntimeError("No H2D-prefetched stream batch is ready to compute.")

        slot = self._h2d_queue.popleft()
        if slot.sequence is None:
            raise RuntimeError("Cannot schedule an unassigned stream slot.")

        label = self._stage_label("compute", slot.sequence, tick)
        with self.compute_stream, time_range(label, color_id=121):
            self.compute_stream.wait_event(slot.h2d_done)
            ready = self.process_batch_device(
                raw_batch_device=slot.raw_batch_device,
                real_batch_device=slot.real_batch_device,
            )
            if ready:
                self.finalize_display_image_device(out=slot.output_device)
            slot.compute_done.record(self.compute_stream)

        slot.display_ready = ready
        self._compute_queue.append(slot)
        return slot.sequence

    def schedule_oldest_compute_batch(self, tick: int | None) -> int:
        if not self._compute_queue:
            raise RuntimeError("No computed stream batch is ready for D2H.")

        slot = self._compute_queue.popleft()
        if slot.sequence is None:
            raise RuntimeError("Cannot schedule D2H for an unassigned stream slot.")

        stage = "D2H" if slot.display_ready else "D2H no-output"
        label = self._stage_label(stage, slot.sequence, tick)
        with self.d2h_stream, time_range(label, color_id=122):
            self.d2h_stream.wait_event(slot.compute_done)
            if slot.display_ready:
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
        slot.display_ready = False
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
        while self._h2d_queue or self._compute_queue:
            if self._h2d_queue:
                self.schedule_oldest_h2d_batch(tick=None)
            if self._compute_queue:
                self.schedule_oldest_compute_batch(tick=None)

        with time_range("streams final H2D sync", color_id=125):
            self.h2d_stream.synchronize()
        with time_range("streams final compute sync", color_id=125):
            self.compute_stream.synchronize()
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

    def _submit_pipeline_tick(self) -> int:
        return self.pipeline.submit_pipeline_tick(next(self.host_batch_iter))

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
        submitted = 0
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
                    while (
                        submitted - completed
                        < self.pipeline.runtime.pipeline_prefetch_batches
                    ):
                        self._submit_pipeline_tick()
                        submitted += 1

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
