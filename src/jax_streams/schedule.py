from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import jax
import numpy as np

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo
from holoflow_benchmarks.stats import RunMeasurement

from .compute import PowerDopplerPipeline
from .nvtx import time_range
from .runtime import (
    DummyGilThread,
    start_dummy_gil_thread,
    stop_dummy_gil_thread,
)


@dataclass(frozen=True)
class JaxManagedStreamRuntimeConfig:
    """Logical in-flight scheduler parameters for JAX-managed CUDA streams.

    JAX/XLA owns CUDA stream creation and dependency tracking. `num_slots`
    therefore acts as a logical upper bound rather than a ring of explicit
    CUDA buffers, and `pipeline_prefetch_batches` controls how many submitted
    display outputs stay ahead of the oldest collected output.
    """

    num_slots: int = 32
    pipeline_prefetch_batches: int = 8

    def __post_init__(self) -> None:
        if self.num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {self.num_slots}.")
        if self.pipeline_prefetch_batches <= 0:
            raise ValueError(
                "pipeline_prefetch_batches must be positive, got "
                f"{self.pipeline_prefetch_batches}."
            )
        if self.pipeline_prefetch_batches > self.num_slots:
            raise ValueError(
                "pipeline_prefetch_batches must be <= num_slots for JAX-managed "
                f"in-flight outputs, got {self.pipeline_prefetch_batches} and "
                f"{self.num_slots}."
            )


@dataclass(frozen=True)
class PendingStreamOutput:
    sequence: int
    image_device: jax.Array


@dataclass(frozen=True)
class StreamOutput:
    sequence: int
    image: np.ndarray


class JaxManagedStreamPowerDopplerPipeline(PowerDopplerPipeline):
    """Submit multiple JAX-dispatched outputs without explicit CUDA streams."""

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
        runtime: JaxManagedStreamRuntimeConfig,
    ) -> None:
        self.runtime = runtime
        super().__init__(
            info=info,
            params=params,
            mode=mode,
            reuse_display_output=False,
        )
        self._next_sequence = 0
        self._pending_outputs: deque[PendingStreamOutput] = deque()

    @property
    def pending_output_count(self) -> int:
        return len(self._pending_outputs)

    def submit_batch(self, host_batch: np.ndarray) -> int:
        return self._submit(host_batch, tick=None)

    def submit_pipeline_tick(self, host_batch: np.ndarray) -> int:
        return self._submit(host_batch, tick=self._next_sequence)

    def _submit(self, host_batch: np.ndarray, tick: int | None) -> int:
        sequence = self._next_sequence
        self._next_sequence += 1

        label = (
            f"jax-streams submit batch {sequence}"
            if tick is None
            else f"jax-streams tick {tick}: submit batch {sequence}"
        )
        with time_range(label, color_id=520):
            if self.sliding_mean.will_be_ready_after_push:
                display_image = self.process_ready_batch_and_finalize_display_device(
                    host_batch
                )
                with time_range("jax-streams async host copy", color_id=522):
                    display_image.copy_to_host_async()
                self._pending_outputs.append(
                    PendingStreamOutput(
                        sequence=sequence,
                        image_device=display_image,
                    )
                )
            else:
                ready = self.process_batch(host_batch)
                if ready:
                    raise RuntimeError(
                        "JAX stream scheduler missed the ready-output path."
                    )

        return sequence

    def wait_for_one_output(self) -> StreamOutput:
        if not self._pending_outputs:
            raise RuntimeError("No JAX-managed stream output is pending.")

        pending = self._pending_outputs.popleft()
        with time_range("jax-streams wait output", color_id=524):
            pending.image_device.block_until_ready()

        with time_range("jax-streams collect output", color_id=525):
            return StreamOutput(
                sequence=pending.sequence,
                image=np.asarray(pending.image_device).copy(),
            )

    def finish(self) -> None:
        while self._pending_outputs:
            self.wait_for_one_output()
        self.synchronize()


class JaxManagedStreamBenchmarkRunner:
    """Prime, warm up, and measure JAX's managed asynchronous dispatch."""

    def __init__(
        self,
        pipeline: JaxManagedStreamPowerDopplerPipeline,
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

        print("Priming JAX-managed stream sliding window...")
        with time_range("jax-streams prime sliding window", color_id=526):
            for _ in range(batches_to_prime):
                self._submit_next_batch()
            self.pipeline.finish()

    def warmup(self) -> None:
        print("Warming up JAX-managed stream pipeline...")
        with time_range("jax-streams warmup outputs", color_id=527):
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

        print("Running JAX-managed stream steady-state benchmark...")

        try:
            dummy_thread, previous_switch_interval = start_dummy_gil_thread(mode)

            with time_range(
                "jax-streams steady-state benchmark loop",
                color_id=528,
            ):
                t0 = perf_counter()
                while elapsed < params.benchmark_seconds:
                    while (
                        self.pipeline.pending_output_count
                        < self.pipeline.runtime.pipeline_prefetch_batches
                    ):
                        self._submit_pipeline_tick()

                    output = self.pipeline.wait_for_one_output()
                    completed += 1
                    last_image = output.image

                    elapsed = perf_counter() - t0

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


__all__ = [
    "JaxManagedStreamBenchmarkRunner",
    "JaxManagedStreamPowerDopplerPipeline",
    "JaxManagedStreamRuntimeConfig",
    "PendingStreamOutput",
    "StreamOutput",
]
