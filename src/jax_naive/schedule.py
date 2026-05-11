from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter

import numpy as np

from holoflow_benchmarks.stats import RunMeasurement

from .compute import PowerDopplerPipeline
from .nvtx import time_range
from .runtime import (
    DummyGilThread,
    start_dummy_gil_thread,
    stop_dummy_gil_thread,
)


class BenchmarkRunner:
    """Explicit three-phase schedule: prime, warm up, then measure."""

    def __init__(
        self,
        pipeline: PowerDopplerPipeline,
        host_batch_iter: Iterator[np.ndarray],
    ) -> None:
        self.pipeline = pipeline
        self.host_batch_iter = host_batch_iter

    def prime(self) -> None:
        """Fill the sliding window with M - 1 batches."""
        batches_to_prime = self.pipeline.params.sliding_window_batches - 1

        print("Priming JAX sliding window...")
        with time_range("jax prime sliding window", color_id=442):
            for batch_index in range(batches_to_prime):
                with time_range(f"jax prime batch {batch_index}", color_id=446):
                    ready = self.pipeline.process_batch(next(self.host_batch_iter))
                if ready:
                    raise RuntimeError("Sliding window became ready too early.")

        with time_range("jax sync after prime", color_id=443):
            self.pipeline.synchronize()

    def warmup(self) -> None:
        """Produce a small number of outputs before timed measurement."""
        print("Warming up JAX pipeline...")
        with time_range("jax warmup outputs", color_id=444):
            for output_index in range(self.pipeline.params.warmup_outputs):
                with time_range(f"jax warmup output {output_index}", color_id=446):
                    display_image = (
                        self.pipeline.process_ready_batch_and_finalize_display_device(
                            next(self.host_batch_iter)
                        )
                    )
                    self.pipeline.copy_display_image_to_host(display_image)

        with time_range("jax sync after warmup", color_id=445):
            self.pipeline.synchronize()

    def run(self) -> RunMeasurement:
        params = self.pipeline.params
        mode = self.pipeline.mode

        total_batches = 0
        total_outputs = 0
        last_image: np.ndarray | None = None
        elapsed = 0.0

        dummy_thread: DummyGilThread | None = None
        previous_switch_interval: float | None = None

        print("Running JAX steady-state benchmark...")

        try:
            dummy_thread, previous_switch_interval = start_dummy_gil_thread(mode)

            with time_range("jax steady-state benchmark loop", color_id=446):
                t0 = perf_counter()
                while elapsed < params.benchmark_seconds:
                    with time_range(
                        f"jax output {total_outputs}: process/export/sync",
                        color_id=447,
                    ):
                        display_image = (
                            self.pipeline.process_ready_batch_and_finalize_display_device(
                                next(self.host_batch_iter)
                            )
                        )
                        total_batches += 1

                        last_image = self.pipeline.copy_display_image_to_host(
                            display_image
                        )
                        total_outputs += 1

                    elapsed = perf_counter() - t0

        finally:
            stop_dummy_gil_thread(dummy_thread, previous_switch_interval)

        if last_image is None:
            raise RuntimeError("Benchmark produced no output image.")

        dummy_iterations = 0 if dummy_thread is None else dummy_thread.iterations
        return RunMeasurement(
            image=last_image,
            seconds=elapsed,
            batches=total_batches,
            outputs=total_outputs,
            dummy_gil_iterations=dummy_iterations,
            dummy_gil_switch_interval_s=mode.dummy_gil_switch_interval_s,
        )


__all__ = ["BenchmarkRunner"]
