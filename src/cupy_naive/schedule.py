from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter

import cupy as cp
import numpy as np
from cupyx.profiler import time_range

from holoflow_benchmarks.runtime import (
    DummyGilThread,
    start_dummy_gil_thread,
    stop_dummy_gil_thread,
)
from holoflow_benchmarks.stats import RunMeasurement

from .compute import PowerDopplerPipeline


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

        print("Priming sliding window...")
        with time_range("prime sliding window", color_id=42):
            for _ in range(batches_to_prime):
                ready = self.pipeline.process_batch(next(self.host_batch_iter))
                if ready:
                    raise RuntimeError("Sliding window became ready too early.")

        with time_range("sync after prime", color_id=43):
            cp.cuda.get_current_stream().synchronize()

    def warmup(self) -> None:
        """Produce a small number of outputs before timed measurement."""
        print("Warming up...")
        with time_range("warmup outputs", color_id=44):
            for _ in range(self.pipeline.params.warmup_outputs):
                ready = self.pipeline.process_batch(next(self.host_batch_iter))
                if not ready:
                    raise RuntimeError("Sliding window should be ready during warmup.")
                self.pipeline.export_display_image()

        with time_range("sync after warmup", color_id=45):
            cp.cuda.get_current_stream().synchronize()

    def run(self) -> RunMeasurement:
        params = self.pipeline.params
        mode = self.pipeline.mode

        total_batches = 0
        total_outputs = 0
        last_image: np.ndarray | None = None
        elapsed = 0.0

        dummy_thread: DummyGilThread | None = None
        previous_switch_interval: float | None = None

        print("Running steady-state benchmark...")

        try:
            dummy_thread, previous_switch_interval = start_dummy_gil_thread(mode)

            with time_range("steady-state benchmark loop", color_id=46):
                t0 = perf_counter()
                while elapsed < params.benchmark_seconds:
                    ready = self.pipeline.process_batch(next(self.host_batch_iter))
                    if not ready:
                        raise RuntimeError(
                            "Sliding window unexpectedly lost readiness."
                        )

                    total_batches += 1

                    last_image = self.pipeline.export_display_image()
                    total_outputs += 1

                    # The naive model measures fully completed display images.
                    with time_range("sync per output", color_id=17):
                        cp.cuda.get_current_stream().synchronize()
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

