from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo, cycle_batches, validate_host_batches
from holoflow_benchmarks.stats import (
    BenchmarkSeries,
    BenchmarkStats,
    make_benchmark_stats,
    run_repeated_modes,
)

from .compute import PowerDopplerPipeline
from .nvtx import time_range
from .runtime import clear_torch_pools
from .schedule import BenchmarkRunner


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
) -> tuple[np.ndarray, BenchmarkStats]:
    validate_host_batches(host_batches, info, params)
    clear_torch_pools()

    with time_range(f"pytorch init mode {mode.name}", color_id=247):
        pipeline = PowerDopplerPipeline(info=info, params=params, mode=mode)
        runner = BenchmarkRunner(
            pipeline=pipeline,
            host_batch_iter=cycle_batches(host_batches),
        )

    print()
    print("=" * 80)
    print(f"Mode: {mode.name}")
    print(
        f"  precompute_static_tensors={mode.precompute_static_tensors} | "
        f"preallocate_work_buffers={mode.preallocate_work_buffers} | "
        f"dummy_gil_thread={mode.enable_dummy_gil_thread}"
    )
    if mode.enable_dummy_gil_thread:
        print(
            f"  dummy_gil_inner_loops={mode.dummy_gil_inner_loops} | "
            f"dummy_gil_switch_interval_s={mode.dummy_gil_switch_interval_s}"
        )

    with time_range(f"pytorch run mode {mode.name}", color_id=248):
        runner.prime()
        runner.warmup()
        measurement = runner.run()

    stats = make_benchmark_stats(
        mode_name=mode.name,
        mode=mode,
        params=params,
        info=info,
        measurement=measurement,
        doppler_bins=pipeline.doppler_bins,
    )

    clear_torch_pools()
    return measurement.image, stats


def benchmark_suite(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    modes: Sequence[ExecutionMode],
) -> list[BenchmarkSeries]:
    def run_mode(mode: ExecutionMode) -> tuple[np.ndarray, BenchmarkStats]:
        return benchmark_mode(
            host_batches=host_batches,
            info=info,
            params=params,
            mode=mode,
        )

    return run_repeated_modes(
        modes=modes,
        repetitions=params.benchmark_repetitions,
        run_mode=run_mode,
    )
