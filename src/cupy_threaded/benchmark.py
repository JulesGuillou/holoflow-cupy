from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo, cycle_batches, validate_host_batches
from holoflow_benchmarks.runtime import clear_cupy_pools
from holoflow_benchmarks.stats import BenchmarkStats, make_benchmark_stats

from .compute import doppler_bin_range
from .schedule import ThreadedBenchmarkRunner, ThreadedRuntimeConfig


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
    runtime: ThreadedRuntimeConfig,
) -> tuple[np.ndarray, BenchmarkStats]:
    validate_host_batches(host_batches, info, params)
    clear_cupy_pools()

    mode_name = _threaded_mode_name(mode)

    print()
    print("=" * 80)
    print(f"Mode: {mode_name}")
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
        measurement = runner.run()

    doppler_bins = doppler_bin_range(
        window_size=params.batch_frames,
        sample_rate_hz=params.sample_rate_hz,
        doppler_low_hz=params.doppler_low_hz,
        doppler_high_hz=params.doppler_high_hz,
    )
    stats = make_benchmark_stats(
        mode_name=mode_name,
        mode=mode,
        params=params,
        info=info,
        measurement=measurement,
        doppler_bins=doppler_bins,
    )

    clear_cupy_pools()
    return measurement.image, stats


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
    name = mode.name
    for prefix in ("cupy-naive", "cupy-streams"):
        if name.startswith(prefix):
            return name.replace(prefix, "cupy-threaded", 1)
    return name

