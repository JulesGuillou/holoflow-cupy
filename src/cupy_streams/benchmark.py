from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from cupyx.profiler import time_range

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo, cycle_batches, validate_host_batches
from holoflow_benchmarks.runtime import clear_cupy_pools
from holoflow_benchmarks.stats import BenchmarkStats, make_benchmark_stats

from .compute import doppler_bin_range
from .schedule import (
    SingleThreadStreamBenchmarkRunner,
    SingleThreadStreamPowerDopplerPipeline,
    SingleThreadStreamRuntimeConfig,
)


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
    runtime: SingleThreadStreamRuntimeConfig,
) -> tuple[np.ndarray, BenchmarkStats]:
    validate_host_batches(host_batches, info, params)
    clear_cupy_pools()

    mode_name = _stream_mode_name(mode, runtime)

    print()
    print("=" * 80)
    print(f"Mode: {mode_name}")
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

