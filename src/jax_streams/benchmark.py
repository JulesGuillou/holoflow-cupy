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

from .compute import doppler_bin_range
from .nvtx import time_range
from .runtime import clear_jax_runtime
from .schedule import (
    JaxManagedStreamBenchmarkRunner,
    JaxManagedStreamPowerDopplerPipeline,
    JaxManagedStreamRuntimeConfig,
)


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
    runtime: JaxManagedStreamRuntimeConfig,
) -> tuple[np.ndarray, BenchmarkStats]:
    validate_host_batches(host_batches, info, params)
    clear_jax_runtime()

    mode_name = _stream_mode_name(mode, runtime)

    print()
    print("=" * 80)
    print(f"Mode: {mode_name}")
    print(
        f"  precompute_static_tensors={mode.precompute_static_tensors} | "
        f"preallocate_work_buffers={mode.preallocate_work_buffers} | "
        f"num_slots={runtime.num_slots} | "
        f"pipeline_prefetch_batches={runtime.pipeline_prefetch_batches}"
    )
    print("  CUDA streams/events are JAX-managed; no explicit stream objects are used.")
    if mode.preallocate_work_buffers:
        print("  JAX preallocation mode uses sliding-window buffer donation hints.")
    if mode.enable_dummy_gil_thread:
        print(
            f"  dummy_gil_inner_loops={mode.dummy_gil_inner_loops} | "
            f"dummy_gil_switch_interval_s={mode.dummy_gil_switch_interval_s}"
        )

    with time_range(f"jax-streams init mode {mode.name}", color_id=529):
        pipeline = JaxManagedStreamPowerDopplerPipeline(
            info=info,
            params=params,
            mode=mode,
            runtime=runtime,
        )
        runner = JaxManagedStreamBenchmarkRunner(
            pipeline=pipeline,
            host_batch_iter=cycle_batches(host_batches),
        )

    with time_range(f"jax-streams run mode {mode.name}", color_id=530):
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

    clear_jax_runtime()
    return measurement.image, stats


def benchmark_suite(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    modes: Sequence[ExecutionMode],
    runtime: JaxManagedStreamRuntimeConfig,
) -> list[BenchmarkSeries]:
    def run_mode(mode: ExecutionMode) -> tuple[np.ndarray, BenchmarkStats]:
        return benchmark_mode(
            host_batches=host_batches,
            info=info,
            params=params,
            mode=mode,
            runtime=runtime,
        )

    return run_repeated_modes(
        modes=modes,
        repetitions=params.benchmark_repetitions,
        run_mode=run_mode,
    )


def _stream_mode_name(
    mode: ExecutionMode,
    runtime: JaxManagedStreamRuntimeConfig,
) -> str:
    name = mode.name
    for prefix in (
        "cupy-naive",
        "cupy-threaded",
        "cupy-streams",
        "pytorch-naive",
        "pytorch-threaded",
        "pytorch-streams",
        "jax-naive",
        "jax-streams",
    ):
        if name.startswith(prefix):
            name = name.replace(prefix, "jax-streams", 1)
            break
    else:
        name = f"jax-streams | {name}"

    return (
        f"{name} | logical_slots={runtime.num_slots} | "
        f"pipeline_prefetch={runtime.pipeline_prefetch_batches}"
    )


__all__ = ["benchmark_mode", "benchmark_suite"]
