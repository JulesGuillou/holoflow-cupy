from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

import numpy as np

from .config import ExecutionMode, Params
from .io import InputInfo, bytes_per_frame


@dataclass(frozen=True)
class RunMeasurement:
    """Small scheduler result kept separate from benchmark reporting."""

    image: np.ndarray
    seconds: float
    batches: int
    outputs: int
    dummy_gil_iterations: int
    dummy_gil_switch_interval_s: float | None


@dataclass(frozen=True)
class BenchmarkStats:
    mode_name: str
    precompute_static_tensors: bool
    preallocate_work_buffers: bool

    seconds: float
    frames: int
    batches: int
    outputs: int

    input_fps: float
    batches_per_second: float
    outputs_per_second: float
    wall_ms_per_output: float

    h2d_gbps: float
    cast_effective_gbps: float
    d2h_output_mbps: float

    shape: tuple[int, int]
    file_dtype: str
    host_dtype: str
    device_input_dtype: str
    real_dtype: str
    complex_dtype: str

    doppler_bins: tuple[int, int]
    doppler_bin_count: int
    batch_frames: int
    batches_per_output: int
    frames_per_output: int
    output_stride_frames: int
    temporal_support_ms: float

    dummy_gil_thread_enabled: bool
    dummy_gil_inner_loops: int
    dummy_gil_iterations: int
    dummy_gil_iterations_per_second: float
    dummy_gil_switch_interval_s: float | None


@dataclass(frozen=True)
class BenchmarkSeries:
    """Repeated steady-state measurements for one execution mode."""

    image: np.ndarray
    runs: tuple[BenchmarkStats, ...]

    def __post_init__(self) -> None:
        if not self.runs:
            raise ValueError("A benchmark series must contain at least one run.")
        mode_names = {run.mode_name for run in self.runs}
        if len(mode_names) != 1:
            raise ValueError("All runs in a benchmark series must use the same mode.")


@dataclass(frozen=True)
class ThroughputSummary:
    repetitions: int
    mean_input_fps: float
    sample_std_input_fps: float | None
    coefficient_of_variation_percent: float | None
    min_input_fps: float
    max_input_fps: float


def summarize_throughput(series: BenchmarkSeries) -> ThroughputSummary:
    values = [run.input_fps for run in series.runs]
    mean = fmean(values)
    sample_std = stdev(values) if len(values) >= 2 else None
    coefficient_of_variation = (
        100.0 * sample_std / mean
        if sample_std is not None and mean != 0.0
        else None
    )
    return ThroughputSummary(
        repetitions=len(values),
        mean_input_fps=mean,
        sample_std_input_fps=sample_std,
        coefficient_of_variation_percent=coefficient_of_variation,
        min_input_fps=min(values),
        max_input_fps=max(values),
    )


def run_repeated_modes(
    *,
    modes: Sequence[ExecutionMode],
    repetitions: int,
    run_mode: Callable[[ExecutionMode], tuple[np.ndarray, BenchmarkStats]],
) -> list[BenchmarkSeries]:
    """Run all repetitions of each mode before advancing to the next mode."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    results: list[BenchmarkSeries] = []
    for mode in modes:
        runs: list[BenchmarkStats] = []
        final_image: np.ndarray | None = None
        for repetition in range(1, repetitions + 1):
            print(f"\nRepetition {repetition}/{repetitions}: {mode.name}")
            final_image, stats = run_mode(mode)
            runs.append(stats)

        if final_image is None:  # Defensive; repetitions is validated above.
            raise RuntimeError("Benchmark mode produced no image.")
        results.append(BenchmarkSeries(image=final_image, runs=tuple(runs)))

    return results


def make_benchmark_stats(
    *,
    mode_name: str,
    mode: ExecutionMode,
    params: Params,
    info: InputInfo,
    measurement: RunMeasurement,
    doppler_bins: tuple[int, int],
) -> BenchmarkStats:
    if measurement.seconds <= 0.0:
        raise ValueError(f"Benchmark duration must be positive, got {measurement.seconds}.")
    if measurement.outputs <= 0:
        raise ValueError(f"Benchmark outputs must be positive, got {measurement.outputs}.")

    total_frames = measurement.batches * params.batch_frames
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

    return BenchmarkStats(
        mode_name=mode_name,
        precompute_static_tensors=mode.precompute_static_tensors,
        preallocate_work_buffers=mode.preallocate_work_buffers,
        seconds=measurement.seconds,
        frames=total_frames,
        batches=measurement.batches,
        outputs=measurement.outputs,
        input_fps=total_frames / measurement.seconds,
        batches_per_second=measurement.batches / measurement.seconds,
        outputs_per_second=measurement.outputs / measurement.seconds,
        wall_ms_per_output=1e3 * measurement.seconds / measurement.outputs,
        h2d_gbps=(total_frames * input_frame_bytes) / measurement.seconds / 1e9,
        cast_effective_gbps=(total_frames * cast_frame_bytes) / measurement.seconds / 1e9,
        d2h_output_mbps=(
            measurement.outputs * output_image_bytes
        )
        / measurement.seconds
        / 1e6,
        shape=(info.height, info.width),
        file_dtype=str(info.dtype),
        host_dtype=str(info.dtype),
        device_input_dtype=str(params.acquisition_dtype),
        real_dtype=str(params.real_dtype),
        complex_dtype=str(params.complex_dtype),
        doppler_bins=doppler_bins,
        doppler_bin_count=doppler_bins[1] - doppler_bins[0],
        batch_frames=params.batch_frames,
        batches_per_output=params.batches_per_output,
        frames_per_output=params.temporal_support_frames,
        output_stride_frames=params.output_stride_frames,
        temporal_support_ms=1e3 * params.temporal_support_seconds,
        dummy_gil_thread_enabled=mode.enable_dummy_gil_thread,
        dummy_gil_inner_loops=mode.dummy_gil_inner_loops,
        dummy_gil_iterations=measurement.dummy_gil_iterations,
        dummy_gil_iterations_per_second=(
            measurement.dummy_gil_iterations / measurement.seconds
        ),
        dummy_gil_switch_interval_s=measurement.dummy_gil_switch_interval_s,
    )
