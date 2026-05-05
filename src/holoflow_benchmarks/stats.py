from __future__ import annotations

from dataclasses import dataclass

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

