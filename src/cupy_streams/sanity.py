from __future__ import annotations

import cupyx
import numpy as np

from cupy_naive.config import ExecutionMode, Params
from cupy_naive.cupy_naive import PowerDopplerPipeline, clear_cupy_pools
from cupy_naive.io import InputInfo, cycle_batches

from .cupy_streams import (
    SingleThreadStreamPowerDopplerPipeline,
    SingleThreadStreamRuntimeConfig,
)


def run_sanity_check() -> None:
    """Compare the stream scheduler against the naive pipeline on tiny data."""
    params = _sanity_params()
    info = InputInfo(height=16, width=16, dtype=np.dtype("uint8"))
    host_batches = _sanity_batches(info, params)

    for mode in _sanity_modes():
        clear_cupy_pools()
        expected = _run_naive_once(host_batches, info, params, mode)

        for num_slots in (2, 3):
            clear_cupy_pools()
            actual = _run_stream_once(host_batches, info, params, mode, num_slots)

            if actual.shape != expected.shape:
                raise AssertionError(
                    f"{mode.name}, num_slots={num_slots}: shape mismatch, "
                    f"got {actual.shape}, expected {expected.shape}."
                )
            if actual.dtype != expected.dtype:
                raise AssertionError(
                    f"{mode.name}, num_slots={num_slots}: dtype mismatch, "
                    f"got {actual.dtype}, expected {expected.dtype}."
                )

            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    clear_cupy_pools()
    print("Stream sanity check passed for num_slots=2 and num_slots=3.")


def _sanity_params() -> Params:
    return Params(
        file_path="synthetic",
        batch_frames=8,
        batches_per_output=3,
        sample_rate_hz=8000.0,
        doppler_low_hz=1000.0,
        doppler_high_hz=3000.0,
        propagation_distance_m=0.486,
        wavelength_m=8.52e-7,
        dx_m=2.0e-5,
        dy_m=2.0e-5,
        benchmark_seconds=0.01,
        warmup_outputs=0,
        show_image=False,
        contrast_roi_radius=0.8,
        contrast_low_percentile=0.2,
        contrast_high_percentile=99.8,
        report_path="cupy_streams_sanity_report.txt",
        acquisition_dtype=np.dtype("uint8"),
        real_dtype=np.dtype("float32"),
        complex_dtype=np.dtype("complex64"),
    )


def _sanity_modes() -> list[ExecutionMode]:
    modes: list[ExecutionMode] = []
    for precompute in (False, True):
        for preallocate in (False, True):
            modes.append(
                ExecutionMode(
                    name=(
                        "cupy-naive | "
                        f"precompute={'on' if precompute else 'off'} | "
                        f"prealloc={'on' if preallocate else 'off'} | "
                        "gil=off"
                    ),
                    precompute_static_tensors=precompute,
                    preallocate_work_buffers=preallocate,
                    enable_dummy_gil_thread=False,
                )
            )
    return modes


def _sanity_batches(info: InputInfo, params: Params) -> np.ndarray:
    rng = np.random.default_rng(1234)
    host_batches = cupyx.empty_pinned(
        (
            params.sliding_window_batches,
            params.batch_frames,
            info.height,
            info.width,
        ),
        dtype=info.dtype,
    )
    host_batches[...] = rng.integers(
        0,
        256,
        size=host_batches.shape,
        dtype=info.dtype,
    )
    return host_batches


def _run_naive_once(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
) -> np.ndarray:
    pipeline = PowerDopplerPipeline(info=info, params=params, mode=mode)
    host_batch_iter = cycle_batches(host_batches)

    for _ in range(params.sliding_window_batches - 1):
        ready = pipeline.process_batch(next(host_batch_iter))
        if ready:
            raise RuntimeError("Naive sanity pipeline became ready too early.")

    ready = pipeline.process_batch(next(host_batch_iter))
    if not ready:
        raise RuntimeError("Naive sanity pipeline did not produce an output.")

    return pipeline.export_display_image()


def _run_stream_once(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
    num_slots: int,
) -> np.ndarray:
    pipeline = SingleThreadStreamPowerDopplerPipeline(
        info=info,
        params=params,
        mode=mode,
        runtime=SingleThreadStreamRuntimeConfig(num_slots=num_slots),
    )
    host_batch_iter = cycle_batches(host_batches)

    for _ in range(params.sliding_window_batches - 1):
        pipeline.submit_batch(next(host_batch_iter))

    pipeline.submit_batch(next(host_batch_iter))
    output = pipeline.wait_for_one_output()
    pipeline.finish()
    return output.image


if __name__ == "__main__":
    run_sanity_check()
