from __future__ import annotations

import torch

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo
from pytorch_naive.compute import (
    PercentileClipDisplay2D,
    SlidingMean2D,
    _accumulate_power,
    _cast_to_dtype,
    _copy_to_real_buffer,
    _fftshift,
    _fresnel_fft,
    _temporal_fft_band_select,
    centered_coordinates,
    doppler_bin_range,
    make_normalized_elliptical_mask,
    make_quadratic_phase,
)

from .dtypes import torch_dtype
from .nvtx import time_range


def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("The PyTorch threaded benchmark requires a CUDA device.")

    return torch.device("cuda")


class BatchPowerComputer:
    """FFT-heavy power Doppler math for the threaded PyTorch compute worker."""

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
        device: torch.device | None = None,
    ) -> None:
        self.info = info
        self.params = params
        self.mode = mode
        self.device = cuda_device() if device is None else device

        self.height = info.height
        self.width = info.width
        self.batch_frames = params.batch_frames
        self.real_dtype = torch_dtype(params.real_dtype)

        self.k0, self.k1 = doppler_bin_range(
            window_size=params.batch_frames,
            sample_rate_hz=params.sample_rate_hz,
            doppler_low_hz=params.doppler_low_hz,
            doppler_high_hz=params.doppler_high_hz,
        )

        with time_range("pytorch-threaded init FFT tensors", color_id=253):
            self.quadratic_phase = (
                make_quadratic_phase(self.height, self.width, params, self.device)
                if mode.precompute_static_tensors
                else None
            )
            self.real_batch_device = (
                torch.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=self.real_dtype,
                    device=self.device,
                )
                if mode.preallocate_work_buffers
                else None
            )

    @property
    def doppler_bins(self) -> tuple[int, int]:
        return self.k0, self.k1

    def _quadratic_phase(self) -> torch.Tensor:
        if self.quadratic_phase is not None:
            return self.quadratic_phase

        return make_quadratic_phase(self.height, self.width, self.params, self.device)

    def compute(self, raw_batch_device: torch.Tensor) -> torch.Tensor:
        with time_range("pytorch-threaded cast to f32", color_id=254):
            if self.real_batch_device is None:
                real_batch = _cast_to_dtype(raw_batch_device, self.real_dtype)
            else:
                real_batch = _copy_to_real_buffer(
                    raw_batch_device,
                    self.real_batch_device,
                )

        with time_range("pytorch-threaded temporal FFT + band select", color_id=254):
            temporal_spectrum = _temporal_fft_band_select(
                real_batch,
                self.k0,
                self.k1,
            )

        with time_range("pytorch-threaded prepare Fresnel phase", color_id=254):
            phase = self._quadratic_phase()

        with time_range("pytorch-threaded Fresnel", color_id=254):
            propagated = _fresnel_fft(temporal_spectrum, phase)

        with time_range("pytorch-threaded accumulate power", color_id=254):
            return _accumulate_power(propagated)


def fftshift_image(image: torch.Tensor) -> torch.Tensor:
    return _fftshift(image)


__all__ = [
    "BatchPowerComputer",
    "PercentileClipDisplay2D",
    "SlidingMean2D",
    "centered_coordinates",
    "cuda_device",
    "doppler_bin_range",
    "fftshift_image",
    "make_normalized_elliptical_mask",
    "make_quadratic_phase",
]
