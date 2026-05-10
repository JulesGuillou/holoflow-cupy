from __future__ import annotations

import numpy as np
import torch

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo

from .dtypes import torch_dtype
from .nvtx import nvtx_range, time_range


COMPILE_MODE = "default"


@torch.compile(mode=COMPILE_MODE)
def _cast_to_dtype(raw_batch_device: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return raw_batch_device.to(dtype=dtype, copy=False)


@torch.compile(mode=COMPILE_MODE)
def _copy_to_real_buffer(
    raw_batch_device: torch.Tensor,
    real_batch_device: torch.Tensor,
) -> torch.Tensor:
    real_batch_device.copy_(raw_batch_device)
    return real_batch_device


@torch.compile(mode=COMPILE_MODE)
def _temporal_fft_band_select(
    real_batch: torch.Tensor,
    k0: int,
    k1: int,
) -> torch.Tensor:
    return torch.fft.rfft(real_batch, dim=0)[k0:k1]


@torch.compile(mode=COMPILE_MODE)
def _fresnel_fft(
    temporal_spectrum: torch.Tensor,
    quadratic_phase: torch.Tensor,
) -> torch.Tensor:
    return torch.fft.fft2(temporal_spectrum * quadratic_phase, dim=(-2, -1))


@torch.compile(mode=COMPILE_MODE)
def _accumulate_power(propagated: torch.Tensor) -> torch.Tensor:
    return propagated.abs().square().sum(dim=0)


@torch.compile(mode=COMPILE_MODE)
def _divide_mean(rolling_sum: torch.Tensor, normalization: float) -> torch.Tensor:
    return rolling_sum / normalization


@torch.compile(mode=COMPILE_MODE)
def _divide_mean_out(
    rolling_sum: torch.Tensor,
    normalization: float,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.div(rolling_sum, normalization, out=out)
    return out


@torch.compile(mode=COMPILE_MODE)
def _fftshift(image: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(image)


@torch.compile(mode=COMPILE_MODE)
def _display_clip(
    image: torch.Tensor,
    roi_indices: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    bounds = torch.quantile(torch.take(image, roi_indices), quantiles)
    return torch.clamp(image, min=bounds[0], max=bounds[1])


@torch.compile(mode=COMPILE_MODE)
def _display_clip_out(
    image: torch.Tensor,
    roi_indices: torch.Tensor,
    quantiles: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    bounds = torch.quantile(torch.take(image, roi_indices), quantiles)
    torch.clamp(image, min=bounds[0], max=bounds[1], out=out)
    return out


def doppler_bin_range(
    window_size: int,
    sample_rate_hz: float,
    doppler_low_hz: float,
    doppler_high_hz: float,
) -> tuple[int, int]:
    """Return the inclusive-exclusive rFFT bin range covering [f0, f1]."""
    freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate_hz)

    k0 = int(np.searchsorted(freqs, doppler_low_hz, side="left"))
    k1 = int(np.searchsorted(freqs, doppler_high_hz, side="right"))

    if k0 >= k1:
        raise ValueError(
            f"Empty Doppler band for window_size={window_size}, "
            f"fs={sample_rate_hz}, f0={doppler_low_hz}, f1={doppler_high_hz}."
        )

    return k0, k1


def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("The PyTorch naive benchmark requires a CUDA device.")

    return torch.device("cuda")


def centered_coordinates(
    length: int,
    pitch: float,
    dtype: np.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return centered coordinates: x[n] = (n - (N - 1) / 2) * pitch."""
    return (
        torch.arange(length, dtype=torch_dtype(dtype), device=device)
        - float((length - 1) / 2.0)
    ) * float(pitch)


def make_quadratic_phase(
    height: int,
    width: int,
    params: Params,
    device: torch.device,
) -> torch.Tensor:
    r"""Build the Fresnel input quadratic phase.

    Q(x, y) = exp(i pi (x^2 + y^2) / (lambda z))
    """
    with time_range("pytorch build Fresnel coordinates", color_id=231):
        x = centered_coordinates(width, params.dx_m, params.real_dtype, device)
        y = centered_coordinates(height, params.dy_m, params.real_dtype, device)

    with time_range("pytorch build Fresnel phase", color_id=232):
        radius_sq = torch.square(y)[:, None] + torch.square(x)[None, :]
        phase = radius_sq * float(
            np.pi / (params.wavelength_m * params.propagation_distance_m)
        )

    with time_range("pytorch materialize Fresnel phase", color_id=233):
        return torch.exp(1j * phase).to(dtype=torch_dtype(params.complex_dtype))


def make_normalized_elliptical_mask(
    height: int,
    width: int,
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the normalized elliptical display ROI mask."""
    dtype = np.dtype(np.float32)

    with time_range("pytorch build ROI coordinates", color_id=234):
        x = centered_coordinates(width, 2.0 / width, dtype, device)
        y = centered_coordinates(height, 2.0 / height, dtype, device)

    with time_range("pytorch build ROI mask", color_id=235):
        radius_sq = torch.square(y)[:, None] + torch.square(x)[None, :]
        return radius_sq <= float(radius * radius)


class SlidingMean2D:
    r"""Causal sliding mean over the last M batch-images."""

    def __init__(
        self,
        window_length: int,
        height: int,
        width: int,
        dtype: np.dtype,
        device: torch.device,
        reuse_mean_buffer: bool,
    ) -> None:
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}.")

        tensor_dtype = torch_dtype(dtype)
        self.window_length = window_length
        self.buffer = torch.zeros(
            (window_length, height, width),
            dtype=tensor_dtype,
            device=device,
        )
        self.rolling_sum = torch.zeros((height, width), dtype=tensor_dtype, device=device)
        self.mean_image = (
            torch.empty((height, width), dtype=tensor_dtype, device=device)
            if reuse_mean_buffer
            else None
        )

        self._head = 0
        self._fill = 0
        self._normalization = float(window_length)

    @property
    def is_full(self) -> bool:
        return self._fill == self.window_length

    def push(self, image: torch.Tensor) -> bool:
        with time_range("pytorch sliding mean update", color_id=212):
            slot = self.buffer[self._head]

            self.rolling_sum.sub_(slot)
            slot.copy_(image)
            self.rolling_sum.add_(slot)

            self._head = (self._head + 1) % self.window_length
            if self._fill < self.window_length:
                self._fill += 1

        return self.is_full

    def mean(self) -> torch.Tensor:
        with time_range("pytorch sliding mean divide", color_id=213):
            if self.mean_image is None:
                return _divide_mean(self.rolling_sum, self._normalization)

            return _divide_mean_out(
                self.rolling_sum,
                self._normalization,
                self.mean_image,
            )


class PercentileClipDisplay2D:
    """Display-stage percentile clipping in a normalized elliptical ROI."""

    def __init__(
        self,
        height: int,
        width: int,
        dtype: np.dtype,
        device: torch.device,
        roi_radius: float,
        low_percentile: float,
        high_percentile: float,
        precompute_mask: bool,
        reuse_output_buffer: bool,
    ) -> None:
        if not (0.0 < roi_radius <= 1.0):
            raise ValueError(f"roi_radius must lie in (0, 1], got {roi_radius}.")
        if not (0.0 <= low_percentile < high_percentile <= 100.0):
            raise ValueError(
                "Percentiles must satisfy 0 <= low < high <= 100, got "
                f"{low_percentile}, {high_percentile}."
            )

        self.height = height
        self.width = width
        self.dtype = dtype
        self.device = device
        self.roi_radius = roi_radius
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.quantiles = torch.tensor(
            [low_percentile / 100.0, high_percentile / 100.0],
            dtype=torch_dtype(dtype),
            device=device,
        )
        self.roi_mask: torch.Tensor | None = None
        self.roi_indices: torch.Tensor | None = None

        with time_range("pytorch init display clipper", color_id=236):
            if precompute_mask:
                self.roi_mask = make_normalized_elliptical_mask(
                    height,
                    width,
                    roi_radius,
                    device,
                )
                self.roi_indices = torch.nonzero(
                    self.roi_mask.ravel(),
                    as_tuple=False,
                ).squeeze(1)

            self.output = (
                torch.empty(
                    (height, width),
                    dtype=torch_dtype(dtype),
                    device=device,
                )
                if reuse_output_buffer
                else None
            )

    def _roi_mask(self) -> torch.Tensor:
        if self.roi_mask is not None:
            return self.roi_mask

        self.roi_mask = make_normalized_elliptical_mask(
            height=self.height,
            width=self.width,
            radius=self.roi_radius,
            device=self.device,
        )
        return self.roi_mask

    def _roi_indices(self) -> torch.Tensor:
        if self.roi_indices is not None:
            return self.roi_indices

        # The compacted index tensor has a fixed shape after construction, so
        # the compiled display clip path avoids dynamic-size mask indexing.
        with time_range("pytorch build ROI indices", color_id=234):
            self.roi_indices = torch.nonzero(
                self._roi_mask().ravel(),
                as_tuple=False,
            ).squeeze(1)
        return self.roi_indices

    def apply(
        self,
        image: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with time_range("pytorch display percentile clip", color_id=216):
            roi_indices = self._roi_indices()
            if out is not None:
                return _display_clip_out(image, roi_indices, self.quantiles, out)

            if self.output is None:
                return _display_clip(image, roi_indices, self.quantiles)

            return _display_clip_out(
                image,
                roi_indices,
                self.quantiles,
                self.output,
            )


class PowerDopplerPipeline:
    r"""Single-thread, default-stream PyTorch pipeline for one temporal batch.

    For a temporal batch U0[t, y, x], the pipeline computes:

    1. temporal_spectrum[k, y, x] = rFFT_t(U0)[k]
    2. keep k in [k0, k1)
    3. propagated[k, y, x] = FFT_xy(temporal_spectrum[k, y, x] * Q[y, x])
    4. batch_power[y, x] = sum_k |propagated[k, y, x]|^2

    Across successive batches, it maintains a causal sliding mean and applies
    display-only fftshift and percentile clipping before export.
    """

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
        *,
        allocate_batch_io_buffers: bool = True,
        reuse_display_output: bool | None = None,
    ) -> None:
        self.info = info
        self.params = params
        self.mode = mode
        self.device = cuda_device()

        self.height = info.height
        self.width = info.width
        self.batch_frames = params.batch_frames
        self.acquisition_dtype = torch_dtype(params.acquisition_dtype)
        self.real_dtype = torch_dtype(params.real_dtype)

        self.k0, self.k1 = doppler_bin_range(
            window_size=params.batch_frames,
            sample_rate_hz=params.sample_rate_hz,
            doppler_low_hz=params.doppler_low_hz,
            doppler_high_hz=params.doppler_high_hz,
        )
        self.doppler_bin_count = self.k1 - self.k0

        with time_range("pytorch init static tensors", color_id=237):
            self.quadratic_phase = (
                make_quadratic_phase(self.height, self.width, params, self.device)
                if mode.precompute_static_tensors
                else None
            )

        with time_range("pytorch init work buffers", color_id=238):
            if mode.preallocate_work_buffers and allocate_batch_io_buffers:
                self.raw_batch_device = torch.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=self.acquisition_dtype,
                    device=self.device,
                )
                self.real_batch_device = torch.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=self.real_dtype,
                    device=self.device,
                )
                self.output_host = torch.empty(
                    (self.height, self.width),
                    dtype=self.real_dtype,
                    pin_memory=True,
                )
            else:
                self.raw_batch_device = None
                self.real_batch_device = None
                self.output_host = None

        with time_range("pytorch init sliding mean", color_id=239):
            self.sliding_mean = SlidingMean2D(
                window_length=params.sliding_window_batches,
                height=self.height,
                width=self.width,
                dtype=params.real_dtype,
                device=self.device,
                reuse_mean_buffer=mode.preallocate_work_buffers,
            )

        self.display_clipper = PercentileClipDisplay2D(
            height=self.height,
            width=self.width,
            dtype=params.real_dtype,
            device=self.device,
            roi_radius=params.contrast_roi_radius,
            low_percentile=params.contrast_low_percentile,
            high_percentile=params.contrast_high_percentile,
            precompute_mask=mode.precompute_static_tensors,
            reuse_output_buffer=(
                mode.preallocate_work_buffers
                if reuse_display_output is None
                else reuse_display_output
            ),
        )

    @property
    def doppler_bins(self) -> tuple[int, int]:
        return self.k0, self.k1

    def _quadratic_phase(self) -> torch.Tensor:
        if self.quadratic_phase is not None:
            return self.quadratic_phase

        return make_quadratic_phase(self.height, self.width, self.params, self.device)

    @nvtx_range("pytorch process_batch", color_id=200)
    def process_batch(self, host_batch: np.ndarray) -> bool:
        batch_power = self._compute_batch_power(host_batch)
        return self.sliding_mean.push(batch_power)

    @nvtx_range("pytorch process_batch_device", color_id=200)
    def process_batch_device(
        self,
        raw_batch_device: torch.Tensor,
        real_batch_device: torch.Tensor | None = None,
    ) -> bool:
        batch_power = self.compute_batch_power_device(
            raw_batch_device=raw_batch_device,
            real_batch_device=real_batch_device,
        )
        return self.sliding_mean.push(batch_power)

    @nvtx_range("pytorch finalize_output", color_id=206)
    def finalize_display_image_device(
        self,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with time_range("pytorch average", color_id=207):
            averaged = self.sliding_mean.mean()

        with time_range("pytorch fftshift", color_id=208):
            shifted = _fftshift(averaged)

        with time_range("pytorch percentile clip", color_id=209):
            return self.display_clipper.apply(shifted, out=out)

    @nvtx_range("pytorch export_display_image", color_id=206)
    def export_display_image(self) -> np.ndarray:
        display_image = self.finalize_display_image_device()

        with time_range("pytorch D2H output", color_id=210):
            if self.output_host is None:
                return display_image.cpu().numpy()

            self.output_host.copy_(display_image, non_blocking=False)
            return self.output_host.numpy().copy()

    def _compute_batch_power(self, host_batch: np.ndarray) -> torch.Tensor:
        with time_range("pytorch H2D upload", color_id=201):
            host_tensor = torch.from_numpy(host_batch)
            if self.raw_batch_device is None:
                raw_batch_device = host_tensor.to(
                    device=self.device,
                    dtype=self.acquisition_dtype,
                    non_blocking=True,
                )
            else:
                self.raw_batch_device.copy_(host_tensor, non_blocking=True)
                raw_batch_device = self.raw_batch_device

        return self.compute_batch_power_device(raw_batch_device=raw_batch_device)

    def compute_batch_power_device(
        self,
        raw_batch_device: torch.Tensor,
        real_batch_device: torch.Tensor | None = None,
    ) -> torch.Tensor:
        real_work_buffer = (
            real_batch_device
            if real_batch_device is not None
            else self.real_batch_device
        )

        with time_range("pytorch cast to f32", color_id=202):
            if real_work_buffer is None:
                real_batch = _cast_to_dtype(raw_batch_device, self.real_dtype)
            else:
                real_batch = _copy_to_real_buffer(
                    raw_batch_device,
                    real_work_buffer,
                )

        with time_range("pytorch temporal FFT + band select", color_id=203):
            temporal_spectrum = _temporal_fft_band_select(
                real_batch,
                self.k0,
                self.k1,
            )

        with time_range("pytorch prepare Fresnel phase", color_id=204):
            quadratic_phase = self._quadratic_phase()

        with time_range("pytorch Fresnel", color_id=205):
            propagated = _fresnel_fft(temporal_spectrum, quadratic_phase)

        with time_range("pytorch accumulate power", color_id=211):
            return _accumulate_power(propagated)
