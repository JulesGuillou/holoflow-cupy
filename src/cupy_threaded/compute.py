from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.profiler import time_range

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo


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


def centered_coordinates(length: int, pitch: float, dtype: np.dtype) -> cp.ndarray:
    """Return centered coordinates: x[n] = (n - (N - 1) / 2) * pitch."""
    scalar = np.dtype(dtype).type
    center = scalar((length - 1) / 2.0)
    step = scalar(pitch)
    return (cp.arange(length, dtype=dtype) - center) * step


def make_quadratic_phase(height: int, width: int, params: Params) -> cp.ndarray:
    r"""Build the Fresnel input quadratic phase.

    Q(x, y) = exp(i pi (x^2 + y^2) / (lambda z))
    """
    real = np.dtype(params.real_dtype).type

    with time_range("threaded build Fresnel coordinates", color_id=53):
        x = centered_coordinates(width, params.dx_m, params.real_dtype)
        y = centered_coordinates(height, params.dy_m, params.real_dtype)

    with time_range("threaded build Fresnel phase", color_id=53):
        radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
        phase = radius_sq * real(
            np.pi / (params.wavelength_m * params.propagation_distance_m)
        )

    with time_range("threaded materialize Fresnel phase", color_id=53):
        return cp.exp(1j * phase).astype(params.complex_dtype, copy=False)


def make_normalized_elliptical_mask(
    height: int,
    width: int,
    radius: float,
) -> cp.ndarray:
    """Return the normalized elliptical display ROI mask."""
    dtype = np.dtype(np.float32)
    real = dtype.type

    with time_range("threaded build ROI coordinates", color_id=61):
        x = centered_coordinates(width, 2.0 / width, dtype)
        y = centered_coordinates(height, 2.0 / height, dtype)

    with time_range("threaded build ROI mask", color_id=61):
        radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
        return radius_sq <= real(radius * radius)


class BatchPowerComputer:
    """FFT-heavy power Doppler math for the threaded compute worker."""

    def __init__(
        self,
        info: InputInfo,
        params: Params,
        mode: ExecutionMode,
    ) -> None:
        self.info = info
        self.params = params
        self.mode = mode
        self.height = info.height
        self.width = info.width
        self.batch_frames = params.batch_frames

        self.k0, self.k1 = doppler_bin_range(
            window_size=params.batch_frames,
            sample_rate_hz=params.sample_rate_hz,
            doppler_low_hz=params.doppler_low_hz,
            doppler_high_hz=params.doppler_high_hz,
        )

        with time_range("threaded init FFT tensors", color_id=53):
            self.quadratic_phase = (
                make_quadratic_phase(self.height, self.width, params)
                if mode.precompute_static_tensors
                else None
            )
            self.real_batch_device = (
                cp.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=params.real_dtype,
                )
                if mode.preallocate_work_buffers
                else None
            )

    @property
    def doppler_bins(self) -> tuple[int, int]:
        return self.k0, self.k1

    def _quadratic_phase(self) -> cp.ndarray:
        if self.quadratic_phase is not None:
            return self.quadratic_phase

        return make_quadratic_phase(self.height, self.width, self.params)

    def compute(self, raw_batch_device: cp.ndarray) -> cp.ndarray:
        with time_range("threaded cast to f32", color_id=54):
            if self.real_batch_device is None:
                real_batch = raw_batch_device.astype(
                    self.params.real_dtype,
                    copy=False,
                )
            else:
                cp.copyto(
                    self.real_batch_device,
                    raw_batch_device,
                    casting="unsafe",
                )
                real_batch = self.real_batch_device

        with time_range("threaded temporal FFT + band select", color_id=54):
            temporal_spectrum = cp.fft.rfft(real_batch, axis=0)[self.k0 : self.k1]

        with time_range("threaded prepare Fresnel phase", color_id=54):
            phase = self._quadratic_phase()

        with time_range("threaded Fresnel", color_id=54):
            propagated = cp.fft.fft2(
                temporal_spectrum * phase,
                axes=(-2, -1),
            )

        with time_range("threaded accumulate power", color_id=54):
            return (cp.abs(propagated) ** 2).sum(axis=0)


class SlidingMean2D:
    """Causal sliding mean over the last M batch-images."""

    def __init__(
        self,
        window_length: int,
        height: int,
        width: int,
        dtype: np.dtype,
        reuse_mean_buffer: bool,
    ) -> None:
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}.")

        self.window_length = window_length
        self.buffer = cp.zeros((window_length, height, width), dtype=dtype)
        self.rolling_sum = cp.zeros((height, width), dtype=dtype)
        self.mean_image = (
            cp.empty((height, width), dtype=dtype) if reuse_mean_buffer else None
        )

        self._head = 0
        self._fill = 0
        self._normalization = np.dtype(dtype).type(window_length)

    @property
    def is_full(self) -> bool:
        return self._fill == self.window_length

    def push(self, image: cp.ndarray) -> bool:
        with time_range("threaded sliding mean update", color_id=62):
            slot = self.buffer[self._head]

            self.rolling_sum[...] -= slot
            slot[...] = image
            self.rolling_sum[...] += slot

            self._head = (self._head + 1) % self.window_length
            if self._fill < self.window_length:
                self._fill += 1

        return self.is_full

    def mean(self) -> cp.ndarray:
        with time_range("threaded sliding mean divide", color_id=62):
            if self.mean_image is None:
                return self.rolling_sum / self._normalization

            cp.divide(self.rolling_sum, self._normalization, out=self.mean_image)
            return self.mean_image


class PercentileClipDisplay2D:
    """Display-stage percentile clipping in a normalized elliptical ROI."""

    def __init__(
        self,
        height: int,
        width: int,
        dtype: np.dtype,
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
        self.roi_radius = roi_radius
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

        with time_range("threaded init display clipper", color_id=61):
            self.roi_mask = (
                make_normalized_elliptical_mask(height, width, roi_radius)
                if precompute_mask
                else None
            )
            self.output = (
                cp.empty((height, width), dtype=dtype) if reuse_output_buffer else None
            )

    def _roi_mask(self) -> cp.ndarray:
        if self.roi_mask is not None:
            return self.roi_mask

        return make_normalized_elliptical_mask(
            height=self.height,
            width=self.width,
            radius=self.roi_radius,
        )

    def apply(self, image: cp.ndarray, out: cp.ndarray | None = None) -> cp.ndarray:
        with time_range("threaded select display ROI", color_id=62):
            roi_values = image[self._roi_mask()]

        with time_range("threaded display percentiles", color_id=62):
            q_low, q_high = cp.percentile(
                roi_values,
                [self.low_percentile, self.high_percentile],
            )

        with time_range("threaded display clip", color_id=62):
            if out is not None:
                cp.clip(image, q_low, q_high, out=out)
                return out

            if self.output is None:
                return cp.clip(image, q_low, q_high)

            cp.clip(image, q_low, q_high, out=self.output)
            return self.output

