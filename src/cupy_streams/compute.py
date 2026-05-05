from __future__ import annotations

import cupy as cp
import cupyx
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

    with time_range("build Fresnel coordinates", color_id=31):
        x = centered_coordinates(width, params.dx_m, params.real_dtype)
        y = centered_coordinates(height, params.dy_m, params.real_dtype)

    with time_range("build Fresnel phase", color_id=32):
        radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
        phase = radius_sq * real(
            np.pi / (params.wavelength_m * params.propagation_distance_m)
        )

    with time_range("materialize Fresnel phase", color_id=33):
        return cp.exp(1j * phase).astype(params.complex_dtype, copy=False)


def make_normalized_elliptical_mask(
    height: int,
    width: int,
    radius: float,
) -> cp.ndarray:
    """Return the normalized elliptical display ROI mask."""
    dtype = np.dtype(np.float32)
    real = dtype.type

    with time_range("build ROI coordinates", color_id=34):
        x = centered_coordinates(width, 2.0 / width, dtype)
        y = centered_coordinates(height, 2.0 / height, dtype)

    with time_range("build ROI mask", color_id=35):
        radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
        return radius_sq <= real(radius * radius)


class SlidingMean2D:
    r"""Causal sliding mean over the last M batch-images."""

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
        with time_range("sliding mean update", color_id=12):
            slot = self.buffer[self._head]

            self.rolling_sum[...] -= slot
            slot[...] = image
            self.rolling_sum[...] += slot

            self._head = (self._head + 1) % self.window_length
            if self._fill < self.window_length:
                self._fill += 1

        return self.is_full

    def mean(self) -> cp.ndarray:
        with time_range("sliding mean divide", color_id=13):
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

        self.roi_mask: cp.ndarray | None = None
        self.roi_indices: cp.ndarray | None = None

        with time_range("init display clipper", color_id=36):
            if precompute_mask:
                self.roi_mask = make_normalized_elliptical_mask(
                    height=height,
                    width=width,
                    radius=roi_radius,
                )
                self.roi_indices = cp.flatnonzero(self.roi_mask.ravel()).astype(
                    cp.int64
                )

            self.output = (
                cp.empty((height, width), dtype=dtype) if reuse_output_buffer else None
            )

    def _roi_mask(self) -> cp.ndarray:
        if self.roi_mask is not None:
            return self.roi_mask

        self.roi_mask = make_normalized_elliptical_mask(
            height=self.height,
            width=self.width,
            radius=self.roi_radius,
        )
        return self.roi_mask

    def _roi_indices(self) -> cp.ndarray:
        if self.roi_indices is not None:
            return self.roi_indices

        # This may synchronize once because flatnonzero also needs a compacted
        # output. That is acceptable during initialization, but not in apply().
        self.roi_indices = cp.flatnonzero(self._roi_mask().ravel()).astype(cp.int64)
        return self.roi_indices

    def apply(self, image: cp.ndarray, out: cp.ndarray | None = None) -> cp.ndarray:
        with time_range("select display ROI", color_id=14):
            roi_values = cp.take(image.ravel(), self._roi_indices())

        with time_range("display percentiles", color_id=15):
            q_low, q_high = cp.percentile(
                roi_values,
                [self.low_percentile, self.high_percentile],
            )

        with time_range("display clip", color_id=16):
            if out is not None:
                cp.clip(image, q_low, q_high, out=out)
                return out

            if self.output is None:
                return cp.clip(image, q_low, q_high)

            cp.clip(image, q_low, q_high, out=self.output)
            return self.output


class PowerDopplerPipeline:
    r"""Single-thread, single-stream device pipeline for one temporal batch.

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

        self.height = info.height
        self.width = info.width
        self.batch_frames = params.batch_frames

        self.k0, self.k1 = doppler_bin_range(
            window_size=params.batch_frames,
            sample_rate_hz=params.sample_rate_hz,
            doppler_low_hz=params.doppler_low_hz,
            doppler_high_hz=params.doppler_high_hz,
        )
        self.doppler_bin_count = self.k1 - self.k0

        with time_range("init static tensors", color_id=37):
            self.quadratic_phase = (
                make_quadratic_phase(self.height, self.width, params)
                if mode.precompute_static_tensors
                else None
            )

        with time_range("init work buffers", color_id=38):
            if mode.preallocate_work_buffers and allocate_batch_io_buffers:
                self.raw_batch_device = cp.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=params.acquisition_dtype,
                )
                self.real_batch_device = cp.empty(
                    (self.batch_frames, self.height, self.width),
                    dtype=params.real_dtype,
                )
                self.output_host = cupyx.empty_pinned(
                    (self.height, self.width),
                    dtype=params.real_dtype,
                )
            else:
                self.raw_batch_device = None
                self.real_batch_device = None
                self.output_host = None

        with time_range("init sliding mean", color_id=39):
            self.sliding_mean = SlidingMean2D(
                window_length=params.sliding_window_batches,
                height=self.height,
                width=self.width,
                dtype=params.real_dtype,
                reuse_mean_buffer=mode.preallocate_work_buffers,
            )

        self.display_clipper = PercentileClipDisplay2D(
            height=self.height,
            width=self.width,
            dtype=params.real_dtype,
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

    def _quadratic_phase(self) -> cp.ndarray:
        if self.quadratic_phase is not None:
            return self.quadratic_phase

        return make_quadratic_phase(self.height, self.width, self.params)

    @time_range("process_batch", color_id=0)
    def process_batch(self, host_batch: np.ndarray) -> bool:
        batch_power = self._compute_batch_power(host_batch)
        return self.sliding_mean.push(batch_power)

    @time_range("process_batch_device", color_id=0)
    def process_batch_device(
        self,
        raw_batch_device: cp.ndarray,
        real_batch_device: cp.ndarray | None = None,
    ) -> bool:
        batch_power = self.compute_batch_power_device(
            raw_batch_device=raw_batch_device,
            real_batch_device=real_batch_device,
        )
        return self.sliding_mean.push(batch_power)

    @time_range("finalize_output", color_id=6)
    def finalize_display_image_device(
        self, out: cp.ndarray | None = None
    ) -> cp.ndarray:
        with time_range("average", color_id=7):
            averaged = self.sliding_mean.mean()

        with time_range("fftshift", color_id=8):
            shifted = cp.fft.fftshift(averaged)

        with time_range("percentile clip", color_id=9):
            return self.display_clipper.apply(shifted, out=out)

    @time_range("export_display_image", color_id=6)
    def export_display_image(self) -> np.ndarray:
        display_image = self.finalize_display_image_device()

        with time_range("D2H output", color_id=10):
            if self.output_host is None:
                return display_image.get(blocking=True)

            display_image.get(out=self.output_host, blocking=True)
            return np.array(self.output_host, copy=True)

    def _compute_batch_power(self, host_batch: np.ndarray) -> cp.ndarray:
        with time_range("H2D upload", color_id=1):
            if self.raw_batch_device is None:
                raw_batch_device = cp.asarray(
                    host_batch,
                    dtype=self.params.acquisition_dtype,
                )
            else:
                self.raw_batch_device.set(host_batch)
                raw_batch_device = self.raw_batch_device

        return self.compute_batch_power_device(raw_batch_device=raw_batch_device)

    def compute_batch_power_device(
        self,
        raw_batch_device: cp.ndarray,
        real_batch_device: cp.ndarray | None = None,
    ) -> cp.ndarray:
        real_work_buffer = (
            real_batch_device
            if real_batch_device is not None
            else self.real_batch_device
        )

        with time_range("cast to f32", color_id=2):
            if real_work_buffer is None:
                real_batch_device = raw_batch_device.astype(
                    self.params.real_dtype,
                    copy=False,
                )
            else:
                cp.copyto(real_work_buffer, raw_batch_device, casting="unsafe")
                real_batch_device = real_work_buffer

        with time_range("temporal FFT + band select", color_id=3):
            temporal_spectrum = cp.fft.rfft(real_batch_device, axis=0)[
                self.k0 : self.k1
            ]

        with time_range("prepare Fresnel phase", color_id=4):
            quadratic_phase = self._quadratic_phase()

        with time_range("Fresnel", color_id=5):
            propagated = cp.fft.fft2(
                temporal_spectrum * quadratic_phase,
                axes=(-2, -1),
            )

        with time_range("accumulate power", color_id=11):
            power = cp.abs(propagated) ** 2
            return power.sum(axis=0)
