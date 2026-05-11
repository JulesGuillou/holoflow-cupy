from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from holoflow_benchmarks.config import ExecutionMode, Params
from holoflow_benchmarks.io import InputInfo

from .dtypes import jax_dtype
from .nvtx import nvtx_range, time_range


def _compute_batch_power_core(
    raw_batch_device: jax.Array,
    quadratic_phase: jax.Array,
    *,
    real_dtype: np.dtype,
    k0: int,
    k1: int,
) -> jax.Array:
    real_batch = raw_batch_device.astype(real_dtype)
    temporal_spectrum = jnp.fft.rfft(real_batch, axis=0)[k0:k1]
    propagated = jnp.fft.fft2(temporal_spectrum * quadratic_phase, axes=(-2, -1))
    return jnp.square(jnp.abs(propagated)).sum(axis=0)


_compute_batch_power_impl = partial(
    jax.jit,
    static_argnames=("real_dtype", "k0", "k1"),
)(_compute_batch_power_core)


def _push_sliding_mean_core(
    buffer: jax.Array,
    rolling_sum: jax.Array,
    image: jax.Array,
    head: int,
) -> tuple[jax.Array, jax.Array]:
    slot = buffer[head]
    next_rolling_sum = rolling_sum - slot + image
    next_buffer = buffer.at[head].set(image)
    return next_buffer, next_rolling_sum


_push_sliding_mean_impl = jax.jit(_push_sliding_mean_core)
_push_sliding_mean_donate_impl = jax.jit(
    _push_sliding_mean_core,
    donate_argnums=(0, 1),
)


@partial(jax.jit, static_argnames=("normalization",))
def _divide_mean(
    rolling_sum: jax.Array,
    *,
    normalization: float,
) -> jax.Array:
    return rolling_sum / normalization


def _display_clip_core(
    image: jax.Array,
    roi_indices: jax.Array,
    *,
    low_percentile: float,
    high_percentile: float,
) -> jax.Array:
    percentiles = jnp.asarray(
        [low_percentile, high_percentile],
        dtype=image.dtype,
    )
    bounds = jnp.percentile(jnp.take(image, roi_indices), percentiles)
    return jnp.clip(image, bounds[0], bounds[1])


_display_clip = partial(
    jax.jit,
    static_argnames=("low_percentile", "high_percentile"),
)(_display_clip_core)


def _finalize_display_image_core(
    rolling_sum: jax.Array,
    roi_indices: jax.Array,
    *,
    normalization: float,
    low_percentile: float,
    high_percentile: float,
) -> jax.Array:
    averaged = rolling_sum / normalization
    shifted = jnp.fft.fftshift(averaged)
    return _display_clip_core(
        shifted,
        roi_indices,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )


_finalize_display_image_impl = partial(
    jax.jit,
    static_argnames=("normalization", "low_percentile", "high_percentile"),
)(_finalize_display_image_core)


def _process_ready_output_core(
    raw_batch_device: jax.Array,
    quadratic_phase: jax.Array,
    buffer: jax.Array,
    rolling_sum: jax.Array,
    roi_indices: jax.Array,
    head: int,
    *,
    real_dtype: np.dtype,
    k0: int,
    k1: int,
    normalization: float,
    low_percentile: float,
    high_percentile: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_power = _compute_batch_power_core(
        raw_batch_device,
        quadratic_phase,
        real_dtype=real_dtype,
        k0=k0,
        k1=k1,
    )
    next_buffer, next_rolling_sum = _push_sliding_mean_core(
        buffer,
        rolling_sum,
        batch_power,
        head,
    )
    display_image = _finalize_display_image_core(
        next_rolling_sum,
        roi_indices,
        normalization=normalization,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    return next_buffer, next_rolling_sum, display_image


_process_ready_output_impl = partial(
    jax.jit,
    static_argnames=(
        "real_dtype",
        "k0",
        "k1",
        "normalization",
        "low_percentile",
        "high_percentile",
    ),
)(_process_ready_output_core)


_process_ready_output_donate_impl = partial(
    jax.jit,
    static_argnames=(
        "real_dtype",
        "k0",
        "k1",
        "normalization",
        "low_percentile",
        "high_percentile",
    ),
    donate_argnums=(2, 3),
)(_process_ready_output_core)


@partial(
    jax.jit,
    static_argnames=(
        "height",
        "width",
        "real_dtype",
        "complex_dtype",
    ),
)
def _make_quadratic_phase_impl(
    *,
    height: int,
    width: int,
    dx_m: float,
    dy_m: float,
    wavelength_m: float,
    propagation_distance_m: float,
    real_dtype: np.dtype,
    complex_dtype: np.dtype,
) -> jax.Array:
    x = (
        jnp.arange(width, dtype=real_dtype) - float((width - 1) / 2.0)
    ) * dx_m
    y = (
        jnp.arange(height, dtype=real_dtype) - float((height - 1) / 2.0)
    ) * dy_m
    radius_sq = jnp.square(y)[:, None] + jnp.square(x)[None, :]
    phase = radius_sq * (
        jnp.asarray(np.pi, dtype=real_dtype)
        / (wavelength_m * propagation_distance_m)
    )
    return jnp.exp(1j * phase).astype(complex_dtype)


@partial(
    jax.jit,
    static_argnames=("height", "width", "radius"),
)
def _make_normalized_elliptical_mask_impl(
    *,
    height: int,
    width: int,
    radius: float,
) -> jax.Array:
    x = (jnp.arange(width, dtype=jnp.float32) - float((width - 1) / 2.0)) * (
        2.0 / width
    )
    y = (jnp.arange(height, dtype=jnp.float32) - float((height - 1) / 2.0)) * (
        2.0 / height
    )
    radius_sq = jnp.square(y)[:, None] + jnp.square(x)[None, :]
    return radius_sq <= float(radius * radius)


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


def cuda_device() -> jax.Device:
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        raise RuntimeError("The JAX benchmark requires a CUDA GPU backend.") from exc

    if not devices:
        raise RuntimeError("The JAX benchmark requires a CUDA GPU backend.")

    return devices[0]


def centered_coordinates(
    length: int,
    pitch: float,
    dtype: np.dtype,
    device: jax.Device,
) -> jax.Array:
    """Return centered coordinates: x[n] = (n - (N - 1) / 2) * pitch."""
    values = (
        jnp.arange(length, dtype=jax_dtype(dtype)) - float((length - 1) / 2.0)
    ) * float(pitch)
    return jax.device_put(values, device)


def make_quadratic_phase(
    height: int,
    width: int,
    params: Params,
    device: jax.Device,
) -> jax.Array:
    r"""Build the Fresnel input quadratic phase.

    Q(x, y) = exp(i pi (x^2 + y^2) / (lambda z))
    """
    with time_range("jax build Fresnel coordinates", color_id=431):
        phase = _make_quadratic_phase_impl(
            height=height,
            width=width,
            dx_m=params.dx_m,
            dy_m=params.dy_m,
            wavelength_m=params.wavelength_m,
            propagation_distance_m=params.propagation_distance_m,
            real_dtype=jax_dtype(params.real_dtype),
            complex_dtype=jax_dtype(params.complex_dtype),
        )

    with time_range("jax materialize Fresnel phase", color_id=433):
        return jax.device_put(phase, device)


def make_normalized_elliptical_mask(
    height: int,
    width: int,
    radius: float,
    device: jax.Device,
) -> jax.Array:
    """Return the normalized elliptical display ROI mask."""
    with time_range("jax build ROI mask", color_id=435):
        mask = _make_normalized_elliptical_mask_impl(
            height=height,
            width=width,
            radius=radius,
        )
        return jax.device_put(mask, device)


class SlidingMean2D:
    r"""Causal sliding mean over the last M batch-images."""

    def __init__(
        self,
        window_length: int,
        height: int,
        width: int,
        dtype: np.dtype,
        device: jax.Device,
        reuse_mean_buffer: bool,
    ) -> None:
        if window_length <= 0:
            raise ValueError(f"window_length must be positive, got {window_length}.")

        array_dtype = jax_dtype(dtype)
        self.window_length = window_length
        self.device = device
        self.buffer = jax.device_put(
            jnp.zeros((window_length, height, width), dtype=array_dtype),
            device,
        )
        self.rolling_sum = jax.device_put(
            jnp.zeros((height, width), dtype=array_dtype),
            device,
        )
        self._head = 0
        self._fill = 0
        self._normalization = float(window_length)
        self._donate_buffers = reuse_mean_buffer

    @property
    def is_full(self) -> bool:
        return self._fill == self.window_length

    def push(self, image: jax.Array) -> bool:
        with time_range("jax sliding mean update", color_id=412):
            update = (
                _push_sliding_mean_donate_impl
                if self._donate_buffers
                else _push_sliding_mean_impl
            )
            self.buffer, self.rolling_sum = update(
                self.buffer,
                self.rolling_sum,
                image,
                self._head,
            )

            self._head = (self._head + 1) % self.window_length
            if self._fill < self.window_length:
                self._fill += 1

        return self.is_full

    def mean(self) -> jax.Array:
        with time_range("jax sliding mean divide", color_id=413):
            return _divide_mean(
                self.rolling_sum,
                normalization=self._normalization,
            )

    def block_until_ready(self) -> None:
        self.rolling_sum.block_until_ready()

    @property
    def head(self) -> int:
        return self._head

    @property
    def normalization(self) -> float:
        return self._normalization

    @property
    def donate_buffers(self) -> bool:
        return self._donate_buffers

    @property
    def will_be_ready_after_push(self) -> bool:
        return self._fill >= self.window_length - 1

    def advance(self) -> bool:
        self._head = (self._head + 1) % self.window_length
        if self._fill < self.window_length:
            self._fill += 1
        return self.is_full

    def replace_state(
        self,
        buffer: jax.Array,
        rolling_sum: jax.Array,
    ) -> bool:
        self.buffer = buffer
        self.rolling_sum = rolling_sum
        return self.advance()


class PercentileClipDisplay2D:
    """Display-stage percentile clipping in a normalized elliptical ROI."""

    def __init__(
        self,
        height: int,
        width: int,
        dtype: np.dtype,
        device: jax.Device,
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
        self.roi_mask: jax.Array | None = None
        self.roi_indices: jax.Array | None = None
        _ = reuse_output_buffer

        with time_range("jax init display clipper", color_id=436):
            if precompute_mask:
                self.roi_mask = make_normalized_elliptical_mask(
                    height,
                    width,
                    roi_radius,
                    device,
                )
                self._roi_indices()

    def _roi_mask(self) -> jax.Array:
        if self.roi_mask is not None:
            return self.roi_mask

        self.roi_mask = make_normalized_elliptical_mask(
            height=self.height,
            width=self.width,
            radius=self.roi_radius,
            device=self.device,
        )
        return self.roi_mask

    def _roi_indices(self) -> jax.Array:
        if self.roi_indices is not None:
            return self.roi_indices

        with time_range("jax build ROI indices", color_id=434):
            mask_host = np.asarray(self._roi_mask())
            indices = np.nonzero(mask_host.ravel())[0].astype(np.int32, copy=False)
            self.roi_indices = jax.device_put(indices, self.device)
        return self.roi_indices

    def apply(
        self,
        image: jax.Array,
        out: jax.Array | None = None,
    ) -> jax.Array:
        _ = out
        with time_range("jax display percentile clip", color_id=416):
            return _display_clip(
                image,
                self._roi_indices(),
                low_percentile=self.low_percentile,
                high_percentile=self.high_percentile,
            )


class PowerDopplerPipeline:
    r"""Single-thread, JAX-dispatched device pipeline for one temporal batch.

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
        reuse_display_output: bool | None = None,
    ) -> None:
        self.info = info
        self.params = params
        self.mode = mode
        self.device = cuda_device()

        self.height = info.height
        self.width = info.width
        self.batch_frames = params.batch_frames
        self.acquisition_dtype = jax_dtype(params.acquisition_dtype)
        self.real_dtype = jax_dtype(params.real_dtype)

        self.k0, self.k1 = doppler_bin_range(
            window_size=params.batch_frames,
            sample_rate_hz=params.sample_rate_hz,
            doppler_low_hz=params.doppler_low_hz,
            doppler_high_hz=params.doppler_high_hz,
        )
        self.doppler_bin_count = self.k1 - self.k0

        with time_range("jax init static tensors", color_id=437):
            self.quadratic_phase = (
                make_quadratic_phase(self.height, self.width, params, self.device)
                if mode.precompute_static_tensors
                else None
            )

        with time_range("jax init sliding mean", color_id=439):
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

    def _quadratic_phase(self) -> jax.Array:
        if self.quadratic_phase is not None:
            return self.quadratic_phase

        return make_quadratic_phase(
            self.height,
            self.width,
            self.params,
            self.device,
        )

    @nvtx_range("jax process_batch", color_id=400)
    def process_batch(self, host_batch: np.ndarray) -> bool:
        batch_power = self._compute_batch_power(host_batch)
        return self.sliding_mean.push(batch_power)

    @nvtx_range("jax process_batch_device", color_id=400)
    def process_batch_device(self, raw_batch_device: jax.Array) -> bool:
        batch_power = self.compute_batch_power_device(raw_batch_device)
        return self.sliding_mean.push(batch_power)

    @nvtx_range("jax process_ready_batch_and_finalize_output", color_id=406)
    def process_ready_batch_and_finalize_display_device(
        self,
        host_batch: np.ndarray,
    ) -> jax.Array:
        with time_range("jax H2D upload", color_id=401):
            host_view = np.asarray(host_batch, dtype=self.params.acquisition_dtype)
            raw_batch_device = jax.device_put(host_view, self.device)

        with time_range("jax ready output step", color_id=418):
            process = (
                _process_ready_output_donate_impl
                if self.sliding_mean.donate_buffers
                else _process_ready_output_impl
            )
            buffer, rolling_sum, display_image = process(
                raw_batch_device,
                self._quadratic_phase(),
                self.sliding_mean.buffer,
                self.sliding_mean.rolling_sum,
                self.display_clipper._roi_indices(),
                self.sliding_mean.head,
                real_dtype=self.real_dtype,
                k0=self.k0,
                k1=self.k1,
                normalization=self.sliding_mean.normalization,
                low_percentile=self.display_clipper.low_percentile,
                high_percentile=self.display_clipper.high_percentile,
            )

        ready = self.sliding_mean.replace_state(buffer, rolling_sum)
        if not ready:
            raise RuntimeError("Ready-output JAX path used before window was ready.")
        return display_image

    @nvtx_range("jax finalize_output", color_id=406)
    def finalize_display_image_device(
        self,
        out: jax.Array | None = None,
    ) -> jax.Array:
        _ = out
        with time_range("jax compiled display finalize", color_id=409):
            return _finalize_display_image_impl(
                self.sliding_mean.rolling_sum,
                self.display_clipper._roi_indices(),
                normalization=self.sliding_mean.normalization,
                low_percentile=self.display_clipper.low_percentile,
                high_percentile=self.display_clipper.high_percentile,
            )

    @nvtx_range("jax export_display_image", color_id=406)
    def export_display_image(self) -> np.ndarray:
        display_image = self.finalize_display_image_device()
        return self.copy_display_image_to_host(display_image)

    def copy_display_image_to_host(self, display_image: jax.Array) -> np.ndarray:
        with time_range("jax D2H output", color_id=410):
            display_image.block_until_ready()
            return np.asarray(display_image).copy()

    def _compute_batch_power(self, host_batch: np.ndarray) -> jax.Array:
        with time_range("jax H2D upload", color_id=401):
            host_view = np.asarray(host_batch, dtype=self.params.acquisition_dtype)
            raw_batch_device = jax.device_put(host_view, self.device)

        return self.compute_batch_power_device(raw_batch_device)

    def compute_batch_power_device(self, raw_batch_device: jax.Array) -> jax.Array:
        with time_range("jax compute batch power", color_id=402):
            return _compute_batch_power_impl(
                raw_batch_device,
                self._quadratic_phase(),
                real_dtype=self.real_dtype,
                k0=self.k0,
                k1=self.k1,
            )

    def synchronize(self) -> None:
        with time_range("jax synchronize pipeline", color_id=417):
            self.sliding_mean.block_until_ready()


__all__ = [
    "PercentileClipDisplay2D",
    "PowerDopplerPipeline",
    "SlidingMean2D",
    "centered_coordinates",
    "cuda_device",
    "doppler_bin_range",
    "make_normalized_elliptical_mask",
    "make_quadratic_phase",
]
