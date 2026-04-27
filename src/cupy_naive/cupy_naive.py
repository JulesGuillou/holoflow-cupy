from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter
import sys
import threading

import cupy as cp
import cupyx
import numpy as np
from cupyx.profiler import time_range

from .config import ExecutionMode, Params
from .io import InputInfo, bytes_per_frame, cycle_batches


# ============================================================================
# Benchmark result model
# ============================================================================


@dataclass(frozen=True)
class BenchmarkStats:
    # Mode description
    mode_name: str
    precompute_static_tensors: bool
    preallocate_work_buffers: bool

    # Wall-clock totals
    seconds: float
    frames: int
    batches: int
    outputs: int

    # Rates
    input_fps: float
    batches_per_second: float
    outputs_per_second: float
    wall_ms_per_output: float

    # Effective bandwidths
    h2d_gbps: float
    cast_effective_gbps: float
    d2h_output_mbps: float

    # Data description
    shape: tuple[int, int]
    file_dtype: str
    host_dtype: str
    device_input_dtype: str
    real_dtype: str
    complex_dtype: str

    # Pipeline description
    doppler_bins: tuple[int, int]
    doppler_bin_count: int
    batch_frames: int
    batches_per_output: int
    frames_per_output: int
    output_stride_frames: int
    temporal_support_ms: float

    # Optional GIL stress
    dummy_gil_thread_enabled: bool
    dummy_gil_inner_loops: int
    dummy_gil_iterations: int
    dummy_gil_iterations_per_second: float
    dummy_gil_switch_interval_s: float | None


# ============================================================================
# Host-side scheduling utilities
# ============================================================================


class DummyGilThread(threading.Thread):
    """Pure-Python background work used to create intentional GIL contention."""

    def __init__(self, inner_loops: int) -> None:
        super().__init__(name="dummy-gil-thread", daemon=True)
        self._inner_loops = inner_loops
        self._stop_requested = threading.Event()
        self._ready = threading.Event()
        self.iterations = 0

    def wait_until_ready(self) -> None:
        self._ready.wait()

    def stop(self) -> None:
        self._stop_requested.set()

    def run(self) -> None:
        x = 0
        self._ready.set()

        while not self._stop_requested.is_set():
            for i in range(self._inner_loops):
                x = (x * 1664525 + 1013904223 + i) & 0xFFFFFFFF
            self.iterations += self._inner_loops

        _ = x


def start_dummy_gil_thread(
    mode: ExecutionMode,
) -> tuple[DummyGilThread | None, float | None]:
    if not mode.enable_dummy_gil_thread:
        return None, None

    previous_switch_interval = None
    if mode.dummy_gil_switch_interval_s is not None:
        previous_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(mode.dummy_gil_switch_interval_s)

    thread = DummyGilThread(mode.dummy_gil_inner_loops)
    thread.start()
    thread.wait_until_ready()
    return thread, previous_switch_interval


def stop_dummy_gil_thread(
    thread: DummyGilThread | None,
    previous_switch_interval: float | None,
) -> None:
    if thread is not None:
        thread.stop()
        thread.join()

    if previous_switch_interval is not None:
        sys.setswitchinterval(previous_switch_interval)


def clear_cupy_pools() -> None:
    """Release unused cached blocks between benchmark modes.

    This does not invalidate still-live arrays. It only frees blocks currently
    held by CuPy's allocators.
    """
    cp.cuda.get_current_stream().synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ============================================================================
# Signal-model helpers
# ============================================================================


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

    x = centered_coordinates(width, params.dx_m, params.real_dtype)
    y = centered_coordinates(height, params.dy_m, params.real_dtype)

    radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
    phase = radius_sq * real(np.pi / (params.wavelength_m * params.propagation_distance_m))

    return cp.exp(1j * phase).astype(params.complex_dtype, copy=False)


def make_normalized_elliptical_mask(
    height: int,
    width: int,
    radius: float,
) -> cp.ndarray:
    """Return the normalized elliptical display ROI mask.

    The support is defined by u^2 + v^2 <= r^2 in normalized image coordinates.
    """
    dtype = np.dtype(np.float32)
    real = dtype.type

    x = centered_coordinates(width, 2.0 / width, dtype)
    y = centered_coordinates(height, 2.0 / height, dtype)

    radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
    return radius_sq <= real(radius * radius)


# ============================================================================
# Device-side operators
# ============================================================================


class SlidingMean2D:
    r"""Causal sliding mean over the last M batch-images.

    If S_n(y, x) is the power image produced by batch n, this object maintains

        R_n(y, x) = sum_{m=n-M+1}^{n} S_m(y, x),

    and returns

        mean_n(y, x) = R_n(y, x) / M

    once the window is full.
    """

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
        slot = self.buffer[self._head]

        self.rolling_sum[...] -= slot
        slot[...] = image
        self.rolling_sum[...] += slot

        self._head = (self._head + 1) % self.window_length
        if self._fill < self.window_length:
            self._fill += 1

        return self.is_full

    def mean(self) -> cp.ndarray:
        if self.mean_image is None:
            return self.rolling_sum / self._normalization

        cp.divide(self.rolling_sum, self._normalization, out=self.mean_image)
        return self.mean_image


class PercentileClipDisplay2D:
    r"""Display-stage percentile clipping in a normalized elliptical ROI.

    Percentile bounds are estimated only inside the ROI and then applied globally
    to the whole image.
    """

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

    def apply(self, image: cp.ndarray) -> cp.ndarray:
        roi_values = image[self._roi_mask()]
        q_low, q_high = cp.percentile(
            roi_values,
            [self.low_percentile, self.high_percentile],
        )

        if self.output is None:
            return cp.clip(image, q_low, q_high)

        cp.clip(image, q_low, q_high, out=self.output)
        return self.output


# ============================================================================
# CuPy-naive benchmark implementation
# ============================================================================


class PowerDopplerPipeline:
    r"""Device-side pipeline from one temporal batch to one displayed image.

    For a temporal batch U0[t, y, x], the CuPy-naive pipeline computes:

        1. temporal_spectrum[k, y, x] = rFFT_t(U0)[k]
        2. keep k in [k0, k1)
        3. propagated[k, y, x] = FFT_xy(temporal_spectrum[k, y, x] * Q[y, x])
        4. batch_power[y, x] = sum_k |propagated[k, y, x]|^2

    Across successive batches, it then maintains the causal sliding mean:

        display_mean_n[y, x] = (1 / M) sum_{m=n-M+1}^{n} batch_power_m[y, x].

    Finally, for visualization only, it applies fftshift, percentile clipping,
    and device-to-host export.
    """

    def __init__(self, info: InputInfo, params: Params, mode: ExecutionMode) -> None:
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

        self.quadratic_phase = (
            make_quadratic_phase(self.height, self.width, params)
            if mode.precompute_static_tensors
            else None
        )

        if mode.preallocate_work_buffers:
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
            reuse_output_buffer=mode.preallocate_work_buffers,
        )

    def _quadratic_phase(self) -> cp.ndarray:
        if self.quadratic_phase is not None:
            return self.quadratic_phase

        return make_quadratic_phase(self.height, self.width, self.params)

    @time_range("process_batch", color_id=0)
    def process_batch(self, host_batch: np.ndarray) -> bool:
        batch_power = self._compute_batch_power(host_batch)
        return self.sliding_mean.push(batch_power)

    @time_range("finalize_output", color_id=6)
    def export_display_image(self) -> np.ndarray:
        with time_range("average", color_id=7):
            averaged = self.sliding_mean.mean()

        with time_range("fftshift", color_id=8):
            shifted = cp.fft.fftshift(averaged)

        with time_range("percentile clip", color_id=9):
            display_image = self.display_clipper.apply(shifted)

        with time_range("D2H output", color_id=10):
            if self.output_host is None:
                return display_image.get()

            display_image.get(out=self.output_host)

            # Return an owning NumPy array so later iterations cannot mutate
            # the image stored by reporting/plotting code.
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

        with time_range("cast to f32", color_id=2):
            if self.real_batch_device is None:
                real_batch_device = raw_batch_device.astype(
                    self.params.real_dtype,
                    copy=False,
                )
            else:
                cp.copyto(self.real_batch_device, raw_batch_device, casting="unsafe")
                real_batch_device = self.real_batch_device

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


# ============================================================================
# Benchmark schedule
# ============================================================================


class BenchmarkRunner:
    """Explicit three-phase schedule: prime, warm up, then measure."""

    def __init__(
        self,
        pipeline: PowerDopplerPipeline,
        host_batch_iter: Iterator[np.ndarray],
    ) -> None:
        self.pipeline = pipeline
        self.host_batch_iter = host_batch_iter

    def prime(self) -> None:
        """Fill the sliding window with M - 1 batches.

        After this phase, exactly one additional processed batch is sufficient
        to produce the first valid output.
        """
        batches_to_prime = self.pipeline.params.sliding_window_batches - 1

        print("Priming sliding window...")
        for _ in range(batches_to_prime):
            ready = self.pipeline.process_batch(next(self.host_batch_iter))
            if ready:
                raise RuntimeError("Sliding window became ready too early.")

        cp.cuda.get_current_stream().synchronize()

    def warmup(self) -> None:
        """Produce a small number of outputs before timed measurement."""
        print("Warming up...")
        for _ in range(self.pipeline.params.warmup_outputs):
            ready = self.pipeline.process_batch(next(self.host_batch_iter))
            if not ready:
                raise RuntimeError("Sliding window should be ready during warmup.")
            self.pipeline.export_display_image()

        cp.cuda.get_current_stream().synchronize()

    def run(self) -> tuple[np.ndarray, BenchmarkStats]:
        params = self.pipeline.params
        info = self.pipeline.info
        mode = self.pipeline.mode

        total_batches = 0
        total_frames = 0
        total_outputs = 0
        last_image: np.ndarray | None = None
        elapsed = 0.0

        dummy_thread: DummyGilThread | None = None
        previous_switch_interval: float | None = None

        print("Running steady-state benchmark...")

        try:
            dummy_thread, previous_switch_interval = start_dummy_gil_thread(mode)

            t0 = perf_counter()
            while elapsed < params.benchmark_seconds:
                ready = self.pipeline.process_batch(next(self.host_batch_iter))
                if not ready:
                    raise RuntimeError("Sliding window unexpectedly lost readiness.")

                total_batches += 1
                total_frames += params.batch_frames

                last_image = self.pipeline.export_display_image()
                total_outputs += 1

                # This is intentionally synchronized per output. It keeps the
                # naive execution model simple and makes wall-clock measurements
                # correspond to completed display images.
                cp.cuda.get_current_stream().synchronize()
                elapsed = perf_counter() - t0

        finally:
            stop_dummy_gil_thread(dummy_thread, previous_switch_interval)

        if last_image is None:
            raise RuntimeError("Benchmark produced no output image.")

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

        dummy_iterations = 0 if dummy_thread is None else dummy_thread.iterations

        stats = BenchmarkStats(
            mode_name=mode.name,
            precompute_static_tensors=mode.precompute_static_tensors,
            preallocate_work_buffers=mode.preallocate_work_buffers,
            seconds=elapsed,
            frames=total_frames,
            batches=total_batches,
            outputs=total_outputs,
            input_fps=total_frames / elapsed,
            batches_per_second=total_batches / elapsed,
            outputs_per_second=total_outputs / elapsed,
            wall_ms_per_output=1e3 * elapsed / total_outputs,
            h2d_gbps=(total_frames * input_frame_bytes) / elapsed / 1e9,
            cast_effective_gbps=(total_frames * cast_frame_bytes) / elapsed / 1e9,
            d2h_output_mbps=(total_outputs * output_image_bytes) / elapsed / 1e6,
            shape=(info.height, info.width),
            file_dtype=str(info.dtype),
            host_dtype=str(info.dtype),
            device_input_dtype=str(params.acquisition_dtype),
            real_dtype=str(params.real_dtype),
            complex_dtype=str(params.complex_dtype),
            doppler_bins=(self.pipeline.k0, self.pipeline.k1),
            doppler_bin_count=self.pipeline.doppler_bin_count,
            batch_frames=params.batch_frames,
            batches_per_output=params.batches_per_output,
            frames_per_output=params.temporal_support_frames,
            output_stride_frames=params.output_stride_frames,
            temporal_support_ms=1e3 * params.temporal_support_seconds,
            dummy_gil_thread_enabled=mode.enable_dummy_gil_thread,
            dummy_gil_inner_loops=mode.dummy_gil_inner_loops,
            dummy_gil_iterations=dummy_iterations,
            dummy_gil_iterations_per_second=dummy_iterations / elapsed,
            dummy_gil_switch_interval_s=mode.dummy_gil_switch_interval_s,
        )

        return last_image, stats


def benchmark_mode(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    mode: ExecutionMode,
) -> tuple[np.ndarray, BenchmarkStats]:
    expected_shape = (
        params.sliding_window_batches,
        params.batch_frames,
        info.height,
        info.width,
    )
    if host_batches.shape != expected_shape:
        raise ValueError(
            f"Unexpected preloaded host batch shape: got {host_batches.shape}, "
            f"expected {expected_shape}."
        )

    clear_cupy_pools()

    pipeline = PowerDopplerPipeline(info=info, params=params, mode=mode)
    runner = BenchmarkRunner(
        pipeline=pipeline,
        host_batch_iter=cycle_batches(host_batches),
    )

    print()
    print("=" * 80)
    print(f"Mode: {mode.name}")
    print(
        f"  precompute_static_tensors={mode.precompute_static_tensors} | "
        f"preallocate_work_buffers={mode.preallocate_work_buffers} | "
        f"dummy_gil_thread={mode.enable_dummy_gil_thread}"
    )
    if mode.enable_dummy_gil_thread:
        print(
            f"  dummy_gil_inner_loops={mode.dummy_gil_inner_loops} | "
            f"dummy_gil_switch_interval_s={mode.dummy_gil_switch_interval_s}"
        )

    runner.prime()
    runner.warmup()
    image, stats = runner.run()

    clear_cupy_pools()
    return image, stats


def benchmark_suite(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
    modes: Sequence[ExecutionMode],
) -> list[tuple[np.ndarray, BenchmarkStats]]:
    results: list[tuple[np.ndarray, BenchmarkStats]] = []

    for mode in modes:
        image, stats = benchmark_mode(
            host_batches=host_batches,
            info=info,
            params=params,
            mode=mode,
        )
        results.append((image, stats))

    return results
