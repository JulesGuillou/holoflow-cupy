from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter
import sys
import threading

import cupy as cp
import cupyx
import holofile
import matplotlib.pyplot as plt
import numpy as np
from cupyx.profiler import time_range
from tabulate import tabulate


# ============================================================================
# Problem description and benchmark modes
# ============================================================================


@dataclass(frozen=True)
class Params:
    """Signal, geometry, display, and benchmark settings.

    This object describes *what* is computed and *how long* it is benchmarked.
    Execution-style choices such as preallocation, precomputation, and host-side
    GIL stress live in `ExecutionMode`.

    Output-transfer batching is part of the schedule: one displayed image is
    still produced per processed temporal batch, but host transfers are grouped
    in batches of size `output_transfer_batch_size`.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    path: str = r"C:\Users\guill\Documents\holofiles_data\250527_GUJ_L_2.holo"

    # ------------------------------------------------------------------
    # Execution schedule
    # ------------------------------------------------------------------
    batch_frames: int = 32
    batches_per_output: int = 16  # Sliding-window length M in batches.
    output_transfer_batch_size: int = 16

    # ------------------------------------------------------------------
    # Signal / physics
    # ------------------------------------------------------------------
    sample_rate_hz: float = 37_000.0
    doppler_low_hz: float = 8_000.0
    doppler_high_hz: float = 16_000.0

    propagation_distance_m: float = 488e-3
    wavelength_m: float = 852e-9
    dx_m: float = 20e-6
    dy_m: float = 20e-6

    # ------------------------------------------------------------------
    # Benchmark control
    # ------------------------------------------------------------------
    benchmark_seconds: float = 10.0
    warmup_outputs: int = 1

    # ------------------------------------------------------------------
    # Display / contrast adjustment
    # ------------------------------------------------------------------
    show_image: bool = True
    contrast_roi_radius: float = 0.8
    contrast_low_percentile: float = 0.2
    contrast_high_percentile: float = 99.8

    # ------------------------------------------------------------------
    # Dtype contract
    # ------------------------------------------------------------------
    acquisition_dtype: type[np.generic] = np.uint8
    real_dtype: type[np.generic] = np.float32
    complex_dtype: type[np.generic] = np.complex64

    @property
    def sliding_window_batches(self) -> int:
        return self.batches_per_output

    @property
    def temporal_support_frames(self) -> int:
        return self.batch_frames * self.sliding_window_batches

    @property
    def output_stride_frames(self) -> int:
        return self.batch_frames

    @property
    def temporal_support_seconds(self) -> float:
        return self.temporal_support_frames / self.sample_rate_hz


@dataclass(frozen=True)
class ExecutionMode:
    """Implementation choices to benchmark.

    `precompute_static_tensors` controls whether constant tensors such as the
    Fresnel quadratic phase and the display ROI mask are built once or rebuilt
    on demand.

    `preallocate_work_buffers` controls whether reusable work buffers are kept
    alive across iterations or recreated as needed. Persistent algorithmic state
    required by the sliding window still exists in both modes; the switch only
    affects reusable temporary/output buffers.
    """

    name: str
    precompute_static_tensors: bool
    preallocate_work_buffers: bool

    enable_dummy_gil_thread: bool = False
    dummy_gil_inner_loops: int = 200_000
    dummy_gil_switch_interval_s: float | None = None


@dataclass(frozen=True)
class InputInfo:
    height: int
    width: int
    dtype: np.dtype


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

    # Output transfer schedule
    output_transfer_batch_size: int
    d2h_transfers: int
    d2h_transfers_per_second: float
    mean_outputs_per_transfer: float

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
# Host-side utilities
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


def read_input_info(path: str) -> InputInfo:
    with holofile.HoloReader(path) as reader:
        return InputInfo(
            height=reader.header.height,
            width=reader.header.width,
            dtype=np.dtype(reader.header.dtype),
        )


def validate_input(info: InputInfo, params: Params) -> None:
    expected_dtype = np.dtype(params.acquisition_dtype)
    if info.dtype != expected_dtype:
        raise ValueError(
            f"This benchmark expects {expected_dtype}, but the file stores {info.dtype}. "
            f"Either convert the file or update Params.acquisition_dtype."
        )

    if params.output_transfer_batch_size <= 0:
        raise ValueError(
            "output_transfer_batch_size must be positive, got "
            f"{params.output_transfer_batch_size}."
        )


def preload_batches(path: str, info: InputInfo, params: Params) -> np.ndarray:
    """Preload exactly M batches in pinned host memory.

    This isolates the compute benchmark from file-I/O noise while keeping the
    host-side data residency explicit.
    """
    host_batches = cupyx.empty_pinned(
        (
            params.sliding_window_batches,
            params.batch_frames,
            info.height,
            info.width,
        ),
        dtype=info.dtype,
    )

    flat_frames = host_batches.reshape(
        params.temporal_support_frames,
        info.height,
        info.width,
    )

    with holofile.HoloReader(path) as reader:
        reader.read_into(flat_frames, 0, params.temporal_support_frames)

    return host_batches


def cycle_batches(host_batches: np.ndarray) -> Iterator[np.ndarray]:
    """Yield preloaded batches forever in cyclic order."""
    num_batches = host_batches.shape[0]
    index = 0

    while True:
        yield host_batches[index]
        index = (index + 1) % num_batches


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


def centered_coordinates(
    length: int,
    pitch: float,
    dtype: type[np.generic],
) -> cp.ndarray:
    """Return centered coordinates:
    x[n] = (n - (N - 1) / 2) * pitch
    """
    center = dtype((length - 1) / 2.0)
    step = dtype(pitch)
    return (cp.arange(length, dtype=dtype) - center) * step


def make_quadratic_phase(height: int, width: int, params: Params) -> cp.ndarray:
    r"""Build the Fresnel input quadratic phase:
    Q(x, y) = exp(i π (x² + y²) / (λ z))
    """
    real = params.real_dtype

    x = centered_coordinates(width, params.dx_m, real)
    y = centered_coordinates(height, params.dy_m, real)

    radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
    phase = radius_sq * real(
        np.pi / (params.wavelength_m * params.propagation_distance_m)
    )

    return cp.exp(1j * phase).astype(params.complex_dtype, copy=False)


def make_normalized_elliptical_mask(
    height: int,
    width: int,
    radius: float,
) -> cp.ndarray:
    """Return the normalized elliptical display ROI mask.

    The support is defined by
        u² + v² <= r²
    in normalized image coordinates.
    """
    real = cp.float32

    x = centered_coordinates(width, 2.0 / width, real)
    y = centered_coordinates(height, 2.0 / height, real)

    radius_sq = cp.square(y)[:, None] + cp.square(x)[None, :]
    return radius_sq <= real(radius * radius)


# ============================================================================
# Device-side operators
# ============================================================================


class SlidingMean2D:
    r"""Causal sliding mean over the last M batch-images.

    If S_n(y, x) is the power image produced by batch n, we maintain

        R_n(y, x) = \sum_{m=n-M+1}^{n} S_m(y, x),

    and return

        \bar{S}_n(y, x) = R_n(y, x) / M

    once the window is full.

    The ring-buffer state is persistent in all modes because it is part of the
    algorithm, not a temporary scratch buffer. Only the output reuse policy
    changes with `reuse_mean_buffer`.
    """

    def __init__(
        self,
        window_length: int,
        height: int,
        width: int,
        dtype: type[np.generic],
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
        self._normalization = dtype(window_length)

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

    Percentile bounds are estimated only inside the ROI, then applied globally
    to the whole image.

    Static-mask and output-buffer reuse are controlled independently so the
    benchmark can isolate precomputation from preallocation.
    """

    def __init__(
        self,
        height: int,
        width: int,
        dtype: type[np.generic],
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


class OutputTransferBatcher:
    """Batch displayed device images before one host transfer.

    The pipeline still produces one displayed image per processed batch. This
    helper stages those device images into a `(N, H, W)` device buffer and
    flushes them to pinned host memory with a single D2H transfer.

    This reduces the frequency of `.get()` calls and stream synchronizations.
    It does not, by itself, introduce true compute/copy overlap on the GPU,
    because all work still runs on the same stream.
    """

    def __init__(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: type[np.generic],
        reuse_buffers: bool,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.dtype = dtype
        self.reuse_buffers = reuse_buffers

        self.device_batch = (
            cp.empty((batch_size, height, width), dtype=dtype)
            if reuse_buffers
            else None
        )
        self.host_batch = (
            cupyx.empty_pinned((batch_size, height, width), dtype=dtype)
            if reuse_buffers
            else None
        )

        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self.batch_size

    def _ensure_buffers(self) -> None:
        if self.device_batch is None:
            self.device_batch = cp.empty(
                (self.batch_size, self.height, self.width),
                dtype=self.dtype,
            )
        if self.host_batch is None:
            self.host_batch = cupyx.empty_pinned(
                (self.batch_size, self.height, self.width),
                dtype=self.dtype,
            )

    def push(self, image: cp.ndarray) -> bool:
        self._ensure_buffers()

        with time_range("stage output image", color_id=10):
            self.device_batch[self._count, ...] = image

        self._count += 1
        return self.is_full

    def flush(self) -> np.ndarray | None:
        if self._count == 0:
            return None

        self._ensure_buffers()

        device_view = self.device_batch[: self._count]
        host_view = self.host_batch[: self._count]

        with time_range("D2H output batch", color_id=11):
            device_view.get(out=host_view, blocking=False)

        with time_range("sync output batch", color_id=12):
            cp.cuda.get_current_stream().synchronize()

        batch = np.array(host_view, copy=True)
        self._count = 0

        if not self.reuse_buffers:
            self.device_batch = None
            self.host_batch = None

        return batch


class PowerDopplerPipeline:
    r"""Device-side pipeline from one temporal batch to one displayed image.

    For a temporal batch U_0[t, y, x], this pipeline computes

        1. temporal_spectrum[k, y, x] = rFFT_t(U_0)[k]
        2. keep only k in [k0, k1)
        3. propagated[k, y, x] = FFT_xy(temporal_spectrum[k, y, x] * Q[y, x])
        4. batch_power[y, x] = sum_k |propagated[k, y, x]|²

    Then, across successive batches, it maintains the causal sliding mean

        display_mean_n[y, x] = (1 / M) sum_{m=n-M+1}^{n} batch_power_m[y, x].

    Finally, for visualization only, it applies:
        - fftshift
        - percentile clipping in a normalized elliptical ROI

    The returned image remains on device. Host transfer is handled separately by
    `OutputTransferBatcher`.
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
        else:
            self.raw_batch_device = None
            self.real_batch_device = None

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
    def render_display_image(self) -> cp.ndarray:
        with time_range("average", color_id=7):
            averaged = self.sliding_mean.mean()

        with time_range("fftshift", color_id=8):
            shifted = cp.fft.fftshift(averaged)

        with time_range("percentile clip", color_id=9):
            return self.display_clipper.apply(shifted)

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

        with time_range("accumulate power", color_id=13):
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
        self.output_batcher = OutputTransferBatcher(
            batch_size=pipeline.params.output_transfer_batch_size,
            height=pipeline.height,
            width=pipeline.width,
            dtype=pipeline.params.real_dtype,
            reuse_buffers=pipeline.mode.preallocate_work_buffers,
        )

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

    def _produce_one_output(self) -> tuple[bool, np.ndarray | None]:
        """Process one batch and stage one displayed image.

        Returns:
            flushed: whether a host transfer was performed.
            host_batch: transferred host batch if flushed, else None.
        """
        ready = self.pipeline.process_batch(next(self.host_batch_iter))
        if not ready:
            raise RuntimeError("Sliding window should be ready when producing output.")

        device_image = self.pipeline.render_display_image()
        if self.output_batcher.push(device_image):
            host_batch = self.output_batcher.flush()
            if host_batch is None:
                raise RuntimeError("Output batcher unexpectedly returned no batch.")
            return True, host_batch

        return False, None

    def _flush_remaining_outputs(self) -> np.ndarray | None:
        return self.output_batcher.flush()

    def warmup(self) -> None:
        """Produce a small number of outputs before timed measurement."""
        print("Warming up...")
        for _ in range(self.pipeline.params.warmup_outputs):
            self._produce_one_output()

        self._flush_remaining_outputs()

    def run(self) -> tuple[np.ndarray, BenchmarkStats]:
        params = self.pipeline.params
        info = self.pipeline.info
        mode = self.pipeline.mode

        total_batches = 0
        total_frames = 0
        total_outputs = 0
        total_d2h_transfers = 0

        last_image: np.ndarray | None = None
        elapsed = 0.0

        dummy_thread: DummyGilThread | None = None
        previous_switch_interval: float | None = None

        print("Running steady-state benchmark...")

        try:
            dummy_thread, previous_switch_interval = start_dummy_gil_thread(mode)

            t0 = perf_counter()
            while elapsed < params.benchmark_seconds:
                flushed, host_batch = self._produce_one_output()

                total_batches += 1
                total_frames += params.batch_frames
                total_outputs += 1

                if flushed:
                    total_d2h_transfers += 1
                    last_image = host_batch[-1]
                    elapsed = perf_counter() - t0

            remaining_batch = self._flush_remaining_outputs()
            if remaining_batch is not None:
                total_d2h_transfers += 1
                last_image = remaining_batch[-1]
                elapsed = perf_counter() - t0

        finally:
            stop_dummy_gil_thread(dummy_thread, previous_switch_interval)

        if last_image is None:
            raise RuntimeError("Benchmark produced no output image.")

        input_frame_bytes = bytes_per_frame(
            height=info.height,
            width=info.width,
            dtype=np.dtype(params.acquisition_dtype),
        )
        cast_frame_bytes = bytes_per_frame(
            height=info.height,
            width=info.width,
            dtype=np.dtype(params.real_dtype),
        )
        output_image_bytes = bytes_per_frame(
            height=info.height,
            width=info.width,
            dtype=np.dtype(params.real_dtype),
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
            output_transfer_batch_size=params.output_transfer_batch_size,
            d2h_transfers=total_d2h_transfers,
            d2h_transfers_per_second=total_d2h_transfers / elapsed,
            mean_outputs_per_transfer=total_outputs / total_d2h_transfers,
            shape=(info.height, info.width),
            file_dtype=str(info.dtype),
            host_dtype=str(info.dtype),
            device_input_dtype=str(np.dtype(params.acquisition_dtype)),
            real_dtype=str(np.dtype(params.real_dtype)),
            complex_dtype=str(np.dtype(params.complex_dtype)),
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
    print(f"  output_transfer_batch_size={params.output_transfer_batch_size}")
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


# ============================================================================
# Reporting
# ============================================================================


def bytes_per_frame(height: int, width: int, dtype: np.dtype) -> int:
    return height * width * dtype.itemsize


def print_table(title: str, rows: list[list[object]]) -> None:
    print(f"\n{title}")
    print(tabulate(rows, tablefmt="github"))


def print_stats(stats: BenchmarkStats) -> None:
    print("\nSteady-state benchmark")
    print("----------------------")

    print_table(
        "Mode",
        [
            ["Name", stats.mode_name],
            ["Precompute static tensors", stats.precompute_static_tensors],
            ["Preallocate work buffers", stats.preallocate_work_buffers],
        ],
    )

    print_table(
        "Timing",
        [
            ["Time", f"{stats.seconds:.3f} s"],
            ["Frames", stats.frames],
            ["Batches", stats.batches],
            ["Outputs", stats.outputs],
            ["Wall time / output", f"{stats.wall_ms_per_output:.2f} ms"],
        ],
    )

    print_table(
        "Rates",
        [
            ["Input FPS", f"{stats.input_fps:.1f} frames/s"],
            ["Batches / s", f"{stats.batches_per_second:.2f}"],
            ["Outputs / s", f"{stats.outputs_per_second:.2f}"],
        ],
    )

    print_table(
        "Bandwidth",
        [
            ["H2D", f"{stats.h2d_gbps:.3f} GB/s"],
            ["Cast throughput", f"{stats.cast_effective_gbps:.3f} GB/s"],
            ["D2H output", f"{stats.d2h_output_mbps:.3f} MB/s"],
        ],
    )

    print_table(
        "Output transfer schedule",
        [
            ["Requested batch size", stats.output_transfer_batch_size],
            ["D2H transfers", stats.d2h_transfers],
            ["Transfers / s", f"{stats.d2h_transfers_per_second:.2f}"],
            ["Outputs / transfer", f"{stats.mean_outputs_per_transfer:.2f}"],
        ],
    )

    print_table(
        "Data types",
        [
            ["File", stats.file_dtype],
            ["Host", stats.host_dtype],
            ["Device input", stats.device_input_dtype],
            ["Real", stats.real_dtype],
            ["Complex", stats.complex_dtype],
        ],
    )

    print_table(
        "Pipeline",
        [
            [
                "Doppler bins",
                f"[{stats.doppler_bins[0]}:{stats.doppler_bins[1]}) "
                f"({stats.doppler_bin_count})",
            ],
            [
                "Sliding window",
                f"{stats.batches_per_output} batches = {stats.frames_per_output} frames",
            ],
            ["Output stride", f"1 batch = {stats.output_stride_frames} frames"],
            ["Temporal support", f"{stats.temporal_support_ms:.3f} ms"],
        ],
    )

    gil_rows = [
        [
            "Dummy GIL thread",
            "enabled" if stats.dummy_gil_thread_enabled else "disabled",
        ],
        ["Dummy inner loops", stats.dummy_gil_inner_loops],
        [
            "Switch interval",
            "default"
            if stats.dummy_gil_switch_interval_s is None
            else f"{stats.dummy_gil_switch_interval_s:.6f} s",
        ],
    ]

    if stats.dummy_gil_thread_enabled:
        gil_rows.extend(
            [
                ["Dummy iterations", stats.dummy_gil_iterations],
                ["Dummy rate", f"{stats.dummy_gil_iterations_per_second:.0f} iter/s"],
            ]
        )

    print_table("Host-side GIL stress", gil_rows)


def print_suite_summary(stats_list: Sequence[BenchmarkStats]) -> None:
    rows: list[list[object]] = []

    for stats in stats_list:
        rows.append(
            [
                stats.mode_name,
                "on" if stats.precompute_static_tensors else "off",
                "on" if stats.preallocate_work_buffers else "off",
                "on" if stats.dummy_gil_thread_enabled else "off",
                "default"
                if stats.dummy_gil_switch_interval_s is None
                else f"{stats.dummy_gil_switch_interval_s:.6f}",
                stats.output_transfer_batch_size,
                f"{stats.mean_outputs_per_transfer:.2f}",
                f"{stats.outputs_per_second:.2f}",
                f"{stats.wall_ms_per_output:.2f}",
                f"{stats.input_fps:.0f}",
                f"{stats.d2h_transfers_per_second:.2f}",
                f"{stats.h2d_gbps:.3f}",
                f"{stats.cast_effective_gbps:.3f}",
                f"{stats.d2h_output_mbps:.3f}",
                f"{stats.dummy_gil_iterations_per_second:.0f}",
            ]
        )

    print("\nSuite summary")
    print("-------------")
    print(
        tabulate(
            rows,
            headers=[
                "Mode",
                "Precompute",
                "Preallocate",
                "GIL",
                "Switch interval (s)",
                "Xfer batch",
                "Out/xfer",
                "Outputs/s",
                "ms/output",
                "Input FPS",
                "Xfers/s",
                "H2D GB/s",
                "Cast GB/s",
                "D2H MB/s",
                "Dummy iter/s",
            ],
            tablefmt="github",
        )
    )


def show_image(image: np.ndarray, stats: BenchmarkStats) -> None:
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(
        f"{stats.mode_name}\n"
        f"Power Doppler sliding average "
        f"({stats.frames_per_output} frames support, "
        f"{stats.outputs_per_second:.2f} outputs/s)"
    )
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# ============================================================================
# Mode builders
# ============================================================================


def make_factorial_modes(
    *,
    dummy_gil_inner_loops: int,
    dummy_gil_switch_interval_s: float | None,
) -> list[ExecutionMode]:
    """Build the full 2 × 2 × 2 benchmark matrix."""
    modes: list[ExecutionMode] = []

    for precompute in (False, True):
        for preallocate in (False, True):
            base_name = (
                f"precompute={'on' if precompute else 'off'} | "
                f"prealloc={'on' if preallocate else 'off'}"
            )

            modes.append(
                ExecutionMode(
                    name=f"{base_name} | gil=off",
                    precompute_static_tensors=precompute,
                    preallocate_work_buffers=preallocate,
                    enable_dummy_gil_thread=False,
                )
            )

            modes.append(
                ExecutionMode(
                    name=f"{base_name} | gil=on",
                    precompute_static_tensors=precompute,
                    preallocate_work_buffers=preallocate,
                    enable_dummy_gil_thread=True,
                    dummy_gil_inner_loops=dummy_gil_inner_loops,
                    dummy_gil_switch_interval_s=dummy_gil_switch_interval_s,
                )
            )

    return modes


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    params = Params(
        benchmark_seconds=10.0,
        output_transfer_batch_size=16,
        show_image=True,
    )

    print("Inspecting input...")
    info = read_input_info(params.path)
    validate_input(info, params)

    print("Preloading host data...")
    host_batches = preload_batches(params.path, info, params)
    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in pinned host memory."
    )

    # Choose either a single mode or a full suite.
    #
    # Example single-mode selection:
    modes = [
        ExecutionMode(
            name="precompute=on | prealloc=on | gil=off",
            precompute_static_tensors=True,
            preallocate_work_buffers=True,
            enable_dummy_gil_thread=False,
        )
    ]
    #
    # Full 2 × 2 × 2 benchmark matrix:
    # modes = make_factorial_modes(
    #     dummy_gil_inner_loops=200_000,
    #     dummy_gil_switch_interval_s=0.0005,
    # )

    results = benchmark_suite(
        host_batches=host_batches,
        info=info,
        params=params,
        modes=modes,
    )

    print_suite_summary([stats for _, stats in results])

    print("\nDetailed results")
    print("----------------")
    for _, stats in results:
        print_stats(stats)

    if params.show_image and results:
        image, stats = results[-1]
        show_image(image, stats)


if __name__ == "__main__":
    main()
