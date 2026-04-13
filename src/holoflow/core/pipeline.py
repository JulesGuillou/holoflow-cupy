"""
pipeline.py — Staged pipeline for real-time Doppler holography.

Architecture
────────────
Frames travel through a chain of StageThreads connected by TensorQueues:

  Reader ──► H2D ──► Compute ──► D2H ──► UI pool
  (disk)  (pinned→GPU) (GPU)  (GPU→pinned)

Each stage runs on a dedicated daemon thread and blocks on its input queue.
Backpressure is automatic: a slow downstream stage fills its input queue,
causing upstream put() calls to block.

TensorQueue holds a fixed pool of pre-allocated arrays so that no allocations
occur in the hot path.  Metadata (e.g. the number of valid frames in a batch)
travels alongside each buffer as a plain Python object.

Load strategies
───────────────
  live — ReaderStage streams frames from disk at runtime.
  cpu  — Frames are preloaded to pinned host RAM; ReaderStage is replaced by a
         PreloadedVirtualQueue that yields zero-copy views.
  gpu  — Frames are preloaded to VRAM; both ReaderStage and H2DStage are
         bypassed.
"""

from __future__ import annotations

import queue
import threading
import time
from abc import ABC, abstractmethod

import cupy as cp
import cupyx
import holofile
import numpy as np


QUEUE_DEPTH = 2  # default slots per inter-stage queue
STAGE_TIMEOUT = 0.05  # seconds; lets threads wake up and check is_running


# ──────────────────────────────────────────────────────────────────────────────
# FrameCounter
# ──────────────────────────────────────────────────────────────────────────────


class FrameCounter:
    """
    Thread-safe accumulator for measuring raw frame throughput.

    Call add(n) from the producer thread each time n frames are consumed.
    Call pop_fps() from the UI thread to read and reset the rate.
    """

    def __init__(self) -> None:
        self._count = 0
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self._count += n

    def pop_fps(self) -> float | None:
        """Return frames/second since the last call, or None if < 1 s elapsed."""
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._t0
            if elapsed < 1.0:
                return None
            fps = self._count / elapsed
            self._count = 0
            self._t0 = now
        return fps


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _alloc(shape, dtype, pinned: bool) -> np.ndarray | cp.ndarray:
    """Allocate a single array, optionally in pinned host memory."""
    if pinned:
        return cupyx.empty_pinned(shape, dtype=dtype)
    return cp.empty(shape, dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────────
# TensorQueue
# ──────────────────────────────────────────────────────────────────────────────


class TensorQueue:
    """
    Bounded queue of pre-allocated arrays with a matching free pool.

    Usage pattern
    ─────────────
    Producer:  buf = q.acquire() → fill buf → q.put(buf, meta)
    Consumer:  buf, meta = q.get() → use buf → q.release(buf)

    The ``meta`` argument is an arbitrary Python object (typically an int
    holding the number of valid frames in a batch) that travels with the
    buffer without being stored on the array itself.
    """

    def __init__(self, shape, dtype, n: int, pinned: bool = False) -> None:
        self._free = queue.Queue(maxsize=n)
        self._ready = queue.Queue(maxsize=n)
        for _ in range(n):
            self._free.put(_alloc(shape, dtype, pinned))

    def acquire(self, timeout: float = STAGE_TIMEOUT) -> np.ndarray | cp.ndarray | None:
        """Return a free buffer to fill, or None on timeout."""
        with cupyx.profiler.time_range("TensorQueue.acquire", color_id=2):
            try:
                return self._free.get(timeout=timeout)
            except queue.Empty:
                return None

    def put(self, buf, meta=None) -> None:
        """Hand a filled buffer downstream.  Blocks under backpressure."""
        with cupyx.profiler.time_range("TensorQueue.put", color_id=1):
            self._ready.put((buf, meta))

    def get(self, block: bool = True, timeout: float = STAGE_TIMEOUT) -> tuple:
        """Return (buf, meta) from the ready queue, or (None, None) on timeout."""
        with cupyx.profiler.time_range("TensorQueue.get", color_id=0):
            try:
                return self._ready.get(block=block, timeout=timeout)
            except queue.Empty:
                return None, None

    def release(self, buf) -> None:
        """Return a consumed buffer to the free pool."""
        with cupyx.profiler.time_range("TensorQueue.release", color_id=3):
            self._free.put(buf)


# ──────────────────────────────────────────────────────────────────────────────
# StageThread
# ──────────────────────────────────────────────────────────────────────────────


class StageThread(ABC):
    """
    Abstract base for a single pipeline stage running on a daemon thread.

    Subclasses implement process(), which is called in a tight loop.  The
    typical body:
      1. inp.get()       — block for an input buffer (return early on timeout).
      2. out.acquire()   — get an output buffer (release inp on timeout).
      3. do work.
      4. inp.release()   — return the input buffer to its free pool.
      5. out.put()       — forward the result downstream.

    setup() and teardown() run once on the stage thread before and after the
    loop, respectively.  Use them for CUDA stream creation and resource cleanup.
    """

    def __init__(
        self,
        inp: TensorQueue | None,
        out: TensorQueue | None,
        frame_counter: FrameCounter | None = None,
    ) -> None:
        self.inp = inp
        self.out = out
        self._frame_counter = frame_counter
        self.is_running = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=type(self).__name__
        )

    def start(self) -> None:
        self.is_running = True
        self._thread.start()

    def stop(self) -> None:
        self.is_running = False
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        self.setup()
        while self.is_running:
            with cupyx.profiler.time_range(
                f"{type(self).__name__}: process", color_id=4
            ):
                self.process()
        self.teardown()

    def setup(self) -> None:
        """Called once on the stage thread before the processing loop."""

    def teardown(self) -> None:
        """Called once on the stage thread after the processing loop exits."""

    @abstractmethod
    def process(self) -> None: ...


# ──────────────────────────────────────────────────────────────────────────────
# Compute kernel
# ──────────────────────────────────────────────────────────────────────────────


def _compute(
    d_in: cp.ndarray,
    n: int,
    d_out: cp.ndarray,
    stream: cp.cuda.Stream,
) -> None:
    """
    Process one batch of raw frames into a single output frame.

    This is a placeholder (temporal mean + normalisation).  Replace the body
    with the real Doppler holography kernel (FFT, wavefront correction, …)
    without changing the signature.

    Parameters
    ----------
    d_in:
        Device array of shape (batch_size, H, W) holding raw integer frames.
    n:
        Number of valid frames in d_in (≤ batch_size).
    d_out:
        Device array of shape (H, W), dtype float32, written in-place.
    stream:
        CUDA stream on which all work is enqueued.
    """
    with stream:
        fs = 37000
        f0 = 8000
        f1 = 16000
        f0_idx = int(f0 / fs * n)
        f1_idx = int(f1 / fs * n)

        with cupyx.profiler.time_range("to_f32", color_id=5):
            d_in_f32 = d_in.astype(cp.float32)

        with cupyx.profiler.time_range("FFT", color_id=6):
            FH = cp.fft.rfft(d_in_f32[:n], axis=0)

        with cupyx.profiler.time_range("band_selection", color_id=7):
            band = FH[f0_idx:f1_idx]

        with cupyx.profiler.time_range("mean", color_id=8):
            d_out[:] = cp.mean(cp.abs(band) ** 2, axis=0, dtype=cp.float32)

        with cupyx.profiler.time_range("normalization", color_id=9):
            min_val = cp.min(d_out)
            max_val = cp.max(d_out)
            d_out[:] = (d_out - min_val) / (max_val - min_val + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Concrete stages
# ──────────────────────────────────────────────────────────────────────────────


class ReaderStage(StageThread):
    """
    Stage 1 — fill pinned host buffers from disk (live) or RAM (cpu).

    Has no input queue; it acquires free buffers from the output queue's free
    pool and fills them with raw frames.  Forwards (buf, n_frames) downstream.
    Wraps around to start_frame when end_frame is reached.
    """

    def __init__(
        self,
        out: TensorQueue,
        loader: DataLoader,
        frame_counter: FrameCounter | None = None,
    ) -> None:
        super().__init__(inp=None, out=out, frame_counter=frame_counter)
        self._loader = loader
        self._idx = loader.start_frame
        self._reader = None

    def setup(self) -> None:
        if self._loader.load_kind == "live":
            self._reader = holofile.HoloReader(self._loader.file_path)
            self._reader.__enter__()

    def teardown(self) -> None:
        if self._reader is not None:
            self._reader.__exit__(None, None, None)

    def process(self) -> None:
        L = self._loader
        buf = self.out.acquire()
        if buf is None:
            return

        if self._idx >= L.end_frame:
            self._idx = L.start_frame

        stop = min(self._idx + L.batch_size, L.end_frame)
        n = stop - self._idx

        if L.load_kind == "live":
            self._reader.read_into(buf[:n], start=self._idx, stop=stop)
        else:
            np.copyto(buf[:n], L.cpu_data[self._idx : stop])

        self._idx += n
        if self._frame_counter is not None:
            self._frame_counter.add(n)
        self.out.put(buf, meta=n)


class H2DStage(StageThread):
    """
    Stage 2 — DMA from a pinned host buffer to a device buffer.

    Receives (host_buf, n), copies host_buf[:n] to dev_buf[:n] via a
    non-blocking CUDA stream, then forwards (dev_buf, n).
    """

    def setup(self) -> None:
        self._stream = cp.cuda.Stream(non_blocking=True)

    def process(self) -> None:
        host_buf, n = self.inp.get()
        if host_buf is None:
            return

        dev_buf = self.out.acquire()
        if dev_buf is None:
            self.inp.release(host_buf)
            return

        with self._stream:
            dev_buf[:n].set(host_buf[:n])
        self._stream.synchronize()

        self.inp.release(host_buf)
        if self._frame_counter is not None:
            self._frame_counter.add(n)
        self.out.put(dev_buf, meta=n)


class ComputeStage(StageThread):
    """
    Stage 3 — GPU compute on a raw device batch.

    Receives (dev_in, n), processes the Doppler holography kernel,
    and forwards (dev_out, None).
    """

    def setup(self) -> None:
        self._stream = cp.cuda.Stream(non_blocking=True)

        # Initialize pre-allocation pointers to None.
        # Allocated on the first pass to perfectly match dynamic dimensions.
        self._d_in_f32 = None
        self._band_power = None

        # Physical constants
        self.fs = 37000
        self.f0 = 8000
        self.f1 = 16000

    def process(self) -> None:
        in_buf, n = self.inp.get()
        if in_buf is None:
            return

        out_buf = self.out.acquire()
        if out_buf is None:
            self.inp.release(in_buf)
            return

        with self._stream:
            # 1. Lazy Pre-allocation (Only runs on the very first batch)
            if self._d_in_f32 is None:
                self._d_in_f32 = cp.empty_like(in_buf, dtype=cp.float32)
                # Allocate a buffer to hold the magnitude-squared calculations
                # to avoid creating implicit arrays during standard CuPy math.
                self._band_power = cp.empty_like(in_buf, dtype=cp.float32)

            # 2. Type casting
            with cupyx.profiler.time_range("to_f32", color_id=5):
                self._d_in_f32[:n] = in_buf[:n]

            # 3. FFT
            with cupyx.profiler.time_range("FFT", color_id=6):
                FH = cp.fft.rfft(self._d_in_f32[:n], axis=0)

            # 4. Band Selection
            with cupyx.profiler.time_range("band_selection", color_id=7):
                f0_idx = int(self.f0 / self.fs * n)
                f1_idx = int(self.f1 / self.fs * n)
                band = FH[f0_idx:f1_idx]
                num_band_frames = f1_idx - f0_idx

            # 5. Standard Math using pre-allocated buffers
            with cupyx.profiler.time_range("mean_and_norm", color_id=8):
                if num_band_frames > 0:
                    # Step 1: Absolute value into our pre-allocated power buffer
                    cp.abs(band, out=self._band_power[:num_band_frames])

                    # Step 2: Square it in-place
                    cp.square(
                        self._band_power[:num_band_frames],
                        out=self._band_power[:num_band_frames],
                    )

                    # Step 3: Mean, written directly into out_buf
                    cp.mean(self._band_power[:num_band_frames], axis=0, out=out_buf)
                else:
                    out_buf[:] = 0.0

                # 6. In-place Normalization
                min_val = cp.min(out_buf)
                max_val = cp.max(out_buf)

                cp.subtract(out_buf, min_val, out=out_buf)
                cp.divide(out_buf, (max_val - min_val + 1e-8), out=out_buf)

        self._stream.synchronize()

        self.inp.release(in_buf)
        if self._frame_counter is not None:
            self._frame_counter.add(n)
        self.out.put(out_buf)


# class ComputeStage(StageThread):
#     """
#     Stage 3 — GPU compute on a raw device batch.

#     Receives (dev_in, n), calls _compute(), forwards (dev_out, None).
#     The frame count is not needed by downstream stages.
#     """

#     def setup(self) -> None:
#         self._stream = cp.cuda.Stream(non_blocking=True)

#     def process(self) -> None:
#         in_buf, n = self.inp.get()
#         if in_buf is None:
#             return

#         out_buf = self.out.acquire()
#         if out_buf is None:
#             self.inp.release(in_buf)
#             return

#         _compute(in_buf, n, out_buf, self._stream)
#         self._stream.synchronize()

#         self.inp.release(in_buf)
#         if self._frame_counter is not None:
#             self._frame_counter.add(n)
#         self.out.put(out_buf)


class D2HStage(StageThread):
    """
    Stage 4 — DMA from a device frame to a pinned host frame.

    Receives (dev_buf, _), copies it to a pinned host buffer via a
    non-blocking CUDA stream, then forwards (host_buf, None) to the UI queue.
    """

    def setup(self) -> None:
        self._stream = cp.cuda.Stream(non_blocking=True)

    def process(self) -> None:
        dev_buf, _ = self.inp.get()
        if dev_buf is None:
            return

        host_buf = self.out.acquire()
        if host_buf is None:
            self.inp.release(dev_buf)
            return

        with self._stream:
            dev_buf.get(out=host_buf)
        self._stream.synchronize()

        self.inp.release(dev_buf)
        self.out.put(host_buf)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────────────────────


class DataLoader:
    """
    Reads metadata from a .holo file and optionally preloads frames into RAM
    or VRAM.

    load_kind
    ─────────
    "live" — No preloading.  ReaderStage streams frames from disk at runtime.
    "cpu"  — All frames preloaded to pinned host RAM.  ReaderStage is replaced
             by a PreloadedVirtualQueue; H2DStage DMA-transfers views to GPU.
    "gpu"  — All frames preloaded to VRAM.  ReaderStage and H2DStage are both
             bypassed; ComputeStage receives GPU views directly.
    """

    def __init__(
        self,
        file_path: str,
        start_frame: int = 0,
        end_frame: int = -1,
        batch_size: int = 128,
        load_kind: str = "live",
    ) -> None:
        self.file_path = file_path
        self.start_frame = start_frame
        self.batch_size = batch_size
        self.load_kind = load_kind.lower()

        with holofile.HoloReader(file_path) as r:
            self.dtype = r.header.dtype
            self.height = r.header.height
            self.width = r.header.width
            self.total_file_frames = r.header.num_frames

        self.end_frame = (
            min(end_frame, self.total_file_frames)
            if end_frame > 0
            else self.total_file_frames
        )
        self.total_frames = self.end_frame - self.start_frame

        self.cpu_data: np.ndarray | None = None
        self.gpu_data: cp.ndarray | None = None

        if self.load_kind in ("cpu", "gpu"):
            self._preload()

    def _preload(self) -> None:
        shape = (self.total_frames, self.height, self.width)
        print(f"[DataLoader] Preloading {self.total_frames} frames…")
        self.cpu_data = cupyx.empty_pinned(shape, dtype=self.dtype)
        with holofile.HoloReader(self.file_path) as r:
            r.read_into(self.cpu_data, start=self.start_frame, stop=self.end_frame)
        if self.load_kind == "gpu":
            self.gpu_data = cp.array(self.cpu_data)
            self.cpu_data = None


# ──────────────────────────────────────────────────────────────────────────────
# PreloadedVirtualQueue
# ──────────────────────────────────────────────────────────────────────────────


class PreloadedVirtualQueue:
    """
    Drop-in replacement for TensorQueue when data is already in RAM or VRAM.

    Instead of waiting for a producer thread, get() immediately returns a
    zero-copy view into the preloaded array and advances an internal pointer.
    Wraps around continuously for uninterrupted playback.

    Only get() and release() are implemented; acquire() and put() raise
    NotImplementedError because this queue is read-only from the pipeline's
    perspective.
    """

    def __init__(self, loader: DataLoader, use_gpu: bool = False) -> None:
        self._loader = loader
        self._use_gpu = use_gpu
        self._idx = loader.start_frame
        self._lock = threading.Lock()

    def get(self, block: bool = True, timeout: float = STAGE_TIMEOUT) -> tuple:
        """Return (view, n) of the next batch slice.  Never blocks."""
        L = self._loader
        with self._lock:
            if self._idx >= L.end_frame:
                self._idx = L.start_frame

            stop = min(self._idx + L.batch_size, L.end_frame)
            n = stop - self._idx
            data = L.gpu_data if self._use_gpu else L.cpu_data
            view = data[self._idx : stop]
            self._idx += n

        return view, n

    def release(self, buf) -> None:
        """No-op: views are slices of the persistent preloaded array."""

    def acquire(self, timeout: float = STAGE_TIMEOUT):
        raise NotImplementedError("PreloadedVirtualQueue is read-only.")

    def put(self, buf, meta=None) -> None:
        raise NotImplementedError("PreloadedVirtualQueue is read-only.")


# ──────────────────────────────────────────────────────────────────────────────
# FramePipeline
# ──────────────────────────────────────────────────────────────────────────────


class FramePipeline:
    """
    Wires the appropriate stages together based on the DataLoader's load_kind.

    Queue layout (live mode)
    ────────────────────────
    q_host_in    pinned (batch_size, H, W)  Reader  → H2D
    q_device_in  device (batch_size, H, W)  H2D     → Compute
    q_device_out device (H, W) float32      Compute → D2H
    q_ui         pinned (H, W) float32      D2H     → UI

    In cpu/gpu modes the front of the pipeline is replaced by a
    PreloadedVirtualQueue, skipping the corresponding stages.
    """

    def __init__(self, loader: DataLoader, queue_depth: int = QUEUE_DEPTH) -> None:
        L = loader
        batch_shape = (L.batch_size, L.height, L.width)
        frame_shape = (L.height, L.width)

        q_device_out = TensorQueue(frame_shape, np.float32, n=queue_depth, pinned=False)
        self._q_ui = TensorQueue(
            frame_shape, np.float32, n=queue_depth + 2, pinned=True
        )

        self._stages: list[StageThread] = []
        self._input_counter = FrameCounter()

        if L.load_kind == "live":
            q_host_in = TensorQueue(batch_shape, L.dtype, n=queue_depth, pinned=True)
            q_device_in = TensorQueue(batch_shape, L.dtype, n=queue_depth, pinned=False)
            self._stages.append(
                ReaderStage(
                    out=q_host_in, loader=loader, frame_counter=self._input_counter
                )
            )
            self._stages.append(H2DStage(inp=q_host_in, out=q_device_in))
            self._stages.append(ComputeStage(inp=q_device_in, out=q_device_out))

        elif L.load_kind == "cpu":
            q_virtual = PreloadedVirtualQueue(loader, use_gpu=False)
            q_device_in = TensorQueue(batch_shape, L.dtype, n=queue_depth, pinned=False)
            self._stages.append(
                H2DStage(
                    inp=q_virtual, out=q_device_in, frame_counter=self._input_counter
                )
            )
            self._stages.append(ComputeStage(inp=q_device_in, out=q_device_out))

        elif L.load_kind == "gpu":
            q_virtual = PreloadedVirtualQueue(loader, use_gpu=True)
            self._stages.append(
                ComputeStage(
                    inp=q_virtual, out=q_device_out, frame_counter=self._input_counter
                )
            )

        else:
            raise ValueError(f"Unknown load_kind: {L.load_kind!r}")

        self._stages.append(D2HStage(inp=q_device_out, out=self._q_ui))

    def start(self) -> None:
        for stage in self._stages:
            stage.start()

    def stop(self) -> None:
        for stage in reversed(self._stages):
            stage.stop()

    def pull_latest_frame(self) -> np.ndarray | None:
        """
        Non-blocking drain of the UI queue.

        Returns only the most recent frame; any older frames are released
        back to the free pool immediately.  Returns None if no frame is ready.
        All drained frames (including dropped ones) are counted for input FPS.
        """
        latest = None
        while True:
            buf, _ = self._q_ui.get(timeout=0)
            if buf is None:
                break
            if latest is not None:
                self._q_ui.release(latest)
            latest = buf
        return latest

    def pop_input_fps(self) -> float | None:
        """
        Return raw frames/second at the pipeline source since the last call,
        or None if less than one second has elapsed.  Resets the counter.
        """
        return self._input_counter.pop_fps()

    def return_frame(self, frame: np.ndarray) -> None:
        """Return a UI frame buffer to the pool after the caller is done with it."""
        self._q_ui.release(frame)
