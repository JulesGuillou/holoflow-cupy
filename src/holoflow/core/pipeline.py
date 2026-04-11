"""
pipeline.py — Simple staged pipeline for real-time Doppler holography.

Pattern
───────
  TensorQueue  : a bounded queue of pre-allocated pinned/device arrays.
                 Stages pull a buffer, fill it, push it forward.
                 The consumer returns it to the producer's free pool when done.

  StageThread  : a daemon thread with one input TensorQueue and one output
                 TensorQueue.  Override process() to do stage work.

  Pipeline stages:

    Reader  ──► H2D ──► Compute ──► D2H ──► UI pool
      (disk)   (stream)  (stream)  (stream)

Each stage blocks on its input queue and pushes to its output queue.
Backpressure is natural: a slow downstream stage fills its input queue,
causing the upstream put() to block.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from unittest import loader

import cupy as cp
import cupyx
import holofile
import numpy as np


QUEUE_DEPTH = 2  # slots per inter-stage queue
STAGE_TIMEOUT = 0.05  # seconds; allows clean shutdown without busy-spinning


# ──────────────────────────────────────────────────────────────────────────────
# TensorQueue
# ──────────────────────────────────────────────────────────────────────────────


def _alloc(shape, dtype, pinned: bool):
    if pinned:
        return cupyx.empty_pinned(shape, dtype=dtype)
    else:
        return cp.empty(shape, dtype=dtype)


class TensorQueue:
    """
    A bounded queue of pre-allocated arrays.

    Two internal queues per instance:
      free  — empty buffers ready to be filled by the producer.
      ready — filled buffers waiting to be consumed.

    The producer calls acquire() → fills the buffer → put(buf, meta).
    The consumer calls get() → uses (buf, meta) → release(buf).

    meta is an arbitrary Python object (typically an int n_frames) that
    travels with the buffer without being stored on the array itself.
    """

    def __init__(self, shape, dtype, n: int, pinned: bool = False) -> None:
        self._free = queue.Queue(maxsize=n)
        self._ready = queue.Queue(maxsize=n)
        for _ in range(n):
            self._free.put(_alloc(shape, dtype, pinned))

    def acquire(self, timeout: float = STAGE_TIMEOUT) -> np.ndarray | cp.ndarray | None:
        """Return a free buffer to fill.  Returns None on timeout."""
        with cupyx.profiler.time_range("TensorQueue.acquire", color_id=2):
            try:
                return self._free.get(timeout=timeout)
            except queue.Empty:
                return None

    def put(self, buf, meta=None) -> None:
        """Hand a filled buffer downstream.  Blocks on backpressure."""
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
    A pipeline stage running on a dedicated daemon thread.

    Subclasses implement process(), which should:
      1. call self.inp.get() to receive a filled buffer (or return on None).
      2. call self.out.acquire() to get an output buffer.
      3. do work.
      4. call self.inp.release(buf) to free the input buffer.
      5. call self.out.put(buf, meta) to forward the result.
    """

    def __init__(self, inp: TensorQueue | None, out: TensorQueue | None) -> None:
        self.inp = inp
        self.out = out
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
        """Called once on thread start. Override for CUDA stream creation etc."""

    def teardown(self) -> None:
        """Called once after the loop exits. Override for resource cleanup."""

    @abstractmethod
    def process(self) -> None: ...


# ──────────────────────────────────────────────────────────────────────────────
# Compute kernel  (replace body with real holography processing)
# ──────────────────────────────────────────────────────────────────────────────


def _compute(
    d_in: cp.ndarray, n: int, d_out: cp.ndarray, stream: cp.cuda.Stream
) -> None:
    """
    Temporal mean + normalisation across n raw frames.

    Replace with the real Doppler holography kernel (FFT, wavefront
    correction, …) without changing the signature.

    Parameters
    ----------
    d_in   : device array (batch_size, H, W), raw integer frames.
    n      : number of valid frames in d_in (≤ batch_size).
    d_out  : device array (H, W) float32, written in-place.
    stream : CUDA stream on which to enqueue all work.
    """
    with stream:
        cp.mean(d_in[:n], axis=0, dtype=cp.float32, out=d_out)
        d_out /= 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Concrete stages
# ──────────────────────────────────────────────────────────────────────────────


class ReaderStage(StageThread):
    """
    Stage 1 — fills pinned host buffers from disk (live) or RAM (cpu).

    Has no input queue; acquires free buffers from its own output queue.
    Forwards (buf, n_frames) downstream.
    """

    def __init__(self, out: TensorQueue, loader: DataLoader) -> None:
        super().__init__(inp=None, out=out)
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
            np.copyto(buf[:n], L.cpu_data[self._idx : self._idx + n])

        self._idx += n
        self.out.put(buf, meta=n)


class H2DStage(StageThread):
    """
    Stage 2 — DMA from pinned host buffer to device buffer.

    Receives (host_buf, n), copies host_buf[:n] → dev_buf[:n],
    forwards (dev_buf, n).
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
        self.out.put(dev_buf, meta=n)


class ComputeStage(StageThread):
    """
    Stage 3 — GPU compute on a raw device batch.

    Receives (dev_in, n), runs _compute, forwards (dev_out, n).
    """

    def setup(self) -> None:
        self._stream = cp.cuda.Stream(non_blocking=True)

    def process(self) -> None:
        in_buf, n = self.inp.get()
        if in_buf is None:
            return

        out_buf = self.out.acquire()
        if out_buf is None:
            self.inp.release(in_buf)
            return

        _compute(in_buf, n, out_buf, self._stream)
        self._stream.synchronize()

        self.inp.release(in_buf)
        self.out.put(out_buf)  # n not needed downstream


class D2HStage(StageThread):
    """
    Stage 4 — DMA from device frame to pinned host frame.

    Receives (dev_buf, _), copies to a pinned host buffer,
    forwards (host_buf, None) into the UI queue.
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

        # host_buf is a plain numpy.ndarray backed by pinned memory
        # (allocated via np.frombuffer(cp.cuda.alloc_pinned_memory(...))).
        # CuPy accepts it as a D2H destination without wrapping.
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
    Reads metadata from a .holo file and optionally preloads frames.

    load_kind
    ─────────
    "live" — Reader streams from disk at runtime.
    "cpu"  — frames preloaded to pinned host RAM; Reader does a fast memcpy.
    "gpu"  — frames preloaded to VRAM; Reader and H2D are not used.
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

        end_frame = end_frame if end_frame > 0 else self.total_file_frames
        self.end_frame = min(end_frame, self.total_file_frames)
        self.total_frames = self.end_frame - self.start_frame

        self.cpu_data = None
        self.gpu_data = None
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


class PreloadedVirtualQueue:
    """
    A 'virtual' queue for preloaded data that acts as a drop-in replacement
    for TensorQueue. Instead of waiting for a producer thread, it immediately
    yields continuous slices of preloaded RAM or VRAM.
    """

    def __init__(self, loader: DataLoader, use_gpu: bool = False) -> None:
        self._loader = loader
        self._use_gpu = use_gpu
        self._idx = loader.start_frame
        self._lock = threading.Lock()  # Ensures thread-safe pointer advancement

    def get(self, block: bool = True, timeout: float = STAGE_TIMEOUT) -> tuple:
        L = self._loader
        with self._lock:
            # Wrap around logic for continuous playback
            if self._idx >= L.end_frame:
                self._idx = L.start_frame

            stop = min(self._idx + L.batch_size, L.end_frame)
            n = stop - self._idx

            # Yield a VIEW of the preloaded data, no copying!
            if self._use_gpu:
                buf_view = L.gpu_data[self._idx : stop]
            else:
                buf_view = L.cpu_data[self._idx : stop]

            self._idx += n

            return buf_view, n

    def release(self, buf) -> None:
        # Since buf is just a slice of the persistent array,
        # there is nothing to return to a free pool.
        pass

    def acquire(self, timeout: float = STAGE_TIMEOUT):
        raise NotImplementedError("Virtual queues are read-only generators.")

    def put(self, buf, meta=None) -> None:
        raise NotImplementedError("Virtual queues are read-only generators.")


# ──────────────────────────────────────────────────────────────────────────────
# FramePipeline
# ──────────────────────────────────────────────────────────────────────────────


class FramePipeline:
    """
    Wires four StageThreads together with TensorQueues.

    Queue layout
    ────────────
    q_host_in   pinned (batch_size, H, W)  Reader  → H2D
    q_device_in device (batch_size, H, W)  H2D     → Compute
    q_device_out device (H, W) float32     Compute → D2H
    q_ui        pinned (H, W) float32      D2H     → UI
    """

    def __init__(self, loader: DataLoader, queue_depth: int = QUEUE_DEPTH) -> None:
        L = loader
        batch_shape = (L.batch_size, L.height, L.width)
        frame_shape = (L.height, L.width)

        # 1. Output queues (Always required)
        q_device_out = TensorQueue(frame_shape, np.float32, n=queue_depth, pinned=False)
        self._q_ui = TensorQueue(
            frame_shape, np.float32, n=queue_depth + 2, pinned=True
        )

        self._stages = []

        # 2. Build the front-end of the pipeline based on load_kind
        if L.load_kind == "live":
            # Disk -> RAM queue -> VRAM queue
            q_host_in = TensorQueue(batch_shape, L.dtype, n=queue_depth, pinned=True)
            q_device_in = TensorQueue(batch_shape, L.dtype, n=queue_depth, pinned=False)

            self._stages.append(ReaderStage(out=q_host_in, loader=loader))
            self._stages.append(H2DStage(inp=q_host_in, out=q_device_in))
            self._stages.append(ComputeStage(inp=q_device_in, out=q_device_out))

        elif L.load_kind == "cpu":
            # RAM Slice -> VRAM queue (Bypasses ReaderStage entirely)
            q_host_virtual = PreloadedVirtualQueue(loader, use_gpu=False)
            q_device_in = TensorQueue(batch_shape, L.dtype, n=queue_depth, pinned=False)

            self._stages.append(H2DStage(inp=q_host_virtual, out=q_device_in))
            self._stages.append(ComputeStage(inp=q_device_in, out=q_device_out))

        elif L.load_kind == "gpu":
            # VRAM Slice -> Compute (Bypasses ReaderStage AND H2DStage entirely)
            q_device_virtual = PreloadedVirtualQueue(loader, use_gpu=True)

            self._stages.append(ComputeStage(inp=q_device_virtual, out=q_device_out))

        else:
            raise ValueError(f"Unknown load_kind: {L.load_kind}")

        # 3. Add the final DMA stage (Always required)
        self._stages.append(D2HStage(inp=q_device_out, out=self._q_ui))

    def start(self) -> None:
        for s in self._stages:
            s.start()

    def stop(self) -> None:
        for s in reversed(self._stages):
            s.stop()

    def pull_latest_frame(self) -> np.ndarray | None:
        """
        Drain the UI queue and return only the most recent frame.
        Stale frames are released back to the free pool immediately.
        """
        latest = None
        while True:
            buf, _ = self._q_ui.get(timeout=0)  # non-blocking drain
            if buf is None:
                break
            if latest is not None:
                self._q_ui.release(latest)
            latest = buf
        return latest

    def return_frame(self, frame: np.ndarray) -> None:
        """Return a UI frame buffer to the pool after rendering."""
        self._q_ui.release(frame)
