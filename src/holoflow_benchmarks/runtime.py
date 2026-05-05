from __future__ import annotations

import sys
import threading

import cupy as cp
from cupyx.profiler import time_range

from .config import ExecutionMode


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

    with time_range("start dummy GIL thread", color_id=40):
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
    with time_range("stop dummy GIL thread", color_id=41):
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
    with time_range("clear CuPy pools", color_id=30):
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

