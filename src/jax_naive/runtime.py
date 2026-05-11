from __future__ import annotations

import sys
import threading

import jax

from holoflow_benchmarks.config import ExecutionMode

from .nvtx import time_range


class DummyGilThread(threading.Thread):
    """Pure-Python background work used to create intentional GIL contention."""

    def __init__(self, inner_loops: int) -> None:
        super().__init__(name="jax-naive-dummy-gil-thread", daemon=True)
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

    with time_range("jax start dummy GIL thread", color_id=440):
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
    with time_range("jax stop dummy GIL thread", color_id=441):
        if thread is not None:
            thread.stop()
            thread.join()

        if previous_switch_interval is not None:
            sys.setswitchinterval(previous_switch_interval)


def clear_jax_runtime() -> None:
    """Run a JAX side-effect barrier between benchmark modes.

    JAX does not expose a CuPy-style memory-pool flush. Individual benchmark
    runners block on their live arrays; this function only gives profiler
    timelines a clear per-mode barrier.
    """
    with time_range("jax runtime barrier", color_id=430):
        effects_barrier = getattr(jax, "effects_barrier", None)
        if effects_barrier is not None:
            effects_barrier()


__all__ = [
    "DummyGilThread",
    "clear_jax_runtime",
    "start_dummy_gil_thread",
    "stop_dummy_gil_thread",
]
