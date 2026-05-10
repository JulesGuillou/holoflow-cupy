from __future__ import annotations

import threading

import torch

from .nvtx import time_range


class DummyGilThread(threading.Thread):
    """Pure-Python background work used to create intentional GIL contention."""

    def __init__(self, inner_loops: int) -> None:
        super().__init__(name="pytorch-threaded-dummy-gil-thread", daemon=True)
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


def clear_torch_pools() -> None:
    """Release cached PyTorch CUDA blocks between benchmark modes."""
    with time_range("clear PyTorch pools", color_id=230):
        if not torch.cuda.is_available():
            return

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
