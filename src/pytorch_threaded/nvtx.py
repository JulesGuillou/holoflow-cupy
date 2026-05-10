from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import torch


P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def time_range(message: str, color_id: int | None = None) -> Iterator[None]:
    """PyTorch NVTX range helper with a CuPy-like call shape."""
    _ = color_id

    if torch.cuda.is_available():
        with torch.cuda.nvtx.range(message):
            yield
        return

    yield


def nvtx_range(
    message: str,
    color_id: int | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with time_range(message, color_id=color_id):
                return function(*args, **kwargs)

        return wrapper

    return decorate
