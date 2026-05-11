from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import jax


P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def time_range(message: str, color_id: int | None = None) -> Iterator[None]:
    """JAX profiler range helper with a CuPy-like call shape."""
    _ = color_id

    annotation = getattr(jax.profiler, "TraceAnnotation", None)
    if annotation is None:
        yield
        return

    with annotation(message):
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


__all__ = ["nvtx_range", "time_range"]
