from __future__ import annotations

from pathlib import Path

from holoflow_benchmarks.config import (
    ExecutionMode,
    Params,
    load_benchmark_config as _load_benchmark_config,
)


def load_benchmark_config(path: str | Path) -> tuple[Params, list[ExecutionMode]]:
    return _load_benchmark_config(path, implementation_name="cupy-naive")

__all__ = ["ExecutionMode", "Params", "load_benchmark_config"]
