from __future__ import annotations

import sys


def require_linux(benchmark_name: str) -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            f"{benchmark_name} is Linux-only. The rest of holoflow can still run "
            "on Windows because JAX CUDA dependencies are platform-gated."
        )


__all__ = ["require_linux"]
