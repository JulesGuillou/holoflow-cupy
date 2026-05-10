from __future__ import annotations

import numpy as np
import torch


def torch_dtype(dtype: np.dtype | type | str) -> torch.dtype:
    """Map a NumPy dtype accepted by the shared config to a PyTorch dtype."""
    normalized = np.dtype(dtype)

    mapping: dict[np.dtype, torch.dtype] = {
        np.dtype("uint8"): torch.uint8,
        np.dtype("int8"): torch.int8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("float16"): torch.float16,
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
    }
    if hasattr(torch, "uint16"):
        mapping[np.dtype("uint16")] = torch.uint16
    if hasattr(torch, "uint32"):
        mapping[np.dtype("uint32")] = torch.uint32
    if hasattr(torch, "uint64"):
        mapping[np.dtype("uint64")] = torch.uint64

    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for PyTorch benchmark: {dtype!r}") from exc

