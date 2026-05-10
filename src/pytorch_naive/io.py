from __future__ import annotations

import holofile
import numpy as np
import torch

from holoflow_benchmarks.config import Params
from holoflow_benchmarks.io import InputInfo

from .dtypes import torch_dtype


def preload_batches(path: str, info: InputInfo, params: Params) -> np.ndarray:
    """Preload exactly M batches into PyTorch-pinned host memory."""
    host_tensor = torch.empty(
        (
            params.sliding_window_batches,
            params.batch_frames,
            info.height,
            info.width,
        ),
        dtype=torch_dtype(info.dtype),
        pin_memory=True,
    )
    host_batches = host_tensor.numpy()

    flat_frames = host_batches.reshape(
        params.temporal_support_frames,
        info.height,
        info.width,
    )

    with holofile.HoloReader(path) as reader:
        reader.read_into(flat_frames, 0, params.temporal_support_frames)

    return host_batches
