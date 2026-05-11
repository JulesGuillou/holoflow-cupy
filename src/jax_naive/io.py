from __future__ import annotations

import holofile
import numpy as np

from holoflow_benchmarks.config import Params
from holoflow_benchmarks.io import InputInfo


def preload_batches(path: str, info: InputInfo, params: Params) -> np.ndarray:
    """Preload exactly M batches into NumPy host memory for JAX transfer."""
    host_batches = np.empty(
        (
            params.sliding_window_batches,
            params.batch_frames,
            info.height,
            info.width,
        ),
        dtype=info.dtype,
    )

    flat_frames = host_batches.reshape(
        params.temporal_support_frames,
        info.height,
        info.width,
    )

    with holofile.HoloReader(path) as reader:
        reader.read_into(flat_frames, 0, params.temporal_support_frames)

    return host_batches


__all__ = ["preload_batches"]
