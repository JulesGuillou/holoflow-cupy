from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cupyx
import holofile
import numpy as np

from .config import Params


@dataclass(frozen=True)
class InputInfo:
    height: int
    width: int
    dtype: np.dtype


def read_input_info(path: str) -> InputInfo:
    with holofile.HoloReader(path) as reader:
        return InputInfo(
            height=int(reader.header.height),
            width=int(reader.header.width),
            dtype=np.dtype(reader.header.dtype),
        )


def validate_input(info: InputInfo, params: Params) -> None:
    if info.dtype != params.acquisition_dtype:
        raise ValueError(
            f"This benchmark expects {params.acquisition_dtype}, "
            f"but the file stores {info.dtype}. "
            f"Either convert the file or update dtypes.acquisition in config.yaml."
        )


def preload_batches(path: str, info: InputInfo, params: Params) -> np.ndarray:
    """Preload exactly M batches in pinned host memory.

    This isolates the compute benchmark from file-I/O noise while keeping the
    host-side data residency explicit.
    """
    host_batches = cupyx.empty_pinned(
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


def cycle_batches(host_batches: np.ndarray) -> Iterator[np.ndarray]:
    """Yield preloaded batches forever in cyclic order."""
    num_batches = host_batches.shape[0]
    index = 0

    while True:
        yield host_batches[index]
        index = (index + 1) % num_batches


def bytes_per_frame(height: int, width: int, dtype: np.dtype) -> int:
    return height * width * dtype.itemsize


def validate_host_batches(
    host_batches: np.ndarray,
    info: InputInfo,
    params: Params,
) -> None:
    expected_shape = (
        params.sliding_window_batches,
        params.batch_frames,
        info.height,
        info.width,
    )
    if host_batches.shape != expected_shape:
        raise ValueError(
            f"Unexpected preloaded host batch shape: got {host_batches.shape}, "
            f"expected {expected_shape}."
        )

