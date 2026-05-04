from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ============================================================================
# Public configuration objects
# ============================================================================


@dataclass(frozen=True)
class Params:
    """Signal, geometry, display, dtype, and benchmark settings.

    This object describes what is computed and how long it is measured.
    Execution-style choices such as preallocation, precomputation, and host-side
    GIL stress live in `ExecutionMode`.
    """

    # Input
    file_path: str

    # Execution schedule
    batch_frames: int
    batches_per_output: int

    # Signal / physics
    sample_rate_hz: float
    doppler_low_hz: float
    doppler_high_hz: float

    propagation_distance_m: float
    wavelength_m: float
    dx_m: float
    dy_m: float

    # Benchmark control
    benchmark_seconds: float
    warmup_outputs: int

    # Display / contrast adjustment
    show_image: bool
    contrast_roi_radius: float
    contrast_low_percentile: float
    contrast_high_percentile: float
    report_path: str

    # Dtype contract
    acquisition_dtype: np.dtype
    real_dtype: np.dtype
    complex_dtype: np.dtype

    @property
    def path(self) -> str:
        return self.file_path

    @property
    def sliding_window_batches(self) -> int:
        return self.batches_per_output

    @property
    def temporal_support_frames(self) -> int:
        return self.batch_frames * self.sliding_window_batches

    @property
    def output_stride_frames(self) -> int:
        return self.batch_frames

    @property
    def temporal_support_seconds(self) -> float:
        return self.temporal_support_frames / self.sample_rate_hz


@dataclass(frozen=True)
class ExecutionMode:
    """Implementation choices to benchmark.

    `precompute_static_tensors` controls whether constant tensors such as the
    Fresnel quadratic phase and display ROI mask are built once or rebuilt on
    demand.

    `preallocate_work_buffers` controls whether reusable work buffers are kept
    alive across iterations or recreated as needed. Persistent algorithmic state
    required by the sliding window still exists in both modes; this switch only
    affects reusable temporary/output buffers.
    """

    name: str
    precompute_static_tensors: bool
    preallocate_work_buffers: bool

    enable_dummy_gil_thread: bool = False
    dummy_gil_inner_loops: int = 200_000
    dummy_gil_switch_interval_s: float | None = None


# ============================================================================
# YAML loading
# ============================================================================


def load_benchmark_config(path: str | Path) -> tuple[Params, list[ExecutionMode]]:
    """Load benchmark parameters and execution modes from a YAML file."""
    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if not isinstance(raw, Mapping):
        raise TypeError(f"{config_path} must contain a YAML mapping at the top level.")

    params = _load_params(raw)
    modes = _load_execution_modes(raw)

    _validate_params(params)
    if not modes:
        raise ValueError("The configuration produced no execution modes.")

    return params, modes


def _load_params(raw: Mapping[str, Any]) -> Params:
    input_cfg = _section(raw, "input")
    signal_cfg = _section(raw, "signal")
    schedule_cfg = _section(raw, "schedule")
    benchmark_cfg = _section(raw, "benchmark")
    display_cfg = _section(raw, "display")
    dtype_cfg = _section(raw, "dtypes")

    return Params(
        file_path=_input_file_path(input_cfg),
        batch_frames=_as_int(_required(schedule_cfg, "batch_frames"), "batch_frames"),
        batches_per_output=_as_int(
            _required(schedule_cfg, "batches_per_output"),
            "batches_per_output",
        ),
        sample_rate_hz=_as_float(
            _required(signal_cfg, "sample_rate_hz"),
            "sample_rate_hz",
        ),
        doppler_low_hz=_as_float(
            _required(signal_cfg, "doppler_low_hz"),
            "doppler_low_hz",
        ),
        doppler_high_hz=_as_float(
            _required(signal_cfg, "doppler_high_hz"),
            "doppler_high_hz",
        ),
        propagation_distance_m=_as_float(
            _required(signal_cfg, "propagation_distance_m"),
            "propagation_distance_m",
        ),
        wavelength_m=_as_float(_required(signal_cfg, "wavelength_m"), "wavelength_m"),
        dx_m=_as_float(_required(signal_cfg, "dx_m"), "dx_m"),
        dy_m=_as_float(_required(signal_cfg, "dy_m"), "dy_m"),
        benchmark_seconds=_as_float(
            _required(benchmark_cfg, "seconds"),
            "benchmark.seconds",
        ),
        warmup_outputs=_as_int(
            _required(benchmark_cfg, "warmup_outputs"),
            "benchmark.warmup_outputs",
        ),
        show_image=_as_bool(_required(display_cfg, "show_image"), "display.show_image"),
        contrast_roi_radius=_as_float(
            _required(display_cfg, "contrast_roi_radius"),
            "display.contrast_roi_radius",
        ),
        contrast_low_percentile=_as_float(
            _required(display_cfg, "contrast_low_percentile"),
            "display.contrast_low_percentile",
        ),
        contrast_high_percentile=_as_float(
            _required(display_cfg, "contrast_high_percentile"),
            "display.contrast_high_percentile",
        ),
        report_path=str(display_cfg.get("report_path", "cupy_naive_report.txt")),
        acquisition_dtype=np.dtype(_required(dtype_cfg, "acquisition")),
        real_dtype=np.dtype(_required(dtype_cfg, "real")),
        complex_dtype=np.dtype(_required(dtype_cfg, "complex")),
    )


def _load_execution_modes(raw: Mapping[str, Any]) -> list[ExecutionMode]:
    execution_cfg = _section(raw, "execution")

    explicit_modes = execution_cfg.get("modes")
    if explicit_modes is not None:
        return _load_explicit_modes(explicit_modes, execution_cfg)

    matrix_cfg = _section(execution_cfg, "mode_matrix")

    precompute_values = _as_bool_list(
        _required(matrix_cfg, "precompute_static_tensors"),
        "execution.mode_matrix.precompute_static_tensors",
    )
    preallocate_values = _as_bool_list(
        _required(matrix_cfg, "preallocate_work_buffers"),
        "execution.mode_matrix.preallocate_work_buffers",
    )
    gil_values = _as_bool_list(
        _required(matrix_cfg, "dummy_gil_thread"),
        "execution.mode_matrix.dummy_gil_thread",
    )

    dummy_inner_loops = _as_int(
        execution_cfg.get("dummy_gil_inner_loops", 200_000),
        "execution.dummy_gil_inner_loops",
    )
    dummy_switch_interval = _optional_float(
        execution_cfg.get("dummy_gil_switch_interval_s"),
        "execution.dummy_gil_switch_interval_s",
    )

    modes: list[ExecutionMode] = []
    for precompute in precompute_values:
        for preallocate in preallocate_values:
            for gil_enabled in gil_values:
                modes.append(
                    ExecutionMode(
                        name=_mode_name(precompute, preallocate, gil_enabled),
                        precompute_static_tensors=precompute,
                        preallocate_work_buffers=preallocate,
                        enable_dummy_gil_thread=gil_enabled,
                        dummy_gil_inner_loops=dummy_inner_loops,
                        dummy_gil_switch_interval_s=(
                            dummy_switch_interval if gil_enabled else None
                        ),
                    )
                )

    return modes


def _load_explicit_modes(
    explicit_modes: Any,
    execution_cfg: Mapping[str, Any],
) -> list[ExecutionMode]:
    if not isinstance(explicit_modes, Sequence) or isinstance(explicit_modes, str):
        raise TypeError("execution.modes must be a YAML sequence.")

    default_inner_loops = _as_int(
        execution_cfg.get("dummy_gil_inner_loops", 200_000),
        "execution.dummy_gil_inner_loops",
    )
    default_switch_interval = _optional_float(
        execution_cfg.get("dummy_gil_switch_interval_s"),
        "execution.dummy_gil_switch_interval_s",
    )

    modes: list[ExecutionMode] = []
    for index, item in enumerate(explicit_modes):
        if not isinstance(item, Mapping):
            raise TypeError(f"execution.modes[{index}] must be a mapping.")

        precompute = _as_bool(
            _required(item, "precompute_static_tensors"),
            f"execution.modes[{index}].precompute_static_tensors",
        )
        preallocate = _as_bool(
            _required(item, "preallocate_work_buffers"),
            f"execution.modes[{index}].preallocate_work_buffers",
        )
        gil_enabled = _as_bool(
            item.get("enable_dummy_gil_thread", False),
            f"execution.modes[{index}].enable_dummy_gil_thread",
        )

        modes.append(
            ExecutionMode(
                name=str(item.get("name", _mode_name(precompute, preallocate, gil_enabled))),
                precompute_static_tensors=precompute,
                preallocate_work_buffers=preallocate,
                enable_dummy_gil_thread=gil_enabled,
                dummy_gil_inner_loops=_as_int(
                    item.get("dummy_gil_inner_loops", default_inner_loops),
                    f"execution.modes[{index}].dummy_gil_inner_loops",
                ),
                dummy_gil_switch_interval_s=(
                    _optional_float(
                        item.get("dummy_gil_switch_interval_s", default_switch_interval),
                        f"execution.modes[{index}].dummy_gil_switch_interval_s",
                    )
                    if gil_enabled
                    else None
                ),
            )
        )

    return modes


# ============================================================================
# Validation and small parsing helpers
# ============================================================================


def _validate_params(params: Params) -> None:
    if params.batch_frames <= 0:
        raise ValueError("schedule.batch_frames must be positive.")
    if params.batches_per_output <= 0:
        raise ValueError("schedule.batches_per_output must be positive.")
    if params.sample_rate_hz <= 0.0:
        raise ValueError("signal.sample_rate_hz must be positive.")
    if params.doppler_low_hz < 0.0:
        raise ValueError("signal.doppler_low_hz must be non-negative.")
    if not params.doppler_low_hz < params.doppler_high_hz:
        raise ValueError("Expected doppler_low_hz < doppler_high_hz.")
    if params.doppler_high_hz > params.sample_rate_hz / 2.0:
        raise ValueError("doppler_high_hz exceeds the Nyquist frequency.")
    if params.propagation_distance_m <= 0.0:
        raise ValueError("signal.propagation_distance_m must be positive.")
    if params.wavelength_m <= 0.0:
        raise ValueError("signal.wavelength_m must be positive.")
    if params.dx_m <= 0.0 or params.dy_m <= 0.0:
        raise ValueError("signal.dx_m and signal.dy_m must be positive.")
    if params.benchmark_seconds <= 0.0:
        raise ValueError("benchmark.seconds must be positive.")
    if params.warmup_outputs < 0:
        raise ValueError("benchmark.warmup_outputs must be non-negative.")
    if not (0.0 < params.contrast_roi_radius <= 1.0):
        raise ValueError("display.contrast_roi_radius must lie in (0, 1].")
    if not (
        0.0
        <= params.contrast_low_percentile
        < params.contrast_high_percentile
        <= 100.0
    ):
        raise ValueError(
            "display percentiles must satisfy 0 <= low < high <= 100."
        )


def _input_file_path(input_cfg: Mapping[str, Any]) -> str:
    value = input_cfg.get("file_path", input_cfg.get("path"))
    if value is None:
        raise KeyError("Missing required configuration key: input.file_path")
    return str(value)


def _mode_name(precompute: bool, preallocate: bool, gil_enabled: bool) -> str:
    return (
        f"cupy-naive | "
        f"precompute={'on' if precompute else 'off'} | "
        f"prealloc={'on' if preallocate else 'off'} | "
        f"gil={'on' if gil_enabled else 'off'}"
    )


def _section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(mapping, key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required configuration key: {key}")
    return mapping[key]


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float, not bool.")
    return float(value)


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, name)


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _as_bool_list(value: Any, name: str) -> list[bool]:
    if isinstance(value, bool):
        return [value]

    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be a boolean or a list of booleans.")

    return [_as_bool(item, f"{name}[{index}]") for index, item in enumerate(value)]
