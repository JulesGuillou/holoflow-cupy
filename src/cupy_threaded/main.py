from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cupyx.profiler import time_range
import yaml

from cupy_naive.config import load_benchmark_config
from cupy_naive.cupy_naive import clear_cupy_pools
from cupy_naive.io import preload_batches, read_input_info, validate_input
from cupy_naive.reporting import format_report, show_image, write_report

from .cupy_threaded import ThreadedRuntimeConfig, benchmark_suite


DEFAULT_CONFIG_PATH = Path("config_cupy_threaded.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-threaded CuPy LDH benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Benchmark YAML file. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def load_threaded_runtime_config(path: str | Path) -> ThreadedRuntimeConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if not isinstance(raw, Mapping):
        raise TypeError(f"{path} must contain a YAML mapping at the top level.")

    threading_cfg = raw.get("threading", {})
    if not isinstance(threading_cfg, Mapping):
        raise TypeError("threading must be a mapping.")

    execution_cfg = raw.get("execution", {})
    if execution_cfg is not None and not isinstance(execution_cfg, Mapping):
        raise TypeError("execution must be a mapping.")

    default_gil_switch_interval = (
        execution_cfg.get("dummy_gil_switch_interval_s")
        if isinstance(execution_cfg, Mapping)
        else None
    )

    return ThreadedRuntimeConfig(
        queue_depth=_as_positive_int(
            threading_cfg.get("queue_depth", ThreadedRuntimeConfig.queue_depth),
            "threading.queue_depth",
        ),
        queue_put_policy=_as_queue_put_policy(
            _first_present(
                threading_cfg,
                "queue_put_policy",
                "producer_submit_policy",
                default=ThreadedRuntimeConfig.queue_put_policy,
            ),
            "threading.queue_put_policy",
        ),
        queue_put_timeout_s=_as_positive_float(
            _first_present(
                threading_cfg,
                "queue_put_timeout_s",
                "producer_submit_timeout_s",
                default=ThreadedRuntimeConfig.queue_put_timeout_s,
            ),
            "threading.queue_put_timeout_s",
        ),
        gil_switch_interval_s=_as_optional_positive_float(
            threading_cfg.get("gil_switch_interval_s", default_gil_switch_interval),
            "threading.gil_switch_interval_s",
        ),
    )


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


def _first_present(
    mapping: Mapping[str, Any],
    primary_key: str,
    legacy_key: str,
    *,
    default: Any,
) -> Any:
    if primary_key in mapping:
        return mapping[primary_key]
    return mapping.get(legacy_key, default)


def _as_queue_put_policy(value: Any, name: str) -> str:
    parsed = str(value)
    if parsed not in {"timed_put", "nowait"}:
        raise ValueError(f"{name} must be 'timed_put' or 'nowait', got {parsed!r}.")
    return parsed


def _as_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float, not bool.")
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


def _as_optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a float, not bool.")
    parsed = float(value)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


def main() -> None:
    args = parse_args()

    with time_range("load threaded benchmark config", color_id=70):
        params, modes = load_benchmark_config(args.config)
        runtime = load_threaded_runtime_config(args.config)

    print(f"Using config: {args.config}")
    print(
        f"Threaded runtime: queue_depth={runtime.queue_depth}, "
        f"queue_put_policy={runtime.queue_put_policy}, "
        f"queue_put_timeout_s={runtime.queue_put_timeout_s}, "
        f"gil_switch_interval_s={runtime.gil_switch_interval_s}"
    )

    with time_range("threaded inspect input", color_id=71):
        print("Inspecting input...")
        info = read_input_info(params.file_path)
        validate_input(info, params)

    with time_range("threaded preload host data", color_id=72):
        print("Preloading host data...")
        host_batches = preload_batches(params.file_path, info, params)

    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in pinned host memory."
    )

    with time_range("threaded benchmark suite", color_id=73):
        results = benchmark_suite(
            host_batches=host_batches,
            info=info,
            params=params,
            modes=modes,
            runtime=runtime,
        )

    stats_list = [stats for _, stats in results]
    report = format_report(stats_list)
    print(report)

    with time_range("write threaded report", color_id=74):
        report_path = write_report(params.report_path, stats_list)
    print(f"Report written to: {report_path}")

    if params.show_image and results:
        with time_range("show threaded image", color_id=75):
            image, stats = results[-1]
            show_image(image, stats)

    clear_cupy_pools()


if __name__ == "__main__":
    main()
