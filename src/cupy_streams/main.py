from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cupyx.profiler import time_range

from holoflow_benchmarks.config import load_benchmark_config, load_yaml_mapping
from holoflow_benchmarks.io import preload_batches, read_input_info, validate_input
from holoflow_benchmarks.reporting import format_report, show_image, write_report
from holoflow_benchmarks.runtime import clear_cupy_pools

from .benchmark import benchmark_suite
from .schedule import SingleThreadStreamRuntimeConfig


DEFAULT_CONFIG_PATH = Path("config_cupy_streams.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the single-thread CUDA-stream CuPy LDH benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Benchmark YAML file. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def load_stream_runtime_config(path: str | Path) -> SingleThreadStreamRuntimeConfig:
    raw = load_yaml_mapping(path)

    streams_cfg = raw.get("streams", {})
    if not isinstance(streams_cfg, Mapping):
        raise TypeError("streams must be a mapping.")

    return SingleThreadStreamRuntimeConfig(
        num_slots=_as_positive_int(
            streams_cfg.get("num_slots", SingleThreadStreamRuntimeConfig.num_slots),
            "streams.num_slots",
        ),
        h2d_prefetch_batches=_as_non_negative_int(
            streams_cfg.get(
                "h2d_prefetch_batches",
                SingleThreadStreamRuntimeConfig.h2d_prefetch_batches,
            ),
            "streams.h2d_prefetch_batches",
        ),
    )


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}.")
    return parsed


def _as_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}.")
    return parsed


def main() -> None:
    args = parse_args()

    with time_range("load stream benchmark config", color_id=131):
        params, modes = load_benchmark_config(
            args.config,
            implementation_name="cupy-streams",
        )
        runtime = load_stream_runtime_config(args.config)

    print(f"Using config: {args.config}")
    print(
        f"Single-thread stream runtime: num_slots={runtime.num_slots}, "
        f"h2d_prefetch_batches={runtime.h2d_prefetch_batches}"
    )

    with time_range("streams inspect input", color_id=132):
        print("Inspecting input...")
        info = read_input_info(params.file_path)
        validate_input(info, params)

    with time_range("streams preload host data", color_id=133):
        print("Preloading host data...")
        host_batches = preload_batches(params.file_path, info, params)

    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in pinned host memory."
    )

    with time_range("streams benchmark suite", color_id=134):
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

    with time_range("write stream report", color_id=135):
        report_path = write_report(params.report_path, stats_list)
    print(f"Report written to: {report_path}")

    if params.show_image and results:
        with time_range("show stream image", color_id=136):
            image, stats = results[-1]
            show_image(image, stats)

    clear_cupy_pools()


if __name__ == "__main__":
    main()
