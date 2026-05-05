from __future__ import annotations

import argparse
from pathlib import Path

from cupyx.profiler import time_range

from holoflow_benchmarks.config import load_benchmark_config
from holoflow_benchmarks.io import preload_batches, read_input_info, validate_input
from holoflow_benchmarks.reporting import format_report, show_image, write_report
from holoflow_benchmarks.runtime import clear_cupy_pools

from .benchmark import benchmark_suite


DEFAULT_CONFIG_PATH = Path("config_cupy_naive.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CuPy-naive LDH benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Benchmark YAML file. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with time_range("load benchmark config", color_id=20):
        params, modes = load_benchmark_config(
            args.config,
            implementation_name="cupy-naive",
        )

    print(f"Using config: {args.config}")

    with time_range("inspect input", color_id=21):
        print("Inspecting input...")
        info = read_input_info(params.file_path)
        validate_input(info, params)

    with time_range("preload host data", color_id=22):
        print("Preloading host data...")
        host_batches = preload_batches(params.file_path, info, params)

    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in pinned host memory."
    )

    with time_range("benchmark suite", color_id=23):
        results = benchmark_suite(
            host_batches=host_batches,
            info=info,
            params=params,
            modes=modes,
        )

    stats_list = [stats for _, stats in results]
    report = format_report(stats_list)
    print(report)

    with time_range("write report", color_id=24):
        report_path = write_report(params.report_path, stats_list)
    print(f"Report written to: {report_path}")

    if params.show_image and results:
        with time_range("show image", color_id=25):
            image, stats = results[-1]
            show_image(image, stats)

    clear_cupy_pools()


if __name__ == "__main__":
    main()
