from __future__ import annotations

import argparse
from pathlib import Path

from holoflow_benchmarks.config import load_benchmark_config
from holoflow_benchmarks.io import read_input_info, validate_input
from holoflow_benchmarks.reporting import format_report, show_image, write_report

from .benchmark import benchmark_suite
from .io import preload_batches
from .nvtx import time_range
from .runtime import clear_torch_pools


DEFAULT_CONFIG_PATH = Path("config_pytorch_naive.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyTorch-naive LDH benchmark.",
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

    with time_range("pytorch load benchmark config", color_id=220):
        params, modes = load_benchmark_config(
            args.config,
            implementation_name="pytorch-naive",
        )

    print(f"Using config: {args.config}")

    with time_range("pytorch inspect input", color_id=221):
        print("Inspecting input...")
        info = read_input_info(params.file_path)
        validate_input(info, params)

    with time_range("pytorch preload host data", color_id=222):
        print("Preloading host data...")
        host_batches = preload_batches(params.file_path, info, params)

    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in PyTorch-pinned host memory."
    )

    with time_range("pytorch benchmark suite", color_id=223):
        results = benchmark_suite(
            host_batches=host_batches,
            info=info,
            params=params,
            modes=modes,
        )

    stats_list = [stats for _, stats in results]
    report = format_report(stats_list)
    print(report)

    with time_range("pytorch write report", color_id=224):
        report_path = write_report(params.report_path, stats_list)
    print(f"Report written to: {report_path}")

    if params.show_image and results:
        with time_range("pytorch show image", color_id=225):
            image, stats = results[-1]
            show_image(image, stats)

    clear_torch_pools()


if __name__ == "__main__":
    main()
