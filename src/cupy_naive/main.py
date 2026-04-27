from __future__ import annotations

import argparse
from pathlib import Path

from cupy_naive.config import load_benchmark_config
from cupy_naive.io import preload_batches, read_input_info, validate_input
from cupy_naive.cupy_naive import benchmark_suite
from cupy_naive.reporting import print_stats, print_suite_summary, show_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CuPy-naive LDH benchmark."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the YAML benchmark configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params, modes = load_benchmark_config(args.config)

    print("Inspecting input...")
    info = read_input_info(params.path)
    validate_input(info, params)

    print("Preloading host data...")
    host_batches = preload_batches(params.path, info, params)
    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in pinned host memory."
    )

    results = benchmark_suite(
        host_batches=host_batches,
        info=info,
        params=params,
        modes=modes,
    )

    stats_list = [stats for _, stats in results]

    print_suite_summary(stats_list)

    print("\nDetailed results")
    print("----------------")
    for stats in stats_list:
        print_stats(stats)

    if params.show_image and results:
        image, stats = results[-1]
        show_image(image, stats)


if __name__ == "__main__":
    main()
