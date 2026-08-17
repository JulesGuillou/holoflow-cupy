from __future__ import annotations

import argparse
from pathlib import Path

from holoflow_benchmarks.config import load_benchmark_config
from holoflow_benchmarks.io import read_input_info, validate_input
from holoflow_benchmarks.reporting import format_report, show_image, write_report

from .platform import require_linux


DEFAULT_CONFIG_PATH = Path("config_jax_naive.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Linux-only JAX-naive LDH benchmark.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Benchmark YAML file. Defaults to {DEFAULT_CONFIG_PATH}.",
    )
    return parser.parse_args()


def main() -> None:
    require_linux("JAX-naive benchmark")

    from .benchmark import benchmark_suite
    from .io import preload_batches
    from .nvtx import time_range
    from .runtime import clear_jax_runtime

    args = parse_args()

    with time_range("jax load benchmark config", color_id=420):
        params, modes = load_benchmark_config(
            args.config,
            implementation_name="jax-naive",
        )

    print(f"Using config: {args.config}")

    with time_range("jax inspect input", color_id=421):
        print("Inspecting input...")
        info = read_input_info(params.file_path)
        validate_input(info, params)

    with time_range("jax preload host data", color_id=422):
        print("Preloading host data...")
        host_batches = preload_batches(params.file_path, info, params)

    print(
        f"Preloaded {params.temporal_support_frames} frames "
        f"of shape ({info.height}, {info.width}) "
        f"in NumPy host memory for JAX transfer."
    )

    with time_range("jax benchmark suite", color_id=423):
        results = benchmark_suite(
            host_batches=host_batches,
            info=info,
            params=params,
            modes=modes,
        )

    report = format_report(results)
    print(report)

    with time_range("jax write report", color_id=424):
        report_path = write_report(params.report_path, results)
    print(f"Report written to: {report_path}")

    if params.show_image and results:
        with time_range("jax show image", color_id=425):
            result = results[-1]
            show_image(result.image, result.runs[-1])

    clear_jax_runtime()


if __name__ == "__main__":
    main()
