from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from .cupy_naive import BenchmarkStats


# ============================================================================
# Console reporting
# ============================================================================


def print_table(title: str, rows: list[list[object]]) -> None:
    print(f"\n{title}")
    print(tabulate(rows, tablefmt="github"))


def print_stats(stats: BenchmarkStats) -> None:
    print("\nSteady-state benchmark")
    print("----------------------")

    print_table(
        "Mode",
        [
            ["Name", stats.mode_name],
            ["Precompute static tensors", stats.precompute_static_tensors],
            ["Preallocate work buffers", stats.preallocate_work_buffers],
        ],
    )

    print_table(
        "Timing",
        [
            ["Time", f"{stats.seconds:.3f} s"],
            ["Frames", stats.frames],
            ["Batches", stats.batches],
            ["Outputs", stats.outputs],
            ["Wall time / output", f"{stats.wall_ms_per_output:.2f} ms"],
        ],
    )

    print_table(
        "Rates",
        [
            ["Input FPS", f"{stats.input_fps:.1f} frames/s"],
            ["Batches / s", f"{stats.batches_per_second:.2f}"],
            ["Outputs / s", f"{stats.outputs_per_second:.2f}"],
        ],
    )

    print_table(
        "Bandwidth",
        [
            ["H2D", f"{stats.h2d_gbps:.3f} GB/s"],
            ["Cast throughput", f"{stats.cast_effective_gbps:.3f} GB/s"],
            ["D2H output", f"{stats.d2h_output_mbps:.3f} MB/s"],
        ],
    )

    print_table(
        "Data types",
        [
            ["File", stats.file_dtype],
            ["Host", stats.host_dtype],
            ["Device input", stats.device_input_dtype],
            ["Real", stats.real_dtype],
            ["Complex", stats.complex_dtype],
        ],
    )

    print_table(
        "Pipeline",
        [
            [
                "Doppler bins",
                f"[{stats.doppler_bins[0]}:{stats.doppler_bins[1]}) "
                f"({stats.doppler_bin_count})",
            ],
            [
                "Sliding window",
                f"{stats.batches_per_output} batches = {stats.frames_per_output} frames",
            ],
            ["Output stride", f"1 batch = {stats.output_stride_frames} frames"],
            ["Temporal support", f"{stats.temporal_support_ms:.3f} ms"],
        ],
    )

    gil_rows = [
        [
            "Dummy GIL thread",
            "enabled" if stats.dummy_gil_thread_enabled else "disabled",
        ],
        ["Dummy inner loops", stats.dummy_gil_inner_loops],
        [
            "Switch interval",
            (
                "default"
                if stats.dummy_gil_switch_interval_s is None
                else f"{stats.dummy_gil_switch_interval_s:.6f} s"
            ),
        ],
    ]

    if stats.dummy_gil_thread_enabled:
        gil_rows.extend(
            [
                ["Dummy iterations", stats.dummy_gil_iterations],
                ["Dummy rate", f"{stats.dummy_gil_iterations_per_second:.0f} iter/s"],
            ]
        )

    print_table("Host-side GIL stress", gil_rows)


def print_suite_summary(stats_list: Sequence[BenchmarkStats]) -> None:
    rows: list[list[object]] = []

    for stats in stats_list:
        rows.append(
            [
                stats.mode_name,
                "on" if stats.precompute_static_tensors else "off",
                "on" if stats.preallocate_work_buffers else "off",
                "on" if stats.dummy_gil_thread_enabled else "off",
                (
                    "default"
                    if stats.dummy_gil_switch_interval_s is None
                    else f"{stats.dummy_gil_switch_interval_s:.6f}"
                ),
                f"{stats.outputs_per_second:.2f}",
                f"{stats.wall_ms_per_output:.2f}",
                f"{stats.input_fps:.0f}",
                f"{stats.h2d_gbps:.3f}",
                f"{stats.cast_effective_gbps:.3f}",
                f"{stats.d2h_output_mbps:.3f}",
                f"{stats.dummy_gil_iterations_per_second:.0f}",
            ]
        )

    print("\nSuite summary")
    print("-------------")
    print(
        tabulate(
            rows,
            headers=[
                "Mode",
                "Precompute",
                "Preallocate",
                "GIL",
                "Switch interval (s)",
                "Outputs/s",
                "ms/output",
                "Input FPS",
                "H2D GB/s",
                "Cast GB/s",
                "D2H MB/s",
                "Dummy iter/s",
            ],
            tablefmt="github",
        )
    )


# ============================================================================
# Display
# ============================================================================


def show_image(image: np.ndarray, stats: BenchmarkStats) -> None:
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(
        f"{stats.mode_name}\n"
        f"Power Doppler sliding average "
        f"({stats.frames_per_output} frames support, "
        f"{stats.outputs_per_second:.2f} outputs/s)"
    )
    plt.colorbar()
    plt.tight_layout()
    plt.show()
