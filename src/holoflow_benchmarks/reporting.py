from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from .stats import BenchmarkSeries, BenchmarkStats, summarize_throughput


def format_table(title: str, rows: list[list[object]]) -> str:
    return f"\n{title}\n{tabulate(rows, tablefmt='github')}"


def format_stats(stats: BenchmarkStats) -> str:
    sections = ["\nSteady-state benchmark", "----------------------"]

    sections.append(
        format_table(
            "Mode",
            [
                ["Name", stats.mode_name],
                ["Precompute static tensors", stats.precompute_static_tensors],
                ["Preallocate work buffers", stats.preallocate_work_buffers],
            ],
        )
    )

    sections.append(
        format_table(
            "Timing",
            [
                ["Time", f"{stats.seconds:.3f} s"],
                ["Frames", stats.frames],
                ["Batches", stats.batches],
                ["Outputs", stats.outputs],
                ["Wall time / output", f"{stats.wall_ms_per_output:.2f} ms"],
            ],
        )
    )

    sections.append(
        format_table(
            "Rates",
            [
                ["Input FPS", f"{stats.input_fps:.1f} frames/s"],
                ["Batches / s", f"{stats.batches_per_second:.2f}"],
                ["Outputs / s", f"{stats.outputs_per_second:.2f}"],
            ],
        )
    )

    sections.append(
        format_table(
            "Bandwidth",
            [
                ["H2D", f"{stats.h2d_gbps:.3f} GB/s"],
                ["Cast throughput", f"{stats.cast_effective_gbps:.3f} GB/s"],
                ["D2H output", f"{stats.d2h_output_mbps:.3f} MB/s"],
            ],
        )
    )

    sections.append(
        format_table(
            "Data types",
            [
                ["File", stats.file_dtype],
                ["Host", stats.host_dtype],
                ["Device input", stats.device_input_dtype],
                ["Real", stats.real_dtype],
                ["Complex", stats.complex_dtype],
            ],
        )
    )

    sections.append(
        format_table(
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

    sections.append(format_table("Host-side GIL stress", gil_rows))

    return "\n".join(sections)


def _mean_stat(series: BenchmarkSeries, field: str) -> float:
    return fmean(float(getattr(run, field)) for run in series.runs)


def _optional_number(value: float | None, digits: int) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def format_suite_summary(series_list: Sequence[BenchmarkSeries]) -> str:
    rows: list[list[object]] = []

    for series in series_list:
        stats = series.runs[0]
        summary = summarize_throughput(series)
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
                summary.repetitions,
                f"{summary.mean_input_fps:.1f}",
                _optional_number(summary.sample_std_input_fps, 1),
                _optional_number(summary.coefficient_of_variation_percent, 2),
                f"{summary.min_input_fps:.1f}",
                f"{summary.max_input_fps:.1f}",
                f"{_mean_stat(series, 'outputs_per_second'):.2f}",
                f"{_mean_stat(series, 'wall_ms_per_output'):.2f}",
                f"{_mean_stat(series, 'h2d_gbps'):.3f}",
                f"{_mean_stat(series, 'cast_effective_gbps'):.3f}",
                f"{_mean_stat(series, 'd2h_output_mbps'):.3f}",
            ]
        )

    return "\n".join(
        [
            "\nSuite summary",
            "-------------",
            tabulate(
                rows,
                headers=[
                    "Mode",
                    "Precompute",
                    "Preallocate",
                    "GIL",
                    "Switch interval (s)",
                    "Runs",
                    "Mean input FPS",
                    "Sample SD FPS",
                    "CV (%)",
                    "Min FPS",
                    "Max FPS",
                    "Mean outputs/s",
                    "Mean ms/output",
                    "Mean H2D GB/s",
                    "Mean cast GB/s",
                    "Mean D2H MB/s",
                ],
                tablefmt="github",
                disable_numparse=True,
            ),
        ]
    )


def format_raw_throughput(series_list: Sequence[BenchmarkSeries]) -> str:
    rows: list[list[object]] = []
    for series in series_list:
        for run_index, stats in enumerate(series.runs, start=1):
            rows.append(
                [
                    stats.mode_name,
                    run_index,
                    f"{stats.seconds:.6f}",
                    stats.frames,
                    stats.batches,
                    stats.outputs,
                    f"{stats.input_fps:.1f}",
                ]
            )

    return "\n".join(
        [
            "\nRaw throughput measurements",
            "---------------------------",
            tabulate(
                rows,
                headers=[
                    "Mode",
                    "Run",
                    "Time (s)",
                    "Frames",
                    "Batches",
                    "Outputs",
                    "Input FPS",
                ],
                tablefmt="github",
                disable_numparse=True,
            ),
        ]
    )


def format_mode_configuration(series: BenchmarkSeries) -> str:
    stats = series.runs[0]
    return "\n".join(
        [
            f"\n{stats.mode_name}",
            "~" * len(stats.mode_name),
            format_table(
                "Execution",
                [
                    ["Repetitions", len(series.runs)],
                    ["Precompute static tensors", stats.precompute_static_tensors],
                    ["Preallocate work buffers", stats.preallocate_work_buffers],
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
                ],
            ),
            format_table(
                "Data types",
                [
                    ["File", stats.file_dtype],
                    ["Host", stats.host_dtype],
                    ["Device input", stats.device_input_dtype],
                    ["Real", stats.real_dtype],
                    ["Complex", stats.complex_dtype],
                ],
            ),
            format_table(
                "Pipeline",
                [
                    [
                        "Doppler bins",
                        f"[{stats.doppler_bins[0]}:{stats.doppler_bins[1]}) "
                        f"({stats.doppler_bin_count})",
                    ],
                    [
                        "Sliding window",
                        f"{stats.batches_per_output} batches = "
                        f"{stats.frames_per_output} frames",
                    ],
                    ["Output stride", f"{stats.output_stride_frames} frames"],
                    ["Temporal support", f"{stats.temporal_support_ms:.3f} ms"],
                ],
            ),
        ]
    )


def format_report(series_list: Sequence[BenchmarkSeries]) -> str:
    sections = [
        format_suite_summary(series_list),
        format_raw_throughput(series_list),
        "\nConfiguration details",
        "---------------------",
    ]
    sections.extend(format_mode_configuration(series) for series in series_list)
    return "\n".join(sections).lstrip() + "\n"


def write_report(path: str | Path, series_list: Sequence[BenchmarkSeries]) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_report(series_list), encoding="utf-8")
    return report_path


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

