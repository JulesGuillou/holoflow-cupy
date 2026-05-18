from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from holoflow_benchmarks.nsys_export_sqlite import export_sqlite, find_nsys
from holoflow_benchmarks.nsys_iteration_metrics import MetricsRow
from holoflow_benchmarks.nsys_iteration_metrics import compute_metrics


SUPPORTED_BENCHMARKS = (
    "cupy_naive",
    "cupy_threaded",
    "cupy_streams",
    "pytorch_naive",
    "pytorch_threaded",
    "pytorch_streams",
    "jax_naive",
    "jax_streams",
)
SUPPORTED_PLATFORMS = ("linux", "windows")

MINIMAL_NSYS_TABLES = (
    "NVTX_EVENTS",
    "StringIds",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_MEMCPY",
)

NAIVE_TABLE_SCOPES = (
    ("iteration", "Total"),
    ("process_batch", "Process batch"),
    ("export_display_image", "Export display"),
)

NAIVE_SUMMARY_SCOPES = (
    "iteration",
    "process_batch",
    "export_display_image",
    "sync_per_output",
)

NAIVE_TABLE_ROWS = (
    ("Duration (\\si{\\micro\\second})", "duration_avg_us", "us"),
    ("CUDA API non-gap (\\si{\\micro\\second})", "cuda_api_non_gap_avg_us", "us"),
    ("CUDA API non-gap (\\%)", "cuda_api_non_gap_pct", "pct"),
    ("CUDA API gap (\\si{\\micro\\second})", "cuda_api_gap_avg_us", "us"),
    ("CUDA API gap (\\%)", "cuda_api_gap_pct", "pct"),
    ("GPU compute (\\si{\\micro\\second})", "gpu_compute_avg_us", "us"),
    ("GPU compute (\\%)", "gpu_compute_pct", "pct"),
    ("GPU memcpy (\\si{\\micro\\second})", "gpu_memcpy_avg_us", "us"),
    ("GPU memcpy (\\%)", "gpu_memcpy_pct", "pct"),
    ("GPU active (\\si{\\micro\\second})", "gpu_any_avg_us", "us"),
    ("GPU active (\\%)", "gpu_any_pct", "pct"),
    ("GPU idle (\\si{\\micro\\second})", "gpu_idle_avg_us", "us"),
    ("GPU idle (\\%)", "gpu_idle_pct", "pct"),
)

THREAD_STAGE_NAMES = {
    "fft": "threaded FFT batch",
    "postprocess": "threaded postprocess batch",
}

THREAD_COPY_STAGES = {
    "h2d": {
        "copy": "threaded H2D upload",
        "sync": "threaded H2D sync before enqueue",
    },
    "d2h": {
        "copy": "threaded D2H output",
        "sync": "threaded D2H sync before enqueue",
    },
}

MAX_COPY_SYNC_GAP_NS = 100_000_000

THREADED_TABLE_ROWS = (
    ("H2D copy avg (\\si{\\micro\\second})", "h2d_copy_avg_us", "us"),
    ("H2D gap+sync avg (\\si{\\micro\\second})", "h2d_gap_sync_avg_us", "us"),
    ("H2D virtual avg (\\si{\\micro\\second})", "h2d_avg_us", "us"),
    ("FFT avg (\\si{\\micro\\second})", "fft_avg_us", "us"),
    ("Postprocess avg (\\si{\\micro\\second})", "postprocess_avg_us", "us"),
    ("D2H copy avg (\\si{\\micro\\second})", "d2h_copy_avg_us", "us"),
    ("D2H gap+sync avg (\\si{\\micro\\second})", "d2h_gap_sync_avg_us", "us"),
    ("D2H virtual avg (\\si{\\micro\\second})", "d2h_avg_us", "us"),
    ("H2D hidden by postprocess (\\%)", "h2d_hidden_by_postprocess_pct", "pct"),
    ("D2H hidden by postprocess (\\%)", "d2h_hidden_by_postprocess_pct", "pct"),
    ("Copy work hidden by postprocess (\\%)", "memcpy_work_hidden_by_postprocess_pct", "pct"),
    ("Postprocess covered by copies (\\%)", "postprocess_covered_by_memcpy_pct", "pct"),
    ("H2D-D2H overlap (\\%)", "h2d_d2h_overlap_pct", "pct"),
)

THREADED_GLOBAL_TABLE_ROWS = NAIVE_TABLE_ROWS
STREAMS_GLOBAL_TABLE_ROWS = NAIVE_TABLE_ROWS
PYTORCH_GLOBAL_TABLE_ROWS = NAIVE_TABLE_ROWS
PYTORCH_NAIVE_SKIP_INITIAL_NS = 1_000_000_000
PYTORCH_THREADED_SKIP_INITIAL_NS = 2_000_000_000
PYTORCH_STREAMS_SKIP_INITIAL_NS = 1_500_000_000
JAX_NAIVE_REFERENCE_NAME = (
    "XlaModule:#hlo_module=jit__process_ready_output_core,program_id=25"
)


@dataclass(frozen=True)
class ReportContext:
    platform: str
    benchmark: str
    report_path: Path
    sqlite_path: Path
    config_path: Path | None
    mode_index: int | None
    mode_name: str
    precompute_static_tensors: bool | None
    preallocate_work_buffers: bool | None
    dummy_gil_thread_enabled: bool | None

    @property
    def mode_label(self) -> str:
        return (
            f"P{_flag(self.precompute_static_tensors)} "
            f"A{_flag(self.preallocate_work_buffers)} "
            f"G{_flag(self.dummy_gil_thread_enabled)}"
        )


@dataclass(frozen=True)
class NvtxRange:
    name: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class CopySyncPair:
    copy_start_ns: int
    copy_end_ns: int
    sync_start_ns: int
    sync_end_ns: int

    @property
    def virtual_interval(self) -> tuple[int, int]:
        return (self.copy_start_ns, self.sync_end_ns)

    @property
    def copy_interval(self) -> tuple[int, int]:
        return (self.copy_start_ns, self.copy_end_ns)

    @property
    def sync_interval(self) -> tuple[int, int]:
        return (self.sync_start_ns, self.sync_end_ns)

    @property
    def gap_ns(self) -> int:
        return max(0, self.sync_start_ns - self.copy_end_ns)

    @property
    def gap_sync_ns(self) -> int:
        return max(0, self.sync_end_ns - self.copy_end_ns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate focused Nsight metrics and LaTeX tables from the "
            "generated/ profiling folder."
        )
    )
    parser.add_argument(
        "generated_root",
        type=Path,
        nargs="?",
        default=Path("generated"),
        help="Folder containing linux/windows reports and Nsight report folders.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=SUPPORTED_BENCHMARKS,
        default=None,
        help=(
            "Benchmark to process. Defaults to cupy_naive, cupy_threaded, "
            "cupy_streams, pytorch_naive, pytorch_threaded, and "
            "pytorch_streams, jax_naive, and jax_streams."
        ),
    )
    parser.add_argument(
        "--platform",
        action="append",
        choices=SUPPORTED_PLATFORMS,
        default=None,
        help="Platform to process. Defaults to all discovered platforms.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV output directory. Defaults to <generated_root>/metrics.",
    )
    parser.add_argument(
        "--latex-dir",
        type=Path,
        default=None,
        help="LaTeX output directory. Defaults to <generated_root>/latex.",
    )
    parser.add_argument(
        "--nsys",
        type=Path,
        default=None,
        help="Optional path to nsys or nsys.exe for missing SQLite exports.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Do not export missing SQLite files from .nsys-rep files.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export SQLite files even if they already exist.",
    )
    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=0,
        help="Complete naive iterations to skip before averaging.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum complete naive iterations to average per report.",
    )
    parser.add_argument(
        "--limit-reports",
        type=int,
        default=None,
        help="Debug helper: process at most this many reports after filtering.",
    )

    args = parser.parse_args()

    generated_root = args.generated_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else generated_root / "metrics"
    )
    latex_dir = (
        args.latex_dir.resolve()
        if args.latex_dir is not None
        else generated_root / "latex"
    )

    benchmarks = tuple(args.benchmark or SUPPORTED_BENCHMARKS)
    platforms = tuple(args.platform) if args.platform else None

    reports = discover_reports(
        generated_root,
        benchmarks=benchmarks,
        platforms=platforms,
    )
    if args.limit_reports is not None:
        reports = reports[: args.limit_reports]

    if not reports:
        raise RuntimeError("No matching .nsys-rep files were found.")

    ensure_sqlite_exports(
        reports,
        nsys_arg=args.nsys,
        skip_export=args.skip_export,
        force_export=args.force_export,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    by_benchmark: dict[str, list[ReportContext]] = defaultdict(list)
    for report in reports:
        by_benchmark[report.benchmark].append(report)

    written: list[Path] = []

    if "cupy_naive" in by_benchmark:
        naive_csv_rows = build_naive_outputs(
            by_benchmark["cupy_naive"],
            output_dir=output_dir,
            latex_dir=latex_dir,
            skip_warmup=args.skip_warmup,
            max_iterations=args.max_iterations,
        )
        written.extend(naive_csv_rows)

    if "cupy_threaded" in by_benchmark:
        threaded_outputs = build_threaded_outputs(
            by_benchmark["cupy_threaded"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(threaded_outputs)

    if "cupy_streams" in by_benchmark:
        streams_outputs = build_streams_outputs(
            by_benchmark["cupy_streams"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(streams_outputs)

    if "pytorch_naive" in by_benchmark:
        pytorch_outputs = build_pytorch_naive_outputs(
            by_benchmark["pytorch_naive"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(pytorch_outputs)

    if "pytorch_threaded" in by_benchmark:
        pytorch_threaded_outputs = build_pytorch_threaded_outputs(
            by_benchmark["pytorch_threaded"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(pytorch_threaded_outputs)

    if "pytorch_streams" in by_benchmark:
        pytorch_streams_outputs = build_pytorch_streams_outputs(
            by_benchmark["pytorch_streams"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(pytorch_streams_outputs)

    if "jax_naive" in by_benchmark:
        jax_naive_outputs = build_jax_naive_outputs(
            by_benchmark["jax_naive"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(jax_naive_outputs)

    if "jax_streams" in by_benchmark:
        jax_streams_outputs = build_jax_streams_outputs(
            by_benchmark["jax_streams"],
            output_dir=output_dir,
            latex_dir=latex_dir,
        )
        written.extend(jax_streams_outputs)

    print("Wrote:")
    for path in written:
        print(f"  {path}")

    return 0


def discover_reports(
    generated_root: Path,
    *,
    benchmarks: tuple[str, ...],
    platforms: tuple[str, ...] | None,
) -> list[ReportContext]:
    reports: list[ReportContext] = []

    for report_path in generated_root.rglob("*.nsys-rep"):
        benchmark = report_path.parent.name
        if benchmark not in benchmarks:
            continue

        platform = infer_platform(report_path)
        if platform is None:
            continue
        if platforms is not None and platform not in platforms:
            continue

        reports.append(
            build_report_context(
                report_path=report_path,
                platform=platform,
                benchmark=benchmark,
            )
        )

    reports.sort(key=report_sort_key)
    return reports


def infer_platform(path: Path) -> str | None:
    for part in path.parts:
        lower = part.lower()
        if "windows" in lower:
            return "windows"
        if "linux" in lower:
            return "linux"
    return None


def build_report_context(
    *,
    report_path: Path,
    platform: str,
    benchmark: str,
) -> ReportContext:
    report_path = report_path.resolve()
    sqlite_path = report_path.with_suffix(".sqlite")
    config_path = find_single_mode_config(report_path, benchmark)
    mode = read_mode_metadata(config_path, report_path.stem)

    return ReportContext(
        platform=platform,
        benchmark=benchmark,
        report_path=report_path,
        sqlite_path=sqlite_path,
        config_path=config_path,
        mode_index=mode["mode_index"],
        mode_name=mode["mode_name"],
        precompute_static_tensors=mode["precompute_static_tensors"],
        preallocate_work_buffers=mode["preallocate_work_buffers"],
        dummy_gil_thread_enabled=mode["dummy_gil_thread_enabled"],
    )


def find_single_mode_config(report_path: Path, benchmark: str) -> Path | None:
    parts = list(report_path.parts)
    try:
        report_root_index = len(parts) - 1 - parts[::-1].index("nsys_reports")
    except ValueError:
        return None

    report_root = Path(*parts[: report_root_index + 1])
    config_path = report_root / "configs" / benchmark / f"{report_path.stem}.yaml"
    if config_path.exists():
        return config_path.resolve()
    return None


def read_mode_metadata(config_path: Path | None, report_stem: str) -> dict[str, Any]:
    fallback = parse_mode_from_stem(report_stem)

    if config_path is None:
        return fallback

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return fallback

    modes = data.get("execution", {}).get("modes", [])
    mode = modes[0] if modes else {}

    return {
        "mode_index": fallback["mode_index"],
        "mode_name": str(mode.get("name") or fallback["mode_name"]),
        "precompute_static_tensors": _optional_bool(
            mode.get(
                "precompute_static_tensors",
                fallback["precompute_static_tensors"],
            )
        ),
        "preallocate_work_buffers": _optional_bool(
            mode.get(
                "preallocate_work_buffers",
                fallback["preallocate_work_buffers"],
            )
        ),
        "dummy_gil_thread_enabled": _optional_bool(
            mode.get(
                "enable_dummy_gil_thread",
                fallback["dummy_gil_thread_enabled"],
            )
        ),
    }


def parse_mode_from_stem(report_stem: str) -> dict[str, Any]:
    match = re.match(r"(?P<index>\d+)_", report_stem)
    mode_index = int(match.group("index")) if match else None
    lower = report_stem.lower()

    return {
        "mode_index": mode_index,
        "mode_name": report_stem,
        "precompute_static_tensors": _flag_from_slug(lower, "precompute"),
        "preallocate_work_buffers": _flag_from_slug(lower, "prealloc"),
        "dummy_gil_thread_enabled": _flag_from_slug(lower, "gil"),
    }


def _flag_from_slug(slug: str, name: str) -> bool | None:
    if f"{name}_on" in slug:
        return True
    if f"{name}_off" in slug:
        return False
    return None


def ensure_sqlite_exports(
    reports: list[ReportContext],
    *,
    nsys_arg: Path | None,
    skip_export: bool,
    force_export: bool,
) -> None:
    missing = [
        report
        for report in reports
        if force_export or not report.sqlite_path.exists()
    ]

    if skip_export:
        absent = [report for report in reports if not report.sqlite_path.exists()]
        if absent:
            sample = "\n".join(f"  {report.sqlite_path}" for report in absent[:5])
            raise FileNotFoundError(
                "Missing SQLite exports and --skip-export was used:\n" + sample
            )
        return

    if not missing:
        return

    nsys_path = find_nsys(nsys_arg)
    for report in missing:
        export_sqlite(
            report.report_path,
            nsys_path=nsys_path,
            force=force_export,
            quiet=True,
            tables=MINIMAL_NSYS_TABLES,
        )


def build_naive_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
    skip_warmup: int,
    max_iterations: int | None,
) -> list[Path]:
    summary_rows: list[dict[str, Any]] = []
    summaries_by_report: dict[Path, dict[str, dict[str, Any]]] = {}

    for report in reports:
        rows = compute_metrics(
            report.sqlite_path,
            process_name="process_batch",
            export_name="export_display_image",
            sync_name="sync per output",
            skip_warmup=skip_warmup,
            max_iterations=max_iterations,
        )
        summary = summarize_naive_metrics(rows)
        summaries_by_report[report.report_path] = summary

        for scope in NAIVE_SUMMARY_SCOPES:
            if scope not in summary:
                continue
            summary_rows.append(
                {
                    **report_metadata_fields(report),
                    "scope": scope,
                    **summary[scope],
                }
            )

    csv_path = output_dir / "cupy_naive_scope_metrics.csv"
    write_csv(csv_path, summary_rows, fieldnames=NAIVE_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        latex_path = latex_dir / f"cupy_naive_metrics_{platform}.tex"
        latex_path.write_text(
            render_naive_latex(
                platform=platform,
                reports=platform_reports,
                summaries_by_report=summaries_by_report,
            ),
            encoding="utf-8",
        )
        written.append(latex_path)

    return written


def summarize_naive_metrics(rows: list[MetricsRow]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[MetricsRow]] = defaultdict(list)
    for row in rows:
        groups[row.scope].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for scope, items in groups.items():
        duration_avg_us = mean(row.duration_us for row in items)
        cuda_api_non_gap_avg_us = mean(row.cuda_api_non_gap_us for row in items)
        cuda_api_gap_avg_us = mean(row.cuda_api_gap_us for row in items)
        gpu_compute_avg_us = mean(row.gpu_compute_us for row in items)
        gpu_memcpy_avg_us = mean(row.gpu_memcpy_us for row in items)
        gpu_any_avg_us = mean(row.gpu_any_us for row in items)
        gpu_idle_avg_us = mean(row.gpu_idle_us for row in items)

        summaries[scope] = {
            "count": len(items),
            "duration_avg_us": duration_avg_us,
            "cuda_api_calls_avg": mean(row.cuda_api_calls for row in items),
            "cuda_api_non_gap_avg_us": cuda_api_non_gap_avg_us,
            "cuda_api_non_gap_pct": pct(cuda_api_non_gap_avg_us, duration_avg_us),
            "cuda_api_gap_avg_us": cuda_api_gap_avg_us,
            "cuda_api_gap_pct": pct(cuda_api_gap_avg_us, duration_avg_us),
            "kernel_calls_avg": mean(row.kernel_calls for row in items),
            "memcpy_calls_avg": mean(row.memcpy_calls for row in items),
            "gpu_compute_avg_us": gpu_compute_avg_us,
            "gpu_compute_pct": pct(gpu_compute_avg_us, duration_avg_us),
            "gpu_memcpy_avg_us": gpu_memcpy_avg_us,
            "gpu_memcpy_pct": pct(gpu_memcpy_avg_us, duration_avg_us),
            "gpu_any_avg_us": gpu_any_avg_us,
            "gpu_any_pct": pct(gpu_any_avg_us, duration_avg_us),
            "gpu_idle_avg_us": gpu_idle_avg_us,
            "gpu_idle_pct": pct(gpu_idle_avg_us, duration_avg_us),
        }

    return summaries


def build_threaded_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows = [compute_threaded_overlap(report) for report in reports]
    global_rows = [compute_threaded_global_metrics(report) for report in reports]

    csv_path = output_dir / "cupy_threaded_overlap.csv"
    write_csv(csv_path, rows, fieldnames=THREADED_CSV_FIELDS)
    global_csv_path = output_dir / "cupy_threaded_global_metrics.csv"
    write_csv(
        global_csv_path,
        global_rows,
        fieldnames=THREADED_GLOBAL_CSV_FIELDS,
    )

    written = [csv_path, global_csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]
        platform_global_rows = [
            row
            for row in global_rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"cupy_threaded_overlap_{platform}.tex"
        latex_path.write_text(
            render_threaded_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            ),
            encoding="utf-8",
        )
        written.append(latex_path)

        global_latex_path = latex_dir / f"cupy_threaded_global_metrics_{platform}.tex"
        global_latex_path.write_text(
            render_threaded_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_global_rows, platform_reports),
            ),
            encoding="utf-8",
        )
        written.append(global_latex_path)

    return written


def build_streams_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows = [compute_streams_global_metrics(report) for report in reports]

    csv_path = output_dir / "cupy_streams_global_metrics.csv"
    write_csv(csv_path, rows, fieldnames=STREAMS_GLOBAL_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"cupy_streams_global_metrics_{platform}.tex"
        latex_path.write_text(
            render_streams_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            ),
            encoding="utf-8",
        )
        written.append(latex_path)

    return written


def build_pytorch_naive_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    failures_by_platform: dict[str, list[str]] = defaultdict(list)

    for report in reports:
        try:
            rows.append(compute_pytorch_naive_global_metrics(report))
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            failures_by_platform[report.platform].append(
                f"{report.report_path.name}: {exc}"
            )

    csv_path = output_dir / "pytorch_naive_global_metrics.csv"
    write_csv(csv_path, rows, fieldnames=PYTORCH_GLOBAL_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"pytorch_naive_global_metrics_{platform}.tex"
        if platform_rows:
            latex = render_pytorch_naive_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            )
        else:
            latex = render_no_data_latex(
                platform=platform,
                benchmark_label="PyTorch naive",
                label_slug="pytorch-naive-global-metrics",
                reason=pytorch_no_data_reason(platform, failures_by_platform),
            )

        latex_path.write_text(latex, encoding="utf-8")
        written.append(latex_path)

    return written


def build_pytorch_threaded_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    failures_by_platform: dict[str, list[str]] = defaultdict(list)

    for report in reports:
        try:
            rows.append(compute_pytorch_threaded_global_metrics(report))
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            failures_by_platform[report.platform].append(
                f"{report.report_path.name}: {exc}"
            )

    csv_path = output_dir / "pytorch_threaded_global_metrics.csv"
    write_csv(csv_path, rows, fieldnames=PYTORCH_GLOBAL_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"pytorch_threaded_global_metrics_{platform}.tex"
        if platform_rows:
            latex = render_pytorch_threaded_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            )
        else:
            latex = render_no_data_latex(
                platform=platform,
                benchmark_label="PyTorch threaded",
                label_slug="pytorch-threaded-global-metrics",
                reason=pytorch_threaded_no_data_reason(
                    platform,
                    failures_by_platform,
                ),
            )

        latex_path.write_text(latex, encoding="utf-8")
        written.append(latex_path)

    return written


def build_pytorch_streams_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    failures_by_platform: dict[str, list[str]] = defaultdict(list)

    for report in reports:
        try:
            rows.append(compute_pytorch_streams_global_metrics(report))
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            failures_by_platform[report.platform].append(
                f"{report.report_path.name}: {exc}"
            )

    csv_path = output_dir / "pytorch_streams_global_metrics.csv"
    write_csv(csv_path, rows, fieldnames=PYTORCH_GLOBAL_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"pytorch_streams_global_metrics_{platform}.tex"
        if platform_rows:
            latex = render_pytorch_streams_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            )
        else:
            latex = render_no_data_latex(
                platform=platform,
                benchmark_label="PyTorch streams",
                label_slug="pytorch-streams-global-metrics",
                reason=pytorch_streams_no_data_reason(
                    platform,
                    failures_by_platform,
                ),
            )

        latex_path.write_text(latex, encoding="utf-8")
        written.append(latex_path)

    return written


def build_jax_naive_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    failures_by_platform: dict[str, list[str]] = defaultdict(list)

    for report in reports:
        try:
            rows.append(compute_jax_naive_global_metrics(report))
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            failures_by_platform[report.platform].append(
                f"{report.report_path.name}: {exc}"
            )

    csv_path = output_dir / "jax_naive_global_metrics.csv"
    write_csv(csv_path, rows, fieldnames=JAX_GLOBAL_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"jax_naive_global_metrics_{platform}.tex"
        if platform_rows:
            latex = render_jax_naive_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            )
        else:
            latex = render_no_data_latex(
                platform=platform,
                benchmark_label="JAX naive",
                label_slug="jax-naive-global-metrics",
                reason=jax_naive_no_data_reason(platform, failures_by_platform),
            )

        latex_path.write_text(latex, encoding="utf-8")
        written.append(latex_path)

    return written


def build_jax_streams_outputs(
    reports: list[ReportContext],
    *,
    output_dir: Path,
    latex_dir: Path,
) -> list[Path]:
    rows: list[dict[str, Any]] = []
    failures_by_platform: dict[str, list[str]] = defaultdict(list)

    for report in reports:
        try:
            rows.append(compute_jax_streams_global_metrics(report))
        except (RuntimeError, sqlite3.DatabaseError) as exc:
            failures_by_platform[report.platform].append(
                f"{report.report_path.name}: {exc}"
            )

    csv_path = output_dir / "jax_streams_global_metrics.csv"
    write_csv(csv_path, rows, fieldnames=JAX_GLOBAL_CSV_FIELDS)

    written = [csv_path]
    for platform, platform_reports in group_reports_by_platform(reports).items():
        platform_rows = [
            row
            for row in rows
            if row["platform"] == platform
        ]

        latex_path = latex_dir / f"jax_streams_global_metrics_{platform}.tex"
        if platform_rows:
            latex = render_jax_streams_global_latex(
                platform=platform,
                rows=sort_rows_like_reports(platform_rows, platform_reports),
            )
        else:
            latex = render_no_data_latex(
                platform=platform,
                benchmark_label="JAX streams",
                label_slug="jax-streams-global-metrics",
                reason=jax_streams_no_data_reason(platform, failures_by_platform),
            )

        latex_path.write_text(latex, encoding="utf-8")
        written.append(latex_path)

    return written


def compute_threaded_overlap(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    simple_stages = {
        stage: [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == nvtx_name
        ]
        for stage, nvtx_name in THREAD_STAGE_NAMES.items()
    }
    copy_ranges = {
        stage: [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == names["copy"]
        ]
        for stage, names in THREAD_COPY_STAGES.items()
    }
    sync_ranges = {
        stage: [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == names["sync"]
        ]
        for stage, names in THREAD_COPY_STAGES.items()
    }

    missing = [
        nvtx_name
        for stage, nvtx_name in THREAD_STAGE_NAMES.items()
        if not simple_stages[stage]
    ]
    for stage, names in THREAD_COPY_STAGES.items():
        if not copy_ranges[stage]:
            missing.append(names["copy"])
        if not sync_ranges[stage]:
            missing.append(names["sync"])

    if missing:
        raise RuntimeError(
            f"{report.sqlite_path} is missing NVTX ranges: {', '.join(missing)}"
        )

    pairs = {
        stage: pair_copy_with_following_sync(
            copy_ranges[stage],
            sync_ranges[stage],
            copy_name=names["copy"],
            sync_name=names["sync"],
            report=report,
        )
        for stage, names in THREAD_COPY_STAGES.items()
    }
    virtual_copy_ranges = {
        stage: [pair.virtual_interval for pair in items]
        for stage, items in pairs.items()
    }

    merged = {
        stage: merge_intervals(intervals)
        for stage, intervals in {
            **simple_stages,
            **virtual_copy_ranges,
        }.items()
    }
    h2d_post_overlap_us = ns_to_us(
        intersection_length_ns(merged["h2d"], merged["postprocess"])
    )
    d2h_post_overlap_us = ns_to_us(
        intersection_length_ns(merged["d2h"], merged["postprocess"])
    )
    memcpy_merged = merge_intervals(merged["h2d"] + merged["d2h"])
    memcpy_post_overlap_us = ns_to_us(
        intersection_length_ns(memcpy_merged, merged["postprocess"])
    )
    h2d_d2h_overlap_us = ns_to_us(
        intersection_length_ns(merged["h2d"], merged["d2h"])
    )

    h2d_total_us = ns_to_us(interval_length_ns(merged["h2d"]))
    d2h_total_us = ns_to_us(interval_length_ns(merged["d2h"]))
    post_total_us = ns_to_us(interval_length_ns(merged["postprocess"]))
    memcpy_union_total_us = ns_to_us(interval_length_ns(memcpy_merged))
    memcpy_work_total_us = h2d_total_us + d2h_total_us
    memcpy_work_hidden_us = h2d_post_overlap_us + d2h_post_overlap_us

    return {
        **report_metadata_fields(report),
        "h2d_count": len(pairs["h2d"]),
        "h2d_copy_count": len(copy_ranges["h2d"]),
        "h2d_sync_count": len(sync_ranges["h2d"]),
        "fft_count": len(simple_stages["fft"]),
        "postprocess_count": len(simple_stages["postprocess"]),
        "d2h_count": len(pairs["d2h"]),
        "d2h_copy_count": len(copy_ranges["d2h"]),
        "d2h_sync_count": len(sync_ranges["d2h"]),
        "h2d_copy_avg_us": mean_interval_us(copy_ranges["h2d"]),
        "h2d_sync_avg_us": mean_interval_us([pair.sync_interval for pair in pairs["h2d"]]),
        "h2d_gap_avg_us": mean_ns(pair.gap_ns for pair in pairs["h2d"]) / 1e3,
        "h2d_gap_sync_avg_us": mean_ns(pair.gap_sync_ns for pair in pairs["h2d"]) / 1e3,
        "h2d_avg_us": mean_interval_us(virtual_copy_ranges["h2d"]),
        "fft_avg_us": mean_interval_us(simple_stages["fft"]),
        "postprocess_avg_us": mean_interval_us(simple_stages["postprocess"]),
        "d2h_copy_avg_us": mean_interval_us(copy_ranges["d2h"]),
        "d2h_sync_avg_us": mean_interval_us([pair.sync_interval for pair in pairs["d2h"]]),
        "d2h_gap_avg_us": mean_ns(pair.gap_ns for pair in pairs["d2h"]) / 1e3,
        "d2h_gap_sync_avg_us": mean_ns(pair.gap_sync_ns for pair in pairs["d2h"]) / 1e3,
        "d2h_avg_us": mean_interval_us(virtual_copy_ranges["d2h"]),
        "h2d_copy_total_us": ns_to_us(interval_length_ns(merge_intervals(copy_ranges["h2d"]))),
        "h2d_sync_total_us": ns_to_us(
            interval_length_ns(merge_intervals([pair.sync_interval for pair in pairs["h2d"]]))
        ),
        "h2d_gap_total_us": ns_to_us(sum(pair.gap_ns for pair in pairs["h2d"])),
        "h2d_total_us": h2d_total_us,
        "fft_total_us": ns_to_us(interval_length_ns(merged["fft"])),
        "postprocess_total_us": post_total_us,
        "d2h_copy_total_us": ns_to_us(interval_length_ns(merge_intervals(copy_ranges["d2h"]))),
        "d2h_sync_total_us": ns_to_us(
            interval_length_ns(merge_intervals([pair.sync_interval for pair in pairs["d2h"]]))
        ),
        "d2h_gap_total_us": ns_to_us(sum(pair.gap_ns for pair in pairs["d2h"])),
        "d2h_total_us": d2h_total_us,
        "h2d_postprocess_overlap_us": h2d_post_overlap_us,
        "d2h_postprocess_overlap_us": d2h_post_overlap_us,
        "memcpy_postprocess_overlap_us": memcpy_post_overlap_us,
        "h2d_d2h_overlap_us": h2d_d2h_overlap_us,
        "h2d_hidden_by_postprocess_pct": pct(h2d_post_overlap_us, h2d_total_us),
        "d2h_hidden_by_postprocess_pct": pct(d2h_post_overlap_us, d2h_total_us),
        "memcpy_work_hidden_by_postprocess_pct": pct(
            memcpy_work_hidden_us,
            memcpy_work_total_us,
        ),
        "postprocess_covered_by_memcpy_pct": pct(
            memcpy_post_overlap_us,
            post_total_us,
        ),
        "h2d_d2h_overlap_pct": pct(
            h2d_d2h_overlap_us,
            memcpy_union_total_us,
        ),
    }


def compute_threaded_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    simple_stages = {
        stage: [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == nvtx_name
        ]
        for stage, nvtx_name in THREAD_STAGE_NAMES.items()
    }
    copy_ranges = {
        stage: [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == names["copy"]
        ]
        for stage, names in THREAD_COPY_STAGES.items()
    }
    sync_ranges = {
        stage: [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == names["sync"]
        ]
        for stage, names in THREAD_COPY_STAGES.items()
    }

    for stage, intervals in simple_stages.items():
        if not intervals:
            raise RuntimeError(
                f"{report.sqlite_path} is missing {THREAD_STAGE_NAMES[stage]!r}."
            )

    pairs = {
        stage: pair_copy_with_following_sync(
            copy_ranges[stage],
            sync_ranges[stage],
            copy_name=names["copy"],
            sync_name=names["sync"],
            report=report,
        )
        for stage, names in THREAD_COPY_STAGES.items()
    }
    stage_intervals = {
        "h2d": [pair.virtual_interval for pair in pairs["h2d"]],
        "fft": simple_stages["fft"],
        "postprocess": simple_stages["postprocess"],
        "d2h": [pair.virtual_interval for pair in pairs["d2h"]],
    }

    window_start_ns = max(min(start for start, _end in intervals) for intervals in stage_intervals.values())
    window_end_ns = min(max(end for _start, end in intervals) for intervals in stage_intervals.values())
    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no shared steady-state window across "
            "H2D, FFT, postprocess, and D2H ranges."
        )

    reference_iteration_stage = "postprocess"
    reference_iterations = count_complete_intervals(
        stage_intervals[reference_iteration_stage],
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage} "
            "ranges in the shared steady-state window."
        )

    cuda_api_intervals = read_timed_intervals(
        report.sqlite_path,
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        group_column="globalTid",
    )
    kernel_intervals = read_timed_intervals(
        report.sqlite_path,
        "CUPTI_ACTIVITY_KIND_KERNEL",
    )
    memcpy_intervals = read_timed_intervals(
        report.sqlite_path,
        "CUPTI_ACTIVITY_KIND_MEMCPY",
    )

    duration_ns = window_end_ns - window_start_ns
    duration_us = ns_to_us(duration_ns)
    api_raw = clip_intervals(cuda_api_intervals, window_start_ns, window_end_ns)
    kernel_raw = clip_intervals(kernel_intervals, window_start_ns, window_end_ns)
    memcpy_raw = clip_intervals(memcpy_intervals, window_start_ns, window_end_ns)

    api_merged = merge_intervals([(start, end) for start, end, _group in api_raw])
    kernel_merged = merge_intervals([(start, end) for start, end, _group in kernel_raw])
    memcpy_merged = merge_intervals([(start, end) for start, end, _group in memcpy_raw])

    cuda_api_non_gap_us = ns_to_us(interval_length_ns(api_merged))
    cuda_api_gap_us = max(0.0, duration_us - cuda_api_non_gap_us)
    gpu_compute_us = ns_to_us(interval_length_ns(kernel_merged))
    gpu_memcpy_us = ns_to_us(interval_length_ns(memcpy_merged))
    gpu_any_us = ns_to_us(interval_length_ns(merge_intervals(kernel_merged + memcpy_merged)))
    gpu_idle_us = max(0.0, duration_us - gpu_any_us)

    origin_ns = min(item.start_ns for item in ranges)
    per_iteration = float(reference_iterations)

    return {
        **report_metadata_fields(report),
        "scope": "global_steady_state",
        "count": reference_iterations,
        "reference_iteration_stage": reference_iteration_stage,
        "reference_iterations": reference_iterations,
        "window_start_ms": (window_start_ns - origin_ns) / 1e6,
        "window_end_ms": (window_end_ns - origin_ns) / 1e6,
        "window_duration_us": duration_us,
        "duration_avg_us": duration_us / per_iteration,
        "cuda_api_calls_avg": len(api_raw) / per_iteration,
        "cuda_api_threads": len(
            {group for _start, _end, group in api_raw if group is not None}
        ),
        "cuda_api_non_gap_avg_us": cuda_api_non_gap_us / per_iteration,
        "cuda_api_non_gap_pct": pct(cuda_api_non_gap_us, duration_us),
        "cuda_api_gap_avg_us": cuda_api_gap_us / per_iteration,
        "cuda_api_gap_pct": pct(cuda_api_gap_us, duration_us),
        "kernel_calls_avg": len(kernel_raw) / per_iteration,
        "memcpy_calls_avg": len(memcpy_raw) / per_iteration,
        "gpu_compute_avg_us": gpu_compute_us / per_iteration,
        "gpu_compute_pct": pct(gpu_compute_us, duration_us),
        "gpu_memcpy_avg_us": gpu_memcpy_us / per_iteration,
        "gpu_memcpy_pct": pct(gpu_memcpy_us, duration_us),
        "gpu_any_avg_us": gpu_any_us / per_iteration,
        "gpu_any_pct": pct(gpu_any_us, duration_us),
        "gpu_idle_avg_us": gpu_idle_us / per_iteration,
        "gpu_idle_pct": pct(gpu_idle_us, duration_us),
    }


def compute_streams_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    tick_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if is_stream_tick_range(item.name)
    ]

    reference_iteration_stage = "process_batch_device"
    reference_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if item.name == reference_iteration_stage
    ]
    if not reference_intervals:
        reference_iteration_stage = "finalize_output"
        reference_intervals = [
            (item.start_ns, item.end_ns)
            for item in ranges
            if item.name == reference_iteration_stage
        ]

    if not reference_intervals:
        raise RuntimeError(
            f"{report.sqlite_path} has no process_batch_device or "
            "finalize_output NVTX ranges."
        )

    reference_intervals = largest_interval_cluster(reference_intervals)
    tick_intervals = largest_interval_cluster(tick_intervals)

    reference_start_ns = min(start for start, _end in reference_intervals)
    reference_end_ns = max(end for _start, end in reference_intervals)
    nearby_tick_intervals = [
        (start, end)
        for start, end in tick_intervals
        if start <= reference_end_ns + MAX_COPY_SYNC_GAP_NS
        and end >= reference_start_ns - MAX_COPY_SYNC_GAP_NS
    ]

    window_source = "streams_tick"
    if nearby_tick_intervals:
        window_start_ns = min(start for start, _end in nearby_tick_intervals)
        window_end_ns = max(end for _start, end in nearby_tick_intervals)
    else:
        window_source = reference_iteration_stage
        window_start_ns = reference_start_ns
        window_end_ns = reference_end_ns

    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no valid CuPy streams steady-state window."
        )

    reference_iterations = count_complete_intervals(
        reference_intervals,
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage} "
            "ranges in the selected streams window."
        )

    row = compute_global_window_metrics(
        report=report,
        ranges=ranges,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        reference_iteration_stage=reference_iteration_stage,
        reference_iterations=reference_iterations,
    )
    row["window_source"] = window_source
    return row


def compute_pytorch_naive_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    if not ranges:
        raise RuntimeError("no completed NVTX ranges are present")

    reference_iteration_stage = "pytorch process_batch"
    reference_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if item.name == reference_iteration_stage
    ]
    if not reference_intervals:
        raise RuntimeError(
            f"no completed {reference_iteration_stage!r} NVTX ranges are present"
        )

    origin_ns = min(item.start_ns for item in ranges)
    window_start_ns = origin_ns + PYTORCH_NAIVE_SKIP_INITIAL_NS
    reference_intervals = largest_interval_cluster(reference_intervals)
    window_reference_intervals = [
        (start, end)
        for start, end in reference_intervals
        if end > window_start_ns
    ]
    if not window_reference_intervals:
        raise RuntimeError(
            f"no {reference_iteration_stage!r} ranges remain after the 1s "
            "initialization cutoff"
        )

    window_end_ns = max(end for _start, end in window_reference_intervals)
    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no valid PyTorch naive steady-state window."
        )

    reference_iterations = count_complete_intervals(
        window_reference_intervals,
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage!r} "
            "ranges in the selected PyTorch naive window."
        )

    row = compute_global_window_metrics(
        report=report,
        ranges=ranges,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        reference_iteration_stage=reference_iteration_stage,
        reference_iterations=reference_iterations,
    )
    row["window_source"] = "trace_origin_plus_1s_to_pytorch_process_batch_cluster"
    return row


def compute_pytorch_threaded_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    if not ranges:
        raise RuntimeError("no completed NVTX ranges are present")

    reference_iteration_stage = "pytorch-threaded FFT batch"
    reference_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if item.name == reference_iteration_stage
    ]
    if not reference_intervals:
        raise RuntimeError(
            f"no completed {reference_iteration_stage!r} NVTX ranges are present"
        )

    origin_ns = min(item.start_ns for item in ranges)
    window_start_ns = origin_ns + PYTORCH_THREADED_SKIP_INITIAL_NS
    reference_intervals = largest_interval_cluster(reference_intervals)
    window_reference_intervals = [
        (start, end)
        for start, end in reference_intervals
        if end > window_start_ns
    ]
    if not window_reference_intervals:
        raise RuntimeError(
            f"no {reference_iteration_stage!r} ranges remain after the 2s "
            "initialization cutoff"
        )

    window_end_ns = max(end for _start, end in window_reference_intervals)
    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no valid PyTorch threaded steady-state window."
        )

    reference_iterations = count_complete_intervals(
        window_reference_intervals,
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage!r} "
            "ranges in the selected PyTorch threaded window."
        )

    row = compute_global_window_metrics(
        report=report,
        ranges=ranges,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        reference_iteration_stage=reference_iteration_stage,
        reference_iterations=reference_iterations,
    )
    row["window_source"] = "trace_origin_plus_2s_to_pytorch_threaded_fft_batch_cluster"
    return row


def compute_pytorch_streams_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    if not ranges:
        raise RuntimeError("no completed NVTX ranges are present")

    reference_iteration_stage = "pytorch process_batch_device"
    reference_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if item.name == reference_iteration_stage
    ]
    if not reference_intervals:
        raise RuntimeError(
            f"no completed {reference_iteration_stage!r} NVTX ranges are present"
        )

    origin_ns = min(item.start_ns for item in ranges)
    window_start_ns = origin_ns + PYTORCH_STREAMS_SKIP_INITIAL_NS
    reference_intervals = largest_interval_cluster(reference_intervals)
    window_reference_intervals = [
        (start, end)
        for start, end in reference_intervals
        if end > window_start_ns
    ]
    if not window_reference_intervals:
        raise RuntimeError(
            f"no {reference_iteration_stage!r} ranges remain after the 1.5s "
            "initialization cutoff"
        )

    window_end_ns = max(end for _start, end in window_reference_intervals)
    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no valid PyTorch streams steady-state window."
        )

    reference_iterations = count_complete_intervals(
        window_reference_intervals,
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage!r} "
            "ranges in the selected PyTorch streams window."
        )

    row = compute_global_window_metrics(
        report=report,
        ranges=ranges,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        reference_iteration_stage=reference_iteration_stage,
        reference_iterations=reference_iterations,
    )
    row["window_source"] = "trace_origin_plus_1p5s_to_pytorch_process_batch_device_cluster"
    return row


def compute_jax_naive_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    if not ranges:
        raise RuntimeError("no completed NVTX ranges are present")

    reference_iteration_stage = JAX_NAIVE_REFERENCE_NAME
    reference_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if is_jax_process_ready_output_core_range(item.name)
    ]
    if not reference_intervals:
        raise RuntimeError(
            f"no completed {reference_iteration_stage!r} NVTX ranges are present"
        )

    reference_intervals = largest_interval_cluster(reference_intervals)
    window_start_ns = min(start for start, _end in reference_intervals)
    window_end_ns = max(end for _start, end in reference_intervals)
    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no valid JAX naive steady-state window."
        )

    reference_iterations = count_complete_intervals(
        reference_intervals,
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage!r} "
            "ranges in the selected JAX naive window."
        )

    row = compute_global_window_metrics(
        report=report,
        ranges=ranges,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        reference_iteration_stage=reference_iteration_stage,
        reference_iterations=reference_iterations,
    )
    row["window_source"] = "jax_xla_process_ready_output_core_cluster"
    return row


def compute_jax_streams_global_metrics(report: ReportContext) -> dict[str, Any]:
    ranges = read_nvtx_ranges(report.sqlite_path)
    if not ranges:
        raise RuntimeError("no completed NVTX ranges are present")

    reference_iteration_stage = JAX_NAIVE_REFERENCE_NAME
    reference_intervals = [
        (item.start_ns, item.end_ns)
        for item in ranges
        if is_jax_process_ready_output_core_range(item.name)
    ]
    if not reference_intervals:
        raise RuntimeError(
            f"no completed {reference_iteration_stage!r} NVTX ranges are present"
        )

    reference_intervals = largest_interval_cluster(reference_intervals)
    window_start_ns = min(start for start, _end in reference_intervals)
    window_end_ns = max(end for _start, end in reference_intervals)
    if window_end_ns <= window_start_ns:
        raise RuntimeError(
            f"{report.sqlite_path} has no valid JAX streams steady-state window."
        )

    reference_iterations = count_complete_intervals(
        reference_intervals,
        window_start_ns,
        window_end_ns,
    )
    if reference_iterations <= 0:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {reference_iteration_stage!r} "
            "ranges in the selected JAX streams window."
        )

    row = compute_global_window_metrics(
        report=report,
        ranges=ranges,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        reference_iteration_stage=reference_iteration_stage,
        reference_iterations=reference_iterations,
    )
    row["window_source"] = "jax_streams_xla_process_ready_output_core_cluster"
    return row


def compute_global_window_metrics(
    *,
    report: ReportContext,
    ranges: list[NvtxRange],
    window_start_ns: int,
    window_end_ns: int,
    reference_iteration_stage: str,
    reference_iterations: int,
) -> dict[str, Any]:
    cuda_api_intervals = read_timed_intervals(
        report.sqlite_path,
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        group_column="globalTid",
    )
    kernel_intervals = read_timed_intervals(
        report.sqlite_path,
        "CUPTI_ACTIVITY_KIND_KERNEL",
    )
    memcpy_intervals = read_timed_intervals(
        report.sqlite_path,
        "CUPTI_ACTIVITY_KIND_MEMCPY",
    )

    duration_ns = window_end_ns - window_start_ns
    duration_us = ns_to_us(duration_ns)
    api_raw = clip_intervals(cuda_api_intervals, window_start_ns, window_end_ns)
    kernel_raw = clip_intervals(kernel_intervals, window_start_ns, window_end_ns)
    memcpy_raw = clip_intervals(memcpy_intervals, window_start_ns, window_end_ns)

    api_merged = merge_intervals([(start, end) for start, end, _group in api_raw])
    kernel_merged = merge_intervals([(start, end) for start, end, _group in kernel_raw])
    memcpy_merged = merge_intervals([(start, end) for start, end, _group in memcpy_raw])

    cuda_api_non_gap_us = ns_to_us(interval_length_ns(api_merged))
    cuda_api_gap_us = max(0.0, duration_us - cuda_api_non_gap_us)
    gpu_compute_us = ns_to_us(interval_length_ns(kernel_merged))
    gpu_memcpy_us = ns_to_us(interval_length_ns(memcpy_merged))
    gpu_any_us = ns_to_us(
        interval_length_ns(merge_intervals(kernel_merged + memcpy_merged))
    )
    gpu_idle_us = max(0.0, duration_us - gpu_any_us)

    origin_ns = min(item.start_ns for item in ranges)
    per_iteration = float(reference_iterations)

    return {
        **report_metadata_fields(report),
        "scope": "global_steady_state",
        "count": reference_iterations,
        "reference_iteration_stage": reference_iteration_stage,
        "reference_iterations": reference_iterations,
        "window_start_ms": (window_start_ns - origin_ns) / 1e6,
        "window_end_ms": (window_end_ns - origin_ns) / 1e6,
        "window_duration_us": duration_us,
        "duration_avg_us": duration_us / per_iteration,
        "cuda_api_calls_avg": len(api_raw) / per_iteration,
        "cuda_api_threads": len(
            {group for _start, _end, group in api_raw if group is not None}
        ),
        "cuda_api_non_gap_avg_us": cuda_api_non_gap_us / per_iteration,
        "cuda_api_non_gap_pct": pct(cuda_api_non_gap_us, duration_us),
        "cuda_api_gap_avg_us": cuda_api_gap_us / per_iteration,
        "cuda_api_gap_pct": pct(cuda_api_gap_us, duration_us),
        "kernel_calls_avg": len(kernel_raw) / per_iteration,
        "memcpy_calls_avg": len(memcpy_raw) / per_iteration,
        "gpu_compute_avg_us": gpu_compute_us / per_iteration,
        "gpu_compute_pct": pct(gpu_compute_us, duration_us),
        "gpu_memcpy_avg_us": gpu_memcpy_us / per_iteration,
        "gpu_memcpy_pct": pct(gpu_memcpy_us, duration_us),
        "gpu_any_avg_us": gpu_any_us / per_iteration,
        "gpu_any_pct": pct(gpu_any_us, duration_us),
        "gpu_idle_avg_us": gpu_idle_us / per_iteration,
        "gpu_idle_pct": pct(gpu_idle_us, duration_us),
    }


def pair_copy_with_following_sync(
    copy_ranges: list[tuple[int, int]],
    sync_ranges: list[tuple[int, int]],
    *,
    copy_name: str,
    sync_name: str,
    report: ReportContext,
) -> list[CopySyncPair]:
    pairs: list[CopySyncPair] = []
    ordered_copies = sorted(copy_ranges)
    ordered_syncs = sorted(sync_ranges)
    sync_index = 0

    for copy_start, copy_end in ordered_copies:
        while (
            sync_index < len(ordered_syncs)
            and (
                ordered_syncs[sync_index][0] < copy_start
                or ordered_syncs[sync_index][1] <= copy_end
            )
        ):
            sync_index += 1

        if sync_index >= len(ordered_syncs):
            break

        sync_start, sync_end = ordered_syncs[sync_index]
        if sync_start - copy_end > MAX_COPY_SYNC_GAP_NS:
            break

        pairs.append(
            CopySyncPair(
                copy_start_ns=copy_start,
                copy_end_ns=copy_end,
                sync_start_ns=sync_start,
                sync_end_ns=sync_end,
            )
        )
        sync_index += 1

    if not pairs:
        raise RuntimeError(
            f"{report.sqlite_path} has no complete {copy_name!r} -> "
            f"{sync_name!r} virtual ranges."
        )

    return pairs


def read_nvtx_ranges(sqlite_path: Path) -> list[NvtxRange]:
    with sqlite3.connect(sqlite_path) as con:
        con.row_factory = sqlite3.Row
        tables = read_sqlite_tables(con)

        if "NVTX_EVENTS" not in tables:
            raise RuntimeError(f"{sqlite_path} has no NVTX_EVENTS table.")

        nvtx_cols = read_sqlite_columns(con, "NVTX_EVENTS")
        if not {"start", "end"} <= nvtx_cols:
            raise RuntimeError(f"{sqlite_path} NVTX_EVENTS has no start/end columns.")

        has_strings = "StringIds" in tables
        string_cols = read_sqlite_columns(con, "StringIds") if has_strings else set()

        name_parts: list[str] = []
        if "text" in nvtx_cols:
            name_parts.append("n.text")

        join = ""
        if "textId" in nvtx_cols and has_strings and {"id", "value"} <= string_cols:
            name_parts.append("s.value")
            join = "LEFT JOIN StringIds s ON s.id = n.textId"

        name_expr = (
            "'<unnamed>'"
            if not name_parts
            else "COALESCE(" + ", ".join(name_parts) + ", '<unnamed>')"
        )

        rows = con.execute(
            f"""
            SELECT
                {name_expr} AS name,
                n.start AS start_ns,
                n.[end] AS end_ns
            FROM NVTX_EVENTS n
            {join}
            WHERE n.[end] IS NOT NULL
              AND n.[end] > n.start
            ORDER BY n.start ASC, n.[end] ASC
            """
        ).fetchall()

    return [
        NvtxRange(
            name=str(row["name"]),
            start_ns=int(row["start_ns"]),
            end_ns=int(row["end_ns"]),
        )
        for row in rows
    ]


def read_timed_intervals(
    sqlite_path: Path,
    table: str,
    *,
    group_column: str | None = None,
) -> list[tuple[int, int, int | None]]:
    with sqlite3.connect(sqlite_path) as con:
        con.row_factory = sqlite3.Row
        if table not in read_sqlite_tables(con):
            return []

        cols = read_sqlite_columns(con, table)
        if not {"start", "end"} <= cols:
            return []

        group_expr = (
            group_column
            if group_column is not None and group_column in cols
            else "NULL"
        )

        rows = con.execute(
            f"""
            SELECT
                start AS start_ns,
                [end] AS end_ns,
                {group_expr} AS group_id
            FROM "{table}"
            WHERE [end] IS NOT NULL
              AND [end] > start
            ORDER BY start ASC, [end] ASC
            """
        ).fetchall()

    return [
        (
            int(row["start_ns"]),
            int(row["end_ns"]),
            None if row["group_id"] is None else int(row["group_id"]),
        )
        for row in rows
    ]


def read_sqlite_tables(con: sqlite3.Connection) -> set[str]:
    rows = con.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    return {str(row["name"]) for row in rows}


def read_sqlite_columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row["name"]) for row in rows}


def is_stream_tick_range(name: str) -> bool:
    return re.fullmatch(
        r"streams tick \d+: H2D batch \d+; compute batch (?:\d+|-); D2H batch (?:\d+|-)",
        name,
    ) is not None


def is_jax_process_ready_output_core_range(name: str) -> bool:
    return name.rstrip("#") == JAX_NAIVE_REFERENCE_NAME


def render_naive_latex(
    *,
    platform: str,
    reports: list[ReportContext],
    summaries_by_report: dict[Path, dict[str, dict[str, Any]]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    reports = sorted(reports, key=report_sort_key)
    midpoint = (len(reports) + 1) // 2
    columns = [column for column in (reports[:midpoint], reports[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Average timing and GPU activity metrics for the CuPy "
            f"naive implementation on {title_platform} under all benchmark "
            "configurations. P denotes precompute, A denotes preallocation, "
            "and G denotes artificial Python GIL contention. Value 0 indicates "
            "disabled, value 1 indicates enabled. }"
        ),
        f"  \\label{{tab:cupy-naive-scope-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_reports in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for report_index, report in enumerate(column_reports):
            if report_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(
                render_naive_mode_table(
                    report=report,
                    summary=summaries_by_report[report.report_path],
                )
            )

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_naive_mode_table(
    *,
    report: ReportContext,
    summary: dict[str, dict[str, Any]],
) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(report.mode_label)}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.34\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.17\\linewidth}"
        ),
        (
            "        >{\\raggedleft\\arraybackslash}p{0.20\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.20\\linewidth}"
        ),
        "      }",
        "      \\hline",
        (
            "      \\textbf{Metric}                       & \\textbf{Total} "
            "& \\textbf{Process batch} & \\textbf{Export display} \\\\"
        ),
        "      \\hline",
    ]

    for label, key, kind in NAIVE_TABLE_ROWS:
        values = []
        for scope, _header in NAIVE_TABLE_SCOPES:
            values.append(format_latex_number(summary.get(scope, {}).get(key), kind))

        lines.append(
            f"      {label:<39} & {values[0]:<14} & {values[1]:<22} & {values[2]:<22} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_threaded_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ CuPy threaded NVTX timing and copy-thread overlap on "
            f"{title_platform}. H2D and D2H virtual ranges start at the copy "
            "range and end at the following sync-before-enqueue range, including "
            "the intermediate gap. Hidden percentages measure how much of those "
            "virtual ranges overlaps threaded postprocess batch. }"
        ),
        f"  \\label{{tab:cupy-threaded-overlap-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_threaded_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_threaded_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Value} \\\\",
        "      \\hline",
    ]

    for label, key, kind in THREADED_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_threaded_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the CuPy threaded implementation on {title_platform}. The "
            "window is the shared wall-clock interval where H2D, FFT, "
            "postprocess, and D2H are all active. CUDA API non-gap is the union "
            "of CUDA runtime API intervals across all submitting threads; CUDA "
            "API gap is the remaining wall-clock time. Microsecond values are "
            "divided by the number of complete postprocess iterations sampled "
            "in that window. }"
        ),
        f"  \\label{{tab:cupy-threaded-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_threaded_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_threaded_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in THREADED_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_streams_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the CuPy streams implementation on {title_platform}. The "
            "window is selected from streams tick ranges when available, "
            "covering the single host thread's H2D, compute, and D2H "
            "submissions. CUDA API non-gap is the union of CUDA runtime API "
            "intervals in that window; CUDA API gap is the remaining wall-clock "
            "time. Microsecond values are divided by the number of complete "
            "process batch device iterations sampled in that window. }"
        ),
        f"  \\label{{tab:cupy-streams-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_streams_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_streams_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in STREAMS_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_pytorch_naive_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the PyTorch naive implementation on {title_platform}. The "
            "window starts 1 second after the first completed NVTX range to "
            "skip initialization, and ends at the end of the clustered "
            "\\texttt{pytorch process\\_batch} ranges. CUDA API non-gap is "
            "the union of CUDA runtime API intervals in that window; CUDA API "
            "gap is the remaining wall-clock time. Microsecond values are "
            "divided by the number of complete \\texttt{pytorch "
            "process\\_batch} iterations sampled in that window. }"
        ),
        f"  \\label{{tab:pytorch-naive-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_pytorch_naive_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_pytorch_naive_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in PYTORCH_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_pytorch_threaded_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the PyTorch threaded implementation on {title_platform}. "
            "The window starts 2 seconds after the first completed NVTX range "
            "to skip initialization, and ends at the end of the clustered "
            "\\texttt{pytorch-threaded FFT batch} ranges. CUDA API non-gap is "
            "the union of CUDA runtime API intervals across all submitting "
            "threads in that window; CUDA API gap is the remaining wall-clock "
            "time. Microsecond values are divided by the number of complete "
            "\\texttt{pytorch-threaded FFT batch} iterations sampled in that "
            "window. }"
        ),
        f"  \\label{{tab:pytorch-threaded-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_pytorch_threaded_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_pytorch_threaded_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in PYTORCH_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_pytorch_streams_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the PyTorch streams implementation on {title_platform}. "
            "The window starts 1.5 seconds after the first completed NVTX "
            "range to skip initialization, and ends at the end of the "
            "clustered \\texttt{pytorch process\\_batch\\_device} ranges. "
            "CUDA API non-gap is the union of CUDA runtime API intervals in "
            "that window; CUDA API gap is the remaining wall-clock time. "
            "Microsecond values are divided by the number of complete "
            "\\texttt{pytorch process\\_batch\\_device} iterations sampled in "
            "that window. }"
        ),
        f"  \\label{{tab:pytorch-streams-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_pytorch_streams_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_pytorch_streams_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in PYTORCH_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_jax_naive_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the JAX naive implementation on {title_platform}. The window "
            "is selected from the clustered "
            "\\texttt{XlaModule:\\#hlo\\_module=jit\\_\\_process\\_ready\\_"
            "output\\_core,program\\_id=25} NVTX ranges. CUDA API non-gap is "
            "the union of CUDA runtime API intervals across all JAX host "
            "threads in that window; CUDA API gap is the remaining wall-clock "
            "time. Microsecond values are divided by the number of complete "
            "\\texttt{jit\\_\\_process\\_ready\\_output\\_core} iterations "
            "sampled in that window. }"
        ),
        f"  \\label{{tab:jax-naive-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_jax_naive_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_jax_naive_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in PYTORCH_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_jax_streams_global_latex(
    *,
    platform: str,
    rows: list[dict[str, Any]],
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    midpoint = (len(rows) + 1) // 2
    columns = [column for column in (rows[:midpoint], rows[midpoint:]) if column]

    lines = [
        "\\begin{table*}[p]",
        "  \\centering",
        "  \\scriptsize",
        "  \\renewcommand{\\arraystretch}{0.90}",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "",
        (
            "  \\caption{ Global steady-state CUDA API and GPU activity metrics "
            f"for the JAX streams implementation on {title_platform}. The "
            "window is selected from the clustered "
            "\\texttt{XlaModule:\\#hlo\\_module=jit\\_\\_process\\_ready\\_"
            "output\\_core,program\\_id=25} NVTX ranges. CUDA API non-gap is "
            "the union of CUDA runtime API intervals across all JAX host "
            "threads in that window; CUDA API gap is the remaining wall-clock "
            "time. Microsecond values are divided by the number of complete "
            "\\texttt{jit\\_\\_process\\_ready\\_output\\_core} iterations "
            "sampled in that window. }"
        ),
        f"  \\label{{tab:jax-streams-global-metrics-{label_platform}}}",
        "",
    ]

    for column_index, column_rows in enumerate(columns):
        if column_index > 0:
            lines.extend(["  \\hfill"])

        lines.append("  \\begin{minipage}[t]{0.49\\textwidth}")
        lines.append("    \\centering")
        lines.append("")

        for row_index, row in enumerate(column_rows):
            if row_index > 0:
                lines.extend(["", "    \\vspace{0.7em}", ""])
            lines.extend(render_jax_streams_global_mode_table(row))

        lines.append("  \\end{minipage}")

    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def render_jax_streams_global_mode_table(row: dict[str, Any]) -> list[str]:
    lines = [
        f"    \\textbf{{{latex_escape(str(row['mode_label']))}}}",
        "",
        (
            "    \\begin{tabular}{ >{\\raggedright\\arraybackslash}p{0.68\\linewidth} "
            ">{\\raggedleft\\arraybackslash}p{0.22\\linewidth}"
        ),
        "      }",
        "      \\hline",
        "      \\textbf{Metric} & \\textbf{Global} \\\\",
        "      \\hline",
    ]

    for label, key, kind in PYTORCH_GLOBAL_TABLE_ROWS:
        lines.append(
            f"      {label:<48} & {format_latex_number(row.get(key), kind):<10} \\\\"
        )

    lines.extend(["      \\hline", "    \\end{tabular}"])
    return lines


def render_no_data_latex(
    *,
    platform: str,
    benchmark_label: str,
    label_slug: str,
    reason: str,
) -> str:
    title_platform = platform.title()
    label_platform = platform.replace("_", "-")
    return "\n".join(
        [
            "\\begin{table*}[p]",
            "  \\centering",
            "  \\scriptsize",
            "  \\renewcommand{\\arraystretch}{0.95}",
            "",
            (
                "  \\caption{ Global steady-state CUDA API and GPU activity "
                f"metrics for the {benchmark_label} implementation on "
                f"{title_platform}. No data is currently present. }}"
            ),
            f"  \\label{{tab:{label_slug}-{label_platform}}}",
            "",
            "  \\begin{tabular}{p{0.82\\textwidth}}",
            "    \\hline",
            f"    {latex_escape(reason)} \\\\",
            "    \\hline",
            "  \\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )


def pytorch_no_data_reason(
    platform: str,
    failures_by_platform: dict[str, list[str]],
) -> str:
    if platform == "linux":
        return (
            "No data currently present: the Linux PyTorch-naive Nsight "
            "analysis failed or produced no completed pytorch process_batch "
            "ranges."
        )

    failures = failures_by_platform.get(platform, [])
    if not failures:
        return "No data currently present for this platform."

    return "No valid PyTorch-naive reports were available: " + "; ".join(failures)


def pytorch_threaded_no_data_reason(
    platform: str,
    failures_by_platform: dict[str, list[str]],
) -> str:
    if platform == "linux":
        return (
            "No data currently present: the Linux PyTorch-threaded Nsight "
            "analysis failed or produced no completed pytorch-threaded FFT "
            "batch ranges."
        )

    failures = failures_by_platform.get(platform, [])
    if not failures:
        return "No data currently present for this platform."

    return "No valid PyTorch-threaded reports were available: " + "; ".join(failures)


def pytorch_streams_no_data_reason(
    platform: str,
    failures_by_platform: dict[str, list[str]],
) -> str:
    if platform == "linux":
        return (
            "No data currently present: the Linux PyTorch-streams Nsight "
            "analysis failed or produced no completed pytorch "
            "process_batch_device ranges."
        )

    failures = failures_by_platform.get(platform, [])
    if not failures:
        return "No data currently present for this platform."

    return "No valid PyTorch-streams reports were available: " + "; ".join(failures)


def jax_naive_no_data_reason(
    platform: str,
    failures_by_platform: dict[str, list[str]],
) -> str:
    failures = failures_by_platform.get(platform, [])
    if not failures:
        return "No data currently present for this platform."

    return "No valid JAX-naive reports were available: " + "; ".join(failures)


def jax_streams_no_data_reason(
    platform: str,
    failures_by_platform: dict[str, list[str]],
) -> str:
    failures = failures_by_platform.get(platform, [])
    if not failures:
        return "No data currently present for this platform."

    return "No valid JAX-streams reports were available: " + "; ".join(failures)


def report_metadata_fields(report: ReportContext) -> dict[str, Any]:
    return {
        "platform": report.platform,
        "benchmark": report.benchmark,
        "report_path": str(report.report_path),
        "sqlite_path": str(report.sqlite_path),
        "config_path": "" if report.config_path is None else str(report.config_path),
        "mode_index": report.mode_index,
        "mode_name": report.mode_name,
        "mode_label": report.mode_label,
        "precompute_static_tensors": report.precompute_static_tensors,
        "preallocate_work_buffers": report.preallocate_work_buffers,
        "dummy_gil_thread_enabled": report.dummy_gil_thread_enabled,
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def group_reports_by_platform(
    reports: list[ReportContext],
) -> dict[str, list[ReportContext]]:
    groups: dict[str, list[ReportContext]] = defaultdict(list)
    for report in reports:
        groups[report.platform].append(report)

    return {
        platform: sorted(items, key=report_sort_key)
        for platform, items in sorted(groups.items())
    }


def sort_rows_like_reports(
    rows: list[dict[str, Any]],
    reports: list[ReportContext],
) -> list[dict[str, Any]]:
    order = {
        str(report.report_path): index
        for index, report in enumerate(sorted(reports, key=report_sort_key))
    }
    return sorted(rows, key=lambda row: order.get(str(row["report_path"]), 9999))


def report_sort_key(report: ReportContext) -> tuple[Any, ...]:
    return (
        SUPPORTED_PLATFORMS.index(report.platform)
        if report.platform in SUPPORTED_PLATFORMS
        else 99,
        report.benchmark,
        report.mode_index if report.mode_index is not None else 9999,
        _none_last_bool(report.precompute_static_tensors),
        _none_last_bool(report.preallocate_work_buffers),
        _none_last_bool(report.dummy_gil_thread_enabled),
        report.report_path.name,
    )


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def clip_intervals(
    intervals: list[tuple[int, int, int | None]],
    start_ns: int,
    end_ns: int,
) -> list[tuple[int, int, int | None]]:
    clipped: list[tuple[int, int, int | None]] = []
    for start, end, group in intervals:
        if start >= end_ns:
            continue
        if end <= start_ns:
            continue

        clipped_start = max(start, start_ns)
        clipped_end = min(end, end_ns)
        if clipped_end > clipped_start:
            clipped.append((clipped_start, clipped_end, group))

    return clipped


def count_complete_intervals(
    intervals: list[tuple[int, int]],
    start_ns: int,
    end_ns: int,
) -> int:
    return sum(
        1
        for start, end in intervals
        if start >= start_ns and end <= end_ns
    )


def largest_interval_cluster(
    intervals: list[tuple[int, int]],
    *,
    max_start_gap_ns: int = MAX_COPY_SYNC_GAP_NS,
) -> list[tuple[int, int]]:
    if not intervals:
        return []

    clusters: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    previous_start: int | None = None

    for start, end in sorted(intervals):
        if (
            previous_start is not None
            and start - previous_start > max_start_gap_ns
        ):
            clusters.append(current)
            current = []

        current.append((start, end))
        previous_start = start

    if current:
        clusters.append(current)

    return max(
        clusters,
        key=lambda cluster: (
            len(cluster),
            interval_length_ns(merge_intervals(cluster)),
        ),
    )


def interval_length_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersection_length_ns(
    left: list[tuple[int, int]],
    right: list[tuple[int, int]],
) -> int:
    i = 0
    j = 0
    total = 0

    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            total += end - start

        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1

    return total


def mean_interval_us(intervals: list[tuple[int, int]]) -> float:
    return mean((end - start) / 1e3 for start, end in intervals)


def mean_ns(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return float(mean(items))


def ns_to_us(value: int | float) -> float:
    return float(value) / 1e3


def pct(part: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    return 100.0 * part / total


def fmt_us(value: Any) -> str:
    if value is None:
        return "--"
    return str(int(round(float(value))))


def fmt_pct(value: Any) -> str:
    if value is None:
        return "--"
    return f"{float(value):.1f}"


def format_latex_number(value: Any, kind: str) -> str:
    if kind == "pct":
        return fmt_pct(value)
    return fmt_us(value)


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _flag(value: bool | None) -> str:
    if value is None:
        return "?"
    return "1" if value else "0"


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in {"true", "on", "1", "yes", "enabled"}:
            return True
        if lower in {"false", "off", "0", "no", "disabled"}:
            return False
    return bool(value)


def _none_last_bool(value: bool | None) -> int:
    if value is None:
        return 2
    return int(value)


COMMON_CSV_FIELDS = (
    "platform",
    "benchmark",
    "report_path",
    "sqlite_path",
    "config_path",
    "mode_index",
    "mode_name",
    "mode_label",
    "precompute_static_tensors",
    "preallocate_work_buffers",
    "dummy_gil_thread_enabled",
)

NAIVE_CSV_FIELDS = COMMON_CSV_FIELDS + (
    "scope",
    "count",
    "duration_avg_us",
    "cuda_api_calls_avg",
    "cuda_api_non_gap_avg_us",
    "cuda_api_non_gap_pct",
    "cuda_api_gap_avg_us",
    "cuda_api_gap_pct",
    "kernel_calls_avg",
    "memcpy_calls_avg",
    "gpu_compute_avg_us",
    "gpu_compute_pct",
    "gpu_memcpy_avg_us",
    "gpu_memcpy_pct",
    "gpu_any_avg_us",
    "gpu_any_pct",
    "gpu_idle_avg_us",
    "gpu_idle_pct",
)

THREADED_CSV_FIELDS = COMMON_CSV_FIELDS + (
    "h2d_count",
    "h2d_copy_count",
    "h2d_sync_count",
    "fft_count",
    "postprocess_count",
    "d2h_count",
    "d2h_copy_count",
    "d2h_sync_count",
    "h2d_copy_avg_us",
    "h2d_sync_avg_us",
    "h2d_gap_avg_us",
    "h2d_gap_sync_avg_us",
    "h2d_avg_us",
    "fft_avg_us",
    "postprocess_avg_us",
    "d2h_copy_avg_us",
    "d2h_sync_avg_us",
    "d2h_gap_avg_us",
    "d2h_gap_sync_avg_us",
    "d2h_avg_us",
    "h2d_copy_total_us",
    "h2d_sync_total_us",
    "h2d_gap_total_us",
    "h2d_total_us",
    "fft_total_us",
    "postprocess_total_us",
    "d2h_copy_total_us",
    "d2h_sync_total_us",
    "d2h_gap_total_us",
    "d2h_total_us",
    "h2d_postprocess_overlap_us",
    "d2h_postprocess_overlap_us",
    "memcpy_postprocess_overlap_us",
    "h2d_d2h_overlap_us",
    "h2d_hidden_by_postprocess_pct",
    "d2h_hidden_by_postprocess_pct",
    "memcpy_work_hidden_by_postprocess_pct",
    "postprocess_covered_by_memcpy_pct",
    "h2d_d2h_overlap_pct",
)

THREADED_GLOBAL_CSV_FIELDS = COMMON_CSV_FIELDS + (
    "scope",
    "count",
    "reference_iteration_stage",
    "reference_iterations",
    "window_start_ms",
    "window_end_ms",
    "window_duration_us",
    "duration_avg_us",
    "cuda_api_calls_avg",
    "cuda_api_threads",
    "cuda_api_non_gap_avg_us",
    "cuda_api_non_gap_pct",
    "cuda_api_gap_avg_us",
    "cuda_api_gap_pct",
    "kernel_calls_avg",
    "memcpy_calls_avg",
    "gpu_compute_avg_us",
    "gpu_compute_pct",
    "gpu_memcpy_avg_us",
    "gpu_memcpy_pct",
    "gpu_any_avg_us",
    "gpu_any_pct",
    "gpu_idle_avg_us",
    "gpu_idle_pct",
)

STREAMS_GLOBAL_CSV_FIELDS = COMMON_CSV_FIELDS + (
    "scope",
    "count",
    "reference_iteration_stage",
    "reference_iterations",
    "window_source",
    "window_start_ms",
    "window_end_ms",
    "window_duration_us",
    "duration_avg_us",
    "cuda_api_calls_avg",
    "cuda_api_threads",
    "cuda_api_non_gap_avg_us",
    "cuda_api_non_gap_pct",
    "cuda_api_gap_avg_us",
    "cuda_api_gap_pct",
    "kernel_calls_avg",
    "memcpy_calls_avg",
    "gpu_compute_avg_us",
    "gpu_compute_pct",
    "gpu_memcpy_avg_us",
    "gpu_memcpy_pct",
    "gpu_any_avg_us",
    "gpu_any_pct",
    "gpu_idle_avg_us",
    "gpu_idle_pct",
)

PYTORCH_GLOBAL_CSV_FIELDS = STREAMS_GLOBAL_CSV_FIELDS
JAX_GLOBAL_CSV_FIELDS = STREAMS_GLOBAL_CSV_FIELDS


if __name__ == "__main__":
    raise SystemExit(main())
