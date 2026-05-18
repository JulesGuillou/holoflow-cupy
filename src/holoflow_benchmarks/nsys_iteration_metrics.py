from __future__ import annotations

import argparse
import csv
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class NvtxRange:
    name: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class Window:
    iteration: int
    scope: str
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class MetricsRow:
    iteration: int
    scope: str
    start_ms: float
    end_ms: float
    duration_us: float

    cuda_api_calls: int
    cuda_api_non_gap_us: float
    cuda_api_gap_us: float
    cuda_api_non_gap_pct: float
    cuda_api_gap_pct: float

    kernel_calls: int
    memcpy_calls: int

    gpu_compute_us: float
    gpu_memcpy_us: float
    gpu_compute_only_us: float
    gpu_memcpy_only_us: float
    gpu_compute_memcpy_overlap_us: float
    gpu_any_us: float
    gpu_idle_us: float

    gpu_compute_pct: float
    gpu_memcpy_pct: float
    gpu_compute_only_pct: float
    gpu_memcpy_only_pct: float
    gpu_compute_memcpy_overlap_pct: float
    gpu_any_pct: float
    gpu_idle_pct: float


class IntervalSelector:
    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = sorted(intervals, key=lambda x: (x.start_ns, x.end_ns))
        self.pos = 0
        self.active: list[Interval] = []

    def candidates(self, window: Window) -> list[Interval]:
        self.active = [
            interval for interval in self.active
            if interval.end_ns > window.start_ns
        ]

        while (
            self.pos < len(self.intervals)
            and self.intervals[self.pos].start_ns < window.end_ns
        ):
            interval = self.intervals[self.pos]
            if interval.end_ns > window.start_ns:
                self.active.append(interval)
            self.pos += 1

        return self.active


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)

    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    return con


def _tables(con: sqlite3.Connection) -> set[str]:
    rows = con.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        """
    ).fetchall()
    return {str(row["name"]) for row in rows}


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row["name"]) for row in rows}


def _read_nvtx_ranges(con: sqlite3.Connection) -> list[NvtxRange]:
    tables = _tables(con)

    if "NVTX_EVENTS" not in tables:
        raise RuntimeError("Missing NVTX_EVENTS table.")

    nvtx_cols = _columns(con, "NVTX_EVENTS")
    has_string_ids = "StringIds" in tables
    string_cols = _columns(con, "StringIds") if has_string_ids else set()

    name_parts: list[str] = []

    if "text" in nvtx_cols:
        name_parts.append("n.text")

    join = ""
    if "textId" in nvtx_cols and has_string_ids and {"id", "value"} <= string_cols:
        name_parts.append("s.value")
        join = "LEFT JOIN StringIds s ON s.id = n.textId"

    if not name_parts:
        name_expr = "'<unnamed>'"
    else:
        name_expr = "COALESCE(" + ", ".join(name_parts) + ", '<unnamed>')"

    rows = con.execute(
        f"""
        SELECT
            {name_expr} AS name,
            n.start AS start_ns,
            n.[end] AS end_ns
        FROM NVTX_EVENTS n
        {join}
        WHERE n.[end] IS NOT NULL
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
        if int(row["end_ns"]) > int(row["start_ns"])
    ]


def _read_intervals(con: sqlite3.Connection, table: str) -> list[Interval]:
    tables = _tables(con)

    if table not in tables:
        return []

    cols = _columns(con, table)
    if not {"start", "end"} <= cols:
        return []

    rows = con.execute(
        f"""
        SELECT
            start AS start_ns,
            [end] AS end_ns
        FROM {table}
        WHERE [end] IS NOT NULL
          AND [end] > start
        ORDER BY start ASC, [end] ASC
        """
    ).fetchall()

    return [
        Interval(
            start_ns=int(row["start_ns"]),
            end_ns=int(row["end_ns"]),
        )
        for row in rows
    ]


def _build_windows(
    nvtx_ranges: list[NvtxRange],
    *,
    process_name: str,
    export_name: str,
    sync_name: str,
    skip_warmup: int,
    max_iterations: int | None,
) -> list[Window]:
    process_ranges = sorted(
        [r for r in nvtx_ranges if r.name == process_name],
        key=lambda r: r.start_ns,
    )
    export_ranges = sorted(
        [r for r in nvtx_ranges if r.name == export_name],
        key=lambda r: r.start_ns,
    )
    sync_ranges = sorted(
        [r for r in nvtx_ranges if r.name == sync_name],
        key=lambda r: r.start_ns,
    )

    if not process_ranges:
        raise RuntimeError(f"No NVTX range named {process_name!r}.")
    if not export_ranges:
        raise RuntimeError(f"No NVTX range named {export_name!r}.")
    if not sync_ranges:
        raise RuntimeError(f"No NVTX range named {sync_name!r}.")

    windows: list[Window] = []

    export_i = 0
    sync_i = 0
    raw_iteration = 0
    kept_iterations = 0

    for process_range in process_ranges:
        while (
            export_i < len(export_ranges)
            and export_ranges[export_i].start_ns < process_range.end_ns
        ):
            export_i += 1

        if export_i >= len(export_ranges):
            break

        export_range = export_ranges[export_i]

        while (
            sync_i < len(sync_ranges)
            and sync_ranges[sync_i].start_ns < export_range.end_ns
        ):
            sync_i += 1

        if sync_i >= len(sync_ranges):
            break

        sync_range = sync_ranges[sync_i]

        raw_iteration += 1

        if raw_iteration <= skip_warmup:
            export_i += 1
            sync_i += 1
            continue

        kept_iterations += 1

        windows.append(
            Window(
                iteration=kept_iterations,
                scope="iteration",
                start_ns=process_range.start_ns,
                end_ns=sync_range.end_ns,
            )
        )
        windows.append(
            Window(
                iteration=kept_iterations,
                scope="process_batch",
                start_ns=process_range.start_ns,
                end_ns=process_range.end_ns,
            )
        )
        windows.append(
            Window(
                iteration=kept_iterations,
                scope="export_display_image",
                start_ns=export_range.start_ns,
                end_ns=export_range.end_ns,
            )
        )
        windows.append(
            Window(
                iteration=kept_iterations,
                scope="sync_per_output",
                start_ns=sync_range.start_ns,
                end_ns=sync_range.end_ns,
            )
        )

        export_i += 1
        sync_i += 1

        if max_iterations is not None and kept_iterations >= max_iterations:
            break

    return windows


def _clip_intervals(
    intervals: list[Interval],
    start_ns: int,
    end_ns: int,
) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []

    for interval in intervals:
        if interval.start_ns >= end_ns:
            continue
        if interval.end_ns <= start_ns:
            continue

        start = max(interval.start_ns, start_ns)
        end = min(interval.end_ns, end_ns)

        if end > start:
            clipped.append((start, end))

    return clipped


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged: list[tuple[int, int]] = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

    return merged


def _interval_length_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def _intersection_length_ns(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> int:
    i = 0
    j = 0
    total = 0

    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])

        if end > start:
            total += end - start

        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

    return total


def _ns_to_us(value: int | float) -> float:
    return float(value) / 1e3


def _pct(part_us: float, total_us: float) -> float:
    if total_us <= 0.0:
        return 0.0
    return 100.0 * part_us / total_us


def _compute_metrics(
    sqlite_path: Path,
    *,
    process_name: str,
    export_name: str,
    sync_name: str,
    skip_warmup: int,
    max_iterations: int | None,
) -> list[MetricsRow]:
    with _connect(sqlite_path) as con:
        nvtx_ranges = _read_nvtx_ranges(con)
        cuda_api_intervals = _read_intervals(con, "CUPTI_ACTIVITY_KIND_RUNTIME")
        kernel_intervals = _read_intervals(con, "CUPTI_ACTIVITY_KIND_KERNEL")
        memcpy_intervals = _read_intervals(con, "CUPTI_ACTIVITY_KIND_MEMCPY")

    windows = _build_windows(
        nvtx_ranges,
        process_name=process_name,
        export_name=export_name,
        sync_name=sync_name,
        skip_warmup=skip_warmup,
        max_iterations=max_iterations,
    )

    if not windows:
        raise RuntimeError("No complete iterations were found.")

    origin_ns = min(window.start_ns for window in windows)

    indexed_windows = sorted(
        list(enumerate(windows)),
        key=lambda item: (item[1].start_ns, item[1].end_ns),
    )

    api_selector = IntervalSelector(cuda_api_intervals)
    kernel_selector = IntervalSelector(kernel_intervals)
    memcpy_selector = IntervalSelector(memcpy_intervals)

    rows_by_index: dict[int, MetricsRow] = {}

    for index, window in indexed_windows:
        duration_ns = window.end_ns - window.start_ns

        api_raw = _clip_intervals(
            api_selector.candidates(window),
            window.start_ns,
            window.end_ns,
        )
        kernel_raw = _clip_intervals(
            kernel_selector.candidates(window),
            window.start_ns,
            window.end_ns,
        )
        memcpy_raw = _clip_intervals(
            memcpy_selector.candidates(window),
            window.start_ns,
            window.end_ns,
        )

        api_merged = _merge_intervals(api_raw)
        kernel_merged = _merge_intervals(kernel_raw)
        memcpy_merged = _merge_intervals(memcpy_raw)

        api_non_gap_ns = _interval_length_ns(api_merged)
        api_gap_ns = max(0, duration_ns - api_non_gap_ns)

        compute_ns = _interval_length_ns(kernel_merged)
        memcpy_ns = _interval_length_ns(memcpy_merged)
        overlap_ns = _intersection_length_ns(kernel_merged, memcpy_merged)

        any_gpu_ns = _interval_length_ns(
            _merge_intervals(kernel_merged + memcpy_merged)
        )

        compute_only_ns = max(0, compute_ns - overlap_ns)
        memcpy_only_ns = max(0, memcpy_ns - overlap_ns)
        gpu_idle_ns = max(0, duration_ns - any_gpu_ns)

        duration_us = _ns_to_us(duration_ns)

        cuda_api_non_gap_us = _ns_to_us(api_non_gap_ns)
        cuda_api_gap_us = _ns_to_us(api_gap_ns)

        gpu_compute_us = _ns_to_us(compute_ns)
        gpu_memcpy_us = _ns_to_us(memcpy_ns)
        gpu_compute_only_us = _ns_to_us(compute_only_ns)
        gpu_memcpy_only_us = _ns_to_us(memcpy_only_ns)
        gpu_compute_memcpy_overlap_us = _ns_to_us(overlap_ns)
        gpu_any_us = _ns_to_us(any_gpu_ns)
        gpu_idle_us = _ns_to_us(gpu_idle_ns)

        rows_by_index[index] = MetricsRow(
            iteration=window.iteration,
            scope=window.scope,
            start_ms=(window.start_ns - origin_ns) / 1e6,
            end_ms=(window.end_ns - origin_ns) / 1e6,
            duration_us=duration_us,

            cuda_api_calls=len(api_raw),
            cuda_api_non_gap_us=cuda_api_non_gap_us,
            cuda_api_gap_us=cuda_api_gap_us,
            cuda_api_non_gap_pct=_pct(cuda_api_non_gap_us, duration_us),
            cuda_api_gap_pct=_pct(cuda_api_gap_us, duration_us),

            kernel_calls=len(kernel_raw),
            memcpy_calls=len(memcpy_raw),

            gpu_compute_us=gpu_compute_us,
            gpu_memcpy_us=gpu_memcpy_us,
            gpu_compute_only_us=gpu_compute_only_us,
            gpu_memcpy_only_us=gpu_memcpy_only_us,
            gpu_compute_memcpy_overlap_us=gpu_compute_memcpy_overlap_us,
            gpu_any_us=gpu_any_us,
            gpu_idle_us=gpu_idle_us,

            gpu_compute_pct=_pct(gpu_compute_us, duration_us),
            gpu_memcpy_pct=_pct(gpu_memcpy_us, duration_us),
            gpu_compute_only_pct=_pct(gpu_compute_only_us, duration_us),
            gpu_memcpy_only_pct=_pct(gpu_memcpy_only_us, duration_us),
            gpu_compute_memcpy_overlap_pct=_pct(
                gpu_compute_memcpy_overlap_us,
                duration_us,
            ),
            gpu_any_pct=_pct(gpu_any_us, duration_us),
            gpu_idle_pct=_pct(gpu_idle_us, duration_us),
        )

    return [rows_by_index[i] for i in range(len(windows))]


def compute_metrics(
    sqlite_path: Path,
    *,
    process_name: str = "process_batch",
    export_name: str = "export_display_image",
    sync_name: str = "sync per output",
    skip_warmup: int = 0,
    max_iterations: int | None = None,
) -> list[MetricsRow]:
    return _compute_metrics(
        sqlite_path,
        process_name=process_name,
        export_name=export_name,
        sync_name=sync_name,
        skip_warmup=skip_warmup,
        max_iterations=max_iterations,
    )


def _write_csv(path: Path, rows: list[MetricsRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    dict_rows = [asdict(row) for row in rows]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dict_rows)


def _print_rows(rows: list[MetricsRow], *, limit: int) -> None:
    dict_rows = [asdict(row) for row in rows]
    if limit > 0:
        dict_rows = dict_rows[:limit]

    try:
        from tabulate import tabulate

        print(tabulate(dict_rows, headers="keys", tablefmt="github", floatfmt=".3f"))
    except Exception:
        for row in dict_rows:
            print(row)


def _print_summary(rows: list[MetricsRow]) -> None:
    groups: dict[str, list[MetricsRow]] = defaultdict(list)

    for row in rows:
        groups[row.scope].append(row)

    summary_rows: list[dict[str, float | int | str]] = []

    for scope, items in groups.items():
        duration_avg_us = mean(r.duration_us for r in items)

        cuda_api_non_gap_avg_us = mean(r.cuda_api_non_gap_us for r in items)
        cuda_api_gap_avg_us = mean(r.cuda_api_gap_us for r in items)

        gpu_compute_avg_us = mean(r.gpu_compute_us for r in items)
        gpu_memcpy_avg_us = mean(r.gpu_memcpy_us for r in items)
        gpu_compute_only_avg_us = mean(r.gpu_compute_only_us for r in items)
        gpu_memcpy_only_avg_us = mean(r.gpu_memcpy_only_us for r in items)
        gpu_overlap_avg_us = mean(r.gpu_compute_memcpy_overlap_us for r in items)
        gpu_any_avg_us = mean(r.gpu_any_us for r in items)
        gpu_idle_avg_us = mean(r.gpu_idle_us for r in items)

        summary_rows.append(
            {
                "scope": scope,
                "count": len(items),

                "duration_avg_us": duration_avg_us,

                "cuda_api_non_gap_avg_us": cuda_api_non_gap_avg_us,
                "cuda_api_non_gap_pct": _pct(
                    cuda_api_non_gap_avg_us,
                    duration_avg_us,
                ),

                "cuda_api_gap_avg_us": cuda_api_gap_avg_us,
                "cuda_api_gap_pct": _pct(
                    cuda_api_gap_avg_us,
                    duration_avg_us,
                ),

                "gpu_compute_avg_us": gpu_compute_avg_us,
                "gpu_compute_pct": _pct(
                    gpu_compute_avg_us,
                    duration_avg_us,
                ),

                "gpu_memcpy_avg_us": gpu_memcpy_avg_us,
                "gpu_memcpy_pct": _pct(
                    gpu_memcpy_avg_us,
                    duration_avg_us,
                ),

                "gpu_compute_only_avg_us": gpu_compute_only_avg_us,
                "gpu_compute_only_pct": _pct(
                    gpu_compute_only_avg_us,
                    duration_avg_us,
                ),

                "gpu_memcpy_only_avg_us": gpu_memcpy_only_avg_us,
                "gpu_memcpy_only_pct": _pct(
                    gpu_memcpy_only_avg_us,
                    duration_avg_us,
                ),

                "gpu_overlap_avg_us": gpu_overlap_avg_us,
                "gpu_overlap_pct": _pct(
                    gpu_overlap_avg_us,
                    duration_avg_us,
                ),

                "gpu_any_avg_us": gpu_any_avg_us,
                "gpu_any_pct": _pct(
                    gpu_any_avg_us,
                    duration_avg_us,
                ),

                "gpu_idle_avg_us": gpu_idle_avg_us,
                "gpu_idle_pct": _pct(
                    gpu_idle_avg_us,
                    duration_avg_us,
                ),
            }
        )

    order = {
        "iteration": 0,
        "process_batch": 1,
        "export_display_image": 2,
        "sync_per_output": 3,
    }
    summary_rows.sort(key=lambda row: order.get(str(row["scope"]), 99))

    try:
        from tabulate import tabulate

        print(tabulate(summary_rows, headers="keys", tablefmt="github", floatfmt=".3f"))
    except Exception:
        for row in summary_rows:
            print(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute iteration-level metrics from an Nsight Systems SQLite export."
    )

    parser.add_argument(
        "sqlite",
        type=Path,
        help="Path to Nsight Systems .sqlite export.",
    )

    parser.add_argument(
        "--process-name",
        default="process_batch",
        help="NVTX range name for the processing step.",
    )

    parser.add_argument(
        "--export-name",
        default="export_display_image",
        help="NVTX range name for the display/export step.",
    )

    parser.add_argument(
        "--sync-name",
        default="sync per output",
        help="NVTX range name for the synchronization step.",
    )

    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=0,
        help="Number of complete iterations to skip.",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of complete iterations to analyze.",
    )

    parser.add_argument(
        "--scope",
        choices=[
            "all",
            "iteration",
            "process_batch",
            "export_display_image",
            "sync_per_output",
        ],
        default="all",
        help="Only print one scope. CSV still follows this filter.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print averaged metrics by scope.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=40,
        help="Maximum printed rows. Use 0 for all.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )

    args = parser.parse_args()

    rows = _compute_metrics(
        args.sqlite,
        process_name=args.process_name,
        export_name=args.export_name,
        sync_name=args.sync_name,
        skip_warmup=args.skip_warmup,
        max_iterations=args.max_iterations,
    )

    if args.scope != "all":
        rows = [row for row in rows if row.scope == args.scope]

    if args.csv is not None:
        _write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")

    if args.summary:
        _print_summary(rows)
    else:
        _print_rows(rows, limit=args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
