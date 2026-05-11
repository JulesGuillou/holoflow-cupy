from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NvtxRange:
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    start_ms: float
    end_ms: float
    duration_us: float
    duration_ms: float
    global_tid: int | None
    pid: int | None
    tid: int | None
    depth: int


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)

    con = sqlite3.connect(sqlite_path)
    con.row_factory = sqlite3.Row
    return con


def _tables(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row["name"]) for row in rows}


def _has_table(con: sqlite3.Connection, table: str) -> bool:
    return table in _tables(con)


def _decode_global_tid(global_tid: int | None) -> tuple[int | None, int | None]:
    if global_tid is None:
        return None, None

    # Nsight serializes pid/tid into one integer.
    # Common decoding used in NVIDIA examples:
    #   pid = globalTid // 0x1000000
    #   tid = globalTid  % 0x1000000
    pid = global_tid // 0x1000000
    tid = global_tid % 0x1000000
    return pid, tid


def _nvtx_query(con: sqlite3.Connection) -> str:
    if not _has_table(con, "NVTX_EVENTS"):
        raise RuntimeError("This SQLite file has no NVTX_EVENTS table.")

    nvtx_cols = _columns(con, "NVTX_EVENTS")
    has_string_ids = _has_table(con, "StringIds")
    string_cols = _columns(con, "StringIds") if has_string_ids else set()

    select_name_parts: list[str] = []

    if "text" in nvtx_cols:
        select_name_parts.append("n.text")

    if "textId" in nvtx_cols and has_string_ids and {"id", "value"} <= string_cols:
        select_name_parts.append("s.value")

    if select_name_parts:
        name_expr = "COALESCE(" + ", ".join(select_name_parts) + ", '<unnamed>')"
    else:
        name_expr = "'<unnamed>'"

    join_expr = ""
    if "textId" in nvtx_cols and has_string_ids and {"id", "value"} <= string_cols:
        join_expr = "LEFT JOIN StringIds s ON s.id = n.textId"

    global_tid_expr = "n.globalTid" if "globalTid" in nvtx_cols else "NULL"

    if "start" not in nvtx_cols or "end" not in nvtx_cols:
        raise RuntimeError("NVTX_EVENTS exists, but it does not have start/end columns.")

    return f"""
        SELECT
            {name_expr} AS name,
            n.start AS start_ns,
            n.end AS end_ns,
            {global_tid_expr} AS global_tid
        FROM NVTX_EVENTS n
        {join_expr}
        WHERE n.end IS NOT NULL
        ORDER BY n.start ASC, n.end DESC
    """


def _compute_depths(rows: list[sqlite3.Row]) -> list[NvtxRange]:
    if not rows:
        return []

    origin_ns = min(int(row["start_ns"]) for row in rows)
    active_by_thread: dict[int | None, list[int]] = {}
    output: list[NvtxRange] = []

    for row in sorted(rows, key=lambda r: (int(r["start_ns"]), int(r["end_ns"]))):
        name = str(row["name"])
        start_ns = int(row["start_ns"])
        end_ns = int(row["end_ns"])
        duration_ns = end_ns - start_ns

        global_tid = row["global_tid"]
        global_tid = int(global_tid) if global_tid is not None else None
        pid, tid = _decode_global_tid(global_tid)

        active = active_by_thread.setdefault(global_tid, [])

        # Important: remove every interval that already ended.
        # Do not assume stack-like nesting.
        active[:] = [active_end for active_end in active if active_end > start_ns]

        depth = len(active)

        active.append(end_ns)

        output.append(
            NvtxRange(
                name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=duration_ns,
                start_ms=(start_ns - origin_ns) / 1e6,
                end_ms=(end_ns - origin_ns) / 1e6,
                duration_us=duration_ns / 1e3,
                duration_ms=duration_ns / 1e6,
                global_tid=global_tid,
                pid=pid,
                tid=tid,
                depth=depth,
            )
        )

    return output


def read_nvtx_ranges(
    sqlite_path: Path,
    *,
    contains: str | None = None,
    regex: str | None = None,
    min_us: float = 0.0,
) -> list[NvtxRange]:
    with _connect(sqlite_path) as con:
        rows = con.execute(_nvtx_query(con)).fetchall()

    ranges = _compute_depths(rows)

    if contains is not None:
        needle = contains.lower()
        ranges = [r for r in ranges if needle in r.name.lower()]

    if regex is not None:
        pattern = re.compile(regex)
        ranges = [r for r in ranges if pattern.search(r.name)]

    if min_us > 0.0:
        ranges = [r for r in ranges if r.duration_us >= min_us]

    return ranges


def _print_tables(sqlite_path: Path) -> None:
    with _connect(sqlite_path) as con:
        for table in _tables(con):
            print(table)


def _print_schema(sqlite_path: Path, table: str) -> None:
    with _connect(sqlite_path) as con:
        if not _has_table(con, table):
            raise RuntimeError(f"No such table: {table}")

        rows = con.execute(f'PRAGMA table_info("{table}")').fetchall()

    for row in rows:
        print(
            f"{row['cid']:>3}  "
            f"{row['name']:<32}  "
            f"{row['type']:<12}  "
            f"notnull={row['notnull']}  "
            f"pk={row['pk']}"
        )


def _print_nvtx_summary(ranges: list[NvtxRange]) -> None:
    by_name: dict[str, list[NvtxRange]] = {}

    for r in ranges:
        by_name.setdefault(r.name, []).append(r)

    summary: list[dict[str, Any]] = []
    for name, items in by_name.items():
        durations = [r.duration_us for r in items]
        summary.append(
            {
                "name": name,
                "calls": len(items),
                "total_ms": sum(durations) / 1e3,
                "avg_us": sum(durations) / len(durations),
                "min_us": min(durations),
                "max_us": max(durations),
            }
        )

    summary.sort(key=lambda row: row["total_ms"], reverse=True)

    _print_rows(
        summary,
        columns=["name", "calls", "total_ms", "avg_us", "min_us", "max_us"],
    )


def _print_rows(rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        print("No rows.")
        return

    try:
        from tabulate import tabulate

        print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".3f"))
    except Exception:
        print(",".join(columns))
        for row in rows:
            print(",".join(str(row.get(col, "")) for col in columns))


def _write_csv(path: Path, ranges: list[NvtxRange]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(r) for r in ranges]
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, ranges: list[NvtxRange]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in ranges], indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Explore an Nsight Systems SQLite export."
    )

    parser.add_argument(
        "sqlite",
        type=Path,
        help="Path to an Nsight Systems .sqlite export.",
    )

    parser.add_argument(
        "--tables",
        action="store_true",
        help="List SQLite tables and exit.",
    )

    parser.add_argument(
        "--schema",
        default=None,
        help="Print schema for one table and exit, for example NVTX_EVENTS.",
    )

    parser.add_argument(
        "--contains",
        default=None,
        help="Only show NVTX ranges whose name contains this text.",
    )

    parser.add_argument(
        "--regex",
        default=None,
        help="Only show NVTX ranges whose name matches this regex.",
    )

    parser.add_argument(
        "--min-us",
        type=float,
        default=0.0,
        help="Only show ranges with duration >= this many microseconds.",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print grouped statistics by NVTX range name.",
    )

    parser.add_argument(
        "--sort",
        choices=["start", "duration", "name"],
        default="start",
        help="Sort displayed ranges.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of ranges to print. Use 0 for all.",
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path for NVTX ranges.",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path for NVTX ranges.",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Only show NVTX ranges at this computed nesting depth.",
    )

    args = parser.parse_args()

    if args.tables:
        _print_tables(args.sqlite)
        return 0

    if args.schema is not None:
        _print_schema(args.sqlite, args.schema)
        return 0

    ranges = read_nvtx_ranges(
        args.sqlite,
        contains=args.contains,
        regex=args.regex,
        min_us=args.min_us,
    )
    
    if args.depth is not None:
        ranges = [r for r in ranges if r.depth == args.depth]

    if args.sort == "duration":
        ranges.sort(key=lambda r: r.duration_ns, reverse=True)
    elif args.sort == "name":
        ranges.sort(key=lambda r: (r.name, r.start_ns))
    else:
        ranges.sort(key=lambda r: r.start_ns)

    if args.csv is not None:
        _write_csv(args.csv, ranges)
        print(f"Wrote CSV: {args.csv}")

    if args.json is not None:
        _write_json(args.json, ranges)
        print(f"Wrote JSON: {args.json}")

    if args.summary:
        _print_nvtx_summary(ranges)
        return 0

    display_ranges = ranges if args.limit == 0 else ranges[: args.limit]

    rows = [
        {
            "name": r.name,
            "start_ms": r.start_ms,
            "end_ms": r.end_ms,
            "duration_us": r.duration_us,
            "duration_ms": r.duration_ms,
            "pid": r.pid,
            "tid": r.tid,
            "depth": r.depth,
        }
        for r in display_ranges
    ]

    _print_rows(
        rows,
        columns=[
            "name",
            "start_ms",
            "end_ms",
            "duration_us",
            "duration_ms",
            "pid",
            "tid",
            "depth",
        ],
    )

    if args.limit and len(ranges) > args.limit:
        print(f"\nDisplayed {args.limit} / {len(ranges)} NVTX ranges.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())