from __future__ import annotations

import argparse
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path


DEFAULT_NSYS = Path(
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.2.1\target-windows-x64\nsys.exe"
)


def find_nsys(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(f"nsys.exe not found: {explicit_path}")

    if DEFAULT_NSYS.exists():
        return DEFAULT_NSYS

    from_path = shutil.which("nsys")
    if from_path is not None:
        return Path(from_path)

    candidates = sorted(
        Path(r"C:\Program Files\NVIDIA Corporation").glob(
            r"Nsight Systems *\target-windows-x64\nsys.exe"
        )
    )
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        "Could not find nsys.exe. Pass --nsys explicitly."
    )


def export_sqlite(
    report_path: Path,
    *,
    nsys_path: Path,
    force: bool,
    output_path: Path | None = None,
    quiet: bool = False,
    tables: Iterable[str] | None = None,
) -> Path:
    report_path = report_path.resolve()

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    if report_path.suffix != ".nsys-rep":
        raise ValueError(f"Expected a .nsys-rep file, got: {report_path}")

    sqlite_path = (
        output_path.resolve()
        if output_path is not None
        else report_path.with_suffix(".sqlite")
    )

    if sqlite_path.exists():
        if force:
            sqlite_path.unlink()
        else:
            raise FileExistsError(
                f"SQLite file already exists: {sqlite_path}\n"
                "Use --force to overwrite it."
            )

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(nsys_path),
        "export",
        "-t",
        "sqlite",
        "-f",
        "true" if force else "false",
        "-o",
        str(sqlite_path),
    ]

    if quiet:
        cmd.append("--quiet=true")

    if tables is not None:
        selected_tables = [table for table in tables if table]
        if selected_tables:
            cmd.append("--tables=" + ",".join(selected_tables))

    cmd.append(str(report_path))

    print(" ".join(f'"{arg}"' if " " in arg else arg for arg in cmd))
    subprocess.run(cmd, check=True)

    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"Nsight export completed, but SQLite file was not found: {sqlite_path}"
        )

    return sqlite_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one or more Nsight Systems .nsys-rep files to SQLite."
    )

    parser.add_argument(
        "reports",
        type=Path,
        nargs="+",
        help="Path(s) to .nsys-rep file(s).",
    )

    parser.add_argument(
        "--nsys",
        type=Path,
        default=None,
        help="Path to nsys.exe.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .sqlite files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for exported SQLite files. Defaults to writing next to "
            "each .nsys-rep file."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet=true to nsys export.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "Export only tables needed by the metric scripts: NVTX, CUDA "
            "runtime, kernel, memcpy, and strings."
        ),
    )

    args = parser.parse_args()

    nsys_path = find_nsys(args.nsys)
    tables = (
        [
            "NVTX_EVENTS",
            "StringIds",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
        ]
        if args.minimal
        else None
    )

    for report_path in args.reports:
        output_path = None
        if args.output_dir is not None:
            output_path = args.output_dir / report_path.with_suffix(".sqlite").name

        sqlite_path = export_sqlite(
            report_path,
            nsys_path=nsys_path,
            force=args.force,
            output_path=output_path,
            quiet=args.quiet,
            tables=tables,
        )
        print(f"Exported: {sqlite_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
