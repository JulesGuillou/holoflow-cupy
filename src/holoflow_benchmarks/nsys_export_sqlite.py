from __future__ import annotations

import argparse
import shutil
import subprocess
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
) -> Path:
    report_path = report_path.resolve()

    if not report_path.exists():
        raise FileNotFoundError(report_path)

    if report_path.suffix != ".nsys-rep":
        raise ValueError(f"Expected a .nsys-rep file, got: {report_path}")

    sqlite_path = report_path.with_suffix(".sqlite")

    if sqlite_path.exists():
        if force:
            sqlite_path.unlink()
        else:
            raise FileExistsError(
                f"SQLite file already exists: {sqlite_path}\n"
                "Use --force to overwrite it."
            )

    cmd = [
        str(nsys_path),
        "export",
        "-t",
        "sqlite",
        str(report_path),
    ]

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

    args = parser.parse_args()

    nsys_path = find_nsys(args.nsys)

    for report_path in args.reports:
        sqlite_path = export_sqlite(
            report_path,
            nsys_path=nsys_path,
            force=args.force,
        )
        print(f"Exported: {sqlite_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())