from __future__ import annotations

import argparse
import copy
import re
import shlex
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ExecutionMode, load_benchmark_config, load_yaml_mapping


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    command: str
    config_path: Path
    implementation_name: str


@dataclass(frozen=True)
class ProfileRun:
    index: int
    total: int
    benchmark: BenchmarkSpec
    mode: ExecutionMode
    config_path: Path
    output_base: Path


@dataclass(frozen=True)
class FileSnapshot:
    exists: bool
    size: int | None
    mtime_ns: int | None


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="cupy_naive",
        command="cupy_naive",
        config_path=Path("config_cupy_naive.yaml"),
        implementation_name="cupy-naive",
    ),
    BenchmarkSpec(
        name="cupy_threaded",
        command="cupy_threaded",
        config_path=Path("config_cupy_threaded.yaml"),
        implementation_name="cupy-threaded",
    ),
    BenchmarkSpec(
        name="cupy_streams",
        command="cupy_streams",
        config_path=Path("config_cupy_streams.yaml"),
        implementation_name="cupy-streams",
    ),
    BenchmarkSpec(
        name="pytorch_naive",
        command="pytorch_naive",
        config_path=Path("config_pytorch_naive.yaml"),
        implementation_name="pytorch-naive",
    ),
    BenchmarkSpec(
        name="pytorch_threaded",
        command="pytorch_threaded",
        config_path=Path("config_pytorch_threaded.yaml"),
        implementation_name="pytorch-threaded",
    ),
    BenchmarkSpec(
        name="pytorch_streams",
        command="pytorch_streams",
        config_path=Path("config_pytorch_streams.yaml"),
        implementation_name="pytorch-streams",
    ),
)
NSYS_DURATION_TERMINATION_EXIT_CODES = {143}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nsight Systems once per resolved benchmark execution mode. "
            "Generated single-mode configs and text reports are written under "
            "the output directory; source YAML files are not modified."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=[benchmark.name for benchmark in BENCHMARKS],
        help=(
            "Benchmark to profile. May be repeated. Defaults to all known "
            "benchmark configs."
        ),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("."),
        help="Directory containing config_*.yaml files. Defaults to the cwd.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nsys_reports"),
        help="Directory for .nsys-rep outputs and generated configs.",
    )
    parser.add_argument(
        "--trace",
        default="cuda,nvtx,python-gil",
        help="Nsight Systems trace set passed to -t.",
    )
    parser.add_argument(
        "--python-backtrace",
        default="cuda",
        help="Value passed to --python-backtrace. Use an empty string to skip it.",
    )
    parser.add_argument(
        "--sample",
        default="cpu",
        help="Value passed to --sample. Use an empty string to skip it.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Nsight capture delay in seconds. Defaults to 3.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Nsight capture duration in seconds. Defaults to 5.",
    )
    parser.add_argument(
        "--no-duration",
        action="store_true",
        help="Do not pass --duration to nsys.",
    )
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not overwrite existing Nsight reports.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Preserve display.show_image=true in generated configs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print commands without running nsys.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue profiling later modes if one nsys command fails.",
    )
    parser.add_argument(
        "--strict-exit-codes",
        action="store_true",
        help=(
            "Treat non-zero nsys exit codes as failures even when a fresh "
            ".nsys-rep file was generated."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose expected .nsys-rep file already exists and is non-empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = selected_benchmarks(args.benchmark)
    runs = prepare_runs(
        benchmarks,
        config_root=args.config_root,
        output_dir=args.output_dir,
        show_image=args.show_image,
    )

    if not runs:
        raise SystemExit("No profiling runs were generated.")

    if not args.dry_run:
        require_executable("nsys")
        require_executable("uv")

    failures = 0
    for run in runs:
        print(
            f"[{run.index}/{run.total}] {run.benchmark.name}: {run.mode.name}",
            flush=True,
        )

        if args.skip_existing and existing_nonempty_report(run):
            print(f"Skipping existing report: {expected_report_path(run)}", flush=True)
            continue

        command = build_nsys_command(run, args)
        print(shlex.join(command), flush=True)

        if args.dry_run:
            continue

        report_before = snapshot_file(expected_report_path(run))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            if nsys_report_completed(
                run,
                args,
                completed.returncode,
                report_before,
            ):
                continue

            failures += 1
            print(
                f"Nsight command failed with exit code {completed.returncode}.",
                flush=True,
            )
            if not args.keep_going:
                raise SystemExit(completed.returncode)

    if args.dry_run:
        print(f"Prepared {len(runs)} Nsight profiling commands.")
    elif failures:
        raise SystemExit(f"{failures} Nsight profiling command(s) failed.")
    else:
        print(f"Completed {len(runs)} Nsight profiling runs.")


def selected_benchmarks(names: Iterable[str] | None) -> list[BenchmarkSpec]:
    if names is None:
        return list(BENCHMARKS)

    wanted = set(names)
    return [benchmark for benchmark in BENCHMARKS if benchmark.name in wanted]


def prepare_runs(
    benchmarks: Iterable[BenchmarkSpec],
    *,
    config_root: Path,
    output_dir: Path,
    show_image: bool,
) -> list[ProfileRun]:
    runs: list[ProfileRun] = []

    for benchmark in benchmarks:
        source_config = config_root / benchmark.config_path
        raw = load_yaml_mapping(source_config)
        _, modes = load_benchmark_config(
            source_config,
            implementation_name=benchmark.implementation_name,
        )

        benchmark_output_dir = output_dir / benchmark.name
        config_output_dir = output_dir / "configs" / benchmark.name
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)
        config_output_dir.mkdir(parents=True, exist_ok=True)

        for mode_index, mode in enumerate(modes, start=1):
            slug = slugify(mode.name)
            stem = f"{mode_index:02d}_{slug}"
            config_path = config_output_dir / f"{stem}.yaml"
            output_base = benchmark_output_dir / stem
            report_path = benchmark_output_dir / f"{stem}.txt"

            generated = single_mode_config(
                raw,
                mode=mode,
                report_path=report_path,
                show_image=show_image,
            )
            write_yaml(config_path, generated)

            runs.append(
                ProfileRun(
                    index=0,
                    total=0,
                    benchmark=benchmark,
                    mode=mode,
                    config_path=config_path,
                    output_base=output_base,
                )
            )

    total = len(runs)
    return [
        ProfileRun(
            index=index,
            total=total,
            benchmark=run.benchmark,
            mode=run.mode,
            config_path=run.config_path,
            output_base=run.output_base,
        )
        for index, run in enumerate(runs, start=1)
    ]


def single_mode_config(
    raw: Mapping[str, Any],
    *,
    mode: ExecutionMode,
    report_path: Path,
    show_image: bool,
) -> dict[str, Any]:
    generated = copy.deepcopy(dict(raw))

    execution_cfg = generated.get("execution", {})
    if not isinstance(execution_cfg, Mapping):
        raise TypeError("execution must be a mapping.")

    execution = dict(execution_cfg)
    execution.pop("mode_matrix", None)
    execution["modes"] = [mode_to_yaml(mode)]
    generated["execution"] = execution

    display_cfg = generated.get("display", {})
    if not isinstance(display_cfg, Mapping):
        raise TypeError("display must be a mapping.")

    display = dict(display_cfg)
    display["report_path"] = str(report_path)
    if not show_image:
        display["show_image"] = False
    generated["display"] = display

    return generated


def mode_to_yaml(mode: ExecutionMode) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": mode.name,
        "precompute_static_tensors": mode.precompute_static_tensors,
        "preallocate_work_buffers": mode.preallocate_work_buffers,
        "enable_dummy_gil_thread": mode.enable_dummy_gil_thread,
        "dummy_gil_inner_loops": mode.dummy_gil_inner_loops,
    }

    if mode.dummy_gil_switch_interval_s is not None:
        item["dummy_gil_switch_interval_s"] = mode.dummy_gil_switch_interval_s

    return item


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def build_nsys_command(run: ProfileRun, args: argparse.Namespace) -> list[str]:
    command = [
        "nsys",
        "profile",
        "-o",
        str(run.output_base),
        "-f",
        "false" if args.no_force else "true",
        "-t",
        args.trace,
    ]

    if args.python_backtrace:
        command.append(f"--python-backtrace={args.python_backtrace}")
    if args.sample:
        command.append(f"--sample={args.sample}")
    if args.delay is not None:
        command.append(f"--delay={format_number(args.delay)}")
    if not args.no_duration:
        command.append(f"--duration={format_number(args.duration)}")

    command.extend(
        [
            "uv",
            "run",
            run.benchmark.command,
            "--config",
            str(run.config_path),
        ]
    )
    return command


def nsys_report_completed(
    run: ProfileRun,
    args: argparse.Namespace,
    returncode: int,
    report_before: FileSnapshot,
) -> bool:
    if args.strict_exit_codes:
        return False

    report_path = expected_report_path(run)
    if not fresh_nonempty_file(report_path, report_before):
        return False

    if not args.no_duration and returncode in NSYS_DURATION_TERMINATION_EXIT_CODES:
        print(
            "Nsight stopped the benchmark at the capture duration "
            f"and generated {report_path}.",
            flush=True,
        )
    else:
        print(
            f"Nsight returned exit code {returncode} after generating "
            f"{report_path}; continuing.",
            flush=True,
        )
    return True


def snapshot_file(path: Path) -> FileSnapshot:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return FileSnapshot(exists=False, size=None, mtime_ns=None)
    return FileSnapshot(exists=True, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def fresh_nonempty_file(path: Path, before: FileSnapshot) -> bool:
    after = snapshot_file(path)
    if not after.exists or after.size is None or after.size <= 0:
        return False
    return (
        not before.exists
        or after.size != before.size
        or after.mtime_ns != before.mtime_ns
    )


def expected_report_path(run: ProfileRun) -> Path:
    return run.output_base.with_suffix(".nsys-rep")


def existing_nonempty_report(run: ProfileRun) -> bool:
    report_path = expected_report_path(run)
    try:
        return report_path.stat().st_size > 0
    except FileNotFoundError:
        return False


def format_number(value: float) -> str:
    return f"{value:g}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    return slug[:120] or "mode"


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required executable not found on PATH: {name}")


if __name__ == "__main__":
    main()
