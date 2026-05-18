# holoflow-cupy

## Benchmark code layout

The benchmark implementations are intentionally isolated from each other:

- `src/cupy_naive/`
- `src/cupy_threaded/`
- `src/cupy_streams/`
- `src/pytorch_naive/`
- `src/pytorch_threaded/`
- `src/pytorch_streams/`
- `src/jax_naive/`
- `src/jax_streams/`

Each implementation follows the same local shape:

- `compute.py`: implementation-local Doppler/Fresnel/display math.
- `schedule.py`: implementation-local execution schedule.
- `benchmark.py`: benchmark-mode orchestration and stats construction.
- `main.py`: CLI config, input preload, report writing, and optional display.

Shared non-compute utilities live in `src/holoflow_benchmarks/`: config loading,
input IO, reporting, memory-pool cleanup, GIL stress helpers, and benchmark
statistics.

## Nsight Systems profiling sweep

Run Nsight Systems once for every resolved mode in every benchmark YAML:

```powershell
uv run holoflow_nsys_profile
```

This expands each config's `execution.mode_matrix` or `execution.modes` into
generated single-mode configs under `nsys_reports/configs/`, then writes Nsight
reports under `nsys_reports/<benchmark>/`. When `--duration` stops a still
running benchmark, Nsight may return a non-zero exit code after writing the
report; the sweep treats that as success if the expected `.nsys-rep` file was
freshly generated. Use `--strict-exit-codes` to make any non-zero Nsight exit
fail the sweep. The default Nsight arguments match the manual profiling command
used for these benchmarks:

```powershell
nsys profile -f true -t cuda,nvtx,python-gil --python-backtrace=cuda --sample=cpu --delay=3 --duration=5
```

Useful variants:

```powershell
uv run holoflow_nsys_profile --dry-run
uv run holoflow_nsys_profile --skip-existing
uv run holoflow_nsys_profile --benchmark cupy_naive --duration 10
uv run holoflow_nsys_profile --output-dir nsys_reports_long --keep-going
```

## Metric and LaTeX generation

`generated/` is local analysis data and is ignored by Git. Do not commit the
raw `.nsys-rep`, exported `.sqlite`, CSV, or generated LaTeX files from that
folder. After profiling on each OS, copy the benchmark text reports and Nsight
report folders back under `generated/` with this layout:

```text
generated/
  linux_reports/
    <benchmark>_report.txt
  windows_reports/
    <benchmark>_report.txt
  nsight_linux/
    nsys_reports/
      configs/
        <benchmark>/
          <report-stem>.yaml
      <benchmark>/
        <report-stem>.nsys-rep
        <report-stem>.sqlite   # optional; regenerated if missing
  nsys_windows/
    nsys_reports/
      configs/
        <benchmark>/
          <report-stem>.yaml
      <benchmark>/
        <report-stem>.nsys-rep
        <report-stem>.sqlite   # optional; regenerated if missing
```

The `<benchmark>` directory names are the Python package names, for example
`cupy_naive`, `cupy_threaded`, `cupy_streams`, `pytorch_naive`,
`pytorch_threaded`, `pytorch_streams`, `jax_naive`, and `jax_streams`. The
config YAML and `.nsys-rep` basenames must match so the generator can recover
the P/A/G mode metadata.

Then generate SQLite exports, CSV metrics, and LaTeX tables with one command:

```powershell
uv run holoflow_metrics_pipeline generated
```

If `nsys` is not on `PATH`, pass it explicitly:

```powershell
uv run holoflow_metrics_pipeline generated --nsys "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.1\target-windows-x64\nsys.exe"
```

The command discovers supported `.nsys-rep` files, exports a minimal SQLite
table set next to each report when needed, parses generated single-mode configs
to recover P/A/G mode metadata, and writes:

- `generated/metrics/cupy_naive_scope_metrics.csv`
- `generated/metrics/cupy_threaded_overlap.csv`
- `generated/metrics/cupy_threaded_global_metrics.csv`
- `generated/metrics/cupy_streams_global_metrics.csv`
- `generated/metrics/pytorch_naive_global_metrics.csv`
- `generated/metrics/pytorch_threaded_global_metrics.csv`
- `generated/metrics/pytorch_streams_global_metrics.csv`
- `generated/metrics/jax_naive_global_metrics.csv`
- `generated/metrics/jax_streams_global_metrics.csv`
- `generated/latex/cupy_naive_metrics_<platform>.tex`
- `generated/latex/cupy_threaded_overlap_<platform>.tex`
- `generated/latex/cupy_threaded_global_metrics_<platform>.tex`
- `generated/latex/cupy_streams_global_metrics_<platform>.tex`
- `generated/latex/pytorch_naive_global_metrics_<platform>.tex`
- `generated/latex/pytorch_threaded_global_metrics_<platform>.tex`
- `generated/latex/pytorch_streams_global_metrics_<platform>.tex`
- `generated/latex/jax_naive_global_metrics_<platform>.tex`
- `generated/latex/jax_streams_global_metrics_<platform>.tex`

The CuPy-naive LaTeX table keeps the Total / Process batch / Export display
scope layout. The CuPy-threaded table uses the same per-mode multi-table
layout. Its H2D and D2H rows are virtual ranges
that start at `threaded H2D upload` or `threaded D2H output` and end at the
following `threaded H2D sync before enqueue` or `threaded D2H sync before
enqueue`, including the intermediate gap. Overlap percentages use those virtual
copy-thread ranges against `threaded postprocess batch`. The CuPy-threaded
global metrics table uses the same rows as the CuPy-naive table, over the shared
steady-state wall-clock window where H2D, FFT, postprocess, and D2H are all
active. CUDA API non-gap is the union of CUDA runtime API intervals across all
submitting threads; CUDA API gap is the remaining window time. Microsecond values
are divided by the number of complete postprocess iterations sampled in that
window. The CuPy-streams global metrics table uses the same global approach, but
selects the steady-state window from `streams tick ...` ranges and divides
microsecond values by the number of complete `process_batch_device` iterations
sampled in that window, falling back to `finalize_output` if needed. The
PyTorch-naive global metrics table uses the same global rows, starts the window
1 second after trace NVTX origin to skip initialization, and divides microsecond
values by the number of complete `pytorch process_batch` iterations sampled in
that window. The current Linux PyTorch-naive Nsight analysis failed, so the
Linux LaTeX output records that no data is currently present. The
PyTorch-threaded global metrics table uses the same global rows across all
submitting threads, starts the window 2 seconds after trace NVTX origin, and
divides microsecond values by the number of complete `pytorch-threaded FFT batch`
iterations sampled in that window. The current Linux PyTorch-threaded
Nsight analysis also has no completed threaded FFT ranges, so its LaTeX output
records that no data is currently present. The PyTorch-streams global metrics
table starts the window 1.5 seconds after trace NVTX origin and divides
microsecond values by the number of complete `pytorch process_batch_device`
iterations sampled in that window. The current Linux PyTorch-streams Nsight
analysis also has no completed process-batch-device ranges, so its LaTeX output
records that no data is currently present. The JAX-naive global metrics table
uses the same global rows across JAX's runtime threads and selects its window
from the clustered `XlaModule:#hlo_module=jit__process_ready_output_core`
ranges for `program_id=25`. Microsecond values are divided by the number of
complete XLA module iterations sampled in that window. The current JAX-naive
analysis was only run on Linux, so only the Linux table is generated from the
current reports. The JAX-streams global metrics table uses the same XLA module
reference and global-thread accounting; the current analysis was also only run
on Linux.

Useful variants:

```powershell
uv run holoflow_metrics_pipeline generated --benchmark cupy_naive --platform windows
uv run holoflow_metrics_pipeline generated --skip-export
uv run holoflow_metrics_pipeline generated --benchmark cupy_threaded --force-export
```

## CuPy-naive benchmark

The CuPy-naive benchmark is a deliberately single-stream, single-threaded
baseline for LDH processing. It performs one host-to-device upload, one CuPy
compute chain, one device-to-host display transfer, and one explicit
synchronization per output image. It does not attempt GPU/CPU overlap, transfer
batching, or multi-threaded API submission.

Run it with:

```powershell
uv run cupy_naive --config config_cupy_naive.yaml
```

Capture CUDA and NVTX ranges with Nsight Systems:

```powershell
nsys profile -t cuda,nvtx -o cupy_naive .\.venv\Scripts\python.exe -m cupy_naive.main --config config_cupy_naive.yaml
```

## PyTorch-naive benchmark

The PyTorch-naive benchmark mirrors the CuPy-naive schedule and math, but uses
PyTorch tensors, `torch.fft`, PyTorch-pinned host preload buffers, and
`torch.cuda.nvtx` ranges. It is still a single host-thread, default-stream
baseline: no explicit CUDA streams, no pipeline overlap, and one synchronization
per exported display image. The fixed-shape tensor kernels are decorated with
`@torch.compile(mode="default")`; the data-dependent ROI percentile
selection stays eager.

Run it with:

```powershell
uv run pytorch_naive --config config_pytorch_naive.yaml
```

Capture CUDA and NVTX ranges with Nsight Systems:

```powershell
nsys profile -t cuda,nvtx -o pytorch_naive .\.venv\Scripts\python.exe -m pytorch_naive.main --config config_pytorch_naive.yaml
```

## PyTorch threaded benchmark

The PyTorch threaded benchmark keeps the PyTorch-naive math path and pinned
host preload buffers, but drives it with the same four-stage host pipeline used
by the CuPy threaded benchmark: H2D upload, FFT-heavy GPU compute, sequential
postprocessing, and D2H output. Each stage owns a PyTorch CUDA stream and
hands work to the next stage through bounded queues after synchronizing that
stage stream.

Run it with:

```powershell
uv run pytorch_threaded --config config_pytorch_threaded.yaml
```

Capture CUDA and NVTX ranges with Nsight Systems:

```powershell
nsys profile -t cuda,nvtx -o pytorch_threaded .\.venv\Scripts\python.exe -m pytorch_threaded.main --config config_pytorch_threaded.yaml
```

## PyTorch single-thread stream benchmark

The PyTorch stream benchmark keeps the PyTorch-naive math path and pinned host
preload buffers, but submits asynchronous H2D, compute, and D2H work into three
`torch.cuda.Stream` instances from one host thread. PyTorch CUDA events connect
the stage dependencies and gate ring-buffer slot reuse, matching the CuPy
stream scheduler shape without adding worker threads.

Run it with:

```powershell
uv run pytorch_streams --config config_pytorch_streams.yaml
```

Capture CUDA and NVTX ranges with Nsight Systems:

```powershell
nsys profile -t cuda,nvtx -o pytorch_streams .\.venv\Scripts\python.exe -m pytorch_streams.main --config config_pytorch_streams.yaml
```

## JAX-naive benchmark

The JAX-naive benchmark is Linux-only and mirrors the single-host-thread naive
schedule. It uses JAX arrays, `jax.numpy.fft`, JIT-compiled fixed-shape kernels,
and one `block_until_ready`/host-copy path per exported display image. The JAX
CUDA dependency is platform-gated in `pyproject.toml`, so the GUI and existing
CuPy/PyTorch benchmarks can still be installed and run on Windows.

Run it with:

```bash
uv run jax_naive --config config_jax_naive.yaml
```

Capture CUDA and JAX profiler ranges with Nsight Systems:

```bash
nsys profile -t cuda,nvtx -o jax_naive uv run jax_naive --config config_jax_naive.yaml
```

## JAX-managed stream benchmark

JAX owns CUDA stream selection and dependency tracking, so the stream benchmark
does not create CUDA stream or event objects. Instead, one host thread keeps a
bounded number of JAX-dispatched display outputs in flight, starts host copies
with `copy_to_host_async`, and collects the oldest output with
`block_until_ready`.

Run it with:

```bash
uv run jax_streams --config config_jax_streams.yaml
```

Capture CUDA and JAX profiler ranges with Nsight Systems:

```bash
nsys profile -t cuda,nvtx -o jax_streams uv run jax_streams --config config_jax_streams.yaml
```

## CuPy threaded benchmark

The threaded benchmark keeps the same computation but splits the runtime into
four host workers: H2D upload, FFT-heavy GPU compute, sequential postprocessing,
and D2H output. Each worker owns a separate CUDA stream and hands work to the
next stage through bounded queues after synchronizing its stream. The threaded
runner also sets Python's GIL switch interval from `threading.gil_switch_interval_s`
for the duration of the benchmark. Optional `dummy_gil_thread` modes add one
extra pure-Python contention thread on top of the pipeline workers. Admission is
queue-driven: the producer fills the H2D input queue to `threading.queue_depth`,
then waits for completed outputs before submitting more work. By default the
pipeline uses timed `queue.put` calls for every queue output operation; set
`threading.queue_put_policy: nowait` to use `put_nowait` attempts instead.

Run it with:

```powershell
uv run cupy_threaded --config config_cupy_threaded.yaml
```

Capture CUDA and NVTX ranges with Nsight Systems:

```powershell
nsys profile -t cuda,nvtx -o cupy_threaded .\.venv\Scripts\python.exe -m cupy_threaded.main --config config_cupy_threaded.yaml
```

## CuPy single-thread stream benchmark

The stream benchmark keeps one host scheduling thread and submits asynchronous
H2D, compute, and D2H work into three CUDA streams. CUDA events connect the
stage dependencies and gate ring-buffer slot reuse, so the stream/event graph
replaces the worker queues used by the threaded implementation.

The stream scheduler submits formal pipeline ticks: each tick queues H2D for
the current batch, compute for the previous batch, and D2H for the batch before
that when available. `streams.pipeline_prefetch_batches` controls how many of
these ticks stay submitted ahead of completed outputs, so Nsight should show a
gap between CUDA API submission and GPU execution when the pipeline is full.

Run it with:

```powershell
uv run cupy_streams --config config_cupy_streams.yaml
```

Run the synthetic sanity check:

```powershell
uv run python -m cupy_streams.sanity
```

Capture CUDA and NVTX ranges with Nsight Systems:

```powershell
nsys profile -t cuda,nvtx -o cupy_streams .\.venv\Scripts\python.exe -m cupy_streams.main --config config_cupy_streams.yaml
```
