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
