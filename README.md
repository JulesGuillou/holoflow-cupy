# holoflow-cupy

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
