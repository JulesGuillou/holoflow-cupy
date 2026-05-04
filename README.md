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
