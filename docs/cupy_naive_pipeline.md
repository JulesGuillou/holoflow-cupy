# CuPy-Naive Pipeline Description

This document describes the CuPy-naive LDH benchmark pipeline implemented in
`src/cupy_naive/cupy_naive.py` and configured by `config_cupy_naive.yaml`. It is
intended as source material for the pipeline and computation section of the
benchmarking article.

## Benchmark Intent

The CuPy-naive pipeline is a direct, single-stream CuPy implementation of the
Power Doppler holography computation. It is meant to be a baseline, not an
optimized runtime architecture.

The benchmark deliberately avoids:

- CPU worker threads for API submission.
- Multiple CUDA streams.
- Explicit overlap between CPU work, host-to-device transfers, GPU kernels, and
  device-to-host transfers.
- Batched output transfers across several display images.
- Custom fused CUDA kernels.
- FFT plan management beyond CuPy/cuFFT defaults.

Each output image is produced by one sequential chain:

1. Transfer one temporal batch from pinned host memory to the GPU.
2. Convert the batch from acquisition dtype to real compute dtype.
3. Compute a temporal FFT and keep the Doppler band of interest.
4. Apply the Fresnel quadratic phase and compute spatial FFTs.
5. Accumulate power over Doppler bins.
6. Update a causal sliding mean over batch-power images.
7. Build a display image with `fftshift` and percentile clipping.
8. Transfer that display image back to host memory.
9. Synchronize the current CUDA stream before measuring the next output.

## Current Article Configuration

The current benchmark configuration is:

| Symbol | Config key | Value | Meaning |
| --- | --- | ---: | --- |
| `H` | input file header | runtime | frame height in pixels |
| `W` | input file header | runtime | frame width in pixels |
| `T` | `schedule.batch_frames` | 32 | frames per temporal batch |
| `M` | `schedule.batches_per_output` | 64 | sliding-window length in batch-power images |
| `N` | `T * M` | 2048 | frame support of one sliding average |
| `fs` | `signal.sample_rate_hz` | 37000 Hz | camera sampling frequency |
| `f0` | `signal.doppler_low_hz` | 8000 Hz | lower Doppler cutoff |
| `f1` | `signal.doppler_high_hz` | 16000 Hz | upper Doppler cutoff |
| `z` | `signal.propagation_distance_m` | 0.486 m | propagation distance |
| `lambda` | `signal.wavelength_m` | 852e-9 m | optical wavelength |
| `dx` | `signal.dx_m` | 20e-6 m | horizontal pixel pitch |
| `dy` | `signal.dy_m` | 20e-6 m | vertical pixel pitch |
| `r` | `display.contrast_roi_radius` | 0.8 | normalized elliptical ROI radius |
| `p_low` | `display.contrast_low_percentile` | 0.2 | display lower percentile |
| `p_high` | `display.contrast_high_percentile` | 99.8 | display upper percentile |

Derived temporal quantities:

- Output stride: `T / fs = 32 / 37000 = 0.8649 ms`.
- Sliding temporal support: `N / fs = 2048 / 37000 = 55.35 ms`.
- Temporal FFT bin spacing: `fs / T = 1156.25 Hz`.
- rFFT bins retained for the current Doppler band: `[k0:k1) = [7:14)`.
- Number of retained Doppler bins: `K = 7`.
- Retained bin centers: 8093.75 Hz through 15031.25 Hz.

The configured dtypes are:

| Role | Dtype |
| --- | --- |
| Acquisition / file / host input | `uint8` |
| Real compute tensors | `float32` |
| Complex compute tensors | `complex64` |
| Display output | `float32` |

## Tensor Inventory

The following table uses `T = batch_frames`, `M = batches_per_output`,
`K = k1 - k0`, and image shape `(H, W)`.

| Tensor | Location | Shape | Dtype | Lifetime | Description |
| --- | --- | --- | --- | --- | --- |
| `host_batches` | pinned CPU | `(M, T, H, W)` | `uint8` | benchmark suite | Preloaded cyclic input. This removes file I/O from the timed compute loop. |
| `host_batch` | pinned CPU view | `(T, H, W)` | `uint8` | one iteration | One temporal batch selected from `host_batches`. |
| `raw_batch_device` | GPU | `(T, H, W)` | `uint8` | one iteration or reused | Device copy of the input batch. Reused only when preallocation is enabled. |
| `real_batch_device` | GPU | `(T, H, W)` | `float32` | one iteration or reused | Real-valued compute input after casting from acquisition dtype. |
| `temporal_spectrum` | GPU | `(K, H, W)` | `complex64` | one iteration | Temporal rFFT result after Doppler band selection. |
| `quadratic_phase` | GPU | `(H, W)` | `complex64` | static or rebuilt | Fresnel input quadratic phase factor. |
| `propagated` | GPU | `(K, H, W)` | `complex64` | one iteration | Spatial FFT result for each retained Doppler bin. |
| `power` | GPU | `(K, H, W)` | `float32` | one iteration | Squared magnitude of `propagated`. |
| `batch_power` | GPU | `(H, W)` | `float32` | one iteration | Doppler power image summed over the retained bins. |
| `sliding_mean.buffer` | GPU | `(M, H, W)` | `float32` | mode lifetime | Ring buffer storing the last `M` batch-power images. |
| `sliding_mean.rolling_sum` | GPU | `(H, W)` | `float32` | mode lifetime | Running sum of the current ring-buffer contents. |
| `mean_image` | GPU | `(H, W)` | `float32` | one output or reused | Sliding average image. Reused only when preallocation is enabled. |
| `shifted` | GPU | `(H, W)` | `float32` | one output | Centered display image after `fftshift`. |
| `roi_mask` | GPU | `(H, W)` | `bool` | static or rebuilt | Normalized elliptical mask used for percentile estimation. |
| `roi_values` | GPU | `(R,)` | `float32` | one output | Values of `shifted` inside the ROI, where `R` is the number of true mask pixels. |
| `q_low`, `q_high` | GPU | scalar | `float32` | one output | Percentile clipping bounds. |
| `display_image` | GPU | `(H, W)` | `float32` | one output or reused | Clipped display image. |
| `output_host` | pinned CPU | `(H, W)` | `float32` | one output or reused | Host-side destination for the final display transfer. |
| returned image | CPU | `(H, W)` | `float32` | reporting | Owning NumPy copy used for reporting and optional plotting. |

## Input Preload

Before benchmarking starts, the code reads exactly `M * T` frames from the
`.holo` file into pinned host memory:

```text
host_batches: uint8[M, T, H, W]
flat_frames = reshape(host_batches, (M * T, H, W))
reader.read_into(flat_frames, start=0, count=M * T)
```

The timed loop cycles through these preloaded batches indefinitely. This makes
the measurement focus on the GPU pipeline and host/device transfer behavior
rather than storage throughput.

## Doppler Bin Selection

For each temporal batch, the temporal FFT length is `T`, not the full sliding
window length `M * T`. CuPy computes:

```text
rfft(real_batch_device, axis=0)
```

The rFFT frequency bins are:

```text
freq[k] = k * fs / T,  for k = 0, ..., T/2
```

The selected Doppler interval is inclusive on the lower bound and exclusive on
the upper index:

```text
k0 = searchsorted(freq, f0, side="left")
k1 = searchsorted(freq, f1, side="right")
temporal_spectrum = rfft_result[k0:k1]
```

For the current configuration, this gives `[7:14)`, so `K = 7`.

## Coordinate System and Fresnel Phase

The spatial coordinates are centered on the image:

```text
x[j] = (j - (W - 1) / 2) * dx,  j = 0, ..., W - 1
y[i] = (i - (H - 1) / 2) * dy,  i = 0, ..., H - 1
```

The quadratic phase tensor is:

```text
Q[i, j] = exp(1j * pi * (x[j]^2 + y[i]^2) / (lambda * z))
```

Its shape is `(H, W)` and dtype is `complex64`. If
`precompute_static_tensors=true`, `Q` is built once when the mode starts. If
`precompute_static_tensors=false`, `Q` is rebuilt for every processed batch.

## Per-Batch Computation

Given one host batch:

```text
U_host: uint8[T, H, W]
```

the benchmark performs the following operations.

### 1. Host-to-Device Upload

```text
U_u8 = asarray(U_host, dtype=uint8)
```

or, when preallocation is enabled:

```text
raw_batch_device.set(U_host)
U_u8 = raw_batch_device
```

Result:

```text
U_u8: uint8[T, H, W]
```

### 2. Cast to Real Compute Dtype

```text
U = U_u8.astype(float32)
```

or, when preallocation is enabled:

```text
copyto(real_batch_device, U_u8, casting="unsafe")
U = real_batch_device
```

Result:

```text
U: float32[T, H, W]
```

### 3. Temporal FFT and Doppler Band Selection

```text
F_full = rfft(U, axis=0)
F = F_full[k0:k1]
```

Result:

```text
F: complex64[K, H, W]
```

### 4. Fresnel Propagation

The Fresnel phase is multiplied into every retained Doppler plane using
broadcasting:

```text
G[k, i, j] = F[k, i, j] * Q[i, j]
```

Then a 2D FFT is applied over the spatial axes:

```text
P = fft2(G, axes=(-2, -1))
```

Result:

```text
P: complex64[K, H, W]
```

### 5. Doppler Power Accumulation

The power is accumulated over Doppler bins:

```text
A[k, i, j] = abs(P[k, i, j])^2
S[i, j] = sum_k A[k, i, j]
```

Result:

```text
S: float32[H, W]
```

This `S` tensor is one batch-power image.

## Sliding Mean Over Batch-Power Images

The pipeline keeps a causal sliding mean over the most recent `M` batch-power
images. Let `S_n` be the batch-power image produced by temporal batch `n`.

The rolling sum is:

```text
R_n = sum_{m=n-M+1}^{n} S_m
```

The displayed mean becomes valid once the ring buffer contains `M` batch-power
images:

```text
B_n = R_n / M
```

Implementation details:

- `sliding_mean.buffer` is a ring buffer with shape `(M, H, W)`.
- `sliding_mean.rolling_sum` stores `R_n`.
- On each new `S_n`, the oldest slot is subtracted, replaced by `S_n`, and then
  added back to the rolling sum.
- The first `M - 1` batches are used to prime the ring buffer.
- After priming, each new processed batch produces one valid output.

The display output rate is therefore one image per `T` input frames, while each
image averages over `M * T` input frames.

## Display Stage

The display stage is part of the measured pipeline because it produces the
final host-visible image used by the benchmark.

### 1. Average

```text
B = R / M
```

Result:

```text
B: float32[H, W]
```

### 2. FFT Shift

```text
C = fftshift(B)
```

Result:

```text
C: float32[H, W]
```

### 3. Elliptical ROI

The ROI uses normalized centered coordinates:

```text
u[j] = (j - (W - 1) / 2) * (2 / W)
v[i] = (i - (H - 1) / 2) * (2 / H)
mask[i, j] = (u[j]^2 + v[i]^2) <= r^2
```

With the current configuration, `r = 0.8`.

If `precompute_static_tensors=true`, this mask is built once per mode. If
`precompute_static_tensors=false`, it is rebuilt for every output image.

### 4. Percentile Clipping

Percentiles are estimated only inside the ROI:

```text
roi_values = C[mask]
q_low, q_high = percentile(roi_values, [0.2, 99.8])
```

The clipping is then applied globally:

```text
D = clip(C, q_low, q_high)
```

Result:

```text
D: float32[H, W]
```

## Device-to-Host Output Transfer

The final display image is transferred to host memory once per output:

```text
host_display = D.get(blocking=True)
```

or, when preallocation is enabled:

```text
D.get(out=output_host, blocking=True)
host_display = np.array(output_host, copy=True)
```

The extra NumPy copy is intentional: it returns an owning array whose contents
cannot be overwritten by later benchmark iterations.

## Benchmark Schedule

Each mode runs three phases.

### Prime

The pipeline processes `M - 1` batches to fill the sliding window up to the
point where the next batch will produce the first valid output. The stream is
synchronized after priming.

### Warmup

The pipeline produces `warmup_outputs` valid outputs before timing starts. The
stream is synchronized after warmup. Warmup outputs are not included in the
reported performance metrics.

### Measurement

The benchmark repeatedly:

1. Processes one batch.
2. Produces one display image.
3. Transfers the display image to host memory.
4. Synchronizes the current CUDA stream.
5. Updates wall-clock elapsed time.

The explicit synchronization after each output is part of the naive baseline.
It ensures that wall-clock timing corresponds to completed images, but it also
prevents overlap between consecutive outputs.

## Execution Modes

The current configuration benchmarks four single-threaded modes:

| `precompute_static_tensors` | `preallocate_work_buffers` | Meaning |
| --- | --- | --- |
| `false` | `false` | Rebuild static tensors and allocate temporary/output buffers on demand. |
| `false` | `true` | Rebuild static tensors on demand, but reuse work/output buffers. |
| `true` | `false` | Reuse static tensors, but allocate temporary/output buffers on demand. |
| `true` | `true` | Reuse static tensors and reusable work/output buffers. |

The optional dummy GIL stress thread is disabled in the article configuration.
When enabled manually, it is a host-side stress condition and should be reported
as a separate experiment rather than as the single-threaded CuPy-naive baseline.

## NVTX Ranges

The benchmark emits NVTX ranges so Nsight Systems can display the pipeline
structure. The main steady-state ranges are:

| NVTX range | Pipeline section |
| --- | --- |
| `process_batch` | Full per-batch compute path through sliding-mean update. |
| `H2D upload` | Host-to-device input transfer. |
| `cast to f32` | Conversion from `uint8` to `float32`. |
| `temporal FFT + band select` | Temporal rFFT and Doppler bin slicing. |
| `prepare Fresnel phase` | Retrieve or rebuild `Q`. |
| `Fresnel` | Broadcast phase multiply and spatial FFTs. |
| `accumulate power` | `abs(.)^2` and sum over Doppler bins. |
| `sliding mean update` | Ring-buffer update and rolling-sum maintenance. |
| `finalize_output` | Full display path and host export. |
| `average` | Divide rolling sum by `M`. |
| `fftshift` | Center display image. |
| `percentile clip` | ROI selection, percentile estimation, and clipping. |
| `D2H output` | Display image transfer to host memory. |
| `sync per output` | Explicit stream synchronization after each output. |

Setup ranges also mark config loading, input inspection, preload, static tensor
creation, buffer allocation, priming, warmup, and report writing.

## Compact Mathematical Summary

For temporal batch `n`, with input `U_n[t, y, x]`, the computation is:

```text
F_n[k, y, x] = rFFT_t(U_n)[k],              k in [k0, k1)
Q[y, x]      = exp(1j * pi * (x^2 + y^2) / (lambda * z))
P_n[k, y, x] = FFT_xy(F_n[k, y, x] * Q[y, x])
S_n[y, x]    = sum_k |P_n[k, y, x]|^2
R_n[y, x]    = sum_{m=n-M+1}^{n} S_m[y, x]
B_n[y, x]    = R_n[y, x] / M
C_n[y, x]    = fftshift(B_n)[y, x]
D_n[y, x]    = clip(C_n[y, x], q_low, q_high)
```

where `q_low` and `q_high` are the `0.2` and `99.8` percentiles of `C_n`
inside the normalized elliptical ROI.

The final benchmark output for each iteration is `D_n` transferred to the CPU
as a `float32[H, W]` NumPy array.
