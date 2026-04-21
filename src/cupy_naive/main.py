from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import holofile
import numpy as np
import cupy as cp
import cupyx
import matplotlib.pyplot as plt


RH = np.uint8
R = np.float32
D8 = cp.uint8
RC = cp.float32
C = cp.complex64


@dataclass(frozen=True)
class P:
    path: str = r"C:\Users\guill\Documents\holofiles_data\250527_GUJ_L_2.holo"

    b: int = 32
    n: int = 128

    fs: float = 37_000.0
    f0: float = 8_000.0
    f1: float = 16_000.0

    z: float = 488e-3
    lam: float = 852e-9
    dx: float = 20e-6
    dy: float = 20e-6

    bench_s: float = 10.0
    warmup_laps: int = 1

    show: bool = True


def preload(path: str, b: int, n: int) -> tuple[np.ndarray, np.dtype]:
    with holofile.HoloReader(path) as r:
        src_dtype = np.dtype(r.header.dtype)
        ny = r.header.height
        nx = r.header.width

    nf = n * b

    h = cupyx.empty_pinned((nf, ny, nx), dtype=src_dtype)
    with holofile.HoloReader(path) as r:
        r.read_into(h, 0, nf)

    return h.reshape(n, b, ny, nx), src_dtype


def doppler_slice(w: int, fs: float, f0: float, f1: float) -> tuple[int, int]:
    f = np.fft.rfftfreq(w, d=1.0 / fs)
    k0 = int(np.searchsorted(f, f0, side="left"))
    k1 = int(np.searchsorted(f, f1, side="right"))

    if k0 >= k1:
        raise ValueError(f"Empty Doppler band: window={w}, fs={fs}, f0={f0}, f1={f1}.")

    return k0, k1


def run(h: np.ndarray, src_dtype: np.dtype, p: P) -> tuple[np.ndarray, dict]:
    n, b, ny, nx = h.shape
    k0, k1 = doppler_slice(b, p.fs, p.f0, p.f1)
    nk = k1 - k0

    x = (cp.arange(nx, dtype=RC) - np.float32(nx // 2)) * np.float32(p.dx)
    y = (cp.arange(ny, dtype=RC) - np.float32(ny // 2)) * np.float32(p.dy)
    x2 = cp.square(x, dtype=RC)
    y2 = cp.square(y, dtype=RC)

    phi = y2[:, None] + x2[None, :]
    phi *= np.float32(np.pi / (p.lam * p.z))
    q = cp.exp((1j * phi).astype(C, copy=False)).astype(C, copy=False)

    # Device input buffers:
    # u8: raw uploaded batch in acquisition dtype
    # u : float32 work buffer for FFT pipeline
    u8 = cp.empty((b, ny, nx), dtype=D8)
    u = cp.empty((b, ny, nx), dtype=RC)

    a = cp.zeros((ny, nx), dtype=RC)

    r2 = cp.empty((nk, ny, nx), dtype=RC)
    i2 = cp.empty((nk, ny, nx), dtype=RC)
    s = cp.empty((ny, nx), dtype=RC)

    y_host = cupyx.empty_pinned((ny, nx), dtype=R)

    stream = cp.cuda.get_current_stream()

    def step(xh: np.ndarray) -> None:
        u8.set(xh)
        cp.copyto(u, u8, casting="unsafe")

        f = cp.fft.rfft(u, axis=0)
        g = f[k0:k1]

        g *= q
        g = cp.fft.fft2(g, axes=(-2, -1))

        r2 = cp.abs(g) ** 2
        s = cp.sum(r2, axis=0)
        cp.add(a, s, out=a)

    def export_output() -> None:
        cp.divide(a, np.float32(n), out=a)
        img = cp.fft.fftshift(a)
        img.get(out=y_host)
        a.fill(np.float32(0.0))

    print("Warming up...")
    for _ in range(p.warmup_laps):
        for xh in h:
            step(xh)
        export_output()

    stream.synchronize()
    a.fill(np.float32(0.0))

    print("Running steady-state benchmark...")
    n_batches = 0
    n_frames = 0
    n_outputs = 0
    t0 = perf_counter()

    while True:
        for xh in h:
            step(xh)
            n_batches += 1
            n_frames += b

        export_output()
        n_outputs += 1

        stream.synchronize()

        if perf_counter() - t0 >= p.bench_s:
            dt = perf_counter() - t0

            stats = {
                "seconds": dt,
                "frames": n_frames,
                "batches": n_batches,
                "outputs": n_outputs,
                "input_fps": n_frames / dt,
                "batches_per_s": n_batches / dt,
                "outputs_per_s": n_outputs / dt,
                "output_latency_ms": 1e3 * dt / n_outputs,
                "input_gbps_h2d": (n_frames * ny * nx * src_dtype.itemsize) / dt / 1e9,
                "cast_gbps_d2d": (n_frames * ny * nx * np.dtype(R).itemsize) / dt / 1e9,
                "output_mbps_d2h": (n_outputs * ny * nx * np.dtype(R).itemsize)
                / dt
                / 1e6,
                "shape": (ny, nx),
                "file_dtype": str(src_dtype),
                "host_dtype": str(src_dtype),
                "device_input_dtype": "uint8",
                "real_dtype": "float32",
                "complex_dtype": "complex64",
                "doppler_bins": (k0, k1),
                "doppler_bin_count": nk,
                "accumulation_batches": n,
                "accumulation_frames": n * b,
            }
            return np.array(y_host, copy=True), stats


def main() -> None:
    p = P()

    print("Preloading host data...")
    h, src_dtype = preload(p.path, p.b, p.n)
    print(
        f"Preloaded {h.shape[0] * h.shape[1]} frames of shape {h.shape[2:]} "
        f"in pinned host memory."
    )

    img, s = run(h, src_dtype, p)

    print()
    print("Steady-state benchmark")
    print("----------------------")
    print(f"time                : {s['seconds']:.3f} s")
    print(f"frames              : {s['frames']}")
    print(f"batches             : {s['batches']}")
    print(f"outputs             : {s['outputs']}")
    print(f"input fps           : {s['input_fps']:.1f} frames/s")
    print(f"batches/s           : {s['batches_per_s']:.2f}")
    print(f"outputs/s           : {s['outputs_per_s']:.2f}")
    print(f"output latency      : {s['output_latency_ms']:.2f} ms/output")
    print(f"H2D bandwidth       : {s['input_gbps_h2d']:.3f} GB/s")
    print(f"D2D cast bandwidth  : {s['cast_gbps_d2d']:.3f} GB/s")
    print(f"D2H output BW       : {s['output_mbps_d2h']:.3f} MB/s")
    print(f"file dtype          : {s['file_dtype']}")
    print(f"host dtype          : {s['host_dtype']}")
    print(f"device input dtype  : {s['device_input_dtype']}")
    print(f"real dtype          : {s['real_dtype']}")
    print(f"complex dtype       : {s['complex_dtype']}")
    print(
        f"doppler bins        : [{s['doppler_bins'][0]}:{s['doppler_bins'][1]}) "
        f"({s['doppler_bin_count']} bins)"
    )
    print(
        f"accumulation        : {s['accumulation_batches']} batches "
        f"= {s['accumulation_frames']} frames per output"
    )

    if p.show:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(
            f"Power Doppler average "
            f"({s['accumulation_frames']} frames/output, {s['outputs_per_s']:.2f} outputs/s)"
        )
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
