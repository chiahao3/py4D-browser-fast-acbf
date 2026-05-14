#!/usr/bin/env python
"""Profile two BF-gather/update strategies for live acquisition.

Path A (legacy host-masking path):
    1) host masking:      vbf_h = dataset[:, :, bf_mask_bool]   # CPU strided gather
    2) host reorganize:   ascontiguousarray(moveaxis(vbf_h, -1, 0))
    3) host_to_pinned:    copy 3D vBF stack into a (Nb, Ry, Rx) pinned host buffer
    4) h2d_small:         non-blocking H2D of the pinned 3D buffer to device

Path B (live pinned-source path):
    1) host_to_pinned:    copy the input numpy 4D into a (Ry, Rx, Ky, Kx) pinned host buffer
    2) h2d_large:         non-blocking H2D of the pinned 4D buffer to a (Ry, Rx, Ky, Kx) device buffer
    3) device_gather:     vbf_d = dataset_d[:, :, bf_mask_bool_d]   # GPU fancy-index
    4) device_reorganize: gathered.permute(2, 0, 1).contiguous()    # to (Nb, Ry, Rx)

Both paths produce the same (Nb, Ry, Rx) float32 tensor on device. Each stage
is timed independently with `torch.cuda.synchronize` between stages so the
reported numbers are honest wall-clock, not asynchronous launch latency.

Per-frame data-source assumption
--------------------------------
The numbers reported below assume the new frame arrives every iteration in a
fresh non-pinned numpy array (the typical Python live-acquisition pipeline).
That makes Path B's `host_to_pinned_large` step (a 1 GB DDR-bandwidth memcpy
from numpy into pinned host memory, ~22 GB/s on this DDR) part of the per-
frame budget, alongside the H2D itself.

If the detector vendor instead DMAs frames *directly into a pinned host
buffer* (or even straight into GPU memory via GPUDirect/RDMA), the
`host_to_pinned_large` step drops out and Path B's per-frame cost becomes
just the H2D + device gather. That second scenario is roughly 2x faster
than the numbers below, and is what the vendor-adapter layer should aim for
once it exists.

The GUI Live Demo now exercises Path B when "Use CUDA pinned source buffer" is
enabled. This script remains useful as a lower-level diagnostic when comparing
BF gather strategies outside the full solver/reconstruction loop.

Usage:
    python scripts/mask_path_profile.py path/to/4d.hdf5 \\
        --max-alpha 25 --dk 0.04 --voltage 80 --iters 20
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from fast_acbf.pipeline import compute_bf_geometry
from py4d_browser_plugin.fast_acbf.calibration import electron_wavelength_angstrom
from py4d_browser_plugin.fast_acbf.live_worker import _cuda_sync


def percentiles(samples_s: list[float]) -> tuple[float, float, float, float]:
    arr = sorted(samples_s)
    mean = statistics.mean(arr) * 1000
    p50 = statistics.median(arr) * 1000
    p95 = arr[int(0.95 * (len(arr) - 1))] * 1000
    p99 = arr[int(0.99 * (len(arr) - 1))] * 1000
    return mean, p50, p95, p99


def report(name: str, stages: dict[str, list[float]]) -> None:
    print(f"\n=== {name} ===")
    total = [sum(v) for v in zip(*stages.values())]
    for stage, samples in stages.items():
        m, p50, p95, p99 = percentiles(samples)
        print(f"  {stage:32s} mean={m:8.2f}  p50={p50:8.2f}  p95={p95:8.2f}  p99={p99:8.2f}  ms")
    m, p50, p95, p99 = percentiles(total)
    print(f"  {'TOTAL':32s} mean={m:8.2f}  p50={p50:8.2f}  p95={p95:8.2f}  p99={p99:8.2f}  ms")


def load_4d(path: Path, key: str | None) -> np.ndarray:
    from ptyrad.io.handlers import load_array_from_file

    arr = load_array_from_file(str(path), key=key, ndims=[4])
    return np.asarray(arr, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--key", default=None)
    p.add_argument("--max-alpha", type=float, default=25.0, help="mrad")
    p.add_argument("--dk", type=float, default=0.04, help="inverse Angstrom")
    p.add_argument("--voltage", type=float, default=80.0, help="kV")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[load] {args.dataset}")
    data = load_4d(args.dataset, args.key)
    Ry, Rx, Ky, Kx = data.shape
    nbytes = data.nbytes
    print(f"[load] shape={data.shape} dtype={data.dtype} size={nbytes / 1e9:.2f} GB")

    wavelength = electron_wavelength_angstrom(args.voltage)
    _, _, bf_mask_bool = compute_bf_geometry(Ky, Kx, args.max_alpha, args.dk, wavelength)
    Nb = int(bf_mask_bool.sum())
    print(f"[mask] Nb={Nb} (max_alpha={args.max_alpha} mrad, dk={args.dk} Å^-1, λ={wavelength:.4g} Å)")

    device = args.device
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise SystemExit("CUDA not available")

    bf_mask_bool_d = torch.from_numpy(bf_mask_bool).to(device)

    # Pinned host buffer for Path B (allocated once, refilled every frame).
    pinned_4d = torch.empty(data.shape, dtype=torch.float32, pin_memory=True)
    # Source numpy array used by Path A (host masking reads this).
    src_np = data

    # Device buffers for each path
    out_3d_a = torch.empty((Nb, Ry, Rx), dtype=torch.float32, device=device)
    pinned_3d_a = torch.empty((Nb, Ry, Rx), dtype=torch.float32, pin_memory=True)
    dataset_d_b = torch.empty(data.shape, dtype=torch.float32, device=device)

    # ---------- Path A: host masking + small H2D ----------
    a_stages: dict[str, list[float]] = {
        "host_gather": [],
        "host_reorganize": [],
        "host_to_pinned": [],
        "h2d_small": [],
    }
    for it in range(args.iters + args.warmup):
        t0 = time.perf_counter()
        masked = src_np[:, :, bf_mask_bool]  # (Ry, Rx, Nb)
        t1 = time.perf_counter()
        masked = np.ascontiguousarray(np.moveaxis(masked, -1, 0)).astype(np.float32, copy=False)
        t2 = time.perf_counter()
        pinned_3d_a.copy_(torch.from_numpy(masked))
        t3 = time.perf_counter()
        out_3d_a.copy_(pinned_3d_a, non_blocking=True)
        _cuda_sync(device)
        t4 = time.perf_counter()
        if it >= args.warmup:
            a_stages["host_gather"].append(t1 - t0)
            a_stages["host_reorganize"].append(t2 - t1)
            a_stages["host_to_pinned"].append(t3 - t2)
            a_stages["h2d_small"].append(t4 - t3)

    # ---------- Path B: large H2D + device masking ----------
    b_stages: dict[str, list[float]] = {
        "host_to_pinned_large": [],
        "h2d_large": [],
        "device_gather": [],
        "device_reorganize": [],
    }
    out_3d_b = None
    for it in range(args.iters + args.warmup):
        t0 = time.perf_counter()
        pinned_4d.copy_(torch.from_numpy(src_np))
        t1 = time.perf_counter()
        dataset_d_b.copy_(pinned_4d, non_blocking=True)
        _cuda_sync(device)
        t2 = time.perf_counter()
        gathered = dataset_d_b[:, :, bf_mask_bool_d]  # (Ry, Rx, Nb)
        _cuda_sync(device)
        t3 = time.perf_counter()
        out_3d_b = gathered.permute(2, 0, 1).contiguous()
        _cuda_sync(device)
        t4 = time.perf_counter()
        if it >= args.warmup:
            b_stages["host_to_pinned_large"].append(t1 - t0)
            b_stages["h2d_large"].append(t2 - t1)
            b_stages["device_gather"].append(t3 - t2)
            b_stages["device_reorganize"].append(t4 - t3)

    # Sanity check: equality of outputs.
    assert out_3d_b is not None
    diff = (out_3d_a - out_3d_b).abs().max().item()
    print(f"\n[sanity] max |Path A - Path B| = {diff:.3g} (should be ~0)")

    report("Path A: legacy host masking + small H2D", a_stages)
    report("Path B: live pinned-source large H2D + device masking", b_stages)

    a_total = statistics.mean([sum(v) for v in zip(*a_stages.values())]) * 1000
    b_total = statistics.mean([sum(v) for v in zip(*b_stages.values())]) * 1000
    print(
        f"\n[summary] Path A mean total = {a_total:.2f} ms"
        f"   Path B mean total = {b_total:.2f} ms"
        f"   speedup = {a_total / b_total:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
