#!/usr/bin/env python
"""Micro-benchmark: isolate per-frame H2D latency for different update strategies.

Simulates a 4D dataset in CUDA-pinned host memory and measures the time to:
  A. H2D only (copy_ into pre-allocated staging buffer)
  B. H2D + GPU bool-mask extraction
  C. H2D + mask + FFT (the full fast update_dataset path, no empty_cache)
  D. As_tensor H2D + mask + FFT + empty_cache (the original broken path)

Usage:
    python scripts/test_h2d_latency.py [--scan SY SX] [--det DY DX] [--nb NB] [--reps N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def sync():
    torch.cuda.synchronize()


def measure(name: str, fn, reps: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    dt_ms = (time.perf_counter() - t0) / reps * 1000
    print(f"  {name:<50s} {dt_ms:8.2f} ms")
    return dt_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", type=int, nargs=2, default=[64, 64], metavar=("SY", "SX"))
    p.add_argument("--det", type=int, nargs=2, default=[256, 256], metavar=("DY", "DX"))
    p.add_argument("--nb", type=int, default=1000, help="Number of BF pixels (aperture size)")
    p.add_argument("--reps", type=int, default=30)
    args = p.parse_args()

    Ry, Rx = args.scan
    Ky, Kx = args.det
    nb = args.nb
    shape = (Ry, Rx, Ky, Kx)
    nbytes = Ry * Rx * Ky * Kx * 4
    dev = torch.device("cuda")

    print(f"Dataset shape: {shape}  ({nbytes / 1e9:.2f} GB)  nb={nb} BF pixels")
    print()

    # Pinned host buffer (simulates CUDA-pinned output from MockStreamer)
    pinned_np = np.random.default_rng(0).random(shape, dtype=np.float32)
    pinned_tensor = torch.from_numpy(pinned_np)
    assert not pinned_tensor.is_cuda

    # Allocate on pinned memory properly
    pinned_pt = torch.empty(shape, dtype=torch.float32, pin_memory=True)
    pinned_pt.copy_(pinned_tensor)
    pinned_np = pinned_pt.numpy()  # numpy view of pinned tensor (shares memory)

    # Pre-allocated GPU staging buffer (reused each frame)
    staging = torch.empty(shape, dtype=torch.float32, device=dev)

    # BF pixel indices on device (precomputed once at solver build time)
    bf_iy = torch.randint(0, Ky, (nb,), device=dev, dtype=torch.long)
    bf_ix = torch.randint(0, Kx, (nb,), device=dev, dtype=torch.long)

    # Precomputed FFT result buffer
    fft_cache = torch.empty((nb, Ry, Rx), dtype=torch.complex64, device=dev)

    print("=== Strategy A: H2D copy_ only (pinned → pre-allocated staging) ===")
    measure("copy_(non_blocking=False)", lambda: staging.copy_(pinned_pt, non_blocking=False), args.reps)
    measure("copy_(non_blocking=True) + sync", lambda: (staging.copy_(pinned_pt, non_blocking=True), sync()), args.reps)

    print()
    print("=== Strategy B: H2D + GPU bool-mask extraction ===")
    def b():
        staging.copy_(pinned_pt, non_blocking=False)
        _ = staging[:, :, bf_iy, bf_ix].permute(2, 0, 1).contiguous()
    measure("copy_ + gather + contiguous", b, args.reps)

    print()
    print("=== Strategy C: H2D + mask + FFT2 (target fast path, no empty_cache) ===")
    def c():
        staging.copy_(pinned_pt, non_blocking=False)
        vbf = staging[:, :, bf_iy, bf_ix].permute(2, 0, 1).contiguous()
        fft = torch.fft.fft2(vbf, dim=(-2, -1))
        # Simulate in-place cache update (no allocation after warmup)
        _ = fft
    measure("copy_ + gather + fft2 (no empty_cache)", c, args.reps)

    print()
    print("=== Strategy D: as_tensor H2D + mask + FFT + empty_cache (old broken path) ===")
    def d():
        arr_gpu = torch.as_tensor(pinned_np, device=dev)
        vbf = arr_gpu[:, :, bf_iy, bf_ix].permute(2, 0, 1).contiguous()
        del arr_gpu
        torch.cuda.empty_cache()
        fft = torch.fft.fft2(vbf, dim=(-2, -1))
        del vbf
        torch.cuda.empty_cache()
    measure("as_tensor + gather + fft2 + 2×empty_cache", d, args.reps)

    print()
    print("=== empty_cache() overhead in isolation ===")
    measure("empty_cache() alone", torch.cuda.empty_cache, args.reps)

    print()
    print("=== Strategy E: imagefft.clear() equivalent (1×empty_cache + cache alloc) ===")
    def e():
        # Simulate imagefft.clear(): null cache + empty_cache
        torch.cuda.empty_cache()
        # Then precompute: as_tensor H2D + gather + FFT + empty_cache
        arr_gpu = torch.as_tensor(pinned_np, device=dev)
        vbf = arr_gpu[:, :, bf_iy, bf_ix].permute(2, 0, 1).contiguous()
        del arr_gpu
        torch.cuda.empty_cache()
        fft = torch.fft.fft2(vbf, dim=(-2, -1))
        del vbf
        torch.cuda.empty_cache()
    measure("clear() + precompute (3×empty_cache total)", e, args.reps)


if __name__ == "__main__":
    main()
