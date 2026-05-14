#!/usr/bin/env python
"""Headless FPS / VRAM benchmark for the live-acquisition path.

Loads a static 4D dataset, wraps it in a ``MockStreamer`` with optional
metadata sweeps, drives a ``LiveSolverEngine`` for N frames, and prints
latency / FPS / VRAM statistics.

Examples
--------
    python scripts/live_fps_benchmark.py path/to/4d.npy --frames 500
    python scripts/live_fps_benchmark.py path/to/4d.h5 --device cuda \\
        --max-alpha 25 --scan-step 0.2 --dk 0.05 --voltage 200 \\
        --pinned-source --rotation-sweep 0.5 --profile
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from py4d_browser_plugin.fast_acbf.calibration import electron_wavelength_angstrom
from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.live_worker import LiveSolverEngine
from py4d_browser_plugin.fast_acbf.utils import choose_device
from py4d_browser_plugin.fast_acbf.streamers import MockStreamer


def load_dataset(
    path: Path,
    *,
    key: str | None = None,
    ndims: list[int] | None = None,
    shape: tuple[int, ...] | None = None,
    offset: int | None = None,
    gap: int | None = None,
    selection=None,
    zarr_kwargs: dict | None = None,
) -> np.ndarray:
    from ptyrad.io.handlers import load_array_from_file

    arr = load_array_from_file(
        str(path),
        key=key,
        ndims=ndims if ndims is not None else [4],
        shape=shape,
        offset=offset,
        gap=gap,
        selection=selection,
        zarr_kwargs=zarr_kwargs,
    )
    return np.asarray(arr, dtype=np.float32)


def vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return None
    return None


def stage_items(stage_times: dict[str, float]):
    order = [
        "prep",
        "pinned_h2d",
        "device_bf_gather",
        "build_image_fft",
        "apply_metadata_or_update_dataset",
        "get_reconstructed_image",
        "tensor_to_numpy",
    ]
    for name in order:
        if name in stage_times:
            yield name, stage_times[name]
    for name, value in stage_times.items():
        if name not in order:
            yield name, value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="Path to a 4D dataset (.npy / .h5 / .mat / .zarr / .tif / .raw)")
    p.add_argument("--key", default=None, help="Dataset key for nested formats (.h5/.mat/.zarr)")
    p.add_argument(
        "--ndims",
        default=None,
        help="Comma-separated allowed dimensionalities for nested-format dataset filtering, e.g. '3,4' (default: '4')",
    )
    p.add_argument(
        "--shape",
        default=None,
        help="Comma-separated dataset shape (required for .raw), e.g. '128,128,256,256'",
    )
    p.add_argument("--offset", type=int, default=None, help="Byte offset for .raw")
    p.add_argument("--gap", type=int, default=None, help="Per-frame byte gap for .raw")
    p.add_argument(
        "--selection",
        default=None,
        help="JSON load-time slicing/indexing for .h5/.hdf5/.mat-v7.3/.zarr",
    )
    p.add_argument(
        "--zarr-kwargs",
        default=None,
        help="JSON kwargs passed through to zarr.open for .zarr files",
    )
    p.add_argument("--frames", type=int, default=200)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--mode", default="tcBF", choices=["tcBF", "acBF"])
    p.add_argument("--cache-mode", default="lazy", choices=["lazy", "full"])
    p.add_argument("--max-alpha", type=float, default=25.0, help="mrad")
    p.add_argument("--scan-step", type=float, default=0.2, help="Angstrom")
    p.add_argument("--dk", type=float, default=0.05, help="inverse Angstrom")
    p.add_argument("--voltage", type=float, default=200.0, help="kV")
    p.add_argument("--rotation", type=float, default=0.0, help="initial rotation_deg")
    p.add_argument("--defocus", type=float, default=0.0, help="initial C10 defocus, Angstrom")
    p.add_argument("--rotation-sweep", type=float, default=0.0, help="linear rotation step per frame, deg")
    p.add_argument("--defocus-sweep", type=float, default=0.0, help="cyclic C10 defocus amplitude, Angstrom")
    p.add_argument("--defocus-period", type=float, default=120.0, help="cyclic C10 defocus period, frames")
    p.add_argument("--display-noise", type=float, default=0.0, help="display Gaussian noise, percent image std")
    p.add_argument("--display-drift-y", type=float, default=0.0, help="display drift y, scan pixels per frame")
    p.add_argument("--display-drift-x", type=float, default=0.0, help="display drift x, scan pixels per frame")
    p.add_argument(
        "--pinned-source",
        action="store_true",
        help="Use the GUI demo pinned-source path on CUDA by filling a pinned host 4D buffer directly",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2, help="frames excluded from FPS stats")
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print per-stage timing breakdown for the first non-warmup frame",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    ndims = [int(x) for x in args.ndims.split(",")] if args.ndims else None
    shape = tuple(int(x) for x in args.shape.split(",")) if args.shape else None
    selection = json.loads(args.selection) if args.selection else None
    zarr_kwargs = json.loads(args.zarr_kwargs) if args.zarr_kwargs else None

    print(f"[load] {args.dataset}")
    data = load_dataset(
        args.dataset,
        key=args.key,
        ndims=ndims,
        shape=shape,
        offset=args.offset,
        gap=args.gap,
        selection=selection,
        zarr_kwargs=zarr_kwargs,
    )
    if data.ndim != 4:
        raise SystemExit(f"Expected 4D dataset, got shape {data.shape}")
    Ry, Rx = data.shape[:2]
    print(f"[load] shape={data.shape} dtype={data.dtype}")

    wavelength = electron_wavelength_angstrom(args.voltage)
    cfg = FastAcbfConfig(
        mode=args.mode,
        device=args.device,
        cache_mode=args.cache_mode,
        max_alpha_mrad=args.max_alpha,
        scan_step_angstrom=args.scan_step,
        dk_inv_angstrom=args.dk,
        voltage_kv=args.voltage,
        wavelength_angstrom=wavelength,
        rotation_deg=args.rotation,
        aberrations={"C10": args.defocus},
        use_calibration=False,
        use_detector_alpha=False,
    )

    base_metadata = {
        "wavelength": wavelength,
        "max_alpha": args.max_alpha,
        "dk": args.dk,
        "scan_shape": (Ry, Rx),
        "scan_step_size": args.scan_step,
        "rotation_deg": args.rotation,
        "flipud": False,
        "fliplr": False,
        "transpose": False,
        "defocus_angstrom": args.defocus,
    }
    linear_sweep = {}
    if args.rotation_sweep != 0:
        linear_sweep["rotation_deg"] = args.rotation_sweep
    cyclic_sweep = {}
    if args.defocus_sweep != 0:
        cyclic_sweep["defocus_angstrom"] = (args.defocus_sweep, args.defocus_period)

    runtime_device = choose_device(args.device)
    initial_data = data
    output_buffer = None
    pinned_source_tensor = None
    if args.pinned_source:
        if not str(runtime_device).startswith("cuda"):
            raise SystemExit("--pinned-source requires CUDA")
        import torch

        pinned_source_tensor = torch.empty(tuple(data.shape), dtype=torch.float32, pin_memory=True)
        output_buffer = pinned_source_tensor.numpy()
        np.copyto(output_buffer, data)
        initial_data = output_buffer

    print(
        f"[build] device={cfg.device} runtime_device={runtime_device} "
        f"cache_mode={cfg.cache_mode} pinned_source={bool(pinned_source_tensor is not None)}"
    )
    t_build_start = time.perf_counter()
    engine = LiveSolverEngine(
        cfg,
        initial_data,
        initial_metadata=base_metadata,
        pinned_source_tensor=pinned_source_tensor,
        drift_per_frame=(args.display_drift_y, args.display_drift_x),
        display_noise_sigma_pct=args.display_noise,
        noise_seed=args.seed,
    )
    print(f"[build] solver ready on {engine.device} in {time.perf_counter() - t_build_start:.2f}s")

    streamer = MockStreamer(
        data,
        base_metadata,
        linear_sweep=linear_sweep or None,
        cyclic_sweep=cyclic_sweep or None,
        n_frames=args.frames,
        output_buffer=output_buffer,
    )

    latencies: list[float] = []
    vram_samples: list[float] = []
    vram0 = vram_mb()
    if vram0 is not None:
        print(f"[vram] post-build: {vram0:.1f} MB")

    profiled_one = False
    for i, (ds, meta) in enumerate(streamer):
        do_profile = args.profile and not profiled_one and i >= args.warmup
        _, metrics = engine.process_one(ds, meta, profile=do_profile)
        if do_profile:
            profiled_one = True
            print("[profile] per-stage timing for one frame (CUDA-synced):")
            for name, dt in stage_items(metrics.stage_times or {}):
                print(f"  {name:36s} {dt * 1000:8.2f} ms")
            print(f"  {'TOTAL':36s} {metrics.latency_s * 1000:8.2f} ms")
        if i >= args.warmup:
            latencies.append(metrics.latency_s)
        v = vram_mb()
        if v is not None:
            vram_samples.append(v)
        if (i + 1) % 50 == 0 or i == 0:
            v_str = f" vram={v:.1f}MB" if v is not None else ""
            print(f"[frame {i + 1}/{args.frames}] latency={metrics.latency_s * 1000:.2f}ms{v_str}")

    if not latencies:
        print("No frames measured (warmup >= frames?)", file=sys.stderr)
        return 1

    mean_lat_ms = statistics.mean(latencies) * 1000
    p50_lat_ms = statistics.median(latencies) * 1000
    p95_lat_ms = sorted(latencies)[int(0.95 * len(latencies))] * 1000
    fps_mean = 1.0 / statistics.mean(latencies)

    print()
    print(f"frames measured: {len(latencies)} (warmup {args.warmup})")
    print(f"latency mean / p50 / p95: {mean_lat_ms:.2f} / {p50_lat_ms:.2f} / {p95_lat_ms:.2f} ms")
    print(f"throughput (mean): {fps_mean:.1f} FPS")
    if vram_samples:
        print(f"VRAM peak: {max(vram_samples):.1f} MB")
        print(f"VRAM delta (last - first): {vram_samples[-1] - vram_samples[0]:+.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
