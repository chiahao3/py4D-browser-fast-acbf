# Live Acquisition: Status & Plan to GUI Demo

## What this document is

A snapshot of where the live-acquisition path stands today (post the
plugin polish + headless work + fast-acbf Path B refactor) and a concrete
step-by-step plan to land an in-GUI demo that visualizes frames refreshing
inside py4D-browser.

---

## Status — done so far

### Plugin polish (`py4D-browser-fast-acbf` `live` branch)

In response to the fast-acbf cache-split refactor:

* `apply_config_to_solver` now uses `BFSolver.apply_metadata` for orientation
  keys (flipud / fliplr / transpose / rotation_deg) instead of mutating
  `solver.coord_transform` and calling `set_rotation_deg` directly. The
  dispatcher owns cache invalidation.
* `solver_signature` no longer pins on `id(datacube_data)`; same-shape data
  swaps reuse the cached solver via the new `update_dataset` Tier-1 hot path.
  This is the bridge that makes live mode possible at all — without it,
  every new frame would rebuild the solver from scratch.
* Empty `fast_acbf_plugin/` placeholder folder removed.

### Headless live-acquisition path (plugin)

* [`src/py4d_browser_plugin/fast_acbf/metadata.py`](src/py4d_browser_plugin/fast_acbf/metadata.py)
  — `MetadataAdapter` diffs incoming metadata dicts against the last known
  state, emits only changed keys, rounded to the same precision as
  `solver_signature` so sub-precision metadata changes do not trigger no-op
  invalidations. Vendor-agnostic.
* [`src/py4d_browser_plugin/fast_acbf/live_worker.py`](src/py4d_browser_plugin/fast_acbf/live_worker.py)
  — `LiveSolverEngine` is the non-Qt core that owns one `BFSolver` plus a
  `MetadataAdapter`, exposing `process_one(dataset, metadata, profile=False)`
  for headless callers. `LiveSolverWorker(QThread)` wraps it with a
  drop-oldest single-slot queue and `frame_ready` / `started_ready` /
  `error` PyQt signals.
* [`src/py4d_browser_plugin/fast_acbf/streamers/mock.py`](src/py4d_browser_plugin/fast_acbf/streamers/mock.py)
  — `MockStreamer` iterates a static 4D array, optionally applying linear or
  cyclic metadata sweeps per frame. Bounded via `n_frames`.
* [`scripts/live_fps_benchmark.py`](scripts/live_fps_benchmark.py)
  — headless FPS / VRAM benchmark that wires `MockStreamer` →
  `LiveSolverEngine`, prints latency mean/p50/p95, FPS, peak VRAM and VRAM
  delta. Loads via `ptyrad.io.handlers.load_array_from_file` so .npy / .h5
  / .mat / .zarr / .tif / .raw all work. It supports the GUI demo's pinned
  source, rotation sweep, defocus sweep, display drift, display noise, and
  `--profile` stage timing.
* Test coverage: 35 tests passing across metadata, mock streamer, live
  controller, live worker, config, dialog, and worker paths.

### fast-acbf Path B refactor (`fast-acbf` `live` branch)

The headless benchmark exposed a 5× speedup waiting in `fast-acbf`'s
`extract_vbf_stack` — the per-frame CPU-side `dataset[:, :, bf_mask_bool]`
fancy index thrashes cache when the dataset is large, bottlenecking at
~2 GB/s. Confirmed by [`scripts/mask_path_profile.py`](scripts/mask_path_profile.py)
on the user's 128⁴ float32 dataset:

* Path A (current host masking): 486 ms / frame
* Path B (large H2D + device masking): 92 ms / frame
* Speedup: **5.3×**

Landed in fast-acbf commit `a640785` ("Switch live update_dataset to
device-side BF mask gather"):

* New `extract_vbf_stack_via_device_mask` in
  `src/fast_acbf/pipeline.py`.
* `BFSolver.update_dataset` routes through it on CUDA; CPU/MPS keep
  Path A.
* Solver state gains `_bf_mask_bool_d`, `_dataset_pinned_buffer_4d`,
  `_dataset_device_staging_4d` (lazy-allocated on first
  `update_dataset` — non-live users pay nothing).
* The 4D buffers are dropped on Tier-3 BF-geometry / scan-shape
  changes so they reallocate at the right shape next frame.
* All 159 CPU + 25 CUDA tests pass (+2 new CUDA tests for buffer reuse
  and Tier-3 invalidation).

### Measured end-to-end on 128×128×128×128 float32 (~1 GB) CUDA

|              | Before    | After      |
|--------------|-----------|------------|
| `update_dataset` | 474.83 ms | 95.62 ms |
| tcBF compute     | 1.52 ms   | 1.37 ms  |
| **Total / frame**| **477 ms**| **97 ms**|
| **Throughput**   | **2.1 FPS** | **10.6 FPS** |
| VRAM             | 140 MB    | 1164 MB (+1 GB staging) |
| VRAM drift over 100 frames | 0 | -0.4 MB (measurement jitter) |

---

## Headroom remaining (not yet implemented)

These are documented for completeness; none are required for the GUI demo.

1. **Vendor adapter writes directly into the pinned host buffer.** Removes
   the ~49 ms `host_to_pinned_large` memcpy step. Brings us to ~22 FPS at
   this dataset size.
2. **Double-buffered CUDA streams.** Overlap the next frame's H2D with the
   current frame's tcBF compute. Could add another ~1.5–2× in steady state.
3. **Vendor-specific metadata adapters.** `MetadataAdapter` is
   vendor-agnostic; concrete adapters that translate Thermo Fisher / Gatan
   metadata to the canonical dict still need to be written when real
   hardware comes online.

---

## Current GUI demo workflow

Goal: open py4D-browser → load a 4D datacube → open **Live Demo** → click
"Start Live" → watch the isolated reconstruction panel refresh while
`MockStreamer` re-streams the loaded datacube through the same pinned-source
transfer and reconstruction path used by the headless benchmark. "Stop Live"
cleanly tears everything down.

Step-by-step user instructions:

```text
1. py4dgui  → File → Load datacube
2. Plugins → fast-acbf → Live Demo
3. Set physics (or trust auto-calibration)
4. In the Live Demo group:
     - Source: current datacube (mock streamer)
     - Rotation sweep: 0.5 deg/frame
     - Frames: 0 (unbounded)
   Click "Start Live"
5. The reconstruction panel refreshes as reconstructed frames complete; the FPS
   readout and timing rundown update in real time.
6. Click "Stop Live" to end.
```

Plus a screenshot or short GIF in `assets/`.

---

## What we are deliberately deferring

* **Vendor adapter implementations.** Mock streamer is enough for the demo;
  real Thermo Fisher / Gatan adapters are downstream work and need
  hardware on hand to validate.
* **GUI calibration overrides during live mode.** For the demo, calibration
  is read once at "Start Live" and held constant; live editing of physics
  fields can come in a follow-up.
* **Pause / scrub / save-frame controls.** Nice-to-have; not required to
  prove the loop works visually.
* **Double-buffered CUDA streams.** Listed under headroom; the pinned-source
  demo already exercises the main transfer/reconstruction cost directly.

---

## Implementation update — GUI demo landed

The implementation followed the commit split above with one adjustment:
live demo controls were moved into a separate **Live Demo** window instead of
remaining inside the Interactive Dashboard. The dashboard is back to the
calibration / preview / refinement workflow; the plugin owns datacube
validation and starts/stops the isolated live session.

Landed pieces:

* `FastAcbfDashboard` is restored to the original refinement-oriented layout.
* `LiveDemoDialog` owns source, mode, pinned-source, rotation sweep, defocus
  sweep, display drift, display Gaussian noise, finite/unbounded frame count,
  Start/Stop, and live FPS/latency/status timing.
* `live_controller.py` creates a `LiveSession` from the resolved config,
  current datacube, `MockStreamer`, `LiveSolverWorker`, and producer thread.
* On CUDA, the demo can allocate a pinned source buffer and let the mock
  streamer fill it directly; the live worker then uploads that buffer without
  the normal NumPy-to-pinned copy.
* Rotation sweep updates `rotation_deg` linearly every frame, forcing the same
  basis-cache rebuild path that rotation metadata changes use in live mode.
* Defocus sweep updates C10 cyclically through live metadata, forcing a
  reconstruction-basis cache rebuild without changing the 1 GB source frame.
* Slow y/x display drift is simulated by rolling the reconstructed scan image
  by a linear per-frame offset.
* Display Gaussian noise is a visual-only effect applied after reconstruction.
* The plugin wires Live Demo signals to session lifecycle, blocks preview and
  refinement while live mode is active, auto-stops on datacube changes and
  close, and surfaces errors through `QMessageBox`.
* Live frames update the Live Demo reconstruction `ImageView`; the first frame
  auto-scales, then subsequent frames keep levels/range fixed.
* `tests/test_live_controller.py` covers metadata sweep translation and a
  finite three-frame CPU live session.
* `README.md` now includes the live mock-acquisition demo recipe.
