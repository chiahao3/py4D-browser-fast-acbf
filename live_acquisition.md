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
  `solver_signature` so sub-precision jitter does not trigger no-op
  invalidations. Vendor-agnostic.
* [`src/py4d_browser_plugin/fast_acbf/live_worker.py`](src/py4d_browser_plugin/fast_acbf/live_worker.py)
  — `LiveSolverEngine` is the non-Qt core that owns one `BFSolver` plus a
  `MetadataAdapter`, exposing `process_one(dataset, metadata, profile=False)`
  for headless callers. `LiveSolverWorker(QThread)` wraps it with a
  drop-oldest single-slot queue and `frame_ready` / `started_ready` /
  `error` PyQt signals.
* [`src/py4d_browser_plugin/fast_acbf/streamers/mock.py`](src/py4d_browser_plugin/fast_acbf/streamers/mock.py)
  — `MockStreamer` iterates a static 4D array, optionally jittering selected
  metadata keys per frame. Reproducible via seed; bounded via `n_frames`.
* [`scripts/live_fps_benchmark.py`](scripts/live_fps_benchmark.py)
  — headless FPS / VRAM benchmark that wires `MockStreamer` →
  `LiveSolverEngine`, prints latency mean/p50/p95, FPS, peak VRAM and VRAM
  delta. Loads via `ptyrad.io.handlers.load_array_from_file` so .npy / .h5
  / .mat / .zarr / .tif / .raw all work. Has a `--profile` flag that breaks
  one frame's latency into prep / apply_metadata / get_reconstructed_image
  / tensor_to_numpy stages with proper CUDA syncs.
* Test coverage: 31 tests passing (12 metadata + 7 mock streamer + 2 live
  worker integration on CPU + the existing config / dialog / worker tests).

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
| VRAM drift over 100 frames | 0 | -0.4 MB (noise) |

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

## Plan: GUI demo with visible frame refreshing

Goal: open py4D-browser → load a 4D datacube → open the fast-acbf dashboard
→ click "Start Live" → watch the reconstruction panel refresh as
`MockStreamer` re-streams the loaded datacube with metadata jitter, with a
visible FPS / latency readout. "Stop Live" cleanly tears everything down.

**Architectural choices already made:**

* Live mode lives **inside the existing dashboard** (`FastAcbfDashboard` in
  [dialogs.py](src/py4d_browser_plugin/fast_acbf/dialogs.py)). Reuses
  `image_view` as the live canvas — no second window.
* The existing one-shot `FastAcbfRunner` path stays alongside; live mode is
  a separate worker (`LiveSolverWorker`) whose lifecycle is managed by the
  dashboard. They never coexist on the same datacube.

### Step 1 — Live-mode group in the dashboard UI (~30 lines)

In [dialogs.py `FastAcbfDashboard._build_ui`](src/py4d_browser_plugin/fast_acbf/dialogs.py#L323),
add a new `QGroupBox("Live Mode")` below the orientation tab containing:

* **Source:** `QComboBox` with one option for now — `"current datacube
  (mock streamer)"`. Future detector backends drop in here.
* **Jitter (scan rotation, deg σ):** `QDoubleSpinBox`, default 0.5.
* **Jitter (scan step, Å σ):** `QDoubleSpinBox`, default 0.0.
* **Frames:** `QSpinBox` (0 = unbounded), default 0.
* **Start Live / Stop Live:** toggle `QPushButton`.
* **Live status:** `QLabel` for FPS / latency text — updated from
  `frame_ready`.

Disable refinement and "Update and Preview" buttons while live. The
existing `set_status` already wires the bottom-bar status; reuse it.

### Step 2 — Wire start/stop to `LiveSolverWorker` (~80 lines)

New file `src/py4d_browser_plugin/fast_acbf/live_controller.py`:

* `LiveSession` — bundle of `(worker: LiveSolverWorker, streamer_thread:
  QThread, base_metadata: dict)`. Single-instance per dashboard.
* `start_live(parent_plugin, config, datacube_data, ui_jitter)`:
  1. Build `base_metadata` from the resolved `FastAcbfConfig` (the
     same translation `resolved_for(parent)` does today, plus
     `scan_shape = datacube_data.shape[:2]`).
  2. Construct `LiveSolverWorker(cfg=config, initial_dataset=datacube_data,
     initial_metadata=base_metadata)`.
  3. Construct `MockStreamer(datacube_data, base_metadata, jitter=ui_jitter,
     n_frames=...)`.
  4. Spin a `QThread`-hosted producer that pulls from the streamer and
     calls `worker.submit(dataset, metadata)` at the streamer's natural
     rate (no sleep — `submit` is drop-oldest).
* `stop_live(session)`: signal both threads to wind down, `wait()`,
  release.

### Step 3 — Display the live frames in `image_view` (~30 lines)

In `FastAcbfDashboard`, connect `worker.frame_ready` to a slot that:

```python
@pyqtSlot(np.ndarray, dict)
def _on_live_frame(self, image: np.ndarray, metrics: dict) -> None:
    self.image_view.setImage(
        image.T, autoLevels=False, autoRange=False, autoHistogramRange=False,
    )
    self.live_status_label.setText(
        f"FPS {metrics['fps']:.1f}   latency {metrics['latency_s'] * 1000:.1f} ms"
    )
```

Critically: `autoLevels=False` + `autoRange=False` so pyqtgraph does not
re-fit the colormap or zoom on every frame (which would make the panel
look like it's flashing rather than refreshing smoothly). Run a single
warmup frame with `autoLevels=True` to set the initial range, then lock
it for the rest of the session.

### Step 4 — Lifecycle & error handling (~30 lines)

* `worker.error.connect(self._on_live_error)` — surface tracebacks via
  `QMessageBox` and auto-stop.
* On dashboard `closeEvent`, if a session is active, call `stop_live`
  before accepting the close — otherwise the QThread leaks.
* Disable refinement / apply buttons in `_set_live_active(True/False)`.

### Step 5 — One end-to-end test (~50 lines)

`tests/test_live_controller.py`: build a small datacube, start a session
on CPU with `n_frames=3`, assert that exactly 3 `frame_ready` emissions
arrive with images of the expected shape, and that `stop_live` joins
both threads within a timeout.

### Step 6 — README demo recipe (~20 lines added to `README.md`)

Step-by-step user instructions for the demo:

```text
1. py4dgui  → File → Load datacube
2. Plugins → fast-acbf → Interactive Dashboard
3. Set physics (or trust auto-calibration)
4. In the Live Mode group:
     - Source: current datacube (mock streamer)
     - Jitter scan rotation: 0.5 deg
     - Frames: 0 (unbounded)
   Click "Start Live"
5. The reconstruction panel refreshes at the streamer's pace; the FPS
   readout updates in real time.
6. Click "Stop Live" to end.
```

Plus a screenshot or short GIF in `assets/`.

---

## Suggested commit split

* commit A — Live Mode UI group in the dashboard (steps 1, 4)
* commit B — `live_controller.py` + worker lifecycle wiring (steps 2, 3)
* commit C — Test + README demo recipe (steps 5, 6)

Each is small, independently verifiable, and the final commit is the
user-visible demo.

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
* **Double-buffered CUDA streams.** Listed under headroom; the demo at
  ~10 FPS on 128⁴ data is already convincing.

---

## Implementation update — GUI demo landed

The implementation followed the commit split above with one adjustment:
live demo controls were moved into a separate **Live Demo** window instead of
remaining inside the Interactive Dashboard. The dashboard is back to the
calibration / preview / refinement workflow; the plugin owns datacube
validation and starts/stops the isolated live session.

Landed pieces:

* `FastAcbfDashboard` is restored to the original refinement-oriented layout.
* `LiveDemoDialog` owns source, mode, pinned-source, jitter, Poisson noise,
  drift, finite/unbounded frame count, Start/Stop, and live FPS/latency status.
* `live_controller.py` creates a `LiveSession` from the resolved config,
  current datacube, `MockStreamer`, `LiveSolverWorker`, and producer thread.
* On CUDA, the demo can allocate a pinned source buffer and let the mock
  streamer fill it directly; the live worker then uploads that buffer without
  the normal NumPy-to-pinned copy.
* Poisson noise is available as
  `poisson(max(dataset, 0) * counts_scale) / counts_scale`.
* Slow y/x scan drift is simulated by rolling the reconstructed scan image by
  a linear per-frame offset.
* The plugin wires Live Demo signals to session lifecycle, blocks preview and
  refinement while live mode is active, auto-stops on datacube changes and
  close, and surfaces errors through `QMessageBox`.
* Live frames update the Live Demo reconstruction `ImageView`; the first frame
  auto-scales, then subsequent frames keep levels/range fixed.
* `tests/test_live_controller.py` covers metadata/jitter translation and a
  finite three-frame CPU live session.
* `README.md` now includes the live mock-acquisition demo recipe.
