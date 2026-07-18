# TODO: Fast-acbf plugin — deferred items

Items that were working before the fast-acbf v0.2.0 refactor and have not yet
been recovered.  Fixed items are noted at the top.

## Architecture note

The plugin maintains two parallel solver roles:

* **`BFSolver`** (core fast-acbf) — time-insensitive interactive workflows:
  Advanced Dashboard and Jupyter notebooks.  Dataset is fixed for a solver
  lifetime; the API optimises for reconstruction quality and ergonomics.

* **`LiveBFSolver`** (`live_view/solver.py`) — frame-rate-sensitive live acquisition.
  Owns the per-frame data path and ImageFFT cache management.  Currently wraps
  a `BFSolver` instance and accesses its private components directly.
  fast-acbf 0.4.0 already exposes `Dataset4D`, `ImageFFT`, `BFReconstructor`,
  `PipelineManager`, `DetectorGeometry`, `ScanGeometry`, `AberrationState`, and
  `CoordinateTransform` as public classes; the private-attr access is a
  plugin-side assembly gap, not an upstream API gap.

---

## Fixed (v0.4.0 recovery pass — 2026-05-18)

- [x] **Solver build crash** (`cache_mode` → `pipeline`): `BFSolver.__init__`
  replaced `cache_mode` with `pipeline` in v0.4.0.  Fixed by renaming the
  config field and updating `build_solver()` and the Configuration dialog.
- [x] **Plugin-side normalization**: `BFSolver(normalize=True)` is now
  supported in v0.4.0.  Removed the temporary `prepare_solver_dataset` /
  `normalize_dataset_by_pacbed_max` helpers from `utils.py`.
- [x] **Live dataset update per frame**: `LiveBFSolver.update_dataset()` owns
  the full GPU data path: pre-allocated staging buffer + `copy_()` DMA from
  pinned host + GPU bool-mask extraction + `fft2`, writing directly into
  `imagefft._cache` without calling `imagefft.clear()`.  Avoids
  `cuda.empty_cache()` entirely; freed tensors recycle inside PyTorch's caching
  allocator at O(1) cost.  Measured ~43 ms / frame for a 1 GB pinned dataset
  on an RTX 5000 Ada (hardware DMA limit; see `scripts/test_h2d_latency.py`).

## Still deferred

- [ ] **Live heavy-metadata update** (wavelength, max_alpha, dk, scan_shape,
  scan_step_size): These physics changes are currently logged as warnings and
  silently skipped in `LiveSolverEngine.process_one()`.  The correct fix is a
  full `LiveBFSolver` rebuild triggered from `_rebuild_for_heavy_delta()`, but
  that requires either a clean construction path for `LiveBFSolver` that doesn't
  go through `BFSolver.__init__`, or a fast-acbf API for rebuilding geometry
  without touching the aberration/basis state.

- [ ] **`LiveBFSolver` direct component assembly** (plugin-side refactor):
  `LiveBFSolver` currently wraps a `BFSolver` instance and reads its private
  attributes (`_dataset`, `_pipeline_manager`, `_recon`).  fast-acbf 0.4.0
  already exposes all required classes publicly.  The refactor is to
  assemble `Dataset4D` → `PipelineManager` → `ImageFFT` → `BFReconstructor`
  directly inside `LiveBFSolver.__init__()`, bypassing `BFSolver` entirely.
  This also unblocks clean heavy-metadata rebuilds (see item above) because
  individual components can be replaced without re-running `BFSolver.__init__`.

- [ ] **Redundant contiguity check for pinned source**: `process_one()` calls
  `np.ascontiguousarray(np.asarray(dataset, dtype=np.float32))` unconditionally.
  When `dataset` is already the pinned numpy view (float32 C-contiguous), this
  touches every element for the dtype/contiguity check.  A future improvement
  would detect that `dataset` is the pinned view and skip the check.
