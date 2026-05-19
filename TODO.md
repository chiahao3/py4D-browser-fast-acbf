# TODO: Fast-acbf plugin — deferred items

Items that were working before the fast-acbf v0.2.0 refactor and have not yet
been recovered.  Fixed items are noted at the top.

## Fixed (v0.4.0 recovery pass — 2026-05-18)

- [x] **Solver build crash** (`cache_mode` → `pipeline`): `BFSolver.__init__`
  replaced `cache_mode` with `pipeline` in v0.4.0.  Fixed by renaming the
  config field and updating `build_solver()` and the Configuration dialog.
- [x] **Plugin-side normalization**: `BFSolver(normalize=True)` is now
  supported in v0.4.0.  Removed the temporary `prepare_solver_dataset` /
  `normalize_dataset_by_pacbed_max` helpers from `utils.py`.
- [x] **Live dataset update per frame**: Implemented `LiveBFSolver.update_dataset()`
  in `live/solver.py`.  Per-frame dataset swap replaces `Dataset4D._array`,
  rebuilds `BFExtractor`, and clears the `ImageFFT` cache without touching
  geometry or aberration state.  `live/engine.py` calls it on every frame.
  CUDA-pinned host buffers flow through naturally (no copy if already float32
  contiguous).

## Still deferred

- [ ] **Live dataset update per frame**: `BFSolver.update_dataset()` was removed
  in fast-acbf v0.2.0. `LiveSolverEngine.process_one()` currently reconstructs
  from the initial dataset frozen at solver build time — new incoming frames
  do not update the solver's data. Fix options:
  - Add `update_dataset()` back to `BFSolver` in the fast-acbf repo, or
  - Rebuild a lightweight solver wrapper inside this plugin that owns dataset
    mutation directly.

- [ ] **Live heavy-metadata update** (wavelength, max_alpha, dk, scan_shape,
  scan_step_size): `BFSolver.apply_metadata()` was removed. These physics
  changes are currently logged as warnings and silently skipped. Recovery
  depends on either restoring `apply_metadata` in fast-acbf or implementing
  a full solver-rebuild path in `LiveSolverEngine._rebuild_for_heavy_delta()`.

- [ ] **Pinned-source live fast path**: The GPU fast-path for live acquisition
  depended on internal `BFSolver` attributes that no longer exist in v0.2.0:
  - `solver.vbf_images` (as a writable attribute)
  - `solver._bf_mask_bool_d`
  - `solver._dataset_device_staging_4d`
  - `solver._dataset_pinned_buffer_4d`
  - `solver._image_fft`
  - `fast_acbf.pipeline.build_image_fft`

  `LiveSolverEngine._update_from_pinned_source()` is stubbed with
  `NotImplementedError`; `_can_use_pinned_source()` always returns `False`.
  The `pinned_source_tensor` constructor argument is accepted but ignored with
  a log warning. Recovery requires either re-exposing the necessary internals
  in the fast-acbf public API or rewriting the fast path against the new
  `ImageFFTProvider` / `BFReconstructor` internals.
