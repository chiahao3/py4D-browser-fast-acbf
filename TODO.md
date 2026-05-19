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
  in `live/solver.py`.  Uses a pre-allocated GPU staging buffer + `copy_()` (DMA from
  pinned host memory) + GPU bool-mask extraction + `fft2`, writing directly into
  `imagefft._cache` without calling `imagefft.clear()`.  Avoids `cuda.empty_cache()`
  entirely: freed tensors stay in PyTorch's caching allocator for O(1) reuse next
  frame.  Measured ~43 ms for a 1 GB pinned dataset on an RTX 5000 Ada
  (hardware DMA limit; see `scripts/test_h2d_latency.py`).

## Still deferred

- [ ] **Live heavy-metadata update** (wavelength, max_alpha, dk, scan_shape,
  scan_step_size): `BFSolver.apply_metadata()` was removed. These physics
  changes are currently logged as warnings and silently skipped. Recovery
  depends on either restoring `apply_metadata` in fast-acbf or implementing
  a full solver-rebuild path in `LiveSolverEngine._rebuild_for_heavy_delta()`.

- [ ] **Redundant contiguity check for pinned source**: `process_one()` calls
  `np.ascontiguousarray(np.asarray(dataset, dtype=np.float32))` unconditionally.
  When `dataset` is already the pinned numpy view (float32 contiguous), this is a
  no-op but still touches every element for the dtype/contiguity check.  A future
  improvement would detect that `dataset is engine._pinned_source_np` and skip
  the check before passing it to `update_dataset()`.
