# TODO: Fast-acbf plugin — deferred after fast-acbf v0.2.0 refactor

Items below were working before the v0.2.0 fast-acbf refactor and need to be
recovered in a follow-up pass.

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
