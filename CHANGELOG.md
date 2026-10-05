# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Advanced Dashboard: in Depth 3D, each image pane's *Export* menu offers *TIFF stack* to
  save every slice of the reconstruction, probe or χ as one multi-page float32 TIFF. The
  pixel size, slice step and the C10 of each slice go in the metadata. The entry is
  greyed out in 2D (ported from ptydy 0.1.1)
- Advanced Dashboard: export file names say what was saved (`tcbf`, `acbf`, `probe_amp`,
  `probe_int`, `probe_cplx`, `chi`, plus `_stack` for a TIFF stack)
### Fixed
- Switching a dashboard pane to a complex image with a complex level stack (Show ψ in 3D
  with one B/C range for the stack) briefly took the levels of the previous real image
  from the stack cast to real (a `ComplexWarning`). A real image now keeps its own
  range, and `stack_levels` refuses a complex stack (ported from ptydy 0.1.1)
- Tests: a `ComplexWarning` is an error in the suite (`[tool.pytest.ini_options]`)

## [0.6.0] - 2026-10-03
### Added
- **Advanced Dashboard views ported from ptydy's fast-acbf app** (same layout and behaviour,
  PySide6 -> PyQt5): Mode / Frame, **Show** (probe |ψ|, |ψ|², complex ψ with hue = phase,
  the aberration surface χ in gray modulo 2π, ∇χ and the vBF image shifts as arrows, with
  an *Axes* choice of output frame or detector grid), and **Depth** 2D / 3D: a cached stack
  of reconstructions, probes, χ and shifts over C10 (*Slices*, *Step*, *Compute stack*), a
  slice slider and an **Ortho View** (xy slice with xz / yz sections through a point;
  drag the round handles). Every view has its own controls underneath (scaling, colour
  map, Auto percentiles, histogram, statistics, cursor readout, Fit / Copy / Export), a
  one-range-for-the-stack or per-slice *B/C Range* in 3D, and *Pipeline* (Ctrl+B) /
  *Controls* toggles in the top row
- `DepthStackJob` and the worker's `optics_diagnostics` / `depth_stack`: complex probe
  with its real-space pixel size, χ over the bright-field disk and the vBF shifts with
  their detector pixels go with every result
### Changed
- The probe view's scale bar uses the probe's real-space pixel (1 / (N dk)), not the
  reconstruction's
- Depth-stack results update only the dashboard (no viewer update, no history row), and
  finished results no longer keep the solver (it stays in the job state only)

## [0.5.0] - 2026-07-31
### Added
- Add a **Live View** dock wired to fast-acbf's live-acquisition path, with calibration synced from the active data source, auto-refinement controls, and result scaling forced to linear
- Add a standalone PyInstaller spec (`fast_acbf.spec`) for building a macOS `py4DGUI.app`, hardened to exclude unwanted plugins and disable UPX
- Add **Refine Scan Rotation** and **Refine Defocus** actions to the Simple Menu using fast-acbf's Brent adaptive line search, a coarse `max`-search defocus button, and `lite_seeded_defocus_fraction` to automatically narrow the defocus search range once orientation is optimized
- Add an acBF upscale option, synced tcBF/acBF upscale spinboxes, and better RBF auto-detection logic
- Add a direct C10 editor/readout and a merged +/- defocus step widget to the Simple Menu toolbar
- Add calibration sync triggers so `QR_rotation`/`QR_flip` follow the active datacube calibration, and pass `pixel_size`/`pixel_units` (proper Å symbol) to all `set_virtual_image` calls
### Changed
- Rename the "Lite" workflow taskbar to **Simple Menu** and "Interactive Dashboard" to **Advanced Dashboard**; rename the Optics tabs to **Aberrations**
- Move aberration reset into the tcBF menu and make the zero button reset aberrations directly; replace the C10 stepper with the direct editor
- Rename "Force Overfocus" to **Focus Sign** and standardize its defaults; make calibration-free alpha resolution safe
- Auto-derive the fine rotation half-width from the joint orientation grid search instead of a fixed default; make the px-mode defocus search range adaptive (`max(20, 0.2*Npix)`)
- Update the upscale method to match fast-acbf 0.6.0's `zero_insert` default for tcBF, and gate the `zero_insert` warning so it doesn't fire when `upscale=1`
- Remove pre-fill values now covered by py4D-browser v1.5.1's own calibration dialog; bump the minimum `py4d-browser` requirement
### Removed
- Remove the "Live Demo" dialog, superseded by Live View and the standalone `py4D-browser-dataset-streamer` plugin; the temporary dataset-streamer dock built while developing Live View was likewise removed and split into that standalone plugin
- Remove the "Quick Run" button, the redundant dock title bar, the Simple Menu settings dialog, and a stale twin-image-ambiguity aberration-refinement attempt
### Fixed
- Reset fast-acbf state on dataset change; stabilize the Simple Menu toolbar lifecycle; restore Live View dock cleanup
- Canonicalize and preserve D4 orientation (rotation/flips/transpose) between the plugin UI and py4D-browser calibration
- Fix a voltage inconsistency introduced by an earlier rebase and a coarse-defocus method that changed unexpectedly

## [0.4.0] - 2026-05-25
### Changed
- Update the upscale method to match fast-acbf 0.6.0's `zero_insert` default
- Add frame interval control to `LiveDemoDialog` and update related tests

## [0.3.0] - 2026-05-22
### Added
- Enable `upscale`, `upscale_methods`, `pad_width`, and finer controls over the refinement methods, tracking fast-acbf 0.5.0's `BFPreparer`

## [0.2.0] - 2026-05-19
### Added
- Add `LiveBFSolver.update_dataset()` for per-frame dataset swap with a staging buffer and GPU mask path (no `empty_cache`)
- Consolidate the live-acquisition modules into a `live/` subpackage; replace the string-dispatch runner with a typed `SolverJob` protocol
### Changed
- Recover plugin compatibility with the fast-acbf v0.3.0 data-layer rewrite and the v0.4.0 `pipeline` API (`cache_mode` renamed to `pipeline`)
- Remove plugin-side PACBED-max normalization; delegate to `BFSolver(normalize=True)` now that fast-acbf supports it natively
- Update `mask_path_profile.py` for the refactored fast-acbf API; update `TODO.md` to mark the v0.4.0 recovery items done

## [0.1.2] - 2026-05-15
### Changed
- Internal refactor: split `utils.py` out with shared helpers, split calibration helpers out of `config.py` into `calibration.py`, split `LiveSolverEngine` out of `live_worker.py` into `live_engine.py`, and split `dialogs.py` into a package with shared widgets
- Add temporary plugin-side PACBED-max normalization before solver build
- Remove dead tests left over by the module split

## [0.1.1] - 2026-05-11
### Added
- Add an Output frame selector to the Advanced Dashboard; the probe-amplitude preview now follows the same output frame selection
### Changed
- Rename "Update and Preview" to **Update Preview** and emphasize/auto-focus it while navigating the dashboard
- Automatically refresh the image/probe preview on mode and output-frame changes
- Put initial focus on **Edit Calibration** when the dashboard opens

## [0.1.0] - 2026-05-10
### Added
- Add live-acquisition support: `MockStreamer` for profiling, `LiveSolverEngine`/`LiveSolverWorker`, `MetadataAdapter` for live metadata diffing, live mode dashboard controls, and a headless FPS/VRAM benchmark script
- Add an end-to-end live-acquisition integration test on CPU and profiling support for live processing tests
### Changed
- Reuse the cached solver across same-shape datasets via `update_dataset` instead of rebuilding; use `BFSolver.apply_metadata` for orientation updates
- Release the cached solver before entering live mode
### Fixed
- Fix pinned source and chunked Poisson noise in the live demo path

## [0.0.1] - 2026-05-09
### Added
- Initial release: a roughly working py4D-browser plugin UI wired to fast-acbf, with two-way sync between preview mode and display mode
