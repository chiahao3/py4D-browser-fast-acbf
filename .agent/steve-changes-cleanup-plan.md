# Cleanup Plan for Steve's Changes

## Scope

Clean up the changes introduced between:

- Baseline: `7ea24131fee872cf6bdc3c1ec4f27ea0f298247f`
- Steve's tip: `a32d47eba9b2272923cd2b4107587b6ce7c2d72a`

The goal is to retain the useful product changes—especially the Simple Menu,
defocus controls, orientation/calibration synchronization, calibrated output
metadata, and macOS packaging—while resolving regressions and making the new
behavior explicit and tested.

This plan does not prescribe reverting the feature direction. It separates
merge-blocking correctness work from UI cleanup, documentation, and packaging
hardening.

## Decisions Incorporated

The following decisions are fixed for this cleanup and should not be reopened
during implementation unless new evidence shows that one is unsafe:

1. Fix existing tests in a reasonable dependency order, then add focused tests
   for every new behavior retained by the cleanup.
2. Implement the work as small, digestible commits on the
   `simple-toolbar-cleanup` branch.
3. Fixing the calibration-free tcBF branch is required.
4. Preserve the current user-visible functionality and behavior of
   `use_detector_alpha` and `max_alpha_px` for now. Characterize it with tests;
   defer any semantic redesign to a later change.
5. Fix toolbar/widget lifecycle and ownership without changing the normal
   simple-toolbar usage or workflow.
6. Fix the incorrect Live View cleanup call.
7. Standardize every default and reset occurrence of `focus_sign` to `"none"`.
8. Update rotation-refinement tooltips and documentation to describe the
   current behavior; do not redesign the refinement strategy in this cleanup.
9. Label the manual defocus readout `C10 (-df)` and its step control
   `C10 (-df) step`.
10. Make orientation/calibration synchronization D4-aware. Convert the plugin's
    `(flipud, fliplr, transpose, rotation_deg)` representation to py4D's
    canonical continuous scan rotation plus transpose/chirality representation.

## Desired End State

1. Calibration-free tcBF and orientation workflows work with unset voltage and
   pixel-unit calibration.
2. The currently retained BF-disk-radius behavior is explicitly characterized
   across calibrated and calibration-free datasets, with semantic refinement
   deferred to follow-up work.
3. The Simple Menu has clear Qt ownership, visibility, close, and disposal
   behavior.
4. Live View continues to use `QDockWidget` lifecycle APIs.
5. Refinement defaults and methods are intentional, visible to users where
   appropriate, and consistent with tooltips and tests.
6. Output images retain the new physical pixel-size metadata.
7. The macOS build path is documented, reproducible, and does not unnecessarily
   hardcode the Python minor version.
8. The full test suite passes without hanging or leaving Qt workers alive.
9. The branch history is reviewable as a sequence of focused commits, each with
   its directly relevant tests.

## Test-First and Commit Strategy

Tests should be repaired and added alongside the behavior they cover, not in one
large test-only commit at the end. Use the following order:

1. Restore the currently failing test baseline where the intended behavior is
   already decided:
   - `focus_sign="none"` everywhere.
   - Simple Menu tests target the new toolbar interface, not the removed
     `QDockWidget` interface.
   - `max_alpha_px` tests describe the behavior being preserved for now.
2. Add failing regression tests for the calibration-free exceptions, then fix
   them.
3. Add lifecycle tests, then fix the Simple Menu/widget ownership.
4. Add Live View removal/restart tests, then fix its cleanup API.
5. Add D4 orientation-conversion tests, then implement canonical
   synchronization.
6. Add tests for new toolbar controls, labels, status messages, refinement
   methods, and output metadata.
7. Finish with documentation/tooltips and full-suite shutdown validation.

Each commit should contain one coherent behavior change plus its tests. Avoid
mixing calibration logic, Qt lifecycle work, orientation algebra, and packaging
changes in the same commit.

Suggested commit sequence:

1. `fix: standardize focus sign defaults`
2. `test: align coverage with the Simple Menu contract`
3. `fix: make calibration-free alpha resolution safe without voltage`
4. `test: characterize current detector-alpha and pixel-radius behavior`
5. `fix: stabilize Simple Menu ownership and visibility`
6. `fix: restore Live View dock cleanup`
7. `fix: canonicalize D4 orientation for py4D calibration`
8. `ui: clarify C10 stepping labels and cover toolbar controls`
9. `docs: align rotation refinement guidance with current behavior`
10. `build: harden macOS application packaging` if packaging remains in scope

Run the focused test file(s) for each commit and record the full-suite result
before the final handoff. Do not make a final commit that combines unrelated
leftovers merely to obtain a clean worktree.

## Phase 1: Fix Merge-Blocking Calibration Regressions

### 1.1 Restore safe wavelength handling

Files:

- `src/py4d_browser_plugin/fast_acbf/config.py`
- `tests/test_config.py`

Actions:

- Use `wavelength_for_conversion` for every pixel-radius-to-mrad conversion in
  `FastAcbfConfig.resolved_for()`.
- Do not multiply by `cfg.wavelength_angstrom` when it may legitimately remain
  `None`.
- Preserve `cfg.voltage_kv` and `cfg.wavelength_angstrom` as unset in the public
  configuration when no real voltage exists; the placeholder should remain an
  internal conversion detail only.
- Replace direct conversion arithmetic with
  `max_alpha_mrad_from_px(...)` so all branches use the same conversion helper.

Regression tests:

- Unset pixel calibration, no voltage, detectable circular ROI.
- Unset pixel calibration, no voltage, automatically detectable BF disk.
- Real scan/diffraction calibration but missing voltage.
- Confirm tcBF/orientation can resolve a usable pixel mask without making a fake
  voltage appear in the configuration.
- Confirm acBF remains blocked by its existing real-voltage gate.

Acceptance criteria:

- No `NoneType` multiplication occurs.
- Calibration-free tcBF reaches solver preparation with a valid
  `max_alpha_px`.
- The UI still displays voltage as unset.

### 1.2 Handle failed BF-disk detection without formatting `None`

Files:

- `src/py4d_browser_plugin/fast_acbf/config.py`
- `tests/test_config.py`

Actions:

- Remove or guard status/debug messages that format
  `max_alpha_mrad` with `:.4f` when it is `None`.
- Preserve a manually configured `max_alpha_mrad` when automatic resolution
  fails.
- If neither a pixel radius nor a manual mrad value exists, allow the existing
  validation/preparation layer to report a clear configuration error rather
  than raising during string formatting.
- Prefer routing diagnostic text through the plugin/runner status mechanism
  instead of unconditional `print()` calls, if status reporting is needed.

Regression tests:

- Featureless CBED with `max_alpha_mrad=None`.
- Featureless CBED with a manual `max_alpha_mrad`.
- Missing `skimage`/failed detection path, if practical to simulate.

Acceptance criteria:

- Detection failure never raises from logging or formatting.
- A valid manual value is not overwritten.
- An unresolved value produces a user-facing configuration error at the proper
  validation boundary.

## Phase 2: Preserve and Characterize BF-Disk Selection Semantics

Files:

- `src/py4d_browser_plugin/fast_acbf/config.py`
- `src/py4d_browser_plugin/fast_acbf/calibration.py`
- `src/py4d_browser_plugin/fast_acbf/dialogs/config_dialog.py`
- `src/py4d_browser_plugin/fast_acbf/dialogs/simple_menu_dialogs.py`
- `tests/test_config.py`
- `tests/test_dialogs.py`
- `tests/test_simple_menu.py`

Decision:

- Do not redesign `use_detector_alpha` or `max_alpha_px` semantics in this
  cleanup. Preserve the current functionality and behavior while fixing the
  calibration-free crashes.
- Where an existing test reflects an older semantic contract, update the test
  to characterize the retained current behavior rather than changing the
  implementation as part of this cleanup.
- Record any semantic concerns as follow-up work, separate from this branch.

Actions:

- Add characterization tests for every current branch before restructuring the
  code:
  - Calibration-free versus calibrated datacube.
  - `use_detector_alpha=True` versus `False`.
  - Live circular ROI present versus absent.
  - Cached/previous `max_alpha_px` present versus absent.
  - Automatic CBED detection success versus failure.
  - Transition from pixel calibration to real calibration.
- Make only the minimum control-flow changes needed to remove the
  calibration-free exceptions and preserve observed behavior.
- Avoid renaming the fields, changing UI meaning, or introducing a new
  precedence policy in this branch.
- Add a follow-up note in the code or project issue tracker summarizing the
  semantic questions identified during review.

Regression tests:

- Update the existing persistent-pixel-radius test if its expectation conflicts
  with the explicitly retained current behavior.
- Lock down the observed live-ROI, cached-radius, and automatic-detection
  behavior for both values of `use_detector_alpha`.
- Confirm repeated `resolved_for()` calls do not introduce a new exception or
  unintended state transition.

Acceptance criteria:

- The cleanup does not intentionally alter the current field behavior.
- All currently supported branches are described by focused tests.
- The calibration-free branch is safe with missing voltage and failed
  detection.

## Phase 3: Stabilize the Simple Menu Lifecycle

Files:

- `src/py4d_browser_plugin/fast_acbf/simple_menu_toolbar.py`
- `src/py4d_browser_plugin/fast_acbf/plugin.py`
- `tests/test_simple_menu.py`

Recommended design:

- Make the simple-menu object a `QToolBar` subclass directly, rather than a
  wrapper `QWidget` that creates a separately parented toolbar.

Constraint:

- Preserve normal simple-toolbar usage: the same menu action, visible toolbar
  layout, tcBF/acBF dropdown workflows, Advanced action, C10 controls, and
  user-facing show/hide behavior should remain available.

Actions:

- Give the toolbar one clear QObject owner.
- Add it to the main window exactly once.
- Use the toolbar itself for `show()`, `hide()`, close-event handling, action
  enablement, and deletion.
- Connect `visibilityChanged` or an equivalent toolbar signal so hiding/closing
  the toolbar reliably updates the check state of “Show Simple Menu.”
- Ensure removal calls `removeToolBar(toolbar)` followed by appropriate Qt
  deferred deletion.
- Verify repeated show/hide/show cycles do not create duplicate toolbars or
  duplicate signal connections.
- Remove unused widget imports while touching the file.

Tests:

- Replace the obsolete `QDockWidget.titleBarWidget()` assertion with toolbar
  contract tests.
- Show, hide, close, and reopen the toolbar.
- Confirm the menu action stays synchronized with toolbar visibility.
- Confirm only one toolbar exists after repeated toggles.
- Confirm all primary and dropdown actions are disabled while a worker is
  running.
- Confirm C10 controls are also disabled during a run.

Acceptance criteria:

- Toolbar lifetime and visibility are deterministic.
- No removed toolbar remains parented to the main window.
- Existing simple-workflow signals fire exactly once.

## Phase 4: Restore Correct Live View Dock Cleanup

Files:

- `src/py4d_browser_plugin/fast_acbf/plugin.py`
- `tests/test_live_view.py`

Actions:

- Change `_remove_live_view_dock()` back to the `QDockWidget` lifecycle:
  `removeDockWidget(dock)`, then deferred deletion.
- Do not access `dock.toolbar`; `LiveViewDock` is not a toolbar.
- Avoid broad exception suppression around an API mismatch. If compatibility
  guarding is necessary, catch only the expected compatibility exception or
  test the parent capability explicitly.
- Check that stopping and restarting Live View does not retain a hidden dock,
  callback, session, or worker thread.

Tests:

- Open and remove the Live View dock.
- Stop with `remove_dock=True` and `remove_dock=False`.
- Restart after removal.
- Verify callback restoration and action check state.
- Ensure the Live View test suite terminates without hanging.

Acceptance criteria:

- Live View is explicitly removed through `removeDockWidget()`.
- No Live View test leaves a Qt worker or event loop alive.

## Phase 5: Align Refinement Defaults and Documentation

Files:

- `src/py4d_browser_plugin/fast_acbf/config.py`
- `src/py4d_browser_plugin/fast_acbf/dialogs/_widgets.py`
- `src/py4d_browser_plugin/fast_acbf/dialogs/config_dialog.py`
- `src/py4d_browser_plugin/fast_acbf/dialogs/simple_menu_dialogs.py`
- `src/py4d_browser_plugin/fast_acbf/solver_job.py`
- `README.md`
- `tests/test_config.py`
- `tests/test_dialog_widgets.py`
- `tests/test_dialogs.py`
- `tests/test_simple_menu.py`
- `tests/test_worker.py`

Decisions:

### Focus sign

Decision:

- Use `focus_sign="none"` for every default, initialization, fallback, and reset
  occurrence.

Apply this consistently to:

- `FastAcbfConfig`
- `OrientationForm` initialization
- Reset behavior
- Simple Menu Orientation dialog
- Configuration dialog
- Tests and documentation

Search the complete repository for hardcoded `"overfocus"` defaults rather than
updating only `FastAcbfConfig`. Explicit non-default test cases for overfocus
and underfocus should remain.

### Defocus search

Preserve the current strategy for now:

- `defocus_points`: 5 → 11
- `defocus_range_tolerance_factor`: 24 → 80
- Default method: `max`
- Simple Menu coarse method: `max`
- Simple Menu refine method: `brent`

Confirm through focused tests that `method="brent"` and `method="max"` are
supported by the declared minimum `fast-acbf >= 0.6.0`.

### Scan rotation

Preserve the current implementation for this cleanup:

- Fine half-width: auto-derived → fixed 10 degrees
- Standalone method: Brent
- Fine points remain 11 even though Brent may not use them as a grid count

Actions:

- Update tooltips and documentation that currently describe standalone Refine
  Scan Rotation as an evenly spaced grid search.
- State that the current standalone action uses Brent and explain which
  tolerance/half-width fields affect it.
- Do not add a method selector or change the algorithm in this cleanup unless a
  current behavior cannot be represented accurately without one.
- Record any desired future grid-versus-Brent UI choice as separate follow-up
  work.

Acceptance criteria:

- Defaults, reset behavior, implementation, tooltips, and tests agree.
- Coarse and fine actions have clearly distinct, documented strategies.

## Phase 6: Make Orientation Synchronization D4-Aware

Files:

- `src/py4d_browser_plugin/fast_acbf/calibration.py`
- `src/py4d_browser_plugin/fast_acbf/config.py`
- `src/py4d_browser_plugin/fast_acbf/utils.py` if solver-to-config
  synchronization belongs there
- `tests/test_calibration.py`
- `tests/test_config.py`
- `tests/test_simple_menu.py`

Problem:

The plugin represents orientation as:

```text
(flipud, fliplr, transpose, rotation_deg)
```

with operations applied in the fast-acbf order:

```text
flipud → fliplr → transpose → continuous rotation
```

py4D calibration stores the canonical O(2) representation:

```text
QR_rotation + QR_flip
```

where `QR_flip` represents chirality/transpose. Writing only
`rotation_deg` and `transpose` is incorrect whenever `flipud` or `fliplr` is
true, because the same physical transform may contain an additional D4
quarter-turn.

Required conversion:

- Reuse or expose the authoritative D4 mapping derived for Orientation
  Optimization in fast-acbf rather than creating an unrelated convention in
  the plugin.
- The existing fast-acbf `_D4_TABLE` maps
  `(is_flipped, quarter_turn)` to `(flipud, fliplr, transpose)`.
- For plugin → py4D synchronization, apply the inverse lookup:
  1. Find `(is_flipped, q)` for the current three Boolean flags.
  2. Compute canonical rotation as `rotation_deg + 90° * q`.
  3. Normalize the angle according to the convention expected by
     `set_QR_rotation()`.
  4. Store `is_flipped` through `set_QR_flip()`.
- Prefer adding/reusing a public conversion helper in fast-acbf if this is a
  cross-project orientation contract. If that cannot be done in this branch,
  implement a small, documented plugin helper verified against fast-acbf's
  `CoordinateTransform` behavior.

For py4D → plugin synchronization:

- Reading canonical `QR_rotation + QR_flip` may set the plugin to the canonical
  representation `(flipud=False, fliplr=False, transpose=QR_flip,
  rotation_deg=QR_rotation)` if normalizing the displayed flags is acceptable.
- If the UI must preserve the current D4 flag choice, compute the residual
  plugin rotation that makes those flags equivalent to the canonical py4D
  transform.
- Whichever approach is used, a read → write → read round trip must preserve
  the physical transform, even if the exact Boolean representation is
  canonicalized.

Tests:

- Exhaustively test all eight `(flipud, fliplr, transpose)` combinations.
- Include residual rotations at `0°`, positive/negative non-quarter angles, and
  quarter-turn boundaries.
- Compare transformed coordinates or transformation matrices, not only stored
  flags, so tests verify physical equivalence.
- Test angle wraparound near `±180°` and `0°/360°`.
- Test plugin → py4D → plugin and py4D → plugin → py4D round trips.
- Test Orientation Optimization output through the same conversion used by
  calibration synchronization.
- Confirm the common case with both flips false is unchanged.

Acceptance criteria:

- `QR_rotation` and `QR_flip` encode the same physical orientation as the full
  plugin D4 state.
- No orientation jump occurs when saving an optimized state with `flipud=True`
  or `fliplr=True`.
- Bidirectional synchronization is stable modulo the documented canonical D4
  representation.

## Phase 7: Validate New Toolbar Features

Files:

- `src/py4d_browser_plugin/fast_acbf/simple_menu_toolbar.py`
- `src/py4d_browser_plugin/fast_acbf/plugin.py`
- `src/py4d_browser_plugin/fast_acbf/worker.py`
- `tests/test_simple_menu.py`
- `tests/test_worker.py`

### Upscale controls

- Confirm the tcBF and acBF spinboxes stay synchronized without duplicate
  application-level updates.
- Initialize both controls from the current configuration rather than always
  showing 1.
- Define whether switching from acBF back to tcBF should restore the
  mode-specific default interpolation method.
- Test `zero_insert` coercion when acBF is selected at upscale greater than 1.

### Manual C10 stepping

- Confirm C10 stepping is based on the solver-synchronized configuration.
- Confirm plus/minus operations update both the solver and displayed
  configuration exactly once.
- Decide whether the step control should support fractional Å values. It
  currently has a minimum of 0.01 but displays zero decimal places.
- Use the proper Å glyph consistently in status messages and widget suffixes.
- Change the displayed readout label from `Defocus` to `C10 (-df)`.
- Change the step label from `Step defocus` to `C10 (-df) step`.
- Add UI tests for both exact labels so subsequent cosmetic changes do not
  erase the sign convention.
- Confirm the C10 display updates after manual previews, coarse search, fine
  refinement, and Simple Menu reconstruction.

### Worker status

- Remove the unused local `result` if job return values will not be consumed, or
  include the returned job result in the emitted payload.
- Test the final status message for manual, defocus, Simple Menu reconstruction,
  orientation, and error paths.
- Decide whether suppressing the prior “complete on device” message is
  intentional.

Acceptance criteria:

- Toolbar state matches configuration state.
- Each user action produces one job and one clear completion message.

## Phase 8: Preserve and Test Output Pixel Metadata

Files:

- `src/py4d_browser_plugin/fast_acbf/plugin.py`
- `tests/test_live_view.py`
- Add or extend plugin display tests as needed.

Actions:

- Keep the new `pixel_size=scan_step_angstrom / upscale` behavior.
- Keep the proper `pixel_units="Å"` value.
- Cover every output route:
  - Standard virtual image
  - Standard result image
  - Live virtual image
  - Live result image
  - Reasserted Live View output
- Confirm py4D-browser 1.5.1 is the first version whose virtual-image API accepts
  these keyword arguments.

Acceptance criteria:

- All output routes carry correct scale metadata.
- The declared minimum dependency matches the API actually used.

## Phase 9: Harden the macOS PyInstaller Build

Files:

- `fast_acbf.spec`
- `launcher.py`
- `pyproject.toml`
- `README.md`
- Optional CI/build script under an appropriate project directory

Actions:

- Derive the Python shared-library filename from `sysconfig` instead of
  hardcoding `libpython3.12.dylib`.
- Decide whether the build officially supports only Python 3.12; document and
  validate that constraint if so.
- Add a packaging/build optional dependency containing PyInstaller.
- Document:
  - Required Conda environment
  - Build command
  - Expected output
  - macOS minimum version
  - Apple Silicon versus Intel expectations
  - Signing/notarization status
- Verify that intentionally excluded py4D-browser plugins are documented.
- Confirm all hidden imports correspond to real modules and remove stale ones.
- Add a macOS smoke check that launches the bundle and verifies plugin
  discovery.
- Keep `upx=False` for macOS unless there is a demonstrated reason to change it.

Acceptance criteria:

- A clean documented environment can produce the `.app`.
- The application launches and discovers fast-acbf.
- The build definition does not fail solely because the Python minor version
  differs from 3.12, unless that restriction is explicit.

## Phase 10: Full Regression and Merge-Readiness Validation

Run from the repository root:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate py4dgui
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m pytest -q -p no:cacheprovider
```

Additional checks:

```bash
git diff --check 7ea24131...HEAD
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m compileall -q src scripts tests
```

Because `compileall` writes bytecode, either run it only when repository writes
are acceptable or direct its cache outside the repository. An AST parse is a
read-only alternative during review.

Manual smoke-test matrix:

1. Pixel-unit datacube, no voltage:
   - Open Simple Menu.
   - Run tcBF.
   - Run orientation optimization.
   - Run coarse and refined defocus.
2. Fully calibrated datacube:
   - Confirm QR rotation/flip are read.
   - Change orientation and confirm calibration is updated through canonical
     D4 conversion.
   - Repeat with every flip/transpose combination.
   - Run acBF.
3. Circular detector ROI enabled and disabled:
   - Confirm documented alpha precedence.
4. No usable detector ROI and featureless CBED:
   - Confirm graceful validation.
5. Upscale greater than 1:
   - Run tcBF and acBF.
   - Confirm interpolation method and output pixel size.
6. Toolbar lifecycle:
   - Show, hide, close, and reopen repeatedly.
7. Live View lifecycle:
   - Start, stop, remove, reopen, and change datacube.

Final acceptance criteria:

- All automated tests pass.
- No test process hangs after completion.
- No Qt worker/thread warnings appear at shutdown.
- `git diff --check` is clean.
- Source changes are accompanied by focused regression tests.
- The final diff contains no accidental generated build output.

## Recommended Implementation Order

1. Align the existing tests with the decided retained behavior.
2. Add regression tests and fix wavelength/failed-detection crashes.
3. Add characterization tests that preserve current `use_detector_alpha` and
   `max_alpha_px` behavior.
4. Add lifecycle tests and refactor simple-toolbar ownership without changing
   normal usage.
5. Add removal/restart tests and repair Live View cleanup.
6. Standardize `focus_sign="none"` everywhere.
7. Add exhaustive D4 tests and implement canonical orientation
   synchronization.
8. Update C10 labels and test the new toolbar controls.
9. Update rotation tooltips and documentation to match the current strategy.
10. Verify output metadata.
11. Harden and smoke-test macOS packaging.
12. Run the full regression and manual workflow matrix.

Calibration-free tcBF, toolbar lifecycle, Live View cleanup, consistent focus
defaults, and D4-correct orientation synchronization should be treated as merge
blockers. Packaging hardening can be separated into a follow-up only if the new
spec and assets are also kept out of the merge until they have a documented,
reproducible validation path.
