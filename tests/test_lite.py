import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolBar

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.dialogs.lite_dialogs import LiteOrientationDialog, LiteSettingsDialog
from py4d_browser_plugin.fast_acbf.dialogs._widgets import SCAN_ROTATION_HELP_TEXT
from py4d_browser_plugin.fast_acbf.lite_dock import LiteTaskbarDock
from py4d_browser_plugin.fast_acbf.solver_job import (
    AutoTuneJob,
    LiteReconstructJob,
    OptimizeOrientationJob,
    RefineDefocusJob,
    _apply_focus_sign_constraint,
)

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


class _AbState:
    def __init__(self, c10=0.0, coeffs=None):
        self.coeffs = dict(coeffs) if coeffs is not None else {"C_1_0": float(c10)}

    def get_physical(self, key):
        return self.coeffs.get(key, 0.0)

    def set_physical(self, key, value):
        self.coeffs[key] = float(value)


class _SpySolver:
    def __init__(self, unit_px=4.0, c10=0.0, tolerance_t1=1.0, raw_scan_shape=(100, 100)):
        self.ab_state = _AbState(c10)
        self._unit_px = unit_px
        self.calls = []
        self.coord_transform = {"flipud": False, "fliplr": False, "transpose": False}
        self.tolerance_factors = {1: tolerance_t1}
        # (100, 100) keeps the default lite_defocus_halfwidth_scan_fraction=0.2 resolving
        # to exactly the floor (20.0), matching pre-scaling-factor test expectations.
        self.raw_scan_shape = raw_scan_shape

    def get_yx_shifts_px(self, frame="scan"):
        # magnitude scales with the current C10; px-mode probes it at C10 == 1.
        scale = abs(self.ab_state.get_physical("C_1_0"))
        return torch.tensor([[0.0, self._unit_px * scale]])

    def set_flips(self, flipud, fliplr, transpose):
        self.coord_transform = {"flipud": flipud, "fliplr": fliplr, "transpose": transpose}

    def refine_defocus(self, **kwargs):
        self.calls.append(("refine_defocus", kwargs))

    def refine_aberrations(self, **kwargs):
        self.calls.append(("refine_aberrations", kwargs))

    def refine_all_params(self, **kwargs):
        self.calls.append(("refine_all_params", kwargs))


def _run(job, solver, config=None):
    config = config or FastAcbfConfig()
    messages = []
    job.execute(solver, config, messages.append)
    return messages


def test_lite_dock_defaults_and_signals():
    _app()
    dock = LiteTaskbarDock()
    assert isinstance(dock, QToolBar)
    assert dock.objectName() == "fastAcbfLiteDock"
    assert dock.title_label.text() == "Fast acBF"
    for name in (
        "orientation_requested",
        "tcbf_requested",
        "calibration_requested",
        "acbf_requested",
        "settings_requested",
        "advanced_requested",
        "closed",
    ):
        assert hasattr(dock, name)
    assert dock.orientation_action.text() == "Set Dataset Orientation..."
    assert dock.tcbf_action.text() == "tcBF"
    assert dock.calibration_action.text() == "Set Calibrations..."
    assert dock.acbf_action.text() == "acBF"
    assert dock.settings_action.text() == "Settings..."
    assert dock.advanced_action.text() == "Advanced..."
    assert dock._defocus_widget._defocus_label.text() == "C10 (-df)"
    assert dock._defocus_widget._label.text() == "C10 (-df) step"
    assert dock._defocus_widget.step_spin.suffix() == " Å"
    dock.set_enabled(False)
    assert dock.tcbf_action.isEnabled() is False
    assert dock.orientation_action.isEnabled() is False
    assert dock.calibration_action.isEnabled() is False
    assert dock.settings_action.isEnabled() is False
    assert dock._defocus_widget.isEnabled() is False
    dock.deleteLater()


def test_lite_toolbar_upscale_controls_initialize_and_emit_once():
    _app()
    toolbar = LiteTaskbarDock(upscale=2.5)
    changes = QSignalSpy(toolbar.upscale_changed)

    assert toolbar.tcbf_upscale_spin.value() == 2.5
    assert toolbar.acbf_upscale_spin.value() == 2.5

    toolbar.tcbf_upscale_spin.setValue(3.5)
    assert toolbar.acbf_upscale_spin.value() == 3.5
    assert len(changes) == 1
    assert changes[0][0] == 3.5

    toolbar.acbf_upscale_spin.setValue(4.25)
    assert toolbar.tcbf_upscale_spin.value() == 4.25
    assert len(changes) == 2
    assert changes[1][0] == 4.25
    toolbar.deleteLater()


def test_lite_toolbar_defocus_controls_emit_once_and_render_c10():
    _app()
    toolbar = LiteTaskbarDock()
    increases = QSignalSpy(toolbar.increase_defocus_requested)
    decreases = QSignalSpy(toolbar.decrease_defocus_requested)
    steps = QSignalSpy(toolbar.defocus_step_changed)

    toolbar._defocus_widget.btn_plus.click()
    toolbar._defocus_widget.btn_minus.click()
    toolbar._defocus_widget.step_spin.setValue(2.5)
    toolbar.set_c10(-125.0)

    assert len(increases) == 1
    assert len(decreases) == 1
    assert len(steps) == 1
    assert steps[0][0] == 2.5
    assert toolbar._defocus_widget.c10_label.text() == "-125 Å"
    toolbar.deleteLater()


def test_lite_toolbar_has_single_owner_and_emits_closed():
    app = _app()
    window = QMainWindow()
    toolbar = LiteTaskbarDock(window)
    closed = QSignalSpy(toolbar.closed)

    window.addToolBar(toolbar)
    window.show()
    app.processEvents()

    assert toolbar.parent() is window
    assert window.findChildren(QToolBar, "fastAcbfLiteDock") == [toolbar]

    toolbar.close()
    app.processEvents()
    assert len(closed) == 1

    window.removeToolBar(toolbar)
    toolbar.deleteLater()
    window.deleteLater()


def test_disabled_level_skips_refinement():
    solver = _SpySolver()
    _run(LiteReconstructJob(aberration_search="disabled"), solver)
    assert solver.calls == []


def test_df_only_calls_refine_defocus():
    solver = _SpySolver()
    _run(LiteReconstructJob(aberration_search="df_only"), solver)
    assert [c[0] for c in solver.calls] == ["refine_defocus"]


def test_first_order_freezes_second_order():
    solver = _SpySolver()
    cfg = FastAcbfConfig(max_order=2)
    _run(LiteReconstructJob(aberration_search="first_order"), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_aberrations"
    assert kwargs["lr_scales"] == [1.0, 0.0]


def test_second_order_refines_both_orders():
    solver = _SpySolver()
    cfg = FastAcbfConfig(max_order=2)
    _run(LiteReconstructJob(aberration_search="second_order"), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_aberrations"
    assert kwargs["lr_scales"] == [1.0, 1.0]


def test_pixel_mode_derives_search_range_from_shifts():
    solver = _SpySolver(unit_px=4.0, c10=0.0)
    cfg = FastAcbfConfig(focus_sign="none")
    job = LiteReconstructJob(pixel_mode=True)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_defocus"
    # unit_px = 4 px per unit C10 -> half_c10 = 20 / 4 = 5 -> range (-5, +5)
    assert kwargs["search_range"] == (-5.0, 5.0)
    # C10 restored to its original value after probing.
    assert solver.ab_state.get_physical("C_1_0") == 0.0


def test_pixel_mode_scales_halfwidth_with_scan_shape():
    """Exercises the actual LiteReconstructJob code path reading solver.raw_scan_shape,
    complementing the direct FastAcbfConfig.resolved_lite_defocus_halfwidth_px unit
    tests in test_config.py."""
    solver = _SpySolver(unit_px=4.0, c10=0.0, raw_scan_shape=(256, 300))
    cfg = FastAcbfConfig(focus_sign="none")
    job = LiteReconstructJob(pixel_mode=True)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    # min(256, 300)=256; 0.2*256=51.2 (above the 20 floor) -> half_c10 = 51.2/4 = 12.8
    assert kwargs["search_range"] == (-12.8, 12.8)


def test_pixel_mode_respects_explicit_halfwidth_regardless_of_scan_shape():
    solver = _SpySolver(unit_px=4.0, c10=0.0, raw_scan_shape=(1024, 1024))
    cfg = FastAcbfConfig(focus_sign="none", lite_defocus_halfwidth_px=8.0)
    job = LiteReconstructJob(pixel_mode=True)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    # explicit 8.0 used as-is despite a large scan shape -> half_c10 = 8/4 = 2
    assert kwargs["search_range"] == (-2.0, 2.0)


def test_optimize_orientation_job_excludes_fine_aberrations():
    solver = _SpySolver()
    msgs = _run(OptimizeOrientationJob(), solver)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    assert kwargs["targets"] == ("orientation_defocus", "coarse_aberrations", "fine_rotation")
    assert "fine_aberrations" not in kwargs["targets"]
    assert any("orientation" in m.lower() for m in msgs)


def test_optimize_orientation_job_forwards_explicit_fine_rotation_settings():
    solver = _SpySolver()
    # An explicit halfwidth is forwarded exactly as given, including values narrower
    # than the derived default (180/12=15.0 for the default rotation_points=12) --
    # no floor is applied once the user has set one (see test_config.py for the
    # unset -> derived-default case).
    cfg = FastAcbfConfig(fine_rotation_halfwidth_deg=2.0, fine_rotation_xatol_deg=0.1)
    _run(OptimizeOrientationJob(), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    assert kwargs["fine_rotation_halfwidth"] == 2.0
    assert kwargs["fine_rotation_xatol"] == 0.1


def test_auto_tune_job_forwards_explicit_fine_rotation_settings():
    solver = _SpySolver()
    cfg = FastAcbfConfig(fine_rotation_halfwidth_deg=2.0, fine_rotation_xatol_deg=0.1)
    _run(AutoTuneJob(), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    assert kwargs["fine_rotation_halfwidth"] == 2.0
    assert kwargs["fine_rotation_xatol"] == 0.1


def test_optimize_orientation_job_derives_fine_rotation_halfwidth_when_unset():
    """Unset (None) halfwidth derives from rotation_points -- exercises the actual
    OptimizeOrientationJob code path, complementing the direct
    FastAcbfConfig.resolved_fine_rotation_halfwidth_deg unit tests in test_config.py."""
    solver = _SpySolver()
    cfg = FastAcbfConfig(rotation_points=12, fine_rotation_halfwidth_deg=None)
    _run(OptimizeOrientationJob(), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    assert kwargs["fine_rotation_halfwidth"] == 15.0


# ---------------------------------------------------------------------------
# focus_sign: defocus ranges clamped to C10 >= 0 (overfocus) or C10 <= 0 (underfocus)
# ---------------------------------------------------------------------------


def test_apply_focus_sign_constraint_is_noop_when_none():
    solver = _SpySolver()
    cfg = FastAcbfConfig(focus_sign="none")
    assert _apply_focus_sign_constraint(solver, cfg, (-5.0, 5.0)) == (-5.0, 5.0)
    assert _apply_focus_sign_constraint(solver, cfg, None) is None


def test_apply_focus_sign_constraint_clamps_explicit_range_overfocus():
    solver = _SpySolver()
    cfg = FastAcbfConfig(focus_sign="overfocus")
    assert _apply_focus_sign_constraint(solver, cfg, (-5.0, 5.0)) == (0.0, 5.0)
    # already all-positive: unchanged
    assert _apply_focus_sign_constraint(solver, cfg, (2.0, 5.0)) == (2.0, 5.0)
    # entirely negative: collapses to (0, 0) rather than an inverted/empty range
    assert _apply_focus_sign_constraint(solver, cfg, (-5.0, -2.0)) == (0.0, 0.0)


def test_apply_focus_sign_constraint_clamps_explicit_range_underfocus():
    solver = _SpySolver()
    cfg = FastAcbfConfig(focus_sign="underfocus")
    assert _apply_focus_sign_constraint(solver, cfg, (-5.0, 5.0)) == (-5.0, 0.0)
    # already all-negative: unchanged
    assert _apply_focus_sign_constraint(solver, cfg, (-5.0, -2.0)) == (-5.0, -2.0)
    # entirely positive: collapses to (0, 0) rather than an inverted/empty range
    assert _apply_focus_sign_constraint(solver, cfg, (2.0, 5.0)) == (0.0, 0.0)


def test_apply_focus_sign_constraint_builds_range_from_halfwidth_when_none_given():
    solver = _SpySolver(c10=3.0)
    cfg = FastAcbfConfig(focus_sign="overfocus")
    assert _apply_focus_sign_constraint(solver, cfg, None, search_halfwidth=10.0) == (0.0, 13.0)


def test_apply_focus_sign_constraint_builds_range_from_tolerance_factor_when_no_halfwidth():
    solver = _SpySolver(c10=3.0, tolerance_t1=2.0)
    cfg = FastAcbfConfig(focus_sign="overfocus", defocus_range_tolerance_factor=4.0)
    # half = 4 * T1(2.0) = 8 -> (3-8, 3+8) = (-5, 11) -> clamped to (0, 11)
    assert _apply_focus_sign_constraint(solver, cfg, None) == (0.0, 11.0)


def test_refine_defocus_job_clamps_range_when_overfocus():
    solver = _SpySolver(c10=3.0, tolerance_t1=2.0)
    cfg = FastAcbfConfig(focus_sign="overfocus", defocus_range_tolerance_factor=4.0)
    _run(RefineDefocusJob(), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_defocus"
    assert kwargs["search_range"] == (0.0, 11.0)
    assert kwargs["search_halfwidth"] is None


def test_lite_df_only_clamps_range_when_underfocus():
    # c10=0.0 -> unseeded (full coarse width), isolating the clamp behavior itself from
    # the seeded-narrowing behavior (see test_lite_df_only_seeded_search_narrows_and_clamps).
    solver = _SpySolver(c10=0.0, tolerance_t1=2.0)
    cfg = FastAcbfConfig(focus_sign="underfocus", defocus_range_tolerance_factor=4.0)
    _run(LiteReconstructJob(aberration_search="df_only"), solver, cfg)
    (name, kwargs), = solver.calls
    # half = 4 * T1(2.0) = 8 -> (0-8, 0+8) = (-8, 8) -> clamped to (-8, 0)
    assert kwargs["search_range"] == (-8.0, 0.0)


def test_lite_df_only_seeded_search_narrows_and_clamps():
    """A non-zero starting C10 (e.g. left behind by Orientation Optimization) triggers the
    seeded, narrower search width instead of the full coarse one."""
    solver = _SpySolver(c10=-3.0, tolerance_t1=2.0)
    cfg = FastAcbfConfig(
        focus_sign="underfocus",
        defocus_range_tolerance_factor=4.0,
        lite_seeded_defocus_fraction=0.5,
    )
    _run(LiteReconstructJob(aberration_search="df_only"), solver, cfg)
    (name, kwargs), = solver.calls
    # coarse half = 4 * T1(2.0) = 8; seeded (fraction=0.5) halves it to 4
    # -> unmodified range (-3-4, -3+4) = (-7, 1) -> clamped to (-7, 0)
    assert kwargs["search_range"] == (-7.0, 0.0)


def test_pixel_mode_clamps_range_when_overfocus():
    # c10=0.0 -> unseeded (full coarse width), isolating the clamp behavior itself from
    # the seeded-narrowing behavior (see test_pixel_mode_seeded_search_narrows_and_clamps).
    solver = _SpySolver(unit_px=4.0, c10=0.0)
    cfg = FastAcbfConfig(focus_sign="overfocus")
    job = LiteReconstructJob(pixel_mode=True)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    # unmodified range would be (0-5, 0+5) = (-5, 5) -> clamped to (0, 5)
    assert kwargs["search_range"] == (0.0, 5.0)


def test_pixel_mode_seeded_search_narrows_and_clamps():
    """A non-zero starting C10 (e.g. left behind by Orientation Optimization) triggers the
    seeded, narrower search width instead of the full coarse one."""
    solver = _SpySolver(unit_px=4.0, c10=-2.0)
    cfg = FastAcbfConfig(focus_sign="overfocus", lite_seeded_defocus_fraction=0.5)
    job = LiteReconstructJob(pixel_mode=True)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    # coarse halfwidth_px = max(20, 0.2*100) = 20; seeded (fraction=0.5) halves it to 10
    # -> half_c10 = 10/4 = 2.5 -> unmodified range (-2-2.5, -2+2.5) = (-4.5, 0.5)
    # -> clamped to (0, 0.5)
    assert kwargs["search_range"] == (0.0, 0.5)


def test_auto_tune_job_clamps_defocus_range_when_overfocus():
    solver = _SpySolver(c10=3.0, tolerance_t1=2.0)
    cfg = FastAcbfConfig(focus_sign="overfocus", defocus_range_tolerance_factor=4.0)
    _run(AutoTuneJob(), solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    assert kwargs["defocus_range"] == (0.0, 11.0)


def test_optimize_orientation_job_default_leaves_defocus_range_to_fast_acbf():
    solver = _SpySolver()
    cfg = FastAcbfConfig(focus_sign="none")
    _run(OptimizeOrientationJob(), solver, cfg)
    (name, kwargs), = solver.calls
    assert kwargs["defocus_range"] is None


def test_optimize_orientation_pixel_mode_derives_defocus_range_from_shifts():
    solver = _SpySolver(unit_px=4.0, c10=2.0)
    cfg = FastAcbfConfig(focus_sign="none")
    job = OptimizeOrientationJob(pixel_mode=True)
    msgs = _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    # unit_px = 4 px per unit C10 -> half_c10 = 20 / 4 = 5 -> range centered on c10=2.0
    assert kwargs["defocus_range"] == (-3.0, 7.0)
    # C10 restored to its original value after probing.
    assert solver.ab_state.get_physical("C_1_0") == 2.0
    assert any("calibration-free" in m.lower() for m in msgs)


def test_optimize_orientation_pixel_mode_respects_explicit_defocus_range():
    solver = _SpySolver(unit_px=4.0, c10=0.0)
    cfg = FastAcbfConfig(
        focus_sign="none", defocus_range_min_angstrom=-1.0, defocus_range_max_angstrom=1.0
    )
    job = OptimizeOrientationJob(pixel_mode=True)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    assert kwargs["defocus_range"] == (-1.0, 1.0)


def test_lite_settings_dialog_round_trips_calibration_free():
    _app()
    dialog = LiteSettingsDialog(FastAcbfConfig(calibration_free=True))
    assert dialog.calibration_free_cb.isChecked() is True

    dialog.calibration_free_cb.setChecked(False)
    values = dialog.values()
    assert values.calibration_free is False
    dialog.close()


def test_lite_settings_dialog_derives_mrad_from_edited_max_alpha_px():
    _app()
    from py4d_browser_plugin.fast_acbf.calibration import PLACEHOLDER_WAVELENGTH_ANGSTROM

    dialog = LiteSettingsDialog(FastAcbfConfig(max_alpha_px=10.0, dk_inv_angstrom=0.05, voltage_kv=None))
    assert dialog.max_alpha_px_line.isReadOnly() is False
    assert dialog.max_alpha_line.isReadOnly() is True

    dialog.max_alpha_px_line.setText("20.0")
    dialog._update_max_alpha_mrad_display()
    expected = 20.0 * 0.05 * PLACEHOLDER_WAVELENGTH_ANGSTROM * 1000.0
    assert abs(float(dialog.max_alpha_line.text()) - expected) < abs(expected) * 1e-4

    values = dialog.values()
    assert values.max_alpha_px == 20.0
    assert abs(values.max_alpha_mrad - expected) < abs(expected) * 1e-4
    dialog.close()


def test_lite_settings_dialog_has_no_force_overfocus_control():
    _app()
    dialog = LiteSettingsDialog(FastAcbfConfig())
    assert not hasattr(dialog, "force_overfocus_cb")
    dialog.close()


def test_lite_orientation_dialog_defaults_to_none_and_round_trips_focus_sign():
    _app()
    dialog = LiteOrientationDialog(FastAcbfConfig())
    assert dialog.orientation_form.focus_sign_combo.currentText() == "None"
    assert dialog.orientation_form.rotation_help_btn.toolTip() == SCAN_ROTATION_HELP_TEXT

    dialog.orientation_form.focus_sign_combo.setCurrentText("Underfocus")
    dialog.accept()
    assert dialog.config.focus_sign == "underfocus"
    dialog.close()
