import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PyQt5.QtWidgets import QApplication

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.dialogs.lite_dialogs import LiteOrientationDialog, LiteSettingsDialog
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
    def __init__(self, unit_px=4.0, c10=0.0, tolerance_t1=1.0):
        self.ab_state = _AbState(c10)
        self._unit_px = unit_px
        self.calls = []
        self.coord_transform = {"flipud": False, "fliplr": False, "transpose": False}
        self.tolerance_factors = {1: tolerance_t1}

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
    # Native title bar collapsed (an empty widget stands in for it).
    assert dock.titleBarWidget() is not None
    assert dock.title_label.text() == "acBF workflow"
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
    assert dock.orientation_btn.text() == "1. Orientation"
    assert dock.tcbf_btn.text() == "2. tcBF"
    assert dock.calibration_btn.text() == "3. Calibration"
    assert dock.acbf_btn.text() == "4. acBF"
    assert dock.settings_btn.text() == "5. Settings"
    assert dock.advanced_btn.text() == "6. Advanced..."
    dock.set_enabled(False)
    assert dock.tcbf_btn.isEnabled() is False
    assert dock.orientation_btn.isEnabled() is False
    assert dock.calibration_btn.isEnabled() is False
    assert dock.settings_btn.isEnabled() is False


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
    job = LiteReconstructJob(pixel_mode=True, defocus_halfwidth_px=20.0)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    assert name == "refine_defocus"
    # unit_px = 4 px per unit C10 -> half_c10 = 20 / 4 = 5 -> range (-5, +5)
    assert kwargs["search_range"] == (-5.0, 5.0)
    # C10 restored to its original value after probing.
    assert solver.ab_state.get_physical("C_1_0") == 0.0


def test_optimize_orientation_job_excludes_fine_aberrations():
    solver = _SpySolver()
    msgs = _run(OptimizeOrientationJob(), solver)
    (name, kwargs), = solver.calls
    assert name == "refine_all_params"
    assert kwargs["targets"] == ("orientation_defocus", "coarse_aberrations", "fine_rotation")
    assert "fine_aberrations" not in kwargs["targets"]
    assert any("orientation" in m.lower() for m in msgs)


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
    solver = _SpySolver(c10=-3.0, tolerance_t1=2.0)
    cfg = FastAcbfConfig(focus_sign="underfocus", defocus_range_tolerance_factor=4.0)
    _run(LiteReconstructJob(aberration_search="df_only"), solver, cfg)
    (name, kwargs), = solver.calls
    # half = 4 * T1(2.0) = 8 -> (-3-8, -3+8) = (-11, 5) -> clamped to (-11, 0)
    assert kwargs["search_range"] == (-11.0, 0.0)


def test_pixel_mode_clamps_range_when_overfocus():
    solver = _SpySolver(unit_px=4.0, c10=-2.0)
    cfg = FastAcbfConfig(focus_sign="overfocus")
    job = LiteReconstructJob(pixel_mode=True, defocus_halfwidth_px=20.0)
    _run(job, solver, cfg)
    (name, kwargs), = solver.calls
    # unmodified range would be (-2-5, -2+5) = (-7, 3) -> clamped to (0, 3)
    assert kwargs["search_range"] == (0.0, 3.0)


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
    job = OptimizeOrientationJob(pixel_mode=True, defocus_halfwidth_px=20.0)
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
    job = OptimizeOrientationJob(pixel_mode=True, defocus_halfwidth_px=20.0)
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


def test_lite_settings_dialog_has_no_force_overfocus_control():
    _app()
    dialog = LiteSettingsDialog(FastAcbfConfig())
    assert not hasattr(dialog, "force_overfocus_cb")
    dialog.close()


def test_lite_orientation_dialog_defaults_to_overfocus_and_round_trips_focus_sign():
    _app()
    dialog = LiteOrientationDialog(FastAcbfConfig())
    assert dialog.orientation_form.focus_sign_combo.currentText() == "Overfocus"

    dialog.orientation_form.focus_sign_combo.setCurrentText("Underfocus")
    dialog.accept()
    assert dialog.config.focus_sign == "underfocus"
    dialog.close()
