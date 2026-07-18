import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch
from PyQt5.QtWidgets import QApplication

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.lite_dock import LiteTaskbarDock
from py4d_browser_plugin.fast_acbf.solver_job import LiteReconstructJob

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


class _AbState:
    def __init__(self, c10=0.0):
        self._c10 = c10

    def get_physical(self, key):
        assert key == "C_1_0"
        return self._c10

    def set_physical(self, key, value):
        assert key == "C_1_0"
        self._c10 = value


class _SpySolver:
    def __init__(self, unit_px=4.0, c10=0.0):
        self.ab_state = _AbState(c10)
        self._unit_px = unit_px
        self.calls = []

    def get_yx_shifts_px(self, frame="scan"):
        # magnitude scales with the current C10; px-mode probes it at C10 == 1.
        scale = abs(self.ab_state.get_physical("C_1_0"))
        return torch.tensor([[0.0, self._unit_px * scale]])

    def refine_defocus(self, **kwargs):
        self.calls.append(("refine_defocus", kwargs))

    def refine_aberrations(self, **kwargs):
        self.calls.append(("refine_aberrations", kwargs))


def _run(job, solver, config=None):
    config = config or FastAcbfConfig()
    messages = []
    job.execute(solver, config, messages.append)
    return messages


def test_lite_dock_defaults_and_signals():
    _app()
    dock = LiteTaskbarDock()
    assert dock.auto_orientations_enabled() is True
    # Native title bar collapsed (an empty widget stands in for it).
    assert dock.titleBarWidget() is not None
    assert dock.title_label.text() == "fast-acbf Lite"
    for name in ("tcbf_requested", "acbf_requested", "advanced_requested", "closed"):
        assert hasattr(dock, name)
    dock.set_enabled(False)
    assert dock.tcbf_btn.isEnabled() is False


def test_disabled_level_skips_refinement():
    solver = _SpySolver()
    _run(LiteReconstructJob(aberration_search="disabled"), solver)
    assert solver.calls == []


def test_df_only_calls_refine_defocus():
    solver = _SpySolver()
    _run(LiteReconstructJob(aberration_search="df_only", auto_orientations=False), solver)
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
    job = LiteReconstructJob(pixel_mode=True, defocus_halfwidth_px=20.0)
    _run(job, solver)
    (name, kwargs), = solver.calls
    assert name == "refine_defocus"
    # unit_px = 4 px per unit C10 -> half_c10 = 20 / 4 = 5 -> range (-5, +5)
    assert kwargs["search_range"] == (-5.0, 5.0)
    # C10 restored to its original value after probing.
    assert solver.ab_state.get_physical("C_1_0") == 0.0


def test_auto_orientations_flag_controls_placeholder_message():
    solver = _SpySolver()
    msgs = _run(LiteReconstructJob(aberration_search="disabled", auto_orientations=True), solver)
    assert any("orientation" in m.lower() for m in msgs)
    msgs = _run(LiteReconstructJob(aberration_search="disabled", auto_orientations=False), solver)
    assert not any("orientation" in m.lower() for m in msgs)
