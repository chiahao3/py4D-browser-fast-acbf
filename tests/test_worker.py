import numpy as np

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.solver_job import (
    AutoTuneJob,
    RefineDefocusJob,
    RefineScanRotationJob,
)
from py4d_browser_plugin.fast_acbf.worker import FastAcbfJobState, FastAcbfRunner


class _EmptyAberrationState:
    coeffs = {}


class _FakeSolver:
    device = "cpu"
    rotation_deg = 0.0
    coord_transform = {}
    ab_state = _EmptyAberrationState()

    def __init__(self):
        self.refine_all_kwargs = None
        self.probe_frames = []

    def refine_flips(self, **kwargs):
        raise AssertionError("auto_tune should not pre-run refine_flips")

    def refine_scan_rotation(self, **kwargs):
        raise AssertionError("auto_tune should not pre-run refine_scan_rotation")

    def refine_defocus(self, **kwargs):
        raise AssertionError("auto_tune should not pre-run refine_defocus")

    def refine_all_params(self, **kwargs):
        self.refine_all_kwargs = kwargs

    def get_probe(self, frame="detector", upscale=None):
        self.probe_frames.append((frame, upscale))
        return _FakeProbe()


class _FakeProbe:
    def abs(self):
        return np.ones((2, 2), dtype=np.float32)


class _FakeRefinementSolver(_FakeSolver):
    def __init__(self):
        super().__init__()
        self.refine_defocus_kwargs = None
        self.refine_rotation_kwargs = None

    def refine_defocus(self, **kwargs):
        self.refine_defocus_kwargs = kwargs

    def refine_scan_rotation(self, **kwargs):
        self.refine_rotation_kwargs = kwargs


def test_refine_defocus_passes_detailed_search_options():
    solver = _FakeRefinementSolver()
    cfg = FastAcbfConfig(
        focus_sign="none",
        defocus_range_min_angstrom=-15.0,
        defocus_range_max_angstrom=25.0,
        defocus_range_tolerance_factor=10.0,
        pad_width=2,
    )

    RefineDefocusJob().execute(solver, cfg, lambda _msg: None)

    assert solver.refine_defocus_kwargs["search_range"] == (-15.0, 25.0)
    assert solver.refine_defocus_kwargs["search_halfwidth"] is None
    assert solver.refine_defocus_kwargs["defocus_range_tolerance_factor"] == 10.0
    assert solver.refine_defocus_kwargs["method"] == "max"
    assert solver.refine_defocus_kwargs["pad_width"] == 2
    assert "fov" not in solver.refine_defocus_kwargs


def test_refine_scan_rotation_passes_detailed_search_options():
    solver = _FakeRefinementSolver()
    cfg = FastAcbfConfig(fine_rotation_halfwidth_deg=20.0, fine_rotation_points=13)

    RefineScanRotationJob().execute(solver, cfg, lambda _msg: None)

    assert solver.refine_rotation_kwargs["search_range"] is None
    assert solver.refine_rotation_kwargs["search_halfwidth"] == 20.0
    assert solver.refine_rotation_kwargs["num_points"] == 13
    assert solver.refine_rotation_kwargs["method"] == "brent"
    assert "xatol" not in solver.refine_rotation_kwargs
    assert "fov" not in solver.refine_rotation_kwargs


def test_auto_tune_calls_fast_acbf_refine_all_params_directly(monkeypatch):
    solver = _FakeSolver()
    reconstructed_modes = []
    results = []
    runner = FastAcbfRunner(
        job=AutoTuneJob(),
        data=np.zeros((1, 1, 2, 2), dtype=np.float32),
        config=FastAcbfConfig(
            mode="acBF",
            output_frame="scan",
            refinement_mode="tcBF",
            pad_width=2,
            upscale=1.5,
            upscale_method="nearest",
            focus_sign="none",
            defocus_range_min_angstrom=-10.0,
            defocus_range_max_angstrom=20.0,
            defocus_range_tolerance_factor=12.0,
            rotation_points=23,
            fine_rotation_halfwidth_deg=10.0,
            fine_rotation_points=13,
        ),
        state=FastAcbfJobState(),
    )
    monkeypatch.setattr(runner, "_get_solver", lambda: solver)

    def reconstruct(_solver, mode):
        reconstructed_modes.append(mode)
        return np.ones((2, 2), dtype=np.float32)

    monkeypatch.setattr(runner, "_reconstruct", reconstruct)

    failures = []
    runner.failed.connect(failures.append)
    runner.finished_result.connect(results.append)
    runner.run()

    assert failures == []
    assert solver.refine_all_kwargs is not None
    assert "targets" not in solver.refine_all_kwargs
    assert solver.refine_all_kwargs["mode"] == "tcBF"
    assert solver.refine_all_kwargs["pad_width"] == 2
    assert "fov" not in solver.refine_all_kwargs
    assert solver.refine_all_kwargs["upscale"] == 1.5
    assert solver.refine_all_kwargs["upscale_method"] == "nearest"
    assert solver.refine_all_kwargs["defocus_range"] == (-10.0, 20.0)
    assert solver.refine_all_kwargs["defocus_range_tolerance_factor"] == 12.0
    assert solver.refine_all_kwargs["rotation_num_points"] == 23
    assert solver.refine_all_kwargs["fine_rotation_halfwidth"] == 10.0
    assert solver.refine_all_kwargs["fine_rotation_xatol"] == 0.1
    assert "fine_rotation_num_points" not in solver.refine_all_kwargs
    assert solver.probe_frames == [("scan", 1.5)]
    assert reconstructed_modes == ["acBF"]
    assert results[0]["mode"] == "acBF"
