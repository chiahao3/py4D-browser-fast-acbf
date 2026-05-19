import numpy as np

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.solver_job import AutoTuneJob
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

    def get_probe(self, frame="detector"):
        self.probe_frames.append(frame)
        return _FakeProbe()


class _FakeProbe:
    def abs(self):
        return np.ones((2, 2), dtype=np.float32)


def test_auto_tune_calls_fast_acbf_refine_all_params_directly(monkeypatch):
    solver = _FakeSolver()
    reconstructed_modes = []
    results = []
    runner = FastAcbfRunner(
        job=AutoTuneJob(),
        data=np.zeros((1, 1, 2, 2), dtype=np.float32),
        config=FastAcbfConfig(mode="acBF", output_frame="scan", refinement_mode="tcBF"),
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
    assert solver.probe_frames == ["scan"]
    assert reconstructed_modes == ["acBF"]
    assert results[0]["mode"] == "acBF"
