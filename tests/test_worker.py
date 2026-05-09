import numpy as np

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
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

    def refine_flips(self, **kwargs):
        raise AssertionError("auto_tune should not pre-run refine_flips")

    def refine_scan_rotation(self, **kwargs):
        raise AssertionError("auto_tune should not pre-run refine_scan_rotation")

    def refine_defocus(self, **kwargs):
        raise AssertionError("auto_tune should not pre-run refine_defocus")

    def refine_all_params(self, **kwargs):
        self.refine_all_kwargs = kwargs

    def get_probe(self, frame="detector"):
        return _FakeProbe()


class _FakeProbe:
    def abs(self):
        return np.ones((2, 2), dtype=np.float32)


def test_auto_tune_calls_fast_acbf_refine_all_params_directly(monkeypatch):
    solver = _FakeSolver()
    runner = FastAcbfRunner(
        command="auto_tune",
        data=np.zeros((1, 1, 2, 2), dtype=np.float32),
        config=FastAcbfConfig(),
        state=FastAcbfJobState(),
    )
    monkeypatch.setattr(runner, "_get_solver", lambda: solver)
    monkeypatch.setattr(runner, "_reconstruct", lambda _solver, _mode: np.ones((2, 2), dtype=np.float32))

    failures = []
    runner.failed.connect(failures.append)
    runner.run()

    assert failures == []
    assert solver.refine_all_kwargs is not None
    assert "targets" not in solver.refine_all_kwargs
