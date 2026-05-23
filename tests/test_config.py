from py4d_browser_plugin.fast_acbf.config import (
    FastAcbfConfig,
    label_dict_to_fast_acbf,
)
from py4d_browser_plugin.fast_acbf.utils import build_solver
from py4d_browser_plugin.fast_acbf.worker import evaluate_metric


def test_label_dict_to_fast_acbf():
    out = label_dict_to_fast_acbf(
        {"C10": 10.0, "C12a": 1.0, "C12b": 2.0, "C21a": 3.0},
        max_order=2,
    )
    assert out[(1, 0)] == 10.0
    assert out[(1, 2)] == {"a": 1.0, "b": 2.0}
    assert out[(2, 1)] == {"a": 3.0, "b": 0.0}
    assert out[(2, 3)] == {"a": 0.0, "b": 0.0}


def test_config_signature_changes_with_runtime_device():
    cfg = FastAcbfConfig(device="cuda")
    data = type("ArrayLike", (), {"shape": (1, 2, 3, 4), "dtype": "float32"})()
    assert "cuda" in cfg.solver_signature(data)


def test_fast_acbf_050_preparation_defaults_and_kwargs():
    cfg = FastAcbfConfig()
    kwargs = cfg.reconstruct_kwargs()

    assert cfg.normalized_pad_width() is None
    assert cfg.output_pixel_size_angstrom() == 1.0
    assert kwargs["pad_width"] is None
    assert kwargs["upscale"] == 1.0
    assert kwargs["upscale_method"] == "bilinear"


def test_fast_acbf_050_preparation_signature_and_pad_normalization():
    data = type("ArrayLike", (), {"shape": (1, 2, 3, 4), "dtype": "float32"})()
    base = FastAcbfConfig(pad_width=0)
    changed = FastAcbfConfig(pad_width=3, upscale=2.0, upscale_method="nearest")

    assert base.normalized_pad_width() is None
    assert changed.normalized_pad_width() == 3
    assert changed.output_pixel_size_angstrom() == 0.5
    assert base.solver_signature(data) != changed.solver_signature(data)


def test_refinement_search_range_helpers():
    cfg = FastAcbfConfig(
        defocus_range_min_angstrom=-10.0,
        defocus_range_max_angstrom=20.0,
        rotation_range_min_deg=-5.0,
        rotation_range_max_deg=5.0,
    )

    assert cfg.defocus_search_range() == (-10.0, 20.0)
    assert cfg.rotation_search_range() == (-5.0, 5.0)
    assert FastAcbfConfig().defocus_search_range() is None
    assert FastAcbfConfig().rotation_search_range() is None


def test_build_solver_passes_fast_acbf_050_preparation_kwargs(monkeypatch):
    import fast_acbf.solver as solver_module
    import numpy as np

    captured = {}

    class _FakeBFSolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(solver_module, "BFSolver", _FakeBFSolver)

    cfg = FastAcbfConfig(pad_width=4, upscale=1.5, upscale_method="nearest")
    solver = build_solver(cfg, np.zeros((1, 1, 2, 2), dtype=np.float32), "cpu")

    assert isinstance(solver, _FakeBFSolver)
    assert captured["pad_width"] == 4
    assert "fov" not in captured
    assert captured["upscale"] == 1.5
    assert captured["upscale_method"] == "nearest"


def test_worker_metric_uses_fast_acbf_metric_names():
    import numpy as np

    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert evaluate_metric(image, "normalized_std") > 0
    assert evaluate_metric(image, "laplacian") >= 0
    assert evaluate_metric(image, "sobel") >= 0
