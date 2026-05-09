import pytest

from py4d_browser_plugin.fast_acbf.config import (
    FastAcbfConfig,
    electron_wavelength_angstrom,
    label_dict_to_fast_acbf,
)
from py4d_browser_plugin.fast_acbf.worker import evaluate_metric


def test_electron_wavelength_300kv():
    assert electron_wavelength_angstrom(300) == pytest.approx(0.019687, abs=1e-6)


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


def test_worker_metric_uses_fast_acbf_metric_names():
    import numpy as np

    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert evaluate_metric(image, "normalized_std") > 0
    assert evaluate_metric(image, "laplacian") >= 0
    assert evaluate_metric(image, "sobel") >= 0

