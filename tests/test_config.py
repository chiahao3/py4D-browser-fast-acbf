import numpy as np
import pytest

from py4d_browser_plugin.fast_acbf.calibration import (
    auto_detect_bf_disk_px,
    infer_alpha_px_from_detector,
    is_calibration_unset,
    resolve_max_alpha_px,
)
from py4d_browser_plugin.fast_acbf.config import (
    FastAcbfConfig,
    VALID_LITE_ABERRATION_SEARCH,
    label_dict_to_fast_acbf,
    lite_search_order,
)
from py4d_browser_plugin.fast_acbf.utils import build_solver
from py4d_browser_plugin.fast_acbf.worker import evaluate_metric


def test_lite_config_defaults():
    cfg = FastAcbfConfig()
    assert cfg.lite_output_target == "virtual_image"
    assert cfg.lite_aberration_search == "first_order"
    assert cfg.lite_defocus_halfwidth_px == 20.0
    cfg.validate_lite_settings()


def test_lite_search_order_mapping():
    assert lite_search_order("disabled") == 0
    assert lite_search_order("df_only") == 1
    assert lite_search_order("first_order") == 1
    assert lite_search_order("second_order") == 2
    # every valid level maps to a known order
    assert all(lite_search_order(v) in (0, 1, 2) for v in VALID_LITE_ABERRATION_SEARCH)


def test_validate_lite_settings_rejects_bad_values():
    cfg = FastAcbfConfig(lite_aberration_search="nonsense")
    with pytest.raises(ValueError):
        cfg.validate_lite_settings()
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=0.0)
    with pytest.raises(ValueError):
        cfg.validate_lite_settings()


class _FakeCalibration:
    def __init__(self, r_size, r_units, q_size, q_units):
        self._r_size, self._r_units = r_size, r_units
        self._q_size, self._q_units = q_size, q_units

    def get_R_pixel_size(self):
        return self._r_size

    def get_R_pixel_units(self):
        return self._r_units

    def get_Q_pixel_size(self):
        return self._q_size

    def get_Q_pixel_units(self):
        return self._q_units


class _FakeDatacube:
    def __init__(self, calibration):
        self.calibration = calibration


def test_is_calibration_unset_detects_pixel_defaults():
    # py4DSTEM default: pixel units, size 1 -> unset
    dc = _FakeDatacube(_FakeCalibration(1, "pixels", 1, "pixels"))
    assert is_calibration_unset(dc) is True
    # no datacube / no calibration -> unset
    assert is_calibration_unset(_FakeDatacube(None)) is True


def test_is_calibration_unset_recognises_real_calibration():
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    assert is_calibration_unset(dc) is False


def test_is_calibration_unset_when_either_axis_unset():
    # real space calibrated but diffraction still at pixel default -> unset
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 1, "pixels"))
    assert is_calibration_unset(dc) is True


class _CircleShape:
    name = "CIRCLE"


class _RectShape:
    name = "RECTANGULAR"


def _make_bf_datacube(radius=10.0, size=64, bg=1.0, fg=100.0):
    yy, xx = np.mgrid[0:size, 0:size]
    cy = cx = (size - 1) / 2.0
    disk = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    pattern = np.full((size, size), bg, dtype=np.float32)
    pattern[disk] = fg
    data = np.tile(pattern, (2, 2, 1, 1)).astype(np.float32)
    return _FakeDatacube2(data)


class _FakeDatacube2:
    def __init__(self, data):
        self.data = data


def test_infer_alpha_px_from_detector_reads_circle_geometry():
    class _Parent:
        def get_diffraction_detector(self):
            return {"shape": _CircleShape(), "geometry": {"R": 12.5}}

    assert infer_alpha_px_from_detector(_Parent()) == 12.5


def test_infer_alpha_px_from_detector_returns_none_without_circle_selection():
    class _Parent:
        def get_diffraction_detector(self):
            return {"shape": _RectShape(), "geometry": {}}

    assert infer_alpha_px_from_detector(_Parent()) is None
    assert infer_alpha_px_from_detector(object()) is None


def test_auto_detect_bf_disk_px_finds_bright_central_disk():
    dc = _make_bf_datacube(radius=10.0, size=64)
    result = auto_detect_bf_disk_px(dc)
    assert result is not None
    radius_px, center_y_px, center_x_px = result
    assert radius_px == pytest.approx(10.0, abs=1.5)
    assert center_y_px == pytest.approx(31.5, abs=1.0)
    assert center_x_px == pytest.approx(31.5, abs=1.0)


def test_auto_detect_bf_disk_px_returns_none_for_featureless_pattern():
    dc = _FakeDatacube2(np.ones((1, 1, 16, 16), dtype=np.float32))
    assert auto_detect_bf_disk_px(dc) is None


def test_resolve_max_alpha_px_prefers_manual_circle_over_auto():
    class _Parent:
        def get_diffraction_detector(self):
            return {"shape": _CircleShape(), "geometry": {"R": 7.0}}

    dc = _make_bf_datacube(radius=10.0, size=64)
    assert resolve_max_alpha_px(_Parent(), dc) == 7.0


def test_resolve_max_alpha_px_falls_back_to_auto_and_caches_on_datacube():
    class _Parent:
        def get_diffraction_detector(self):
            raise RuntimeError("no detector selection")

    dc = _make_bf_datacube(radius=10.0, size=64)
    radius_px = resolve_max_alpha_px(_Parent(), dc)
    assert radius_px == pytest.approx(10.0, abs=1.5)
    assert dc._fast_acbf_auto_alpha_px == pytest.approx(radius_px)
    # second call hits the cache rather than rescanning the dataset
    assert resolve_max_alpha_px(_Parent(), dc) == radius_px


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


def test_fast_acbf_060_preparation_defaults_and_kwargs():
    cfg = FastAcbfConfig()
    kwargs = cfg.reconstruct_kwargs()

    assert cfg.normalized_pad_width() is None
    assert cfg.output_pixel_size_angstrom() == 1.0
    assert cfg.live_virtual_output == "None"
    assert cfg.live_result_output == "tcBF"
    assert cfg.live_auto_focus_interval_s == 5.0
    assert cfg.live_auto_aberrations_interval_s == 30.0
    assert kwargs["pad_width"] is None
    assert kwargs["upscale"] == 1.0
    assert kwargs["upscale_method"] == "zero_insert"


def test_live_view_output_settings_are_validated_and_affect_acbf_upscale():
    cfg = FastAcbfConfig(live_virtual_output="probe", live_result_output="chi")
    cfg.validate_live_output_settings()
    assert cfg.normalized_live_output("acbf") == "acBF"

    cfg.live_result_output = "not-a-panel-output"
    with pytest.raises(ValueError, match="Live result output"):
        cfg.validate_live_output_settings()

    cfg = FastAcbfConfig(live_result_output="acBF", upscale_method="zero_insert")
    messages = cfg.coerce_upscale_method_for_mode()
    assert cfg.upscale_method == "nearest"
    assert "not supported for acBF" in messages[0]


def test_live_view_auto_refinement_intervals_are_validated():
    cfg = FastAcbfConfig(live_auto_focus_interval_s=0.0)
    with pytest.raises(ValueError, match="Auto Focus interval"):
        cfg.validate_live_auto_refinement_settings()

    cfg = FastAcbfConfig(live_auto_aberrations_interval_s=-1.0)
    with pytest.raises(ValueError, match="Auto Aberrations interval"):
        cfg.validate_upscale_settings()


def test_fast_acbf_060_preparation_signature_and_pad_normalization():
    data = type("ArrayLike", (), {"shape": (1, 2, 3, 4), "dtype": "float32"})()
    base = FastAcbfConfig(pad_width=0)
    changed = FastAcbfConfig(pad_width=3, upscale=2.0, upscale_method="nearest")

    assert base.normalized_pad_width() is None
    assert changed.normalized_pad_width() == 3
    assert changed.output_pixel_size_angstrom() == 0.5
    assert base.solver_signature(data) != changed.solver_signature(data)


def test_zero_insert_requires_integer_upscale():
    cfg = FastAcbfConfig(upscale=1.5, upscale_method="zero_insert")

    with pytest.raises(ValueError, match="integer upscale factor"):
        cfg.validate_upscale_settings()


def test_zero_insert_is_coerced_for_acbf_modes():
    cfg = FastAcbfConfig(mode="acBF", upscale_method="zero_insert")

    messages = cfg.coerce_upscale_method_for_mode()

    assert cfg.upscale_method == "nearest"
    assert "not supported for acBF" in messages[0]


def test_zero_insert_rejected_for_acbf_refinement():
    cfg = FastAcbfConfig(refinement_mode="acBF", upscale_method="zero_insert")

    with pytest.raises(ValueError, match="only supported for tcBF"):
        cfg.validate_upscale_settings()


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


def test_resolved_fine_rotation_halfwidth_derives_from_rotation_points_when_unset():
    cfg = FastAcbfConfig(rotation_points=12, fine_rotation_halfwidth_deg=None)
    assert cfg.resolved_fine_rotation_halfwidth_deg() == pytest.approx(15.0)

    cfg = FastAcbfConfig(rotation_points=18, fine_rotation_halfwidth_deg=None)
    assert cfg.resolved_fine_rotation_halfwidth_deg() == pytest.approx(10.0)


def test_resolved_fine_rotation_halfwidth_respects_explicit_value_narrower_than_derived():
    # derived default would be 180/12 = 15.0, but an explicit value is used as-is,
    # even when narrower -- no floor is applied once the user has set one.
    cfg = FastAcbfConfig(rotation_points=12, fine_rotation_halfwidth_deg=2.0)
    assert cfg.resolved_fine_rotation_halfwidth_deg() == pytest.approx(2.0)


def test_resolved_fine_rotation_halfwidth_respects_explicit_value_wider_than_derived():
    cfg = FastAcbfConfig(rotation_points=12, fine_rotation_halfwidth_deg=25.0)
    assert cfg.resolved_fine_rotation_halfwidth_deg() == pytest.approx(25.0)


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


class _CalibrationFreeParent:
    def __init__(self, datacube, radius_px=20.0):
        self.datacube = datacube
        self._radius_px = radius_px

    def get_diffraction_detector(self):
        return {"shape": _CircleShape(), "geometry": {"R": self._radius_px}}


def test_resolved_for_calibration_free_derives_max_alpha_from_px():
    dc = _FakeDatacube(_FakeCalibration(1, "pixels", 1, "pixels"))
    parent = _CalibrationFreeParent(dc, radius_px=20.0)
    cfg = FastAcbfConfig(calibration_free=True, voltage_kv=300.0)

    resolved = cfg.resolved_for(parent)

    assert resolved.max_alpha_px == 20.0
    expected_mrad = 20.0 * resolved.dk_inv_angstrom * resolved.wavelength_angstrom * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_mrad)


def test_resolved_for_ignores_calibration_free_once_calibration_is_real():
    # Real calibration -> is_calibration_unset is False, so the normal
    # use_detector_alpha/infer_alpha_mrad_from_detector path runs instead, even
    # though calibration_free is still enabled.
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    parent = _CalibrationFreeParent(dc, radius_px=20.0)
    cfg = FastAcbfConfig(calibration_free=True, use_detector_alpha=False, max_alpha_mrad=25.0)

    resolved = cfg.resolved_for(parent)

    assert resolved.max_alpha_px is None
    assert resolved.max_alpha_mrad == 25.0


def test_resolved_for_refreshes_stale_max_alpha_mrad_once_calibration_is_real():
    # Simulates config state persisted from a prior calibration-free run: max_alpha_px
    # measured in pixels, and max_alpha_mrad derived under placeholder dk/wavelength
    # that no longer matches the real calibration below.
    class _Parent:
        def __init__(self, datacube):
            self.datacube = datacube

        def get_diffraction_detector(self):
            raise AssertionError("no circular selection needed for this test")

    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    stale_cfg = FastAcbfConfig(
        calibration_free=True,
        use_detector_alpha=False,
        max_alpha_px=20.0,
        max_alpha_mrad=1234.5,  # stale: computed earlier under placeholder dk/wavelength
        dk_inv_angstrom=1.0,
        wavelength_angstrom=0.0197,
    )

    resolved = stale_cfg.resolved_for(_Parent(dc))

    assert resolved.max_alpha_px is None
    expected_mrad = 20.0 * resolved.dk_inv_angstrom * resolved.wavelength_angstrom * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_mrad)
    assert resolved.max_alpha_mrad != pytest.approx(1234.5)


def test_resolved_for_live_detector_selection_overrides_stale_px_refresh():
    # Same stale-state setup, but now a live circular selection exists and
    # use_detector_alpha is on, so it should win over the px-based refresh.
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    parent = _CalibrationFreeParent(dc, radius_px=8.0)
    stale_cfg = FastAcbfConfig(
        calibration_free=True,
        use_detector_alpha=True,
        max_alpha_px=20.0,
        max_alpha_mrad=1234.5,
        dk_inv_angstrom=1.0,
        wavelength_angstrom=0.0197,
    )

    resolved = stale_cfg.resolved_for(parent)

    assert resolved.max_alpha_px is None
    expected_mrad = 8.0 * resolved.dk_inv_angstrom * resolved.wavelength_angstrom * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_mrad)


def test_resolved_for_calibration_free_disabled_keeps_default_alpha():
    class _Parent:
        def __init__(self, datacube):
            self.datacube = datacube

        def get_diffraction_detector(self):
            raise AssertionError("should not be consulted when calibration_free is off")

    dc = _FakeDatacube(_FakeCalibration(1, "pixels", 1, "pixels"))
    cfg = FastAcbfConfig(calibration_free=False, use_detector_alpha=False, max_alpha_mrad=25.0)

    resolved = cfg.resolved_for(_Parent(dc))

    assert resolved.max_alpha_mrad == 25.0
    assert resolved.max_alpha_px is None
