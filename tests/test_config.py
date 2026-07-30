import numpy as np
import pytest

from py4d_browser_plugin.fast_acbf.calibration import (
    auto_detect_bf_disk_px,
    infer_alpha_px_from_detector,
    is_calibration_unset,
    is_voltage_unset,
    resolve_max_alpha_px,
    resolved_wavelength_angstrom,
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
    assert cfg.lite_defocus_halfwidth_px is None
    assert cfg.lite_defocus_halfwidth_scan_fraction == 0.2
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
    def __init__(self, calibration, data=None):
        self.calibration = calibration
        if data is not None:
            self.data = data


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


def test_is_voltage_unset_true_when_missing_or_no_calibration():
    # _FakeCalibration has no __getitem__ at all -> not subscriptable -> unset
    assert is_voltage_unset(_FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))) is True
    assert is_voltage_unset(_FakeDatacube(None)) is True


def test_is_voltage_unset_false_for_positive_voltage():
    class _Cal:
        def __getitem__(self, key):
            if key == "voltage":
                return 300_000.0  # stored in Volts
            raise KeyError(key)

    assert is_voltage_unset(_FakeDatacube(_Cal())) is False


def test_is_voltage_unset_true_for_non_positive_voltage():
    class _Cal:
        def __getitem__(self, key):
            if key == "voltage":
                return 0.0
            raise KeyError(key)

    assert is_voltage_unset(_FakeDatacube(_Cal())) is True


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
    cfg = FastAcbfConfig(device="cuda", max_alpha_mrad=25.0, voltage_kv=300.0, wavelength_angstrom=0.019687)
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

    cfg = FastAcbfConfig(live_result_output="acBF", upscale_method="zero_insert", upscale=2.0)
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
    common = dict(max_alpha_mrad=25.0, voltage_kv=300.0, wavelength_angstrom=0.019687)
    base = FastAcbfConfig(pad_width=0, **common)
    changed = FastAcbfConfig(pad_width=3, upscale=2.0, upscale_method="nearest", **common)

    assert base.normalized_pad_width() is None
    assert changed.normalized_pad_width() == 3
    assert changed.output_pixel_size_angstrom() == 0.5
    assert base.solver_signature(data) != changed.solver_signature(data)


def test_zero_insert_requires_integer_upscale():
    cfg = FastAcbfConfig(upscale=1.5, upscale_method="zero_insert")

    with pytest.raises(ValueError, match="integer upscale factor"):
        cfg.validate_upscale_settings()


def test_zero_insert_is_coerced_for_acbf_modes():
    cfg = FastAcbfConfig(mode="acBF", upscale_method="zero_insert", upscale=2.0)

    messages = cfg.coerce_upscale_method_for_mode()

    assert cfg.upscale_method == "nearest"
    assert "not supported for acBF" in messages[0]


def test_zero_insert_rejected_for_acbf_refinement():
    cfg = FastAcbfConfig(refinement_mode="acBF", upscale_method="zero_insert", upscale=2.0)

    with pytest.raises(ValueError, match="only supported for tcBF"):
        cfg.validate_upscale_settings()


def test_zero_insert_allowed_for_acbf_when_upscale_is_one():
    # upscale=1.0 (the default) means zero_insert is a no-op regardless of method,
    # so it shouldn't be rejected or silently coerced away just because acBF is active.
    cfg = FastAcbfConfig(mode="acBF", upscale_method="zero_insert", upscale=1.0)

    cfg.validate_upscale_settings()  # must not raise
    messages = cfg.coerce_upscale_method_for_mode()

    assert cfg.upscale_method == "zero_insert"
    assert messages == []


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


def test_resolved_lite_defocus_halfwidth_uses_floor_for_small_scans():
    # scan_fraction=0.2 * 64 = 12.8, below the fixed floor of 20 -> floor wins.
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=None)
    assert cfg.resolved_lite_defocus_halfwidth_px(min_scan_dim=64) == pytest.approx(20.0)


def test_resolved_lite_defocus_halfwidth_scales_for_large_scans():
    # scan_fraction=0.2 * 256 = 51.2, above the floor -> proportional term wins.
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=None)
    assert cfg.resolved_lite_defocus_halfwidth_px(min_scan_dim=256) == pytest.approx(51.2)


def test_resolved_lite_defocus_halfwidth_respects_custom_scan_fraction():
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=None, lite_defocus_halfwidth_scan_fraction=0.5)
    assert cfg.resolved_lite_defocus_halfwidth_px(min_scan_dim=256) == pytest.approx(128.0)


def test_resolved_lite_defocus_halfwidth_respects_explicit_value_regardless_of_scan_size():
    # An explicit value is used as-is, even where the derived default (floor or
    # scale term) would differ -- no floor/scaling applied once the user has set one.
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=5.0)
    assert cfg.resolved_lite_defocus_halfwidth_px(min_scan_dim=1024) == pytest.approx(5.0)


def test_resolved_lite_defocus_halfwidth_seeded_scales_down_auto_derived_value():
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=None, lite_seeded_defocus_fraction=0.5)
    # unseeded: max(20, 0.2*256) = 51.2
    assert cfg.resolved_lite_defocus_halfwidth_px(min_scan_dim=256) == pytest.approx(51.2)
    # seeded: 51.2 * 0.5 = 25.6
    assert cfg.resolved_lite_defocus_halfwidth_px(
        min_scan_dim=256, seeded=True
    ) == pytest.approx(25.6)


def test_resolved_lite_defocus_halfwidth_seeded_has_no_effect_on_explicit_value():
    cfg = FastAcbfConfig(lite_defocus_halfwidth_px=5.0, lite_seeded_defocus_fraction=0.5)
    assert cfg.resolved_lite_defocus_halfwidth_px(
        min_scan_dim=1024, seeded=True
    ) == pytest.approx(5.0)


def test_build_solver_passes_fast_acbf_050_preparation_kwargs(monkeypatch):
    import fast_acbf.solver as solver_module
    import numpy as np

    captured = {}

    class _FakeBFSolver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(solver_module, "BFSolver", _FakeBFSolver)

    cfg = FastAcbfConfig(
        pad_width=4,
        upscale=1.5,
        upscale_method="nearest",
        max_alpha_mrad=25.0,
        voltage_kv=300.0,
        wavelength_angstrom=0.019687,
    )
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


def test_resolved_for_derives_max_alpha_from_px_when_calibration_unset():
    # calibration_free has no bearing on max_alpha resolution any more (see the
    # companion test below with calibration_free=False) -- left at its True default here.
    dc = _FakeDatacube(_FakeCalibration(1, "pixels", 1, "pixels"))
    parent = _CalibrationFreeParent(dc, radius_px=20.0)
    cfg = FastAcbfConfig(voltage_kv=300.0)

    resolved = cfg.resolved_for(parent)

    assert resolved.max_alpha_px == 20.0
    expected_mrad = 20.0 * resolved.dk_inv_angstrom * resolved.wavelength_angstrom * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_mrad)


def test_resolved_for_derives_max_alpha_from_px_even_with_real_calibration():
    # max_alpha resolution no longer waits for calibration to be unset: a precalibrated
    # dataset with a live circular selection (or an auto-fittable CBED) should get
    # max_alpha_px/mrad populated immediately, calibration_free notwithstanding, so
    # tcBF/Orientation never block on it.
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    parent = _CalibrationFreeParent(dc, radius_px=20.0)
    cfg = FastAcbfConfig(calibration_free=False, voltage_kv=300.0)

    resolved = cfg.resolved_for(parent)

    assert resolved.max_alpha_px == 20.0
    expected_mrad = 20.0 * resolved.dk_inv_angstrom * resolved.wavelength_angstrom * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_mrad)


def test_resolved_for_keeps_manual_alpha_when_px_cannot_be_resolved():
    # use_detector_alpha off and no CBED data to auto-fit from (_FakeDatacube has no
    # .data) -> max_alpha_px stays unresolved, so the manually-set mrad is preserved
    # untouched rather than being overwritten with a guess.
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    parent = _CalibrationFreeParent(dc, radius_px=20.0)
    cfg = FastAcbfConfig(calibration_free=True, use_detector_alpha=False, max_alpha_mrad=25.0)

    resolved = cfg.resolved_for(parent)

    assert resolved.max_alpha_px is None
    assert resolved.max_alpha_mrad == 25.0


def test_resolved_for_converts_and_clears_px_once_calibration_is_real_when_detector_disabled():
    # Characterize the current behavior: max_alpha_px acts as a calibration-free
    # marker. The calibrated branch converts it to mrad and clears it when detector
    # alpha resolution is disabled.
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


def test_resolved_for_live_detector_selection_overrides_persistent_px():
    # Same persistent-px setup, but now a live circular selection exists and
    # use_detector_alpha is on, so it should win over the previously-cached px.
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

    assert resolved.max_alpha_px == 8.0
    expected_mrad = 8.0 * resolved.dk_inv_angstrom * resolved.wavelength_angstrom * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_mrad)


def test_resolved_for_keeps_manual_alpha_when_detector_alpha_disabled_and_no_data():
    class _Parent:
        def __init__(self, datacube):
            self.datacube = datacube

        def get_diffraction_detector(self):
            raise AssertionError("should not be consulted when use_detector_alpha is off")

    dc = _FakeDatacube(_FakeCalibration(1, "pixels", 1, "pixels"))
    cfg = FastAcbfConfig(calibration_free=False, use_detector_alpha=False, max_alpha_mrad=25.0)

    resolved = cfg.resolved_for(_Parent(dc))

    assert resolved.max_alpha_mrad == 25.0
    assert resolved.max_alpha_px is None


def test_resolved_for_calibration_free_circle_uses_placeholder_without_voltage():
    dc = _FakeDatacube(_FakeCalibration(1, "pixels", 1, "pixels"))
    parent = _CalibrationFreeParent(dc, radius_px=20.0)

    resolved = FastAcbfConfig().resolved_for(parent)

    assert resolved.voltage_kv is None
    assert resolved.wavelength_angstrom is None
    assert resolved.max_alpha_px == 20.0
    assert resolved.max_alpha_mrad == pytest.approx(
        20.0
        * resolved.dk_inv_angstrom
        * resolved_wavelength_angstrom(None)
        * 1000.0
    )


def test_resolved_for_calibration_free_auto_detects_without_voltage():
    class _Parent:
        def __init__(self, datacube):
            self.datacube = datacube

        def get_diffraction_detector(self):
            return {"shape": _RectShape(), "geometry": {}}

    source = _make_bf_datacube(radius=9.0, size=48)
    dc = _FakeDatacube(
        _FakeCalibration(1, "pixels", 1, "pixels"),
        data=source.data,
    )

    resolved = FastAcbfConfig().resolved_for(_Parent(dc))

    assert resolved.wavelength_angstrom is None
    assert resolved.max_alpha_px == pytest.approx(9.0, abs=1.5)
    assert resolved.max_alpha_mrad == pytest.approx(
        resolved.max_alpha_px
        * resolved.dk_inv_angstrom
        * resolved_wavelength_angstrom(None)
        * 1000.0
    )


def test_resolved_for_failed_calibration_free_detection_preserves_unset_alpha():
    class _Parent:
        def __init__(self, datacube):
            self.datacube = datacube

        def get_diffraction_detector(self):
            return {"shape": _RectShape(), "geometry": {}}

    dc = _FakeDatacube(
        _FakeCalibration(1, "pixels", 1, "pixels"),
        data=np.ones((1, 1, 16, 16), dtype=np.float32),
    )

    resolved = FastAcbfConfig().resolved_for(_Parent(dc))

    assert resolved.max_alpha_px is None
    assert resolved.max_alpha_mrad is None


def test_resolved_for_failed_calibration_free_detection_preserves_manual_alpha():
    class _Parent:
        def __init__(self, datacube):
            self.datacube = datacube

        def get_diffraction_detector(self):
            return {"shape": _RectShape(), "geometry": {}}

    dc = _FakeDatacube(
        _FakeCalibration(1, "pixels", 1, "pixels"),
        data=np.ones((1, 1, 16, 16), dtype=np.float32),
    )

    resolved = FastAcbfConfig(max_alpha_mrad=25.0).resolved_for(_Parent(dc))

    assert resolved.max_alpha_px is None
    assert resolved.max_alpha_mrad == 25.0


def test_resolved_for_calibrated_axes_without_voltage_uses_placeholder_for_detector_alpha():
    dc = _FakeDatacube(_FakeCalibration(0.2, "A", 0.01, "A^-1"))
    parent = _CalibrationFreeParent(dc, radius_px=12.0)

    resolved = FastAcbfConfig().resolved_for(parent)

    assert resolved.voltage_kv is None
    assert resolved.wavelength_angstrom is None
    assert resolved.max_alpha_px == 12.0
    assert resolved.max_alpha_mrad == pytest.approx(
        12.0 * 0.01 * resolved_wavelength_angstrom(None) * 1000.0
    )
