from py4d_browser_plugin.fast_acbf.live_view.metadata import (
    APPLY_METADATA_KEYS,
    MetadataAdapter,
    normalize_metadata_value,
)


def test_apply_metadata_keys_match_solver_contract():
    expected = {
        "wavelength",
        "max_alpha",
        "dk",
        "scan_shape",
        "scan_step_size",
        "rotation_deg",
        "flipud",
        "fliplr",
        "transpose",
    }
    assert set(APPLY_METADATA_KEYS) == expected


def test_first_call_returns_full_state():
    adapter = MetadataAdapter()
    state = {"rotation_deg": 30.0, "flipud": False, "max_alpha": 25.0}
    assert adapter.diff(state) == {
        "rotation_deg": 30.0,
        "flipud": False,
        "max_alpha": 25.0,
    }


def test_diff_returns_empty_when_unchanged():
    adapter = MetadataAdapter()
    state = {"rotation_deg": 30.0, "flipud": False, "scan_step_size": 0.2}
    adapter.diff(state)
    assert adapter.diff(state) == {}


def test_diff_returns_only_changed_keys():
    adapter = MetadataAdapter()
    adapter.diff({"rotation_deg": 30.0, "flipud": False})
    delta = adapter.diff({"rotation_deg": 31.0, "flipud": False})
    assert delta == {"rotation_deg": 31.0}


def test_diff_float_tolerance_below_precision():
    adapter = MetadataAdapter()
    adapter.diff({"rotation_deg": 30.0})
    assert adapter.diff({"rotation_deg": 30.0 + 1e-12}) == {}


def test_diff_float_above_precision_is_change():
    adapter = MetadataAdapter()
    adapter.diff({"rotation_deg": 30.0})
    assert adapter.diff({"rotation_deg": 30.001}) == {"rotation_deg": 30.001}


def test_diff_ignores_unknown_keys():
    adapter = MetadataAdapter()
    delta = adapter.diff({"rotation_deg": 30.0, "vendor_specific": "ignored"})
    assert delta == {"rotation_deg": 30.0}


def test_scan_shape_normalized_to_tuple_and_compared_structurally():
    adapter = MetadataAdapter()
    assert adapter.diff({"scan_shape": [4, 5]}) == {"scan_shape": (4, 5)}
    assert adapter.diff({"scan_shape": (4, 5)}) == {}


def test_bool_keys_normalized():
    adapter = MetadataAdapter()
    assert adapter.diff({"flipud": 1, "transpose": 0}) == {
        "flipud": True,
        "transpose": False,
    }
    assert adapter.diff({"flipud": True, "transpose": False}) == {}


def test_reset_clears_state():
    adapter = MetadataAdapter()
    adapter.diff({"rotation_deg": 30.0})
    adapter.reset()
    assert adapter.diff({"rotation_deg": 30.0}) == {"rotation_deg": 30.0}


def test_partial_update_preserves_unmentioned_keys():
    adapter = MetadataAdapter()
    adapter.diff({"rotation_deg": 30.0, "flipud": True})
    adapter.diff({"rotation_deg": 30.0})
    assert adapter.last_state == {"rotation_deg": 30.0, "flipud": True}


def test_normalize_metadata_value_passthrough_for_unknown_key():
    assert normalize_metadata_value("custom", "abc") == "abc"
