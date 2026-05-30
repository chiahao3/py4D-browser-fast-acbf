"""Tests for py4d_browser_plugin.fast_acbf.calibration.

Covers the helpers that scrape py4D-browser calibration state and convert
units into the conventions fast-acbf expects (Angstrom, inverse Angstrom).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from py4d_browser_plugin.fast_acbf.calibration import (
    electron_wavelength_angstrom,
    infer_alpha_mrad_from_detector,
    infer_dk_inv_angstrom,
    infer_scan_step_angstrom,
    infer_voltage_kv,
    normalize_length_to_angstrom,
    q_pixel_to_inv_angstrom,
    sync_config_to_datacube_calibration,
)


# --- unit conversion / wavelength ------------------------------------------------


def test_electron_wavelength_300kv_matches_known_value():
    assert electron_wavelength_angstrom(300) == pytest.approx(0.019687, abs=1e-6)


def test_electron_wavelength_200kv_matches_known_value():
    assert electron_wavelength_angstrom(200) == pytest.approx(0.02508, abs=1e-4)


def test_electron_wavelength_rejects_non_positive_voltage():
    with pytest.raises(ValueError):
        electron_wavelength_angstrom(0)
    with pytest.raises(ValueError):
        electron_wavelength_angstrom(-100)


@pytest.mark.parametrize(
    "units, value, expected",
    [
        ("A", 1.5, 1.5),
        ("Å", 1.5, 1.5),
        ("nm", 1.5, 15.0),
        ("nanometer", 0.5, 5.0),
        ("Nanometers", 2.0, 20.0),
        (None, 1.5, 1.5),
        ("", 1.5, 1.5),
    ],
)
def test_normalize_length_to_angstrom(units, value, expected):
    assert normalize_length_to_angstrom(value, units) == pytest.approx(expected)


@pytest.mark.parametrize(
    "units, value, wavelength, expected",
    [
        ("A^-1", 0.05, 0.025, 0.05),
        ("1/A", 0.05, 0.025, 0.05),
        ("Å^-1", 0.05, 0.025, 0.05),
        ("1/Å", 0.05, 0.025, 0.05),
        ("nm^-1", 0.5, 0.025, 0.05),
        ("1/nm", 0.5, 0.025, 0.05),
        ("mrad", 1.0, 0.025, 1.0 / 1000.0 / 0.025),
        ("unknown", 0.05, 0.025, 0.05),
        (None, 0.05, 0.025, 0.05),
    ],
)
def test_q_pixel_to_inv_angstrom(units, value, wavelength, expected):
    assert q_pixel_to_inv_angstrom(value, units, wavelength) == pytest.approx(expected)


# --- calibration inference (mocked datacube/parent) ------------------------------


def _make_calibration(
    *,
    voltage=None,
    r_size=None,
    r_units=None,
    q_size=None,
    q_units=None,
):
    cal = SimpleNamespace()
    if voltage is not None:
        cal.__getitem__ = lambda self, key, _v=voltage: _v if key == "voltage" else (_ for _ in ()).throw(KeyError(key))
    if r_size is not None:
        cal.get_R_pixel_size = lambda _s=r_size: _s
    if r_units is not None:
        cal.get_R_pixel_units = lambda _u=r_units: _u
    if q_size is not None:
        cal.get_Q_pixel_size = lambda _s=q_size: _s
    if q_units is not None:
        cal.get_Q_pixel_units = lambda _u=q_units: _u
    return cal


def test_infer_voltage_kv_reads_from_calibration():
    class _Cal:
        def __getitem__(self, key):
            if key == "voltage":
                return 200.0
            raise KeyError(key)

    datacube = SimpleNamespace(calibration=_Cal())
    assert infer_voltage_kv(datacube, default=80.0) == 200.0


def test_infer_voltage_kv_falls_back_to_default_on_missing():
    class _Cal:
        def __getitem__(self, key):
            raise KeyError(key)

    datacube = SimpleNamespace(calibration=_Cal())
    assert infer_voltage_kv(datacube, default=80.0) == 80.0


def test_infer_voltage_kv_handles_no_calibration():
    datacube = SimpleNamespace()
    assert infer_voltage_kv(datacube, default=120.0) == 120.0


def test_infer_scan_step_angstrom_passthrough_for_angstrom_units():
    cal = SimpleNamespace(
        get_R_pixel_size=lambda: 0.25,
        get_R_pixel_units=lambda: "A",
    )
    datacube = SimpleNamespace(calibration=cal)
    assert infer_scan_step_angstrom(datacube, default=1.0) == pytest.approx(0.25)


def test_infer_scan_step_angstrom_converts_nm_to_angstrom():
    cal = SimpleNamespace(
        get_R_pixel_size=lambda: 0.025,
        get_R_pixel_units=lambda: "nm",
    )
    datacube = SimpleNamespace(calibration=cal)
    assert infer_scan_step_angstrom(datacube, default=1.0) == pytest.approx(0.25)


def test_infer_scan_step_angstrom_uses_default_when_getter_raises():
    cal = SimpleNamespace(
        get_R_pixel_size=lambda: (_ for _ in ()).throw(RuntimeError("nope")),
        get_R_pixel_units=lambda: "A",
    )
    datacube = SimpleNamespace(calibration=cal)
    assert infer_scan_step_angstrom(datacube, default=0.3) == 0.3


def test_infer_dk_inv_angstrom_passthrough():
    cal = SimpleNamespace(
        get_Q_pixel_size=lambda: 0.05,
        get_Q_pixel_units=lambda: "A^-1",
    )
    datacube = SimpleNamespace(calibration=cal)
    assert infer_dk_inv_angstrom(datacube, wavelength_angstrom=0.025, default=0.01) == pytest.approx(0.05)


def test_infer_dk_inv_angstrom_from_mrad():
    cal = SimpleNamespace(
        get_Q_pixel_size=lambda: 1.0,
        get_Q_pixel_units=lambda: "mrad",
    )
    datacube = SimpleNamespace(calibration=cal)
    # 1 mrad at lambda=0.025 A -> 0.001 / 0.025 = 0.04 1/A
    assert infer_dk_inv_angstrom(datacube, wavelength_angstrom=0.025, default=0.0) == pytest.approx(0.04)


def test_infer_alpha_mrad_from_detector_with_circle():
    cal = SimpleNamespace(
        get_Q_pixel_size=lambda: 0.05,
        get_Q_pixel_units=lambda: "A^-1",
    )
    datacube = SimpleNamespace(calibration=cal)

    class _DetShape:
        name = "CIRCLE"

    parent = SimpleNamespace(
        datacube=datacube,
        get_diffraction_detector=lambda: {"shape": _DetShape(), "geometry": {"R": 10.0}},
    )
    # 10 px * 0.05 1/A * 0.025 A * 1000 = 12.5 mrad
    assert infer_alpha_mrad_from_detector(parent, wavelength_angstrom=0.025, default=99.0) == pytest.approx(12.5)


def test_infer_alpha_mrad_falls_back_for_non_circular_detector():
    class _DetShape:
        name = "RECTANGLE"

    parent = SimpleNamespace(
        get_diffraction_detector=lambda: {"shape": _DetShape(), "geometry": {}},
    )
    assert infer_alpha_mrad_from_detector(parent, wavelength_angstrom=0.025, default=42.0) == 42.0


def test_infer_alpha_mrad_falls_back_when_detector_call_raises():
    def _boom():
        raise RuntimeError("no detector")

    parent = SimpleNamespace(get_diffraction_detector=_boom)
    assert infer_alpha_mrad_from_detector(parent, wavelength_angstrom=0.025, default=42.0) == 42.0


# --- config integration ----------------------------------------------------------


def test_config_resolved_for_pulls_from_calibration():
    from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig

    class _Cal:
        def __getitem__(self, key):
            if key == "voltage":
                return 200.0
            raise KeyError(key)

        def get_R_pixel_size(self):
            return 0.5

        def get_R_pixel_units(self):
            return "A"

        def get_Q_pixel_size(self):
            return 0.04

        def get_Q_pixel_units(self):
            return "A^-1"

    class _DetShape:
        name = "CIRCLE"

    datacube = SimpleNamespace(calibration=_Cal())
    parent = SimpleNamespace(
        datacube=datacube,
        get_diffraction_detector=lambda: {"shape": _DetShape(), "geometry": {"R": 12.0}},
    )

    cfg = FastAcbfConfig(voltage_kv=80.0, use_calibration=True, use_detector_alpha=True)
    resolved = cfg.resolved_for(parent)

    assert resolved.voltage_kv == 200.0
    assert resolved.wavelength_angstrom == pytest.approx(electron_wavelength_angstrom(200.0))
    assert resolved.scan_step_angstrom == pytest.approx(0.5)
    assert resolved.dk_inv_angstrom == pytest.approx(0.04)
    # 12 * 0.04 * lambda(200kV) * 1000
    expected_alpha = 12.0 * 0.04 * electron_wavelength_angstrom(200.0) * 1000.0
    assert resolved.max_alpha_mrad == pytest.approx(expected_alpha)


def test_sync_config_to_datacube_calibration_writes_py4d_fields():
    from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig

    class _Cal:
        def __init__(self):
            self.values = {}

        def set_R_pixel_size(self, value):
            self.values["r_size"] = value

        def set_R_pixel_units(self, value):
            self.values["r_units"] = value

        def set_Q_pixel_size(self, value):
            self.values["q_size"] = value

        def set_Q_pixel_units(self, value):
            self.values["q_units"] = value

        def __setitem__(self, key, value):
            self.values[key] = value

    cal = _Cal()
    datacube = SimpleNamespace(calibration=cal)
    cfg = FastAcbfConfig(
        scan_step_angstrom=3.0,
        dk_inv_angstrom=0.125,
        voltage_kv=200.0,
    )

    sync_config_to_datacube_calibration(datacube, cfg)

    assert cal.values == {
        "r_size": 3.0,
        "r_units": "A",
        "q_size": 0.125,
        "q_units": "A^-1",
        "voltage": 200.0,
    }
