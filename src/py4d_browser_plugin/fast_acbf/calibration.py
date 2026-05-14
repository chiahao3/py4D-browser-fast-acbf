"""Calibration helpers that talk to py4D-browser.

Scrapes voltage, scan step, detector pixel size, and detector radius from a
``DataViewer``/``DataCube`` calibration object and converts them into the
units fast-acbf expects (Angstrom, inverse Angstrom, mrad). Kept separate
from :mod:`config` so that the plugin's dataclass remains a pure data
container with no coupling to py4D-browser internals.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def electron_wavelength_angstrom(kv: float) -> float:
    """Relativistic electron wavelength in Angstrom for accelerating voltage in kV."""
    voltage = float(kv) * 1000.0
    if voltage <= 0:
        raise ValueError("Accelerating voltage must be positive.")
    h = 6.62607015e-34
    m0 = 9.1093837015e-31
    e = 1.602176634e-19
    c = 299792458.0
    wavelength_m = h / np.sqrt(2.0 * m0 * e * voltage * (1.0 + e * voltage / (2.0 * m0 * c * c)))
    return float(wavelength_m * 1e10)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _cal_get(calibration, getter_name: str, default: Any = None) -> Any:
    getter = getattr(calibration, getter_name, None)
    if getter is None:
        return default
    try:
        return getter()
    except Exception:
        return default


def normalize_length_to_angstrom(value: float, units: str | None) -> float:
    units = (units or "").strip().lower()
    if units in {"nm", "nanometer", "nanometers"}:
        return float(value) * 10.0
    return float(value)


def q_pixel_to_inv_angstrom(value: float, units: str | None, wavelength_angstrom: float) -> float:
    units = (units or "").strip().lower().replace(" ", "")
    if units in {"a^-1", "1/a", "angstrom^-1", "angstroms^-1", "å^-1", "1/å"}:
        return float(value)
    if units in {"nm^-1", "1/nm"}:
        return float(value) / 10.0
    if units == "mrad":
        return (float(value) / 1000.0) / float(wavelength_angstrom)
    return float(value)


def infer_voltage_kv(datacube, default: float) -> float:
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return float(default)
    try:
        value = calibration["voltage"]
    except Exception:
        value = None
    return _as_float(value, default)


def infer_scan_step_angstrom(datacube, default: float) -> float:
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return float(default)
    size = _as_float(_cal_get(calibration, "get_R_pixel_size", default), default)
    units = _cal_get(calibration, "get_R_pixel_units", "A")
    return normalize_length_to_angstrom(size, units)


def infer_dk_inv_angstrom(datacube, wavelength_angstrom: float, default: float) -> float:
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return float(default)
    size = _as_float(_cal_get(calibration, "get_Q_pixel_size", default), default)
    units = _cal_get(calibration, "get_Q_pixel_units", "A^-1")
    return q_pixel_to_inv_angstrom(size, units, wavelength_angstrom)


def infer_alpha_mrad_from_detector(parent, wavelength_angstrom: float, default: float) -> float:
    try:
        detector = parent.get_diffraction_detector()
    except Exception:
        return float(default)

    try:
        shape_name = detector["shape"].name
    except Exception:
        shape_name = str(detector.get("shape", ""))
    if shape_name != "CIRCLE":
        return float(default)

    radius_px = None
    geometry = detector.get("geometry")
    if isinstance(geometry, dict):
        radius_px = geometry.get("R")
    if radius_px is None:
        return float(default)

    dk = infer_dk_inv_angstrom(parent.datacube, wavelength_angstrom, default=np.nan)
    if not np.isfinite(dk):
        return float(default)
    return float(radius_px) * float(dk) * float(wavelength_angstrom) * 1000.0
