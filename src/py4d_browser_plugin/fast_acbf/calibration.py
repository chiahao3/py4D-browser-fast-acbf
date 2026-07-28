"""Calibration helpers that talk to py4D-browser.

Scrapes voltage, scan step, detector pixel size, and detector radius from a
``DataViewer``/``DataCube`` calibration object and converts them into the
units fast-acbf expects (Angstrom, inverse Angstrom, mrad). Kept separate
from :mod:`config` so that the plugin's dataclass remains a pure data
container with no coupling to py4D-browser internals.
"""

from __future__ import annotations

import math

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


# Used only to convert a calibration-free BF-disk radius (px) into mrad for display when no
# real voltage is known. Not physically meaningful on its own -- only the product
# max_alpha_px * dk * wavelength matters in that path, and both dk and wavelength are already
# placeholders there -- so this must never be surfaced as the user's actual accelerating voltage.
PLACEHOLDER_WAVELENGTH_ANGSTROM = electron_wavelength_angstrom(300.0)


def resolved_wavelength_angstrom(wavelength_angstrom: float | None) -> float:
    """The wavelength to use for internal max_alpha/dk unit conversions: the real
    value once voltage is known, else the calibration-free placeholder above."""
    return float(wavelength_angstrom) if wavelength_angstrom is not None else PLACEHOLDER_WAVELENGTH_ANGSTROM


def max_alpha_mrad_from_px(max_alpha_px: float, dk_inv_angstrom: float, wavelength_angstrom: float) -> float:
    """Convert a BF-disk radius in raw detector pixels to mrad under the given dk/wavelength."""
    return float(max_alpha_px) * float(dk_inv_angstrom) * float(wavelength_angstrom) * 1000.0


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


def infer_voltage_kv(datacube, default: float | None) -> float | None:
    """Accelerating voltage in kV, converted from the Volts stored in py4D calibration."""
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return default
    try:
        value_volts = calibration["voltage"]
    except Exception:
        return default
    try:
        return float(value_volts) / 1000.0
    except Exception:
        return default


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


def infer_alpha_px_from_detector(parent) -> float | None:
    """BF-disk radius in raw detector pixels from a user-drawn circular selection.

    Needs no ``dk``/wavelength, so it stays meaningful even when the datacube
    calibration is completely unset.
    """
    try:
        detector = parent.get_diffraction_detector()
    except Exception:
        return None

    try:
        shape_name = detector["shape"].name
    except Exception:
        shape_name = str(detector.get("shape", ""))
    if shape_name != "CIRCLE":
        return None

    geometry = detector.get("geometry")
    radius_px = geometry.get("R") if isinstance(geometry, dict) else None
    if radius_px is None:
        return None
    radius_px = float(radius_px)
    return radius_px if radius_px > 0 else None


def auto_detect_bf_disk_px(datacube) -> tuple[float, float, float] | None:
    """Estimate (radius_px, center_y_px, center_x_px) of the BF disk, calibration-free.

    Squares the position-averaged CBED to suppress the (comparatively weak) dark-field
    and diffuse background relative to the bright central disk, Otsu-thresholds the
    result, and fits a circle via the equivalent-area radius and centroid of the bright
    region nearest the detector center. Returns ``None`` if no usable disk is found.
    """
    data = getattr(datacube, "data", None)
    if data is None:
        return None
    try:
        mean_cbed = np.asarray(data).mean(axis=(0, 1))
    except Exception:
        return None
    if mean_cbed.ndim != 2 or mean_cbed.size == 0:
        return None

    squared = np.square(mean_cbed.astype(np.float64))
    finite = squared[np.isfinite(squared)]
    if finite.size == 0 or float(finite.max()) <= 0:
        return None

    try:
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops
    except ImportError:
        return None

    try:
        threshold = threshold_otsu(squared)
    except Exception:
        return None
    mask = squared > threshold
    if not mask.any():
        return None

    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        return None

    # Prefer the bright region that actually contains the detector center (the true
    # BF disk); fall back to the largest bright region if none does.
    center_row = (mean_cbed.shape[0] - 1) / 2.0
    center_col = (mean_cbed.shape[1] - 1) / 2.0
    region = None
    for candidate in regions:
        min_row, min_col, max_row, max_col = candidate.bbox
        if min_row <= center_row < max_row and min_col <= center_col < max_col:
            region = candidate
            break
    if region is None:
        region = max(regions, key=lambda r: r.area)

    radius_px = float(np.sqrt(region.area / np.pi))
    if radius_px <= 0:
        return None
    center_y_px, center_x_px = (float(v) for v in region.centroid)
    return radius_px, center_y_px, center_x_px


def resolve_max_alpha_px(parent, datacube, *, use_detector: bool = True) -> float | None:
    """BF-disk radius in pixels, calibration-free: prefer a circular detector
    selection drawn by the user (unless ``use_detector`` is False); otherwise
    auto-detect from the mean CBED.

    Cached on the datacube instance (auto-invalidated whenever a new datacube is
    loaded) since the auto-detect path scans the full dataset.
    """
    if use_detector:
        manual = infer_alpha_px_from_detector(parent)
        if manual is not None:
            return manual

    cached = getattr(datacube, "_fast_acbf_auto_alpha_px", None)
    if cached is not None:
        return float(cached)

    auto = auto_detect_bf_disk_px(datacube)
    if auto is None:
        return None
    radius_px, _center_y_px, _center_x_px = auto
    try:
        datacube._fast_acbf_auto_alpha_px = radius_px
    except Exception:
        pass
    return radius_px


def _is_pixel_units(units: Any) -> bool:
    text = str(units or "").strip().lower()
    return text in {"", "pixels", "pixel", "px"}


def _axis_uncalibrated(calibration, size_getter: str, units_getter: str) -> bool:
    """A real/reciprocal axis is 'unset' at py4D pixel defaults (pixel units or size == 1)."""
    units = _cal_get(calibration, units_getter, None)
    if _is_pixel_units(units):
        return True
    size = _cal_get(calibration, size_getter, None)
    if size is None:
        return True
    try:
        return float(size) == 1.0
    except Exception:
        return False


def is_calibration_unset(datacube) -> bool:
    """True when the py4D calibration is still at pixel defaults for real or reciprocal space.

    Detection is conservative: an axis counts as uncalibrated when its pixel units are
    pixel-like (the py4DSTEM default) or its pixel size is exactly 1. Either the real-space
    (scan step) or the reciprocal-space (``dk``) axis being unset makes the calibration
    unusable for acBF, so we treat the whole calibration as unset if *either* is.
    """
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return True
    r_unset = _axis_uncalibrated(calibration, "get_R_pixel_size", "get_R_pixel_units")
    q_unset = _axis_uncalibrated(calibration, "get_Q_pixel_size", "get_Q_pixel_units")
    return r_unset or q_unset


def is_voltage_unset(datacube) -> bool:
    """True when py4D calibration has no usable accelerating voltage.

    acBF's phase-based aberration correction needs a real wavelength (unlike tcBF's pure
    shift-and-add), so voltage completeness is checked separately from
    :func:`is_calibration_unset`, which only covers the real/reciprocal-space axes.
    """
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return True
    try:
        value = calibration["voltage"]
    except Exception:
        return True
    try:
        return not (float(value) > 0)
    except Exception:
        return True


def sync_config_to_datacube_calibration(datacube, config) -> None:
    """Write fast-acbf calibration fields into py4D's datacube calibration."""
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return
    try:
        calibration.set_R_pixel_size(float(config.scan_step_angstrom))
        calibration.set_R_pixel_units("A")
    except Exception:
        pass
    try:
        calibration.set_Q_pixel_size(float(config.dk_inv_angstrom))
        calibration.set_Q_pixel_units("A^-1")
    except Exception:
        pass
    try:
        calibration["voltage"] = float(config.voltage_kv) * 1e3
    except Exception:
        pass
    try:
        calibration.set_QR_rotation(math.radians(float(config.rotation_deg)))
    except Exception:
        pass
    try:
        calibration.set_QR_flip(bool(config.transpose))
    except Exception:
        pass


def infer_qr_rotation(datacube, default: float) -> float:
    """Infer scan rotation from datacube calibration key QR_rotation.

    QR_rotation is stored in radians; the config field rotation_deg expects
    degrees, so a conversion is applied.
    """
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return default
    try:
        value = calibration.get_QR_rotation()
        if value is None:
            return default
        return math.degrees(float(value))
    except Exception:
        pass
    return default


def infer_qr_flip(datacube, default: bool) -> bool:
    """Infer diffraction pattern transpose from datacube calibration key QR_flip."""
    calibration = getattr(datacube, "calibration", None)
    if calibration is None:
        return default
    try:
        value = calibration.get_QR_flip()
        if value is None:
            return default
        return bool(value)
    except Exception:
        pass
    return default
