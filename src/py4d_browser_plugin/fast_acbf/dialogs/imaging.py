"""Display helpers for 2D images: intensity scaling, autoscale levels and complex colour.

Ported from ptydy (``ptydy.core.imaging``) for the advanced dashboard's views; keep
the two in step when either changes.
"""

from __future__ import annotations

import enum

import numpy as np

LOG_FLOOR = 1e-3
"""Smallest value taken into log scaling (log of zero or negative counts is undefined)."""


class Scaling(enum.Enum):
    LINEAR = "Linear"
    LOG = "Log"
    SQRT = "Square root"


def scale(image: np.ndarray, mode: Scaling) -> np.ndarray:
    """Intensity scaling applied before display (never to the data itself)."""
    image = np.asarray(image, dtype=np.float64)
    match mode:
        case Scaling.LINEAR:
            return image
        case Scaling.LOG:
            return np.log10(np.maximum(image, LOG_FLOOR))
        case Scaling.SQRT:
            return np.sqrt(np.maximum(image, 0.0))
    raise ValueError(f"Unknown scaling {mode!r}")


def percentile_levels(image: np.ndarray, low: float = 0.1,
                      high: float = 99.9) -> tuple[float, float]:
    """Display levels at the given percentiles of the finite values (never equal)."""
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(finite, [low, high]))
    if hi <= lo:
        hi = lo + (abs(lo) * 1e-6 or 1.0)
    return lo, hi








def complex_to_rgb(z: np.ndarray, low: float = 1.0, high: float = 99.0,
                   levels: tuple[float, float] | None = None) -> np.ndarray:
    """Colour image of a complex array: hue = phase, brightness = magnitude.

    Magnitudes are scaled between their ``low`` and ``high`` percentiles, or between
    ``levels`` (e.g. one range for a whole stack). Returns ``(..., 3)`` float RGB in
    ``[0, 1]``.
    """
    z = np.asarray(z)
    mag = np.abs(z)
    lo, hi = levels if levels is not None else percentile_levels(mag, low, high)
    hi = hi if hi > lo else lo + 1.0
    v = np.clip((mag - lo) / (hi - lo), 0.0, 1.0)
    h = (np.angle(z) + np.pi) / (2 * np.pi)  # 0..1
    return _hsv_to_rgb(h, np.ones_like(v), v)


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    i = np.floor(h * 6.0).astype(int) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    choices = [
        (v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q),
    ]
    rgb = np.zeros(h.shape + (3,))
    for k, (r, g, b) in enumerate(choices):
        sel = i == k
        rgb[sel, 0], rgb[sel, 1], rgb[sel, 2] = r[sel], g[sel], b[sel]
    return rgb


def image_stats(image: np.ndarray) -> dict[str, float]:
    """Min, max, mean, sum and standard deviation of the finite values."""
    a = np.asarray(image)
    a = np.abs(a) if np.iscomplexobj(a) else a
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return {k: float("nan") for k in ("min", "max", "mean", "sum", "std")}
    return {"min": float(finite.min()), "max": float(finite.max()),
            "mean": float(finite.mean()), "sum": float(finite.sum()),
            "std": float(finite.std())}


def fill_mask(image: np.ndarray, rim: int = 2, min_fraction: float = 0.002) -> np.ndarray | None:
    """Pixels that hold data, or ``None`` when the image has no fill region.

    Images rotated or padded after reconstruction (e.g. fast-acbf's detector-frame
    output) carry a constant fill (exact zeros) connected to the image border, and a rim
    of values interpolated between fill and data. Both would dominate percentile display
    levels. Returns ``False`` on that border-connected zero region grown by ``rim``
    pixels, ``True`` elsewhere. Zeros inside the image (e.g. counts of a counting
    detector) are kept; a fill smaller than ``min_fraction`` of the image is ignored.
    """
    from scipy import ndimage

    a = np.asarray(image)
    if a.ndim != 2 or a.size == 0:
        return None
    zero = a == 0
    if zero.sum() < min_fraction * a.size:
        return None
    labels, n = ndimage.label(zero)
    if n == 0:
        return None
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    fill = np.isin(labels, border[border > 0])
    if fill.sum() < min_fraction * a.size:
        return None
    if rim > 0:
        fill = ndimage.binary_dilation(fill, iterations=rim)
    valid = ~fill
    return valid if valid.any() else None


def stack_levels(stack: np.ndarray, low: float = 0.5, high: float = 99.5,
                 scaling: Scaling = Scaling.LINEAR,
                 max_zero_fraction: float = 0.1) -> tuple[float, float]:
    """One display range for every slice of a stack (in ``scaling`` units): the
    ``low`` / ``high`` percentiles of the data pixels of all slices together
    (:func:`fill_mask`, found on each unscaled slice), so no slice is clipped more than
    the percentiles say.

    Slices whose data pixels are more than ``max_zero_fraction`` exact zeros are left
    out (e.g. a tcBF slice at C10 ≈ 0 with zero-insert upscaling, where no shifts fill
    the inserted pixels), unless every slice is like that.
    """
    kept, degenerate = [], []
    for s in np.asarray(stack):
        valid = fill_mask(s)
        values = scale(s if valid is None else s[valid], scaling).ravel()
        raw = s if valid is None else s[valid]
        (degenerate if np.mean(raw == 0) > max_zero_fraction else kept).append(values)
    pooled = np.concatenate(kept or degenerate)
    return percentile_levels(pooled, low, high)
