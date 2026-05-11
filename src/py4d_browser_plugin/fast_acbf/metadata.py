"""Metadata adapter for the live-acquisition path.

Diffs incoming metadata dicts against the last known solver state and emits
only the changed keys, in the exact contract that ``BFSolver.apply_metadata``
consumes. Float values are rounded to the same precision used by
``FastAcbfConfig.solver_signature`` so that sub-precision metadata changes do
not trigger no-op cache invalidations.

Vendor-specific translation (proprietary metadata format -> dict in this
contract) is left to the caller; this module is intentionally vendor-agnostic.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


APPLY_METADATA_KEYS: tuple[str, ...] = (
    "wavelength",
    "max_alpha",
    "dk",
    "scan_shape",
    "scan_step_size",
    "rotation_deg",
    "flipud",
    "fliplr",
    "transpose",
)

_FLOAT_PRECISION: dict[str, int] = {
    "wavelength": 12,
    "max_alpha": 9,
    "dk": 12,
    "scan_step_size": 9,
    "rotation_deg": 9,
}

_BOOL_KEYS: frozenset[str] = frozenset({"flipud", "fliplr", "transpose"})


def normalize_metadata_value(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in _FLOAT_PRECISION:
        return round(float(value), _FLOAT_PRECISION[key])
    if key in _BOOL_KEYS:
        return bool(value)
    if key == "scan_shape":
        return tuple(int(x) for x in value)
    return value


class MetadataAdapter:
    """Track the last metadata state and emit per-frame deltas for apply_metadata."""

    def __init__(self) -> None:
        self._last_state: dict[str, Any] = {}

    @property
    def last_state(self) -> dict[str, Any]:
        return deepcopy(self._last_state)

    def reset(self) -> None:
        self._last_state = {}

    def diff(self, new_state: dict[str, Any]) -> dict[str, Any]:
        """Return the subset of apply_metadata keys whose values changed.

        Unknown keys in ``new_state`` are ignored. Missing keys are treated as
        "no information"; the cached state for those keys is preserved.
        """
        delta: dict[str, Any] = {}
        for key in APPLY_METADATA_KEYS:
            if key not in new_state:
                continue
            normalized = normalize_metadata_value(key, new_state[key])
            if self._last_state.get(key) != normalized:
                delta[key] = normalized
        for key, value in delta.items():
            self._last_state[key] = value
        return delta
