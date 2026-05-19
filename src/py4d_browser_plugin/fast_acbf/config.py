"""Configuration dataclass for the fast-acbf py4D-browser plugin."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .calibration import (
    electron_wavelength_angstrom,
    infer_alpha_mrad_from_detector,
    infer_dk_inv_angstrom,
    infer_scan_step_angstrom,
    infer_voltage_kv,
)


ABERRATION_LABELS_BY_ORDER = {
    1: ("C10", "C12a", "C12b"),
    2: ("C21a", "C21b", "C23a", "C23b"),
    3: ("C30", "C32a", "C32b", "C34a", "C34b"),
    4: ("C41a", "C41b", "C43a", "C43b", "C45a", "C45b"),
}

LABEL_TO_STATE_KEY = {
    "C10": "C_1_0",
    "C12a": "C_1_2_a",
    "C12b": "C_1_2_b",
    "C21a": "C_2_1_a",
    "C21b": "C_2_1_b",
    "C23a": "C_2_3_a",
    "C23b": "C_2_3_b",
    "C30": "C_3_0",
    "C32a": "C_3_2_a",
    "C32b": "C_3_2_b",
    "C34a": "C_3_4_a",
    "C34b": "C_3_4_b",
    "C41a": "C_4_1_a",
    "C41b": "C_4_1_b",
    "C43a": "C_4_3_a",
    "C43b": "C_4_3_b",
    "C45a": "C_4_5_a",
    "C45b": "C_4_5_b",
}


def labels_for_order(max_order: int) -> list[str]:
    labels: list[str] = []
    for order in range(1, max_order + 1):
        labels.extend(ABERRATION_LABELS_BY_ORDER.get(order, ()))
    return labels


def label_dict_to_fast_acbf(labels: dict[str, float], max_order: int) -> dict:
    """Convert UI coefficient labels to fast_acbf/Aberrations nested cartesian dict."""
    out: dict[tuple[int, int], float | dict[str, float]] = {}
    for order in range(1, max_order + 1):
        start_m = (order + 1) % 2
        for m in range(start_m, order + 2, 2):
            if m == 0:
                out[(order, m)] = float(labels.get(f"C{order}{m}", 0.0))
            else:
                out[(order, m)] = {
                    "a": float(labels.get(f"C{order}{m}a", 0.0)),
                    "b": float(labels.get(f"C{order}{m}b", 0.0)),
                }
    return out


@dataclass
class FastAcbfConfig:
    mode: str = "tcBF"
    acbf_algorithm: str = "phase_only"
    output_target: str = "virtual_image"
    output_frame: str = "scan"
    device: str = "auto"
    max_order: int = 2
    max_alpha_mrad: float = 25.0
    scan_step_angstrom: float = 1.0
    dk_inv_angstrom: float = 0.01
    voltage_kv: float = 300.0
    wavelength_angstrom: float = 0.019687
    use_calibration: bool = True
    use_detector_alpha: bool = True
    pipeline: str = "balanced"
    basis_mode: str = "on_the_fly"
    chunk_size: int = 64
    eps: float = 1e-3
    rolloff: float = 0.0
    regularization: float = 1e-3
    support_threshold: float = 1e-6
    rotation_deg: float = 0.0
    flipud: bool = False
    fliplr: bool = False
    transpose: bool = False
    metric: str = "normalized_std"
    defocus_points: int = 7
    rotation_points: int = 18
    aberration_lr: float = 1.0
    aberration_iters: int = 20
    refinement_mode: str = "tcBF"
    aberrations: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "FastAcbfConfig":
        return deepcopy(self)

    def resolved_for(self, parent) -> "FastAcbfConfig":
        cfg = self.copy()
        datacube = getattr(parent, "datacube", None)
        if datacube is not None and cfg.use_calibration:
            cfg.voltage_kv = infer_voltage_kv(datacube, cfg.voltage_kv)
            cfg.wavelength_angstrom = electron_wavelength_angstrom(cfg.voltage_kv)
            cfg.scan_step_angstrom = infer_scan_step_angstrom(datacube, cfg.scan_step_angstrom)
            cfg.dk_inv_angstrom = infer_dk_inv_angstrom(
                datacube, cfg.wavelength_angstrom, cfg.dk_inv_angstrom
            )
            if cfg.use_detector_alpha:
                cfg.max_alpha_mrad = infer_alpha_mrad_from_detector(
                    parent, cfg.wavelength_angstrom, cfg.max_alpha_mrad
                )
        return cfg

    def aberration_dict(self) -> dict:
        return label_dict_to_fast_acbf(self.aberrations, self.max_order)

    def coord_transform(self) -> dict[str, bool | float]:
        return {
            "flipud": bool(self.flipud),
            "fliplr": bool(self.fliplr),
            "transpose": bool(self.transpose),
            "rotation_deg": float(self.rotation_deg),
        }

    def solver_signature(self, datacube_data) -> tuple:
        return (
            tuple(getattr(datacube_data, "shape", ())),
            str(getattr(datacube_data, "dtype", "")),
            round(float(self.max_alpha_mrad), 9),
            round(float(self.scan_step_angstrom), 9),
            round(float(self.dk_inv_angstrom), 12),
            round(float(self.wavelength_angstrom), 12),
            int(self.max_order),
            str(self.device),
            str(self.pipeline),
            str(self.basis_mode),
            round(float(self.eps), 12),
        )

    def reconstruct_kwargs(self) -> dict[str, float | int | str]:
        kwargs: dict[str, float | int | str] = {
            "chunk_size": int(self.chunk_size),
        }
        if self.mode.lower() == "acbf" or self.refinement_mode.lower() == "acbf":
            kwargs.update(
                {
                    "rolloff": float(self.rolloff),
                    "acbf_algorithm": self.acbf_algorithm,
                    "regularization": float(self.regularization),
                    "support_threshold": float(self.support_threshold),
                }
            )
        return kwargs
