"""Configuration dataclass for the fast-acbf py4D-browser plugin."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .calibration import (
    electron_wavelength_angstrom,
    infer_alpha_px_from_detector,
    infer_dk_inv_angstrom,
    infer_scan_step_angstrom,
    infer_voltage_kv,
    max_alpha_mrad_from_px,
    resolve_max_alpha_px,
    resolved_wavelength_angstrom,
    is_calibration_unset,
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

VALID_FOCUS_SIGNS = ("none", "overfocus", "underfocus")
VALID_UPSCALE_METHODS = ("zero_insert", "nearest", "bilinear")
VALID_LIVE_OUTPUTS = ("None", "tcBF", "acBF", "probe", "chi")
UPSCALE_METHOD_DEFAULTS_BY_MODE = {
    "tcbf": "zero_insert",
    "acbf": "nearest",
}

# Lite taskbar -------------------------------------------------------------
VALID_LITE_OUTPUT_TARGETS = ("virtual_image", "result_image")
VALID_LITE_ABERRATION_SEARCH = ("disabled", "df_only", "first_order", "second_order")
LITE_ABERRATION_ORDER = {
    "disabled": 0,
    "df_only": 1,
    "first_order": 1,
    "second_order": 2,
}
# Minimum calibration-free defocus search half-width (scan px), regardless of how
# small the scan-fraction-derived term is for a small scan grid. See
# FastAcbfConfig.resolved_lite_defocus_halfwidth_px.
LITE_DEFOCUS_HALFWIDTH_PX_FLOOR = 20.0


def lite_search_order(value: str) -> int:
    """Aberration order implied by a Lite taskbar search level (0 when disabled)."""
    return LITE_ABERRATION_ORDER.get(str(value).strip().lower(), 1)


def default_upscale_method_for_mode(mode: str) -> str:
    return UPSCALE_METHOD_DEFAULTS_BY_MODE.get(str(mode).strip().lower(), "zero_insert")


def is_integer_upscale(value: float) -> bool:
    return abs(float(value) - round(float(value))) <= 1e-6


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
    max_alpha_mrad: float | None = None
    scan_step_angstrom: float = 1.0
    dk_inv_angstrom: float = 0.01
    voltage_kv: float | None = None
    wavelength_angstrom: float | None = None
    use_calibration: bool = True
    use_detector_alpha: bool = True
    calibration_free: bool = True
    max_alpha_px: float | None = None
    pipeline: str = "balanced"
    basis_mode: str = "on_the_fly"
    chunk_size: int = 64
    pad_width: int | None = None
    upscale: float = 1.0
    upscale_method: str = "zero_insert"
    eps: float = 1e-3
    rolloff: float = 0.0
    regularization: float = 1e-3
    support_threshold: float = 1e-6
    rotation_deg: float = 0.0
    flipud: bool = False
    fliplr: bool = False
    transpose: bool = False
    metric: str = "sobel"
    defocus_points: int = 5
    defocus_range_min_angstrom: float | None = None
    defocus_range_max_angstrom: float | None = None
    defocus_search_halfwidth_angstrom: float | None = None
    defocus_range_tolerance_factor: float = 192.0
    focus_sign: str = "none"
    rotation_points: int = 12
    rotation_range_min_deg: float | None = None
    rotation_range_max_deg: float | None = None
    fine_rotation_halfwidth_deg: float | None = None
    fine_rotation_points: int = 11
    fine_rotation_xatol_deg: float = 0.1
    aberration_lr: float = 1.0
    aberration_iters: int = 20
    refinement_mode: str = "tcBF"
    live_virtual_output: str = "None"
    live_result_output: str = "tcBF"
    live_auto_focus_interval_s: float = 5.0
    live_auto_aberrations_interval_s: float = 30.0
    lite_output_target: str = "virtual_image"
    lite_aberration_search: str = "first_order"
    lite_defocus_halfwidth_px: float | None = None
    lite_defocus_halfwidth_scan_fraction: float = 0.2
    lite_seeded_defocus_fraction: float = 0.5
    aberrations: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "FastAcbfConfig":
        return deepcopy(self)

    def resolved_for(self, parent) -> "FastAcbfConfig":
        cfg = self.copy()
        datacube = getattr(parent, "datacube", None)
        if datacube is not None and cfg.use_calibration:
            cfg.voltage_kv = infer_voltage_kv(datacube, cfg.voltage_kv)
            if cfg.voltage_kv is not None:
                cfg.wavelength_angstrom = electron_wavelength_angstrom(cfg.voltage_kv)
            cfg.scan_step_angstrom = infer_scan_step_angstrom(datacube, cfg.scan_step_angstrom)
            # dk/max_alpha unit conversions below need *some* wavelength number even when
            # voltage is genuinely unknown; fall back to a placeholder for those internal
            # conversions only -- cfg.voltage_kv/wavelength_angstrom themselves stay None so
            # the UI shows them as unset and the acBF run gate still blocks on missing voltage.
            wavelength_for_conversion = resolved_wavelength_angstrom(cfg.wavelength_angstrom)
            cfg.dk_inv_angstrom = infer_dk_inv_angstrom(
                datacube, wavelength_for_conversion, cfg.dk_inv_angstrom
            )
            if cfg.calibration_free and is_calibration_unset(datacube):
                # infer_alpha_mrad_from_detector needs a real dk to convert pixels to
                # mrad; under unset calibration dk is a meaningless placeholder, so
                # measure the BF disk radius directly in pixels instead and convert
                # using whatever dk/wavelength placeholders are in effect. The bf_mask
                # this produces is exact regardless of how "real" those placeholders
                # are, since only the product max_alpha_px = max_alpha/(1000*dk*wavelength)
                # is used to build it.
                print(f"[fast-acbf] Auto-detecting BF disk radius (calibration-free case)...")
                alpha_px = resolve_max_alpha_px(parent, datacube)
                if alpha_px is not None:
                    val = alpha_px * cfg.dk_inv_angstrom * cfg.wavelength_angstrom * 1000.0
                    print(f"[fast-acbf] Detected BF disk radius: {alpha_px:.2f} px -> max_alpha_mrad: {val:.4f}")
                    cfg.max_alpha_px = alpha_px
                    cfg.max_alpha_mrad = val
                else:
                    print(f"[fast-acbf] BF disk auto-detection failed; using default max_alpha_mrad: {cfg.max_alpha_mrad:.4f}")
            else:
                if cfg.max_alpha_px is not None:
                    # A previous calibration-free run left max_alpha_mrad derived from
                    # placeholder dk/wavelength (max_alpha_px is the marker for that).
                    # The BF-disk radius in pixels is a property of the raw data, not
                    # the calibration, so re-express it in the now-real mrad instead of
                    # leaving max_alpha_mrad pinned to that stale value now that real
                    # calibration is available (or calibration_free was turned off).
                    cfg.max_alpha_mrad = (
                        cfg.max_alpha_px * cfg.dk_inv_angstrom * cfg.wavelength_angstrom * 1000.0
                    )
                    cfg.max_alpha_px = None
                if cfg.use_detector_alpha:
                    # Prefer a manual circular detector; if none exists, try to auto-detect
                    # the BF disk radius from the data.
                    print(f"[fast-acbf] Auto-detecting BF disk radius (calibrated case)...")
                    alpha_px = resolve_max_alpha_px(parent, datacube)
                    if alpha_px is not None:
                        val = alpha_px * cfg.dk_inv_angstrom * cfg.wavelength_angstrom * 1000.0
                        print(f"[fast-acbf] Detected BF disk radius: {alpha_px:.2f} px -> max_alpha_mrad: {val:.4f}")
                        cfg.max_alpha_mrad = val
                    else:
                        print(f"[fast-acbf] BF disk auto-detection failed; using default max_alpha_mrad: {cfg.max_alpha_mrad:.4f}")
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
            self.normalized_pad_width(),
            round(float(self.upscale), 12),
            str(self.upscale_method),
        )

    def normalized_pad_width(self) -> int | None:
        if self.pad_width is None:
            return None
        value = int(self.pad_width)
        return value if value > 0 else None

    def output_pixel_size_angstrom(self) -> float:
        return float(self.scan_step_angstrom) / float(self.upscale)

    def uses_acbf_reconstruction(self) -> bool:
        live_outputs = {
            str(self.live_virtual_output).strip().lower(),
            str(self.live_result_output).strip().lower(),
        }
        return (
            self.mode.lower() == "acbf"
            or self.refinement_mode.lower() == "acbf"
            or "acbf" in live_outputs
        )

    def validate_live_output_settings(self) -> None:
        valid = {value.lower(): value for value in VALID_LIVE_OUTPUTS}
        for label, value in (
            ("Live virtual image output", self.live_virtual_output),
            ("Live result output", self.live_result_output),
        ):
            if str(value).strip().lower() not in valid:
                raise ValueError(
                    f"{label} must be one of {', '.join(VALID_LIVE_OUTPUTS)}."
                )

    def validate_live_auto_refinement_settings(self) -> None:
        for label, value in (
            ("Live Auto Focus interval", self.live_auto_focus_interval_s),
            ("Live Auto Aberrations interval", self.live_auto_aberrations_interval_s),
        ):
            if float(value) <= 0:
                raise ValueError(f"{label} must be positive.")

    def validate_lite_settings(self) -> None:
        if str(self.lite_output_target).strip().lower() not in VALID_LITE_OUTPUT_TARGETS:
            raise ValueError(
                f"Lite output panel must be one of {', '.join(VALID_LITE_OUTPUT_TARGETS)}."
            )
        if str(self.lite_aberration_search).strip().lower() not in VALID_LITE_ABERRATION_SEARCH:
            raise ValueError(
                "Lite aberration search must be one of "
                f"{', '.join(VALID_LITE_ABERRATION_SEARCH)}."
            )
        if self.lite_defocus_halfwidth_px is not None and float(self.lite_defocus_halfwidth_px) <= 0:
            raise ValueError("Lite defocus half width (px) must be positive.")
        if float(self.lite_defocus_halfwidth_scan_fraction) <= 0:
            raise ValueError("Lite defocus scan fraction must be positive.")
        if float(self.lite_seeded_defocus_fraction) <= 0:
            raise ValueError("Lite seeded defocus fraction must be positive.")

    def normalized_live_output(self, value: str) -> str:
        valid = {item.lower(): item for item in VALID_LIVE_OUTPUTS}
        return valid[str(value).strip().lower()]

    def validate_upscale_settings(self) -> None:
        self.validate_live_output_settings()
        self.validate_live_auto_refinement_settings()
        method = str(self.upscale_method).strip().lower()
        if method not in VALID_UPSCALE_METHODS:
            raise ValueError(
                f"Upscale method must be one of {', '.join(VALID_UPSCALE_METHODS)}."
            )
        if float(self.upscale) < 1.0:
            raise ValueError("Upscale must be >= 1.0.")
        if method == "zero_insert" and not is_integer_upscale(float(self.upscale)):
            raise ValueError(
                "Upscale method zero_insert requires an integer upscale factor. "
                "Use nearest or bilinear for fractional upscale."
            )
        if (
            method == "zero_insert"
            and float(self.upscale) != 1.0
            and self.uses_acbf_reconstruction()
        ):
            raise ValueError(
                "Upscale method zero_insert is only supported for tcBF. "
                "Use nearest or bilinear when Display mode or Refinement mode is acBF."
            )

    def coerce_upscale_method_for_mode(self) -> list[str]:
        """Coerce impossible method/mode combinations and return user-facing notes."""
        messages: list[str] = []
        method = str(self.upscale_method).strip().lower()
        if method not in VALID_UPSCALE_METHODS:
            self.upscale_method = default_upscale_method_for_mode(self.mode)
            messages.append(
                f"Unknown upscale method {method!r}; using {self.upscale_method}."
            )
            return messages
        self.upscale_method = method
        if (
            method == "zero_insert"
            and float(self.upscale) != 1.0
            and self.uses_acbf_reconstruction()
        ):
            self.upscale_method = "nearest"
            messages.append(
                "zero_insert is not supported for acBF; upscale method was changed to nearest."
            )
        return messages

    def defocus_search_range(self) -> tuple[float, float] | None:
        if self.defocus_range_min_angstrom is None and self.defocus_range_max_angstrom is None:
            return None
        if self.defocus_range_min_angstrom is None or self.defocus_range_max_angstrom is None:
            raise ValueError("Defocus search range requires both min and max.")
        return (float(self.defocus_range_min_angstrom), float(self.defocus_range_max_angstrom))

    def rotation_search_range(self) -> tuple[float, float] | None:
        if self.rotation_range_min_deg is None and self.rotation_range_max_deg is None:
            return None
        if self.rotation_range_min_deg is None or self.rotation_range_max_deg is None:
            raise ValueError("Rotation search range requires both min and max.")
        return (float(self.rotation_range_min_deg), float(self.rotation_range_max_deg))

    def resolved_fine_rotation_halfwidth_deg(self) -> float:
        """Fine-rotation Brent search half-width in degrees.

        The coarse orientation grid (``rotation_points`` samples spread evenly
        over the full 360 deg) can land its winning sample up to half its own
        spacing away from the true optimum, so a half-width narrower than that
        risks Brent's search window missing the true optimum entirely. When
        left unset, this derives a safe default from that spacing
        (180/rotation_points). An explicit value is used exactly as given,
        with no floor applied -- if you've set one, you're assumed to know
        why it's narrower than the derived default.
        """
        if self.fine_rotation_halfwidth_deg is None:
            return 180.0 / max(1, int(self.rotation_points))
        return float(self.fine_rotation_halfwidth_deg)

    def resolved_lite_defocus_halfwidth_px(
        self, min_scan_dim: int, *, seeded: bool = False
    ) -> float:
        """Calibration-free defocus search half-width in raw scan pixels.

        Pixel shift is a relative quantity -- it depends on scan_step_angstrom,
        itself a placeholder under calibration-free operation -- so there's no
        physically derivable bound on how far the true, uncalibrated defocus
        could be from the search's starting guess. This instead scales with how
        much scan field of view is actually available: past roughly half the
        scan FOV, "shift these sub-images and sum" stops corresponding to any
        physically meaningful overlap, regardless of how the search got there.
        Floored at ``LITE_DEFOCUS_HALFWIDTH_PX_FLOOR`` so small scans still get
        a reasonably wide search rather than shrinking below it.

        If left unset, derives ``max(LITE_DEFOCUS_HALFWIDTH_PX_FLOOR,
        lite_defocus_halfwidth_scan_fraction * min_scan_dim)``. An explicit
        value is used exactly as given, with no floor or scaling applied.

        ``seeded`` scales that auto-derived value down by
        ``lite_seeded_defocus_fraction`` -- for use once a defocus estimate is
        already in hand (e.g. after Orientation Optimization has set a non-zero
        C10), where searching the full "assume nothing" width just risks
        wandering away from an already-good value. Has no effect on an explicit
        ``lite_defocus_halfwidth_px``, which is always used exactly as given.
        """
        if self.lite_defocus_halfwidth_px is not None:
            return float(self.lite_defocus_halfwidth_px)
        base = max(
            LITE_DEFOCUS_HALFWIDTH_PX_FLOOR,
            float(self.lite_defocus_halfwidth_scan_fraction) * float(min_scan_dim),
        )
        if seeded:
            return base * float(self.lite_seeded_defocus_fraction)
        return base

    def reconstruct_kwargs(self) -> dict[str, float | int | str | None]:
        self.validate_upscale_settings()
        kwargs: dict[str, float | int | str | None] = {
            "chunk_size": int(self.chunk_size),
            "pad_width": self.normalized_pad_width(),
            "upscale": float(self.upscale),
            "upscale_method": str(self.upscale_method),
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
