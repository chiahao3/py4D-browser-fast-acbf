"""Output selection helpers for real py4D-browser Live View."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import FastAcbfConfig, VALID_LIVE_OUTPUTS
from ..utils import tensor_to_numpy

LIVE_OUTPUT_NONE = "None"


@dataclass
class LiveViewOutputs:
    images: dict[str, np.ndarray]
    routes: dict[str, str]


def normalize_live_output(value: str) -> str:
    valid = {item.lower(): item for item in VALID_LIVE_OUTPUTS}
    key = str(value).strip().lower()
    if key not in valid:
        raise ValueError(f"Live output must be one of {', '.join(VALID_LIVE_OUTPUTS)}.")
    return valid[key]


def live_output_routes(config: FastAcbfConfig) -> dict[str, str]:
    return {
        "virtual": normalize_live_output(config.live_virtual_output),
        "result": normalize_live_output(config.live_result_output),
    }


def live_output_title(kind: str) -> str:
    return f"fast-acbf Live View {kind}"


def _reconstruction_kwargs(config: FastAcbfConfig, mode: str) -> dict[str, Any]:
    cfg = config.copy()
    cfg.mode = mode
    return cfg.reconstruct_kwargs()


def _as_float32(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def compute_one_live_output(solver: Any, config: FastAcbfConfig, kind: str) -> np.ndarray:
    kind = normalize_live_output(kind)
    if kind == LIVE_OUTPUT_NONE:
        raise ValueError("Cannot compute live output 'None'.")
    if kind in {"tcBF", "acBF"}:
        image = solver.get_reconstructed_image(
            mode=kind,
            frame=config.output_frame,
            **_reconstruction_kwargs(config, kind),
        )
        return _as_float32(tensor_to_numpy(image))
    if kind == "probe":
        probe = solver.get_probe(frame=config.output_frame, upscale=config.upscale).abs()
        return _as_float32(tensor_to_numpy(probe))
    if kind == "chi":
        chi = tensor_to_numpy(solver.get_chi_surface(frame=config.output_frame))
        return _as_float32(np.angle(np.exp(1j * chi)))
    raise AssertionError(f"Unhandled live output {kind!r}.")


def compute_live_view_outputs(solver: Any, config: FastAcbfConfig) -> LiveViewOutputs:
    routes = live_output_routes(config)
    requested = [kind for kind in routes.values() if kind != LIVE_OUTPUT_NONE]
    images: dict[str, np.ndarray] = {}
    for kind in dict.fromkeys(requested):
        images[kind] = compute_one_live_output(solver, config, kind)
    return LiveViewOutputs(images=images, routes=routes)
