"""Non-Qt core of the live-acquisition path.

``LiveSolverEngine`` owns one ``BFSolver`` plus a ``MetadataAdapter`` for
its lifetime and exposes ``process_one(dataset, metadata)`` for headless
callers (benchmark scripts, integration tests). It must not import PyQt;
the Qt wrapper lives in :mod:`live.worker`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import FastAcbfConfig, LABEL_TO_STATE_KEY
from .metadata import MetadataAdapter
from ..utils import build_solver, choose_device, sync_config_from_solver, tensor_to_numpy

logger = logging.getLogger(__name__)

_HEAVY_METADATA_KEYS = frozenset({"wavelength", "max_alpha", "dk", "scan_shape", "scan_step_size"})
_ORIENTATION_METADATA_KEYS = frozenset({"rotation_deg", "flipud", "fliplr", "transpose"})


@dataclass
class FrameMetrics:
    latency_s: float
    fps: float
    device: str
    mask_path: str
    bf_pixels: int | None = None
    mode: str | None = None
    max_alpha_mrad: float | None = None
    display_noise_sigma_pct: float = 0.0
    stage_times: dict[str, float] | None = None


def _cuda_sync(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


class LiveSolverEngine:
    """Holds one BFSolver + MetadataAdapter and reconstructs frames in place."""

    def __init__(
        self,
        cfg: FastAcbfConfig,
        initial_dataset: np.ndarray,
        initial_metadata: dict[str, Any] | None = None,
        pinned_source_tensor=None,
        drift_per_frame: tuple[float, float] = (0.0, 0.0),
        display_noise_sigma_pct: float = 0.0,
        noise_seed: int | None = None,
    ) -> None:
        self.cfg = cfg.copy()
        self.runtime_device = choose_device(self.cfg.device)
        data = np.ascontiguousarray(np.asarray(initial_dataset, dtype=np.float32))
        self.solver = build_solver(self.cfg, data, self.runtime_device)
        self.adapter = MetadataAdapter()
        if initial_metadata is not None:
            self.adapter.diff(initial_metadata)
        self._last_frame_t: float | None = None
        # pinned_source_tensor stored as a lifetime anchor only — keeps the
        # CUDA-pinned buffer alive while output_buffer (its numpy view) is in use.
        self._pinned_source_tensor = pinned_source_tensor
        self._drift_y_per_frame = float(drift_per_frame[0])
        self._drift_x_per_frame = float(drift_per_frame[1])
        self._display_noise_sigma_pct = max(float(display_noise_sigma_pct), 0.0)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._frame_index = 0
        self._last_defocus_angstrom: float | None = None

    @property
    def device(self) -> str:
        return self.runtime_device

    def process_one(
        self,
        dataset: np.ndarray,
        metadata: dict[str, Any] | None = None,
        *,
        profile: bool = False,
    ) -> tuple[np.ndarray, FrameMetrics]:
        device = self.runtime_device
        metadata = metadata or {}
        t0 = time.perf_counter()
        data = np.ascontiguousarray(np.asarray(dataset, dtype=np.float32))
        self._apply_defocus_sweep(metadata)
        delta = self.adapter.diff(metadata)
        t_prep = time.perf_counter()

        if delta:
            orientation_delta = {k: v for k, v in delta.items() if k in _ORIENTATION_METADATA_KEYS}
            heavy_delta = {k: v for k, v in delta.items() if k in _HEAVY_METADATA_KEYS}

            if orientation_delta:
                ct = self.solver.coord_transform
                self.solver.set_flips(
                    bool(orientation_delta.get("flipud", ct.get("flipud", False))),
                    bool(orientation_delta.get("fliplr", ct.get("fliplr", False))),
                    bool(orientation_delta.get("transpose", ct.get("transpose", False))),
                )
                if "rotation_deg" in orientation_delta:
                    self.solver.set_rotation_deg(float(orientation_delta["rotation_deg"]))

            if heavy_delta:
                logger.warning(
                    "Live metadata change for physics params %s cannot be applied "
                    "without a full solver rebuild. Reconstruction will use previous "
                    "physics parameters.",
                    sorted(heavy_delta),
                )

        if profile:
            _cuda_sync(device)
        t_apply = time.perf_counter()
        recon_kwargs = self.cfg.reconstruct_kwargs()
        image = self.solver.get_reconstructed_image(
            mode=self.cfg.mode,
            frame=self.cfg.output_frame,
            **recon_kwargs,
        )
        if profile:
            _cuda_sync(device)
        t_recon = time.perf_counter()
        image_np = tensor_to_numpy(image)
        image_np = self._apply_display_noise(image_np)
        image_np = self._apply_scan_drift(image_np)
        t1 = time.perf_counter()
        latency = t1 - t0
        if self._last_frame_t is None:
            fps = (1.0 / latency) if latency > 0 else float("inf")
        else:
            dt = t1 - self._last_frame_t
            fps = (1.0 / dt) if dt > 0 else float("inf")
        self._last_frame_t = t1
        stage_times: dict[str, float] | None = None
        if profile:
            stage_times = {
                "prep": t_prep - t0,
                "apply_metadata": t_apply - t_prep,
                "get_reconstructed_image": t_recon - t_apply,
                "tensor_to_numpy": t1 - t_recon,
            }
        bf_pixels = int(self.solver.bf_mask.bool().sum().item())
        mask_path = "cuda" if str(device).startswith("cuda") else "host-mask"
        return image_np, FrameMetrics(
            latency_s=latency,
            fps=fps,
            device=device,
            mask_path=mask_path,
            bf_pixels=bf_pixels,
            mode=self.cfg.mode,
            max_alpha_mrad=float(self.cfg.max_alpha_mrad),
            display_noise_sigma_pct=self._display_noise_sigma_pct,
            stage_times=stage_times,
        )

    def _apply_defocus_sweep(self, metadata: dict[str, Any]) -> None:
        if "defocus_angstrom" not in metadata:
            return
        value = round(float(metadata["defocus_angstrom"]), 9)
        if self._last_defocus_angstrom == value:
            return
        state_key = LABEL_TO_STATE_KEY["C10"]
        if state_key in self.solver.ab_state.coeffs:
            self.solver.ab_state.set_physical(state_key, value)
            self.solver.clear_basis_cache()
        self._last_defocus_angstrom = value

    def _apply_scan_drift(self, image: np.ndarray) -> np.ndarray:
        dy = int(round(self._frame_index * self._drift_y_per_frame))
        dx = int(round(self._frame_index * self._drift_x_per_frame))
        self._frame_index += 1
        if dy == 0 and dx == 0:
            return image
        return np.roll(image, shift=(dy, dx), axis=(0, 1))

    def _apply_display_noise(self, image: np.ndarray) -> np.ndarray:
        if self._display_noise_sigma_pct <= 0:
            return image
        sigma = float(np.std(image)) * self._display_noise_sigma_pct / 100.0
        if sigma <= 0:
            return image
        noisy = image + self._noise_rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
        return noisy.astype(np.float32, copy=False)
