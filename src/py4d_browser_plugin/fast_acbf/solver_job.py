"""Typed job objects for fast-acbf runner dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class SolverJob(Protocol):
    command: str

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None: ...


@dataclass
class PreviewJob:
    command: str = "manual"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        pass  # reconstruct-only; no refinement step


@dataclass
class RefineDefocusJob:
    command: str = "refine_defocus"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        emit("Refining defocus...")
        solver.refine_defocus(
            search_range=config.defocus_search_range(),
            num_points=int(config.defocus_points),
            metric=config.metric,
            plot_search=False,
            mode=config.refinement_mode,
            search_halfwidth=config.defocus_search_halfwidth_angstrom,
            defocus_range_tolerance_factor=float(config.defocus_range_tolerance_factor),
            **config.reconstruct_kwargs(),
        )


@dataclass
class RefineFlipsJob:
    command: str = "refine_flips"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        emit("Refining flips...")
        solver.refine_flips(
            metric=config.metric,
            plot_search=False,
            mode=config.refinement_mode,
            **config.reconstruct_kwargs(),
        )


@dataclass
class RefineScanRotationJob:
    command: str = "refine_scan_rotation"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        emit("Refining scan rotation...")
        solver.refine_scan_rotation(
            search_range=config.rotation_search_range(),
            num_points=int(config.fine_rotation_points),
            metric=config.metric,
            plot_search=False,
            mode=config.refinement_mode,
            search_halfwidth=(
                None
                if config.rotation_search_range() is not None
                else float(config.fine_rotation_halfwidth_deg)
            ),
            **config.reconstruct_kwargs(),
        )


@dataclass
class RefineAberrationsJob:
    command: str = "refine_aberrations"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        emit("Refining aberrations...")
        solver.refine_aberrations(
            lr=float(config.aberration_lr),
            iters=int(config.aberration_iters),
            metric=config.metric,
            mode=config.refinement_mode,
            **config.reconstruct_kwargs(),
        )


@dataclass
class LiteReconstructJob:
    """Composite job for the Lite taskbar: optional orientation search + aberration search.

    The runner reconstructs and emits the display ``mode`` after ``execute`` returns, so this
    job only performs the refinement steps. Refinement always runs in ``config.refinement_mode``
    (set to ``tcBF`` by the Lite handler) regardless of the final display mode.
    """

    auto_orientations: bool = True
    aberration_search: str = "first_order"
    pixel_mode: bool = False
    defocus_halfwidth_px: float = 20.0
    command: str = "lite_reconstruct"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        if self.auto_orientations:
            emit("Auto orientation search (placeholder)...")
            # TODO: plug in the new orientation refinement algorithm here. Intentional no-op for now.

        level = str(self.aberration_search).strip().lower()
        if level == "disabled":
            return

        if self.pixel_mode:
            self._refine_defocus_px(solver, config, emit)
            return

        if level == "df_only":
            emit("Refining defocus...")
            solver.refine_defocus(
                search_range=config.defocus_search_range(),
                num_points=int(config.defocus_points),
                metric=config.metric,
                plot_search=False,
                mode=config.refinement_mode,
                search_halfwidth=config.defocus_search_halfwidth_angstrom,
                defocus_range_tolerance_factor=float(config.defocus_range_tolerance_factor),
                **config.reconstruct_kwargs(),
            )
            return

        order = 2 if level == "second_order" else 1
        max_order = int(config.max_order)
        lr_scales = [1.0] * min(order, max_order) + [0.0] * max(0, max_order - order)
        emit(f"Refining aberrations up to order {order}...")
        solver.refine_aberrations(
            lr=float(config.aberration_lr),
            lr_scales=lr_scales,
            iters=int(config.aberration_iters),
            metric=config.metric,
            mode=config.refinement_mode,
            **config.reconstruct_kwargs(),
        )

    def _refine_defocus_px(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        """Calibration-free defocus search parametrised by a pixel-shift half-width.

        Derives an equivalent C10 (Angstrom) search range from the solver's own px-shift
        sensitivity (``get_yx_shifts_px`` at unit C10), then reuses ``refine_defocus``. This
        makes tcBF defocus focusing work even when the datacube calibration is unset.
        """
        import torch

        emit("Pixel-mode defocus search...")
        with torch.no_grad():
            c10_0 = float(solver.ab_state.get_physical("C_1_0"))
            solver.ab_state.set_physical("C_1_0", 1.0)
            unit_px = float(solver.get_yx_shifts_px(frame="scan").abs().max().item())
            solver.ab_state.set_physical("C_1_0", c10_0)
        if unit_px <= 0:
            raise ValueError("Cannot derive pixel-mode defocus range (zero shift sensitivity).")
        half_c10 = float(self.defocus_halfwidth_px) / unit_px
        solver.refine_defocus(
            search_range=(c10_0 - half_c10, c10_0 + half_c10),
            num_points=int(config.defocus_points),
            metric=config.metric,
            plot_search=False,
            mode=config.refinement_mode,
            **config.reconstruct_kwargs(),
        )


@dataclass
class AutoTuneJob:
    command: str = "auto_tune"

    def execute(self, solver: Any, config: Any, emit: Callable[[str], None]) -> None:
        emit("Refining all fast-acbf parameters...")
        solver.refine_all_params(
            metric=config.metric,
            mode=config.refinement_mode,
            defocus_range=config.defocus_search_range(),
            defocus_range_tolerance_factor=float(config.defocus_range_tolerance_factor),
            rotation_num_points=int(config.rotation_points),
            defocus_num_points=int(config.defocus_points),
            aberration_lr=float(config.aberration_lr),
            aberration_iters=int(config.aberration_iters),
            **config.reconstruct_kwargs(),
        )
