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
