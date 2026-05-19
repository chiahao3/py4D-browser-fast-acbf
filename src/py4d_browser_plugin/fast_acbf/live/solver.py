"""LiveBFSolver — BFSolver wrapper with per-frame dataset-swap capability."""

from __future__ import annotations

import numpy as np


class LiveBFSolver:
    """Thin wrapper around BFSolver that adds update_dataset().

    Wraps a fully-constructed BFSolver and exposes update_dataset(new_array),
    which replaces the raw data without rebuilding geometry, aberrations, or
    any basis caches.

    NOTE: Accesses private internals of BFSolver / Dataset4D / PipelineManager:
      _dataset._array, _pipeline_manager.extractor, _pipeline_manager.imagefft,
      _pipeline_manager.device, _pipeline_manager.detector_geom,
      _pipeline_manager.resolution.extractor_strategy.
    A future fast-acbf release should expose Dataset4D.swap_array() and
    BFSolver.update_dataset() so this wrapper can delegate cleanly.
    See TODO.md.
    """

    def __init__(self, solver) -> None:
        self._solver = solver

    def update_dataset(self, new_array: np.ndarray) -> None:
        """Swap the underlying raw data without rebuilding geometry or aberration state.

        Safe to call every live frame.  Cost per call: one BFExtractor
        construction + imagefft.clear() (+ optional precompute).
        All geometry/aberration/basis caches in BFReconstructor are preserved
        because they depend only on detector geometry and aberration coefficients,
        not on raw pixel values.

        If new_array is already a float32 C-contiguous array (e.g. a CUDA-pinned
        host buffer), np.ascontiguousarray is a no-op and its pinned-memory
        characteristics are preserved through the extraction pipeline.
        """
        solver = self._solver
        ds = solver._dataset
        pm = solver._pipeline_manager

        new_array = np.ascontiguousarray(np.asarray(new_array, dtype=np.float32))
        expected = tuple(ds.shape)
        if new_array.shape != expected:
            raise ValueError(
                f"Shape mismatch: new_array {new_array.shape} != solver dataset {expected}."
            )

        # 1. Swap the raw array inside Dataset4D
        ds._array = new_array

        # 2. Rebuild BFExtractor against the updated dataset
        from fast_acbf.data.bf_extractor import BFExtractor
        new_extractor = BFExtractor(
            ds,
            pm.detector_geom,
            device=pm.device,
            strategy=pm.resolution.extractor_strategy,  # use already-resolved strategy
        )

        # 3. Wire new extractor into the pipeline
        #    pm.imagefft and solver._recon.imagefft are the SAME object
        pm.extractor = new_extractor
        pm.imagefft.extractor = new_extractor

        # 4. Invalidate FFT cache; re-precompute immediately if required
        pm.imagefft.clear()
        if pm.imagefft.fill == "precompute":
            pm.imagefft.precompute()
        # _tcbf_cache / _acbf_cache in BFReconstructor are geometry-only — preserved

    def __getattr__(self, name: str):
        return getattr(self._solver, name)
