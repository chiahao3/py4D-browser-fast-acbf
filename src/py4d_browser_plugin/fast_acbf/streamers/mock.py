"""Mock streamer for live-acquisition profiling.

Yields ``(dataset, metadata)`` pairs from a static in-memory 4D array,
optionally jittering selected metadata keys per frame to exercise the
Tier-1 / Tier-2 / Tier-3 paths in BFSolver.apply_metadata.

The streamer is a plain iterable. The benchmark script (and any future
GUI driver) is responsible for wrapping it in a producer thread.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import numpy as np


class MockStreamer:
    def __init__(
        self,
        dataset: np.ndarray,
        base_metadata: dict,
        jitter: dict | None = None,
        n_frames: int | None = None,
        copy_dataset: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        dataset:
            4D STEM array shaped ``(Ry, Rx, Ky, Kx)``. Held by reference; the
            same buffer is yielded every frame unless ``copy_dataset`` is set,
            so the live-acquisition path is exercised in the in-place mutation
            mode that real hardware will use.
        base_metadata:
            Baseline metadata dict matching the ``BFSolver.apply_metadata``
            contract.
        jitter:
            Optional ``{key: sigma}`` dict. For each yielded frame, every
            listed key is set to ``base + N(0, sigma)``. Keys absent from
            this dict are emitted unchanged from ``base_metadata``.
        n_frames:
            Number of frames to yield. ``None`` (default) means infinite.
        copy_dataset:
            If True, ``np.copy`` the dataset on every yield. Default False.
        seed:
            RNG seed for reproducible jitter.
        """
        self.dataset = dataset
        self.base_metadata = deepcopy(base_metadata)
        self.jitter = dict(jitter) if jitter else {}
        self.n_frames = n_frames
        self.copy_dataset = bool(copy_dataset)
        self._rng = np.random.default_rng(seed)

    def _next_metadata(self) -> dict:
        meta = deepcopy(self.base_metadata)
        for key, sigma in self.jitter.items():
            base = float(meta.get(key, 0.0))
            meta[key] = base + float(self._rng.normal(0.0, float(sigma)))
        return meta

    def __iter__(self) -> Iterator[tuple[np.ndarray, dict]]:
        i = 0
        while self.n_frames is None or i < self.n_frames:
            data = np.copy(self.dataset) if self.copy_dataset else self.dataset
            yield data, self._next_metadata()
            i += 1

    def frames(self) -> Iterator[tuple[np.ndarray, dict]]:
        return iter(self)
