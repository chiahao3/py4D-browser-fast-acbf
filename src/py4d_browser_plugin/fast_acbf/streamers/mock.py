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
        output_buffer: np.ndarray | None = None,
        poisson_scale: float | None = None,
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
        output_buffer:
            Optional writable float32 buffer to receive generated frames. This
            is useful for live demos that want a CUDA-pinned host array filled
            directly by the simulated acquisition source.
        poisson_scale:
            Optional count scale for Poisson noise. When positive, each frame
            is sampled as ``poisson(max(dataset, 0) * scale) / scale``.
        seed:
            RNG seed for reproducible jitter.
        """
        self.dataset = dataset
        self.base_metadata = deepcopy(base_metadata)
        self.jitter = dict(jitter) if jitter else {}
        self.n_frames = n_frames
        self.copy_dataset = bool(copy_dataset)
        self.output_buffer = output_buffer
        self.poisson_scale = float(poisson_scale) if poisson_scale else None
        self._rng = np.random.default_rng(seed)
        if self.output_buffer is not None:
            if self.output_buffer.shape != self.dataset.shape:
                raise ValueError(
                    f"output_buffer shape {self.output_buffer.shape} does not match "
                    f"dataset shape {self.dataset.shape}."
                )
            np.copyto(self.output_buffer, np.asarray(self.dataset, dtype=np.float32))

    def _next_metadata(self) -> dict:
        meta = deepcopy(self.base_metadata)
        for key, sigma in self.jitter.items():
            base = float(meta.get(key, 0.0))
            meta[key] = base + float(self._rng.normal(0.0, float(sigma)))
        return meta

    def __iter__(self) -> Iterator[tuple[np.ndarray, dict]]:
        i = 0
        while self.n_frames is None or i < self.n_frames:
            if self.poisson_scale is not None and self.poisson_scale > 0:
                lam = np.clip(self.dataset, 0.0, None) * self.poisson_scale
                noisy = self._rng.poisson(lam).astype(np.float32, copy=False)
                noisy /= self.poisson_scale
                if self.output_buffer is not None:
                    np.copyto(self.output_buffer, noisy)
                    data = self.output_buffer
                else:
                    data = noisy
            elif self.output_buffer is not None:
                data = self.output_buffer
            else:
                data = np.copy(self.dataset) if self.copy_dataset else self.dataset
            yield data, self._next_metadata()
            i += 1

    def frames(self) -> Iterator[tuple[np.ndarray, dict]]:
        return iter(self)
