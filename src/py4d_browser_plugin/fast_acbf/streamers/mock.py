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
        poisson_chunk_rows: int | None = None,
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
        poisson_chunk_rows:
            Optional number of scan rows to sample at once. Defaults to a
            conservative chunk size to avoid full-frame temporary arrays.
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
        self.poisson_chunk_rows = poisson_chunk_rows
        self._rng = np.random.default_rng(seed)
        if self.output_buffer is not None:
            if self.output_buffer.shape != self.dataset.shape:
                raise ValueError(
                    f"output_buffer shape {self.output_buffer.shape} does not match "
                    f"dataset shape {self.dataset.shape}."
                )
            np.copyto(self.output_buffer, np.asarray(self.dataset, dtype=np.float32))

    def _default_poisson_chunk_rows(self) -> int:
        if self.dataset.ndim < 4:
            return 1
        row_bytes = int(np.prod(self.dataset.shape[1:], dtype=np.int64)) * 4
        # Keep float lam + int64 sample temporaries in the tens of MB range.
        target_bytes = 32 * 1024 * 1024
        return max(1, target_bytes // max(row_bytes * 3, 1))

    def _poisson_frame(self) -> np.ndarray:
        scale = float(self.poisson_scale)
        out = self.output_buffer
        if out is None:
            out = np.empty_like(self.dataset, dtype=np.float32)
        rows = int(self.poisson_chunk_rows or self._default_poisson_chunk_rows())
        rows = max(1, rows)
        for y0 in range(0, self.dataset.shape[0], rows):
            y1 = min(y0 + rows, self.dataset.shape[0])
            src = self.dataset[y0:y1]
            lam = np.clip(src, 0.0, None) * scale
            noisy = self._rng.poisson(lam).astype(np.float32, copy=False)
            noisy /= scale
            np.copyto(out[y0:y1], noisy)
        return out

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
                data = self._poisson_frame()
            elif self.output_buffer is not None:
                data = self.output_buffer
            else:
                data = np.copy(self.dataset) if self.copy_dataset else self.dataset
            yield data, self._next_metadata()
            i += 1

    def frames(self) -> Iterator[tuple[np.ndarray, dict]]:
        return iter(self)
