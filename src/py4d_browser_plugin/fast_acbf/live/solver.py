"""LiveBFSolver — BFSolver wrapper with per-frame dataset-swap capability."""

from __future__ import annotations

import numpy as np


class LiveBFSolver:
    """Thin wrapper around BFSolver that adds update_dataset().

    Wraps a fully-constructed BFSolver and exposes update_dataset(new_array),
    which replaces the raw data without rebuilding geometry, aberrations, or
    any basis caches.

    For CUDA devices the fast path avoids torch.cuda.empty_cache() entirely.
    A pre-allocated GPU staging buffer is filled via copy_() (DMA from pinned
    host memory when available), followed by GPU bool-mask extraction and FFT.
    The resulting FFT tensor is stored directly in imagefft._cache, bypassing
    imagefft.clear() which unconditionally calls empty_cache() and adds
    ~20 ms of CUDA allocator overhead per frame.

    NOTE: Accesses private internals of BFSolver / Dataset4D / PipelineManager:
      _dataset._array, _pipeline_manager.imagefft._cache,
      _pipeline_manager.imagefft._filled, _pipeline_manager.imagefft.storage,
      _pipeline_manager.imagefft.nb, _pipeline_manager.extractor.bf_iy/bf_ix,
      _pipeline_manager.device.
    A future fast-acbf release should expose Dataset4D.swap_array() and a
    lightweight cache-invalidation hook so this wrapper can delegate cleanly.
    See TODO.md.
    """

    def __init__(self, solver) -> None:
        self._solver = solver
        # Lazy-init GPU state (allocated once on first update_dataset call)
        self._staging_4d = None   # pre-allocated GPU staging buffer
        self._bf_iy_d = None      # BF row indices on device
        self._bf_ix_d = None      # BF column indices on device

    def update_dataset(self, new_array: np.ndarray) -> None:
        """Swap the underlying raw data per live frame without rebuilding geometry.

        Fast path (CUDA, storage != 'none'):
          1. ds._array = new_array          (pointer swap, no copy)
          2. staging.copy_(pinned, non_blocking=True)  (DMA H2D)
          3. gather BF pixels on GPU via precomputed bool-mask indices
          4. torch.fft.fft2 on GPU
          5. imagefft._cache = fft          (in-place cache replace)
          No cuda.empty_cache() at any point — freed tensors stay in
          PyTorch's caching allocator for O(1) reuse next frame.

        on_the_fly path (storage == 'none'):
          Just swaps ds._array and nulls the (already-None) cache refs.

        The BFExtractor is NOT rebuilt — it holds a reference to Dataset4D
        and picks up new_array via ds.raw_array() on the next extraction call.
        """
        import torch

        solver = self._solver
        ds = solver._dataset
        pm = solver._pipeline_manager
        imagefft = pm.imagefft

        new_array = np.ascontiguousarray(np.asarray(new_array, dtype=np.float32))
        expected = tuple(ds.shape)
        if new_array.shape != expected:
            raise ValueError(
                f"Shape mismatch: new_array {new_array.shape} != solver dataset {expected}."
            )

        # 1. Swap raw array; BFExtractor picks it up via ds.raw_array() by reference.
        ds._array = new_array

        if imagefft.storage == "none":
            # on_the_fly: just reset cache refs (they are already None after each use).
            imagefft._cache = None
            imagefft._filled = None
            return

        # 2. Lazy-init: allocate staging buffer and BF index tensors once per solver.
        dev = torch.device(pm.device)
        if self._staging_4d is None or tuple(self._staging_4d.shape) != expected:
            self._staging_4d = torch.empty(expected, dtype=torch.float32, device=dev)
            ext = pm.extractor
            self._bf_iy_d = torch.as_tensor(ext.bf_iy, dtype=torch.long, device=dev)
            self._bf_ix_d = torch.as_tensor(ext.bf_ix, dtype=torch.long, device=dev)

        # 3. H2D: copy pinned host buffer into pre-allocated staging tensor.
        #    torch.from_numpy shares the numpy array's pinned memory (no CPU copy),
        #    so copy_() triggers a DMA transfer from pinned host to device.
        self._staging_4d.copy_(torch.from_numpy(new_array), non_blocking=True)

        # 4. GPU bool-mask extraction (gather BF pixels) + FFT.
        vbf = self._staging_4d[:, :, self._bf_iy_d, self._bf_ix_d].permute(2, 0, 1).contiguous()
        fft = torch.fft.fft2(vbf, dim=(-2, -1))
        del vbf

        # 5. Write FFT into imagefft cache, bypassing imagefft.clear()'s empty_cache().
        #    Freed tensors (vbf, old _cache) stay in PyTorch's allocator pool for
        #    immediate reuse next frame — no CUDA memory manager round-trip.
        if imagefft.storage == "device":
            imagefft._cache = fft
        elif imagefft.storage == "host":
            imagefft._cache = fft.cpu().numpy()
            del fft
        imagefft._filled = np.ones(imagefft.nb, dtype=bool)

    def __getattr__(self, name: str):
        return getattr(self._solver, name)
