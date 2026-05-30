"""Temporary HDF5 reader used by the Dataset Streamer test harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DatasetStreamPaths:
    data: str = "/dp"
    scan_step_angstrom: str = "/calibration/dx_ang"
    dk_inv_angstrom: str = "/calibration/dk_y_inv_ang"
    voltage_kv: str = "/calibration/kv"


DEFAULT_STREAM_PATHS = DatasetStreamPaths()


@dataclass
class DatasetStreamFrame:
    path: Path
    datacube: object
    title: str


class StreamReadError(RuntimeError):
    """Raised when a test-stream HDF5 file cannot produce a calibrated datacube."""


def discover_hdf5_stream_files(folder: str | Path) -> list[Path]:
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise StreamReadError(f"Dataset stream folder does not exist: {root}")
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )


def _read_scalar(h5, path: str, label: str) -> float:
    normalized = _normalize_hdf5_path(path)
    try:
        value = h5[normalized][()]
    except Exception:
        value = _read_attr_scalar(h5, normalized)
        if value is None:
            raise StreamReadError(f"Missing {label} at HDF5 path {path!r}.")
    array = np.asarray(value)
    if array.size != 1:
        raise StreamReadError(f"{label} at HDF5 path {path!r} must be scalar.")
    return float(array.reshape(-1)[0])


def _normalize_hdf5_path(path: str) -> str:
    value = str(path).strip()
    if not value:
        raise StreamReadError("HDF5 path cannot be empty.")
    return value if value.startswith("/") else f"/{value}"


def _read_attr_scalar(h5, path: str):
    parent_path, _, attr_name = path.rstrip("/").rpartition("/")
    if not parent_path:
        parent_path = "/"
    try:
        group = h5[parent_path]
    except Exception:
        return None
    attrs = getattr(group, "attrs", None)
    if attrs is None or attr_name not in attrs:
        return None
    return attrs[attr_name]


def read_hdf5_stream_frame(path: str | Path, paths: DatasetStreamPaths) -> DatasetStreamFrame:
    import h5py
    from py4DSTEM import DataCube

    file_path = Path(path)
    try:
        with h5py.File(file_path, "r") as h5:
            try:
                data = np.asarray(h5[_normalize_hdf5_path(paths.data)][()], dtype=np.float32)
            except Exception as exc:
                raise StreamReadError(f"Missing 4D data at HDF5 path {paths.data!r}.") from exc
            if data.ndim == 5:
                data = data[-1]
            if data.ndim != 4:
                raise StreamReadError(
                    f"Data at HDF5 path {paths.data!r} must be 4D, got shape {data.shape}."
                )
            scan_step = _read_scalar(h5, paths.scan_step_angstrom, "scan step")
            dk = _read_scalar(h5, paths.dk_inv_angstrom, "dk")
            voltage = _read_scalar(h5, paths.voltage_kv, "voltage")
    except StreamReadError:
        raise
    except Exception as exc:
        raise StreamReadError(f"Could not read dataset stream file {file_path}: {exc}") from exc

    datacube = DataCube(data)
    datacube.calibration.set_R_pixel_size(scan_step)
    datacube.calibration.set_R_pixel_units("A")
    datacube.calibration.set_Q_pixel_size(dk)
    datacube.calibration.set_Q_pixel_units("A^-1")
    datacube.calibration["voltage"] = voltage
    return DatasetStreamFrame(path=file_path, datacube=datacube, title=str(file_path))


class DatasetStreamSequence:
    """Looping sequence of calibrated HDF5 datacubes for the temporary streamer."""

    def __init__(
        self,
        files: list[Path],
        paths: DatasetStreamPaths = DEFAULT_STREAM_PATHS,
        *,
        preload: bool = False,
        reader: Callable[[Path, DatasetStreamPaths], DatasetStreamFrame] = read_hdf5_stream_frame,
    ) -> None:
        if not files:
            raise StreamReadError("No .h5 or .hdf5 files found for dataset streaming.")
        self.files = list(files)
        self.paths = paths
        self.preload = bool(preload)
        self._reader = reader
        self._index = 0
        self._frames: list[DatasetStreamFrame] | None = None
        self.preload_errors: list[str] = []
        self.last_position: int | None = None
        if self.preload:
            frames: list[DatasetStreamFrame] = []
            for file_path in self.files:
                try:
                    frames.append(self._reader(file_path, self.paths))
                except Exception as exc:
                    self.preload_errors.append(f"{file_path.name}: {exc}")
            if not frames:
                detail = "\n".join(self.preload_errors)
                raise StreamReadError(f"All dataset stream files failed to preload.\n{detail}")
            self._frames = frames

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        paths: DatasetStreamPaths = DEFAULT_STREAM_PATHS,
        *,
        preload: bool = False,
        reader: Callable[[Path, DatasetStreamPaths], DatasetStreamFrame] = read_hdf5_stream_frame,
    ) -> "DatasetStreamSequence":
        return cls(
            discover_hdf5_stream_files(folder),
            paths,
            preload=preload,
            reader=reader,
        )

    def next_frame(self) -> tuple[DatasetStreamFrame, list[str]]:
        skipped: list[str] = []
        if self._frames is not None:
            self.last_position = self._index % len(self._frames)
            frame = self._frames[self.last_position]
            self._index += 1
            return frame, skipped

        for _ in range(len(self.files)):
            self.last_position = self._index % len(self.files)
            file_path = self.files[self.last_position]
            self._index += 1
            try:
                return self._reader(file_path, self.paths), skipped
            except Exception as exc:
                skipped.append(f"{file_path.name}: {exc}")
        detail = "\n".join(skipped)
        raise StreamReadError(f"All dataset stream files failed to load.\n{detail}")
