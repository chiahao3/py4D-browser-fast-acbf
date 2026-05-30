import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu

from py4d_browser_plugin.fast_acbf.dataset_streamer import (
    DatasetStreamFrame,
    DatasetStreamPaths,
    DatasetStreamSequence,
    StreamReadError,
    TemporaryDatasetStreamerDialog,
    discover_hdf5_stream_files,
    read_hdf5_stream_frame,
)
from py4d_browser_plugin.fast_acbf.plugin import FastAcbfPlugin


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


def _write_stream_file(
    path: Path,
    *,
    data_shape=(2, 3, 4, 5),
    scan_step=1.5,
    dk=0.25,
    voltage=200.0,
    paths=DatasetStreamPaths(),
):
    with h5py.File(path, "w") as h5:
        h5.create_dataset(paths.data, data=np.ones(data_shape, dtype=np.float32))
        h5.create_dataset(paths.scan_step_angstrom, data=scan_step)
        h5.create_dataset(paths.dk_inv_angstrom, data=dk)
        h5.create_dataset(paths.voltage_kv, data=voltage)


def test_hdf5_stream_reader_reads_4d_data_and_calibration(tmp_path):
    file_path = tmp_path / "frame_001.h5"
    _write_stream_file(file_path, scan_step=2.0, dk=0.125, voltage=80.0)

    frame = read_hdf5_stream_frame(file_path, DatasetStreamPaths())

    assert frame.path == file_path
    assert frame.datacube.data.shape == (2, 3, 4, 5)
    assert frame.datacube.calibration.get_R_pixel_size() == 2.0
    assert frame.datacube.calibration.get_R_pixel_units() == "A"
    assert frame.datacube.calibration.get_Q_pixel_size() == 0.125
    assert frame.datacube.calibration.get_Q_pixel_units() == "A^-1"
    assert frame.datacube.calibration["voltage"] == 80.0


def test_hdf5_stream_reader_reports_missing_and_non_4d_data(tmp_path):
    missing_cal = tmp_path / "missing_cal.h5"
    with h5py.File(missing_cal, "w") as h5:
        h5.create_dataset("/data", data=np.ones((2, 2, 4, 4), dtype=np.float32))

    with pytest.raises(StreamReadError, match="scan step"):
        read_hdf5_stream_frame(missing_cal, DatasetStreamPaths())

    non_4d = tmp_path / "non_4d.h5"
    _write_stream_file(non_4d, data_shape=(2, 3, 4))
    with pytest.raises(StreamReadError, match="must be 4D"):
        read_hdf5_stream_frame(non_4d, DatasetStreamPaths())


def test_dataset_stream_sequence_discovers_sorted_files_and_loops_lazy(tmp_path):
    for name in ("b.h5", "a.hdf5", "ignore.txt"):
        (tmp_path / name).write_text("x")
    files = discover_hdf5_stream_files(tmp_path)
    assert [path.name for path in files] == ["a.hdf5", "b.h5"]

    calls = []

    def _reader(path, paths):
        calls.append(path.name)
        return DatasetStreamFrame(path=path, datacube=object(), title=path.name)

    sequence = DatasetStreamSequence(files, reader=_reader)
    assert sequence.next_frame()[0].title == "a.hdf5"
    assert sequence.next_frame()[0].title == "b.h5"
    assert sequence.next_frame()[0].title == "a.hdf5"
    assert calls == ["a.hdf5", "b.h5", "a.hdf5"]


def test_dataset_stream_sequence_preloads_and_reuses_cached_frames(tmp_path):
    files = [tmp_path / "a.h5", tmp_path / "b.h5"]
    for file_path in files:
        file_path.write_text("x")
    calls = []

    def _reader(path, paths):
        calls.append(path.name)
        return DatasetStreamFrame(path=path, datacube=object(), title=path.name)

    sequence = DatasetStreamSequence(files, preload=True, reader=_reader)
    first = sequence.next_frame()[0]
    second = sequence.next_frame()[0]
    first_again = sequence.next_frame()[0]

    assert calls == ["a.h5", "b.h5"]
    assert first.title == "a.h5"
    assert second.title == "b.h5"
    assert first_again is first


def test_dataset_stream_sequence_skips_bad_lazy_files_and_stops_if_all_fail(tmp_path):
    files = [tmp_path / "bad.h5", tmp_path / "good.h5"]
    for file_path in files:
        file_path.write_text("x")

    def _reader(path, paths):
        if path.name == "bad.h5":
            raise StreamReadError("bad file")
        return DatasetStreamFrame(path=path, datacube=object(), title=path.name)

    sequence = DatasetStreamSequence(files, reader=_reader)
    frame, skipped = sequence.next_frame()
    assert frame.title == "good.h5"
    assert skipped and "bad.h5" in skipped[0]

    def _always_bad(path, paths):
        raise StreamReadError("bad file")

    sequence = DatasetStreamSequence(files, reader=_always_bad)
    with pytest.raises(StreamReadError, match="All dataset stream files failed"):
        sequence.next_frame()


class _FakeViewer(QMainWindow):
    signal_datacube_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.frames = []

    def set_datacube(self, datacube, window_title):
        self.frames.append((datacube, window_title))
        self.signal_datacube_changed.emit()


def test_temporary_dataset_streamer_dialog_updates_parent_datacube(monkeypatch, tmp_path):
    _app()
    import py4d_browser_plugin.fast_acbf.dataset_streamer.dialog as dialog_module

    parent = _FakeViewer()
    frames = [
        DatasetStreamFrame(tmp_path / "a.h5", object(), "a.h5"),
        DatasetStreamFrame(tmp_path / "b.h5", object(), "b.h5"),
    ]
    captured = {}

    class _Sequence:
        def __init__(self):
            self.index = 0

        def next_frame(self):
            frame = frames[self.index % len(frames)]
            self.index += 1
            return frame, []

    def _from_folder(folder, paths, *, preload=False):
        captured.update({"folder": folder, "paths": paths, "preload": preload})
        return _Sequence()

    monkeypatch.setattr(dialog_module.DatasetStreamSequence, "from_folder", _from_folder)

    dialog = TemporaryDatasetStreamerDialog(parent, parent=parent)
    dialog.folder_line.setText(str(tmp_path))
    dialog.interval_spin.setValue(25)
    dialog.preload_cb.setChecked(True)
    dialog.data_path_line.setText("/frames/data")

    dialog.start_stream()
    dialog._timer.stop()
    dialog._on_tick()

    assert captured["folder"] == str(tmp_path)
    assert captured["paths"].data == "/frames/data"
    assert captured["preload"] is True
    assert [title for _datacube, title in parent.frames] == ["a.h5", "b.h5"]
    assert "streamed b.h5" in dialog.status_label.text()
    dialog.close()


def test_plugin_exposes_temporary_dataset_streamer_action(monkeypatch):
    _app()
    parent = _FakeViewer()
    menu = QMenu(parent)
    plugin = FastAcbfPlugin(parent, menu)
    shown = []

    class _Dialog:
        def __init__(self, parent_viewer, parent=None):
            self.parent_viewer = parent_viewer
            self.destroyed = SimpleNamespace(connect=lambda callback: None)

        def show(self):
            shown.append("show")

        def raise_(self):
            shown.append("raise")

        def close(self):
            shown.append("close")

    monkeypatch.setattr("py4d_browser_plugin.fast_acbf.plugin.TemporaryDatasetStreamerDialog", _Dialog)

    labels = [action.text() for action in menu.actions()]
    assert "Dataset Streamer (Temporary)" in labels

    plugin.launch_dataset_streamer()

    assert shown == ["show", "raise"]
