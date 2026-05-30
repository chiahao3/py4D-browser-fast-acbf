"""Temporary Dataset Streamer controller for Live View testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from .reader import DEFAULT_STREAM_PATHS, DatasetStreamPaths, DatasetStreamSequence, StreamReadError


@dataclass(frozen=True)
class DatasetStreamerSettings:
    folder: str = ""
    interval_ms: int = 500
    preload: bool = False
    paths: DatasetStreamPaths = DEFAULT_STREAM_PATHS


class TemporaryDatasetStreamerController(QObject):
    """Temporary test harness that mimics an external datacube source."""

    active_changed = pyqtSignal(bool)
    status_changed = pyqtSignal(str)
    error = pyqtSignal(str)
    settings_changed = pyqtSignal(object)

    def __init__(self, parent_viewer, parent=None) -> None:
        super().__init__(parent=parent)
        self.parent_viewer = parent_viewer
        self._settings = DatasetStreamerSettings()
        self._sequence: DatasetStreamSequence | None = None
        self._active = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def settings(self) -> DatasetStreamerSettings:
        return self._settings

    @property
    def sequence(self) -> DatasetStreamSequence | None:
        return self._sequence

    def set_settings(self, settings: DatasetStreamerSettings) -> None:
        interval_changed = int(settings.interval_ms) != int(self._settings.interval_ms)
        self._settings = DatasetStreamerSettings(
            folder=settings.folder,
            interval_ms=max(1, int(settings.interval_ms)),
            preload=bool(settings.preload),
            paths=settings.paths,
        )
        if self._active and interval_changed:
            self._timer.setInterval(self._settings.interval_ms)
            self._emit_progress(f"Dataset Streamer interval set to {self._settings.interval_ms} ms")
        self.settings_changed.emit(self._settings)

    def set_interval_ms(self, interval_ms: int) -> None:
        self.set_settings(
            DatasetStreamerSettings(
                folder=self._settings.folder,
                interval_ms=int(interval_ms),
                preload=self._settings.preload,
                paths=self._settings.paths,
            )
        )

    def start(self) -> bool:
        if self._active:
            return True
        try:
            self._sequence = DatasetStreamSequence.from_folder(
                self._settings.folder,
                self._settings.paths,
                preload=self._settings.preload,
            )
        except Exception as exc:
            self._sequence = None
            message = str(exc)
            self.status_changed.emit(message)
            self.error.emit(message)
            return False
        self._active = True
        self.active_changed.emit(True)
        self._emit_progress(
            f"Dataset Streamer started: {len(self._sequence.files)} file(s), "
            f"preload={self._settings.preload}, interval={self._settings.interval_ms} ms"
        )
        self._on_tick()
        if self._active:
            self._timer.start(self._settings.interval_ms)
        return True

    def stop(self, status: str = "stopped") -> None:
        was_active = self._active
        self._timer.stop()
        self._sequence = None
        self._active = False
        if was_active:
            self.active_changed.emit(False)
        self._emit_progress(f"Dataset Streamer {status}")

    def _on_tick(self) -> None:
        if self._sequence is None:
            return
        try:
            frame, skipped = self._sequence.next_frame()
        except StreamReadError as exc:
            self.stop("stopped: all files failed")
            self.error.emit(str(exc))
            return
        self.parent_viewer.set_datacube(frame.datacube, frame.title)
        message = self._format_frame_message(frame, skipped)
        self._emit_progress(message)

    def _format_frame_message(self, frame, skipped: list[str]) -> str:
        if self._sequence is None:
            prefix = "?/?"
        else:
            position = self._sequence.last_position
            total = (
                len(self._sequence._frames)
                if self._sequence._frames is not None
                else len(self._sequence.files)
            )
            prefix = f"{position + 1}/{total}" if position is not None and total else "?/?"
        data = getattr(frame.datacube, "data", None)
        shape = tuple(getattr(data, "shape", ()))
        calibration = getattr(frame.datacube, "calibration", None)
        try:
            step = calibration.get_R_pixel_size()
            dk = calibration.get_Q_pixel_size()
            voltage = calibration["voltage"]
            cal_text = f"step {float(step):.5g} A, dk {float(dk):.5g} 1/A, {float(voltage):.5g} kV"
        except Exception:
            cal_text = "calibration unavailable"
        message = f"Dataset Streamer {prefix}: {Path(frame.path).name}, shape {shape}, {cal_text}"
        if skipped:
            message += f"; skipped {len(skipped)} file(s)"
        return message

    def _emit_progress(self, message: str) -> None:
        print(message, flush=True)
        self.status_changed.emit(message)
        status_bar = getattr(self.parent_viewer, "statusBar", None)
        if status_bar is None:
            return
        try:
            status_bar().showMessage(message, 5000)
        except Exception:
            pass
