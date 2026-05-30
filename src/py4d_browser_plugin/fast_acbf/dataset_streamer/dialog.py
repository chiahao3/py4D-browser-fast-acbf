"""Temporary Dataset Streamer dialog for Live View testing."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from .reader import DEFAULT_STREAM_PATHS, DatasetStreamPaths, DatasetStreamSequence, StreamReadError


class TemporaryDatasetStreamerDialog(QDialog):
    """Temporary test harness that mimics an external datacube source."""

    def __init__(self, parent_viewer, parent=None) -> None:
        super().__init__(parent=parent)
        self.parent_viewer = parent_viewer
        self.setWindowTitle("fast-acbf: Dataset Streamer (Temporary)")
        self.resize(640, 320)
        self._sequence: DatasetStreamSequence | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        form = QFormLayout()
        self.folder_line = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_line)
        folder_row.addWidget(browse_btn)
        form.addRow("Folder", folder_row)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 3_600_000)
        self.interval_spin.setValue(500)
        form.addRow("Interval [ms]", self.interval_spin)

        self.preload_cb = QCheckBox("Preload all datasets into RAM")
        form.addRow("", self.preload_cb)

        self.data_path_line = QLineEdit(DEFAULT_STREAM_PATHS.data)
        self.scan_step_path_line = QLineEdit(DEFAULT_STREAM_PATHS.scan_step_angstrom)
        self.dk_path_line = QLineEdit(DEFAULT_STREAM_PATHS.dk_inv_angstrom)
        self.voltage_path_line = QLineEdit(DEFAULT_STREAM_PATHS.voltage_kv)
        form.addRow("Data path", self.data_path_line)
        form.addRow("Scan step path", self.scan_step_path_line)
        form.addRow("dk path", self.dk_path_line)
        form.addRow("Voltage path", self.voltage_path_line)
        main.addLayout(form)

        self.toggle_btn = QPushButton("Start Stream")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._toggle_stream)
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)
        main.addWidget(self.toggle_btn)
        main.addWidget(self.status_label)

    def stream_paths(self) -> DatasetStreamPaths:
        return DatasetStreamPaths(
            data=self.data_path_line.text().strip() or DEFAULT_STREAM_PATHS.data,
            scan_step_angstrom=(
                self.scan_step_path_line.text().strip()
                or DEFAULT_STREAM_PATHS.scan_step_angstrom
            ),
            dk_inv_angstrom=self.dk_path_line.text().strip() or DEFAULT_STREAM_PATHS.dk_inv_angstrom,
            voltage_kv=self.voltage_path_line.text().strip() or DEFAULT_STREAM_PATHS.voltage_kv,
        )

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dataset Streamer Folder")
        if folder:
            self.folder_line.setText(folder)

    def _toggle_stream(self, checked: bool) -> None:
        if checked:
            self.start_stream()
        else:
            self.stop_stream("stopped")

    def start_stream(self) -> None:
        try:
            self._sequence = DatasetStreamSequence.from_folder(
                self.folder_line.text().strip(),
                self.stream_paths(),
                preload=self.preload_cb.isChecked(),
            )
        except Exception as exc:
            self._sequence = None
            self._set_toggle_checked(False)
            self.status_label.setText(str(exc))
            QMessageBox.warning(self, "Dataset Streamer", str(exc))
            return
        self._set_controls_enabled(False)
        self._set_toggle_checked(True)
        self.status_label.setText("streaming")
        self._on_tick()
        self._timer.start(int(self.interval_spin.value()))

    def stop_stream(self, status: str = "stopped") -> None:
        self._timer.stop()
        self._sequence = None
        self._set_controls_enabled(True)
        self._set_toggle_checked(False)
        self.status_label.setText(status)

    def _on_tick(self) -> None:
        if self._sequence is None:
            return
        try:
            frame, skipped = self._sequence.next_frame()
        except StreamReadError as exc:
            self.stop_stream("stopped: all files failed")
            QMessageBox.warning(self, "Dataset Streamer", str(exc))
            return
        self.parent_viewer.set_datacube(frame.datacube, frame.title)
        message = f"streamed {Path(frame.path).name}"
        if skipped:
            message += f"; skipped {len(skipped)} file(s)"
        self.status_label.setText(message)

    def _set_toggle_checked(self, checked: bool) -> None:
        previous = self.toggle_btn.blockSignals(True)
        self.toggle_btn.setChecked(checked)
        self.toggle_btn.setText("Stop Stream" if checked else "Start Stream")
        self.toggle_btn.blockSignals(previous)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.folder_line,
            self.interval_spin,
            self.preload_cb,
            self.data_path_line,
            self.scan_step_path_line,
            self.dk_path_line,
            self.voltage_path_line,
        ):
            widget.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        self.stop_stream("stopped")
        super().closeEvent(event)
