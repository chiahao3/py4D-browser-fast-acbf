"""Temporary Dataset Streamer configuration dialog for Live View testing."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
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

from .controller import DatasetStreamerSettings, TemporaryDatasetStreamerController
from .reader import DEFAULT_STREAM_PATHS, DatasetStreamPaths


class TemporaryDatasetStreamerDialog(QDialog):
    """Temporary settings window for the HDF5 dataset streamer test harness."""

    settings_applied = pyqtSignal(object)

    def __init__(self, parent_viewer, parent=None, controller=None) -> None:
        super().__init__(parent=parent)
        self.parent_viewer = parent_viewer
        self.controller = controller or TemporaryDatasetStreamerController(parent_viewer, parent=self)
        self.setWindowTitle("fast-acbf: Dataset Streamer")
        self.resize(640, 320)
        self._build_ui()
        self._load_settings(self.controller.settings)
        self.controller.active_changed.connect(self._stream_active_changed)
        self.controller.status_changed.connect(self.status_label.setText)
        self.controller.error.connect(self._show_error)
        self.controller.settings_changed.connect(self._load_settings)

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

        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)
        main.addWidget(self.apply_btn)
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

    def settings(self) -> DatasetStreamerSettings:
        return DatasetStreamerSettings(
            folder=self.folder_line.text().strip(),
            interval_ms=int(self.interval_spin.value()),
            preload=self.preload_cb.isChecked(),
            paths=self.stream_paths(),
        )

    def apply_settings(self) -> None:
        settings = self.settings()
        self.controller.set_settings(settings)
        self.settings_applied.emit(settings)
        if not self.controller.is_active:
            self.status_label.setText("settings applied")

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Dataset Streamer Folder")
        if folder:
            self.folder_line.setText(folder)

    def _stream_active_changed(self, active: bool) -> None:
        self._set_controls_enabled(not active)
        if active:
            self.status_label.setText("streaming")

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Dataset Streamer", message)

    def _load_settings(self, settings: DatasetStreamerSettings) -> None:
        previous = [
            self.folder_line.blockSignals(True),
            self.interval_spin.blockSignals(True),
            self.preload_cb.blockSignals(True),
            self.data_path_line.blockSignals(True),
            self.scan_step_path_line.blockSignals(True),
            self.dk_path_line.blockSignals(True),
            self.voltage_path_line.blockSignals(True),
        ]
        self.folder_line.setText(settings.folder)
        self.interval_spin.setValue(int(settings.interval_ms))
        self.preload_cb.setChecked(bool(settings.preload))
        self.data_path_line.setText(settings.paths.data)
        self.scan_step_path_line.setText(settings.paths.scan_step_angstrom)
        self.dk_path_line.setText(settings.paths.dk_inv_angstrom)
        self.voltage_path_line.setText(settings.paths.voltage_kv)
        for widget, blocked in zip(
            (
                self.folder_line,
                self.interval_spin,
                self.preload_cb,
                self.data_path_line,
                self.scan_step_path_line,
                self.dk_path_line,
                self.voltage_path_line,
            ),
            previous,
        ):
            widget.blockSignals(blocked)

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.folder_line,
            self.interval_spin,
            self.preload_cb,
            self.data_path_line,
            self.scan_step_path_line,
            self.dk_path_line,
            self.voltage_path_line,
            self.apply_btn,
        ):
            widget.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
