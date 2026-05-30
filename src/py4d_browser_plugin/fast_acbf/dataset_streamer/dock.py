"""Temporary Dataset Streamer dock controls."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from .controller import TemporaryDatasetStreamerController


class TemporaryDatasetStreamerDock(QDockWidget):
    """Compact controls for the temporary HDF5 dataset streamer."""

    closed = pyqtSignal()

    def __init__(
        self,
        controller: TemporaryDatasetStreamerController,
        *,
        configure_callback=None,
        parent=None,
    ) -> None:
        super().__init__("fast-acbf Dataset Streamer", parent)
        self.setObjectName("fastAcbfDatasetStreamerDock")
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.controller = controller
        self.configure_callback = configure_callback

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 3_600_000)
        self.interval_spin.setValue(controller.settings.interval_ms)
        self.interval_label = QLabel("Interval [ms]")
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(False)
        self.configure_btn = QPushButton("Configure...")

        body = QWidget(self)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.interval_label)
        layout.addWidget(self.interval_spin)
        layout.addWidget(self.configure_btn)
        layout.addWidget(self.status_label, 1)
        self.setWidget(body)

        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(lambda: self.controller.stop("stopped"))
        self.interval_spin.valueChanged.connect(self.controller.set_interval_ms)
        self.configure_btn.clicked.connect(self._configure)
        self.controller.active_changed.connect(self.set_active)
        self.controller.status_changed.connect(self.set_status)
        self.controller.settings_changed.connect(self._settings_changed)
        self.set_active(self.controller.is_active)

    def set_active(self, active: bool) -> None:
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _settings_changed(self, settings) -> None:
        previous = self.interval_spin.blockSignals(True)
        self.interval_spin.setValue(int(settings.interval_ms))
        self.interval_spin.blockSignals(previous)

    def _configure(self) -> None:
        if self.configure_callback is not None:
            self.configure_callback()

    def _start(self) -> None:
        if not self.controller.settings.folder.strip():
            message = "Choose a dataset folder before starting the stream."
            self.set_status(message)
            if self.configure_callback is not None:
                self.configure_callback()
            else:
                QMessageBox.information(self, "Dataset Streamer", message)
            return
        self.controller.start()

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
