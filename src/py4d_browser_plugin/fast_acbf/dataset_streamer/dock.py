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
    QStyle,
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
        self._active = controller.is_active
        self._playing = controller.is_playing

        self.reverse_btn = QPushButton("Reverse")
        self.play_btn = QPushButton("Play")
        self.next_btn = QPushButton("Next")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.start_btn = self.play_btn
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 3_600_000)
        self.interval_spin.setValue(controller.settings.interval_ms)
        self.interval_label = QLabel("Interval [ms]")
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(False)
        self.configure_btn = QPushButton("Configure...")
        self._set_button_icons()

        body = QWidget(self)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addWidget(self.reverse_btn)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.interval_label)
        layout.addWidget(self.interval_spin)
        layout.addWidget(self.configure_btn)
        layout.addWidget(self.status_label, 1)
        self.setWidget(body)

        self.reverse_btn.clicked.connect(self._reverse)
        self.play_btn.clicked.connect(self._play)
        self.next_btn.clicked.connect(self._next)
        self.pause_btn.clicked.connect(self.controller.pause)
        self.stop_btn.clicked.connect(lambda: self.controller.stop("stopped"))
        self.interval_spin.valueChanged.connect(self.controller.set_interval_ms)
        self.configure_btn.clicked.connect(self._configure)
        self.controller.active_changed.connect(self.set_active)
        self.controller.playback_changed.connect(self.set_playing)
        self.controller.status_changed.connect(self.set_status)
        self.controller.settings_changed.connect(self._settings_changed)
        self.set_active(self.controller.is_active)
        self.set_playing(self.controller.is_playing)

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_transport_state()

    def set_playing(self, playing: bool) -> None:
        self._playing = playing
        self._refresh_transport_state()

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _settings_changed(self, settings) -> None:
        previous = self.interval_spin.blockSignals(True)
        self.interval_spin.setValue(int(settings.interval_ms))
        self.interval_spin.blockSignals(previous)

    def _configure(self) -> None:
        if self.configure_callback is not None:
            self.configure_callback()

    def _play(self) -> None:
        if self.controller.is_active:
            self.controller.play()
            return
        if not self._ensure_folder():
            return
        self.controller.play()

    def _next(self) -> None:
        if self._ensure_folder():
            self.controller.next()

    def _reverse(self) -> None:
        if self._ensure_folder():
            self.controller.reverse()

    def _ensure_folder(self) -> bool:
        if self.controller.settings.folder.strip():
            return True
        message = "Choose a dataset folder before using the stream controls."
        self.set_status(message)
        if self.configure_callback is not None:
            self.configure_callback()
        else:
            QMessageBox.information(self, "Dataset Streamer", message)
        return False

    def _refresh_transport_state(self) -> None:
        self.play_btn.setEnabled(not self._playing)
        self.pause_btn.setEnabled(self._active and self._playing)
        self.stop_btn.setEnabled(self._active)
        self.reverse_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

    def _set_button_icons(self) -> None:
        style = self.style()
        self.reverse_btn.setIcon(style.standardIcon(QStyle.SP_MediaSeekBackward))
        self.play_btn.setIcon(style.standardIcon(QStyle.SP_MediaPlay))
        self.next_btn.setIcon(style.standardIcon(QStyle.SP_MediaSeekForward))
        self.pause_btn.setIcon(style.standardIcon(QStyle.SP_MediaPause))
        self.stop_btn.setIcon(style.standardIcon(QStyle.SP_MediaStop))

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
