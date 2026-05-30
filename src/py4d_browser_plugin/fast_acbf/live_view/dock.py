"""Top dock widget for fast-acbf Live View controls and status."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QPushButton, QWidget

from ..config import FastAcbfConfig


class LiveViewDock(QDockWidget):
    """Compact horizontal dock shown while real Live View is active."""

    closed = pyqtSignal()

    def __init__(
        self,
        config: FastAcbfConfig,
        *,
        start_callback=None,
        stop_callback=None,
        configure_callback=None,
        parent=None,
    ) -> None:
        super().__init__("fast-acbf Live View", parent)
        self.setObjectName("fastAcbfLiveViewDock")
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.configure_callback = configure_callback

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.configure_btn = QPushButton("Configure...")
        self.status_label = QLabel("Live View stopped")
        self.outputs_label = QLabel()
        self.perf_label = QLabel("waiting for datacube updates")
        self.c10_label = QLabel()
        self.alpha_label = QLabel()

        body = QWidget(self)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.configure_btn)
        layout.addWidget(self.status_label)
        layout.addWidget(self.outputs_label)
        layout.addWidget(self.perf_label)
        layout.addWidget(self.c10_label)
        layout.addWidget(self.alpha_label)
        layout.addStretch(1)
        self.setWidget(body)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.configure_btn.clicked.connect(self._configure)
        self.set_config(config)
        self.set_active(False)

    def set_config(self, config: FastAcbfConfig) -> None:
        self.outputs_label.setText(
            "Virtual: "
            f"{config.live_virtual_output}   Result: {config.live_result_output}"
        )
        self.set_c10(float(config.aberrations.get("C10", 0.0)))
        self.set_max_alpha(float(config.max_alpha_mrad))

    def set_c10(self, value: float) -> None:
        self.c10_label.setText(f"C10(-df): {float(value):.5g} Ang")

    def set_max_alpha(self, value: float) -> None:
        self.alpha_label.setText(f"max alpha: {float(value):.5g} mrad")

    def set_active(self, active: bool) -> None:
        self.start_btn.setEnabled(not active)
        self.stop_btn.setEnabled(active)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_metrics(self, metrics: dict) -> None:
        fps = float(metrics.get("fps") or 0.0)
        latency_ms = float(metrics.get("latency_s") or 0.0) * 1000.0
        device = str(metrics.get("device") or "?")
        self.perf_label.setText(
            f"{device}   FPS {fps:.1f}   latency {latency_ms:.1f} ms"
        )
        if "c10_angstrom" in metrics:
            self.set_c10(float(metrics["c10_angstrom"]))
        if "max_alpha_mrad" in metrics:
            self.set_max_alpha(float(metrics["max_alpha_mrad"]))

    def _start(self) -> None:
        if self.start_callback is not None:
            self.start_callback()

    def _stop(self) -> None:
        if self.stop_callback is not None:
            self.stop_callback()

    def _configure(self) -> None:
        if self.configure_callback is not None:
            self.configure_callback()

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
