"""Top dock widget for fast-acbf Live View controls and status."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDockWidget, QHBoxLayout, QLabel, QWidget

from ..config import FastAcbfConfig


class LiveViewDock(QDockWidget):
    """Compact horizontal dock shown while real Live View is active."""

    def __init__(self, config: FastAcbfConfig, parent=None) -> None:
        super().__init__("fast-acbf Live View", parent)
        self.setObjectName("fastAcbfLiveViewDock")
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        self.status_label = QLabel("Live View active")
        self.outputs_label = QLabel()
        self.perf_label = QLabel("waiting for datacube updates")
        self.c10_label = QLabel()

        body = QWidget(self)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(16)
        layout.addWidget(self.status_label)
        layout.addWidget(self.outputs_label)
        layout.addWidget(self.perf_label)
        layout.addWidget(self.c10_label)
        layout.addStretch(1)
        self.setWidget(body)
        self.set_config(config)

    def set_config(self, config: FastAcbfConfig) -> None:
        self.outputs_label.setText(
            "Virtual: "
            f"{config.live_virtual_output}   Result: {config.live_result_output}"
        )
        self.set_c10(float(config.aberrations.get("C10", 0.0)))

    def set_c10(self, value: float) -> None:
        self.c10_label.setText(f"C10(-df): {float(value):.5g} Ang")

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
