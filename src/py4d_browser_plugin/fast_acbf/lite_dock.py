"""Compact 'acBF workflow' dock: a beginner-friendly one-click reconstruction bar.

Mirrors :class:`~py4d_browser_plugin.fast_acbf.live_view.dock.LiveViewDock` — the native
title bar is collapsed and the name is shown inline with a divider to save vertical space.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class LiteTaskbarDock(QDockWidget):
    """Horizontal dock with numbered Orientation/tcBF/Calibration/acBF/Settings/Advanced buttons."""

    orientation_requested = pyqtSignal()
    tcbf_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    acbf_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    advanced_requested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("acBF Workflow Taskbar", parent)
        self.setObjectName("fastAcbfLiteDock")
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        # Collapse the native title bar row; the title is shown inline instead.
        self.setTitleBarWidget(QWidget(self))

        self.title_label = QLabel("acBF workflow")
        self.title_divider = QFrame()
        self.title_divider.setFrameShape(QFrame.VLine)
        self.title_divider.setFrameShadow(QFrame.Sunken)
        self.title_divider.setFixedHeight(20)

        self.orientation_btn = QPushButton("1. Orientation")
        self.tcbf_btn = QPushButton("2. tcBF")
        self.calibration_btn = QPushButton("3. Calibration")
        self.acbf_btn = QPushButton("4. acBF")
        self.settings_btn = QPushButton("5. Settings")
        self.advanced_btn = QPushButton("6. Advanced...")

        body = QWidget(self)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.title_divider, 0, Qt.AlignVCenter)
        layout.addWidget(self.orientation_btn)
        layout.addWidget(self.tcbf_btn)
        layout.addWidget(self.calibration_btn)
        layout.addWidget(self.acbf_btn)
        layout.addWidget(self.settings_btn)
        layout.addWidget(self.advanced_btn)
        layout.addStretch(1)
        self.setWidget(body)

        self.orientation_btn.clicked.connect(self.orientation_requested.emit)
        self.tcbf_btn.clicked.connect(self.tcbf_requested.emit)
        self.calibration_btn.clicked.connect(self.calibration_requested.emit)
        self.acbf_btn.clicked.connect(self.acbf_requested.emit)
        self.settings_btn.clicked.connect(self.settings_requested.emit)
        self.advanced_btn.clicked.connect(self.advanced_requested.emit)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the action buttons (e.g. while a job runs)."""
        self.orientation_btn.setEnabled(enabled)
        self.tcbf_btn.setEnabled(enabled)
        self.calibration_btn.setEnabled(enabled)
        self.acbf_btn.setEnabled(enabled)
        self.settings_btn.setEnabled(enabled)
        self.advanced_btn.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
