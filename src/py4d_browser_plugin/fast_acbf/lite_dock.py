"""Compact 'Lite taskbar' dock: a beginner-friendly one-click reconstruction bar.

Mirrors :class:`~py4d_browser_plugin.fast_acbf.live_view.dock.LiveViewDock` — the native
title bar is collapsed and the name is shown inline with a divider to save vertical space.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class LiteTaskbarDock(QDockWidget):
    """Horizontal dock with an Auto Orientations toggle and tcBF/acBF/Advanced buttons."""

    tcbf_requested = pyqtSignal()
    acbf_requested = pyqtSignal()
    advanced_requested = pyqtSignal()
    closed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("fast-acbf Lite Taskbar", parent)
        self.setObjectName("fastAcbfLiteDock")
        self.setAllowedAreas(Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea)
        # Collapse the native title bar row; the title is shown inline instead.
        self.setTitleBarWidget(QWidget(self))

        self.title_label = QLabel("fast-acbf Lite")
        self.title_divider = QFrame()
        self.title_divider.setFrameShape(QFrame.VLine)
        self.title_divider.setFrameShadow(QFrame.Sunken)
        self.title_divider.setFixedHeight(20)

        self.auto_orientations_cb = QCheckBox("Auto Orientations")
        self.auto_orientations_cb.setChecked(True)
        self.tcbf_btn = QPushButton("tcBF")
        self.acbf_btn = QPushButton("acBF")
        self.advanced_btn = QPushButton("Advanced...")

        body = QWidget(self)
        layout = QHBoxLayout(body)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.title_divider, 0, Qt.AlignVCenter)
        layout.addWidget(self.auto_orientations_cb)
        layout.addWidget(self.tcbf_btn)
        layout.addWidget(self.acbf_btn)
        layout.addWidget(self.advanced_btn)
        layout.addStretch(1)
        self.setWidget(body)

        self.tcbf_btn.clicked.connect(self.tcbf_requested.emit)
        self.acbf_btn.clicked.connect(self.acbf_requested.emit)
        self.advanced_btn.clicked.connect(self.advanced_requested.emit)

    def auto_orientations_enabled(self) -> bool:
        return self.auto_orientations_cb.isChecked()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the action buttons (e.g. while a job runs)."""
        self.tcbf_btn.setEnabled(enabled)
        self.acbf_btn.setEnabled(enabled)
        self.advanced_btn.setEnabled(enabled)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
