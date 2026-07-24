"""Compact 'acBF workflow' dock: a beginner-friendly one-click reconstruction bar.

Mirrors :class:`~py4d_browser_plugin.fast_acbf.live_view.dock.LiveViewDock` — the native
title bar is collapsed and the name is shown inline with a divider to save vertical space.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QToolBar,
    QFrame,
    QLabel,
    QAction,
    QWidget,
    QSizePolicy,
    QMenu,
    QToolButton,
    QWidgetAction,
    QComboBox,
    QSpinBox,
    QHBoxLayout,

)


class LiteTaskbarDock(QWidget):
    """Horizontal dock with numbered Orientation/tcBF/Calibration/acBF/Settings/Advanced buttons."""

    orientation_requested = pyqtSignal()
    tcbf_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    acbf_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    advanced_requested = pyqtSignal()
    coarse_defocus_requested = pyqtSignal()
    upscale_changed = pyqtSignal(float)
    closed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.toolbar = QToolBar(parent)
        self.toolbar.setObjectName("fastAcbfLiteDock")

        self.title_label = QLabel("acBF workflow")
        self.title_divider = QFrame()
        self.title_divider.setFrameShape(QFrame.VLine)
        self.title_divider.setFrameShadow(QFrame.Sunken)
        self.title_divider.setFixedHeight(20)

        self.toolbar.addWidget(self.title_label)
        self.toolbar.addWidget(self.title_divider)

        self.orientation_action = QAction("Set Dataset Orientation...", self.toolbar)
        self.tcbf_action = QAction("tcBF", self.toolbar)
        self.calibration_action = QAction("Set Calibrations...", self.toolbar)
        self.acbf_action = QAction("acBF", self.toolbar)
        self.settings_action = QAction("5. Settings", self.toolbar)
        self.advanced_action = QAction("Advanced...", self.toolbar)
        self.coarse_defocus_action = QAction("Coarse Defocus Search", self.toolbar)

        self.actions = [
            self.tcbf_action,
            self.acbf_action,
            self.advanced_action,
        ]

        for action in self.actions:
            self.toolbar.addAction(action)

        # Setup popups for buttons 2 and 4
        # Button 2: tcBF -> Popup: Orientation, Upscale
        btn_tcbf = self.toolbar.widgetForAction(self.tcbf_action)
        if btn_tcbf:
            menu_tcbf = QMenu(self.toolbar)
            menu_tcbf.addAction(self.orientation_action)
            menu_tcbf.addAction(self.coarse_defocus_action)

            # Add Upscale SpinBox via QWidgetAction
            upscale_widget = QWidget()
            upscale_layout = QHBoxLayout(upscale_widget)
            upscale_layout.setContentsMargins(10, 2, 10, 2)
            upscale_spin = QSpinBox(upscale_widget)
            upscale_spin.setMinimum(1)
            upscale_spin.setValue(1)
            upscale_spin.valueChanged.connect(lambda v: self.upscale_changed.emit(float(v)))
            upscale_layout.addWidget(QLabel("Upscale:"))
            upscale_layout.addWidget(upscale_spin)

            upscale_action = QWidgetAction(menu_tcbf)
            upscale_action.setDefaultWidget(upscale_widget)
            menu_tcbf.addAction(upscale_action)

            btn_tcbf.setMenu(menu_tcbf)
            btn_tcbf.setPopupMode(QToolButton.MenuButtonPopup)

        # Button 4: acBF -> Popup: Calibration, Settings
        btn_acbf = self.toolbar.widgetForAction(self.acbf_action)
        if btn_acbf:
            menu_acbf = QMenu(self.toolbar)
            menu_acbf.addAction(self.calibration_action)
            menu_acbf.addAction(self.settings_action)

            # Add Upscale SpinBox via QWidgetAction
            upscale_widget_acbf = QWidget()
            upscale_layout_acbf = QHBoxLayout(upscale_widget_acbf)
            upscale_layout_acbf.setContentsMargins(10, 2, 10, 2)
            upscale_spin_acbf = QSpinBox(upscale_widget_acbf)
            upscale_spin_acbf.setMinimum(1)
            upscale_spin_acbf.setValue(1)
            upscale_spin_acbf.valueChanged.connect(lambda v: self.upscale_changed.emit(float(v)))
            upscale_layout_acbf.addWidget(QLabel("Upscale:"))
            upscale_layout_acbf.addWidget(upscale_spin_acbf)

            upscale_action_acbf = QWidgetAction(menu_acbf)
            upscale_action_acbf.setDefaultWidget(upscale_widget_acbf)
            menu_acbf.addAction(upscale_action_acbf)

            btn_acbf.setMenu(menu_acbf)
            btn_acbf.setPopupMode(QToolButton.MenuButtonPopup)


        # Mirror the 'addStretch(1)' behavior to keep items left-aligned.
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.toolbar.addWidget(spacer)

        self.orientation_action.triggered.connect(self.orientation_requested.emit)
        self.tcbf_action.triggered.connect(self.tcbf_requested.emit)
        self.calibration_action.triggered.connect(self.calibration_requested.emit)
        self.acbf_action.triggered.connect(self.acbf_requested.emit)
        self.settings_action.triggered.connect(self.settings_requested.emit)
        self.advanced_action.triggered.connect(self.advanced_requested.emit)
        self.coarse_defocus_action.triggered.connect(self.coarse_defocus_requested.emit)

        if parent is not None and hasattr(parent, 'addToolBar'):
            parent.addToolBar(self.toolbar)

    def show(self) -> None:
        self.toolbar.show()

    def hide(self) -> None:
        self.toolbar.hide()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the action buttons (e.g. while a job runs)."""
        for action in self.actions:
            action.setEnabled(enabled)

    def hideEvent(self, event) -> None:
        self.closed.emit()
        super().hideEvent(event)
