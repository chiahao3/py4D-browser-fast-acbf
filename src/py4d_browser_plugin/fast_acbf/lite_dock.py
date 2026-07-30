"""Compact beginner-friendly fast-acBF workflow toolbar."""

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
    QSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QGridLayout,
)


class _DefocusWidget(QWidget):
    """Compact +/- defocus button pair with a single shared step spinbox, styled to appear as one merged unit."""

    increase_clicked = pyqtSignal()
    decrease_clicked = pyqtSignal()
    step_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(2)
        layout.setHorizontalSpacing(5)

        style = (
            "QToolButton {"
            "  background: palette(button); border: 1px solid palette(midlight); border-radius: 3px;"
            "  color: palette(windowText); font-weight: bold; padding: 2px 10px;"
            "  outline: none;"
            "}"
            "QToolButton:hover { background: palette(highlight); color: palette(highlightedText); }"
        )

        small_label_style = "font-size: 11px; padding: 0;"

        self.c10_label = QLabel("--", self)
        self.c10_label.setStyleSheet("font-size: 12px; padding: 0;")
        self.c10_label.setAlignment(Qt.AlignCenter)

        self._defocus_label = QLabel("Defocus", self)
        self._defocus_label.setStyleSheet(small_label_style)
        self._defocus_label.setAlignment(Qt.AlignCenter)

        self.btn_plus = QToolButton(self)
        self.btn_plus.setText("⬆︎")
        self.btn_plus.setStyleSheet(style)
        self.btn_plus.setFocusPolicy(Qt.NoFocus)

        self.step_spin = QDoubleSpinBox(self)
        self.step_spin.setRange(0.01, 100000)
        self.step_spin.setDecimals(0)
        self.step_spin.setSingleStep(10)
        self.step_spin.setValue(10.0)
        self.step_spin.setSuffix(" Å")
        self.step_spin.setMaximumWidth(100)

        self.btn_minus = QToolButton(self)
        self.btn_minus.setText("⬇︎")
        self.btn_minus.setStyleSheet(style)
        self.btn_minus.setFocusPolicy(Qt.NoFocus)

        self._label = QLabel("Step defocus", self)
        self._label.setStyleSheet(small_label_style)

        layout.addWidget(self.c10_label, 0, 0)
        layout.addWidget(self.btn_plus, 0, 1)
        layout.addWidget(self.step_spin, 0, 2)
        layout.addWidget(self._defocus_label, 1, 0)
        layout.addWidget(self.btn_minus, 1, 1)
        layout.addWidget(self._label, 1, 2, Qt.AlignCenter)

        self.btn_plus.clicked.connect(self.increase_clicked.emit)
        self.btn_minus.clicked.connect(self.decrease_clicked.emit)
        self.step_spin.valueChanged.connect(self.step_changed.emit)

    def set_c10(self, value: float | None) -> None:
        """Display the current defocus value from the solver."""
        if value is not None:
            self.c10_label.setText(f"{value:.0f} Å")
        else:
            self.c10_label.setText("--")


class LiteTaskbarDock(QToolBar):
    """Horizontal toolbar for the simple tcBF/acBF workflow."""

    orientation_requested = pyqtSignal()
    tcbf_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    acbf_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    advanced_requested = pyqtSignal()
    coarse_defocus_requested = pyqtSignal()
    refine_defocus_requested = pyqtSignal()
    upscale_changed = pyqtSignal(float)
    increase_defocus_requested = pyqtSignal()
    decrease_defocus_requested = pyqtSignal()
    defocus_step_changed = pyqtSignal(float)
    closed = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__("Fast acBF", parent)
        self.setObjectName("fastAcbfLiteDock")

        self.title_label = QLabel("Fast acBF")
        self.title_divider = QFrame()
        self.title_divider.setFrameShape(QFrame.VLine)
        self.title_divider.setFrameShadow(QFrame.Sunken)
        self.title_divider.setFixedHeight(20)

        self.addWidget(self.title_label)
        self.addWidget(self.title_divider)

        self.orientation_action = QAction("Set Dataset Orientation...", self)
        self.tcbf_action = QAction("tcBF", self)
        self.calibration_action = QAction("Set Calibrations...", self)
        self.acbf_action = QAction("acBF", self)
        self.settings_action = QAction("Settings...", self)
        self.advanced_action = QAction("Advanced...", self)
        self.coarse_defocus_action = QAction("Coarse Defocus Search", self)
        self.refine_defocus_action = QAction("Refine Defocus", self)

        self._workflow_actions = [
            self.orientation_action,
            self.tcbf_action,
            self.coarse_defocus_action,
            self.refine_defocus_action,
            self.calibration_action,
            self.acbf_action,
            self.settings_action,
            self.advanced_action,
        ]

        for action in (self.tcbf_action, self.acbf_action, self.advanced_action):
            self.addAction(action)

        # Setup popups for buttons 2 and 4
        # Button 2: tcBF -> Popup: Orientation, Upscale
        btn_tcbf = self.widgetForAction(self.tcbf_action)
        if btn_tcbf:
            menu_tcbf = QMenu(self)
            menu_tcbf.addAction(self.orientation_action)
            menu_tcbf.addAction(self.coarse_defocus_action)
            menu_tcbf.addAction(self.refine_defocus_action)

            # Add Upscale SpinBox via QWidgetAction
            upscale_widget = QWidget()
            upscale_layout = QHBoxLayout(upscale_widget)
            self.tcbf_upscale_spin = QSpinBox(upscale_widget)
            self.tcbf_upscale_spin.setMinimum(1)
            self.tcbf_upscale_spin.setValue(1)
            self.tcbf_upscale_spin.valueChanged.connect(
                lambda value: self.upscale_changed.emit(float(value))
            )
            upscale_layout.addWidget(QLabel("Upscale:"))
            upscale_layout.addWidget(self.tcbf_upscale_spin)

            upscale_action = QWidgetAction(menu_tcbf)
            upscale_action.setDefaultWidget(upscale_widget)
            menu_tcbf.addAction(upscale_action)

            btn_tcbf.setMenu(menu_tcbf)
            btn_tcbf.setPopupMode(QToolButton.MenuButtonPopup)

        # Button 4: acBF -> Popup: Calibration, Settings
        btn_acbf = self.widgetForAction(self.acbf_action)
        if btn_acbf:
            menu_acbf = QMenu(self)
            menu_acbf.addAction(self.calibration_action)
            menu_acbf.addAction(self.settings_action)

            # Add Upscale SpinBox via QWidgetAction
            upscale_widget_acbf = QWidget()
            upscale_layout_acbf = QHBoxLayout(upscale_widget_acbf)
            self.acbf_upscale_spin = QSpinBox(upscale_widget_acbf)
            self.acbf_upscale_spin.setMinimum(1)
            self.acbf_upscale_spin.setValue(1)
            self.acbf_upscale_spin.valueChanged.connect(
                lambda value: self.upscale_changed.emit(float(value))
            )
            self.acbf_upscale_spin.valueChanged.connect(self.tcbf_upscale_spin.setValue)
            upscale_layout_acbf.addWidget(QLabel("Upscale:"))
            upscale_layout_acbf.addWidget(self.acbf_upscale_spin)

            upscale_action_acbf = QWidgetAction(menu_acbf)
            upscale_action_acbf.setDefaultWidget(upscale_widget_acbf)
            menu_acbf.addAction(upscale_action_acbf)

            btn_acbf.setMenu(menu_acbf)
            btn_acbf.setPopupMode(QToolButton.MenuButtonPopup)

            # Sync tcBF spinbox → acBF spinbox (reverse direction)
            self.tcbf_upscale_spin.valueChanged.connect(self.acbf_upscale_spin.setValue)

        # Separator + merged defocus +/- widget with shared spinbox
        self.addSeparator()

        self._defocus_widget = _DefocusWidget(self)
        self.addWidget(self._defocus_widget)
        self._defocus_widget.increase_clicked.connect(self.increase_defocus_requested.emit)
        self._defocus_widget.decrease_clicked.connect(self.decrease_defocus_requested.emit)
        self._defocus_widget.step_changed.connect(self.defocus_step_changed.emit)
        # Mirror the 'addStretch(1)' behavior to keep items left-aligned.
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.addWidget(spacer)

        self.orientation_action.triggered.connect(self.orientation_requested.emit)
        self.tcbf_action.triggered.connect(self.tcbf_requested.emit)
        self.calibration_action.triggered.connect(self.calibration_requested.emit)
        self.acbf_action.triggered.connect(self.acbf_requested.emit)
        self.settings_action.triggered.connect(self.settings_requested.emit)
        self.advanced_action.triggered.connect(self.advanced_requested.emit)
        self.coarse_defocus_action.triggered.connect(self.coarse_defocus_requested.emit)
        self.refine_defocus_action.triggered.connect(self.refine_defocus_requested.emit)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the action buttons (e.g. while a job runs)."""
        for action in self._workflow_actions:
            action.setEnabled(enabled)
        self._defocus_widget.setEnabled(enabled)

    def set_c10(self, value: float | None) -> None:
        """Update the displayed C10 defocus value."""
        self._defocus_widget.set_c10(value)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
