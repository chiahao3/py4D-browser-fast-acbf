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
    QDoubleSpinBox,
    QHBoxLayout,
)

_COMPACT_BUTTON_STYLE = (
    "QToolButton {"
    "  background: palette(button); border: 1px solid palette(midlight); border-radius: 3px;"
    "  color: palette(windowText); padding: 2px 10px;"
    "  outline: none;"
    "}"
    "QToolButton:hover { background: palette(highlight); color: palette(highlightedText); }"
)


class _C10Control(QWidget):
    """One-row editor for the current C10 value and its adjustment step."""

    value_requested = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(5)

        self.label = QLabel("C10 (-df)", self)
        self.c10_spin = QDoubleSpinBox(self)
        self.c10_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.c10_spin.setDecimals(2)
        self.c10_spin.setSingleStep(10.0)
        self.c10_spin.setKeyboardTracking(False)
        self.c10_spin.setSuffix(" Å")
        self.c10_spin.setMaximumWidth(120)
        self.c10_spin.setToolTip(
            "Current C10 (-df) in Å. Enter a value and press Enter or leave the field "
            "to update the reconstruction."
        )

        self.gear_button = QToolButton(self)
        self.gear_button.setText("⚙")
        self.gear_button.setToolTip("Change the step size used by the C10 spin box.")
        self.gear_button.setStyleSheet(_COMPACT_BUTTON_STYLE)
        self.gear_button.setFocusPolicy(Qt.NoFocus)
        self.gear_button.setPopupMode(QToolButton.InstantPopup)

        self.step_menu = QMenu(self.gear_button)
        step_widget = QWidget(self.step_menu)
        step_layout = QHBoxLayout(step_widget)
        step_layout.setContentsMargins(8, 4, 8, 4)
        self.step_spin = QDoubleSpinBox(step_widget)
        self.step_spin.setRange(0.01, 100_000.0)
        self.step_spin.setDecimals(2)
        self.step_spin.setValue(10.0)
        self.step_spin.setSuffix(" Å")
        step_layout.addWidget(QLabel("C10 step:", step_widget))
        step_layout.addWidget(self.step_spin)
        step_action = QWidgetAction(self.step_menu)
        step_action.setDefaultWidget(step_widget)
        self.step_menu.addAction(step_action)
        self.gear_button.setMenu(self.step_menu)

        layout.addWidget(self.label)
        layout.addWidget(self.c10_spin)
        layout.addWidget(self.gear_button)

        self.c10_spin.valueChanged.connect(self.value_requested.emit)
        self.step_spin.valueChanged.connect(self.c10_spin.setSingleStep)

    def set_c10(self, value: float | None) -> None:
        """Synchronize the editor without requesting another reconstruction."""
        previous = self.c10_spin.blockSignals(True)
        self.c10_spin.setValue(0.0 if value is None else float(value))
        self.c10_spin.blockSignals(previous)


class SimpleMenuToolbar(QToolBar):
    """Horizontal toolbar for the simple tcBF/acBF workflow."""

    orientation_requested = pyqtSignal()
    tcbf_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    acbf_requested = pyqtSignal()
    advanced_requested = pyqtSignal()
    coarse_defocus_requested = pyqtSignal()
    refine_defocus_requested = pyqtSignal()
    reset_aberrations_requested = pyqtSignal()
    upscale_changed = pyqtSignal(float)
    c10_value_requested = pyqtSignal(float)
    closed = pyqtSignal()

    def __init__(self, parent=None, *, upscale: float = 1.0) -> None:
        super().__init__("Fast acBF", parent)
        self.setObjectName("fastAcbfSimpleMenuToolbar")

        self.title_label = QLabel("Fast acBF")
        self.title_divider = QFrame()
        self.title_divider.setFrameShape(QFrame.VLine)
        self.title_divider.setFrameShadow(QFrame.Sunken)

        self.title_widget = QWidget(self)
        self.title_layout = QHBoxLayout(self.title_widget)
        self.title_layout.setContentsMargins(8, 0, 8, 0)
        self.title_layout.setSpacing(12)
        self.title_layout.addWidget(self.title_label)
        self.title_layout.addWidget(self.title_divider, 0, Qt.AlignVCenter)
        self.addWidget(self.title_widget)

        self.orientation_action = QAction("Set Dataset Orientation...", self)
        self.tcbf_action = QAction("tcBF", self)
        self.calibration_action = QAction("Set Calibrations...", self)
        self.acbf_action = QAction("acBF", self)
        self.advanced_action = QAction("Advanced", self)
        self.coarse_defocus_action = QAction("Coarse Defocus Search", self)
        self.refine_defocus_action = QAction("Refine Defocus", self)
        self.reset_aberrations_action = QAction("Reset Aberrations", self)
        self.reset_aberrations_action.setToolTip(
            "Set all aberrations, including C10 (-df), to zero and update the reconstruction."
        )

        self._workflow_actions = [
            self.orientation_action,
            self.tcbf_action,
            self.coarse_defocus_action,
            self.refine_defocus_action,
            self.reset_aberrations_action,
            self.calibration_action,
            self.acbf_action,
            self.advanced_action,
        ]

        for action in (self.tcbf_action, self.acbf_action, self.advanced_action):
            self.addAction(action)

        # Setup popups for buttons 2 and 4
        # Button 2: tcBF -> Popup: Orientation, Upscale
        self.tcbf_button = self.widgetForAction(self.tcbf_action)
        if self.tcbf_button:
            menu_tcbf = QMenu(self)
            menu_tcbf.addAction(self.orientation_action)
            menu_tcbf.addAction(self.coarse_defocus_action)
            menu_tcbf.addAction(self.refine_defocus_action)
            menu_tcbf.addSeparator()
            menu_tcbf.addAction(self.reset_aberrations_action)

            # Add Upscale SpinBox via QWidgetAction
            upscale_widget = QWidget()
            upscale_layout = QHBoxLayout(upscale_widget)
            self.tcbf_upscale_spin = QDoubleSpinBox(upscale_widget)
            self.tcbf_upscale_spin.setRange(1.0, 100.0)
            self.tcbf_upscale_spin.setDecimals(2)
            self.tcbf_upscale_spin.setSingleStep(1.0)
            self.tcbf_upscale_spin.setValue(float(upscale))
            upscale_layout.addWidget(QLabel("Upscale:"))
            upscale_layout.addWidget(self.tcbf_upscale_spin)

            upscale_action = QWidgetAction(menu_tcbf)
            upscale_action.setDefaultWidget(upscale_widget)
            menu_tcbf.addAction(upscale_action)

            self.tcbf_button.setMenu(menu_tcbf)
            self.tcbf_button.setPopupMode(QToolButton.MenuButtonPopup)

        # Button 4: acBF -> Popup: Calibration, Upscale
        self.acbf_button = self.widgetForAction(self.acbf_action)
        if self.acbf_button:
            menu_acbf = QMenu(self)
            menu_acbf.addAction(self.calibration_action)

            # Add Upscale SpinBox via QWidgetAction
            upscale_widget_acbf = QWidget()
            upscale_layout_acbf = QHBoxLayout(upscale_widget_acbf)
            self.acbf_upscale_spin = QDoubleSpinBox(upscale_widget_acbf)
            self.acbf_upscale_spin.setRange(1.0, 100.0)
            self.acbf_upscale_spin.setDecimals(2)
            self.acbf_upscale_spin.setSingleStep(1.0)
            self.acbf_upscale_spin.setValue(float(upscale))
            self.acbf_upscale_spin.valueChanged.connect(
                lambda value: self._sync_upscale(value, self.tcbf_upscale_spin)
            )
            upscale_layout_acbf.addWidget(QLabel("Upscale:"))
            upscale_layout_acbf.addWidget(self.acbf_upscale_spin)

            upscale_action_acbf = QWidgetAction(menu_acbf)
            upscale_action_acbf.setDefaultWidget(upscale_widget_acbf)
            menu_acbf.addAction(upscale_action_acbf)

            self.acbf_button.setMenu(menu_acbf)
            self.acbf_button.setPopupMode(QToolButton.MenuButtonPopup)

            self.tcbf_upscale_spin.valueChanged.connect(
                lambda value: self._sync_upscale(value, self.acbf_upscale_spin)
            )

        # Separator + direct C10 editor
        self.addSeparator()

        self._c10_control = _C10Control(self)
        self.addWidget(self._c10_control)
        self.advanced_button = self.widgetForAction(self.advanced_action)
        self._style_primary_buttons()
        self._c10_control.value_requested.connect(self.c10_value_requested.emit)
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
        self.advanced_action.triggered.connect(self.advanced_requested.emit)
        self.coarse_defocus_action.triggered.connect(self.coarse_defocus_requested.emit)
        self.refine_defocus_action.triggered.connect(self.refine_defocus_requested.emit)
        self.reset_aberrations_action.triggered.connect(
            self.reset_aberrations_requested.emit
        )

    def _style_primary_buttons(self) -> None:
        """Give the three main actions the same subtle outline as the compact controls."""
        for button in (self.tcbf_button, self.acbf_button, self.advanced_button):
            if button is not None:
                button.setStyleSheet(_COMPACT_BUTTON_STYLE)
        self.title_divider.setFixedHeight(20)

    def _sync_upscale(self, value: float, other: QDoubleSpinBox) -> None:
        previous = other.blockSignals(True)
        other.setValue(float(value))
        other.blockSignals(previous)
        self.upscale_changed.emit(float(value))

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the action buttons (e.g. while a job runs)."""
        for action in self._workflow_actions:
            action.setEnabled(enabled)
        self._c10_control.setEnabled(enabled)

    def set_c10(self, value: float | None) -> None:
        """Update the displayed C10 defocus value."""
        self._c10_control.set_c10(value)

    def reset_for_new_dataset(self, config) -> None:
        """Restore dataset-dependent controls to a fresh configuration."""
        for spin in (self.tcbf_upscale_spin, self.acbf_upscale_spin):
            previous = spin.blockSignals(True)
            spin.setValue(float(config.upscale))
            spin.blockSignals(previous)
        previous = self._c10_control.step_spin.blockSignals(True)
        self._c10_control.step_spin.setValue(10.0)
        self._c10_control.step_spin.blockSignals(previous)
        self._c10_control.c10_spin.setSingleStep(10.0)
        self.set_c10(config.aberrations.get("C10", 0.0))

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)
