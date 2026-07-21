"""Shared widgets used by multiple fast-acbf dialogs.

Both ``ConfigurationDialog`` and ``FastAcbfDashboard`` need the same
aberration coefficient form and the same orientation (rotation + flips)
form. Keeping these as standalone QWidgets prevents the two dialogs
from drifting apart silently when one is edited.
"""

from __future__ import annotations

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import labels_for_order

# Combo labels <-> FastAcbfConfig.focus_sign values. "Overfocus" is index 1 == default.
FOCUS_SIGN_LABELS = [
    ("None", "none"),
    ("Overfocus", "overfocus"),
    ("Underfocus", "underfocus"),
]

FOCUS_SIGN_HELP_TEXT = (
    "tcBF's shift-and-add model is exactly degenerate under a 180° rotation "
    "(equivalently, a simultaneous flipud+fliplr toggle) combined with negating C10 "
    "and any other odd-order aberrations — orientation/defocus optimization can't "
    "tell the two apart from defocus contrast alone.\n\n"
    "Overfocus/Underfocus constrains every defocus search to the chosen sign, resolving "
    "it directly. \"None\" leaves the sign unconstrained; a coma-based tie-break then "
    "runs automatically after Orientation optimization to pick the sharper branch."
)


class AberrationForm(QWidget):
    """Editable form of aberration coefficients keyed by short labels.

    Rebuilds its rows when ``set_max_order`` changes, preserving any
    text the user has already typed for labels that survive the order
    change.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._form = QFormLayout()
        self._layout.addLayout(self._form)
        self.aberration_inputs: dict[str, QLineEdit] = {}
        self._max_order = 0

    def set_max_order(self, max_order: int) -> None:
        previous = self.read_values(quiet=True)
        self._clear()
        self._max_order = int(max_order)
        for label in labels_for_order(self._max_order):
            line = QLineEdit()
            line.setValidator(QDoubleValidator())
            line.setText(f"{float(previous.get(label, 0.0)):g}")
            self.aberration_inputs[label] = line
            self._form.addRow(f"{label} [A]", line)

    def set_values(self, values: dict[str, float]) -> None:
        for label, value in values.items():
            line = self.aberration_inputs.get(label)
            if line is None:
                continue
            line.setText(f"{float(value):g}")

    def read_values(self, *, quiet: bool = False) -> dict[str, float]:
        result: dict[str, float] = {}
        for label, line in self.aberration_inputs.items():
            text = line.text().strip()
            try:
                result[label] = float(text or 0.0)
            except ValueError:
                if quiet:
                    result[label] = 0.0
                else:
                    raise ValueError(f"{label} must be numeric.")
        return result

    def zero_all(self) -> None:
        for line in self.aberration_inputs.values():
            line.setText("0")

    def add_zero_all_button(self) -> QPushButton:
        button = QPushButton("Zero All")
        button.clicked.connect(self.zero_all)
        self._layout.addWidget(button)
        return button

    def _clear(self) -> None:
        while self._form.count():
            item = self._form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.aberration_inputs = {}


class OrientationForm(QWidget):
    """Rotation angle + three orientation flip checkboxes + focus-sign constraint.

    Used identically by ``ConfigurationDialog``'s Orientation tab, the Advanced
    Dashboard's Orientation tab, and the Workflow taskbar's Orientation popup, so all
    three present the same fields for orientation/defocus-sign optimization.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self._layout.addLayout(form)

        self.rotation_line = QLineEdit()
        self.rotation_line.setValidator(QDoubleValidator())
        self.flipud_cb = QCheckBox()
        self.fliplr_cb = QCheckBox()
        self.transpose_cb = QCheckBox()
        form.addRow("Scan rotation [deg]", self.rotation_line)
        form.addRow("Flip up/down", self.flipud_cb)
        form.addRow("Flip left/right", self.fliplr_cb)
        form.addRow("Transpose detector x/y", self.transpose_cb)

        self.focus_sign_combo = QComboBox()
        for label, _value in FOCUS_SIGN_LABELS:
            self.focus_sign_combo.addItem(label)
        self.focus_sign_combo.setCurrentIndex(1)  # default: Overfocus
        self.focus_sign_help_btn = QToolButton()
        self.focus_sign_help_btn.setText("?")
        self.focus_sign_help_btn.setToolTip(FOCUS_SIGN_HELP_TEXT)
        self.focus_sign_help_btn.setAutoRaise(True)
        focus_sign_row = QHBoxLayout()
        focus_sign_row.setContentsMargins(0, 0, 0, 0)
        focus_sign_row.addWidget(self.focus_sign_combo)
        focus_sign_row.addWidget(self.focus_sign_help_btn)
        form.addRow("Focus sign", focus_sign_row)

    def set_values(
        self,
        *,
        rotation_deg: float,
        flipud: bool,
        fliplr: bool,
        transpose: bool,
        focus_sign: str = "overfocus",
    ) -> None:
        self.rotation_line.setText(f"{float(rotation_deg):g}")
        self.flipud_cb.setChecked(bool(flipud))
        self.fliplr_cb.setChecked(bool(fliplr))
        self.transpose_cb.setChecked(bool(transpose))
        self._set_focus_sign(focus_sign)

    def read_values(self) -> dict[str, float | bool | str]:
        text = self.rotation_line.text().strip()
        rotation = float(text) if text else 0.0
        return {
            "rotation_deg": rotation,
            "flipud": self.flipud_cb.isChecked(),
            "fliplr": self.fliplr_cb.isChecked(),
            "transpose": self.transpose_cb.isChecked(),
            "focus_sign": self._focus_sign_value(),
        }

    def _set_focus_sign(self, value: str) -> None:
        target = str(value).strip().lower()
        for index, (_label, item_value) in enumerate(FOCUS_SIGN_LABELS):
            if item_value == target:
                self.focus_sign_combo.setCurrentIndex(index)
                return
        self.focus_sign_combo.setCurrentIndex(1)  # unrecognised -> default: Overfocus

    def _focus_sign_value(self) -> str:
        index = self.focus_sign_combo.currentIndex()
        if 0 <= index < len(FOCUS_SIGN_LABELS):
            return FOCUS_SIGN_LABELS[index][1]
        return "overfocus"

    def reset(self) -> None:
        self.rotation_line.setText("0")
        self.flipud_cb.setChecked(False)
        self.fliplr_cb.setChecked(False)
        self.transpose_cb.setChecked(False)
        self._set_focus_sign("overfocus")

    def add_reset_button(self) -> QPushButton:
        button = QPushButton("Reset Orientation")
        button.clicked.connect(self.reset)
        self._layout.addWidget(button)
        return button
