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
    QFormLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import labels_for_order


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
    """Rotation angle + three orientation flip checkboxes."""

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

    def set_values(
        self,
        *,
        rotation_deg: float,
        flipud: bool,
        fliplr: bool,
        transpose: bool,
    ) -> None:
        self.rotation_line.setText(f"{float(rotation_deg):g}")
        self.flipud_cb.setChecked(bool(flipud))
        self.fliplr_cb.setChecked(bool(fliplr))
        self.transpose_cb.setChecked(bool(transpose))

    def read_values(self) -> dict[str, float | bool]:
        text = self.rotation_line.text().strip()
        rotation = float(text) if text else 0.0
        return {
            "rotation_deg": rotation,
            "flipud": self.flipud_cb.isChecked(),
            "fliplr": self.fliplr_cb.isChecked(),
            "transpose": self.transpose_cb.isChecked(),
        }

    def reset(self) -> None:
        self.rotation_line.setText("0")
        self.flipud_cb.setChecked(False)
        self.fliplr_cb.setChecked(False)
        self.transpose_cb.setChecked(False)

    def add_reset_button(self) -> QPushButton:
        button = QPushButton("Reset Orientation")
        button.clicked.connect(self.reset)
        self._layout.addWidget(button)
        return button
