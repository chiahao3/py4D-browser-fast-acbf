"""Lightweight popup dialogs for the Simple Menu workflow."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..config import FastAcbfConfig
from ..solver_job import OptimizeOrientationJob, RefineScanRotationJob
from ._widgets import OrientationForm


class SimpleMenuOrientationDialog(QDialog):
    """Non-modal popup mirroring the Advanced Dashboard's Orientation tab.

    Adds an "Optimize Orientation" button that runs :class:`OptimizeOrientationJob`
    (``refine_all_params`` without the ``fine_aberrations`` target) on the current config.
    """

    run_requested = pyqtSignal(object)
    config_changed = pyqtSignal(object)

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf: Orientation")
        self.config = config.copy()
        self._build_ui()
        self.set_config(self.config)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.orientation_form = OrientationForm()
        layout.addWidget(self.orientation_form)
        self.orientation_form.add_reset_button()

        self.optimize_btn = QPushButton("Optimize Orientation")
        self.optimize_btn.setToolTip(
            "Refine flips, defocus, scan rotation, and 1st/2nd-order aberrations"
        )
        self.optimize_btn.clicked.connect(self._optimize)
        layout.addWidget(self.optimize_btn)

        self.refine_rotation_btn = QPushButton("Refine Scan Rotation")
        self.refine_rotation_btn.setToolTip("Refine only the scan rotation angle")
        self.refine_rotation_btn.clicked.connect(self._refine_rotation)
        layout.addWidget(self.refine_rotation_btn)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_config(self, config: FastAcbfConfig) -> None:
        self.config = config.copy()
        self.orientation_form.set_values(
            rotation_deg=float(self.config.rotation_deg),
            flipud=bool(self.config.flipud),
            fliplr=bool(self.config.fliplr),
            transpose=bool(self.config.transpose),
            focus_sign=str(self.config.focus_sign),
        )

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _config_with_form_values(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        orient = self.orientation_form.read_values()
        cfg.rotation_deg = float(orient["rotation_deg"])
        cfg.flipud = bool(orient["flipud"])
        cfg.fliplr = bool(orient["fliplr"])
        cfg.transpose = bool(orient["transpose"])
        cfg.focus_sign = str(orient["focus_sign"])
        return cfg

    def _optimize(self) -> None:
        self.config = self._config_with_form_values()
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit(OptimizeOrientationJob())

    def _refine_rotation(self) -> None:
        self.config = self._config_with_form_values()
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit(RefineScanRotationJob())

    def accept(self) -> None:
        self.config = self._config_with_form_values()
        self.config_changed.emit(self.config.copy())
        super().accept()

    def reject(self) -> None:
        self.set_config(self.config)
        super().reject()
