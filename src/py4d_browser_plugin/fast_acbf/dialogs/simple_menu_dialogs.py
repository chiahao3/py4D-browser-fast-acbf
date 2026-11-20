"""Lightweight popup dialogs for the Simple Menu workflow."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ..calibration import resolved_wavelength_angstrom
from ..config import FastAcbfConfig
from ..solver_job import OptimizeOrientationJob, ReconstructVolumeJob, RefineScanRotationJob
from ._widgets import OrientationForm


def _t1_tolerance_angstrom(config: FastAcbfConfig) -> float | None:
    """T1 tolerance factor (2*lambda / (8*alpha^2)) in Angstrom, or None when max alpha is unset."""
    alpha_mrad = config.max_alpha_mrad
    if alpha_mrad is None or float(alpha_mrad) <= 0.0:
        return None
    wavelength = resolved_wavelength_angstrom(config.wavelength_angstrom)
    alpha_rad = float(alpha_mrad) / 1000.0
    return 2.0 * float(wavelength) / (8.0 * alpha_rad * alpha_rad)


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


class SimpleMenuVolumeDialog(QDialog):
    """Modal dialog for the "Reconstruct Volume..." workflow.

    Asks for a C10 defocus half-width (±, in Å) and a slice count, then emits a
    :class:`ReconstructVolumeJob` on accept.
    """

    run_requested = pyqtSignal(object)
    config_changed = pyqtSignal(object)

    def __init__(self, config: FastAcbfConfig, mode: str, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f"fast-acbf: Reconstruct Volume ({mode})")
        self.mode = mode
        self._build_ui()
        t1 = _t1_tolerance_angstrom(config)
        if t1 is None:
            self.info_label.setText("Max alpha is unset; using a default range of ±100 Å.")
            self.halfwidth_spin.setValue(100.0)
        else:
            halfwidth = 50.0 * t1
            self.info_label.setText(f"T1 ≈ {t1:.4g} Å — defaulting to ±{halfwidth:.4g} Å (50×T1).")
            self.halfwidth_spin.setValue(halfwidth)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.halfwidth_spin = QDoubleSpinBox()
        self.halfwidth_spin.setRange(0.1, 1e9)
        self.halfwidth_spin.setDecimals(1)
        self.halfwidth_spin.setSingleStep(100.0)
        self.halfwidth_spin.setSuffix(" Å")
        self.halfwidth_spin.setToolTip(
            "Defocus half-width: the stack spans C10 ± this value (in Angstrom)."
        )
        form.addRow("Defocus range (±):", self.halfwidth_spin)

        self.slices_spin = QSpinBox()
        self.slices_spin.setRange(2, 1000)
        self.slices_spin.setValue(25)
        self.slices_spin.setToolTip("Number of defocus slices to reconstruct.")
        form.addRow("Number of slices:", self.slices_spin)

        layout.addLayout(form)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        job = ReconstructVolumeJob(
            mode=self.mode,
            halfwidth_angstrom=float(self.halfwidth_spin.value()),
            n_slices=int(self.slices_spin.value()),
        )
        self.run_requested.emit(job)
        super().accept()
