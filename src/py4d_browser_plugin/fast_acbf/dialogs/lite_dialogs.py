"""Lightweight popup dialogs for the Lite ("acBF workflow") taskbar."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..config import FastAcbfConfig, VALID_UPSCALE_METHODS
from ..solver_job import OptimizeOrientationJob
from ._widgets import OrientationForm


class LiteOrientationDialog(QDialog):
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
            "Refine flips, defocus, and scan rotation (refine_all_params without "
            "the full-order aberration pass)."
        )
        self.optimize_btn.clicked.connect(self._optimize)
        layout.addWidget(self.optimize_btn)

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

    def accept(self) -> None:
        self.config = self._config_with_form_values()
        self.config_changed.emit(self.config.copy())
        super().accept()

    def reject(self) -> None:
        self.set_config(self.config)
        super().reject()


class LiteSettingsDialog(QDialog):
    """Simplest-possible settings popup for the Lite taskbar.

    Exposes only the handful of knobs beginners are likely to tweak day-to-day;
    everything else stays in the full Configuration dialog.
    """

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf: Settings")
        self.config = config.copy()
        self._build_ui()
        self.set_from_config(self.config)

    def _line(self, validator=None) -> QLineEdit:
        line = QLineEdit()
        if validator is not None:
            line.setValidator(validator)
        return line

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.calibration_free_cb = QCheckBox("Calibration-free tcBF/Orientation when uncalibrated")
        self.calibration_free_cb.setToolTip(
            "Measures the BF-disk radius directly in pixels (circular detector selection "
            "if present, otherwise auto-detected) so tcBF and Orientation still work when "
            "the datacube calibration is unset. Automatically stops applying once real "
            "calibration is set."
        )
        self.max_alpha_line = self._line(QDoubleValidator())
        self.upscale_line = self._line(QDoubleValidator())
        self.upscale_method_combo = QComboBox()
        self.upscale_method_combo.addItems(list(VALID_UPSCALE_METHODS))
        self.pad_width_line = self._line(QIntValidator(0, 1000000))
        form.addRow("", self.calibration_free_cb)
        form.addRow("Max alpha [mrad]", self.max_alpha_line)
        form.addRow("Upscale", self.upscale_line)
        form.addRow("Upscale method", self.upscale_method_combo)
        form.addRow("Pad width", self.pad_width_line)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_from_config(self, config: FastAcbfConfig) -> None:
        self.calibration_free_cb.setChecked(bool(config.calibration_free))
        self.max_alpha_line.setText(f"{config.max_alpha_mrad:g}")
        self.upscale_line.setText(f"{config.upscale:g}")
        self.upscale_method_combo.setCurrentText(config.upscale_method)
        pad_width = config.normalized_pad_width()
        self.pad_width_line.setText("0" if pad_width is None else str(pad_width))

    def _float(self, line: QLineEdit, label: str) -> float:
        text = line.text().strip()
        if text == "":
            raise ValueError(f"{label} is required.")
        return float(text)

    def _optional_int(self, line: QLineEdit, label: str) -> int | None:
        text = line.text().strip()
        if text == "":
            return None
        value = int(text)
        if value < 0:
            raise ValueError(f"{label} must be zero or positive.")
        return value if value > 0 else None

    def values(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        cfg.calibration_free = self.calibration_free_cb.isChecked()
        cfg.max_alpha_mrad = self._float(self.max_alpha_line, "Max alpha")
        cfg.upscale = self._float(self.upscale_line, "Upscale")
        cfg.upscale_method = self.upscale_method_combo.currentText()
        cfg.pad_width = self._optional_int(self.pad_width_line, "Pad width")
        cfg.validate_upscale_settings()
        return cfg

    def accept(self) -> None:
        try:
            self.config = self.values()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid fast-acbf settings", str(exc))
            return
        super().accept()
