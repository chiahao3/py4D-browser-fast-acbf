"""Lightweight popup dialogs for the Simple Menu workflow."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPalette
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

from ..calibration import max_alpha_mrad_from_px, resolved_wavelength_angstrom
from ..config import FastAcbfConfig, VALID_UPSCALE_METHODS
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
        self.refine_rotation_btn.setToolTip(
            "Refine only the scan rotation angle"
        )
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


class SimpleMenuSettingsDialog(QDialog):
    """Simplest-possible settings popup for the Simple Menu.

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

    def _read_only_line(self) -> QLineEdit:
        """A derived-value display: read-only, and visually muted so it reads as
        uneditable regardless of the active Qt style/theme (matches the palette's
        own disabled-window shade rather than a hardcoded color)."""
        line = self._line()
        line.setReadOnly(True)
        palette = line.palette()
        palette.setColor(line.backgroundRole(), palette.color(QPalette.Window))
        line.setPalette(palette)
        return line

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.calibration_free_cb = QCheckBox("Calibration-free tcBF/Orientation when uncalibrated")
        self.calibration_free_cb.setToolTip(
            "Operates on scan step/dk directly in pixels so tcBF and Orientation still "
            "work when the datacube calibration is unset. Automatically stops applying "
            "once real calibration is set. Max alpha is always resolved from the BF-disk "
            "radius in pixels regardless of this setting -- see Max alpha [px] below."
        )
        self.max_alpha_px_line = self._line(QDoubleValidator())
        self.max_alpha_px_line.setToolTip(
            "BF-disk radius in detector pixels: a circular detector selection if present, "
            "else auto-detected from the position-averaged CBED. Editable -- correct it "
            "here if the auto-fit got the disk edge wrong. Changing this only resizes the "
            "reconstruction mask (smaller than the true BF disk trims how much of the "
            "diffraction pattern feeds the virtual BF image); it does not touch the dk "
            "calibration. Note: a live circular detector selection still wins over a "
            "manual edit here on the next refresh."
        )
        self.max_alpha_line = self._read_only_line()
        self.max_alpha_line.setToolTip(
            "Derived from Max alpha [px] x dk x wavelength; not directly editable -- edit "
            "Max alpha [px] instead."
        )
        self.upscale_line = self._line(QDoubleValidator())
        self.upscale_method_combo = QComboBox()
        self.upscale_method_combo.addItems(list(VALID_UPSCALE_METHODS))
        self.pad_width_line = self._line(QIntValidator(0, 1000000))
        self.max_alpha_px_line.textEdited.connect(self._update_max_alpha_mrad_display)
        form.addRow("", self.calibration_free_cb)
        form.addRow("Max alpha [px]", self.max_alpha_px_line)
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
        self.max_alpha_line.setText(self._optional_float_text(config.max_alpha_mrad))
        self.max_alpha_px_line.setText(self._optional_float_text(config.max_alpha_px))
        self.upscale_line.setText(f"{config.upscale:g}")
        self.upscale_method_combo.setCurrentText(config.upscale_method)
        pad_width = config.normalized_pad_width()
        self.pad_width_line.setText("0" if pad_width is None else str(pad_width))

    def _float(self, line: QLineEdit, label: str) -> float:
        text = line.text().strip()
        if text == "":
            raise ValueError(f"{label} is required.")
        return float(text)

    def _optional_float_text(self, value: float | None) -> str:
        return "" if value is None else f"{float(value):g}"

    def _optional_float(self, line: QLineEdit, label: str) -> float | None:
        text = line.text().strip()
        if text == "":
            return None
        return float(text)

    def _optional_int(self, line: QLineEdit, label: str) -> int | None:
        text = line.text().strip()
        if text == "":
            return None
        value = int(text)
        if value < 0:
            raise ValueError(f"{label} must be zero or positive.")
        return value if value > 0 else None

    def _update_max_alpha_mrad_display(self, *_args) -> None:
        """Keep the read-only Max alpha [mrad] display in sync with Max alpha [px]
        (the editable source of truth), using this dialog's fixed dk/wavelength --
        mirroring the conversion FastAcbfConfig.resolved_for applies."""
        px_text = self.max_alpha_px_line.text().strip()
        if px_text == "":
            self.max_alpha_line.setText("")
            return
        try:
            px = float(px_text)
        except ValueError:
            return
        dk = float(self.config.dk_inv_angstrom)
        wavelength = resolved_wavelength_angstrom(self.config.wavelength_angstrom)
        self.max_alpha_line.setText(f"{max_alpha_mrad_from_px(px, dk, wavelength):g}")

    def values(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        cfg.calibration_free = self.calibration_free_cb.isChecked()
        cfg.max_alpha_px = self._optional_float(self.max_alpha_px_line, "Max alpha (px)")
        cfg.upscale = self._float(self.upscale_line, "Upscale")
        cfg.upscale_method = self.upscale_method_combo.currentText()
        cfg.pad_width = self._optional_int(self.pad_width_line, "Pad width")
        if cfg.max_alpha_px is not None:
            wavelength_for_conversion = resolved_wavelength_angstrom(cfg.wavelength_angstrom)
            cfg.max_alpha_mrad = max_alpha_mrad_from_px(
                cfg.max_alpha_px, cfg.dk_inv_angstrom, wavelength_for_conversion
            )
        else:
            cfg.max_alpha_mrad = None
        cfg.validate_upscale_settings()
        return cfg

    def accept(self) -> None:
        try:
            self.config = self.values()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid fast-acbf settings", str(exc))
            return
        super().accept()
