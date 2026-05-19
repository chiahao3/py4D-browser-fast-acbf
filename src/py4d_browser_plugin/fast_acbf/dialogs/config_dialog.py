"""Modal configuration dialog for the fast-acbf plugin."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import FastAcbfConfig
from ._widgets import AberrationForm, OrientationForm


class ConfigurationDialog(QDialog):
    request_calibration = pyqtSignal()

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf Configuration")
        self.resize(560, 680)
        self.config = config.copy()
        self._build_ui()
        self.set_from_config(self.config)

    # ------------------------------------------------------------------ helpers

    def _line(self, validator=None) -> QLineEdit:
        line = QLineEdit()
        if validator is not None:
            line.setValidator(validator)
        return line

    def _float_line(self) -> QLineEdit:
        return self._line(QDoubleValidator())

    def _int_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    @property
    def aberration_inputs(self) -> dict[str, QLineEdit]:
        """Back-compat accessor used by older tests/callers."""
        return self.aberration_form.aberration_inputs

    @property
    def flipud_cb(self) -> QCheckBox:
        return self.orientation_form.flipud_cb

    @property
    def fliplr_cb(self) -> QCheckBox:
        return self.orientation_form.fliplr_cb

    @property
    def transpose_cb(self) -> QCheckBox:
        return self.orientation_form.transpose_cb

    @property
    def rotation_line(self) -> QLineEdit:
        return self.orientation_form.rotation_line

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        run_tab = QWidget()
        run_form = QFormLayout(run_tab)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["tcBF", "acBF"])
        self.acbf_combo = QComboBox()
        self.acbf_combo.addItems(["phase_only", "complex_inversion"])
        self.output_combo = QComboBox()
        self.output_combo.addItems(["virtual_image", "result_image"])
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["scan", "detector"])
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "mps", "cpu"])
        self.cache_combo = QComboBox()
        self.cache_combo.addItems(["balanced", "speed", "memory"])
        self.basis_combo = QComboBox()
        self.basis_combo.addItems(["on_the_fly", "precompute"])
        self.chunk_spin = self._int_spin(1, 4096, 64)
        self.eps_line = self._float_line()
        self.rolloff_line = self._float_line()
        self.regularization_line = self._float_line()
        self.support_line = self._float_line()
        run_form.addRow("Display mode", self.mode_combo)
        run_form.addRow("acBF algorithm", self.acbf_combo)
        run_form.addRow("Output target", self.output_combo)
        run_form.addRow("Output frame", self.frame_combo)
        run_form.addRow("Device", self.device_combo)
        run_form.addRow("Pipeline", self.cache_combo)
        run_form.addRow("Basis mode", self.basis_combo)
        run_form.addRow("Chunk size", self.chunk_spin)
        run_form.addRow("Phase eps", self.eps_line)
        run_form.addRow("Rolloff", self.rolloff_line)
        run_form.addRow("Complex regularization", self.regularization_line)
        run_form.addRow("Support threshold", self.support_line)
        tabs.addTab(run_tab, "Run")

        physics_tab = QWidget()
        physics_form = QFormLayout(physics_tab)
        self.use_calibration_cb = QCheckBox("Read scan step, dk, voltage from py4D calibration")
        self.use_detector_cb = QCheckBox("Use current circular detector radius for max alpha")
        self.max_alpha_line = self._float_line()
        self.scan_step_line = self._float_line()
        self.dk_line = self._float_line()
        self.voltage_line = self._float_line()
        self.wavelength_line = self._float_line()
        self.max_order_spin = self._int_spin(1, 4, 2)
        physics_form.addRow("", self.use_calibration_cb)
        physics_form.addRow("", self.use_detector_cb)
        physics_form.addRow("Max alpha [mrad]", self.max_alpha_line)
        physics_form.addRow("Scan step [A]", self.scan_step_line)
        physics_form.addRow("dk [1/A]", self.dk_line)
        physics_form.addRow("Voltage [kV]", self.voltage_line)
        physics_form.addRow("Wavelength [A]", self.wavelength_line)
        physics_form.addRow("Max aberration order", self.max_order_spin)
        calib_btn = QPushButton("Edit py4D Calibration...")
        calib_btn.clicked.connect(self.request_calibration.emit)
        physics_form.addRow("", calib_btn)
        tabs.addTab(physics_tab, "Physics")

        optics_wrap = QWidget()
        optics_outer = QVBoxLayout(optics_wrap)
        optics_outer.setContentsMargins(0, 0, 0, 0)
        self.aberration_form = AberrationForm()
        optics_outer.addWidget(self.aberration_form)
        self.aberration_form.add_zero_all_button()
        optics_outer.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(optics_wrap)
        tabs.addTab(scroll, "Optics")
        self.max_order_spin.valueChanged.connect(self.aberration_form.set_max_order)

        orient_tab = QWidget()
        orient_layout = QVBoxLayout(orient_tab)
        self.orientation_form = OrientationForm()
        orient_layout.addWidget(self.orientation_form)
        self.orientation_form.add_reset_button()
        orient_layout.addStretch()
        tabs.addTab(orient_tab, "Orientation")

        refine_tab = QWidget()
        refine_form = QFormLayout(refine_tab)
        self.refine_mode_combo = QComboBox()
        self.refine_mode_combo.addItems(["tcBF", "acBF"])
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["normalized_std", "laplacian", "sobel"])
        self.defocus_points_spin = self._int_spin(3, 101, 7)
        self.rotation_points_spin = self._int_spin(3, 360, 18)
        self.lr_line = self._float_line()
        self.iters_spin = self._int_spin(1, 5000, 20)
        refine_form.addRow("Refinement mode", self.refine_mode_combo)
        refine_form.addRow("Quality metric", self.metric_combo)
        refine_form.addRow("Defocus points", self.defocus_points_spin)
        refine_form.addRow("Rotation points", self.rotation_points_spin)
        refine_form.addRow("Aberration learning rate", self.lr_line)
        refine_form.addRow("Aberration iterations", self.iters_spin)
        tabs.addTab(refine_tab, "Refinement")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ config I/O

    def set_from_config(self, config: FastAcbfConfig) -> None:
        self.mode_combo.setCurrentText(config.mode)
        self.acbf_combo.setCurrentText(config.acbf_algorithm)
        self.output_combo.setCurrentText(config.output_target)
        self.frame_combo.setCurrentText(config.output_frame)
        self.device_combo.setCurrentText(config.device)
        self.cache_combo.setCurrentText(config.pipeline)
        self.basis_combo.setCurrentText(config.basis_mode)
        self.chunk_spin.setValue(int(config.chunk_size))
        self.eps_line.setText(f"{config.eps:g}")
        self.rolloff_line.setText(f"{config.rolloff:g}")
        self.regularization_line.setText(f"{config.regularization:g}")
        self.support_line.setText(f"{config.support_threshold:g}")
        self.use_calibration_cb.setChecked(bool(config.use_calibration))
        self.use_detector_cb.setChecked(bool(config.use_detector_alpha))
        self.max_alpha_line.setText(f"{config.max_alpha_mrad:g}")
        self.scan_step_line.setText(f"{config.scan_step_angstrom:g}")
        self.dk_line.setText(f"{config.dk_inv_angstrom:g}")
        self.voltage_line.setText(f"{config.voltage_kv:g}")
        self.wavelength_line.setText(f"{config.wavelength_angstrom:g}")
        self.max_order_spin.setValue(int(config.max_order))
        self.aberration_form.set_max_order(int(config.max_order))
        self.aberration_form.set_values(config.aberrations)
        self.orientation_form.set_values(
            rotation_deg=float(config.rotation_deg),
            flipud=bool(config.flipud),
            fliplr=bool(config.fliplr),
            transpose=bool(config.transpose),
        )
        self.refine_mode_combo.setCurrentText(config.refinement_mode)
        self.metric_combo.setCurrentText(config.metric)
        self.defocus_points_spin.setValue(int(config.defocus_points))
        self.rotation_points_spin.setValue(int(config.rotation_points))
        self.lr_line.setText(f"{config.aberration_lr:g}")
        self.iters_spin.setValue(int(config.aberration_iters))

    def _float(self, line: QLineEdit, label: str) -> float:
        text = line.text().strip()
        if text == "":
            raise ValueError(f"{label} is required.")
        return float(text)

    def values(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        cfg.mode = self.mode_combo.currentText()
        cfg.acbf_algorithm = self.acbf_combo.currentText()
        cfg.output_target = self.output_combo.currentText()
        cfg.output_frame = self.frame_combo.currentText()
        cfg.device = self.device_combo.currentText()
        cfg.pipeline = self.cache_combo.currentText()
        cfg.basis_mode = self.basis_combo.currentText()
        cfg.chunk_size = int(self.chunk_spin.value())
        cfg.eps = self._float(self.eps_line, "Phase eps")
        cfg.rolloff = self._float(self.rolloff_line, "Rolloff")
        cfg.regularization = self._float(self.regularization_line, "Complex regularization")
        cfg.support_threshold = self._float(self.support_line, "Support threshold")
        cfg.use_calibration = self.use_calibration_cb.isChecked()
        cfg.use_detector_alpha = self.use_detector_cb.isChecked()
        cfg.max_alpha_mrad = self._float(self.max_alpha_line, "Max alpha")
        cfg.scan_step_angstrom = self._float(self.scan_step_line, "Scan step")
        cfg.dk_inv_angstrom = self._float(self.dk_line, "dk")
        cfg.voltage_kv = self._float(self.voltage_line, "Voltage")
        cfg.wavelength_angstrom = self._float(self.wavelength_line, "Wavelength")
        cfg.max_order = int(self.max_order_spin.value())
        cfg.aberrations = self.aberration_form.read_values()
        orient = self.orientation_form.read_values()
        text = self.orientation_form.rotation_line.text().strip()
        if text == "":
            raise ValueError("Scan rotation is required.")
        cfg.rotation_deg = float(orient["rotation_deg"])
        cfg.flipud = bool(orient["flipud"])
        cfg.fliplr = bool(orient["fliplr"])
        cfg.transpose = bool(orient["transpose"])
        cfg.refinement_mode = self.refine_mode_combo.currentText()
        cfg.metric = self.metric_combo.currentText()
        cfg.defocus_points = int(self.defocus_points_spin.value())
        cfg.rotation_points = int(self.rotation_points_spin.value())
        cfg.aberration_lr = self._float(self.lr_line, "Aberration learning rate")
        cfg.aberration_iters = int(self.iters_spin.value())
        return cfg

    def accept(self) -> None:
        try:
            self.config = self.values()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid fast-acbf settings", str(exc))
            return
        super().accept()
