"""Qt dialogs for the fast-acbf py4D-browser plugin."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QHeaderView,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.scalebar import ScaleBar

from .config import FastAcbfConfig, labels_for_order


class ConfigurationDialog(QDialog):
    request_calibration = pyqtSignal()

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf Configuration")
        self.resize(560, 680)
        self.config = config.copy()
        self.aberration_inputs: dict[str, QLineEdit] = {}
        self._build_ui()
        self.set_from_config(self.config)

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
        self.cache_combo.addItems(["lazy", "full"])
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
        run_form.addRow("Cache mode", self.cache_combo)
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
        self.optics_form = QFormLayout()
        optics_outer.addLayout(self.optics_form)
        zero_all_btn = QPushButton("Zero All")
        zero_all_btn.clicked.connect(self._zero_all_aberrations)
        optics_outer.addWidget(zero_all_btn)
        optics_outer.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(optics_wrap)
        tabs.addTab(scroll, "Optics")
        self.max_order_spin.valueChanged.connect(self._rebuild_aberration_fields)

        orient_tab = QWidget()
        orient_form = QFormLayout(orient_tab)
        self.rotation_line = self._float_line()
        self.flipud_cb = QCheckBox()
        self.fliplr_cb = QCheckBox()
        self.transpose_cb = QCheckBox()
        orient_form.addRow("Scan rotation [deg]", self.rotation_line)
        orient_form.addRow("Flip up/down", self.flipud_cb)
        orient_form.addRow("Flip left/right", self.fliplr_cb)
        orient_form.addRow("Transpose detector x/y", self.transpose_cb)
        reset_orientation_btn = QPushButton("Reset Orientation")
        reset_orientation_btn.clicked.connect(self._reset_orientation)
        orient_form.addRow("", reset_orientation_btn)
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

    def _clear_form(self, form: QFormLayout) -> None:
        while form.count():
            item = form.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            layout = item.layout()
            if layout is not None:
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget() is not None:
                        child.widget().deleteLater()

    def _rebuild_aberration_fields(self) -> None:
        old_values = self._read_aberrations(quiet=True)
        merged_values = dict(self.config.aberrations)
        merged_values.update(old_values)
        self._clear_form(self.optics_form)
        self.aberration_inputs = {}
        for label in labels_for_order(self.max_order_spin.value()):
            line = self._float_line()
            line.setText(f"{float(merged_values.get(label, 0.0)):g}")
            self.aberration_inputs[label] = line
            self.optics_form.addRow(f"{label} [A]", line)

    def set_from_config(self, config: FastAcbfConfig) -> None:
        self.mode_combo.setCurrentText(config.mode)
        self.acbf_combo.setCurrentText(config.acbf_algorithm)
        self.output_combo.setCurrentText(config.output_target)
        self.frame_combo.setCurrentText(config.output_frame)
        self.device_combo.setCurrentText(config.device)
        self.cache_combo.setCurrentText(config.cache_mode)
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
        self._rebuild_aberration_fields()
        for label, value in config.aberrations.items():
            if label in self.aberration_inputs:
                self.aberration_inputs[label].setText(f"{float(value):g}")
        self.rotation_line.setText(f"{config.rotation_deg:g}")
        self.flipud_cb.setChecked(bool(config.flipud))
        self.fliplr_cb.setChecked(bool(config.fliplr))
        self.transpose_cb.setChecked(bool(config.transpose))
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

    def _read_aberrations(self, quiet: bool = False) -> dict[str, float]:
        values = {}
        for label, line in self.aberration_inputs.items():
            text = line.text().strip()
            try:
                values[label] = float(text or 0.0)
            except ValueError:
                if quiet:
                    values[label] = 0.0
                else:
                    raise ValueError(f"{label} must be numeric.")
        return values

    def values(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        cfg.mode = self.mode_combo.currentText()
        cfg.acbf_algorithm = self.acbf_combo.currentText()
        cfg.output_target = self.output_combo.currentText()
        cfg.output_frame = self.frame_combo.currentText()
        cfg.device = self.device_combo.currentText()
        cfg.cache_mode = self.cache_combo.currentText()
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
        cfg.aberrations = self._read_aberrations()
        cfg.rotation_deg = self._float(self.rotation_line, "Scan rotation")
        cfg.flipud = self.flipud_cb.isChecked()
        cfg.fliplr = self.fliplr_cb.isChecked()
        cfg.transpose = self.transpose_cb.isChecked()
        cfg.refinement_mode = self.refine_mode_combo.currentText()
        cfg.metric = self.metric_combo.currentText()
        cfg.defocus_points = int(self.defocus_points_spin.value())
        cfg.rotation_points = int(self.rotation_points_spin.value())
        cfg.aberration_lr = self._float(self.lr_line, "Aberration learning rate")
        cfg.aberration_iters = int(self.iters_spin.value())
        return cfg

    def _zero_all_aberrations(self) -> None:
        for line in self.aberration_inputs.values():
            line.setText("0")

    def _reset_orientation(self) -> None:
        self.rotation_line.setText("0")
        self.flipud_cb.setChecked(False)
        self.fliplr_cb.setChecked(False)
        self.transpose_cb.setChecked(False)

    def accept(self) -> None:
        try:
            self.config = self.values()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid fast-acbf settings", str(exc))
            return
        super().accept()


class FastAcbfDashboard(QDialog):
    run_requested = pyqtSignal(str)
    config_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    config_changed = pyqtSignal(object)

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf: Interactive Dashboard")
        self.resize(1200, 800)
        self.config = config.copy()
        self.aberration_inputs: dict[str, QLineEdit] = {}
        self.image_scale_bar = None
        self.probe_scale_bar = None
        self._build_ui()
        self.set_config(config)

    def _build_ui(self) -> None:
        try:
            import pyqtgraph as pg
        except Exception:
            pg = None
        self.pg = pg

        main = QHBoxLayout(self)
        left = QVBoxLayout()
        main.addLayout(left, stretch=2)

        calib = QGroupBox("Global Calibration")
        calib_layout = QVBoxLayout(calib)
        self.calib_label = QLabel()
        calib_layout.addWidget(self.calib_label)
        btns = QHBoxLayout()
        edit_calib = QPushButton("Edit Calibration...")
        edit_calib.clicked.connect(self.calibration_requested.emit)
        open_config = QPushButton("Configuration")
        open_config.clicked.connect(self.config_requested.emit)
        btns.addWidget(edit_calib)
        btns.addWidget(open_config)
        calib_layout.addLayout(btns)
        left.addWidget(calib)

        tabs = QTabWidget()
        left.addWidget(tabs)
        optics = QWidget()
        optics_layout = QVBoxLayout(optics)
        self.optics_form = QFormLayout()
        optics_layout.addLayout(self.optics_form)
        zero_all_btn = QPushButton("Zero All")
        zero_all_btn.clicked.connect(self._zero_all_aberrations)
        optics_layout.addWidget(zero_all_btn)
        optics_layout.addStretch()
        tabs.addTab(optics, "Optics")
        orient = QWidget()
        orient_form = QFormLayout(orient)
        self.rotation_line = QLineEdit()
        self.rotation_line.setValidator(QDoubleValidator())
        self.flipud_cb = QCheckBox()
        self.fliplr_cb = QCheckBox()
        self.transpose_cb = QCheckBox()
        orient_form.addRow("Scan rotation [deg]", self.rotation_line)
        orient_form.addRow("Flip up/down", self.flipud_cb)
        orient_form.addRow("Flip left/right", self.fliplr_cb)
        orient_form.addRow("Transpose", self.transpose_cb)
        reset_orientation_btn = QPushButton("Reset Orientation")
        reset_orientation_btn.clicked.connect(self._reset_orientation)
        orient_form.addRow("", reset_orientation_btn)
        tabs.addTab(orient, "Orientation")

        self.apply_btn = QPushButton("Update Preview")
        self.apply_btn.clicked.connect(self._apply_overrides)
        left.addWidget(self.apply_btn)

        actions = QGroupBox("Automated Refinement")
        action_layout = QVBoxLayout(actions)
        self.auto_btn = QPushButton("Refine All Params")
        self.auto_btn.clicked.connect(lambda: self.run_requested.emit("auto_tune"))
        action_layout.addWidget(self.auto_btn)
        sub = QGridLayout()
        self.flips_btn = QPushButton("Refine Flips")
        self.flips_btn.clicked.connect(lambda: self.run_requested.emit("refine_flips"))
        self.rotation_btn = QPushButton("Refine Scan Rotation")
        self.rotation_btn.clicked.connect(lambda: self.run_requested.emit("refine_scan_rotation"))
        self.defocus_btn = QPushButton("Refine Defocus")
        self.defocus_btn.clicked.connect(lambda: self.run_requested.emit("refine_defocus"))
        self.ad_btn = QPushButton("Refine Aberrations")
        self.ad_btn.clicked.connect(lambda: self.run_requested.emit("refine_aberrations"))
        sub.addWidget(self.flips_btn, 0, 0)
        sub.addWidget(self.rotation_btn, 0, 1)
        sub.addWidget(self.defocus_btn, 1, 0)
        sub.addWidget(self.ad_btn, 1, 1)
        action_layout.addLayout(sub)
        left.addWidget(actions)

        left.addWidget(QLabel("Refinement History:"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Step", "C10", "C12a", "C12b", "Rot", "Metric"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMaximumHeight(170)
        left.addWidget(self.table)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        close_btn = QPushButton("Return to py4D-browser")
        close_btn.clicked.connect(self.close)
        left.addWidget(close_btn)

        right = QVBoxLayout()
        main.addLayout(right, stretch=5)
        controls = QFormLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["tcBF", "acBF"])
        self.mode_combo.currentTextChanged.connect(self._mode_changed)
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["scan", "detector"])
        self.frame_combo.currentTextChanged.connect(self._output_frame_changed)
        controls.addRow("Display mode", self.mode_combo)
        controls.addRow("Output frame", self.frame_combo)
        right.addLayout(controls)

        if pg is not None:
            self.image_view = pg.ImageView()
            self.probe_view = pg.ImageView()
            self.image_view.ui.roiBtn.hide()
            self.image_view.ui.menuBtn.hide()
            self.probe_view.ui.roiBtn.hide()
            self.probe_view.ui.menuBtn.hide()
            self.image_scale_bar = ScaleBar(pixel_size=1, units="A", width=10)
            self.image_scale_bar.setParentItem(self.image_view.getView())
            self.image_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))
            self.probe_scale_bar = ScaleBar(pixel_size=1, units="A", width=10)
            self.probe_scale_bar.setParentItem(self.probe_view.getView())
            self.probe_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))
            grid = QGridLayout()
            grid.addWidget(QLabel("Reconstruction"), 0, 0)
            grid.addWidget(self.image_view, 1, 0)
            grid.addWidget(QLabel("Probe amplitude"), 2, 0)
            grid.addWidget(self.probe_view, 3, 0)
            right.addLayout(grid)
        else:
            self.image_view = self.probe_view = None
            right.addWidget(QLabel("pyqtgraph is required for dashboard previews."))

    def _mode_changed(self, mode: str) -> None:
        self.config.mode = mode
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit("apply")

    def _output_frame_changed(self, frame: str) -> None:
        self.config.output_frame = frame
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit("apply")

    def _apply_overrides(self) -> None:
        try:
            self.config = self.config_with_overrides()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid overrides", str(exc))
            return
        self.run_requested.emit("apply")

    def config_with_overrides(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        cfg.rotation_deg = float(self.rotation_line.text() or 0.0)
        cfg.flipud = self.flipud_cb.isChecked()
        cfg.fliplr = self.fliplr_cb.isChecked()
        cfg.transpose = self.transpose_cb.isChecked()
        for label, line in self.aberration_inputs.items():
            cfg.aberrations[label] = float(line.text() or 0.0)
        return cfg

    def _rebuild_aberrations(self) -> None:
        while self.optics_form.count():
            item = self.optics_form.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.aberration_inputs = {}
        for label in labels_for_order(self.config.max_order):
            line = QLineEdit(f"{float(self.config.aberrations.get(label, 0.0)):g}")
            line.setValidator(QDoubleValidator())
            self.aberration_inputs[label] = line
            self.optics_form.addRow(f"{label} [A]", line)

    def set_config(self, config: FastAcbfConfig) -> None:
        self.config = config.copy()
        previous = self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(config.mode)
        self.mode_combo.blockSignals(previous)
        previous = self.frame_combo.blockSignals(True)
        self.frame_combo.setCurrentText(config.output_frame)
        self.frame_combo.blockSignals(previous)
        self.rotation_line.setText(f"{config.rotation_deg:g}")
        self.flipud_cb.setChecked(bool(config.flipud))
        self.fliplr_cb.setChecked(bool(config.fliplr))
        self.transpose_cb.setChecked(bool(config.transpose))
        self.calib_label.setText(
            f"kV: {config.voltage_kv:g}    step: {config.scan_step_angstrom:g} A    "
            f"dk: {config.dk_inv_angstrom:g} 1/A    alpha: {config.max_alpha_mrad:g} mrad"
        )
        self._update_scale_bars(config)
        self._rebuild_aberrations()

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def set_result(self, result: dict) -> None:
        image = result.get("image")
        probe = result.get("probe")
        if self.pg is not None and image is not None:
            self.image_view.setImage(image.T, autoLevels=True, autoRange=True)
        if self.pg is not None and probe is not None:
            self.probe_view.setImage(probe.T, autoLevels=True, autoRange=True)
        self.add_history(result)

    def add_history(self, result: dict) -> None:
        cfg: FastAcbfConfig = result.get("config", self.config)
        row = self.table.rowCount()
        self.table.insertRow(row)
        step_labels = {
            "manual": "manual",
            "run": "Preview",
            "auto_tune": "Refine All Params",
            "refine_defocus": "Refine Defocus",
            "refine_scan_rotation": "Refine Scan Rotation",
            "refine_flips": "Refine Flips",
            "refine_orientation": "Refine Orientation",
            "refine_aberrations": "Refine Aberrations",
        }
        values = [
            step_labels.get(result.get("command", "run"), result.get("command", "run")),
            cfg.aberrations.get("C10", 0.0),
            cfg.aberrations.get("C12a", 0.0),
            cfg.aberrations.get("C12b", 0.0),
            cfg.rotation_deg,
            result.get("metric_text", ""),
        ]
        for col, value in enumerate(values):
            if isinstance(value, float):
                text = f"{value:.5g}"
            else:
                text = str(value)
            self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.selectRow(row)

    def _zero_all_aberrations(self) -> None:
        for line in self.aberration_inputs.values():
            line.setText("0")
        for label in self.config.aberrations:
            self.config.aberrations[label] = 0.0

    def _reset_orientation(self) -> None:
        self.rotation_line.setText("0")
        self.flipud_cb.setChecked(False)
        self.fliplr_cb.setChecked(False)
        self.transpose_cb.setChecked(False)
        self.config.rotation_deg = 0.0
        self.config.flipud = False
        self.config.fliplr = False
        self.config.transpose = False

    def _update_scale_bars(self, config: FastAcbfConfig) -> None:
        if self.image_scale_bar is not None:
            self.image_scale_bar.pixel_size = float(config.scan_step_angstrom)
            self.image_scale_bar.units = "A"
            self.image_scale_bar.updateBar()
        if self.probe_scale_bar is not None:
            self.probe_scale_bar.pixel_size = float(config.scan_step_angstrom)
            self.probe_scale_bar.units = "A"
            self.probe_scale_bar.updateBar()


class LiveDemoDialog(QDialog):
    start_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal()
    config_requested = pyqtSignal()

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf: Live Demo")
        self.resize(1000, 720)
        self.config = config.copy()
        self._live_active = False
        self._seen_frame = False
        self.image_scale_bar = None
        self._build_ui()
        self.set_config(config)

    def _build_ui(self) -> None:
        try:
            import pyqtgraph as pg
        except Exception:
            pg = None
        self.pg = pg

        main = QHBoxLayout(self)
        left = QVBoxLayout()
        main.addLayout(left, stretch=2)

        calib = QGroupBox("Configuration")
        calib_layout = QVBoxLayout(calib)
        self.calib_label = QLabel()
        self.calib_label.setWordWrap(True)
        calib_layout.addWidget(self.calib_label)
        config_btn = QPushButton("Configuration")
        config_btn.clicked.connect(self.config_requested.emit)
        calib_layout.addWidget(config_btn)
        left.addWidget(calib)

        controls = QGroupBox("Live Demo")
        form = QFormLayout(controls)
        self.source_combo = QComboBox()
        self.source_combo.addItem("current datacube (mock streamer)")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["tcBF", "acBF"])
        self.pinned_cb = QCheckBox("Use CUDA pinned source buffer")
        self.pinned_cb.setChecked(True)
        self.rotation_sweep_spin = QDoubleSpinBox()
        self.rotation_sweep_spin.setRange(-360.0, 360.0)
        self.rotation_sweep_spin.setDecimals(3)
        self.rotation_sweep_spin.setSingleStep(0.1)
        self.rotation_sweep_spin.setValue(0.5)
        self.defocus_sweep_spin = QDoubleSpinBox()
        self.defocus_sweep_spin.setRange(0.0, 100000.0)
        self.defocus_sweep_spin.setDecimals(3)
        self.defocus_sweep_spin.setSingleStep(10.0)
        self.defocus_period_spin = QSpinBox()
        self.defocus_period_spin.setRange(1, 100000)
        self.defocus_period_spin.setValue(120)
        self.display_noise_spin = QDoubleSpinBox()
        self.display_noise_spin.setRange(0.0, 1000.0)
        self.display_noise_spin.setDecimals(3)
        self.display_noise_spin.setSingleStep(1.0)
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(0, 1_000_000)
        self.drift_y_spin = QDoubleSpinBox()
        self.drift_y_spin.setRange(-1000.0, 1000.0)
        self.drift_y_spin.setDecimals(3)
        self.drift_y_spin.setSingleStep(0.05)
        self.drift_x_spin = QDoubleSpinBox()
        self.drift_x_spin.setRange(-1000.0, 1000.0)
        self.drift_x_spin.setDecimals(3)
        self.drift_x_spin.setSingleStep(0.05)
        self.toggle_btn = QPushButton("Start Live")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self._toggle_clicked)
        self.status_label = QLabel("idle")
        self.status_label.setWordWrap(True)

        form.addRow("Source", self.source_combo)
        form.addRow("Display mode", self.mode_combo)
        form.addRow("", self.pinned_cb)
        form.addRow("Rotation sweep [deg/frame]", self.rotation_sweep_spin)
        form.addRow("Defocus sweep [A amplitude]", self.defocus_sweep_spin)
        form.addRow("Defocus period [frames]", self.defocus_period_spin)
        form.addRow("Display Gaussian noise [% image std]", self.display_noise_spin)
        form.addRow("Frames", self.frames_spin)
        form.addRow("Display drift y [scan px/frame]", self.drift_y_spin)
        form.addRow("Display drift x [scan px/frame]", self.drift_x_spin)
        form.addRow("", self.toggle_btn)
        form.addRow("Status", self.status_label)
        left.addWidget(controls)
        left.addStretch()

        right = QVBoxLayout()
        main.addLayout(right, stretch=5)
        right.addWidget(QLabel("Live Reconstruction"))
        if pg is not None:
            self.image_view = pg.ImageView()
            self.image_view.ui.roiBtn.hide()
            self.image_view.ui.menuBtn.hide()
            self.image_scale_bar = ScaleBar(pixel_size=1, units="A", width=10)
            self.image_scale_bar.setParentItem(self.image_view.getView())
            self.image_scale_bar.anchor((1, 1), (1, 1), offset=(-40, -40))
            right.addWidget(self.image_view)
        else:
            self.image_view = None
            right.addWidget(QLabel("pyqtgraph is required for live previews."))

    def set_config(self, config: FastAcbfConfig) -> None:
        self.config = config.copy()
        previous = self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(config.mode)
        self.mode_combo.blockSignals(previous)
        self.calib_label.setText(
            f"kV: {config.voltage_kv:g}    step: {config.scan_step_angstrom:g} A    "
            f"dk: {config.dk_inv_angstrom:g} 1/A    alpha: {config.max_alpha_mrad:g} mrad"
        )
        if self.image_scale_bar is not None:
            self.image_scale_bar.pixel_size = float(config.scan_step_angstrom)
            self.image_scale_bar.units = "A"
            self.image_scale_bar.updateBar()

    def live_options(self) -> dict:
        n_frames = int(self.frames_spin.value())
        return {
            "source": self.source_combo.currentText(),
            "mode": self.mode_combo.currentText(),
            "use_pinned_source": self.pinned_cb.isChecked(),
            "rotation_sweep_deg_per_frame": float(self.rotation_sweep_spin.value()),
            "defocus_sweep_angstrom": float(self.defocus_sweep_spin.value()),
            "defocus_sweep_period_frames": int(self.defocus_period_spin.value()),
            "display_noise_sigma_pct": float(self.display_noise_spin.value()),
            "n_frames": n_frames if n_frames > 0 else None,
            "drift_y_per_frame": float(self.drift_y_spin.value()),
            "drift_x_per_frame": float(self.drift_x_spin.value()),
        }

    def _toggle_clicked(self, checked: bool) -> None:
        if checked:
            self.start_requested.emit(self.live_options())
        else:
            self.stop_requested.emit()

    def set_live_active(self, active: bool, status: str | None = None) -> None:
        self._live_active = bool(active)
        if not self._live_active:
            self._seen_frame = False
        previous = self.toggle_btn.blockSignals(True)
        self.toggle_btn.setChecked(self._live_active)
        self.toggle_btn.setText("Stop Live" if self._live_active else "Start Live")
        self.toggle_btn.blockSignals(previous)
        for widget in (
            self.source_combo,
            self.mode_combo,
            self.pinned_cb,
            self.rotation_sweep_spin,
            self.defocus_sweep_spin,
            self.defocus_period_spin,
            self.display_noise_spin,
            self.frames_spin,
            self.drift_y_spin,
            self.drift_x_spin,
        ):
            widget.setEnabled(not self._live_active)
        if status is not None:
            self.status_label.setText(status)

    def on_live_frame(self, image, metrics: dict) -> None:
        if self.pg is not None and self.image_view is not None:
            first_frame = not self._seen_frame
            self.image_view.setImage(
                image.T,
                autoLevels=first_frame,
                autoRange=first_frame,
                autoHistogramRange=first_frame,
            )
            self._seen_frame = True
        fps = float(metrics.get("fps", 0.0))
        latency_ms = float(metrics.get("latency_s", 0.0)) * 1000.0
        device = str(metrics.get("device") or "?")
        mask_path = str(metrics.get("mask_path") or "?")
        mode = str(metrics.get("mode") or self.mode_combo.currentText())
        bf_pixels = metrics.get("bf_pixels")
        alpha = metrics.get("max_alpha_mrad")
        details = f"FPS {fps:.1f}   latency {latency_ms:.1f} ms   {mode} {device} {mask_path}"
        if bf_pixels is not None:
            details += f"   BF {int(bf_pixels)} px"
        if alpha is not None:
            details += f"   alpha {float(alpha):.3g} mrad"
        noise = float(metrics.get("display_noise_sigma_pct") or 0.0)
        if noise > 0:
            details += f"   display noise {noise:.3g}%"
        stage_text = self._format_stage_times(metrics.get("stage_times") or {})
        if stage_text:
            details += "\n" + stage_text
        self.status_label.setText(details)

    def _format_stage_times(self, stage_times: dict) -> str:
        if not stage_times:
            return ""
        labels = [
            ("prep", "prep"),
            ("pinned_h2d", "transfer"),
            ("device_bf_gather", "gather"),
            ("build_image_fft", "fft"),
            ("apply_metadata_or_update_dataset", "update total"),
            ("get_reconstructed_image", "compute"),
            ("tensor_to_numpy", "numpy"),
        ]
        parts = []
        for key, label in labels:
            if key in stage_times:
                parts.append(f"{label} {float(stage_times[key]) * 1000.0:.1f} ms")
        return " | ".join(parts)

    def closeEvent(self, event) -> None:
        if self._live_active:
            self.stop_requested.emit()
        super().closeEvent(event)
