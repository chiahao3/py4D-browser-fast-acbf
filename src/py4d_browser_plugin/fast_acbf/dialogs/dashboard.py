"""Advanced dashboard dialog for fast-acbf."""

from __future__ import annotations

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.scalebar import ScaleBar

from ..config import FastAcbfConfig
from ..solver_job import (
    AutoTuneJob,
    PreviewJob,
    RefineAberrationsJob,
    RefineDefocusJob,
    RefineFlipsJob,
    RefineScanRotationJob,
)
from ._widgets import AberrationForm, OrientationForm


class FastAcbfDashboard(QDialog):
    run_requested = pyqtSignal(object)
    config_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    config_changed = pyqtSignal(object)

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf: Advanced Dashboard")
        self.resize(1200, 800)
        self.config = config.copy()
        self.image_scale_bar = None
        self.probe_scale_bar = None
        self._build_ui()
        self.set_config(config)
        QTimer.singleShot(0, self._focus_global_calibration)

    # ---- back-compat accessors (existing tests/callers look these up directly)

    @property
    def aberration_inputs(self) -> dict[str, QLineEdit]:
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

    # ---- build

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
        self.edit_calib_btn = QPushButton("Edit Calibration...")
        self.edit_calib_btn.clicked.connect(self.calibration_requested.emit)
        open_config = QPushButton("Configuration")
        open_config.clicked.connect(self.config_requested.emit)
        btns.addWidget(self.edit_calib_btn)
        btns.addWidget(open_config)
        calib_layout.addLayout(btns)
        left.addWidget(calib)

        from PyQt5.QtWidgets import QTabWidget

        tabs = QTabWidget()
        self.parameter_tabs = tabs
        left.addWidget(tabs)

        aberrations_tab = QWidget()
        aberrations_layout = QVBoxLayout(aberrations_tab)
        aberrations_layout.setContentsMargins(0, 0, 0, 0)
        self.aberration_form = AberrationForm()
        aberrations_layout.addWidget(self.aberration_form)
        self.aberration_form.add_zero_all_button()
        aberrations_layout.addStretch()
        tabs.addTab(aberrations_tab, "Aberrations")

        orient = QWidget()
        orient_layout = QVBoxLayout(orient)
        orient_layout.setContentsMargins(0, 0, 0, 0)
        self.orientation_form = OrientationForm()
        orient_layout.addWidget(self.orientation_form)
        self.orientation_form.add_reset_button()
        orient_layout.addStretch()
        tabs.addTab(orient, "Orientation")

        self.apply_btn = QPushButton("Update Preview")
        self.apply_btn.setAutoDefault(True)
        self.apply_btn.setDefault(True)
        self.apply_btn.setMinimumHeight(34)
        font = self.apply_btn.font()
        font.setBold(True)
        self.apply_btn.setFont(font)
        self.apply_btn.setStyleSheet(
            "QPushButton { padding: 6px 10px; }"
            "QPushButton:default { border: 2px solid #2a82da; }"
        )
        self.apply_btn.clicked.connect(self._apply_overrides)
        left.addWidget(self.apply_btn)
        self.parameter_tabs.currentChanged.connect(self._schedule_update_preview_focus)

        actions = QGroupBox("Automated Refinement")
        action_layout = QVBoxLayout(actions)
        self.auto_btn = QPushButton("Refine All Params")
        self.auto_btn.clicked.connect(lambda: self.run_requested.emit(AutoTuneJob()))
        action_layout.addWidget(self.auto_btn)
        sub = QGridLayout()
        self.flips_btn = QPushButton("Refine Flips")
        self.flips_btn.clicked.connect(lambda: self.run_requested.emit(RefineFlipsJob()))
        self.rotation_btn = QPushButton("Refine Scan Rotation")
        self.rotation_btn.clicked.connect(lambda: self.run_requested.emit(RefineScanRotationJob()))
        self.defocus_btn = QPushButton("Refine Defocus")
        self.defocus_btn.clicked.connect(lambda: self.run_requested.emit(RefineDefocusJob()))
        self.ad_btn = QPushButton("Refine Aberrations")
        self.ad_btn.clicked.connect(lambda: self.run_requested.emit(RefineAberrationsJob()))
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

    # ---- behavior

    def _mode_changed(self, mode: str) -> None:
        self.config.mode = mode
        messages = self.config.coerce_upscale_method_for_mode()
        if messages:
            self.set_status(" ".join(messages))
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit(PreviewJob())

    def _output_frame_changed(self, frame: str) -> None:
        self.config.output_frame = frame
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit(PreviewJob())

    def _schedule_update_preview_focus(self, *_args) -> None:
        QTimer.singleShot(0, self._focus_update_preview)

    def _focus_update_preview(self) -> None:
        self.apply_btn.setFocus(Qt.OtherFocusReason)

    def _focus_global_calibration(self) -> None:
        self.edit_calib_btn.setFocus(Qt.OtherFocusReason)

    def _apply_overrides(self) -> None:
        try:
            self.config = self.config_with_overrides()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid overrides", str(exc))
            return
        self.config_changed.emit(self.config.copy())
        self.run_requested.emit(PreviewJob())

    def config_with_overrides(self) -> FastAcbfConfig:
        cfg = self.config.copy()
        orient = self.orientation_form.read_values()
        cfg.rotation_deg = float(orient["rotation_deg"])
        cfg.flipud = bool(orient["flipud"])
        cfg.fliplr = bool(orient["fliplr"])
        cfg.transpose = bool(orient["transpose"])
        cfg.focus_sign = str(orient["focus_sign"])
        cfg.aberrations.update(self.aberration_form.read_values(quiet=False))
        return cfg

    def set_config(self, config: FastAcbfConfig) -> None:
        self.config = config.copy()
        messages = self.config.coerce_upscale_method_for_mode()
        if messages:
            self.set_status(" ".join(messages))
        previous = self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(self.config.mode)
        self.mode_combo.blockSignals(previous)
        previous = self.frame_combo.blockSignals(True)
        self.frame_combo.setCurrentText(self.config.output_frame)
        self.frame_combo.blockSignals(previous)
        self.orientation_form.set_values(
            rotation_deg=float(self.config.rotation_deg),
            flipud=bool(self.config.flipud),
            fliplr=bool(self.config.fliplr),
            transpose=bool(self.config.transpose),
            focus_sign=str(self.config.focus_sign),
        )
        kv_text = "unset" if self.config.voltage_kv is None else f"{self.config.voltage_kv:.3g}"
        alpha_mrad_text = (
            "unset" if self.config.max_alpha_mrad is None else f"{self.config.max_alpha_mrad:.3g}"
        )
        alpha_px_text = (
            "unset" if self.config.max_alpha_px is None else f"{self.config.max_alpha_px:.3g}"
        )
        self.calib_label.setText(
            f"kV: {kv_text}    step: {self.config.scan_step_angstrom:.3g} Å    "
            f"dk: {self.config.dk_inv_angstrom:.3g} 1/Å    "
            f"alpha: {alpha_mrad_text} mrad ({alpha_px_text} px)"
        )
        self._update_scale_bars(self.config)
        self.aberration_form.set_max_order(int(self.config.max_order))
        self.aberration_form.set_values(self.config.aberrations)

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

    def reset_for_new_dataset(self, config: FastAcbfConfig) -> None:
        """Clear dataset-specific history and previews, then show fresh settings."""
        self.aberration_form.zero_all()
        self.set_config(config)
        self.table.setRowCount(0)
        if self.image_view is not None:
            self.image_view.clear()
        if self.probe_view is not None:
            self.probe_view.clear()

    def add_history(self, result: dict) -> None:
        cfg: FastAcbfConfig = result.get("config", self.config)
        row = self.table.rowCount()
        self.table.insertRow(row)
        step_labels = {
            "manual": "manual",
            "auto_tune": "Refine All Params",
            "refine_defocus": "Refine Defocus",
            "refine_scan_rotation": "Refine Scan Rotation",
            "refine_flips": "Refine Flips",
            "refine_aberrations": "Refine Aberrations",
            "simple_menu_reconstruct": "Simple Menu",
        }
        values = [
            step_labels.get(result["command"], result["command"]),
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

    def _update_scale_bars(self, config: FastAcbfConfig) -> None:
        pixel_size = config.output_pixel_size_angstrom()
        if self.image_scale_bar is not None:
            self.image_scale_bar.pixel_size = pixel_size
            self.image_scale_bar.units = "A"
            self.image_scale_bar.updateBar()
        if self.probe_scale_bar is not None:
            self.probe_scale_bar.pixel_size = pixel_size
            self.probe_scale_bar.units = "A"
            self.probe_scale_bar.updateBar()
