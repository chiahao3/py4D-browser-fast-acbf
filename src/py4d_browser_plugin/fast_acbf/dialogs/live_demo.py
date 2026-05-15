"""Live-demo dialog for the fast-acbf plugin."""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from py4D_browser.scalebar import ScaleBar

from ..config import FastAcbfConfig


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
