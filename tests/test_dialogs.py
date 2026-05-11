import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.dialogs import (
    ConfigurationDialog,
    FastAcbfDashboard,
    LiveDemoDialog,
)

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


def test_flip_controls_map_directly_to_fast_acbf_axes():
    _app()
    cfg = FastAcbfConfig(flipud=True, fliplr=False)
    dialog = ConfigurationDialog(cfg)

    assert dialog.flipud_cb.isChecked() is True
    assert dialog.fliplr_cb.isChecked() is False

    dialog.flipud_cb.setChecked(False)
    dialog.fliplr_cb.setChecked(True)
    values = dialog.values()
    assert values.flipud is False
    assert values.fliplr is True
    dialog.close()


def test_dashboard_labels_and_history_metric():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig())
    button_labels = [child.text() for child in dashboard.findChildren(QPushButton)]
    assert "Run Reconstruction" not in button_labels
    assert "Refine All Params" in button_labels
    assert "Refine Aberrations" in button_labels
    assert "Zero All" in button_labels
    assert "Reset Orientation" in button_labels
    assert "Start Live" not in button_labels
    assert button_labels.index("Refine Flips") < button_labels.index("Refine Defocus")
    label_texts = [child.text() for child in dashboard.findChildren(QLabel)]
    assert "Display mode" in label_texts
    assert "Output frame" in label_texts

    dashboard.add_history(
        {
            "command": "manual",
            "config": FastAcbfConfig(aberrations={"C10": 1.0}, rotation_deg=2.0),
            "metric_text": "3.14",
        }
    )
    assert dashboard.table.horizontalHeaderItem(5).text() == "Metric"
    assert dashboard.table.item(0, 0).text() == "manual"
    assert dashboard.table.item(0, 5).text() == "3.14"
    dashboard.close()


def test_update_preview_button_is_primary_dashboard_action():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig())

    assert dashboard.apply_btn.isDefault() is True
    assert dashboard.apply_btn.autoDefault() is True
    assert dashboard.apply_btn.minimumHeight() >= 34
    assert dashboard.apply_btn.font().bold() is True
    assert "QPushButton:default" in dashboard.apply_btn.styleSheet()

    dashboard.close()


def test_dashboard_display_mode_updates_config():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig(mode="tcBF"))
    changes = []
    runs = []
    dashboard.config_changed.connect(changes.append)
    dashboard.run_requested.connect(runs.append)

    dashboard.mode_combo.setCurrentText("acBF")

    assert dashboard.config.mode == "acBF"
    assert changes[-1].mode == "acBF"
    assert runs == ["apply"]
    dashboard.close()


def test_dashboard_output_frame_updates_config():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig(output_frame="scan"))
    changes = []
    runs = []
    dashboard.config_changed.connect(changes.append)
    dashboard.run_requested.connect(runs.append)

    dashboard.frame_combo.setCurrentText("detector")

    assert dashboard.config.output_frame == "detector"
    assert changes[-1].output_frame == "detector"
    assert runs == ["apply"]
    dashboard.close()


def test_zero_and_reset_controls_update_dashboard_fields():
    _app()
    dashboard = FastAcbfDashboard(
        FastAcbfConfig(
            aberrations={"C10": 5.0, "C12a": 2.0},
            rotation_deg=9.0,
            flipud=True,
            fliplr=True,
            transpose=True,
        )
    )
    dashboard._zero_all_aberrations()
    assert all(line.text() == "0" for line in dashboard.aberration_inputs.values())

    dashboard._reset_orientation()
    assert dashboard.rotation_line.text() == "0"
    assert dashboard.flipud_cb.isChecked() is False
    assert dashboard.fliplr_cb.isChecked() is False
    assert dashboard.transpose_cb.isChecked() is False
    dashboard.close()


def test_probe_scale_bar_matches_reconstruction_scale():
    _app()
    dashboard = FastAcbfDashboard(
        FastAcbfConfig(scan_step_angstrom=2.5, dk_inv_angstrom=0.125)
    )

    assert dashboard.image_scale_bar.pixel_size == 2.5
    assert dashboard.image_scale_bar.units == "A"
    assert dashboard.probe_scale_bar.pixel_size == 2.5
    assert dashboard.probe_scale_bar.units == "A"
    dashboard.close()


def test_live_demo_options_and_active_state():
    _app()
    dialog = LiveDemoDialog(FastAcbfConfig())
    starts = []
    stops = []
    dialog.start_requested.connect(starts.append)
    dialog.stop_requested.connect(lambda: stops.append(True))

    dialog.mode_combo.setCurrentText("acBF")
    dialog.rotation_sweep_spin.setValue(1.25)
    dialog.defocus_sweep_spin.setValue(150.0)
    dialog.defocus_period_spin.setValue(80)
    dialog.display_noise_spin.setValue(2.5)
    dialog.frames_spin.setValue(10)
    dialog.drift_y_spin.setValue(0.25)
    dialog.drift_x_spin.setValue(-0.5)
    dialog.toggle_btn.click()

    assert starts[-1] == {
        "source": "current datacube (mock streamer)",
        "mode": "acBF",
        "use_pinned_source": True,
        "rotation_sweep_deg_per_frame": 1.25,
        "defocus_sweep_angstrom": 150.0,
        "defocus_sweep_period_frames": 80,
        "display_noise_sigma_pct": 2.5,
        "n_frames": 10,
        "drift_y_per_frame": 0.25,
        "drift_x_per_frame": -0.5,
    }

    dialog.set_live_active(True, "warming up")
    assert dialog.toggle_btn.text() == "Stop Live"
    assert dialog.mode_combo.isEnabled() is False
    assert dialog.status_label.text() == "warming up"

    dialog.toggle_btn.click()
    assert stops == [True]
    dialog.set_live_active(False, "stopped")
    assert dialog.toggle_btn.text() == "Start Live"
    assert dialog.mode_combo.isEnabled() is True
    assert dialog.status_label.text() == "stopped"

    stage_text = dialog._format_stage_times(
        {"pinned_h2d": 0.05, "apply_metadata_or_update_dataset": 0.055}
    )
    assert stage_text == "transfer 50.0 ms | update total 55.0 ms"
    dialog.close()
