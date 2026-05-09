import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QPushButton

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.dialogs import ConfigurationDialog, FastAcbfDashboard

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
    assert button_labels.index("Refine Flips") < button_labels.index("Refine Defocus")

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
