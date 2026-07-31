import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QTabWidget

from py4d_browser_plugin.fast_acbf.calibration import PLACEHOLDER_WAVELENGTH_ANGSTROM
from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig
from py4d_browser_plugin.fast_acbf.dialogs import ConfigurationDialog, FastAcbfDashboard
from py4d_browser_plugin.fast_acbf.dialogs._widgets import SCAN_ROTATION_HELP_TEXT

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


def test_simple_menu_tab_is_last_and_round_trips():
    _app()
    cfg = FastAcbfConfig(
        simple_menu_output_target="result_image",
        simple_menu_aberration_search="second_order",
        simple_menu_defocus_halfwidth_px=42.0,
        simple_menu_defocus_halfwidth_scan_fraction=0.3,
        simple_menu_seeded_defocus_fraction=0.4,
    )
    dialog = ConfigurationDialog(cfg)

    tabs = dialog.findChild(QTabWidget)
    assert "Aberrations" in [tabs.tabText(index) for index in range(tabs.count())]
    assert "Optics" not in [tabs.tabText(index) for index in range(tabs.count())]
    assert tabs.tabText(tabs.count() - 1) == "Simple Menu"
    assert dialog.simple_menu_output_combo.currentText() == "Result image"
    assert dialog.simple_menu_aberration_combo.currentText() == "Up to 2nd order"
    assert dialog.simple_menu_defocus_halfwidth_line.text() == "42"
    assert dialog.simple_menu_defocus_halfwidth_scan_fraction_line.text() == "0.3"
    assert dialog.simple_menu_seeded_defocus_fraction_line.text() == "0.4"

    dialog.simple_menu_output_combo.setCurrentText("Virtual image")
    dialog.simple_menu_aberration_combo.setCurrentText("df only")
    dialog.simple_menu_defocus_halfwidth_line.setText("8")
    dialog.simple_menu_defocus_halfwidth_scan_fraction_line.setText("0.1")
    dialog.simple_menu_seeded_defocus_fraction_line.setText("0.6")
    values = dialog.values()
    assert values.simple_menu_output_target == "virtual_image"
    assert values.simple_menu_aberration_search == "df_only"
    assert values.simple_menu_defocus_halfwidth_px == 8.0
    assert values.simple_menu_defocus_halfwidth_scan_fraction == 0.1
    assert values.simple_menu_seeded_defocus_fraction == 0.6
    dialog.close()


def test_simple_menu_defocus_halfwidth_blank_means_auto():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig(simple_menu_defocus_halfwidth_px=42.0))
    assert dialog.simple_menu_defocus_halfwidth_line.text() == "42"

    dialog.simple_menu_defocus_halfwidth_line.setText("")
    values = dialog.values()

    assert values.simple_menu_defocus_halfwidth_px is None
    # blank means "derive from scan size" -- still resolves to a usable value.
    assert values.resolved_simple_menu_defocus_halfwidth_px(min_scan_dim=64) == 20.0
    dialog.close()


def test_default_width_fits_all_tabs_without_overflow():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig())
    dialog.show()
    _app().processEvents()
    tabs = dialog.findChild(QTabWidget)
    tab_bar = tabs.tabBar()
    last_tab_right = tab_bar.tabRect(tab_bar.count() - 1).right()
    assert last_tab_right <= tab_bar.width()
    dialog.close()


def test_calibration_free_checkbox_round_trips_and_shows_alpha_px_editable():
    _app()
    cfg = FastAcbfConfig(calibration_free=False, max_alpha_px=42.5, max_alpha_mrad=99.0)
    dialog = ConfigurationDialog(cfg)

    assert dialog.calibration_free_cb.isChecked() is False
    assert dialog.max_alpha_px_line.text() == "42.5"
    assert dialog.max_alpha_px_line.isReadOnly() is False
    assert dialog.max_alpha_line.isReadOnly() is True

    dialog.calibration_free_cb.setChecked(True)
    values = dialog.values()
    assert values.calibration_free is True
    dialog.close()


def test_configuration_dialog_derives_mrad_from_edited_max_alpha_px():
    _app()
    cfg = FastAcbfConfig(max_alpha_px=10.0, dk_inv_angstrom=0.05, voltage_kv=None)
    dialog = ConfigurationDialog(cfg)

    # No real voltage set -> the live display and values() both fall back to the
    # same calibration-free placeholder wavelength FastAcbfConfig.resolved_for uses.
    dialog.max_alpha_px_line.setText("20.0")
    dialog._update_max_alpha_mrad_display()
    expected = 20.0 * 0.05 * PLACEHOLDER_WAVELENGTH_ANGSTROM * 1000.0
    assert float(dialog.max_alpha_line.text()) == pytest.approx(expected, rel=1e-4)

    values = dialog.values()
    assert values.max_alpha_px == pytest.approx(20.0)
    assert values.max_alpha_mrad == pytest.approx(expected, rel=1e-4)
    dialog.close()


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


def test_configuration_dialog_round_trips_fast_acbf_060_preparation_fields():
    _app()
    cfg = FastAcbfConfig(pad_width=5, upscale=2.0, upscale_method="nearest")
    dialog = ConfigurationDialog(cfg)

    assert dialog.pad_width_line.text() == "5"
    assert dialog.upscale_line.text() == "2"
    assert dialog.upscale_method_combo.currentText() == "nearest"

    dialog.pad_width_line.setText("0")
    dialog.upscale_line.setText("1.5")
    dialog.upscale_method_combo.setCurrentText("bilinear")
    values = dialog.values()

    assert values.pad_width is None
    assert values.upscale == 1.5
    assert values.upscale_method == "bilinear"
    dialog.close()


def test_configuration_dialog_disables_zero_insert_for_acbf():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig(mode="tcBF", upscale_method="zero_insert"))

    zero_insert_index = dialog.upscale_method_combo.findText("zero_insert")
    assert zero_insert_index >= 0
    assert dialog.upscale_method_combo.model().item(zero_insert_index).isEnabled() is True

    dialog.mode_combo.setCurrentText("acBF")

    assert dialog.upscale_method_combo.currentText() == "nearest"
    assert dialog.upscale_method_combo.model().item(zero_insert_index).isEnabled() is False
    assert dialog.values().upscale_method == "nearest"
    dialog.close()


def test_configuration_dialog_rejects_fractional_zero_insert():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig(upscale_method="zero_insert"))
    dialog.upscale_line.setText("1.5")

    try:
        dialog.values()
    except ValueError as exc:
        assert "integer upscale factor" in str(exc)
    else:
        raise AssertionError("fractional zero_insert upscale should fail")
    dialog.close()


def test_configuration_dialog_round_trips_refinement_search_fields():
    _app()
    cfg = FastAcbfConfig(
        defocus_range_min_angstrom=-10.0,
        defocus_range_max_angstrom=20.0,
        defocus_range_tolerance_factor=12.0,
        fine_rotation_halfwidth_deg=2.5,
        fine_rotation_points=13,
        fine_rotation_xatol_deg=0.2,
    )
    dialog = ConfigurationDialog(cfg)

    assert dialog.defocus_min_line.text() == "-10"
    assert dialog.defocus_max_line.text() == "20"
    assert dialog.defocus_tolerance_line.text() == "12"
    assert dialog.fine_rotation_halfwidth_line.text() == "2.5"
    assert dialog.fine_rotation_points_spin.value() == 13
    assert dialog.fine_rotation_xatol_line.text() == "0.2"

    dialog.defocus_min_line.setText("")
    dialog.defocus_max_line.setText("")
    dialog.defocus_halfwidth_line.setText("15")
    dialog.rotation_min_line.setText("-7")
    dialog.rotation_max_line.setText("8")
    values = dialog.values()

    assert values.defocus_search_range() is None
    assert values.defocus_search_halfwidth_angstrom == 15.0
    assert values.rotation_search_range() == (-7.0, 8.0)
    assert values.fine_rotation_halfwidth_deg == 2.5
    assert values.fine_rotation_xatol_deg == 0.2
    dialog.close()


def test_configuration_dialog_rejects_non_positive_fine_rotation_xatol():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig())
    dialog.fine_rotation_xatol_line.setText("0")

    try:
        dialog.values()
    except ValueError as exc:
        assert "tolerance" in str(exc).lower()
    else:
        raise AssertionError("non-positive fine_rotation_xatol should fail")
    dialog.close()


def test_configuration_dialog_blank_fine_rotation_halfwidth_means_auto():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig(fine_rotation_halfwidth_deg=2.5))
    assert dialog.fine_rotation_halfwidth_line.text() == "2.5"

    dialog.fine_rotation_halfwidth_line.setText("")
    values = dialog.values()

    assert values.fine_rotation_halfwidth_deg is None
    # blank means "derive from rotation_points" -- still resolves to a usable value
    # (default rotation_points=12 -> 180/12=15.0).
    assert values.resolved_fine_rotation_halfwidth_deg() == 180.0 / values.rotation_points
    dialog.close()
    dialog.close()


def test_configuration_dialog_has_no_refinement_tab_focus_sign_controls():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig())
    assert not hasattr(dialog, "force_overfocus_cb")
    assert not hasattr(dialog, "orientation_note_label")
    dialog.close()


def test_configuration_dialog_orientation_tab_round_trips_focus_sign():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig(focus_sign="none"))
    assert dialog.orientation_form.focus_sign_combo.currentText() == "None"

    dialog.orientation_form.focus_sign_combo.setCurrentText("Underfocus")
    values = dialog.values()
    assert values.focus_sign == "underfocus"
    dialog.close()


def test_configuration_and_dashboard_orientation_tabs_show_rotation_help():
    _app()
    config_dialog = ConfigurationDialog(FastAcbfConfig())
    dashboard = FastAcbfDashboard(FastAcbfConfig())

    assert config_dialog.orientation_form.rotation_help_btn.toolTip() == SCAN_ROTATION_HELP_TEXT
    assert dashboard.orientation_form.rotation_help_btn.toolTip() == SCAN_ROTATION_HELP_TEXT

    config_dialog.close()
    dashboard.close()


def test_configuration_dialog_prefills_disabled_pad_width_as_zero():
    _app()
    dialog = ConfigurationDialog(FastAcbfConfig(pad_width=None))

    assert dialog.pad_width_line.text() == "0"
    assert dialog.values().pad_width is None
    dialog.close()


def test_configuration_dialog_round_trips_live_view_outputs():
    _app()
    cfg = FastAcbfConfig(
        live_virtual_output="probe",
        live_result_output="chi",
        live_auto_focus_interval_s=7.5,
        live_auto_aberrations_interval_s=45.0,
    )
    dialog = ConfigurationDialog(cfg)

    assert dialog.live_virtual_output_combo.currentText() == "probe"
    assert dialog.live_result_output_combo.currentText() == "chi"
    assert dialog.live_auto_focus_interval_spin.value() == 7.5
    assert dialog.live_auto_aberrations_interval_spin.value() == 45.0

    dialog.live_virtual_output_combo.setCurrentText("None")
    dialog.live_result_output_combo.setCurrentText("acBF")
    dialog.live_auto_focus_interval_spin.setValue(2.25)
    dialog.live_auto_aberrations_interval_spin.setValue(12.5)

    values = dialog.values()
    assert values.live_virtual_output == "None"
    assert values.live_result_output == "acBF"
    assert values.live_auto_focus_interval_s == 2.25
    assert values.live_auto_aberrations_interval_s == 12.5
    assert values.upscale_method == "nearest"
    dialog.close()


def test_dashboard_labels_and_history_metric():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig())
    tab_labels = [
        dashboard.parameter_tabs.tabText(index)
        for index in range(dashboard.parameter_tabs.count())
    ]
    assert "Aberrations" in tab_labels
    assert "Optics" not in tab_labels
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

    dashboard.add_history(
        {
            "command": "simple_menu_reconstruct",
            "config": FastAcbfConfig(),
        }
    )
    assert dashboard.table.item(1, 0).text() == "Simple Menu"
    dashboard.close()


def test_dashboard_dataset_reset_clears_history_and_restores_defaults():
    _app()
    dashboard = FastAcbfDashboard(
        FastAcbfConfig(rotation_deg=17.0, aberrations={"C10": -30.0})
    )
    dashboard.add_history(
        {
            "command": "manual",
            "config": dashboard.config.copy(),
        }
    )

    dashboard.reset_for_new_dataset(FastAcbfConfig())

    assert dashboard.table.rowCount() == 0
    assert dashboard.orientation_form.rotation_line.text() == "0"
    assert dashboard.aberration_form.aberration_inputs["C10"].text() == "0"
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


def test_dashboard_startup_focus_targets_global_calibration():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig())
    dashboard._focus_global_calibration()

    assert dashboard.focusWidget() is dashboard.edit_calib_btn

    dashboard.close()


def test_dashboard_display_mode_updates_config():
    _app()
    # upscale=2.0 so switching to acBF actually needs to coerce zero_insert away
    # (at upscale=1.0 zero_insert is a no-op regardless of mode, see
    # test_dashboard_display_mode_keeps_zero_insert_when_upscale_is_one below).
    dashboard = FastAcbfDashboard(FastAcbfConfig(mode="tcBF", upscale=2.0))
    changes = []
    runs = []
    dashboard.config_changed.connect(changes.append)
    dashboard.run_requested.connect(runs.append)

    dashboard.mode_combo.setCurrentText("acBF")

    assert dashboard.config.mode == "acBF"
    assert dashboard.config.upscale_method == "nearest"
    assert changes[-1].mode == "acBF"
    assert changes[-1].upscale_method == "nearest"
    assert len(runs) == 1
    assert runs[0].command == "manual"
    dashboard.close()


def test_dashboard_display_mode_keeps_zero_insert_when_upscale_is_one():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig(mode="tcBF", upscale=1.0))
    dashboard.mode_combo.setCurrentText("acBF")

    assert dashboard.config.mode == "acBF"
    assert dashboard.config.upscale_method == "zero_insert"
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
    assert len(runs) == 1
    assert runs[0].command == "manual"
    dashboard.close()


def test_dashboard_update_publishes_orientation_before_requesting_run():
    _app()
    dashboard = FastAcbfDashboard(FastAcbfConfig(rotation_deg=3.0))
    events = []
    dashboard.config_changed.connect(lambda config: events.append(("config", config)))
    dashboard.run_requested.connect(lambda job: events.append(("run", job)))

    dashboard.orientation_form.rotation_line.setText("17.5")
    dashboard.apply_btn.click()

    assert [event for event, _value in events] == ["config", "run"]
    assert events[0][1].rotation_deg == 17.5
    assert dashboard.config.rotation_deg == 17.5
    dashboard.close()


def test_probe_scale_bar_matches_reconstruction_scale():
    _app()
    dashboard = FastAcbfDashboard(
        FastAcbfConfig(scan_step_angstrom=2.5, dk_inv_angstrom=0.125, upscale=2.0)
    )

    assert dashboard.image_scale_bar.pixel_size == 1.25
    assert dashboard.image_scale_bar.units == "A"
    assert dashboard.probe_scale_bar.pixel_size == 1.25
    assert dashboard.probe_scale_bar.units == "A"
    dashboard.close()
