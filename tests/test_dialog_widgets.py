"""Tests for the shared widgets extracted from dialogs.

``AberrationForm`` and ``OrientationForm`` are the two widgets that were
duplicated between ``ConfigurationDialog`` and ``FastAcbfDashboard``
before the dialogs package split. These tests pin down their behavior
independently of any specific dialog so future changes don't silently
diverge between the two consumers.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from py4d_browser_plugin.fast_acbf.dialogs._widgets import (
    AberrationForm,
    OrientationForm,
)


_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    return _APP


def test_aberration_form_builds_labels_for_max_order():
    _app()
    form = AberrationForm()
    form.set_max_order(2)
    expected = ["C10", "C12a", "C12b", "C21a", "C21b", "C23a", "C23b"]
    assert list(form.aberration_inputs.keys()) == expected
    form.deleteLater()


def test_aberration_form_rebuild_preserves_user_edits():
    _app()
    form = AberrationForm()
    form.set_max_order(1)
    form.aberration_inputs["C10"].setText("5")
    form.aberration_inputs["C12a"].setText("1.5")

    form.set_max_order(2)
    assert form.aberration_inputs["C10"].text() == "5"
    assert form.aberration_inputs["C12a"].text() == "1.5"
    assert "C21a" in form.aberration_inputs
    assert form.aberration_inputs["C21a"].text() == "0"
    form.deleteLater()


def test_aberration_form_set_values_writes_known_fields_and_skips_unknown():
    _app()
    form = AberrationForm()
    form.set_max_order(1)
    form.set_values({"C10": 7.5, "C12a": 0.25, "C30": 99.0})  # C30 unknown at order 1
    assert form.aberration_inputs["C10"].text() == "7.5"
    assert form.aberration_inputs["C12a"].text() == "0.25"
    assert "C30" not in form.aberration_inputs
    form.deleteLater()


def test_aberration_form_read_values_strict_and_quiet():
    _app()
    form = AberrationForm()
    form.set_max_order(1)
    form.aberration_inputs["C10"].setText("garbage")
    form.aberration_inputs["C12a"].setText("2.0")
    form.aberration_inputs["C12b"].setText("")

    quiet = form.read_values(quiet=True)
    assert quiet["C10"] == 0.0
    assert quiet["C12a"] == 2.0
    assert quiet["C12b"] == 0.0

    try:
        form.read_values(quiet=False)
    except ValueError as exc:
        assert "C10" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-numeric input")
    form.deleteLater()


def test_aberration_form_zero_all():
    _app()
    form = AberrationForm()
    form.set_max_order(2)
    for line in form.aberration_inputs.values():
        line.setText("3")
    form.zero_all()
    assert all(line.text() == "0" for line in form.aberration_inputs.values())
    form.deleteLater()


def test_orientation_form_round_trip():
    _app()
    form = OrientationForm()
    form.set_values(
        rotation_deg=12.5, flipud=True, fliplr=False, transpose=True, focus_sign="underfocus"
    )
    assert form.rotation_line.text() == "12.5"
    assert form.flipud_cb.isChecked() is True
    assert form.fliplr_cb.isChecked() is False
    assert form.transpose_cb.isChecked() is True
    assert form.focus_sign_combo.currentText() == "Underfocus"

    read = form.read_values()
    assert read == {
        "rotation_deg": 12.5,
        "flipud": True,
        "fliplr": False,
        "transpose": True,
        "focus_sign": "underfocus",
    }
    form.deleteLater()


def test_orientation_form_focus_sign_defaults_to_none():
    _app()
    form = OrientationForm()
    assert form.focus_sign_combo.currentText() == "None"
    assert form.read_values()["focus_sign"] == "none"
    form.deleteLater()


def test_orientation_form_reset_clears_state():
    _app()
    form = OrientationForm()
    form.set_values(
        rotation_deg=9.0, flipud=True, fliplr=True, transpose=True, focus_sign="underfocus"
    )
    form.reset()
    assert form.read_values() == {
        "rotation_deg": 0.0,
        "flipud": False,
        "fliplr": False,
        "transpose": False,
        "focus_sign": "none",
    }
    form.deleteLater()


def test_orientation_form_empty_rotation_text_reads_as_zero():
    _app()
    form = OrientationForm()
    form.rotation_line.setText("")
    assert form.read_values()["rotation_deg"] == 0.0
    form.deleteLater()


def test_dialogs_package_re_exports_existing_classes():
    """Existing imports `from .dialogs import ConfigurationDialog, ...` must keep working."""
    from py4d_browser_plugin.fast_acbf.dialogs import ConfigurationDialog, FastAcbfDashboard

    assert ConfigurationDialog is not None
    assert FastAcbfDashboard is not None
