"""The advanced dashboard's views: side views, depth stack, ortho, history, panel.

Ported from ptydy's tests of the same views (its fast-acbf app); ``qtbot`` is a minimal
local stand-in for pytest-qt's (keep a widget until the test ends, process events).
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from py4d_browser_plugin.fast_acbf.config import FastAcbfConfig  # noqa: E402
from py4d_browser_plugin.fast_acbf.dialogs import FastAcbfDashboard  # noqa: E402
from py4d_browser_plugin.fast_acbf.solver_job import DepthStackJob  # noqa: E402


class _QtBot:
    def __init__(self):
        self.widgets = []

    def addWidget(self, widget):  # noqa: N802 (pytest-qt's name)
        self.widgets.append(widget)

    def wait(self, ms: int) -> None:
        from PyQt5.QtCore import QElapsedTimer

        timer = QElapsedTimer()
        timer.start()
        while timer.elapsed() < ms:
            QApplication.processEvents()


_APP = None


@pytest.fixture
def qtbot():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication(sys.argv)
    bot = _QtBot()
    yield bot
    for w in bot.widgets:
        w.close()
        w.deleteLater()
    QApplication.processEvents()


def _result(cfg, command="manual", **extra):
    rng = np.random.default_rng(0)
    probe = (rng.random((16, 16)) + 1j * rng.random((16, 16))).astype(np.complex64)
    chi = np.full((8, 8), np.nan, np.float32)
    chi[2:6, 2:6] = 1.5
    rows, cols = np.nonzero(np.isfinite(chi))
    return {"image": rng.random((12, 20)), "probe_complex": probe, "probe": np.abs(probe),
            "probe_pixel_size": 0.3, "chi": chi, "chi_pixel_size": 0.05,
            "shifts_yx": rng.normal(size=(len(rows), 2)), "bf_rows": rows, "bf_cols": cols,
            "command": command, "config": cfg, "metric_text": "1", "mode": cfg.mode, **extra}


def _stack_result(cfg, n=5):
    rng = np.random.default_rng(1)
    base = _result(cfg, "depth_stack")
    defocus = np.arange(n) - n // 2  # χ and the shifts grow away from the centre slice
    return {**base, "stack": rng.random((n, 12, 20)).astype(np.float32),
            "stack_c10": 50.0 + 20.0 * defocus,
            "probe_stack": (rng.random((n, 16, 16)) + 1j * rng.random((n, 16, 16))
                            ).astype(np.complex64),
            "chi_stack": np.stack([base["chi"] * (1 + d) for d in defocus]),
            "shifts_stack": np.stack([base["shifts_yx"] * (1 + abs(d)) for d in defocus]),
            "stack_step": 20.0}


@pytest.fixture
def dash(qtbot):
    d = FastAcbfDashboard(FastAcbfConfig(scan_step_angstrom=1.0, upscale=1.0))
    qtbot.addWidget(d)
    d.show()
    return d


def test_history_is_read_only(dash):
    from PyQt5.QtWidgets import QAbstractItemView

    cfg = dash.config.copy()
    cfg.aberrations = {"C10": -30.0, "C12a": 4.0}
    cfg.rotation_deg = 12.0
    dash.set_result(_result(cfg))
    dash.set_result(_result(dash.config.copy()))
    assert dash.table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers
    assert dash.table.rowCount() == 2 and dash.table.item(0, 1).text() == "-30"


def test_side_views(dash):
    res = _result(dash.config)
    dash.set_result(res)
    assert dash.side_stack.currentWidget() is dash.side_pane
    np.testing.assert_allclose(dash.side_pane.raw, np.abs(res["probe_complex"]))
    assert dash.side_pane.pixel_size == 0.3
    dash.side_buttons.buttons["probe_int"].click()
    np.testing.assert_allclose(dash.side_pane.raw, np.abs(res["probe_complex"]) ** 2,
                               rtol=1e-6)
    dash.side_buttons.buttons["probe_complex"].click()
    assert np.iscomplexobj(dash.side_pane.raw)
    dash.side_buttons.buttons["chi"].click()
    assert dash.side_pane.raw.shape == (8, 8)  # cropped to the disk (4 px) + 2 px margin
    assert np.isnan(dash.side_pane.raw[0, 0]) and dash.side_pane.units == "A^-1"
    dash.side_buttons.buttons["shifts"].click()
    assert dash.side_stack.currentWidget() is dash.quiver
    assert len(dash.quiver.segments) == 2 * len(res["bf_rows"])  # 16 pixels, all drawn


def test_depth_stack_cache_slices_and_ortho(dash):
    runs = []
    dash.run_requested.connect(runs.append)
    dash.set_result(_result(dash.config))
    dash.slices_spin.setValue(5)
    dash.depth_buttons.buttons["3D"].click()  # no stack yet: asks for one
    assert isinstance(runs[-1], DepthStackJob) and runs[-1].n_slices == 5
    stack = _stack_result(dash.config)
    dash.set_result(stack)  # the worker's answer
    assert dash.depth_mode == "3D" and dash.stack_is_current()
    assert dash.slice_slider.maximum() == 4 and dash.slice_spin.value() == 2
    assert "cached" in dash.stack_label.text()
    np.testing.assert_array_equal(dash.recon_pane.raw, stack["stack"][2])
    levels = dash.recon_pane.image_item.getLevels()
    dash.slice_slider.setValue(4)
    np.testing.assert_array_equal(dash.recon_pane.raw, stack["stack"][4])
    np.testing.assert_allclose(dash.recon_pane.image_item.getLevels(), levels)  # one range
    np.testing.assert_allclose(dash.side_pane.raw, np.abs(stack["probe_stack"][4]))
    assert "Δ +40.0 Å" in dash.slice_label.text()
    assert dash.table.rowCount() == 1  # a stack is not a history step

    assert dash.ortho_btn.isVisible()  # a display choice of the stack: in the slice row
    dash.ortho_btn.setChecked(True)
    assert dash.recon_stack.currentWidget() is dash.recon_ortho
    assert dash.side_stack.currentWidget() is dash.side_ortho
    assert dash.recon_ortho.img_xz.image.shape == (5, 20)  # z by x
    assert dash.recon_ortho.img_yz.image.shape == (12, 5)  # y by z
    dash.slice_spin.setValue(1)
    assert dash.recon_ortho.z == 1 and dash.side_ortho.z == 1

    n_runs = len(runs)
    dash.depth_buttons.buttons["2D"].click()
    dash.depth_buttons.buttons["3D"].click()  # cached: no new run
    assert len(runs) == n_runs
    dash.slices_spin.setValue(7)  # a different request: the cache is out of date
    assert not dash.stack_is_current() and "earlier" in dash.stack_label.text()


def test_rotated_detector_frame_output_ignores_the_fill(dash):
    from scipy import ndimage

    img = 100 + np.random.default_rng(2).normal(0, 1, (48, 48))
    rotated = ndimage.rotate(img, 30, reshape=False, order=1, cval=0.0)
    dash.set_result({**_result(dash.config), "image": rotated})
    lo, hi = dash.recon_pane.image_item.getLevels()
    assert dash.recon_pane.valid is not None and 90 < lo < hi < 110


def test_pipeline_and_controls_actions(dash):
    assert not hasattr(dash, "flap")  # the ribbon chip and Ctrl+B are enough
    assert dash.panel_action.text() == "Pipeline"
    assert dash.left_panel.isVisible()
    dash.panel_action.trigger()
    assert not dash.left_panel.isVisible() and not dash.panel_action.isChecked()
    dash.panel_action.trigger()
    assert dash.left_panel.isVisible()
    dash.controls_action.trigger()
    assert not any(v.footer.isVisible() for v in dash.views())
    dash.controls_action.trigger()
    assert dash.recon_pane.footer.isVisible() and dash.side_pane.footer.isVisible()


def test_new_dataset_clears_stack_and_history(dash):
    dash.set_result(_result(dash.config))
    dash.set_result(_stack_result(dash.config))
    dash.reset_for_new_dataset(FastAcbfConfig())
    assert dash.stack is None and dash.history == [] and dash.depth_mode == "2D"
    assert not dash.slice_row.isVisible()


def test_stack_levels_ignore_a_degenerate_slice():
    from py4d_browser_plugin.fast_acbf.dialogs.views import stack_levels

    stack = 0.97 + 0.02 * np.random.default_rng(4).random((21, 32, 32))
    stack[8, :, :] = 0.0
    stack[8, ::2, ::2] = 1.0  # zero-insert upscaling at C10 = 0: 3/4 of the slice empty
    lo, hi = stack_levels(stack)
    assert 0.96 < lo < hi < 1.0
    assert lo < 0.972 and hi > 0.988  # the percentiles of all other slices together


def test_ortho_sections_are_drawn_where_their_axes_say(qtbot):
    from py4d_browser_plugin.fast_acbf.dialogs.views import OrthoView

    v = OrthoView()
    qtbot.addWidget(v)
    v.set_stack(np.random.default_rng(5).random((21, 64, 48)), step=1.0, pixel_size=1.0)
    xz = v.img_xz.mapRectToParent(v.img_xz.boundingRect())
    yz = v.img_yz.mapRectToParent(v.img_yz.boundingRect())
    assert (xz.width(), xz.height()) == pytest.approx((48, 21 * v.z_height))
    assert (yz.width(), yz.height()) == pytest.approx((21 * v.z_height, 64))
    assert v.caption.text() == "depth to scale"  # 21 slices, 1 px apart: fits as is
    assert v.scale_bar.isVisible()


def test_ortho_view_has_pane_controls_and_draggable_handles(qtbot):
    from py4d_browser_plugin.fast_acbf.dialogs.imaging import stack_levels
    from py4d_browser_plugin.fast_acbf.dialogs.views import OrthoView

    v = OrthoView()
    qtbot.addWidget(v)
    v.show()
    stack = np.random.default_rng(6).random((9, 32, 40)).astype(np.float32)
    stack[4] *= 10  # one slice brighter: the levels stay those of the whole stack
    v.set_stack(stack, step=1.0, pixel_size=1.0)
    assert v.levels == pytest.approx(stack_levels(stack, 0.5, 99.5))
    v.set_z(4)
    assert v.levels == pytest.approx(stack_levels(stack, 0.5, 99.5))
    v.set_colormap("viridis")  # the sections follow the colour map and the levels
    assert v.img_xz.lut is v.img_xy.lut and v.img_yz.lut is v.img_xy.lut
    v.set_fixed_levels(0.2, 0.3)
    assert v.img_xz.levels == pytest.approx((0.2, 0.3))

    zs = []
    v.z_changed.connect(zs.append)
    v.handle_xy.setPos(10.5, 20.5)  # what a drag does
    assert v.point() == (20, 10)
    np.testing.assert_allclose(v.img_xz.image, stack[:, 20, :])
    v.handle_xz.setPos(30.5, (6 + 0.5) * v.z_height)  # x and the slice
    assert v.point() == (20, 30) and v.z == 6 and zs == [6]
    v.handle_yz.setPos((2 + 0.5) * v.z_height, 5.5)  # y and the slice
    assert v.point() == (5, 30) and v.z == 2 and zs == [6, 2]


def test_chi_is_gray_mod_2pi_by_default_and_side_views_keep_their_colour_maps(dash):
    from py4d_browser_plugin.fast_acbf.dialogs.image_pane import CYCLIC, GRAY_2PI

    dash.set_result(_result(dash.config))
    assert dash.side_pane.cmap == "gray"
    dash.set_side_view("chi")
    assert dash.side_pane.cmap == GRAY_2PI
    assert dash.side_pane.image_item.getLevels() == pytest.approx((0, 2 * np.pi))
    assert dash.side_pane.cmap_combo.findData(CYCLIC) >= 0  # the coloured one is offered
    assert "aperture plane" in dash.side_pane.title.text()
    dash.side_pane.set_colormap("viridis")
    dash.set_side_view("probe_amp")
    assert dash.side_pane.cmap == "gray"
    dash.set_side_view("chi")
    assert dash.side_pane.cmap == "viridis"


def test_grad_chi_quiver(dash):
    chi = np.full((10, 10), np.nan)
    rr, cc = np.mgrid[:10, :10]
    inside = (rr - 5) ** 2 + (cc - 5) ** 2 <= 9
    chi[inside] = 2.0 * cc[inside] - 1.0 * rr[inside]  # constant gradient
    dash.set_result({**_result(dash.config), "chi": chi, "chi_pixel_size": 0.5})
    dash.set_side_view("grad_chi")
    assert dash.side_stack.currentWidget() is dash.quiver
    v = dash.quiver._data["v"]
    # the rim too (one-sided differences), except the four tips of the disk, which have
    # no neighbour along one axis
    assert len(v) == inside.sum() - 4
    np.testing.assert_allclose(v, np.tile([-2.0, 4.0], (len(v), 1)))  # (d/dy, d/dx) / 0.5
    assert dash.axes_buttons.isHidden() or not dash.axes_buttons.isVisible()
    dash.set_side_view("shifts")  # only the shifts have a choice of axes
    assert dash.axes_buttons.isVisible()


def test_quiver_arrows_have_heads_and_controls(dash):
    dash.set_result(_result(dash.config))
    dash.set_side_view("shifts")
    q = dash.quiver
    assert not q.heads.path().isEmpty()
    width = q.arrows.opts["pen"].widthF()
    q.width_spin.setValue(5.0)
    assert q.arrows.opts["pen"].widthF() == 5.0 > width
    tips = q.segments[1::2] - q.segments[0::2]
    q.length_spin.setValue(2.0)
    np.testing.assert_allclose(q.segments[1::2] - q.segments[0::2], 2 * tips)
    q.count_spin.setValue(10)  # 16 pixels over 10 arrows: every 2nd pixel on a grid
    assert len(q.segments) < 2 * 16


def test_footers_have_one_height(dash, qtbot):
    assert len({v.footer.minimumHeight() for v in dash.views()}) == 1
    dash.set_result(_result(dash.config))
    qtbot.wait(10)
    height = dash.recon_pane.footer.height()
    dash.set_result(_stack_result(dash.config))  # 3D: the Range rows appear
    qtbot.wait(10)
    assert dash.range_buttons["recon"].isVisible()
    assert dash.recon_pane.footer.height() == height
    dash.set_side_view("shifts")
    qtbot.wait(10)
    assert dash.quiver.footer.height() == height


def test_3d_range_is_one_for_the_stack_unless_per_slice(dash):
    dash.set_result(_result(dash.config))
    stack = _stack_result(dash.config)
    stack["stack"][4] += 5.0
    dash.set_result(stack)
    assert dash.range_buttons["recon"].current() == "stack"
    levels = dash.recon_pane.image_item.getLevels()
    dash.set_slice(4)
    np.testing.assert_allclose(dash.recon_pane.image_item.getLevels(), levels)
    assert dash.recon_pane.autoscale  # Auto stays on, over the stack
    side = dash.side_pane.image_item.getLevels()
    dash.set_slice(0)
    np.testing.assert_allclose(dash.side_pane.image_item.getLevels(), side)
    dash.range_buttons["recon"].buttons["slice"].click()  # opt in: a range per slice
    dash.set_slice(4)
    assert dash.recon_pane.image_item.getLevels()[0] > 5.0
    dash.depth_buttons.buttons["2D"].click()
    assert not dash.range_buttons["recon"].isVisible()  # 3D only
    assert dash.recon_pane.level_stack is None


def test_shift_axes_on_the_detector_grid_follow_grad_chi():
    """Detector-grid arrows (orientation undone) are ∇χ_det / 2π of the fast-acbf
    solver, for every flip combination and a scan rotation (both frames)."""
    import itertools
    import logging

    fast_acbf = pytest.importorskip("fast_acbf")
    from py4d_browser_plugin.fast_acbf.dialogs.dashboard import nan_gradient, shifts_on_detector_grid
    from py4d_browser_plugin.fast_acbf.worker import optics_diagnostics

    logging.disable(logging.CRITICAL)
    try:
        data = np.random.default_rng(0).random((4, 4, 48, 48)).astype(np.float32) + 1
        for flags in itertools.product([False, True], repeat=3):
            cfg = FastAcbfConfig(flipud=flags[0], fliplr=flags[1], transpose=flags[2],
                                 rotation_deg=30.0)
            solver = fast_acbf.BFSolver(
                dataset=data, max_alpha=25.0, scan_step_size=1.0, dk=0.03, wavelength=0.0418,
                max_order=2, aberrations={"C10": 50.0, "C12a": 80.0, "C12b": -30.0,
                                          "C21a": 2000.0, "C21b": 900.0},
                device="cpu", coord_transform={**dict(zip(("flipud", "fliplr", "transpose"),
                                                          flags, strict=True)), "rotation_deg": 30.0})
            det = optics_diagnostics(solver, "detector", 1.0)
            gy, gx = nan_gradient(det["chi"], det["chi_pixel_size"])
            r, c = det["bf_rows"], det["bf_cols"]
            disk = np.isfinite(det["chi"])  # central differences: all 4 neighbours inside
            inner = disk[r - 1, c] & disk[r + 1, c] & disk[r, c - 1] & disk[r, c + 1]
            expected = np.column_stack([gy[r, c], gx[r, c]])[inner] / (2 * np.pi)
            for frame in ("detector", "scan"):
                out = optics_diagnostics(solver, frame, 1.0)
                grid = shifts_on_detector_grid(out["shifts_yx"], cfg, frame)[inner]
                np.testing.assert_allclose(grid, expected, atol=2e-3 * np.abs(expected).max())
    finally:
        logging.disable(logging.NOTSET)


def test_mode_label_lines_up_with_the_other_rows(dash):
    def first_label(layout):
        return next(layout.itemAt(i).widget() for i in range(layout.count())
                    if layout.itemAt(i).widget() is not None)

    mode = first_label(dash.top_row)
    assert mode.text() == "Mode"
    show = first_label(dash.side_buttons.parentWidget().layout().itemAt(1).layout())
    assert mode.width() == show.width() == mode.minimumWidth()


def test_side_views_follow_the_slice_with_one_scale(dash):
    dash.set_result(_result(dash.config))
    stack = _stack_result(dash.config)
    dash.set_result(stack)
    dash.set_side_view("chi")
    dash.set_slice(4)
    np.testing.assert_allclose(dash.side_pane.raw[2:6, 2:6], stack["chi_stack"][4][2:6, 2:6])
    assert "slice 4" in dash.side_pane.title.text()

    dash.set_side_view("shifts")
    seg4 = dash.quiver.segments.copy()
    np.testing.assert_allclose(dash.quiver._data["v"], stack["shifts_stack"][4])
    dash.set_slice(2)  # in focus: the same vectors / 3, drawn at the same scale
    seg2 = dash.quiver.segments
    np.testing.assert_allclose(seg2[1::2] - seg2[0::2], (seg4[1::2] - seg4[0::2]) / 3)
    assert "stack:" in dash.quiver.caption.text()

    dash.set_side_view("grad_chi")
    g2 = dash.quiver.segments.copy()
    dash.set_slice(4)
    g4 = dash.quiver.segments
    np.testing.assert_allclose(g4[1::2] - g4[0::2], 3 * (g2[1::2] - g2[0::2]))


def test_bc_range_label_and_stack_range_does_not_clip_the_contrasty_slices(dash):
    row = dash.recon_pane._rows[dash.range_buttons["recon"]]
    assert row.layout().itemAt(0).widget().text() == "B/C Range"
    dash.set_result(_result(dash.config))
    stack = _stack_result(dash.config)
    rng = np.random.default_rng(7)
    # contrast grows away from focus (as in a real defocus series)
    stack["stack"] = np.stack([1 + (0.1 + abs(d)) * rng.normal(size=(12, 20))
                               for d in range(-2, 3)]).astype(np.float32)
    dash.set_result(stack)
    lo, hi = dash.recon_pane.image_item.getLevels()
    widest = stack["stack"][0]
    # the stack range reaches (nearly) the contrasty slices' own percentiles; the old
    # median of per-slice ranges clipped them to the middle slice's
    assert hi > np.percentile(widest, 95) and lo < np.percentile(widest, 5)
    dash.set_slice(0)
    assert dash.recon_pane.image_item.getLevels() == pytest.approx((lo, hi))


def test_ortho_view_follows_the_probe_view(dash):
    dash.set_result(_result(dash.config))
    stack = _stack_result(dash.config)
    dash.set_result(stack)
    dash.ortho_btn.setChecked(True)
    assert dash.side_stack.currentWidget() is dash.side_ortho
    np.testing.assert_allclose(dash.side_ortho.stack, np.abs(stack["probe_stack"]), rtol=1e-6)
    dash.side_buttons.buttons["probe_int"].click()
    assert dash.side_stack.currentWidget() is dash.side_ortho
    np.testing.assert_allclose(dash.side_ortho.stack, np.abs(stack["probe_stack"]) ** 2,
                               rtol=1e-5)
    assert dash.side_ortho.title.text() == "Probe |ψ|²"
    dash.side_buttons.buttons["probe_complex"].click()
    assert np.iscomplexobj(dash.side_ortho.stack)
    assert dash.side_ortho.img_xz.image.shape[-1] == 3  # hue = phase in the sections too
    dash.set_slice(1)
    assert dash.side_ortho.z == 1
    dash.side_buttons.buttons["chi"].click()  # χ has no ortho view: the slice image
    assert dash.side_stack.currentWidget() is dash.side_pane


def test_images_keep_their_orientation_under_py4d_column_major_default(qtbot):
    """py4D-browser leaves pyqtgraph column-major; every image item here is row-major on
    its own, so an (rows, cols) image is cols wide and rows tall, and the ortho sections
    are (nz, nx) and (ny, nz) as their rectangles say (a transposed section was stretched
    into its rectangle)."""
    import pyqtgraph as pg

    from py4d_browser_plugin.fast_acbf.dialogs.image_pane import ImagePane
    from py4d_browser_plugin.fast_acbf.dialogs.views import OrthoView

    assert pg.getConfigOption("imageAxisOrder") == "col-major"
    pane = ImagePane("t")
    qtbot.addWidget(pane)
    pane.set_image(np.arange(32.0).reshape(4, 8), reset=True)
    assert (pane.image_item.width(), pane.image_item.height()) == (8, 4)

    v = OrthoView("o")
    qtbot.addWidget(v)
    nz, ny, nx = 3, 5, 7
    v.set_stack(np.random.default_rng(0).random((nz, ny, nx)), step=1.0, pixel_size=1.0)
    assert (v.img_xy.width(), v.img_xy.height()) == (nx, ny)
    assert (v.img_xz.width(), v.img_xz.height()) == (nx, nz)
    assert (v.img_yz.width(), v.img_yz.height()) == (nz, ny)


def test_zero_vector_draws_no_shaft_or_head(qtbot):
    """A zero shift (the pixel at the disk centre) keeps its base dot but no shaft or head
    (Qt 5 drew its zero-length segment as a long horizontal bar; Qt 6 draws a dot)."""
    from py4d_browser_plugin.fast_acbf.dialogs.views import QuiverView

    q = QuiverView()
    qtbot.addWidget(q)
    rows, cols = np.array([0, 0, 1]), np.array([0, 1, 0])
    q.set_data(rows, cols, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), (2, 2))
    x, _y = q.arrows.getData()
    assert len(x) == 4  # two shafts, none for the zero vector
    assert q.heads.path().elementCount() == 2 * 3  # two triangles (3 points each)
    assert len(q.bases.data) == 3 and len(q.segments) == 6
