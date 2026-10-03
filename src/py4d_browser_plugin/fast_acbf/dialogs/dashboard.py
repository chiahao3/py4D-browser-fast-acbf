"""Advanced dashboard dialog for fast-acbf.

The views (right side) are ported from ptydy's fast-acbf app (its dashboard, views and
image panes, moved from PySide6 to PyQt5), so the dashboard looks and behaves like it;
keep them in step when either changes.

Left panel, the *pipeline* (shown or hidden with *Pipeline* in the top row or Ctrl+B):
calibration summary, per-aberration values and orientation with *Update Preview*,
automated refinement, the refinement history (read-only) and *Return to py4D-browser*.

Right side, under rows of buttons like pane controls:

- **Mode** tcBF / acBF and **Frame** scan / detector of the reconstruction
- **Show** what the second view holds: probe amplitude, probe intensity, complex probe
  (hue = phase), the aberration surface χ (gray modulo 2π by default), its gradient ∇χ,
  or the vBF image shifts as arrows
- **Depth** 2D / 3D: in 3D a stack of reconstructions (and probes) at *Slices* values of
  C10, *Step* Å apart around the current C10, is computed once and cached; the slice
  slider steps through it, and *Ortho View* next to it shows xz / yz sections

Every view has the pane controls underneath (*Controls* in the top row hides them).
In 3D every side view follows the slice (probe, χ, ∇χ and shifts at that C10), and each
pane's *B/C Range* row picks one brightness / contrast range for the whole stack
(default: the Auto percentiles over all slices together) or one per slice; arrows keep
one length scale for the stack, so they visibly grow away from focus. The ortho view
shows the chosen probe view (|ψ|, |ψ|² or ψ).

Frames of χ and the shifts (checked numerically against fast-acbf): both are drawn on
the raw detector grid. χ at a pixel is the aperture's χ at that pixel's k *after* the
orientation (flips, transpose; in the scan frame the coefficients are rotated too), i.e.
what the probe sees. The shift arrows sit at the recorded pixels (before any flip or
rotation), but their vectors are in the output frame's axes, ``s = R F ∇χ_det / 2π``
(``F`` the flips and transpose, ``R`` the scan rotation in the scan frame). The *Axes*
row of the shift view undoes ``R F`` so the arrows follow the detector grid (then they
are parallel to ∇χ of the detector-frame χ).

Images rotated into the detector frame have a fill outside the data; it is left out of
the display range (``ImagePane(ignore_fill=True)``).
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import FastAcbfConfig
from ..solver_job import (
    AutoTuneJob,
    DepthStackJob,
    PreviewJob,
    RefineAberrationsJob,
    RefineDefocusJob,
    RefineFlipsJob,
    RefineScanRotationJob,
)
from ._widgets import AberrationForm, OrientationForm
from .image_pane import (
    COLORMAPS,
    CYCLIC,
    GRAY_2PI,
    ImagePane,
    Segmented,
    save_png,
    save_tiff,
    to_qimage,
    tool_button,
)
from .views import OrthoView, QuiverView

SIDE_VIEWS = {"probe_amp": "Probe |ψ|", "probe_int": "Probe |ψ|²",
              "probe_complex": "Probe ψ", "chi": "χ", "grad_chi": "∇χ", "shifts": "Shifts"}
SIDE_TIPS = {"probe_amp": "Probe amplitude in real space",
             "probe_int": "Probe intensity in real space",
             "probe_complex": "Complex probe in real space (brightness = amplitude, "
                              "hue = phase)",
             "chi": "Aberration surface χ (rad) of the probe-forming aperture, over the "
                    "bright-field disk: each detector pixel shows χ at its k after the "
                    "orientation (flips, transpose, rotation), i.e. what the probe sees",
             "grad_chi": "Gradient of the χ image (rad·Å) as arrows, on the detector grid's "
                         "axes",
             "shifts": "Image shift of each bright-field pixel's vBF image, drawn at the "
                       "recorded detector pixel (before flips and rotation); the Axes row "
                       "picks the axes of the vectors"}
IMAGE_SIDE_VIEWS = ("probe_amp", "probe_int", "probe_complex", "chi")
QUIVER_SIDE_VIEWS = ("grad_chi", "shifts")
SHIFT_AXES = {"frame": "Output frame", "grid": "Detector grid"}
SHIFT_AXES_TIPS = {"frame": "Vectors in the axes of the output frame (what the "
                            "reconstruction shifts each vBF image by)",
                   "grid": "Vectors on the detector grid's axes: flips, transpose and "
                           "rotation undone (parallel to ∇χ of the detector-frame χ)"}
RANGE_TIPS = {"stack": "One brightness / contrast range for every slice: the Auto "
                       "percentiles of all slices together",
              "slice": "Each slice gets its own brightness / contrast range"}
PROBE_VIEWS = ("probe_amp", "probe_int", "probe_complex")
STEP_LABELS = {
    "manual": "manual",
    "auto_tune": "Refine All Params",
    "refine_defocus": "Refine Defocus",
    "refine_scan_rotation": "Refine Scan Rotation",
    "refine_flips": "Refine Flips",
    "refine_aberrations": "Refine Aberrations",
    "simple_menu_reconstruct": "Simple Menu",
    "optimize_orientation": "Optimize Orientation",
}
ROW_LABEL_WIDTH = 52
MODE_FRAME_GAP = 28
SLICE_SLIDER_WIDTH = 240
VIEW_MIN_WIDTH = 380
"""Views never get narrower; their minimum is fixed when first shown (a long cursor
readout used to widen them while hovering)."""


def _label(text: str, width: int | None = None) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("paneLabel")
    if width:
        lab.setFixedWidth(width)
    return lab


def _row(label: str, *widgets) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(6)
    if label:
        row.addWidget(_label(label, ROW_LABEL_WIDTH))
    for w in widgets:
        row.addWidget(w)
    row.addStretch(1)
    return row


def finite_box(image: np.ndarray, margin: int = 2) -> tuple[slice, slice]:
    """The bounding box of the finite values of ``image`` (the bright-field disk of χ),
    plus ``margin`` pixels."""
    rows, cols = np.nonzero(np.isfinite(image))
    if rows.size == 0:
        return slice(None), slice(None)
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + 1 + margin, image.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + 1 + margin, image.shape[1])
    return slice(r0, r1), slice(c0, c1)


def nan_gradient(image: np.ndarray, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(d/drow, d/dcol)`` of ``image`` with NaN outside the data: central differences
    where both neighbours are finite, one-sided at the rim, NaN where neither is."""
    a = np.asarray(image, dtype=np.float64)
    out = []
    for axis in (0, 1):
        fwd = np.full_like(a, np.nan)
        bwd = np.full_like(a, np.nan)
        hi = [slice(None)] * 2
        lo = [slice(None)] * 2
        hi[axis], lo[axis] = slice(1, None), slice(None, -1)
        fwd[tuple(lo)] = a[tuple(hi)] - a[tuple(lo)]
        bwd[tuple(hi)] = a[tuple(hi)] - a[tuple(lo)]
        both = np.isfinite(fwd) & np.isfinite(bwd)
        g = np.where(both, 0.5 * (fwd + bwd), np.where(np.isfinite(fwd), fwd, bwd))
        out.append(g / spacing)
    return out[0], out[1]


def orientation_matrix(cfg: FastAcbfConfig, frame: str) -> np.ndarray:
    """``M`` with ``v_frame = M v_grid`` for (x, y) column vectors: fast-acbf's flipud →
    fliplr → transpose on k, then (scan frame only) the scan rotation."""
    m = np.eye(2)
    if cfg.flipud:
        m = np.diag([1.0, -1.0]) @ m
    if cfg.fliplr:
        m = np.diag([-1.0, 1.0]) @ m
    if cfg.transpose:
        m = np.array([[0.0, 1.0], [1.0, 0.0]]) @ m
    if frame == "scan" and cfg.rotation_deg:
        t = np.deg2rad(float(cfg.rotation_deg))
        m = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ m
    return m


def shifts_on_detector_grid(shifts_yx: np.ndarray, cfg: FastAcbfConfig,
                            frame: str) -> np.ndarray:
    """vBF shifts (``(n, 2)`` as y, x in ``frame``) with the orientation undone: the
    same vectors on the raw detector grid's axes (= ∇χ_det / 2π per pixel)."""
    s = np.asarray(shifts_yx, dtype=np.float64)[:, ::-1]  # (x, y) rows
    grid = s @ orientation_matrix(cfg, frame)  # rows: (M^T v)^T = v^T M
    return grid[:, ::-1]


def restorable(config: FastAcbfConfig) -> dict:
    """The part of a configuration that a history row brings back."""
    return {"aberrations": dict(config.aberrations), "rotation_deg": float(config.rotation_deg),
            "flipud": bool(config.flipud), "fliplr": bool(config.fliplr),
            "transpose": bool(config.transpose)}


class FastAcbfDashboard(QDialog):
    run_requested = pyqtSignal(object)
    config_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    config_changed = pyqtSignal(object)

    def __init__(self, config: FastAcbfConfig, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle("fast-acbf: Advanced Dashboard")
        self.resize(1400, 850)
        self.config = config.copy()
        self.history: list[dict] = []
        self.result: dict | None = None
        self.stack: dict | None = None
        self.stack_key: tuple | None = None
        self._pending_stack_key: tuple | None = None
        self.side_view = "probe_amp"
        self.depth_mode = "2D"
        self.side_cmaps = {"chi": GRAY_2PI}
        """Colour map of each image side view (the pane keeps one; χ starts as gray
        modulo 2π)."""
        self.stack_range = {"recon": "stack", "side": "stack"}
        """3D display range of each pane: one for the stack, or one per slice."""
        self.shift_axes = "frame"
        self._side_ortho_key: str | None = None
        self._footers_aligned = False
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
        self.panel_action = QAction("Pipeline", self)
        self.panel_action.setCheckable(True)
        self.panel_action.setChecked(True)
        self.panel_action.setShortcut(QKeySequence("Ctrl+B"))
        self.panel_action.setToolTip("Show or hide the pipeline on the left: calibration, "
                                     "aberrations, refinement and history (Ctrl+B)")
        self.panel_action.toggled.connect(self.set_panel_visible)
        self.addAction(self.panel_action)
        self.controls_action = QAction("Controls", self)
        self.controls_action.setCheckable(True)
        self.controls_action.setChecked(True)
        self.controls_action.setToolTip("Show the statistics and display controls under "
                                        "each view (hide them for the largest images)")
        self.controls_action.toggled.connect(self.set_controls_visible)

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(self.splitter)
        self.left_panel = QScrollArea()
        self.left_panel.setWidgetResizable(True)
        self.left_panel.setMinimumWidth(340)
        inner = QWidget()
        self.left_panel.setWidget(inner)
        self._build_left(QVBoxLayout(inner))
        # as wide as its content: py4D-browser's native style is larger than ptydy's, and
        # 340 px clipped the buttons and the calibration line
        self.left_panel.setMinimumWidth(max(340, inner.sizeHint().width() + 24))
        self.splitter.addWidget(self.left_panel)
        right = QWidget()
        self._build_right(QVBoxLayout(right))
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setChildrenCollapsible(False)

    def _build_left(self, left: QVBoxLayout) -> None:
        left.setContentsMargins(8, 8, 8, 8)
        calib = QGroupBox("Global Calibration")
        calib_layout = QVBoxLayout(calib)
        self.calib_label = QLabel()
        self.calib_label.setWordWrap(True)
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
        # min-height here too: a style sheet's min-height beats the widget's own minimum
        self.apply_btn.setStyleSheet(
            "QPushButton { padding: 6px 10px; min-height: 34px; }"
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
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(170)
        left.addWidget(self.table, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        close_btn = QPushButton("Return to py4D-browser")
        close_btn.clicked.connect(self.close)
        left.addWidget(close_btn)

    def _build_right(self, right: QVBoxLayout) -> None:
        right.setContentsMargins(6, 6, 6, 0)
        right.setSpacing(5)
        self.mode_buttons = Segmented([("tcBF", "tcBF"), ("acBF", "acBF")])
        self.mode_buttons.picked.connect(self._mode_changed)
        self.frame_buttons = Segmented([("Scan", "scan"), ("Detector", "detector")])
        self.frame_buttons.setToolTip("Frame of the reconstruction, probe, χ and shifts "
                                      "(detector = PtyRAD's frame)")
        self.frame_buttons.picked.connect(self._output_frame_changed)
        self.top_row = _row("Mode", self.mode_buttons)
        self.top_row.insertSpacing(self.top_row.count() - 1, MODE_FRAME_GAP)
        self.top_row.insertWidget(self.top_row.count() - 1, _label("Frame"))
        self.top_row.insertWidget(self.top_row.count() - 1, self.frame_buttons)
        # ptydy puts these two in its ribbon; py4D-browser has none: end of the top row
        self.panel_btn = tool_button("", action=self.panel_action)
        self.controls_btn = tool_button("", action=self.controls_action)
        for b in (self.panel_btn, self.controls_btn):
            b.setObjectName("chipButton")
            self.top_row.addWidget(b)
        right.addLayout(self.top_row)

        self.side_buttons = Segmented(((label, key) for key, label in SIDE_VIEWS.items()),
                                      tips=SIDE_TIPS)
        self.side_buttons.picked.connect(self.set_side_view)
        right.addLayout(_row("Show", self.side_buttons))

        self.depth_buttons = Segmented([("2D", "2D"), ("3D", "3D")])
        self.depth_buttons.setToolTip("3D: a stack of reconstructions over C10 (depth)")
        self.depth_buttons.picked.connect(self.set_depth_mode)
        self.slices_spin = QSpinBox()
        self.slices_spin.setRange(3, 201)
        self.slices_spin.setSingleStep(2)
        self.slices_spin.setValue(11)
        self.slices_spin.setToolTip("Number of slices, centred on the current C10")
        self.step_spin = QDoubleSpinBox()
        self.step_spin.setRange(0.1, 1e5)
        self.step_spin.setDecimals(1)
        self.step_spin.setValue(20.0)
        self.step_spin.setSuffix(" Å")
        self.step_spin.setToolTip("C10 step between slices")
        for spin in (self.slices_spin, self.step_spin):
            spin.valueChanged.connect(lambda *_: self._update_stack_label())
        self.compute_btn = QPushButton("Compute stack")
        self.compute_btn.clicked.connect(self.compute_stack)
        self.stack_label = QLabel("")
        self.stack_label.setObjectName("paneInfo")
        right.addLayout(_row("Depth", self.depth_buttons, _label("Slices"), self.slices_spin,
                             _label("Step"), self.step_spin, self.compute_btn,
                             self.stack_label))

        # the slice row (3D with a stack only) holds the display choices of the stack
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setFixedWidth(SLICE_SLIDER_WIDTH)
        self.slice_spin = QSpinBox()
        self.slice_slider.valueChanged.connect(self.set_slice)
        self.slice_spin.valueChanged.connect(self.set_slice)
        self.ortho_btn = tool_button("Ortho View", checkable=True,
                                     tip="Orthogonal sections (xz below, yz beside) through "
                                         "a point; drag the round handles to move it")
        self.ortho_btn.setObjectName("chipButton")
        self.ortho_btn.toggled.connect(lambda _on: self._show_depth())
        self.slice_label = QLabel("")
        self.slice_label.setObjectName("paneInfo")
        self.slice_label.setMinimumWidth(220)
        self.slice_row = QWidget()
        sl = _row("Slice", self.slice_slider, self.slice_spin, self.ortho_btn,
                  self.slice_label)
        sl.setContentsMargins(0, 0, 0, 0)
        self.slice_row.setLayout(sl)
        right.addWidget(self.slice_row)

        views = QSplitter(Qt.Orientation.Horizontal)
        self.recon_pane = ImagePane("Reconstruction", ignore_fill=True, percentiles=(0.5, 99.5))
        self.recon_ortho = OrthoView("Reconstruction")
        self.recon_stack = QStackedWidget()
        self.recon_stack.addWidget(self.recon_pane)
        self.recon_stack.addWidget(self.recon_ortho)
        self.side_pane = ImagePane("Probe", ignore_fill=True, percentiles=(0.5, 99.5),
                                   colormaps=(*COLORMAPS, GRAY_2PI, CYCLIC))
        self.range_buttons = {}
        for key, pane in (("recon", self.recon_pane), ("side", self.side_pane)):
            seg = Segmented([("Stack", "stack"), ("Slice", "slice")], tips=RANGE_TIPS)
            seg.set_current(self.stack_range[key])
            seg.picked.connect(lambda mode, k=key: self.set_stack_range(k, mode))
            self.range_buttons[key] = pane.add_control(seg, "B/C Range")
        self.quiver = QuiverView()
        self.axes_buttons = Segmented(((label, key) for key, label in SHIFT_AXES.items()),
                                      tips=SHIFT_AXES_TIPS)
        self.axes_buttons.set_current(self.shift_axes)
        self.axes_buttons.picked.connect(self.set_shift_axes)
        self.quiver.add_control(self.axes_buttons, "Axes")
        self.side_ortho = OrthoView("Probe |ψ|")
        self.side_stack = QStackedWidget()
        for w in (self.side_pane, self.quiver, self.side_ortho):
            self.side_stack.addWidget(w)
        views.addWidget(self.recon_stack)
        views.addWidget(self.side_stack)
        views.setSizes([600, 400])
        views.setChildrenCollapsible(False)
        right.addWidget(views, 1)
        self.recon_ortho.z_changed.connect(self.set_slice)
        self.side_ortho.z_changed.connect(self.set_slice)
        for pane in (self.recon_pane, self.side_pane, self.recon_ortho, self.side_ortho):
            pane.exportRequested.connect(lambda fmt, p=pane: self._export(p, fmt))
            pane.copyRequested.connect(lambda p=pane: self._copy(p))
        self.side_buttons.set_current(self.side_view)
        self.depth_buttons.set_current(self.depth_mode)
        self._show_depth()

    def views(self) -> tuple:
        """Every view with a controls footer."""
        return (self.recon_pane, self.side_pane, self.quiver, self.recon_ortho,
                self.side_ortho)

    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        if not self._footers_aligned:
            self.align_footers()

    def align_footers(self) -> None:
        """Give every view's footer the height of the tallest with all its rows shown,
        so switching views, 2D / 3D (rows that only apply to one are hidden) or a
        complex probe (no histogram) never changes the image height."""
        rows = [r for v in self.views() for r in v.control_row_widgets()]
        hidden = [r for r in rows if r.isHidden()]
        for r in hidden:
            r.show()
        height = max(v.footer.sizeHint().height() for v in self.views())
        for r in hidden:
            r.hide()
        # fixed, not only a minimum as in ptydy: with py4D-browser's native style (no
        # style sheet) an image pane's footer otherwise ends up a pixel taller than this
        for v in self.views():
            v.footer.setFixedHeight(height)
        # a fixed minimum width (the content's, at least VIEW_MIN_WIDTH): nothing shown
        # later (e.g. a longer cursor readout) may widen a view
        for w in (self.recon_stack, self.side_stack):
            w.setMinimumWidth(max(VIEW_MIN_WIDTH, w.minimumSizeHint().width()))
        self._footers_aligned = True

    # ---- left panel (the pipeline) and the controls

    def set_panel_visible(self, on: bool) -> None:
        self.left_panel.setVisible(on)
        self.panel_action.blockSignals(True)
        self.panel_action.setChecked(on)
        self.panel_action.blockSignals(False)

    def set_controls_visible(self, on: bool) -> None:
        for v in self.views():
            v.set_controls_visible(on)
        self.controls_action.blockSignals(True)
        self.controls_action.setChecked(on)
        self.controls_action.blockSignals(False)

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
        self.apply_btn.setFocus(Qt.FocusReason.OtherFocusReason)

    def _focus_global_calibration(self) -> None:
        self.edit_calib_btn.setFocus(Qt.FocusReason.OtherFocusReason)

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
        self.mode_buttons.set_current(self.config.mode)
        self.frame_buttons.set_current(self.config.output_frame)
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
        self.aberration_form.set_max_order(int(self.config.max_order))
        self.aberration_form.set_values(self.config.aberrations)
        self._update_stack_label()

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    # ---- results

    def set_result(self, result: dict) -> None:
        """Show a finished job: the reconstruction, the side view and a history row (a
        depth-stack job fills the stack instead)."""
        if "stack" in result:
            self.set_stack(result)
            return
        self.result = result
        self.add_history(result)
        self._show_depth()
        self._update_stack_label()

    def add_history(self, result: dict) -> None:
        cfg: FastAcbfConfig = result.get("config", self.config)
        entry = {"label": result.get("label") or STEP_LABELS.get(result["command"],
                                                                 result["command"]),
                 "state": restorable(cfg), "metric_text": result.get("metric_text", "")}
        self.history.append(entry)
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            entry["label"],
            cfg.aberrations.get("C10", 0.0),
            cfg.aberrations.get("C12a", 0.0),
            cfg.aberrations.get("C12b", 0.0),
            cfg.rotation_deg,
            entry["metric_text"],
        ]
        for col, value in enumerate(values):
            text = f"{value:.5g}" if isinstance(value, float) else str(value)
            self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.selectRow(row)

    def reset_for_new_dataset(self, config: FastAcbfConfig) -> None:
        """Clear dataset-specific history, previews and the depth stack, then show fresh
        settings."""
        self.aberration_form.zero_all()
        self.set_config(config)
        self.table.setRowCount(0)
        self.history = []
        self.result = None
        self.stack = self.stack_key = self._pending_stack_key = None
        for pane in (self.recon_pane, self.side_pane):
            pane.clear()
        for view in (self.quiver, self.recon_ortho, self.side_ortho):
            view.clear()
        self.depth_mode = "2D"
        self.depth_buttons.set_current("2D")
        self._show_depth()

    # ---- side view (probe, χ, shifts)

    def set_side_view(self, key: str) -> None:
        previous, self.side_view = self.side_view, key
        self.side_buttons.set_current(key)
        if previous in IMAGE_SIDE_VIEWS:
            self.side_cmaps[previous] = self.side_pane.cmap
        if key in IMAGE_SIDE_VIEWS:
            cmap = self.side_cmaps.get(key, "gray")
            if cmap != self.side_pane.cmap:
                self.side_pane.set_colormap(cmap)
        self._show_side()

    def set_shift_axes(self, axes: str) -> None:
        self.shift_axes = axes
        self.axes_buttons.set_current(axes)
        self._show_side()

    def set_stack_range(self, pane: str, mode: str) -> None:
        """``pane`` ``"recon"`` or ``"side"``: one display range for the stack or per
        slice (3D)."""
        self.stack_range[pane] = mode
        self.range_buttons[pane].set_current(mode)
        self._show_depth()

    def _in_stack(self) -> bool:
        return self.depth_mode == "3D" and self.stack is not None

    def _chi(self) -> np.ndarray | None:
        """χ of the current slice in 3D (when the stack has it), else the result's."""
        if self._in_stack() and "chi_stack" in self.stack:
            return self.stack["chi_stack"][self._slice()]
        return self.result.get("chi")

    def _shifts(self) -> np.ndarray | None:
        if self._in_stack() and "shifts_stack" in self.stack:
            return self.stack["shifts_stack"][self._slice()]
        return self.result.get("shifts_yx")

    def _stack_longest(self, key: str) -> float | None:
        """The longest arrow over every slice (3D), so the arrows keep one scale and
        grow or shrink as the slider moves through defocus."""
        if not self._in_stack():
            return None
        cache = f"longest_{key}"
        if cache not in self.stack:
            longest = None
            if key == "shifts" and "shifts_stack" in self.stack:
                longest = float(np.hypot(*np.moveaxis(self.stack["shifts_stack"], -1, 0)).max())
            elif key == "grad_chi" and "chi_stack" in self.stack:
                size = self.result.get("chi_pixel_size", 1.0)
                longest = float(max(np.nanmax(np.hypot(*nan_gradient(c, size)))
                                    for c in self.stack["chi_stack"]))
            self.stack[cache] = longest
        return self.stack[cache]

    def _show_quiver(self, key: str, frame: str) -> None:
        res = self.result
        cfg = res.get("config", self.config)
        chi = self._chi()
        longest = self._stack_longest(key)
        suffix = self._slice_suffix()
        if key == "grad_chi":
            self.quiver.set_control_visible(self.axes_buttons, False)
            if chi is None:
                self.quiver.clear()
                return
            gy, gx = nan_gradient(chi, res.get("chi_pixel_size", 1.0))
            rows, cols = np.nonzero(np.isfinite(gy) & np.isfinite(gx))
            self.quiver.set_data(rows, cols, np.column_stack([gy[rows, cols], gx[rows, cols]]),
                                 chi.shape, units="rad·Å",
                                 note="axes of the detector grid, like the χ image",
                                 title=f"∇χ of the χ image ({frame} frame){suffix}",
                                 longest=longest)
            return
        self.quiver.set_control_visible(self.axes_buttons, True)
        shifts = self._shifts()
        if shifts is None:
            self.quiver.clear()
            return
        shape = chi.shape if chi is not None else (
            int(res["bf_rows"].max()) + 1, int(res["bf_cols"].max()) + 1)
        if self.shift_axes == "grid":
            vectors = shifts_on_detector_grid(shifts, cfg, frame)
            note = "vectors on the detector grid's axes (orientation undone)"
        else:
            vectors = shifts
            note = f"vectors in the {frame} frame's axes"
        self.quiver.set_data(res["bf_rows"], res["bf_cols"], vectors, shape, note=note,
                             title="vBF image shifts at the recorded detector pixels" + suffix,
                             longest=longest)

    def _probe(self) -> np.ndarray | None:
        """The complex probe to show: the current slice's in 3D, else the result's."""
        if self.depth_mode == "3D" and self.stack is not None:
            return self.stack["probe_stack"][self._slice()]
        if self.result is not None:
            probe = self.result.get("probe_complex")
            return probe if probe is not None else self.result.get("probe")
        return None

    def _show_side(self) -> None:
        key = self.side_view
        frame = self.config.output_frame
        if self.result is None:
            return
        if key in QUIVER_SIDE_VIEWS:
            self.side_stack.setCurrentWidget(self.quiver)
            self._show_quiver(key, frame)
            return
        if key in PROBE_VIEWS and self._ortho_on():
            self._sync_side_ortho(key)
            self.side_stack.setCurrentWidget(self.side_ortho)
            return
        self.side_stack.setCurrentWidget(self.side_pane)
        one_range = self._in_stack() and self.stack_range["side"] == "stack"
        if key == "chi":
            chi = self.result.get("chi")
            if chi is not None:
                box = finite_box(chi)  # the disk: the same for every slice
                shown = self._chi()[box]
                levels = None
                if one_range and "chi_stack" in self.stack:
                    if "chi_crop" not in self.stack:
                        self.stack["chi_crop"] = self.stack["chi_stack"][(slice(None), *box)]
                    levels = self.stack["chi_crop"]
                self.side_pane.set_level_stack(levels)
                self.side_pane.set_image(shown, reset=self._new_shape(shown),
                                         pixel_size=self.result.get("chi_pixel_size", 1.0),
                                         units="A^-1",
                                         title=f"χ (rad), aperture plane ({frame} frame)"
                                               + self._slice_suffix())
                self.side_pane.info.setText("orientation applied to k")
            return
        probe = self._probe()
        if probe is None:
            return
        shown = self._probe_view(key, probe)
        self.side_pane.set_level_stack(self._probe_level_stack(key) if one_range else None)
        size = self.result.get("probe_pixel_size")
        self.side_pane.set_image(
            shown, reset=self._new_shape(shown), pixel_size=size or 1.0,
            units="A" if size else "pixels",
            title=f"{SIDE_VIEWS[key]} ({frame} frame)" + self._slice_suffix())
        self.side_pane.info.setText("")

    @staticmethod
    def _probe_view(key: str, probe) -> np.ndarray:
        return {"probe_amp": np.abs, "probe_int": lambda p: np.abs(p) ** 2,
                "probe_complex": lambda p: p}[key](np.asarray(probe))

    def _probe_level_stack(self, key: str) -> np.ndarray:
        """|ψ|, |ψ|² or ψ of every slice (cached with the stack)."""
        cache_key = f"levels_{key}"
        if cache_key not in self.stack:
            self.stack[cache_key] = self._probe_view(key, self.stack["probe_stack"])
        return self.stack[cache_key]

    def _sync_side_ortho(self, key: str) -> None:
        """Give the probe's ortho view the stack of the chosen probe view."""
        if self._side_ortho_key == key:
            return
        self._side_ortho_key = key
        self.side_ortho.title.setText(SIDE_VIEWS[key])
        self.side_ortho.set_stack(self._probe_level_stack(key), self.stack["stack_step"],
                                  self.stack["probe_pixel_size"])
        self.side_ortho.set_z(self._slice())

    def _new_shape(self, image) -> bool:
        return self.side_pane.raw is None or self.side_pane.raw.shape != np.shape(image)

    # ---- depth stack

    def stack_request_key(self) -> tuple:
        """What a cached stack depends on: aberrations, orientation, mode, frame,
        upscale, slices and step."""
        c = self.config
        return (tuple(sorted((k, round(float(v), 6)) for k, v in c.aberrations.items())),
                round(float(c.rotation_deg), 6), bool(c.flipud), bool(c.fliplr),
                bool(c.transpose), c.mode, c.output_frame, float(c.upscale),
                int(self.slices_spin.value()), round(float(self.step_spin.value()), 6))

    def stack_is_current(self) -> bool:
        return self.stack is not None and self.stack_key == self.stack_request_key()

    def compute_stack(self) -> None:
        self._pending_stack_key = self.stack_request_key()
        self.run_requested.emit(DepthStackJob(n_slices=int(self.slices_spin.value()),
                                              step=float(self.step_spin.value())))

    def set_stack(self, result: dict) -> None:
        self.stack = {k: result[k] for k in ("stack", "stack_c10", "probe_stack",
                                             "stack_step", "chi_stack", "shifts_stack")
                      if k in result}
        self.stack_key = self._pending_stack_key or self.stack_request_key()
        self._pending_stack_key = None
        cfg = result.get("config", self.config)
        self.stack["pixel_size"] = cfg.output_pixel_size_angstrom()
        self.stack["probe_pixel_size"] = result.get("probe_pixel_size", 1.0)
        n = len(self.stack["stack"])
        for w in (self.slice_slider, self.slice_spin):
            w.blockSignals(True)
            w.setRange(0, n - 1)
            w.setValue(n // 2)
            w.blockSignals(False)
        self.recon_pane.set_level_stack(None)  # a new stack: new ranges
        self.recon_ortho.set_stack(self.stack["stack"], self.stack["stack_step"],
                                   self.stack["pixel_size"])
        self._side_ortho_key = None  # filled with the chosen probe view when shown
        if self.result is None:
            self.result = result
        self.depth_mode = "3D"
        self.depth_buttons.set_current("3D")
        self._show_depth()
        self._update_stack_label()

    def set_depth_mode(self, mode: str) -> None:
        self.depth_mode = mode
        self.depth_buttons.set_current(mode)
        if mode == "3D" and not self.stack_is_current() and self.result is not None:
            self.compute_stack()  # the stack comes back through set_stack
        self._show_depth()

    def _slice(self) -> int:
        return int(self.slice_spin.value())

    def _ortho_on(self) -> bool:
        return self.depth_mode == "3D" and self.stack is not None and self.ortho_btn.isChecked()

    def _slice_suffix(self) -> str:
        if self.depth_mode != "3D" or self.stack is None:
            return ""
        return f", slice {self._slice()}"

    def set_slice(self, z: int) -> None:
        for w in (self.slice_slider, self.slice_spin):
            if w.value() != z:
                w.blockSignals(True)
                w.setValue(int(z))
                w.blockSignals(False)
        self._show_depth()

    def _show_depth(self) -> None:
        """Put the 2D result or the current slice of the stack into the views."""
        in_3d = self.depth_mode == "3D"
        for w in (self.slices_spin, self.step_spin, self.compute_btn):
            w.setEnabled(in_3d)
        have_stack = in_3d and self.stack is not None
        self.slice_row.setVisible(have_stack)
        for key, pane in (("recon", self.recon_pane), ("side", self.side_pane)):
            pane.set_control_visible(self.range_buttons[key], have_stack)
        if not have_stack:
            self.recon_stack.setCurrentWidget(self.recon_pane)
            self.recon_pane.set_level_stack(None)
            image = None if self.result is None else self.result.get("image")
            if image is not None:
                cfg = self.result.get("config", self.config)
                image = np.asarray(image)
                reset = self.recon_pane.raw is None or self.recon_pane.raw.shape != image.shape
                self.recon_pane.set_image(
                    image, reset=reset, pixel_size=cfg.output_pixel_size_angstrom(), units="A",
                    title=f"{self.result.get('mode', cfg.mode)} ({cfg.output_frame} frame)")
            self._show_side()
            return
        z = self._slice()
        c10 = self.stack["stack_c10"]
        centre = c10[len(c10) // 2]
        self.slice_label.setText(f"C10 = {c10[z]:.1f} Å  (Δ {c10[z] - centre:+.1f} Å)")
        if self._ortho_on():
            self.recon_stack.setCurrentWidget(self.recon_ortho)
            self.recon_ortho.set_z(z)
            if self._side_ortho_key is not None:
                self.side_ortho.set_z(z)
        else:
            self.recon_stack.setCurrentWidget(self.recon_pane)
            img = self.stack["stack"][z]
            reset = self.recon_pane.raw is None or self.recon_pane.raw.shape != img.shape
            one_range = self.stack_range["recon"] == "stack"
            if (self.recon_pane.level_stack is not None) != one_range:
                self.recon_pane.set_level_stack(self.stack["stack"] if one_range else None)
            self.recon_pane.set_image(img, reset=reset, pixel_size=self.stack["pixel_size"],
                                      units="A", title=f"{self.config.mode} depth stack "
                                                       f"({self.config.output_frame} frame)")
        self._show_side()

    def _update_stack_label(self) -> None:
        if self.stack is None:
            self.stack_label.setText("")
        elif self.stack_is_current():
            self.stack_label.setText(f"cached: {len(self.stack['stack'])} slices")
        else:
            self.stack_label.setText("from an earlier state: Compute stack to update")

    # ---- export

    def _export(self, pane: ImagePane, fmt: str) -> None:
        if pane.raw is None:
            return
        ext = {"tiff": "tif", "png": "png"}[fmt]
        path, _ = QFileDialog.getSaveFileName(self, "Export", f"fast_acbf.{ext}", f"*.{ext}")
        if not path:
            return
        if fmt == "tiff":
            save_tiff(path, pane.raw, pane.pixel_size, pane.units)
        else:
            save_png(path, pane.displayed())

    def _copy(self, pane: ImagePane) -> None:
        from PyQt5.QtWidgets import QApplication

        img = pane.displayed()
        if img is not None:
            QApplication.clipboard().setImage(to_qimage(img))
