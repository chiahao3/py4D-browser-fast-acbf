"""One image view of the advanced dashboard, with its display controls next to it.

Ported from ptydy (``ptydy.gui.widgets.image_pane``, with its scale bar and
``tool_button``) to PyQt5 so the dashboard shows the same views as ptydy's fast-acbf app;
keep the two in step when either changes. ptydy's theme colours are the constants
``ACCENT`` / ``MUTED`` / ``IMAGE_BG`` here (py4D-browser has no theme).

Layout, top to bottom:

- header: title, a short description (e.g. the scan position), extra header tools, and
  *Fit* / *Copy* / *Export*
- the image (row-major, row 0 at the top) with a scale bar
- statistics of the image (mean, min, max, std) and the value under the cursor
- footer: the pane's own controls (e.g. the detector shape, added with
  :meth:`add_control`), colour map, intensity scaling, *Auto* and a small histogram whose
  handles set the display levels

The pane keeps the *unscaled* image it was given (``raw``); scaling (linear / log /
square root), colour map and levels only change what is drawn. While *Auto* is on, the
levels are set at the chosen percentiles on every update; dragging the histogram handles
turns it off. With :meth:`ImagePane.set_level_stack` the percentiles are taken over a
whole stack instead (one range for every slice; for a complex stack, one magnitude
range). The wrapped colour maps (offered to panes that list them, e.g. for phases), gray
or cyclic, draw the value modulo 2π over a fixed 0 – 2π range.

The footer keeps its height when the histogram is hidden (complex images), and the
cursor readout has a fixed width, so neither makes the pane jump.

An image can be *placed*: drawn through an affine map from its pixels to view
coordinates instead of one view unit per pixel (``set_image(..., transform=M)`` with
``M`` 2 × 3, ``(x, y)_view = M @ (col, row, 1)``, or ``rect=(x, y, w, h)`` for a plain
rectangle); the scale bar then follows the image's own pixel size. The dashboard does not
place images; it is kept so the pane stays the same as ptydy's.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .imaging import (
    Scaling,
    complex_to_rgb,
    fill_mask,
    image_stats,
    percentile_levels,
    scale,
    stack_levels,
)

ACCENT = "#2a82da"
"""Accent colour (arrows, crosshairs; the blue of the Update Preview button)."""
MUTED = "#8a8f98"
"""Statistics labels."""
IMAGE_BG = "k"
"""Background behind images and histograms (pyqtgraph's default black)."""

COLORMAPS = ("gray", "inferno", "magma", "viridis", "cividis", "turbo")
CYCLIC = "cyclic"
"""Colour map for phases: the value modulo 2π on a cyclic map (CET-C2), levels 0 – 2π."""
GRAY_2PI = "gray_2pi"
"""The value modulo 2π in gray, levels 0 – 2π."""
WRAPPED = (GRAY_2PI, CYCLIC)
CMAP_LABELS = {CYCLIC: "Cyclic 2π", GRAY_2PI: "Gray 2π"}
TWO_PI = 2 * np.pi
READOUT_TEMPLATE = "[y, x] = [0000, 0000]   -0.0000e-00∠-000°"
"""Longest cursor readout; its width is reserved so the pane never changes width."""
SCALING_LABELS = {Scaling.LINEAR: "Linear", Scaling.LOG: "Log", Scaling.SQRT: "Sqrt"}
PERCENTILES = ((0.0, 100.0), (0.1, 99.9), (0.5, 99.5), (1.0, 99.0), (2.0, 98.0), (5.0, 95.0))
HISTOGRAM_WIDTH = (120, 220)
"""Minimum and maximum; the histogram takes what the controls leave."""
HISTOGRAM_MIN_HEIGHT = 84
ROW_LABEL_WIDTH = 64


# ---- scale bar (ptydy.gui.widgets.scalebar) ----------------------------------------

NICE_STEPS = (1, 2, 5)


def nice_length(target: float) -> float:
    """The largest 1/2/5 × 10^n not above ``target`` (> 0)."""
    if target <= 0 or not np.isfinite(target):
        return 1.0
    exp = np.floor(np.log10(target))
    for step in reversed(NICE_STEPS):
        v = step * 10**exp
        if v <= target:
            return float(v)
    return float(10**exp)


def format_units(units: str) -> str:
    """ASCII calibration units as display text (``A^-1`` -> ``Å⁻¹``)."""
    table = {"A": "Å", "A^-1": "Å⁻¹", "nm^-1": "nm⁻¹", "um": "µm", "pixels": "px"}
    return table.get(units, units.replace("^-1", "⁻¹").replace("A", "Å"))


class CalibratedScaleBar(pg.ScaleBar):
    def __init__(self, pixel_size: float = 1.0, units: str = "pixels", width: int = 5,
                 fraction: float = 0.25):
        super().__init__(size=10, width=width, suffix="")
        self.pixel_size = float(pixel_size)
        self.units = str(units)
        self.fraction = fraction
        self._vb = None

    def setParentItem(self, parent) -> None:  # noqa: N802 (Qt naming)
        super().setParentItem(parent)
        vb = parent if isinstance(parent, pg.ViewBox) else None
        if vb is not None and vb is not self._vb:
            self._vb = vb
            vb.sigRangeChanged.connect(lambda *_: self.updateBar())

    def updateBar(self) -> None:  # noqa: N802
        vb = self._vb or self.parentItem()
        if vb is None or not isinstance(vb, pg.ViewBox):
            return super().updateBar()
        if self.units in ("", "pixels", "px") or not self.pixel_size > 0:
            self.hide()
            return
        (x0, x1), _ = vb.viewRange()
        length = nice_length(self.fraction * max(x1 - x0, 1.0) * self.pixel_size)
        self.size = length / self.pixel_size
        self.text.setText(f"{length:g} {format_units(self.units)}")
        self.show()
        super().updateBar()


# ---- tool button (ptydy.gui.widgets.ribbon) ----------------------------------------

def tool_button(text: str, slot: Callable | None = None, *, tip: str = "",
                menu: QtWidgets.QMenu | None = None, checkable: bool = False,
                action: QtWidgets.QAction | None = None) -> QtWidgets.QToolButton:
    b = QtWidgets.QToolButton()
    b.setObjectName("ribbonButton")
    b.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    if action is not None:
        b.setDefaultAction(action)
    else:
        b.setText(text)
        b.setCheckable(checkable)
    if tip:
        b.setToolTip(tip)
    if slot is not None:
        b.clicked.connect(lambda *_: slot())
    if menu is not None:
        b.setMenu(menu)
        b.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup
                       if slot is not None or action is not None
                       else QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    return b


def colormap(name: str) -> pg.ColorMap:
    if name == CYCLIC:
        return pg.colormap.get("CET-C2")
    if name in ("gray", GRAY_2PI):
        return pg.ColorMap([0.0, 1.0], [(0, 0, 0), (255, 255, 255)])
    return pg.colormap.get(name)


def _fmt(v: float) -> str:
    return "–" if not np.isfinite(v) else f"{v:.4g}"


def control_row(widget: QtWidgets.QWidget, label: str) -> QtWidgets.QWidget:
    """``label`` (fixed width, so rows line up) and ``widget`` as one footer row."""
    row = QtWidgets.QWidget()
    r = QtWidgets.QHBoxLayout(row)
    r.setContentsMargins(0, 0, 0, 0)
    r.setSpacing(6)
    lab = QtWidgets.QLabel(label)
    lab.setObjectName("paneLabel")
    lab.setFixedWidth(ROW_LABEL_WIDTH)
    r.addWidget(lab)
    r.addWidget(widget)
    r.addStretch(1)
    return row


class Segmented(QtWidgets.QWidget):
    """A row of exclusive toggle buttons (all choices visible, nothing to open)."""

    picked = QtCore.pyqtSignal(object)
    """The data of the button the user clicked."""

    def __init__(self, items, tips: dict | None = None, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict = {}
        items = list(items)
        for i, (label, data) in enumerate(items):
            b = QtWidgets.QToolButton()
            b.setObjectName("segment")
            b.setText(label)
            b.setCheckable(True)
            # never narrower than the label (a squeezed pane elided "Linear" to "L…r")
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                            QtWidgets.QSizePolicy.Policy.Fixed)
            b.setProperty("first", i == 0)
            b.setProperty("last", i == len(items) - 1)
            if tips and data in tips:
                b.setToolTip(tips[data])
            b.clicked.connect(lambda *_, d=data: self.picked.emit(d))
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[data] = b

    def set_current(self, data) -> None:
        if data in self.buttons:
            self.buttons[data].setChecked(True)

    def current(self):
        return next((d for d, b in self.buttons.items() if b.isChecked()), None)

    def set_item_enabled(self, data, on: bool) -> None:
        self.buttons[data].setEnabled(on)


class ImagePane(QtWidgets.QWidget):
    exportRequested = QtCore.pyqtSignal(str)
    """``"tiff"`` or ``"png"``: the user asked to save this pane."""
    copyRequested = QtCore.pyqtSignal()

    def __init__(self, title: str, parent=None, scaling: Scaling = Scaling.LINEAR,
                 cmap: str = "gray", ignore_fill: bool = False,
                 percentiles: tuple[float, float] = (0.1, 99.9),
                 colormaps: tuple[str, ...] = COLORMAPS):
        super().__init__(parent)
        self.colormaps = tuple(colormaps)
        self.setObjectName("imagePane")
        self.raw: np.ndarray | None = None
        self.valid: np.ndarray | None = None
        """Pixels that hold data (``None``: all); see ``ignore_fill``."""
        self.ignore_fill = ignore_fill
        """Leave a border-connected fill (rotated / padded output) out of the levels and
        statistics (:func:`.imaging.fill_mask`)."""
        self.scaling = scaling
        self.cmap = cmap
        self.percentiles = (float(percentiles[0]), float(percentiles[1]))
        self.autoscale = True
        self.level_stack: np.ndarray | None = None
        """Take the Auto levels over this stack (all slices) instead of the image."""
        self._stack_levels: dict = {}
        self.pixel_size = 1.0
        self.units = "pixels"
        self.transform: np.ndarray | None = None
        """2 × 3 map from image pixels ``(col, row, 1)`` to view ``(x, y)`` of a placed
        image; None: one view unit per pixel from (0, 0)."""
        self.placed_label = "scan"
        """Name of the view coordinates in the cursor readout of a placed image."""
        self._setting_levels = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_header(title))

        self.view = pg.GraphicsLayoutWidget()
        self.vb: pg.ViewBox = self.view.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMenuEnabled(False)
        self.image_item = pg.ImageItem()
        self.vb.addItem(self.image_item)
        self.scale_bar = CalibratedScaleBar()
        self.scale_bar.setParentItem(self.vb)
        self.scale_bar.anchor((1, 1), (1, 1), offset=(-16, -16))
        self.scale_bar.hide()
        self.view.scene().sigMouseMoved.connect(self._mouse_moved)
        lay.addWidget(self.view, 1)

        self.footer = QtWidgets.QWidget()
        self.footer.setObjectName("paneFooter")
        self.footer.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        fl = QtWidgets.QVBoxLayout(self.footer)
        fl.setContentsMargins(8, 4, 8, 6)
        fl.setSpacing(4)
        fl.addLayout(self._build_stats())
        fl.addLayout(self._build_controls())
        lay.addWidget(self.footer)
        self._sync_controls()
        self.restyle()

    # ------------------------------------------------------------------ building
    def _build_header(self, title: str) -> QtWidgets.QWidget:
        head = QtWidgets.QWidget()
        head.setObjectName("paneHeader")
        head.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(8, 2, 4, 2)
        hl.setSpacing(4)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("paneTitle")
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("paneInfo")
        hl.addWidget(self.title)
        hl.addSpacing(6)
        hl.addWidget(self.info)
        hl.addStretch(1)
        self.header_tools = QtWidgets.QHBoxLayout()
        self.header_tools.setSpacing(4)
        hl.addLayout(self.header_tools)
        fit = tool_button("Fit", self.fit, tip="Show the whole image")
        copy = tool_button("Copy", self.copyRequested.emit,
                           tip="Copy the image as displayed to the clipboard")
        menu = QtWidgets.QMenu(self)
        menu.addAction("TIFF (raw values)…").triggered.connect(
            lambda *_: self.exportRequested.emit("tiff"))
        menu.addAction("PNG (as displayed)…").triggered.connect(
            lambda *_: self.exportRequested.emit("png"))
        export = tool_button("Export", menu=menu, tip="Save the image")
        for b in (fit, copy, export):
            b.setObjectName("paneTool")
            hl.addWidget(b)
        return head

    def _build_stats(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self.stats_label = QtWidgets.QLabel("")
        self.stats_label.setObjectName("paneStats")
        # neither label may widen the pane: statistics are clipped, the readout's width
        # is reserved for its longest text
        self.stats_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                       QtWidgets.QSizePolicy.Policy.Preferred)
        self.stats_label.setMinimumWidth(40)
        self.readout = QtWidgets.QLabel("")
        self.readout.setObjectName("paneReadout")
        self.readout.setFixedWidth(self.readout.fontMetrics().horizontalAdvance(
            READOUT_TEMPLATE) + 8)
        self.readout.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                  | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.stats_label, 1)
        row.addWidget(self.readout)
        return row

    def _build_controls(self) -> QtWidgets.QHBoxLayout:
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(12)
        self.control_rows = QtWidgets.QVBoxLayout()
        self.control_rows.setSpacing(5)
        self._rows: dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}

        self.cmap_combo = QtWidgets.QComboBox()
        for name in self.colormaps:
            self.cmap_combo.addItem(CMAP_LABELS.get(name, name.capitalize()), name)
        self.cmap_combo.setMaxVisibleItems(len(self.colormaps))
        self.cmap_combo.activated.connect(
            lambda i: self.set_colormap(self.cmap_combo.itemData(i)))
        self.scale_buttons = Segmented([(label, mode) for mode, label in SCALING_LABELS.items()])
        self.scale_buttons.setToolTip("Intensity scaling (display only)")
        self.scale_buttons.picked.connect(self.set_scaling)
        range_menu = QtWidgets.QMenu(self)
        self.range_group = QtWidgets.QActionGroup(self)
        for lo, hi in PERCENTILES:
            a = range_menu.addAction(f"{lo:g} – {hi:g} %")
            a.setCheckable(True)
            a.setData((lo, hi))
            self.range_group.addAction(a)
        self.range_group.triggered.connect(lambda a: self.set_percentiles(*a.data()))
        range_menu.addSeparator()
        range_menu.addAction("Autoscale now").triggered.connect(
            lambda *_: self.autoscale_once())
        self.auto_button = tool_button("Auto", menu=range_menu, checkable=True,
                                       tip="Set the levels at the chosen percentiles on "
                                           "every update (arrow: range, autoscale once)")
        self.auto_button.setObjectName("chipButton")
        self.auto_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.auto_button.toggled.connect(self.set_autoscale)
        display = QtWidgets.QWidget()
        dl = QtWidgets.QHBoxLayout(display)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(6)
        dl.addWidget(self.scale_buttons)
        dl.addWidget(self.cmap_combo)
        dl.addWidget(self.auto_button)
        self._add_row(display, "Display")
        row.addLayout(self.control_rows, 0)

        self.hist = pg.HistogramLUTWidget(orientation="horizontal", gradientPosition="bottom")
        self.hist.item.axis.hide()
        self.hist.item.layout.setContentsMargins(0, 0, 0, 0)
        self.hist.item.gradient.showTicks(False)
        self.hist.setMinimumSize(HISTOGRAM_WIDTH[0], HISTOGRAM_MIN_HEIGHT)
        self.hist.setMaximumWidth(HISTOGRAM_WIDTH[1])
        policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                                       QtWidgets.QSizePolicy.Policy.Expanding)
        policy.setRetainSizeWhenHidden(True)  # a complex image must not shrink the footer
        self.hist.setSizePolicy(policy)
        self.hist.setImageItem(self.image_item)
        self.hist.item.sigLevelChangeFinished.connect(self._levels_edited)
        self.hist.setToolTip("Drag the handles to set the display levels")
        row.addWidget(self.hist)
        self.set_colormap(self.cmap)
        return row

    def _add_row(self, widget: QtWidgets.QWidget, label: str, index: int | None = None) -> None:
        row = control_row(widget, label)
        self._rows[widget] = row
        if index is None:
            self.control_rows.addWidget(row)
        else:
            self.control_rows.insertWidget(index, row)

    def add_control(self, widget: QtWidgets.QWidget, label: str = "") -> QtWidgets.QWidget:
        """Add a row of pane-specific controls (e.g. the detector shape) to the footer,
        above the display row."""
        self._add_row(widget, label, self.control_rows.count() - 1)
        return widget

    def set_control_visible(self, widget: QtWidgets.QWidget, on: bool) -> None:
        """Show or hide the row of a control added with :meth:`add_control`."""
        self._rows[widget].setVisible(on)

    def control_row_widgets(self) -> list[QtWidgets.QWidget]:
        return list(self._rows.values())

    def add_header_tool(self, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        widget.setObjectName(widget.objectName() or "paneTool")
        self.header_tools.addWidget(widget)
        return widget

    def set_controls_visible(self, on: bool) -> None:
        self.footer.setVisible(on)

    # ------------------------------------------------------------------ image
    def set_image(self, raw: np.ndarray | None, reset: bool = False,
                  pixel_size: float | None = None, units: str | None = None,
                  title: str | None = None,
                  rect: tuple[float, float, float, float] | None = None,
                  transform: np.ndarray | None = None) -> None:
        """Show ``raw``; ``pixel_size`` / ``units`` are those of its pixels. ``transform``
        (2 × 3) or ``rect`` place it in view coordinates (see the module docstring;
        neither = the pixel grid)."""
        if title is not None:
            self.title.setText(title)
        if pixel_size is not None:
            self.pixel_size = float(pixel_size)
        if units is not None:
            self.units = str(units)
        self.raw = None if raw is None else np.asarray(raw)
        self.transform = None
        if transform is not None:
            self.transform = np.asarray(transform, dtype=np.float64).reshape(2, 3)
        elif rect is not None and self.raw is not None:
            x, y, w, h = (float(v) for v in rect)
            rows, cols = self.raw.shape[:2]
            self.transform = np.array([[w / cols, 0.0, x], [0.0, h / rows, y]])
        self.valid = (fill_mask(self.raw) if self.ignore_fill and self.raw is not None
                      and self.raw.ndim == 2 else None)
        self.render(reset=reset)

    def clear(self) -> None:
        self.raw = None
        self.valid = None
        self.transform = None
        self.image_item.clear()
        self.scale_bar.hide()
        self.info.setText("")
        self.stats_label.setText("")
        self.readout.setText("")

    def render(self, reset: bool = False) -> None:
        if self.raw is None:
            return
        raw = self.raw
        is_complex = np.iscomplexobj(raw)
        self._setting_levels = True
        try:
            kwargs = {}
            if is_complex:
                self.image_item.setImage(self._complex_rgb(raw), autoLevels=False,
                                         levels=(0, 1), **kwargs)
            else:
                shown = self._shown(raw)
                if reset or self.autoscale:
                    kwargs["levels"] = self._levels(shown)
                self.image_item.setImage(shown, autoLevels=False, **kwargs)
        finally:
            self._setting_levels = False
        m = self.transform if self.transform is not None else np.array([[1.0, 0, 0],
                                                                         [0, 1.0, 0]])
        # QTransform(m11, m12, m21, m22, dx, dy): x' = m11 x + m21 y + dx, y' = m12 x + ...
        self.image_item.setTransform(QtGui.QTransform(m[0, 0], m[1, 0], m[0, 1], m[1, 1],
                                                      m[0, 2], m[1, 2]))
        if reset:
            self.fit()
        self.hist.setVisible(not is_complex)
        self.scale_buttons.setEnabled(not is_complex and self.cmap not in WRAPPED)
        self.cmap_combo.setEnabled(not is_complex)
        self._update_stats()
        self._update_scale_bar()

    def view_pixel_size(self) -> float:
        """Calibrated length of one view unit (the image's pixel size unless placed)."""
        if self.transform is None:
            return self.pixel_size
        return self.pixel_size / np.sqrt(abs(np.linalg.det(self.transform[:, :2])))

    def view_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Image ``(row, col)`` (continuous) under view point ``(x, y)``."""
        if self.transform is None:
            return y, x
        col, row = np.linalg.solve(self.transform[:, :2], [x - self.transform[0, 2],
                                                           y - self.transform[1, 2]])
        return float(row), float(col)

    def fit(self) -> None:
        if self.raw is not None:
            self.vb.autoRange(padding=0.02)

    def displayed(self) -> np.ndarray | None:
        """The image as drawn (scaled and clipped to the levels, then coloured), for PNG
        export and the clipboard: ``(rows, cols, 3)`` float in ``[0, 1]``, or
        ``(rows, cols)`` for the gray colour map."""
        if self.raw is None:
            return None
        if np.iscomplexobj(self.raw):
            return self._complex_rgb(self.raw)
        shown = self._shown(self.raw)
        lo, hi = self.image_item.getLevels() if self.image_item.levels is not None \
            else self._levels(shown)
        norm = np.clip((shown - lo) / (hi - lo or 1.0), 0, 1)
        if self.cmap in ("gray", GRAY_2PI):
            return norm
        lut = colormap(self.cmap).getLookupTable(0.0, 1.0, 256, alpha=False) / 255.0
        return lut[np.round(np.nan_to_num(norm) * 255).astype(int)]

    def _shown(self, raw: np.ndarray) -> np.ndarray:
        """``raw`` as drawn before the levels: scaled, or modulo 2π (wrapped maps)."""
        if self.cmap in WRAPPED:
            return np.mod(np.asarray(raw, dtype=np.float64), TWO_PI)
        return scale(raw, self.scaling)

    def _complex_rgb(self, raw: np.ndarray) -> np.ndarray:
        """Hue = phase, brightness = magnitude between the percentiles of this image, or
        of the whole level stack when one is set."""
        levels = None
        if self.level_stack is not None and np.iscomplexobj(self.level_stack):
            key = ("complex", self.percentiles)
            if key not in self._stack_levels:
                self._stack_levels[key] = percentile_levels(np.abs(self.level_stack),
                                                            *self.percentiles)
            levels = self._stack_levels[key]
        return complex_to_rgb(raw, *self.percentiles, levels=levels)

    def _levels(self, shown: np.ndarray) -> tuple[float, float]:
        if self.cmap in WRAPPED:
            return (0.0, TWO_PI)
        if self.level_stack is not None:
            key = (self.scaling, self.percentiles)
            if key not in self._stack_levels:
                self._stack_levels[key] = stack_levels(self.level_stack, *self.percentiles,
                                                       scaling=self.scaling)
            return self._stack_levels[key]
        return percentile_levels(shown if self.valid is None else shown[self.valid],
                                 *self.percentiles)

    def set_level_stack(self, stack: np.ndarray | None) -> None:
        """Take the Auto levels over every slice of ``stack`` (real, ``(n, rows, cols)``;
        e.g. the depth stack the pane shows a slice of), or over the image (``None``)."""
        if stack is self.level_stack:
            return  # same stack: keep its cached levels
        self.level_stack = None if stack is None else np.asarray(stack)
        self._stack_levels = {}
        if self.raw is not None and self.autoscale:
            self.render()

    def stats(self) -> dict[str, float]:
        if self.raw is None:
            return {}
        return image_stats(self.raw if self.valid is None else self.raw[self.valid])

    def _update_stats(self) -> None:
        s = self.stats()
        muted = MUTED
        self.stats_label.setText("&nbsp;&nbsp;&nbsp;".join(
            f"<span style='color:{muted}'>{k.capitalize()}</span> {_fmt(s[k])}"
            for k in ("mean", "min", "max", "std")) if s else "")

    # ------------------------------------------------------------------ display settings
    def set_scaling(self, mode: Scaling) -> None:
        self.scaling = mode
        self._sync_controls()
        self.render(reset=False)
        if self.raw is not None and not self.autoscale:  # new value range: rescale once
            self.autoscale_once()

    def set_colormap(self, name: str) -> None:
        wraps = (name in WRAPPED) != (self.cmap in WRAPPED)
        self.cmap = name
        self.hist.item.gradient.setColorMap(colormap(name))
        self._sync_controls()
        if wraps and self.raw is not None:  # other values drawn: new levels too
            self.render(reset=False)
            self.autoscale_once()

    def set_percentiles(self, low: float, high: float) -> None:
        self.percentiles = (float(low), float(high))
        self._sync_controls()
        self.autoscale_once()

    def set_autoscale(self, on: bool) -> None:
        self.autoscale = bool(on)
        self._sync_controls()
        if on:
            self.render()

    def set_fixed_levels(self, low: float, high: float) -> None:
        """Hold these levels (in scaled units) until Auto is turned back on, e.g. one
        range for every slice of a stack."""
        self.autoscale = False
        self._sync_controls()
        self._setting_levels = True
        try:
            self.image_item.setLevels((float(low), float(high)))
        finally:
            self._setting_levels = False

    def autoscale_once(self) -> None:
        if self.raw is None:
            return
        if np.iscomplexobj(self.raw):
            self.render()
            return
        self._setting_levels = True
        try:
            self.image_item.setLevels(self._levels(self._shown(self.raw)))
        finally:
            self._setting_levels = False

    def _levels_edited(self, *_):
        if not self._setting_levels:
            self.autoscale = False
            self._sync_controls()

    def _sync_controls(self) -> None:
        self.scale_buttons.set_current(self.scaling)
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.setCurrentIndex(self.cmap_combo.findData(self.cmap))
        self.cmap_combo.blockSignals(False)
        self.auto_button.blockSignals(True)
        self.auto_button.setChecked(self.autoscale)
        self.auto_button.blockSignals(False)
        for a in self.range_group.actions():
            a.setChecked(a.data() == self.percentiles)

    # ------------------------------------------------------------------ scale bar
    def _update_scale_bar(self) -> None:
        if self.raw is None:
            self.scale_bar.hide()
            return
        self.scale_bar.pixel_size = self.view_pixel_size()
        self.scale_bar.units = self.units
        self.scale_bar.updateBar()

    # ------------------------------------------------------------------ cursor
    def _mouse_moved(self, pos) -> None:
        if self.raw is None or not self.vb.sceneBoundingRect().contains(pos):
            self.readout.setText("")
            return
        p = self.vb.mapSceneToView(pos)
        rows, cols = self.raw.shape[:2]
        row, col = (int(np.floor(v)) for v in self.view_to_pixel(p.x(), p.y()))
        if 0 <= row < rows and 0 <= col < cols:
            v = self.raw[row, col]
            text = (f"{abs(v):.4g}∠{np.degrees(np.angle(v)):.0f}°" if np.iscomplexobj(v)
                    else f"{float(v):.5g}")
            if self.transform is None:
                self.readout.setText(f"[y, x] = [{row}, {col}]   {text}")
            else:  # a placed image: where the cursor is on the view's grid (e.g. the scan)
                self.readout.setText(f"{self.placed_label} [y, x] = "
                                     f"[{int(np.floor(p.y()))}, {int(np.floor(p.x()))}]   {text}")
        else:
            self.readout.setText("")

    def restyle(self) -> None:
        self.view.setBackground(IMAGE_BG)
        self.hist.setBackground(IMAGE_BG)
        self._update_stats()


# ---- saving what a pane shows (ptydy.gui.export) ----------------------------------


def save_tiff(path: str, image: np.ndarray, pixel_size: float = 1.0,
              units: str = "pixels") -> None:
    """Raw values as float32 TIFF (complex images: two pages, real then imaginary)."""
    import tifffile

    img = np.asarray(image)
    meta = {"pixel_size": float(pixel_size), "units": str(units)}
    if np.iscomplexobj(img):
        tifffile.imwrite(path, np.stack([img.real, img.imag]).astype(np.float32),
                         metadata=meta)
    else:
        tifffile.imwrite(path, img.astype(np.float32), metadata=meta)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """``[0, 1]`` floats (grey or RGB) as 8-bit."""
    return (np.clip(np.nan_to_num(image), 0, 1) * 255 + 0.5).astype(np.uint8)


def to_qimage(image: np.ndarray) -> QtGui.QImage:
    a = np.ascontiguousarray(to_uint8(image))
    if a.ndim == 2:
        h, w = a.shape
        q = QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
    else:
        h, w, _ = a.shape
        q = QtGui.QImage(a.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    return q.copy()  # detach from the numpy buffer


def save_png(path: str, displayed: np.ndarray) -> None:
    if not to_qimage(displayed).save(path):
        raise OSError(f"Could not write {path}")
