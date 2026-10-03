"""Views of the advanced dashboard besides the image panes.

Ported from ptydy's fast-acbf app (``ptydy.plugins.fast_acbf.dialogs.views``) to
PyQt5; keep the two in step when either changes.

- :class:`QuiverView`: a vector per bright-field pixel (the vBF image shifts, or ∇χ),
  drawn as an arrow at that pixel on the detector grid (same grid, orientation and origin
  as the χ and probe-in-k-space images: x = column, y = row, row 0 at the top), with a
  footer of controls for the arrow length, width and density
- :class:`OrthoView`: a depth stack as three sections through a point: the xy slice,
  the xz section under it and the yz section beside it. It is an :class:`ImagePane`, so
  it has the same controls (scaling, colour map, Auto, histogram), statistics and scale
  bar; its levels are taken over the whole stack. Round handles on the crosshair move the
  point (xy) and the slice (xz, yz)
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .image_pane import ACCENT, ROW_MAJOR, ImagePane, control_row
from .imaging import stack_levels  # noqa: F401 (re-exported)

MAX_ARROWS = 400
HANDLE_SIZE = 15
"""Crosshair handle diameter in screen pixels (a line alone is too thin to grab)."""


def _pane_label(text: str) -> QtWidgets.QLabel:
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("paneLabel")
    return lab


def arrow_heads(bases: np.ndarray, tips: np.ndarray, head: float = 0.35,
                half_width: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Shaft ends and head triangles of arrows from ``bases`` to ``tips`` (``(n, 2)``,
    x / y). The head is ``head`` of the arrow long and ``half_width`` of its own length
    wide on each side; the shaft stops at the head so it does not poke through the tip.
    Returns ``(shaft_ends (n, 2), triangles (n, 3, 2))``."""
    v = tips - bases
    length = np.hypot(v[:, 0], v[:, 1])[:, None]
    u = np.divide(v, length, out=np.zeros_like(v), where=length > 0)
    normal = np.column_stack([-u[:, 1], u[:, 0]])
    h = head * length
    neck = tips - h * u
    tri = np.stack([tips, neck + half_width * h * normal, neck - half_width * h * normal], 1)
    return neck, tri


class QuiverView(QtWidgets.QWidget):
    """Arrows on the detector grid with a footer: a caption with the scale, then rows of
    controls (length ×, width in px, most arrows; :meth:`add_control` adds rows)."""

    def __init__(self, title: str = "vBF image shifts", parent=None):
        super().__init__(parent)
        self.setObjectName("imagePane")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QtWidgets.QWidget()
        head.setObjectName("paneHeader")
        head.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(8, 2, 4, 2)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("paneTitle")
        hl.addWidget(self.title)
        hl.addStretch(1)
        lay.addWidget(head)

        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.setMenuEnabled(False)
        self.plot.setLabel("bottom", "detector column")
        self.plot.setLabel("left", "detector row")
        self.aperture = pg.ImageItem(axisOrder=ROW_MAJOR)
        self.aperture.setOpacity(0.35)
        self.plot.addItem(self.aperture)
        accent = ACCENT
        self.arrows = pg.PlotDataItem(connect="pairs")
        self.heads = QtWidgets.QGraphicsPathItem()
        self.heads.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.heads.setBrush(pg.mkBrush(accent))
        self.bases = pg.ScatterPlotItem(size=3, pen=None, brush=pg.mkBrush(accent))
        self.plot.addItem(self.arrows)
        self.plot.addItem(self.heads)
        self.plot.addItem(self.bases)
        lay.addWidget(self.plot, 1)

        self.footer = QtWidgets.QWidget()
        self.footer.setObjectName("paneFooter")
        self.footer.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        fl = QtWidgets.QVBoxLayout(self.footer)
        fl.setContentsMargins(8, 4, 8, 6)
        fl.setSpacing(4)
        self.caption = QtWidgets.QLabel("")
        self.caption.setObjectName("paneStats")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                   QtWidgets.QSizePolicy.Policy.Preferred)
        fl.addWidget(self.caption)
        self.control_rows = QtWidgets.QVBoxLayout()
        self.control_rows.setSpacing(5)
        self._rows: dict[QtWidgets.QWidget, QtWidgets.QWidget] = {}
        fl.addLayout(self.control_rows)
        fl.addStretch(1)
        lay.addWidget(self.footer)

        self.length_spin = QtWidgets.QDoubleSpinBox()
        self.length_spin.setRange(0.1, 20.0)
        self.length_spin.setSingleStep(0.25)
        self.length_spin.setValue(1.0)
        self.length_spin.setPrefix("× ")
        self.length_spin.setToolTip("Arrow length (× 1: the longest arrow spans one grid step)")
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.width_spin.setRange(0.5, 10.0)
        self.width_spin.setSingleStep(0.5)
        self.width_spin.setValue(2.5)
        self.width_spin.setSuffix(" px")
        self.width_spin.setToolTip("Width of the arrow shafts on screen")
        self.count_spin = QtWidgets.QSpinBox()
        self.count_spin.setRange(10, 20000)
        self.count_spin.setSingleStep(50)
        self.count_spin.setValue(MAX_ARROWS)
        self.count_spin.setToolTip("At most this many arrows (every k-th pixel on a grid)")
        for spin in (self.length_spin, self.width_spin, self.count_spin):
            spin.valueChanged.connect(lambda *_: self._draw())
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        for text, spin in (("Length", self.length_spin), ("Width", self.width_spin),
                           ("Max", self.count_spin)):
            rl.addWidget(_pane_label(text))
            rl.addWidget(spin)
        self.add_control(row, "Arrows")

        self.segments: np.ndarray | None = None
        """Base and tip of each arrow drawn, ``(2 n, 2)`` as x / y."""
        self._data: dict | None = None

    def add_control(self, widget: QtWidgets.QWidget, label: str = "") -> QtWidgets.QWidget:
        row = control_row(widget, label)
        self._rows[widget] = row
        self.control_rows.addWidget(row)
        return widget

    def set_control_visible(self, widget: QtWidgets.QWidget, on: bool) -> None:
        self._rows[widget].setVisible(on)

    def control_row_widgets(self) -> list[QtWidgets.QWidget]:
        return list(self._rows.values())

    def set_controls_visible(self, on: bool) -> None:
        self.footer.setVisible(on)

    def set_data(self, rows, cols, vectors_yx, mask_shape, units: str = "Å",
                 note: str = "", title: str | None = None,
                 longest: float | None = None) -> None:
        """``vectors_yx[i]`` (in ``units``) belongs to detector pixel ``(rows[i],
        cols[i])``; ``note`` goes into the caption (e.g. the axes of the vectors).
        ``longest``: the length drawn one grid step long (default: the longest vector
        here), e.g. the longest over a stack so every slice has the same scale."""
        if title is not None:
            self.title.setText(title)
        rows, cols = np.asarray(rows), np.asarray(cols)
        v = np.asarray(vectors_yx, dtype=np.float64).reshape(-1, 2)
        if len(v) != len(rows) or len(v) == 0:
            self.clear()
            return
        refit = self._data is None or self._data["shape"] != tuple(mask_shape)
        self._data = {"rows": rows, "cols": cols, "v": v, "shape": tuple(mask_shape),
                      "units": units, "note": note, "longest": longest}
        mask = np.zeros(mask_shape, np.float32)
        mask[rows, cols] = 1.0
        self.aperture.setImage(mask, levels=(0, 1))
        self._draw()
        if refit:
            self.fit()

    def _step(self) -> int:
        """Draw every k-th pixel (on a regular grid) to stay under the arrow limit."""
        return max(1, int(np.ceil(np.sqrt(len(self._data["v"]) / self.count_spin.value()))))

    def _draw(self) -> None:
        d = self._data
        if d is None:
            return
        rows, cols, v = d["rows"], d["cols"], d["v"]
        k = self._step()
        keep = (rows % k == 0) & (cols % k == 0)
        r, c, vy, vx = rows[keep] + 0.5, cols[keep] + 0.5, v[keep, 0], v[keep, 1]
        here = float(np.hypot(v[:, 0], v[:, 1]).max())
        longest = d["longest"] if d["longest"] else here
        # the longest arrow (here, or over the stack) spans one grid step at length × 1
        scale = 0.9 * k * self.length_spin.value() / longest if longest > 0 else 0.0
        bases = np.column_stack([c, r])
        tips = np.column_stack([c + scale * vx, r + scale * vy])
        neck, tri = arrow_heads(bases, tips)
        seg = np.empty((2 * len(r), 2))
        seg[0::2], seg[1::2] = bases, tips
        self.segments = seg
        # a zero vector (e.g. the pixel at the disk centre) has no shaft or head, only its
        # base dot: under Qt 5 (PyQt5) a zero-length "pairs" segment with a wide pen is
        # drawn as a long horizontal bar (Qt 6, ptydy, draws a dot)
        moves = np.hypot(*(tips - bases).T) > 0
        shaft = np.empty((2 * int(moves.sum()), 2))
        shaft[0::2], shaft[1::2] = bases[moves], neck[moves]
        self.arrows.setData(shaft[:, 0], shaft[:, 1],
                            pen=pg.mkPen(ACCENT, width=self.width_spin.value()))
        path = QtGui.QPainterPath()
        for t in tri[moves]:
            path.addPolygon(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in t]))
        self.heads.setPath(path)
        self.bases.setData(c, r)
        per = "every bright-field pixel" if k == 1 else f"one arrow per {k} × {k} pixels"
        note = f"; {d['note']}" if d["note"] else ""
        scope = f" (stack: {longest:.3g})" if d["longest"] else ""
        self.caption.setText(f"Longest = {here:.3g} {d['units']}{scope}{note}; {per}; "
                             f"length × {scale:.3g} px per {d['units']}")

    def fit(self) -> None:
        if self._data is None:
            return
        rows, cols = self._data["rows"], self._data["cols"]
        pad = 2 + self._step()
        self.plot.setRange(xRange=(cols.min() - pad, cols.max() + 1 + pad),
                           yRange=(rows.min() - pad, rows.max() + 1 + pad), padding=0)

    def clear(self) -> None:
        self.segments = None
        self._data = None
        self.arrows.setData([], [])
        self.heads.setPath(QtGui.QPainterPath())
        self.bases.setData([], [])
        self.aperture.clear()
        self.caption.setText("")


class OrthoView(ImagePane):
    """xy slice with its xz (below) and yz (right) sections through a movable point, and
    the controls of an :class:`ImagePane` (levels over the whole stack)."""

    z_changed = QtCore.pyqtSignal(int)

    def __init__(self, title: str = "", parent=None):
        super().__init__(title, parent, ignore_fill=True, percentiles=(0.5, 99.5))
        self.vb_xy = self.vb
        self.vb_yz = self.view.addViewBox(row=0, col=1)
        self.vb_xz = self.view.addViewBox(row=1, col=0)
        for vb in (self.vb_yz, self.vb_xz):
            vb.invertY(True)
            vb.setMenuEnabled(False)
        self.vb_xz.setXLink(self.vb_xy)
        self.vb_yz.setYLink(self.vb_xy)
        grid = self.view.ci.layout
        grid.setColumnStretchFactor(0, 3)
        grid.setColumnStretchFactor(1, 1)
        grid.setRowStretchFactor(0, 3)
        grid.setRowStretchFactor(1, 1)
        self.img_xy = self.image_item
        self.img_xz = pg.ImageItem(axisOrder=ROW_MAJOR)
        self.img_yz = pg.ImageItem(axisOrder=ROW_MAJOR)
        self.vb_xz.addItem(self.img_xz)
        self.vb_yz.addItem(self.img_yz)

        accent = QtGui.QColor(ACCENT)
        fill = QtGui.QColor(accent)
        fill.setAlpha(70)
        pen = pg.mkPen(accent, width=1)
        self.line_x = pg.InfiniteLine(angle=90, pen=pen)
        self.line_y = pg.InfiniteLine(angle=0, pen=pen)
        self.xz_x = pg.InfiniteLine(angle=90, pen=pen)
        self.xz_z = pg.InfiniteLine(angle=0, pen=pen)
        self.yz_y = pg.InfiniteLine(angle=0, pen=pen)
        self.yz_z = pg.InfiniteLine(angle=90, pen=pen)
        handle = {"size": HANDLE_SIZE, "symbol": "o", "pen": pg.mkPen(accent, width=2),
                  "brush": pg.mkBrush(fill), "hoverBrush": pg.mkBrush(accent)}
        self.handle_xy = pg.TargetItem(**handle)
        self.handle_xz = pg.TargetItem(**handle)
        self.handle_yz = pg.TargetItem(**handle)
        for h, tip in ((self.handle_xy, "Drag to move the point the sections go through"),
                       (self.handle_xz, "Drag to move x and the slice"),
                       (self.handle_yz, "Drag to move y and the slice")):
            h.setToolTip(tip)
        for vb, items in ((self.vb_xy, (self.line_x, self.line_y, self.handle_xy)),
                          (self.vb_xz, (self.xz_x, self.xz_z, self.handle_xz)),
                          (self.vb_yz, (self.yz_y, self.yz_z, self.handle_yz))):
            for it in items:
                vb.addItem(it, ignoreBounds=True)
        self.handle_xy.sigPositionChanged.connect(self._xy_dragged)
        self.handle_xz.sigPositionChanged.connect(self._xz_dragged)
        self.handle_yz.sigPositionChanged.connect(self._yz_dragged)
        for h in (self.handle_xz, self.handle_yz):  # snap to the slice centre
            h.sigPositionChangeFinished.connect(lambda *_: self._place_handles())
        # the sections follow the xy image's levels and colour map
        self.hist.item.sigLevelsChanged.connect(lambda *_: self._sync_sections())
        self.hist.item.sigLookupTableChanged.connect(lambda *_: self._sync_sections())

        self.stack: np.ndarray | None = None
        self.z = 0
        self.z_height = 1.0
        """Height of one slice in the sections, in xy pixels."""
        self._xy = (0.5, 0.5)
        """The point (x, y) in xy pixels."""
        self.rect_xz = self.rect_yz = QtCore.QRectF(0, 0, 1, 1)

    @property
    def caption(self) -> QtWidgets.QLabel:
        """The depth-scale note (in the header)."""
        return self.info

    @property
    def levels(self) -> tuple[float, float]:
        return tuple(float(v) for v in self.image_item.getLevels())

    def set_stack(self, stack: np.ndarray, step: float, pixel_size: float,
                  units: str = "A") -> None:
        """``stack`` (Nz, Ny, Nx), real or complex (hue = phase, one magnitude range);
        slices ``step`` apart, pixels ``pixel_size`` apart (same unit). Depth is drawn to
        scale unless that would make the sections thinner than a quarter or taller than
        the full image."""
        stack = np.asarray(stack)
        stack = stack.astype(np.complex64 if np.iscomplexobj(stack) else np.float32,
                             copy=False)
        nz, ny, nx = stack.shape
        first = self.stack is None or self.stack.shape != stack.shape
        self.stack = stack
        true_height = nz * step / pixel_size if pixel_size > 0 else nz
        shown = float(np.clip(true_height, 0.25 * max(ny, nx), 1.0 * max(ny, nx)))
        self.z_height = shown / nz
        # the sections' rectangles go with each setImage (a rect set before the first
        # image is scaled against a 1-pixel image and ends up far too large)
        self.rect_xz = QtCore.QRectF(0, 0, nx, nz * self.z_height)
        self.rect_yz = QtCore.QRectF(0, 0, nz * self.z_height, ny)
        if first:
            self._xy = (nx / 2, ny / 2)
        self.z = min(self.z, nz - 1)
        self.level_stack = stack
        self._stack_levels = {}
        self.set_image(stack[self.z], reset=first, pixel_size=pixel_size, units=units)
        self.info.setText("depth to scale" if np.isclose(shown, true_height)
                          else f"depth scaled ×{shown / true_height:.3g} to fit")
        self._place_handles()
        if first:
            self.vb_xz.setYRange(0, nz * self.z_height, padding=0.02)
            self.vb_yz.setXRange(0, nz * self.z_height, padding=0.02)

    # ---- drawing
    def render(self, reset: bool = False) -> None:
        super().render(reset=reset)
        self._draw_sections()

    def autoscale_once(self) -> None:
        super().autoscale_once()
        self._sync_sections()

    def set_fixed_levels(self, low: float, high: float) -> None:
        super().set_fixed_levels(low, high)
        self._sync_sections()

    def _draw_sections(self) -> None:
        if self.stack is None or self.image_item.image is None:
            return
        y, x = self.point()
        xz, yz = self.stack[:, y, :], self.stack[:, :, x].T
        if np.iscomplexobj(self.stack):
            self.img_xz.setImage(self._complex_rgb(xz), autoLevels=False, levels=(0, 1),
                                 rect=self.rect_xz)
            self.img_yz.setImage(self._complex_rgb(yz), autoLevels=False, levels=(0, 1),
                                 rect=self.rect_yz)
            return
        self.img_xz.setImage(self._shown(xz), autoLevels=False, levels=self.levels,
                             rect=self.rect_xz)
        self.img_yz.setImage(self._shown(yz), autoLevels=False, levels=self.levels,
                             rect=self.rect_yz)
        self._sync_sections()

    def _sync_sections(self) -> None:
        if self.stack is None or self.image_item.image is None or np.iscomplexobj(self.stack):
            return
        for img in (self.img_xz, self.img_yz):
            img.setLookupTable(self.image_item.lut)
            img.setLevels(self.levels)

    # ---- the point and the slice
    def set_z(self, z: int) -> None:
        if self.stack is None:
            return
        self.z = int(np.clip(z, 0, len(self.stack) - 1))
        self.set_image(self.stack[self.z])
        self._place_handles()

    def set_point(self, y: float, x: float) -> None:
        """Move the sections through ``(y, x)`` (xy pixel coordinates)."""
        if self.stack is None:
            return
        ny, nx = self.stack.shape[1:]
        self._xy = (float(np.clip(x, 0, nx - 1e-3)), float(np.clip(y, 0, ny - 1e-3)))
        self._draw_sections()
        self._place_handles()

    def point(self) -> tuple[int, int]:
        """The (y, x) pixel the sections go through."""
        ny, nx = self.stack.shape[1:]
        return (int(np.clip(np.floor(self._xy[1]), 0, ny - 1)),
                int(np.clip(np.floor(self._xy[0]), 0, nx - 1)))

    def _place_handles(self) -> None:
        x, y = self._xy
        zp = (self.z + 0.5) * self.z_height
        for item, value in ((self.line_x, x), (self.line_y, y), (self.xz_x, x),
                            (self.xz_z, zp), (self.yz_y, y), (self.yz_z, zp)):
            item.setValue(value)
        for h, pos in ((self.handle_xy, (x, y)), (self.handle_xz, (x, zp)),
                       (self.handle_yz, (zp, y))):
            h.blockSignals(True)
            h.setPos(*pos)
            h.blockSignals(False)

    def _z_from(self, pos: float) -> None:
        z = int(np.clip(pos // self.z_height, 0, len(self.stack) - 1))
        if z != self.z:
            self.z = z
            self.set_image(self.stack[z])
            self.z_changed.emit(z)

    def _xy_dragged(self, *_) -> None:
        if self.stack is not None:
            p = self.handle_xy.pos()
            self.set_point(p.y(), p.x())

    def _xz_dragged(self, *_) -> None:
        if self.stack is not None:
            p = self.handle_xz.pos()
            self._z_from(p.y())
            self.set_point(self._xy[1], p.x())

    def _yz_dragged(self, *_) -> None:
        if self.stack is not None:
            p = self.handle_yz.pos()
            self._z_from(p.x())
            self.set_point(p.y(), self._xy[0])

    def clear(self) -> None:
        super().clear()
        self.stack = None
        self.level_stack = None
        for img in (self.img_xz, self.img_yz):
            img.clear()
