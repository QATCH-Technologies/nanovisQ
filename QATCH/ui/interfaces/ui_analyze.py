# Standard Library
import atexit
import datetime as dt
import hashlib
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO, StringIO
from time import monotonic
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)
from xml.dom import minidom

import numpy as np
import pyqtgraph as pg
import pyzipper
from numpy import loadtxt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from scipy.signal import argrelextrema, savgol_filter

from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.common.fileStorage import FileStorage, secure_open
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.processors.CurveOptimizer import (
    DifferenceFactorOptimizer,
    DropEffectCorrection,
)
from QATCH.QModel import QModelIndus, QModelOnyx, QModelTweed, QModelVolta
from QATCH.ui.components import (
    AnimatedComboBox,
    AnimatedDoubleSpinBox,
    LabeledToggle,
    POIChipField,
    QATCHLineEdit,
    QATCHPushButton,
)
from QATCH.ui.components.analyze_action_bar import AnalyzeActionBar
from QATCH.ui.components.analyze_plot_cards import (
    SIGNAL_COLORS,
    DetailPlotCard,
    SignalOverviewCard,
)
from QATCH.ui.components.glass_axis_item import (
    GlassAxisItem,
    apply_glass_plot_style,
    glass_curve_pen,
)
from QATCH.ui.components.pill_stepper import PillCellButton, PillStepper
from QATCH.ui.components.plot_grid_item import PlotGridItem
from QATCH.ui.dialogs.pop_up_dialog import PopUp
from QATCH.ui.dialogs.signature_dialog import (
    SignatureDialog,
    auto_sign_matches_session,
    persist_auto_sign_key,
)
from QATCH.ui.interfaces.ui_plots import PlotContainer
from QATCH.ui.labels.section_label import SectionHeader
from QATCH.ui.styles.theme_manager import ThemeManager, desc_label_qss, tok_css
from QATCH.ui.widgets.account_popup import AccountPopup
from QATCH.ui.widgets.advanced_main_widget import AdvancedMainWidget, _InfoIcon
from QATCH.ui.widgets.query_run_info_widget import QueryRunInfoWidget
from QATCH.ui.widgets.run_filter_popover import RunFilterPopover
from QATCH.ui.widgets.table_view_widget import TableView
from QATCH.ui.workers.analyze_worker import AnalyzeWorker
from QATCH.ui.workers.run_scan_worker import RunScanWorker

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow
    from QATCH.ui.windows.analyze_window import AnalyzeWindow
TAG = "[UIAnalyze]"
USE_NEW_FILL_METHOD = True

_DEV_MODE_TTL_SECONDS = 30.0

_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qmodel-preload")


def _shutdown_executor() -> None:
    """
    Ensures the background executor tears down cleanly when the application exits.

    - wait=False: Prevents the main application thread from hanging indefinitely
                  if a background task is currently stuck or executing.
    - cancel_futures=True: Ensures any pending tasks in the queue that haven't
                           started yet are immediately discarded (Requires Python 3.9+).
    """
    _LOAD_EXECUTOR.shutdown(wait=False, cancel_futures=True)


# Register the cleanup handler to fire automatically upon Python interpreter exit
atexit.register(_shutdown_executor)


class ResistantViewBox(pg.ViewBox):
    """A `pg.ViewBox` whose wheel-zoom grows progressively more resistant
    once the view is zoomed out past its plot's soft "resting" bounds (see
    `UIAnalyze._apply_plot_limits`, which stashes those bounds on this
    ViewBox as `_pan_zoom_limits`) - each further wheel-out tick moves the
    view less than the last, rather than scaling at a constant rate
    forever.

    This is also what keeps a long, continuous zoom-out scroll (one that
    never pauses long enough to trigger `UIAnalyze`'s debounced
    bounce-back) from compounding pyqtgraph's own `1.02 ** delta`
    per-tick scale factor toward float overflow - which otherwise reaches
    `scaleBy()`/`setRange()` as a literal "Cannot set range [nan, nan]"
    exception. `wheelEvent` below also catches that exception directly as
    a last resort, since resistance alone is a mitigation, not a hard
    guarantee, against every possible float edge case.

    Panning is untouched here - only wheel-zoom gets resisted. Pan
    overscroll is instead handled after the fact by `UIAnalyze`'s
    debounced bounce-back, which already covers both pan and zoom.
    """

    # How aggressively resistance ramps up per multiple of "overshoot"
    # past the resting max span (see _resist_zoom_scale) - higher pulls
    # harder.
    _RESISTANCE_STRENGTH = 3.0

    def wheelEvent(self, ev, axis=None) -> None:
        limits = getattr(self, "_pan_zoom_limits", None)
        if limits is None:
            # No run loaded yet (_apply_plot_limits hasn't run) - defer to
            # pyqtgraph's own zoom, still guarded against the same "Cannot
            # set range [nan, nan]" crash the resisted path below handles.
            try:
                super().wheelEvent(ev, axis=axis)
            except Exception as exc:
                Log.w(
                    TAG,
                    f"Ignored a degenerate wheel-zoom (view already at a numeric extreme): {exc}",
                )
                ev.accept()
            return

        if axis in (0, 1):
            mask = [False, False]
            mask[axis] = self.state["mouseEnabled"][axis]
        else:
            mask = self.state["mouseEnabled"][:]

        raw_s = 1.02 ** (ev.delta() * self.state["wheelScaleFactor"])
        s = self._resist_zoom_scale(raw_s, limits)
        s = [(None if m is False else s) for m in mask]
        center = pg.Point(pg.invertQTransform(self.childGroup.transform()).map(ev.pos()))

        self._resetTarget()
        try:
            self.scaleBy(s, center)
        except Exception as exc:
            # pyqtgraph raises a bare Exception here (ViewBox.setRange,
            # "Cannot set range [nan, nan]") - no narrower type to catch.
            # Resistance below should make this unreachable in practice,
            # but this is the actual reported crash, so it's still worth
            # a hard backstop: drop this one tick rather than let a stray
            # numeric edge case propagate into an uncaught exception.
            Log.w(
                TAG, f"Ignored a degenerate wheel-zoom (view already at a numeric extreme): {exc}"
            )
            ev.accept()
            return
        ev.accept()
        self.sigRangeChangedManually.emit(mask)

    def _resist_zoom_scale(self, raw_s: float, limits: dict) -> float:
        """Dampens `raw_s` (pyqtgraph's raw per-tick scale factor - >1
        zooms out, <1 zooms in) the further the *current* view already
        exceeds its resting max span. Zooming back in (`raw_s <= 1.0`) is
        never resisted, so recovering from an over-zoomed state always
        feels normal.
        """
        if raw_s <= 1.0:
            return raw_s
        try:
            (x0, x1), (y0, y1) = self.viewRange()
        except Exception:
            return raw_s
        max_x = max(limits.get("maxXRange") or 0.0, 1e-9)
        max_y = max(limits.get("maxYRange") or 0.0, 1e-9)
        overshoot = max(0.0, (x1 - x0) / max_x - 1.0, (y1 - y0) / max_y - 1.0)
        if overshoot <= 0.0:
            return raw_s
        resistance = 1.0 / (1.0 + overshoot * self._RESISTANCE_STRENGTH)
        return 1.0 + (raw_s - 1.0) * resistance


def _new_glass_plot_widget() -> pg.PlotWidget:
    """A `pg.PlotWidget` pre-built with `GlassAxisItem` bottom/left axes -
    the same no-spine/no-tick-marks look PlotsUI's plots use (see
    QATCH.ui.main_window._configure_plot), so Analyze's four plot cards
    read as the same family rather than plain default pyqtgraph axes.

    Uses a `ResistantViewBox` instead of a plain `pg.ViewBox` so wheel-zoom
    on every one of these four plots (main overview + the three POI detail
    graphs) gets progressively more resistant past `UIAnalyze._apply_plot_
    limits`'s soft bounds, in step with that method's debounced pan/zoom
    bounce-back - see `ResistantViewBox` for why.
    """
    w = pg.PlotWidget(
        viewBox=ResistantViewBox(),
        axisItems={
            "bottom": GlassAxisItem(orientation="bottom"),
            "left": GlassAxisItem(orientation="left"),
        },
    )

    # Same transparency setup as PlotsUI's own plot widgets (see
    # ui_plots._make_plot_widget). setBackground(None) (see _apply_pg_theme)
    # plus these widget attributes still aren't enough on their own -
    # QAbstractScrollArea/GraphicsView's viewport keeps auto-filling an
    # opaque widget-level background underneath the scene regardless, which
    # rendered as a flat white/grey patch instead of letting PlotContainer's
    # own card paint show through. The objectName + matching
    # "#analyzeGlassPlot { background: transparent; }" rule in
    # app_theme.qss is the piece that actually fixes it: once any
    # stylesheet applies to a QAbstractScrollArea subclass, Qt propagates
    # its background to the viewport too (the same mechanism PlotsUI's own
    # "#plt, #pltB, #plt_temp" rule relies on).
    w.setObjectName("analyzeGlassPlot")
    w.setAutoFillBackground(False)
    w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    w.setFrameShape(QtWidgets.QFrame.NoFrame)
    w.setFrameShadow(QtWidgets.QFrame.Plain)
    w.setLineWidth(0)

    return w


def _enable_adaptive_resolution(item: pg.PlotDataItem) -> pg.PlotDataItem:
    """Downsamples what's actually *rendered* based on the current zoom/
    pixel width, instead of always drawing every raw sample - a run's curves
    can have many thousands of points, most of which land on the same
    screen pixel column at the overview graph's fully-zoomed-out range, so
    plotting all of them costs paint time for no visible benefit.

    `clipToView` skips samples outside the visible x-range entirely; `auto`
    downsampling then picks a stride so roughly one sample gets drawn per
    pixel of the item's current width. `method='peak'` (rather than the
    faster 'subsample'/'mean') keeps each pixel-bucket's min/max envelope
    instead of smoothing it away, since the spikes/blips right at a channel
    fill point are exactly what the model and the user both rely on to
    place POIs - flattening them here would be counterproductive.

    Both settings only affect painting: the item's own `.xData`/`.yData`
    attributes still hold every original point untouched (it's specifically
    `.getData()` - "the displayed data... after mapping and data reduction",
    per its own docstring - that returns the reduced view; nothing in this
    file calls it). POI marker positions and everything else here read from
    `self.xs`/`self.ys_*` directly, never from the plotted items, so none of
    that is affected. Adapts automatically on zoom - it's a no-op once the
    visible point count already drops below roughly one per pixel, so it's
    safe to apply uniformly rather than only at the overview's default
    zoomed-out range.
    """
    item.setClipToView(True)
    item.setDownsampling(auto=True, method="peak")
    return item


def _make_target_symbol() -> QtGui.QPainterPath:
    """Builds a "target"/crosshair `QPainterPath` - a ring, a center dot,
    and four diagonal corner ticks, matching `icons/detail-target.svg`'s
    geometry - normalized to pyqtgraph's own [-0.5, 0.5] symbol coordinate
    space (see `pyqtgraph.graphicsItems.ScatterPlotItem.Symbols`, which
    every built-in symbol string like `"star"` resolves to in that same
    space, and which itself accepts a raw `QPainterPath` as a `symbol=`
    value in place of one of those strings).

    Passed as the detail sub-graphs' current-POI highlight markers'
    `symbol=` (star1/2/3, gstars1/2/3 - see `_plot_signal_curves`) in
    place of the built-in `"star"` shape.

    The ring is two concentric circles under an odd-even fill rule
    (covered twice near the center - "even", left unfilled; covered once
    in the band between them - "odd", filled), rather than a single solid
    disk, matching the source SVG's stroke-only outer circle.
    """
    path = QtGui.QPainterPath()
    path.setFillRule(QtCore.Qt.FillRule.OddEvenFill)

    outer_r, ring_w, dot_r = 0.292, 0.035, 0.083
    path.addEllipse(QtCore.QRectF(-outer_r, -outer_r, outer_r * 2, outer_r * 2))
    inner_r = outer_r - ring_w
    path.addEllipse(QtCore.QRectF(-inner_r, -inner_r, inner_r * 2, inner_r * 2))
    path.addEllipse(QtCore.QRectF(-dot_r, -dot_r, dot_r * 2, dot_r * 2))

    for (x0, y0), (x1, y1) in (
        ((-0.206, -0.206), (-0.333, -0.333)),
        ((0.206, -0.206), (0.333, -0.333)),
        ((0.333, 0.333), (0.206, 0.206)),
        ((-0.333, 0.333), (-0.206, 0.206)),
    ):
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)

    return path


_TARGET_SYMBOL = _make_target_symbol()


def _target_pen(color) -> QtGui.QPen:
    """Builds the outline pen for a `_TARGET_SYMBOL` scatter marker (see
    `UIAnalyze._plot_signal_curves` / `_apply_pg_theme`, which color
    `star1/2/3`/`gstars1/2/3` with this). `pen=` (not just `brush=`)
    matters here: the target symbol's four corner ticks are open line
    segments (see `_make_target_symbol`) - fill alone never renders those,
    only a stroke does. Cosmetic, since pyqtgraph's `drawSymbol` scales
    the painter by `size` before applying the pen - a non-cosmetic width
    would scale right along with it (1.5 * 25 = 37.5px for `star1/2/3`'s
    size=25), while a cosmetic one stays a constant, reasonable
    device-pixel width regardless of `size`.
    """
    pen = pg.mkPen(color, width=1.5)
    pen.setCosmetic(True)
    return pen


class POIMarker(pg.InfiniteLine):
    """A vertical POI marker with a finite extent and a circular handle.

    `pg.InfiniteLine` always spans the plot's full visible height. This
    instead draws only a short segment around the run's actual
    plotted-data y-range (set via `setDataRange`, padded a little past the
    data rather than stopping exactly at it) with a themed circular handle
    centered on that segment, so the marker reads as scoped to the data
    instead of stretching edge-to-edge.

    `setAngle` is pinned to 90 (vertical) and skips `InfiniteLine`'s own
    `setRotation` call - staying unrotated is what makes the item's local
    y-axis line up directly with data-space y, which is what lets
    `setDataRange` treat `_y0`/`_y1` as literal data coordinates in
    `_computeBoundingRect`/`paint` below. (`InfiniteLine`'s rotate-then-
    remap-to-the-current-view math exists specifically so an *infinite*
    line can always reach the view's edges; a fixed-extent marker doesn't
    need any of that.)

    Movement is constrained to strictly horizontal via the `setPos`
    override: every caller elsewhere in ui_analyze.py already treats a POI
    marker's position as a single x (time) scalar via `.value()`/
    `.setValue()`. `InfiniteLine` itself never actually zeroes the y
    component of a drag - it just never showed, since an infinite line
    looks identical regardless of exactly where its (off-screen) origin
    sits. A fixed-extent line would visibly drift vertically on every drag
    without this.
    """

    _HANDLE_RADIUS = 6.0  # device pixels - constant on screen at any zoom

    # The overview graph's own drag-handle glyph - a plain filled dot, so
    # no rotation is needed (unlike a directional shape) - see _handle_icon.
    _HANDLE_ICON_PATH = os.path.join(Architecture.get_path(), "QATCH", "icons", "overview-dot.svg")

    # Fraction of a marker's own entrance animation (see setRevealProgress)
    # spent popping in the handle before the vertical line starts expanding
    # out from it - handle-then-line rather than both at once, so the
    # sequence reads as "the grip lands, then the marker unfurls" instead of
    # everything growing in a single undifferentiated blob.
    _HANDLE_REVEAL_FRAC = 0.35

    def __init__(self, *args, **kwargs):
        self._y0 = 0.0
        self._y1 = 1.0
        self._handle_outline = QtGui.QColor(255, 255, 255)
        # Set by UIAnalyze._style_poi_marker - whether this marker is
        # currently drawn in its full accent color vs a muted tone. Read
        # back by UIAnalyze._apply_pg_theme to preserve that look across a
        # theme switch (see _style_poi_marker's docstring for why this
        # can't just be re-derived from setMovable()).
        self._active_style = True
        # Entrance-animation progress (see setRevealProgress) - 1.0 (fully
        # revealed) by default so a marker never driven through the reveal
        # animation (e.g. one added outside _reveal_poi_markers) still just
        # paints normally, at full size, immediately.
        self._reveal_frac = 1.0
        # Cache for _handle_icon - tinting+rotating the SVG is only redone
        # when the handle's color actually changes (active/muted toggle,
        # theme switch), not on every paint (i.e. every drag tick).
        self._icon_cache_key = None
        self._icon_cache_pixmap = None
        super().__init__(*args, **kwargs)

    def setAngle(self, angle: float) -> None:
        self.angle = 90
        self.update()

    def setDataRange(self, y0: float, y1: float, pad_frac: float = 0.08) -> None:
        """Sets the line segment's finite vertical extent from the plotted
        data's [y0, y1], padded by `pad_frac` of that span on each side so
        it pokes out slightly past the data instead of terminating exactly
        at it.
        """
        if y1 < y0:
            y0, y1 = y1, y0
        span = y1 - y0
        pad = span * pad_frac if span > 0 else max(abs(y1), 1.0) * pad_frac
        self._y0 = y0 - pad
        self._y1 = y1 + pad
        self._boundingRect = None
        self.update()

    def setHandleOutlineColor(self, color: QtGui.QColor) -> None:
        self._handle_outline = color
        self.update()

    def setRevealProgress(self, frac: float) -> None:
        """Drives this marker's entrance effect (see `_reveal_poi_markers`):
        `frac` 0.0 draws nothing at all, 1.0 draws the marker at its full,
        normal size. In between, the handle scales in first (over the
        first `_HANDLE_REVEAL_FRAC` of `frac`'s range), then - once it's
        fully sized - the vertical line expands outward from the handle's
        position (the marker's vertical midpoint) to its full `_y0`/`_y1`
        extent over the remainder.
        """
        self._reveal_frac = max(0.0, min(1.0, frac))
        self.update()

    def setPos(self, pos) -> None:
        # Horizontal-only - see class docstring.
        if isinstance(pos, (list, tuple, np.ndarray)) and not np.ndim(pos) == 0:
            pos = [pos[0], 0]
        elif isinstance(pos, QtCore.QPointF):
            pos = [pos.x(), 0]
        super().setPos(pos)

    def _computeBoundingRect(self) -> QtCore.QRectF:
        px = self.pixelWidth() or 0.0
        py = self.pixelHeight() or 0.0
        half_w = max(self.pen.width() / 2, self.hoverPen.width() / 2, self._HANDLE_RADIUS) * px
        # Symmetric padding top/bottom for the handle, which now sits at
        # the segment's vertical midpoint rather than bulging past one end.
        v_pad = self._HANDLE_RADIUS * py
        br = QtCore.QRectF(
            -half_w, self._y0 - v_pad, 2 * half_w, (self._y1 - self._y0) + 2 * v_pad
        ).normalized()

        if self._bounds != br:
            self._bounds = br
            self.prepareGeometryChange()

        return br

    def _handle_icon(self, color: QtGui.QColor) -> QtGui.QPixmap:
        """Returns this marker's handle glyph - `overview-dot.svg`
        (see `_HANDLE_ICON_PATH`), tinted to `color` (the same color that
        used to fill the plain circle this replaces - see
        `UIAnalyze._style_poi_marker`, which drives it via the marker's own
        pen). Cached per color so repeated paints (every drag tick) don't
        reload/retint the SVG from scratch each time - only actually
        rebuilt when `color` changes (an active/muted toggle or a theme
        switch).
        """
        key = color.rgba()
        if self._icon_cache_key != key:
            size = max(1, round(self._HANDLE_RADIUS * 2))
            self._icon_cache_pixmap = PlotContainer._tinted_icon(
                self._HANDLE_ICON_PATH, color, size=size
            ).pixmap(size, size)
            self._icon_cache_key = key
        return self._icon_cache_pixmap

    def paint(self, p, *args) -> None:
        pen = self.currentPen
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)

        mid_y = (self._y0 + self._y1) / 2.0
        if self._reveal_frac >= 1.0:
            handle_frac, expand_frac = 1.0, 1.0
        else:
            handle_frac = min(1.0, self._reveal_frac / self._HANDLE_REVEAL_FRAC)
            expand_frac = max(
                0.0,
                (self._reveal_frac - self._HANDLE_REVEAL_FRAC) / (1.0 - self._HANDLE_REVEAL_FRAC),
            )

        if expand_frac > 0:
            # Grows from the handle's own position (the vertical midpoint)
            # outward toward _y0/_y1 symmetrically, rather than e.g. only
            # downward - the handle stays put as the anchor the line
            # unfurls from in both directions.
            y0 = mid_y - expand_frac * (mid_y - self._y0)
            y1 = mid_y + expand_frac * (self._y1 - mid_y)
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(0, y0), QtCore.QPointF(0, y1))

        if handle_frac > 0:
            # Handle drawn in device pixels (reset transform, same trick
            # InfiniteLine's own addMarker() glyphs use) so it stays a
            # constant size on screen regardless of the current zoom level.
            mid_device = p.transform().map(QtCore.QPointF(0, mid_y))
            tr = p.transform()
            p.resetTransform()
            icon = self._handle_icon(pen.color())
            size = 2 * self._HANDLE_RADIUS * handle_frac
            target = QtCore.QRectF(mid_device.x() - size / 2, mid_device.y() - size / 2, size, size)
            p.drawPixmap(target, icon, QtCore.QRectF(icon.rect()))
            p.setTransform(tr)

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if axis == 0:
            return None  # x axis should never be auto-scaled
        return (self._y0, self._y1)


def _hairline() -> QtWidgets.QFrame:
    """Creates a 1px divider matching the account dropdown's hairline separators.

    Same objectName-driven pattern as UIControls' own `_hairline()` (see
    app_theme.qss's app-wide `QFrame#CtrlHairline` rule, not Controls-scoped),
    duplicated locally here rather than imported cross-module.

    Returns:
        QtWidgets.QFrame: A configured frame object representing the hairline.
    """
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setObjectName("CtrlHairline")
    return line


###############################################################################
# Elaborate on the raw data gathered from the SerialProcess in parallel timing
###############################################################################
class UIAnalyze(QtWidgets.QWidget):
    progressValue = QtCore.pyqtSignal(int)
    progressFormat = QtCore.pyqtSignal(str)
    progressUpdate = QtCore.pyqtSignal()
    indus_predict_progress = QtCore.pyqtSignal(int, str)
    volta_predict_progress = QtCore.pyqtSignal(int, str)
    onyx_predict_progress = QtCore.pyqtSignal(int, str)
    # The run-load pipeline (analyze_data) runs its I/O + model-inference +
    # signal-processing work on a background thread (see _RunLoadThread /
    # _run_analysis_pipeline) to keep the UI thread responsive. These
    # signals marshal the handful of Qt widget updates that pipeline used
    # to perform directly back onto the main thread.
    _model_status_changed = QtCore.pyqtSignal(str)
    _model_status_cleared = QtCore.pyqtSignal()
    _diff_factor_value_changed = QtCore.pyqtSignal(float)

    # Plot-card gear menu wiring: maps each SIGNAL_COLORS key to the curve
    # attribute names _plot_signal_curves() assigns on self (main-graph fit
    # + scatter, then the matching detail sub-graph's fit + scatter), and to
    # the POI star-marker attributes shown alongside them. Same fixed line
    # alpha convention as PlotsUI's grid menu (see ThemedGridItem).
    _SERIES_CURVE_ATTRS = {
        "resonance": ("fit1", "scat1", "fit_1", "scat_1"),
        "difference": ("fit2", "scat2", "fit_2", "scat_2"),
        "dissipation": ("fit3", "scat3", "fit_3", "scat_3"),
    }
    _SERIES_STAR_ATTRS = {
        "resonance": ("star1", "gstars1"),
        "difference": ("star2", "gstars2"),
        "dissipation": ("star3", "gstars3"),
    }
    # This series's raw-data "point cloud" attr on the overview graph vs.
    # its matching detail sub-graph - each pair is a subset of
    # _SERIES_CURVE_ATTRS above, broken out separately since each has its
    # own independent Point-to-Point toggle (_overview_point_to_point /
    # _detail_point_to_point) that _SERIES_CURVE_ATTRS's plain per-series
    # visibility loop must not blindly override - see
    # _apply_overview_point_cloud_visibility / _apply_detail_point_cloud_
    # visibility, both applied as an AND with that per-series visibility.
    _OVERVIEW_SCATTER_ATTR = {"resonance": "scat1", "difference": "scat2", "dissipation": "scat3"}
    _DETAIL_SCATTER_ATTR = {"resonance": "scat_1", "difference": "scat_2", "dissipation": "scat_3"}
    _GRID_MAJOR_ALPHA = 45
    _GRID_MINOR_ALPHA = 18
    # Duration of the main overview graph's left-to-right curve draw-in
    # when a run's curves first appear (see _animate_curve_reveal). Linear
    # rather than eased, so it reads as a steady draw sweeping across the
    # plot rather than a fast-start/slow-end fade. Each POI marker grows
    # into place within this same window, triggered exactly when the
    # drawing front reaches that marker's x (see _reveal_poi_markers) -
    # slow enough here that each marker's own entrance reads clearly
    # rather than the whole thing rushing by.
    _CURVE_REVEAL_MS = 1400
    # Size of the looping GIF spinner shared by every "work is happening"
    # overlay (loading a run, QModel auto-fitting - see _build_gif_spinner).
    # LABEL_SIZE is the QLabel's on-screen box; RENDER_SIZE is the pixmap
    # resolution rendered into it (kept equal-ish, RENDER_SIZE a hair
    # smaller so SmoothTransformation upscaling isn't stretching a
    # perfectly-sized source).
    _SPINNER_LABEL_SIZE = 72
    _SPINNER_RENDER_SIZE = 64

    # The 5 removable stepper steps, in fixed tail-to-head removal order
    # (Channel 3 first, Fill Start last - see _on_remove_step_requested/
    # _on_add_step_requested), paired with each one's `poi_markers` index.
    # Index 2 (the permanently-hidden POI3/"Post" marker) is deliberately
    # absent - it's never independently shown/hidden by the user, it only
    # ever rides along with Fill End (see _INTERMEDIATE_STEPS's one
    # special-cased transition in _on_remove_step_requested).
    _INTERMEDIATE_STEPS = (
        ("Fill Start", 0),
        ("Fill End", 1),
        ("Channel 1", 3),
        ("Channel 2", 4),
        ("Channel 3", 5),
    )
    # The same 5 user-facing poi_markers indices as _INTERMEDIATE_STEPS
    # above, without the step labels - the Custom POIs field (POIChipField)
    # reads/writes exactly these 5 slots, in order, and never touches index
    # 2 (POI3). Derived from _INTERMEDIATE_STEPS rather than repeated as a
    # separate literal, so there's one place that knows "POI3 is hidden".
    _VISIBLE_POI_SLOTS = tuple(idx for _, idx in _INTERMEDIATE_STEPS)
    # Duration of one marker's animated fade in/out when a stepper step is
    # added/removed via +/- - matches PillStepper._ANIM_MS so the marker
    # and its pill read as one cohesive transition rather than two
    # independently-timed animations.
    _STEP_VIS_ANIM_MS = 190

    def setup_ui(self, analyze_window: "AnalyzeWindow", parent: "MainWindow"):
        super(UIAnalyze, self).__init__(None)
        self.parent: "MainWindow" = parent
        assert (
            self.parent is not None
        ), "AnalyzeProcess requires a valid MainWindow parent for proper operation."
        self.stateStep = -1
        self.zoomLevel = 1
        self.xml_path = None
        self.poi_markers = []
        self.sort_order = 1  # by date, default
        self.scan_for_most_recent_run = True
        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.run_is_new = {}  # dict_key -> bool, see _scan_run's is_new / the "New" sort filter
        # dict_keys ("{folder}:{device}") whose device was actually confirmed
        # from that run's own XML (see _scan_run's device_from_xml) at some
        # point - used by _prune_unconfirmed_devices to tell a real device
        # folder apart from filesystem cruft with no genuine capture inside.
        self._device_xml_confirmed_runs = set()

        # Run-filter popover state (independent of sort_order's Name/Date
        # choice - see _refresh_cbox_runs). date bounds are "YYYY-MM-DD"
        # strings or None ("Any"), matching the captured_date format
        # _refresh_cbox_runs already derives from run_timestamps.
        self._filter_new_only = False
        self._filter_date_from = None
        self._filter_date_to = None
        self._run_filter_popover = None

        # Advanced Settings popup state (see action_advanced/_build_advanced_layout).
        self._advanced_popup = None
        self._advanced_popup_closed_at = 0.0

        # Filesystem-watcher state that keeps the run list auto-maintained
        # instead of requiring a manual Rescan button or a full rescan on
        # every mode switch (see _ensure_watcher_armed/_rearm_watcher).
        self._run_watcher = QtCore.QFileSystemWatcher(self)
        self._run_watcher.directoryChanged.connect(self._on_watched_dir_changed)
        self._watched_load_path: Optional[str] = None
        self._pending_dirty_paths: set = set()
        self._watch_debounce_timer = QtCore.QTimer(self)
        self._watch_debounce_timer.setSingleShot(True)
        self._watch_debounce_timer.setInterval(750)
        self._watch_debounce_timer.timeout.connect(self._process_pending_watch_events)
        self._incremental_workers: list = []

        self.step_direction = "forwards"
        self.allow_modify = False
        self.moved_markers = [False, False, False, False, False, False]
        self.parent.signed_at = "[NEVER]"
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"
        self.analyzer_task = QtCore.QThread()
        self.qmodel_tweed_predictor = QModelTweed()

        self.qmodel_indus_modules_loaded = False
        self.qmodel_indus_predictor = None

        # QModel Volta  Constants
        self.QModel_volta_modules_loaded = False
        self.QModel_volta_predictor = None

        # QModel Onyx Constants
        self.QModel_onyx_modules_loaded = False
        self.QModel_onyx_predictor = None
        screen = QtWidgets.QDesktopWidget().availableGeometry()
        pct_width = 75
        pct_height = 75
        self.resize(
            int(screen.width() * pct_width / 100),
            int(screen.height() * pct_height / 100),
        )
        self.move(
            int(screen.width() * (100 - pct_width) / 200),
            int(screen.height() * (100 - pct_width) / 200),
        )

        self.layout = QtWidgets.QVBoxLayout(self)
        # Flush against the window edges, same as UIControls' toolLayout
        # (`self.toolLayout.setContentsMargins(0, 0, 0, 0)`) and PlotsUI's
        # root_layout (`(0, 8, 0, 0)`) - otherwise this QVBoxLayout's default
        # margins push the action bar down/inward and wrap the plot area in
        # an unwanted border of empty space that Controls/Plots don't have.
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Fixes #30
        self.text_Devices = QtWidgets.QLabel("Device")
        self.cBox_Devices = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )

        self.btn_Load = QtWidgets.QPushButton("Load")
        self.btn_Back = QtWidgets.QPushButton("Back")
        self.btn_Next = QtWidgets.QPushButton("Next")
        self.text_Created = QtWidgets.QLabel("[NONE]")
        self.btn_Info = QtWidgets.QPushButton("Run Info")
        self.cBox_Runs = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )

        self.graphStack = (
            QtWidgets.QStackedWidget()
        )  # must define this here, before connecting "self._update_progress_value" in the next section

        # Progressbar -------------------------------------------------------------
        # No inline stylesheet here - QProgressBar#progressBar is themed
        # app-wide (see app_theme.qss), driven by the ctrl_progress_* tokens
        # so it repaints correctly on light/dark switch.
        self.progressBar = QtWidgets.QProgressBar()
        # self.progressBar.setProperty("value", 0)
        self.progressBar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.progressBar.setObjectName("progressBar")
        # self.progressBar.setFixedHeight(50)
        self.progressBar.valueChanged.connect(self._update_progress_value)
        self.progressBar.setValue(0)
        self.progressBar.setHidden(True)
        # Themed action bar: run selector + Load/Auto-Fit/Run Info +
        # Back/Next/Modify/Analyze + Advanced/User, as one card matching
        # PlotsUI/ControlsUI's visual language (see AnalyzeActionBar). It
        # only builds/lays out the widgets - every callback below is wired
        # here since those methods live on UIAnalyze, not the bar itself.
        self.actionbar = AnalyzeActionBar()
        self.text_Created = self.actionbar.text_Created
        self.cBox_Runs = self.actionbar.cBox_Runs
        self.saved_state_dot = self.actionbar.saved_state_dot
        self.saved_state_label = self.actionbar.saved_state_label
        self.saved_state_widget = self.actionbar.saved_state_widget
        self.tBtn_Predict = self.actionbar.tBtn_Predict
        self.tBtn_Info = self.actionbar.tBtn_Info
        self.tool_Cancel = self.actionbar.tool_Cancel
        self.tool_Back = self.actionbar.tool_Back
        self.tool_Next = self.actionbar.tool_Next
        self.tool_Modify = self.actionbar.tool_Modify
        self.tool_Analyze = self.actionbar.tool_Analyze
        self.tool_Advanced = self.actionbar.tool_Advanced
        self.tool_User = self.actionbar.tool_User

        self.tBtn_Predict.clicked.connect(self._restore_qmodel_predictions)
        self.tBtn_Info.clicked.connect(self.getRunInfo)
        self.saved_state_widget.mousePressEvent = lambda _evt: self.gotoStepNum(None, 1)
        self.actionbar.filter_action.triggered.connect(self._open_run_filter_popover)
        self.actionbar.clear_filter_action.triggered.connect(self._on_filter_clear)

        self.tool_Cancel.clicked.connect(
            lambda: self.action_cancel(exit_batched_processing_mode=True)
        )
        self.tool_Back.clicked.connect(self.action_back)
        self.tool_Next.clicked.connect(self.action_next)
        self.tool_Modify.clicked.connect(self.action_modify)
        self.tool_Analyze.clicked.connect(
            self.action_analyze
        )  # TODO: skip ahead to analyze (if pois are all set)
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_Advanced.toggled.connect(
            lambda _: self.parent.controls_window.ui._refresh_checkable_style(self.tool_Advanced)
        )
        self.tool_User.clicked.connect(self._toggle_account_popup)
        self.tool_User.toggled.connect(
            lambda _: self.parent.controls_window.ui._refresh_checkable_style(self.tool_User)
        )
        self._refresh_account_button_state()

        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolLayout.addWidget(self.actionbar)
        self.toolLayout.addWidget(self.progressBar)

        # Devices ------------------------------------------------------
        # Fixes #30
        self.showRunsFromAllDevices = LabeledToggle("Show all available runs")
        self.showRunsFromAllDevices.setToolTip(
            "Show runs from every known device instead of only the one selected above."
        )
        self.showRunsFromAllDevices.setChecked(True)
        self.showRunsFromAllDevices.clicked.connect(self.showRunsFromAllDevices_clicked)
        self.cBox_Devices.setEnabled(False)

        # Parameters ------------------------------------------------------
        self.tbox_diff_factor = AnimatedDoubleSpinBox(
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.tbox_diff_factor.setDecimals(3)
        self.tbox_diff_factor.setRange(0.5, 2.0)
        self.tbox_diff_factor.setSingleStep(0.05)
        self.tbox_diff_factor.setFixedWidth(100)
        self.tbox_diff_factor.setValue(Constants.default_diff_factor)
        # Enter / focus-loss now commits directly (previously only the
        # deleted "Set/Reload" button did) - see set_new_diff_factor.
        self.tbox_diff_factor.editingFinished.connect(self.set_new_diff_factor)

        # Displayed/edited in micrometers (Constants.channel_thickness itself
        # stays in meters - the SI unit analyze_worker.py's viscosity formulas
        # expect) so the field shows e.g. "2.250" instead of "2.25e-06".
        self.validThickness = QtGui.QDoubleValidator(0, 1e6, 3)
        self.tbox_ch_thick = QATCHLineEdit()
        self.tbox_ch_thick.setValidator(self.validThickness)
        self.tbox_ch_thick.setFixedWidth(75)
        self.tbox_ch_thick.setText(f"{Constants.channel_thickness * 1e6:.3f}")
        self.tbox_ch_thick.textEdited.connect(self.set_new_ch_thick)
        self.h0 = _InfoIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "question-circle.svg"),
            tooltip="<b>Changes here apply to this session ONLY</b> Modify 'constants.py' to make a constant change value forever.",
        )

        # Real (but not directly user-facing) data store for the Custom
        # POIs chip field below - see QATCH.ui.components.poi_chip_field
        # and update_custom_pois for the full backend contract. Kept as a
        # QATCHLineEdit instance so the several `setText(f"{poi_vals}")`
        # call sites elsewhere in this file keep working unmodified.
        self.custom_poi_text = QATCHLineEdit()
        self.custom_poi_text.setFixedWidth(250)
        self.custom_poi_text.editingFinished.connect(self.update_custom_pois)

        # Options ------------------------------------------------------
        self.option_remove_dups = LabeledToggle("Remove duplicate analysis output files")
        self.option_remove_dups.setToolTip(
            "If a re-analysis produces output identical to the previous save, delete the "
            "redundant older copy instead of keeping both."
        )
        self.option_remove_dups.setChecked(True)
        # self.correct_drop_effect = QtWidgets.QCheckBox(
        #     "Apply drop effect vectors")
        # # per issue #26, disable by default
        # self.correct_drop_effect.setChecked(False)
        # self.correct_drop_effect.clicked.connect(self.change_drop_effect)

        # Add the checkbox and call-backs for using the curve-optimizer
        # utility. Compact + relocated into its own row beside the
        # "Difference Factor" caption (see _build_advanced_layout) rather
        # than living in the Processing/Options list.
        self.difference_factor_optimizer_checkbox = LabeledToggle("Auto-calculate", compact=True)
        self.difference_factor_optimizer_checkbox.setToolTip(
            "Automatically calculate the resonance/dissipation difference factor from the "
            "run data instead of using the fixed value set below."
        )
        self.difference_factor_optimizer_checkbox.setChecked(False)
        self.difference_factor_optimizer_checkbox.clicked.connect(
            self.use_difference_factor_optimizer
        )

        self.drop_effect_cancelation_checkbox = LabeledToggle("Drop effect correction")
        self.drop_effect_cancelation_checkbox.setToolTip(
            "Detect and correct anomalies in the dissipation/resonance curves caused by the "
            "initial liquid drop onto the sensor."
        )
        self.drop_effect_cancelation_checkbox.setChecked(True)
        self.drop_effect_cancelation_checkbox.clicked.connect(self.use_drop_effect_cancelation)

        self.partial_fills_checkbox = LabeledToggle("Enable Partial-Fills")
        self.partial_fills_checkbox.setToolTip(
            "Let the auto-fit prediction models account for runs where the channel wasn't "
            "completely filled."
        )
        self.partial_fills_checkbox.setChecked(False)

        # Predict Model ------------------------------------------------------
        self.cBox_Models = AnimatedComboBox(
            icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "down-chevron.svg")
        )
        self.cBox_Models.setToolTip(
            "Selects which QModel version predicts this run's points of interest (POIs) "
            "automatically."
        )
        self.cBox_Models.addItems(Constants.list_predict_models)
        if Constants.qmodel_onyx_predict:
            self.cBox_Models.setCurrentIndex(3)
        elif Constants.qmodel_volta_predict:
            self.cBox_Models.setCurrentIndex(2)
        elif Constants.qmodel_indus_predict:
            self.cBox_Models.setCurrentIndex(1)
        elif Constants.qmodel_tweed_predict:
            self.cBox_Models.setCurrentIndex(0)
        self.cBox_Models.currentTextChanged.connect(self.set_new_prediction_model)

        # Advanced Settings popup - anchored/animated flat popup shared with
        # UIControls (see AdvancedMainWidget/_build_advanced_layout), replacing
        # the old always-on-top Dialog window this used to be (see git history
        # for the previous self.advancedwidget/self.l0..l3 implementation).
        self._advanced_controls_layout = self._build_advanced_layout()
        self.advanced_container = AdvancedMainWidget.build_container(self._advanced_controls_layout)
        self._advanced_content_container = self.advanced_container
        AdvancedMainWidget.install_entrance_animation(self, self.advanced_container)

        # Numbered step indicator (was dot2..dot7, dot9, dot10 - dot8 was
        # already permanently hidden, "for POI3 removal"; dot1, the "Loaded &
        # saved" status, now lives on AnalyzeActionBar beneath cBox_Runs
        # instead of here). See _STEP_NUMS for the mapping between Stepper
        # index and the legacy 1-based step_num values gotoStepNum/
        # setDotStepMarkers use everywhere else. No longer a docked toolbar
        # row - PillStepper floats over the Signal Overview plot instead
        # (see _embed_stepper_overlay, called once graphWidget exists below).
        self.stepper = PillStepper(
            ["Load", "Fill Start", "Fill End", "Channel 1", "Channel 2", "Channel 3", "Analyze"]
        )
        self.stepper.stepClicked.connect(self._on_stepper_clicked)
        # +/- step add/remove (see _INTERMEDIATE_STEPS, _on_add_step_requested/
        # _on_remove_step_requested) - a view-only feature (see those methods'
        # docstrings): the backend still always sees all 6 POI markers,
        # exactly as before this existed. The actual +/- buttons live
        # *beside* the stepper (own widgets, not part of PillStepper itself)
        # - built in _embed_stepper_overlay, once graphWidget exists below.
        self.active_count = len(self._INTERMEDIATE_STEPS)
        self._parked_marker_values: Dict[int, float] = {}
        # Cached 0/1/2/3-channel auto-fit hypotheses (channel count -> full
        # 6-point POI list), populated by _cache_channel_hypotheses whenever
        # Onyx/Volta runs (they're the only engines that support forcing a
        # channel count) - see _apply_cached_channel_config, which +/- uses
        # to snap markers to a model-predicted layout instead of parking/
        # restoring a stale position. Reset alongside self.model_result in
        # clear()/_run_model_prediction/_restore_qmodel_predictions.
        self._channel_config_cache: Dict[int, List[int]] = {}

        self.graphWidget = _new_glass_plot_widget()
        # Background/axis colors are applied by _apply_pg_theme() (called at
        # the end of setup_ui and on every themeChanged) rather than a
        # hardcoded literal, since pyqtgraph doesn't consume QSS.
        # The overview always shows the whole run at a fixed scale - drag/
        # zoom precision is what the three detail sub-graphs below are for -
        # so interactive pan/zoom here is disabled outright rather than left
        # for the user to accidentally trigger. This only touches this one
        # ViewBox, not _new_glass_plot_widget() itself (graphWidget1/2/3 -
        # the detail sub-graphs - still need pan/zoom for precise POI
        # placement). ResistantViewBox.wheelEvent already checks
        # state["mouseEnabled"] and no-ops when both axes are off, so this
        # also kills wheel-zoom, not just drag-pan. POI marker dragging is
        # unaffected - that's handled by the marker items themselves, not
        # routed through the ViewBox's own pan/zoom.
        overview_vb = self.graphWidget.getViewBox()
        overview_vb.setMouseEnabled(x=False, y=False)
        overview_vb.setMenuEnabled(False)
        self.overview_card = SignalOverviewCard(self.graphWidget)
        self.overview_card.btn_zoom_in.clicked.connect(lambda: self.zoomFinderPlots(0.5))
        self.overview_card.btn_zoom_out.clicked.connect(lambda: self.zoomFinderPlots(2.0))
        self.overview_card.btn_move_left.clicked.connect(lambda: self.moveCurrentMarker(-1))
        self.overview_card.btn_move_right.clicked.connect(lambda: self.moveCurrentMarker(+1))
        self.overview_card.point_to_point_toggled.connect(self._set_overview_point_to_point)

        data, rows, cols = [
            {
                "A": ["", "", "", ""],
                "B": ["", "", "", ""],
                "C": ["", "", "", ""],
                "D": ["", "", "", ""],
            },
            4,
            4,
        ]
        results_table = TableView(data, rows, cols)
        results_figure = pg.PlotWidget()
        results_figure.setBackground("w")
        plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        plot_text.setHtml("<span style='font-size: 10pt'><b>No Results To View</b><br/> \
                            Load a run, follow the prompts to select points,<br/> \
                            and press \"Analyze\" action to view results.</span>")
        it = plot_text.textItem
        option = it.document().defaultTextOption()
        option.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        it.document().setDefaultTextOption(option)
        it.setTextWidth(it.boundingRect().width())
        plot_text.setPos(0.5, 0.5)
        results_figure.addItem(plot_text, ignoreBounds=True)

        self.results_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.results_split.addWidget(results_table)
        self.results_split.addWidget(results_figure)
        # self.results_split.setEnabled(False)
        # self.results_split.setSizes([1, 1])

        # self.graphStack = QtWidgets.QStackedWidget()
        self.graphStack.addWidget(self.overview_card)
        self.graphStack.addWidget(self.results_split)
        self.graphStack.setCurrentIndex(0)

        # self.QModel_widget = QtWidgets.QWidget(self)
        # self.QModel_widget.setWindowFlags(
        #     QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.WindowType.FramelessWindowHint)
        # self.QModel_widget.setWindowTitle("QModel Widget")
        # self.QModel_runBtn = QtWidgets.QPushButton("Run QModel Again")
        # self.QModel_runBtn.clicked.connect(self._restore_qmodel_predictions)
        # # self.layout.addWidget(self.QModel_runBtn)
        # # self.QModel_runBtn.setParent(None)
        # self.QModel_widget.setFixedSize(self.QModel_runBtn.sizeHint())
        # self.QModel_widget.hide()
        # floating_layout = QtWidgets.QHBoxLayout()
        # floating_layout.setContentsMargins(0, 0, 0, 0)
        # floating_layout.addWidget(self.QModel_runBtn)
        # self.QModel_widget.setLayout(floating_layout)
        # # floating_widget.move(100, 100)  # Position relative to main window
        # # floating_widget.show()
        # # layout_v4.addWidget(floating_widget)

        self.graphWidget1 = _new_glass_plot_widget()
        self.graphWidget2 = _new_glass_plot_widget()
        self.graphWidget3 = _new_glass_plot_widget()
        # Background/axis colors for graphWidget1/2/3 are applied by
        # _apply_pg_theme() alongside graphWidget, above.

        self.resonance_card = DetailPlotCard(self.graphWidget1, "Resonance", "resonance")
        self.difference_card = DetailPlotCard(self.graphWidget2, "Difference", "difference")
        self.dissipation_card = DetailPlotCard(self.graphWidget3, "Dissipation", "dissipation")

        # A QSplitter (not a plain QHBoxLayout) so any one detail card can be
        # expanded to fill the row - see _toggle_analyze_fullscreen, which
        # animates this splitter's sizes the same way graph_split's are.
        self.lowerGraphs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.lowerGraphs.setHandleWidth(6)
        self.lowerGraphs.addWidget(self.resonance_card)
        self.lowerGraphs.addWidget(self.difference_card)
        self.lowerGraphs.addWidget(self.dissipation_card)
        self.lowerGraphs.setSizes([1, 1, 1])

        self.graph_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.graph_split.addWidget(self.graphStack)
        self.graph_split.addWidget(self.lowerGraphs)
        self.graph_split.setSizes([1, 1])

        # Expand/configure parity with PlotsUI: each plot card's fullscreen
        # toggle animates graph_split/lowerGraphs to give it the whole
        # plot area, mirroring UIPlots._toggle_fullscreen.
        self._fullscreen_active_widget: Optional[QtWidgets.QWidget] = None
        for card in (
            self.overview_card,
            self.resonance_card,
            self.difference_card,
            self.dissipation_card,
        ):
            card.fullscreen_requested.connect(partial(self._toggle_analyze_fullscreen, card))

        # Configure-menu parity with PlotsUI: gear-menu grid toggles drive a
        # ThemedGridItem overlay per plot widget (see _on_grid_toggle /
        # _apply_grid_item), and per-series color/visibility toggles recolor
        # or show/hide the real plotted curves (see
        # _on_analyze_section_color_changed / _on_analyze_section_visibility_changed).
        # Keyed by SIGNAL_COLORS key rather than by card, since the overview
        # card's three sections and each detail card's single section refer
        # to the same underlying series.
        self._grid_flags: Dict[QtWidgets.QWidget, Dict[str, bool]] = {}
        self._series_colors: Dict[str, QtGui.QColor] = dict(SIGNAL_COLORS)
        self._series_visible: Dict[str, bool] = {
            "resonance": True,
            "difference": True,
            "dissipation": True,
        }
        # User preference from the overview card's "Point-to-Point
        # Rendering" gear-menu toggle (see _set_overview_point_to_point) -
        # shows/hides the overview graph's raw-data "point cloud" (the
        # near-invisible scatter dots the fit lines are a smoothed average
        # through). Defaults off: the overview graph plots a whole run's
        # raw sample count at once, the most expensive case, so its point
        # cloud starts hidden - matching the overview card's own gear-menu
        # checkbox default (see SignalOverviewCard._build_extra_menu_rows).
        # Persists across run reloads within this session; re-applied to
        # each newly plotted run's own point-cloud layer (see
        # _apply_overview_point_cloud_visibility, called from
        # _plot_signal_curves).
        self._overview_point_to_point: bool = False
        # Same preference, but per detail card (see _set_detail_point_to_
        # point / _apply_detail_point_cloud_visibility) - each covers only
        # one series' narrower POI window rather than a whole run, so these
        # default on, matching each detail card's own gear-menu checkbox
        # default (see DetailPlotCard._build_extra_menu_rows).
        self._detail_point_to_point: Dict[str, bool] = {
            "resonance": True,
            "difference": True,
            "dissipation": True,
        }
        for card, plot_widget in (
            (self.overview_card, self.graphWidget),
            (self.resonance_card, self.graphWidget1),
            (self.difference_card, self.graphWidget2),
            (self.dissipation_card, self.graphWidget3),
        ):
            card.grid_changed.connect(partial(self._on_grid_toggle, plot_widget))
            card.section_color_changed.connect(self._on_analyze_section_color_changed)
            card.section_visibility_changed.connect(self._on_analyze_section_visibility_changed)
        for key, card in (
            ("resonance", self.resonance_card),
            ("difference", self.difference_card),
            ("dissipation", self.dissipation_card),
        ):
            card.point_to_point_toggled.connect(partial(self._set_detail_point_to_point, key))
        # Starts disabled (self.stateStep is -1 here - see _update_detail_
        # point_cloud_state) since no run's wizard has reached a Channel
        # step yet; getPoints() keeps this in sync as the step changes.
        self._update_detail_point_cloud_state()

        # Same visible/draggable gap as lowerGraphs and PlotsUI's own
        # splitters (main_splitter/right_splitter in ui_plots.py), so the
        # overview card and the detail-card row read as separate panels
        # instead of touching with no seam.
        self.graph_split.setHandleWidth(6)

        # add collapse/expand icon arrows
        self.results_split.setHandleWidth(12)
        handle2 = self.results_split.handle(1)
        layout_s2 = QtWidgets.QVBoxLayout()
        layout_s2.setContentsMargins(0, 0, 0, 0)
        layout_s2.addStretch()
        self.btnMovable2 = QtWidgets.QToolButton(handle2)
        self.btnMovable2.setText("||")
        self.btnMovable2.setFont(QtGui.QFont("Arial", 8))
        self.btnMovable2.clicked.connect(
            lambda: self.results_split.setSizes(self.get_results_split_auto_sizes())
        )
        layout_s2.addWidget(self.btnMovable2)
        layout_s2.addStretch()
        handle2.setLayout(layout_s2)

        # Drag-marker hint (left) + terse keyboard-shortcut hints (right),
        # matching the target layout - previously reversed (keyboard hints
        # were on the left, drag hint on the right).
        self.footerText_hint = QtWidgets.QLabel(
            "<i>Drag markers for rough placement &nbsp;·&nbsp; "
            "click the detail plots for precise placement.</i>"
        )
        self.footerText_keys = QtWidgets.QLabel(
            "<b>Esc</b> | Back &nbsp;&nbsp; <b>Enter</b> | Next"
        )
        self.footerText_hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.footerText_keys.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.footerText_hint.setStyleSheet(desc_label_qss())
        self.footerText_keys.setStyleSheet(desc_label_qss())

        layout_h3 = QtWidgets.QHBoxLayout()
        layout_h3.addWidget(self.footerText_hint)
        layout_h3.addWidget(self.footerText_keys)

        # Add widgets to layout - hint bar sits at the very bottom, under
        # the detail-plot row, per the target layout. The workflow stepper is
        # no longer a docked row here - it floats over the plot itself (see
        # _embed_stepper_overlay), so the action bar sits directly above the
        # plot area now.
        self.layout.addLayout(self.toolLayout)
        self.layout.addWidget(self.graph_split)
        self.layout.addLayout(layout_h3)

        self.setLayout(self.layout)
        self.setWindowTitle("Analyze Data")
        # Embedding is deferred to the first showEvent (see below) rather
        # than done here - a QGraphicsProxyWidget created before this window
        # has ever actually been shown on screen doesn't reliably route
        # mouse clicks to its embedded widget until some time after the
        # window is first shown, which read as "the stepper's clicks don't
        # do anything, but only on the very first run load."

        # self.cBox_Devices.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        # self.cBox_Devices.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cBox_Runs.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.cBox_Runs.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cBox_Runs.addItem("No Runs Found")
        self.cBox_Runs.setEnabled(False)
        self.cBox_Runs.currentIndexChanged.connect(self.updateDev)
        # Loads implicitly on genuine user selection (click or keyboard
        # confirm) - `activated` only fires for real interaction, unlike
        # `currentIndexChanged` above, which also fires for the programmatic
        # clear()/addItems()/setCurrentText() calls background refreshes
        # (the run-list filesystem watcher, force_full_resync, etc.) make.
        self.cBox_Runs.activated.connect(self._on_run_activated)
        self.cBox_Devices.currentIndexChanged.connect(self.updateRunOnChange)
        self.btn_Load.pressed.connect(self.load_run)
        self.btn_Back.pressed.connect(self.goBack)
        self.btn_Next.pressed.connect(self.getPoints)
        self.btn_Info.pressed.connect(self.getRunInfo)
        # self.graphWidget.scene().sigMouseClicked.connect(self.summaryClick)
        self.graphWidget1.scene().sigMouseClicked.connect(self.onClick)
        self.graphWidget2.scene().sigMouseClicked.connect(self.onClick)
        self.graphWidget3.scene().sigMouseClicked.connect(self.onClick)

        self.askForPOIs = True

        """
        # create main graph summary point selection tool (initially hidden)
        self.AI_SelectTool_At = 0
        self.AI_Guess_Idxs = [0, 0, 0, 0, 0, 0]
        self.AI_Guess_Maxs = [5, 5, 5, 5, 5, 5]
        self.AI_Start_Vals = []
        self.AI_has_starting_values = False
        self.AI_moving_marker = False

        self.AI_SelectTool_Frame = QtWidgets.QWidget(self)
        self.AI_SelectTool_Layout = QtWidgets.QVBoxLayout()
        self.AI_SelectTool_Layout.setSpacing(0)
        self.AI_SelectTool_TitleBar = QtWidgets.QWidget()
        self.ai_layout_t = QtWidgets.QHBoxLayout()
        self.ai_layout_t.setSpacing(5)
        self.ai_layout_t.setContentsMargins(5, 0, 5, 0)
        self.AI_SelectTool_TitleBar.setLayout(self.ai_layout_t)
        self.ai_title = QtWidgets.QLabel("AI Point Selection Tool")
        self.ai_layout_t.addWidget(self.ai_title)
        self.ai_layout_t.addStretch()
        self.ai_close = QtWidgets.QLabel("X")
        self.ai_close.mouseReleaseEvent = self.hideSelectTool
        self.ai_layout_t.addWidget(self.ai_close)
        self.AI_SelectTool_TitleBar.setObjectName("AI_TitleBar")
        self.AI_SelectTool_TitleBar.setStyleSheet(
            "#AI_TitleBar { background: #DDDDDD; border: 1px solid black; border-bottom: 0; }"
        )
        self.AI_SelectTool_Layout.addWidget(self.AI_SelectTool_TitleBar)
        self.AI_SelectTool_Body = QtWidgets.QWidget()
        self.AI_SelectTool_Layout.addWidget(self.AI_SelectTool_Body)
        self.AI_SelectTool_Frame.setLayout(self.AI_SelectTool_Layout)
        self.AI_SelectTool_Frame.setVisible(False)
        self.layout.addChildWidget(self.AI_SelectTool_Frame)

        self.ai_layout = QtWidgets.QGridLayout()
        self.ai_layout.setSpacing(5)
        self.ai_layout.setContentsMargins(0, 0, 0, 0)
        self.ai_layout.setRowMinimumHeight(0, self.ai_layout.spacing())
        self.ai_layout.setRowMinimumHeight(5, self.ai_layout.spacing())
        self.AI_SelectTool_Body.setLayout(self.ai_layout)
        self.ai_backBtn = QtWidgets.QToolButton()
        self.ai_backBtn.setArrowType(QtCore.Qt.LeftArrow)
        self.ai_backBtn.adjustSize()
        self.ai_backBtn.clicked.connect(
            self.AI_Prev_Guess)  # (self.summaryBack)
        self.ai_layout.addWidget(self.ai_backBtn, 0, 0, 6, 1)
        self.ai_nextBtn = QtWidgets.QToolButton()
        self.ai_nextBtn.setArrowType(QtCore.Qt.RightArrow)
        self.ai_nextBtn.adjustSize()
        self.ai_nextBtn.clicked.connect(
            self.AI_Next_Guess)  # (self.summaryNext)
        self.ai_layout.addWidget(self.ai_nextBtn, 0, 3, 6, 1)
        self.ai_label = QtWidgets.QLabel("Point [Unknown]")
        self.ai_layout.addWidget(
            self.ai_label, 1, 1, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_score = QtWidgets.QLabel("Confidence Score: 95%")
        self.ai_layout.addWidget(
            self.ai_score, 2, 1, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_guess = QtWidgets.QLabel("Guess #: 1 of 5")
        self.ai_layout.addWidget(
            self.ai_guess, 3, 1, 1, 2, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_prev = QtWidgets.QPushButton("&Prev")
        self.ai_prev.setFixedWidth(50)
        self.ai_prev.clicked.connect(self.AI_Prev_Guess)
        self.ai_layout.addWidget(
            self.ai_prev, 4, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.ai_next = QtWidgets.QPushButton("&Next")
        self.ai_next.setFixedWidth(50)
        self.ai_next.clicked.connect(self.AI_Next_Guess)
        self.ai_layout.addWidget(
            self.ai_next, 4, 2, 1, 1, QtCore.Qt.AlignmentFlag.AlignCenter
        )
        self.AI_SelectTool_Body.setObjectName("AI_Tool")
        self.AI_SelectTool_Body.setStyleSheet(
            "#AI_Tool { background: #A9E1FA; border: 1px solid black; }"
        )
        self.ai_prev.setVisible(False)
        self.ai_next.setVisible(False)
        """

        self.progressValue.connect(lambda value: self.progressBar.setValue(value))
        self.progressFormat.connect(lambda value: self.progressBar.setFormat(value))
        self.progressUpdate.connect(self.progressBar.repaint)
        self.progressUpdate.connect(QtCore.QCoreApplication.processEvents)
        self.indus_predict_progress.connect(self._qmodel_indus_progress_update)
        self.volta_predict_progress.connect(self._QModel_volta_progress_update)
        self.onyx_predict_progress.connect(self._QModel_onyx_progress_update)
        self._model_status_changed.connect(self._set_model_status_text)
        self._model_status_cleared.connect(self._clear_model_status_text)
        self._diff_factor_value_changed.connect(self.tbox_diff_factor.setValue)

        # Arm the run-list watcher and do the one full scan now, so the run
        # list is already warm by the time the user first opens Analyze mode
        # instead of rescanning on every mode switch (see
        # _ensure_watcher_armed).
        self._rearm_watcher()

        # Apply the current theme once at startup, then keep it live -
        # unlike PlotsUI/ControlsUI, this panel previously had no
        # themeChanged subscription at all.
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._apply_theme)

    def _apply_theme(self, _mode: Optional[str] = None) -> None:
        """Re-applies token-driven colors to the chrome this panel still
        styles with inline QSS and to the pyqtgraph plot widgets, which
        don't consume QSS at all.

        The Advanced Settings panel's group titles/panel/toggles
        (`SectionHeader`/`AdvancedMainWidget`/`LabeledToggle`) already
        self-theme via their own `ThemeManager.themeChanged` subscriptions
        (see `_build_advanced_layout`/`advanced_main_widget.py`) - only its
        plain field-caption QLabels (not a themed component of their own)
        still need re-styling here, same as footerText_hint/keys.

        Args:
            _mode: Optional theme mode string, provided when connected
                directly to ThemeManager.themeChanged. Unused - the tokens
                are always re-read fresh from ThemeManager.instance().
        """
        for label in (
            self.footerText_hint,
            self.footerText_keys,
            self._lbl_diff_factor,
            self._lbl_ch_thick,
            self._lbl_custom_poi,
            self._lbl_ch_thick_unit,
            self._poi_copied_label,
        ):
            label.setStyleSheet(desc_label_qss())
        # text_Devices/_diff_hint_label have their own state-dependent
        # styling (dimmed/hint text differs by toggle state, not just
        # theme) - see _style_device_label / _set_diff_hint_text.
        self._style_device_label()
        self._style_diff_hint_label()
        self._apply_pg_theme()

    def _style_device_label(self) -> None:
        """Dims `text_Devices` to a further-muted `flat_text_muted` while
        `cBox_Devices` is disabled (i.e. "Show all available runs" is on) -
        previously only the combo itself dimmed/disabled, leaving its
        caption a fixed color regardless of state.
        """
        tok = ThemeManager.instance().tokens()
        if self.showRunsFromAllDevices.isChecked():
            r, g, b, a = tok["flat_text_muted"]
            color = f"rgba({r}, {g}, {b}, {max(0, int(a * 0.55))})"
        else:
            color = tok_css(tok["flat_text_muted"])
        self.text_Devices.setStyleSheet(
            f"QLabel {{ color: {color}; font-size: 12px; background: transparent; }}"
        )

    def _style_diff_hint_label(self) -> None:
        """Re-applies the muted/italic style to the Difference Factor hint
        label ("0.5 - 2.0" / "computed from run") on every theme change."""
        if not hasattr(self, "_diff_hint_label"):
            return
        tok = ThemeManager.instance().tokens()
        self._diff_hint_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 11px; "
            "font-style: italic; background: transparent; }"
        )

    def _set_diff_hint_text(self) -> None:
        """Sets the Difference Factor hint label text for the current
        auto-calculate state ("computed from run" vs. the valid range)."""
        auto = self.difference_factor_optimizer_checkbox.isChecked()
        self._diff_hint_label.setText("computed from run" if auto else "0.5 – 2.0")
        self._style_diff_hint_label()

    def _apply_pg_theme(self) -> None:
        """Applies token-driven background/axis colors to the pyqtgraph plot
        widgets. pyqtgraph ignores QSS entirely, so this must be called
        explicitly on init, on every themeChanged, and after ax.clear()
        (which resets axis pens on some pyqtgraph versions).
        """
        tok = ThemeManager.instance().tokens()
        text_pen = pg.mkPen(QtGui.QColor(*tok["plot_text_muted"][:3]))

        for plot_widget in (
            self.graphWidget,
            self.graphWidget1,
            self.graphWidget2,
            self.graphWidget3,
        ):
            # Transparent, matching PlotsUI's own plt/pltB/plt_temp (see
            # MainWindow._configure_plot's `setBackground(None)`) - "surface"
            # is a translucent RGBA token meant to be alpha-blended by
            # PlotContainer's own paintEvent, not filled in opaquely here.
            # Dropping the alpha (the old `QColor(*tok["surface"][:3])`) drew
            # a flat, fully-opaque patch that didn't match the actual
            # (blended) card color behind it - this lets the same card
            # background painted by PlotContainer show straight through.
            plot_widget.setBackground(None)
            # Same "glass" axis look PlotsUI uses (no spine, no tick marks,
            # muted uniform text) - see QATCH.ui.components.glass_axis_item.
            apply_glass_plot_style(plot_widget.getPlotItem(), text_pen)

        # POI markers persist across a theme switch (they're only rebuilt on
        # the next run load) - re-theme whichever ones currently exist,
        # preserving each one's current active/muted look (see
        # _style_poi_marker's docstring for why that's its own flag rather
        # than derived from setMovable()).
        for marker in getattr(self, "poi_markers", []):
            self._style_poi_marker(marker, active=getattr(marker, "_active_style", True))

        # Same idea for the detail sub-graphs' current-POI target markers
        # (star1/2/3, gstars1/2/3) - see _style_target_markers.
        self._style_target_markers()

    def _toggle_analyze_fullscreen(self, target_widget: QtWidgets.QWidget) -> None:
        """Toggle one plot card between fullscreen and normal splitter
        layout, mirroring `UIPlots._toggle_fullscreen` (see ui_plots.py) but
        adapted to Analyze's two-level splitter (`graph_split` = graphStack vs
        lowerGraphs; `lowerGraphs` itself splits the three detail cards).

        Args:
            target_widget: The plot card (overview_card/resonance_card/
                difference_card/dissipation_card) whose fullscreen state
                should be toggled.
        """
        if self._fullscreen_active_widget == target_widget:
            target_graph_sizes = self._normal_graph_sizes
            target_lower_sizes = self._normal_lower_sizes

            target_widget._is_fullscreen = False
            target_widget._apply_icon_theme()
            target_widget.btn_fs.setToolTip("Toggle Fullscreen")

            self._fullscreen_active_widget = None
        else:
            previous_widget = self._fullscreen_active_widget

            if previous_widget is None:
                self._normal_graph_sizes = self.graph_split.sizes()
                self._normal_lower_sizes = self.lowerGraphs.sizes()
            else:
                previous_widget._is_fullscreen = False
                previous_widget._apply_icon_theme()

            self._fullscreen_active_widget = target_widget

            target_widget._is_fullscreen = True
            target_widget._apply_icon_theme()
            target_widget.btn_fs.setToolTip("Restore Size")

            total_graph = sum(self.graph_split.sizes())
            total_lower = sum(self.lowerGraphs.sizes())

            fullscreen_sizes = {
                self.overview_card: (
                    [total_graph, 0],
                    self._normal_lower_sizes,
                ),
                self.resonance_card: (
                    [0, total_graph],
                    [total_lower, 0, 0],
                ),
                self.difference_card: (
                    [0, total_graph],
                    [0, total_lower, 0],
                ),
                self.dissipation_card: (
                    [0, total_graph],
                    [0, 0, total_lower],
                ),
            }

            if target_widget not in fullscreen_sizes:
                return

            target_graph_sizes, target_lower_sizes = fullscreen_sizes[target_widget]

        self._update_overview_fullscreen_enabled()

        if hasattr(self, "_fs_timer") and self._fs_timer.isActive():
            self._fs_timer.stop()
            self._freeze_analyze_plots(False)

        self._freeze_analyze_plots(True)

        duration_ms = 380
        interval_ms = 16  # ~60 FPS
        total_steps = max(1, duration_ms // interval_ms)

        start_graph_sizes = list(self.graph_split.sizes())
        start_lower_sizes = list(self.lowerGraphs.sizes())
        step_count = [0]

        def _ease_in_out_quart(progress: float) -> float:
            if progress < 0.5:
                return 8.0 * progress**4
            progress -= 1.0
            return 1.0 - 8.0 * progress**4

        def _tick() -> None:
            step_count[0] += 1

            progress = min(step_count[0] / total_steps, 1.0)
            eased_progress = _ease_in_out_quart(progress)

            graph_sizes = [
                int(start + (end - start) * eased_progress)
                for start, end in zip(start_graph_sizes, target_graph_sizes)
            ]
            lower_sizes = [
                int(start + (end - start) * eased_progress)
                for start, end in zip(start_lower_sizes, target_lower_sizes)
            ]

            self.graph_split.setSizes(graph_sizes)
            self.lowerGraphs.setSizes(lower_sizes)

            if progress >= 1.0:
                self._fs_timer.stop()
                self._freeze_analyze_plots(False)

        self._fs_timer = QtCore.QTimer(self)
        self._fs_timer.setInterval(interval_ms)
        self._fs_timer.timeout.connect(_tick)
        self._fs_timer.start()

    def _update_overview_fullscreen_enabled(self) -> None:
        """Disables the overview card's fullscreen toggle when there's no
        detail-card row to actually toggle against - either because
        `lowerGraphs` itself is currently hidden (see
        `_set_lower_graphs_visible` - no run loaded yet, or the current
        point-picking step doesn't show it) or because every individual
        detail card has been hidden via `_on_analyze_section_visibility_
        changed` (hiding a series on the overview hides its detail card
        entirely). Fullscreen-ing the overview when there's nothing in the
        detail row to hide/reveal would just be a no-op that looks like a
        broken button.

        Always leaves the button enabled while the overview is the
        *currently active* fullscreen widget, regardless of the above -
        restoring out of fullscreen must always be reachable, even if,
        say, every series got hidden while already fullscreen.
        """
        btn = getattr(self.overview_card, "btn_fs", None)
        if btn is None:
            return
        if self._fullscreen_active_widget is self.overview_card:
            btn.setEnabled(True)
            return
        any_detail_visible = self.lowerGraphs.isVisible() and any(
            card.isVisible()
            for card in (self.resonance_card, self.difference_card, self.dissipation_card)
        )
        btn.setEnabled(any_detail_visible)

    def _set_lower_graphs_visible(self, visible: bool) -> None:
        """Shows/hides the detail-card row (`lowerGraphs`) and keeps the
        overview card's fullscreen toggle in sync with it - see
        `_update_overview_fullscreen_enabled`. Use this instead of calling
        `self.lowerGraphs.setVisible(...)` directly so that recompute never
        gets missed at a new call site.
        """
        self.lowerGraphs.setVisible(visible)
        self._update_overview_fullscreen_enabled()

    # Duration of the detail-row reflow animation below - shorter than
    # _toggle_analyze_fullscreen's 380ms since this only resizes within
    # lowerGraphs itself, not the whole graph_split layout.
    _DETAIL_VISIBILITY_ANIM_MS = 300

    def _animate_detail_card_visibility(self, card: QtWidgets.QWidget, visible: bool) -> None:
        """Hides/shows one detail card (resonance_card/difference_card/
        dissipation_card) by tweening `lowerGraphs`'s splitter sizes rather
        than calling `card.setVisible()` outright - Qt would otherwise
        relayout the row instantly the moment a child's visibility
        changes, which reads as a jump-cut rather than the other card(s)
        smoothly reclaiming (or yielding) that space. Mirrors
        `_toggle_analyze_fullscreen`'s splitter-size tween, scoped to just
        this one row.

        Hiding: animates `card`'s share down to 0 while the *other*
        currently-visible cards grow to fill it, then calls
        `card.setVisible(False)` only once it's already at zero width -
        by then there's nothing left to visually snap.

        Showing: calls `card.setVisible(True)` immediately (a hidden
        splitter child can't be given a nonzero size), pins its share
        back to 0 to undo whatever Qt just auto-assigned it, then animates
        every currently-visible card - including this one - to an even
        split of the row.

        A no-op if `card` is already in the requested state, or isn't one
        of the three detail cards at all.
        """
        cards = (self.resonance_card, self.difference_card, self.dissipation_card)
        if card not in cards or card.isVisible() == visible:
            return

        prev_anim = getattr(self, "_detail_vis_anim", None)
        prev_pending = getattr(self, "_detail_vis_pending", None)
        if prev_anim is not None:
            prev_anim.stop()
            if prev_pending is not None:
                # Snap whatever animation this interrupts straight to its
                # own true end state - never leave a card stranded mid-tween.
                prev_card, prev_visible = prev_pending
                prev_card.setVisible(prev_visible)
            self._detail_vis_anim = None
            self._detail_vis_pending = None

        idx = cards.index(card)
        if visible:
            start_sizes = list(self.lowerGraphs.sizes())
            start_sizes[idx] = 0
            card.setVisible(True)
            # Undo whatever share Qt's own relayout just auto-assigned the
            # newly-shown card, so the animation's start point is exactly
            # "0, as if still hidden" rather than wherever Qt jumped it to.
            self.lowerGraphs.setSizes(start_sizes)
            visible_indices = [i for i, c in enumerate(cards) if c.isVisible()]
        else:
            start_sizes = list(self.lowerGraphs.sizes())
            visible_indices = [i for i, c in enumerate(cards) if c.isVisible() and c is not card]

        total = sum(start_sizes) or 1
        target_sizes = [0] * len(cards)
        if visible_indices:
            share, remainder = divmod(total, len(visible_indices))
            for i in visible_indices:
                target_sizes[i] = share
            target_sizes[visible_indices[0]] += remainder  # give any leftover to the first slot

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self._DETAIL_VISIBILITY_ANIM_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        def _apply(t, start=start_sizes, target=target_sizes) -> None:
            try:
                self.lowerGraphs.setSizes([int(s + (e - s) * t) for s, e in zip(start, target)])
            except RuntimeError:
                pass

        anim.valueChanged.connect(_apply)

        def _finish(anim=anim, card=card, visible=visible) -> None:
            if getattr(self, "_detail_vis_anim", None) is anim:
                self._detail_vis_anim = None
                self._detail_vis_pending = None
            if not visible:
                card.setVisible(False)
            self._update_overview_fullscreen_enabled()

        anim.finished.connect(_finish)
        self._detail_vis_anim = anim
        self._detail_vis_pending = (card, visible)
        anim.start()

    def _freeze_analyze_plots(self, freeze: bool) -> None:
        """Suppress/restore pyqtgraph viewport redraws during the fullscreen
        splitter animation (see `_toggle_analyze_fullscreen`), the same
        technique `UIPlots._freeze_plots` uses.

        Args:
            freeze: If True, disables viewport updates; if False, restores
                normal updates and requests a repaint.
        """
        update_mode = (
            QtWidgets.QGraphicsView.NoViewportUpdate
            if freeze
            else QtWidgets.QGraphicsView.MinimalViewportUpdate
        )

        for plot_widget in (
            self.graphWidget,
            self.graphWidget1,
            self.graphWidget2,
            self.graphWidget3,
        ):
            try:
                plot_widget.setViewportUpdateMode(update_mode)
                if not freeze:
                    plot_widget.viewport().update()
            except Exception:
                continue

    def _on_grid_toggle(self, plot_widget: pg.PlotWidget, key: str, visible: bool) -> None:
        """Handles a gear-menu grid checkbox toggle for one plot widget.

        Args:
            plot_widget: The graphWidget/graphWidget1/2/3 the toggle applies to.
            key: "grid_major" or "grid_minor" (see GridMenuRow in ui_plots.py).
            visible: The new checkbox state.
        """
        flags = self._grid_flags.setdefault(plot_widget, {})
        flags[key] = visible
        self._apply_grid_item(plot_widget, key, visible)

    def _apply_grid_item(self, plot_widget: pg.PlotWidget, key: str, visible: bool) -> None:
        """Add or toggle a `ThemedGridItem` on a plot's ViewBox for major or
        minor grid lines - the same technique `MainWindow._apply_grid_item`
        uses for the Plots window (see ThemedGridItem's docstring for why a
        custom GridItem is used instead of pg's native `showGrid()`).

        Args:
            plot_widget: The graphWidget/graphWidget1/2/3 to add/toggle the
                grid item on.
            key: "grid_major" or "grid_minor".
            visible: Whether the grid should be shown or hidden.
        """
        plot_item = plot_widget.getPlotItem()
        vb = plot_item.getViewBox()
        is_major = "_major" in key
        attr = "_grid_major" if is_major else "_grid_minor"
        grid = getattr(vb, attr, None)

        if visible:
            tok = ThemeManager.instance().tokens()
            base_color = QtGui.QColor(*tok["plot_text_muted"][:3])
            grid_pen = pg.mkPen(base_color)
            alpha = self._GRID_MAJOR_ALPHA if is_major else self._GRID_MINOR_ALPHA

            if grid is None:
                grid = PlotGridItem(
                    pen=grid_pen,
                    alpha=alpha,
                    x_axis=plot_item.getAxis("bottom"),
                    y_axis=plot_item.getAxis("left"),
                    include_minor_ticks=not is_major,
                    textPen=None,
                )
                grid.setZValue(-10 if is_major else -9)
                vb.addItem(grid)
                setattr(vb, attr, grid)
            else:
                grid.setPen(grid_pen)
                grid._fixed_alpha = alpha

            grid.show()
        elif grid is not None:
            grid.hide()

    def _on_analyze_section_color_changed(self, key: str, color: QtGui.QColor) -> None:
        """Recolors the real fit/scatter curves for one series, and
        remembers the choice so it survives the next `_plot_signal_curves()`
        call (e.g. loading a different run).

        Every fit line (fit1/2/3 on the overview, fit_1/2/3 on the detail
        sub-graphs) re-applies the new color through `glass_curve_pen`
        rather than setting it as a bare pen - matching both graphs'
        "glass" line style (see where these are first plotted in
        `_plot_signal_curves`) means every recolor has to go through the
        same translucency, or the line would revert to a flat opaque
        stroke the moment a user picks a new color.

        Args:
            key: One of "resonance"/"difference"/"dissipation".
            color: The newly chosen color.
        """
        self._series_colors[key] = QtGui.QColor(color)
        for attr in self._SERIES_CURVE_ATTRS.get(key, ()):
            item = getattr(self, attr, None)
            if item is None:
                continue
            if attr.startswith("fit"):
                item.setPen(glass_curve_pen(color))
            else:
                item.setSymbolBrush(color)

        # Keep every card's legend dot in sync, not just whichever card's
        # gear menu was actually used - the Signal Overview legend and the
        # matching detail card's title dot both show this same series.
        for card in (
            self.overview_card,
            self.resonance_card,
            self.difference_card,
            self.dissipation_card,
        ):
            card.set_section_color(key, color)

    def _on_analyze_section_visibility_changed(self, key: str, visible: bool) -> None:
        """Shows or hides the real curve/marker items for one series, hides
        or shows that series's entire detail card in the details row (see
        `lowerGraphs`), and remembers the choice so it survives the next
        `_plot_signal_curves()` call (e.g. loading a different run).

        Both graphs' point-cloud layers for this series (scat1/scat2/scat3
        on the overview, scat_1/scat_2/scat_3 on the matching detail
        sub-graph) are deliberately excluded from the direct apply below
        and instead routed through `_apply_overview_point_cloud_visibility`
        / `_apply_detail_point_cloud_visibility`, since each one's actual
        visibility is the AND of this per-series toggle and that graph's
        own separate "Point-to-Point" toggle - blindly setting either to
        `visible` here would force it back on even if its own toggle is
        off.

        Hiding this series's entire detail card (rather than just its
        curves within it) is what makes the details row read as "only the
        remaining visible series" (e.g. hiding Dissipation leaves just
        Resonance and Difference side by side) instead of an empty-looking
        panel still taking up a third of the row - animated via
        `_animate_detail_card_visibility` rather than an instant Qt-driven
        splitter snap, so the remaining cards visibly grow/shrink to
        reclaim or yield that space instead of jumping straight there.

        Args:
            key: One of "resonance"/"difference"/"dissipation".
            visible: The new visibility state.
        """
        self._series_visible[key] = visible
        skip_attrs = {self._OVERVIEW_SCATTER_ATTR.get(key), self._DETAIL_SCATTER_ATTR.get(key)}
        attrs = self._SERIES_CURVE_ATTRS.get(key, ()) + self._SERIES_STAR_ATTRS.get(key, ())
        for attr in attrs:
            if attr in skip_attrs:
                continue
            item = getattr(self, attr, None)
            if item is not None:
                item.setVisible(visible)
        self._apply_overview_point_cloud_visibility()
        self._apply_detail_point_cloud_visibility(key)

        detail_card = {
            "resonance": getattr(self, "resonance_card", None),
            "difference": getattr(self, "difference_card", None),
            "dissipation": getattr(self, "dissipation_card", None),
        }.get(key)
        if detail_card is not None:
            self._animate_detail_card_visibility(detail_card, visible)
        self._update_overview_fullscreen_enabled()

    def _show_analyze_plot_overlay(self) -> None:
        """Creates and displays a progress overlay on the analysis plot.

        This method initializes a placeholder `pg.PlotWidget`, embeds a
        semi-transparent status overlay (containing a label and a progress bar),
        and inserts it into the results splitter. The overlay is wrapped in a
        `QGraphicsProxyWidget` to float above the `PlotItem`.

        The overlay is automatically centered within the ViewBox using a
        single-shot timer to ensure layout calculations are complete before
        positioning.
        """
        self._hide_analyze_plot_overlay()
        results_figure = pg.PlotWidget()
        results_figure.setBackground("w")

        plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        plot_text.setHtml("<span style='font-size: 10pt'>Analyze in-progress...</span>")
        plot_text.setPos(0.5, 0.5)

        plot_text.setFlag(plot_text.GraphicsItemFlag.ItemHasNoContents, False)
        results_figure.addItem(plot_text)
        self.results_split.replaceWidget(1, results_figure)
        self.results_split.setEnabled(False)
        self._analyze_results_figure = results_figure

        # Progress overlay
        plot_item = results_figure.getPlotItem()

        container = QtWidgets.QWidget()
        container.setFixedSize(380, 62)
        container.setStyleSheet(
            "QWidget {"
            "  background: rgba(255, 255, 255, 230);"
            "}"
            "QLabel {"
            "  background: transparent;"
            "  border: none;"
            "  font-size: 10pt;"
            "  color: #333333;"
            "}"
            "QProgressBar {"
            "  border: none;"
            "  border-radius: 3px;"
            "  background: #e8f4fb;"
            "}"
            "QProgressBar::chunk {"
            "  background: #2E9BDA;"
            "  border-radius: 3px;"
            "}"
        )

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        status_label = QtWidgets.QLabel("Starting\u2026")
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        progress_bar = QtWidgets.QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        progress_bar.setFixedHeight(6)

        layout.addWidget(status_label)
        layout.addWidget(progress_bar)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(container)
        plot_item = cast(pg.PlotItem, results_figure.getPlotItem())
        graphics_container = plot_item.graphicsItem()

        if graphics_container:
            proxy.setParentItem(graphics_container)
        proxy.setZValue(1000)

        def _center() -> None:
            try:
                plot_item = results_figure.getPlotItem()
                if plot_item is None:
                    return
                vb = plot_item.getViewBox()
                vb_rect = vb.mapRectToItem(plot_item.graphicsItem(), vb.boundingRect())
                pw = proxy.boundingRect().width()
                ph = proxy.boundingRect().height()
                proxy.setPos(
                    vb_rect.x() + (vb_rect.width() - pw) / 2.0,
                    vb_rect.y() + (vb_rect.height() - ph) / 2.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _center)

        self._analyze_overlay = (proxy, progress_bar, status_label)

    def _update_analyze_plot_overlay(self, value: int, status: str) -> None:
        """Updates the progress percentage and status message on the plot overlay.

        This method updates the visual state of the `QProgressBar` and
        `QLabel` stored in the overlay tuple. It caps the progress bar value
        at 99 to prevent it from showing a "Complete" state before the
        finalization logic triggers. It also forces a UI event loop process
        to ensure the display refreshes during long-running operations.

        Args:
            value: The current progress percentage, typically between 0 and 100.
                The visual bar is capped at 99 internally.
            status: A string description of the current analysis step (e.g.,
                "Calculating FFT...", "Filtering data...").

        Note:
            This method calls `QCoreApplication.processEvents()`, which
            temporarily allows the UI to stay responsive but should be used
            cautiously to avoid re-entrancy issues.
        """
        overlay = getattr(self, "_analyze_overlay", None)
        if overlay is None:
            return
        _proxy, progress_bar, status_label = overlay
        progress_bar.setValue(min(value, 99))
        if status and len(status):
            progress_bar.setFormat(status)
            status_label.setText(status)
        QtCore.QCoreApplication.processEvents()

    def _hide_analyze_plot_overlay(self) -> None:
        """Removes the analysis progress overlay and handles task termination state.

        This method cleans up the graphics overlay by detaching the proxy widget
        from the scene. If the underlying analysis task finished successfully,
        the progress bar is briefly set to 100%. If the task failed (based on
        the worker's exit code), a warning dialog is displayed to the user.

        The method is designed to be idempotent and is safe to call even if the
        overlay has already been removed or was never initialized.

        Raises:
            RuntimeError: Silently handles cases where the underlying Qt objects
                have already been deleted by the C++ runtime.
        """
        overlay = getattr(self, "_analyze_overlay", None)
        if overlay is None:
            return

        proxy, progress_bar, _status_label = overlay

        failed = hasattr(self, "analyze_work") and not self.analyze_work.exitCode()
        if failed:
            PopUp.warning(self, Constants.app_title, "Analyze task failed.")
        else:
            progress_bar.setValue(100)
            progress_bar.setFormat("Progress: Finished")
            QtCore.QCoreApplication.processEvents()

        try:
            proxy.setParentItem(None)
            scene = proxy.scene()
            if scene is not None:
                scene.removeItem(proxy)
        except RuntimeError:
            pass

        self._analyze_overlay = None

    def _build_gif_spinner(self) -> Dict[str, Any]:
        """Builds a themed, looping GIF spinner label shared by every
        "work is happening" overlay (loading a run, QModel auto-fitting -
        see `_show_loading_run_overlay`/`_show_qmodel_plot_overlay`).

        The glyph is an animated GIF (a bouncing three-dot loader) played
        via QMovie rather than a hand-rolled QTimer/progress tracker -
        QMovie natively decodes and times GIF frames. Its raw frames are
        pre-tinted into a pixmap cache per theme (a SourceAtop recolor):
        the source GIF is a single flat color, which would look wrong (or
        invisible) against one of the two card backgrounds without being
        retinted to the active theme's accent color on every show/
        theme-change.

        Sized via `_SPINNER_LABEL_SIZE`/`_SPINNER_RENDER_SIZE` so every
        caller renders an identical spinner.

        Returns:
            dict: `label` (the QLabel to place in a layout), `movie` (the
            QMovie - caller starts/stops and `deleteLater()`s it), `apply_theme`
            (call once up front and connect to `ThemeManager.instance().
            themeChanged`), and `show_frame` (steps the cached frames
            directly - used for a manual reverse wind-down since PyQt5's
            QMovie can't play in reverse).
        """
        spinner_label = QtWidgets.QLabel()
        spinner_label.setFixedSize(self._SPINNER_LABEL_SIZE, self._SPINNER_LABEL_SIZE)
        spinner_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        icon_path = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "animations", "loading.gif"
        )
        movie = QtGui.QMovie(icon_path, QtCore.QByteArray(), self)
        movie.setCacheMode(QtGui.QMovie.CacheMode.CacheAll)
        movie.jumpToFrame(0)
        total_frames = max(1, movie.frameCount())
        render_size = self._SPINNER_RENDER_SIZE

        anim_state = {
            "color": QtGui.QColor(40, 50, 62),
            "frames": [],
        }

        def _rebuild_frame_cache() -> None:
            color = anim_state["color"]
            frames = []
            for f in range(total_frames):
                movie.jumpToFrame(f)
                base = QtGui.QPixmap.fromImage(movie.currentImage()).scaled(
                    render_size,
                    render_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                tinted = QtGui.QPixmap(base.size())
                tinted.fill(QtCore.Qt.GlobalColor.transparent)
                painter = QtGui.QPainter(tinted)
                painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                painter.drawPixmap(0, 0, base)
                painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
                painter.fillRect(tinted.rect(), color)
                painter.end()
                frames.append(tinted)
            anim_state["frames"] = frames

        def _show_frame(index: int) -> None:
            frames = anim_state["frames"]
            if frames:
                spinner_label.setPixmap(frames[max(0, min(index, len(frames) - 1))])

        movie.frameChanged.connect(_show_frame)

        def _apply_theme(_mode: str | None = None) -> None:
            tok = ThemeManager.instance().tokens()
            anim_state["color"] = QtGui.QColor(*tok["flat_accent"])
            _rebuild_frame_cache()
            _show_frame(movie.currentFrameNumber())

        return {
            "label": spinner_label,
            "movie": movie,
            "apply_theme": _apply_theme,
            "show_frame": _show_frame,
        }

    def _show_qmodel_plot_overlay(self) -> None:
        """Embeds a frosted-glass dimming layer and spinner overlay into the
        main graph widget.

        This method initializes a visual overlay for QModel inference. Unlike the
        analysis plot, this does not replace the widget; instead, it applies a
        real `QGraphicsBlurEffect` to the plot's ViewBox/axes to dim it, then
        places the shared GIF spinner (see `_build_gif_spinner`) and a
        status label on top - same look as `_show_loading_run_overlay`'s
        "Loading run..." card, since both represent the same kind of
        "work is happening" state.

        Only shown when QModel auto-fitting is triggered with no run-load
        overlay already active (e.g. the "Run QModel Again" button, after a
        run is already displayed) - see `_handle_qmodel_progress`, which
        routes progress into the existing `_loading_run_overlay` instead of
        calling this during the initial load.

        Note:
            Uses a single-shot timer to execute centering logic (`_center`)
            to ensure that the `ViewBox` geometry is fully calculated before
            centering the card.
        """
        self._hide_qmodel_plot_overlay()

        plot_item = self.graphWidget.getPlotItem()
        vb = plot_item.getViewBox()

        # Same frosted-glass technique as _show_no_run_overlay/
        # _show_loading_run_overlay: a real QGraphicsBlurEffect on the
        # ViewBox/axes, animated in lockstep with the card fade below.
        # This used to be a flat white QGraphicsRectItem, which read as a
        # tint *brightening* the canvas in dark mode instead of dimming it -
        # see those two methods' own comments for the same fix.
        frost_targets = [vb, plot_item.getAxis("bottom"), plot_item.getAxis("left")]

        def _set_frost(progress: float) -> None:
            for target in frost_targets:
                if target is None:
                    continue
                try:
                    if progress > 0.01:
                        effect = target.graphicsEffect()
                        if not isinstance(effect, QtWidgets.QGraphicsBlurEffect):
                            effect = QtWidgets.QGraphicsBlurEffect()
                            effect.setBlurHints(
                                QtWidgets.QGraphicsBlurEffect.BlurHint.PerformanceHint
                            )
                            target.setGraphicsEffect(effect)
                        effect.setBlurRadius(progress * 10.0)
                    else:
                        target.setGraphicsEffect(None)
                except RuntimeError:
                    pass

        # Spinner container
        container = QtWidgets.QWidget()
        container.setFixedWidth(280)
        container.setAutoFillBackground(False)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        container.setStyleSheet("background: transparent;")

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(10)

        spinner = self._build_gif_spinner()
        spinner_label = spinner["label"]
        movie = spinner["movie"]

        status_label = QtWidgets.QLabel("Auto-fitting points\u2026")
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        status_label.setWordWrap(True)

        spinner_row = QtWidgets.QHBoxLayout()
        spinner_row.addStretch(1)
        spinner_row.addWidget(spinner_label)
        spinner_row.addStretch(1)

        layout.addLayout(spinner_row)
        layout.addWidget(status_label)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(container)
        proxy.setParentItem(plot_item.graphicsItem())
        proxy.setZValue(1000)
        _set_frost(1.0)

        def _apply_theme(_mode: str | None = None) -> None:
            tok = ThemeManager.instance().tokens()
            text_color = tok_css(tok["flat_text"])
            status_label.setStyleSheet(f"font-size: 10pt; font-weight: 600; color: {text_color};")
            spinner["apply_theme"](_mode)

        _apply_theme()
        ThemeManager.instance().themeChanged.connect(_apply_theme)
        movie.start()

        def _center() -> None:
            try:
                vb_rect = vb.mapRectToItem(plot_item.graphicsItem(), vb.boundingRect())
                pw = proxy.boundingRect().width()
                ph = proxy.boundingRect().height()
                proxy.setPos(
                    vb_rect.x() + (vb_rect.width() - pw) / 2.0,
                    vb_rect.y() + (vb_rect.height() - ph) / 2.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _center)

        self._qmodel_overlay = {
            "proxy": proxy,
            "frost": _set_frost,
            "movie": movie,
            "status_label": status_label,
            "apply_theme": _apply_theme,
        }

    def _update_qmodel_plot_overlay(self, pct: int, status: str, is_error: bool = False) -> None:
        """Updates the shared QModel overlay's status text and, once
        finished, fades the whole card out.

        There's no numeric progress dial anymore (the GIF spinner just
        loops continuously - see `_build_gif_spinner`), so `pct` is only
        used to detect completion.

        Args:
            pct: The current progress percentage (0-100). >=100 (or an
                error) triggers the fade-out.
            status: A human-readable string describing the current inference
                step.
            is_error: If True, forces the overlay to treat the state as a failure.
        """
        overlay = getattr(self, "_qmodel_overlay", None)
        if overlay is None:
            return

        proxy, frost = overlay["proxy"], overlay["frost"]
        status_label = overlay["status_label"]
        error_detected = is_error or "error" in status.lower() or "failed" in status.lower()
        is_finished = pct >= 100 or error_detected

        if status:
            status_label.setText(status)
            if error_detected:
                status_label.setStyleSheet("font-size: 10pt; font-weight: 600; color: #DA2E2E;")

        QtCore.QCoreApplication.processEvents()
        if is_finished:
            if getattr(self, "_qmodel_is_fading", False):
                return
            self._qmodel_is_fading = True

            anim = QtCore.QVariantAnimation(self)
            anim.setDuration(400)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)

            def update_opacity(val: float):
                proxy.setOpacity(val)
                frost(val)

            anim.valueChanged.connect(update_opacity)
            anim.finished.connect(lambda: self._hide_qmodel_plot_overlay(failed=error_detected))
            anim.finished.connect(lambda: setattr(self, "_qmodel_is_fading", False))

            self._qmodel_fade_anim = anim

            QtCore.QTimer.singleShot(800, anim.start)

    def _hide_qmodel_plot_overlay(self, failed: bool = False) -> None:
        """Removes the QModel frost/blur layer and spinner overlay.

        Stops/tears down the spinner's `QMovie`, clears the blur effect off
        the ViewBox/axes, and removes the `QGraphicsProxyWidget` from the
        scene.

        Args:
            failed: If True, shows a warning popup. Defaults to False.
        """
        overlay = getattr(self, "_qmodel_overlay", None)
        if overlay is None:
            return

        proxy, frost = overlay["proxy"], overlay["frost"]
        movie = overlay["movie"]
        try:
            ThemeManager.instance().themeChanged.disconnect(overlay["apply_theme"])
        except (RuntimeError, TypeError):
            pass
        movie.stop()
        movie.deleteLater()
        frost(0.0)

        try:
            proxy.setParentItem(None)
            scene = proxy.scene()
            if scene is not None:
                scene.removeItem(proxy)
        except RuntimeError:
            pass

        self._qmodel_overlay = None

        if failed:
            PopUp.warning(self, Constants.app_title, "QModel inference failed.")

    def _fade_overlay_opacity(
        self,
        items: List[QtWidgets.QGraphicsItem],
        target_opacity: float,
        duration_ms: int = 220,
        on_finished: Optional[Callable[[], None]] = None,
    ) -> QtCore.QVariantAnimation:
        """Animates a group of QGraphicsItems' opacity to `target_opacity`,
        starting from their current opacity - used to fade the plot
        placeholder cards (no-run / loading-run) in when shown and out when
        hidden, instead of an abrupt appear/disappear.

        Args:
            items: The QGraphicsItems (dim rect + card proxy) to fade together.
            target_opacity: 1.0 to fade in, 0.0 to fade out.
            duration_ms: Fade duration.
            on_finished: Optional callback invoked once the fade completes -
                hide methods use this to defer the actual scene teardown
                until the fade-out has visually finished.

        Returns:
            The running QVariantAnimation, so the caller can stop() it if a
            new fade needs to interrupt/replace it.
        """
        start_opacity = items[0].opacity() if items else 0.0
        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(float(start_opacity))
        anim.setEndValue(float(target_opacity))
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutCubic)

        def _tick(value) -> None:
            for item in items:
                try:
                    item.setOpacity(float(value))
                except RuntimeError:
                    pass

        anim.valueChanged.connect(_tick)
        if on_finished is not None:
            anim.finished.connect(on_finished)
        anim.start()
        return anim

    def _make_step_button(self, icon_filename: str, tooltip: str) -> PillCellButton:
        """Builds one themed, icon-only round button for the stepper
        overlay's external "+"/"-" controls (see `_embed_stepper_overlay`)
        - the same self-painted `PillCellButton` the numbered pills use
        (see its own docstring for why: QSS-rendered circles come out
        jagged once embedded via `QGraphicsProxyWidget`), sized to match
        (`PillStepper._CIRCLE`) so the three read as one control cluster,
        just holding a centered icon instead of a caption/number.
        """
        btn = PillCellButton()
        btn.setFixedSize(PillStepper._CIRCLE, PillStepper._CIRCLE)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setToolTip(tooltip)
        btn.setProperty("_icon_filename", icon_filename)
        return btn

    def _restyle_step_button(self, btn: PillCellButton) -> None:
        """Re-tints one stepper +/- button for the current theme and its
        own current enabled state - a disabled `PillCellButton` doesn't
        dim itself automatically the way a normal QSS-styled `QToolButton`
        would, since it paints itself directly (see `PillCellButton`), so
        this explicitly picks a muted tint when disabled.
        """
        tok = ThemeManager.instance().tokens()
        enabled = btn.isEnabled()
        icon_color = QtGui.QColor(*(tok["flat_accent"] if enabled else tok["flat_text_muted"]))
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        icon_path = os.path.join(icons_dir, btn.property("_icon_filename"))
        pixmap = PlotContainer._tinted_icon(icon_path, icon_color, size=12).pixmap(12, 12)
        btn.set_icon_pixmap(pixmap)
        btn.set_colors(
            QtGui.QColor(*tok["flat_surface2"]),
            QtGui.QColor(*tok["flat_border"]),
            QtGui.QColor(0, 0, 0, 0),
        )

    def _update_step_buttons_enabled(self) -> None:
        """Enables/disables the stepper overlay's "+"/"-" buttons to match
        `self.active_count`'s bounds (0 to `len(_INTERMEDIATE_STEPS)`) and
        re-tints both for the current enabled state - see
        `_restyle_step_button`. Called whenever `active_count` changes
        (`_on_add_step_requested`/`_on_remove_step_requested`/
        `_reset_step_visibility`/`_reveal_steps_for_poi_vals`) and on
        every theme switch.
        """
        minus_btn = getattr(self, "_stepper_minus_btn", None)
        plus_btn = getattr(self, "_stepper_plus_btn", None)
        if minus_btn is None or plus_btn is None:
            return
        minus_btn.setEnabled(self.active_count > 0)
        plus_btn.setEnabled(self.active_count < len(self._INTERMEDIATE_STEPS))
        self._restyle_step_button(minus_btn)
        self._restyle_step_button(plus_btn)

    def _embed_stepper_overlay(self) -> None:
        """Embeds self.stepper (a PillStepper), flanked by its own external
        "+"/"-" buttons, as a persistent overlay floating at the top-center
        of the Signal Overview plot, using the same
        QGraphicsProxyWidget-on-a-PlotItem technique as
        `_show_no_run_overlay`/`_show_loading_run_overlay` below - just
        built once here (it's always present once the plots exist, not
        shown/hidden per transient state) and top-anchored instead of
        centered. Sits at a lower zValue than those two transient overlays
        and fades out while either is showing (see
        `_update_stepper_overlay_visibility`), since they already occupy the
        same top-of-plot real estate with their own card + blur treatment.

        The +/- buttons are deliberately separate widgets sitting *beside*
        the stepper in one shared row, rather than built into
        `PillStepper`'s own layout/stadium card - giving them a bit of
        breathing room from the pill row instead of crowding its card
        background.
        """
        plot_item = self.graphWidget.getPlotItem()
        vb = plot_item.getViewBox()

        self._stepper_minus_btn = self._make_step_button("subtract.svg", "Remove a workflow step")
        self._stepper_plus_btn = self._make_step_button("add.svg", "Add a workflow step")
        self._stepper_minus_btn.clicked.connect(self._on_remove_step_requested)
        # QToolButton.clicked emits clicked(bool checked=False) - connecting
        # directly would bind that `checked` value to _on_add_step_requested's
        # sole parameter (restore_position: bool = True), silently forcing
        # restore_position=False on every real click and skipping both the
        # parked-value restore and the cached-channel-config reposition (see
        # _apply_cached_channel_config). Wrap in a no-arg lambda so the
        # method's own default (True) is used instead.
        self._stepper_plus_btn.clicked.connect(lambda: self._on_add_step_requested())
        ThemeManager.instance().themeChanged.connect(lambda _: self._update_step_buttons_enabled())

        wrapper = QtWidgets.QWidget()
        wrapper.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        wrapper.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        wrapper.setAutoFillBackground(False)
        wrapper_layout = QtWidgets.QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(10)  # breathing room either side of the pill card
        wrapper_layout.addWidget(self._stepper_minus_btn, 0, QtCore.Qt.AlignVCenter)
        wrapper_layout.addWidget(self.stepper, 0, QtCore.Qt.AlignVCenter)
        wrapper_layout.addWidget(self._stepper_plus_btn, 0, QtCore.Qt.AlignVCenter)
        self._stepper_overlay_wrapper = wrapper

        self.stepper.show()
        wrapper.show()
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(wrapper)
        proxy.setParentItem(plot_item.graphicsItem())
        proxy.setZValue(900)

        def _position() -> None:
            try:
                full_rect = plot_item.boundingRect()
                pw = proxy.boundingRect().width()
                proxy.setPos(
                    full_rect.x() + (full_rect.width() - pw) / 2.0,
                    full_rect.y() + 10.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _position)
        vb.sigResized.connect(_position)
        # A step expanding/collapsing changes the stepper's own width, which
        # would otherwise leave the whole row off-center until the next
        # plot resize.
        self.stepper.sizeChanged.connect(_position)

        self._stepper_overlay_proxy = proxy
        self._update_step_buttons_enabled()
        self._update_stepper_overlay_visibility(animate=False)

    def _update_stepper_overlay_visibility(self, animate: bool = True) -> None:
        """Fades the floating stepper out while the "no run loaded" or
        "loading run" placeholder card is showing over the same plot region,
        and back in once neither is. Called from those overlays' own
        show/hide methods rather than the other way around, so this stays a
        pure follower of their state."""
        proxy = getattr(self, "_stepper_overlay_proxy", None)
        if proxy is None:
            return
        hide = (
            getattr(self, "_no_run_overlay", None) is not None
            or getattr(self, "_loading_run_overlay", None) is not None
        )
        target = 0.0 if hide else 1.0
        if not animate:
            proxy.setOpacity(target)
            return
        prev_fade = getattr(self, "_stepper_overlay_fade", None)
        if prev_fade is not None:
            try:
                prev_fade.stop()
            except RuntimeError:
                pass
        self._stepper_overlay_fade = self._fade_overlay_opacity([proxy], target, duration_ms=160)

    def _show_no_run_overlay(self) -> None:
        """Shows a placeholder card over the Signal Overview plot while no
        run is loaded: a blurred backdrop plus a centered card (icon,
        title, subtext, and a "Load from folder..." button) - same
        QGraphicsProxyWidget technique as `_show_qmodel_plot_overlay`, but
        this one stays up for as long as `clear()`'s "nothing loaded" state
        holds rather than for the duration of a single background task, so
        it also keeps itself centered (and themed) across resizes and
        light/dark switches. Fades in via `_fade_overlay_opacity` rather
        than appearing abruptly.

        `clear()` (and therefore this) reruns every time Analyze mode is
        (re-)entered even when no run was ever loaded, e.g. tabbing away and
        back - a no-op in that case since the card is still showing, so skip
        the teardown/rebuild/fade-in rather than flashing it out and back in
        for no visible change.
        """
        if getattr(self, "_no_run_overlay", None) is not None:
            return

        self._hide_no_run_overlay(animate=False)
        self._hide_loading_run_overlay(animate=True)

        plot_item = self.graphWidget.getPlotItem()
        vb = plot_item.getViewBox()

        # Same frosted-glass technique as PlotsUI's own plot-dimming (see
        # MainWindow._apply_plot_dim): just a real QGraphicsBlurEffect on
        # the ViewBox/axes, its radius animated in lockstep with the card
        # fade below - no separate colored overlay, which read as a tint
        # darkening/brightening the canvas rather than a plain blur.
        frost_targets = [vb, plot_item.getAxis("bottom"), plot_item.getAxis("left")]

        def _set_frost(progress: float) -> None:
            for target in frost_targets:
                if target is None:
                    continue
                try:
                    if progress > 0.01:
                        effect = target.graphicsEffect()
                        if not isinstance(effect, QtWidgets.QGraphicsBlurEffect):
                            effect = QtWidgets.QGraphicsBlurEffect()
                            effect.setBlurHints(
                                QtWidgets.QGraphicsBlurEffect.BlurHint.PerformanceHint
                            )
                            target.setGraphicsEffect(effect)
                        effect.setBlurRadius(progress * 10.0)
                    else:
                        target.setGraphicsEffect(None)
                except RuntimeError:
                    pass

        container = QtWidgets.QWidget()
        container.setFixedWidth(360)
        # Transparent by default a plain QWidget hosted in a
        # QGraphicsProxyWidget still paints an opaque palette background,
        # which would otherwise show as a flat white patch over the
        # blurred plot behind it.
        container.setAutoFillBackground(False)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        container.setStyleSheet("background: transparent;")

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(8)

        icon_badge = QtWidgets.QLabel()
        icon_badge.setFixedSize(56, 56)
        icon_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "import.svg")

        icon_row = QtWidgets.QHBoxLayout()
        icon_row.addStretch(1)
        icon_row.addWidget(icon_badge)
        icon_row.addStretch(1)

        title_label = QtWidgets.QLabel("Load a run to begin analysis")
        title_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        subtext_label = QtWidgets.QLabel(
            "Choose a run from the menu above. Any prior point selections "
            "will be restored automatically from saved data."
        )
        subtext_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subtext_label.setWordWrap(True)

        folder_btn = QATCHPushButton("Load from folder...", variant="primary")
        folder_btn.clicked.connect(self.load_all_from_folder)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(folder_btn)
        btn_row.addStretch(1)

        layout.addLayout(icon_row)
        layout.addWidget(title_label)
        layout.addWidget(subtext_label)
        layout.addSpacing(6)
        layout.addLayout(btn_row)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(container)
        proxy.setParentItem(plot_item.graphicsItem())
        proxy.setZValue(1000)
        proxy.setOpacity(0.0)

        def _apply_theme(_mode: str | None = None) -> None:
            """Re-tints the card's own text/icon colors to match the active
            theme, so dark mode gets light text instead of a fixed dark
            palette (which read as illegible against a dark-themed plot)."""
            tok = ThemeManager.instance().tokens()
            text_color = tok_css(tok["flat_text"])
            muted_color = tok_css(tok["flat_text_muted"])
            badge_bg = tok_css(tok["flat_accent_weak"])
            accent = QtGui.QColor(*tok["flat_accent"])

            icon_badge.setStyleSheet(f"background: {badge_bg}; border-radius: 14px;")
            icon_badge.setPixmap(
                PlotContainer._tinted_icon(icon_path, accent, size=24).pixmap(24, 24)
            )
            title_label.setStyleSheet(f"font-size: 12pt; font-weight: 600; color: {text_color};")
            subtext_label.setStyleSheet(f"font-size: 9pt; color: {muted_color};")

        _apply_theme()
        ThemeManager.instance().themeChanged.connect(_apply_theme)

        def _center() -> None:
            try:
                full_rect = plot_item.boundingRect()
                pw = proxy.boundingRect().width()
                ph = proxy.boundingRect().height()
                proxy.setPos(
                    full_rect.x() + (full_rect.width() - pw) / 2.0,
                    full_rect.y() + (full_rect.height() - ph) / 2.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _center)
        # Unlike the transient QModel/analyze overlays (one-shot centering
        # is enough for their short lifetime), this one can sit visible
        # across window resizes, so it needs to keep re-centering itself.
        vb.sigResized.connect(_center)

        fade_anim = self._fade_overlay_opacity([proxy], 1.0)
        fade_anim.valueChanged.connect(_set_frost)

        self._no_run_overlay = {
            "proxy": proxy,
            "center": _center,
            "theme": _apply_theme,
            "frost": _set_frost,
            "fade": fade_anim,
        }
        self._update_stepper_overlay_visibility()

    def _hide_no_run_overlay(self, animate: bool = True) -> None:
        """Removes the "no run loaded" placeholder card, if shown.

        Safe to call even if the overlay was never shown or was already
        removed.

        Args:
            animate: If True (default), fades the card out before removing
                it from the scene. Callers that are just clearing a stale
                instance before immediately building a fresh one (the
                defensive call at the top of `_show_no_run_overlay` itself)
                pass False, since that isn't a user-visible transition.
        """
        overlay = getattr(self, "_no_run_overlay", None)
        if overlay is None:
            return

        proxy = overlay["proxy"]
        try:
            self.graphWidget.getPlotItem().getViewBox().sigResized.disconnect(overlay["center"])
        except (RuntimeError, TypeError):
            pass
        try:
            ThemeManager.instance().themeChanged.disconnect(overlay["theme"])
        except (RuntimeError, TypeError):
            pass
        prev_fade = overlay.get("fade")
        if prev_fade is not None:
            try:
                prev_fade.stop()
            except RuntimeError:
                pass

        frost = overlay.get("frost")

        def _cleanup() -> None:
            if frost is not None:
                frost(0.0)
            try:
                proxy.setParentItem(None)
                scene = proxy.scene()
                if scene is not None:
                    scene.removeItem(proxy)
            except RuntimeError:
                pass

        self._no_run_overlay = None
        self._update_stepper_overlay_visibility()
        if animate:
            unfade_anim = self._fade_overlay_opacity(
                [proxy], 0.0, duration_ms=180, on_finished=_cleanup
            )
            if frost is not None:
                unfade_anim.valueChanged.connect(frost)
        else:
            _cleanup()

    def _show_loading_run_overlay(self) -> None:
        """Shows a spinning-loader card over the Signal Overview plot while
        a run is being read/processed (see `analyze_data`, which shows this
        at the top and hides it once the background `_RunLoadThread`
        result is unpacked), replacing the static "no run loaded" card so
        the user sees active progress instead of a "load a run" prompt.

        Same QGraphicsProxyWidget technique as `_show_no_run_overlay`,
        including the fade-in via `_fade_overlay_opacity`. The status label
        this builds is also what `_set_model_status_text`/
        `_clear_model_status_text` update with finer-grained progress (e.g.
        QModel auto-fit status) during the load.
        """
        self._hide_no_run_overlay(animate=True)
        self._hide_loading_run_overlay(animate=False)
        # Single choke point (per this method's own docstring, every load
        # path routes through here) for the "actively loading" indicator -
        # settles to "saved"/"error" once the load actually finishes, at
        # the existing _set_saved_state call sites.
        self._set_saved_state("loading", "Loading run…")

        plot_item = self.graphWidget.getPlotItem()
        vb = plot_item.getViewBox()

        # Same frosted-glass technique as PlotsUI's own plot-dimming (see
        # MainWindow._apply_plot_dim/_show_no_run_overlay above) - just a
        # real QGraphicsBlurEffect on the ViewBox/axes, no separate colored
        # overlay.
        frost_targets = [vb, plot_item.getAxis("bottom"), plot_item.getAxis("left")]

        def _set_frost(progress: float) -> None:
            for target in frost_targets:
                if target is None:
                    continue
                try:
                    if progress > 0.01:
                        effect = target.graphicsEffect()
                        if not isinstance(effect, QtWidgets.QGraphicsBlurEffect):
                            effect = QtWidgets.QGraphicsBlurEffect()
                            effect.setBlurHints(
                                QtWidgets.QGraphicsBlurEffect.BlurHint.PerformanceHint
                            )
                            target.setGraphicsEffect(effect)
                        effect.setBlurRadius(progress * 10.0)
                    else:
                        target.setGraphicsEffect(None)
                except RuntimeError:
                    pass

        container = QtWidgets.QWidget()
        container.setFixedWidth(280)
        container.setAutoFillBackground(False)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        container.setStyleSheet("background: transparent;")

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(24, 28, 24, 28)
        layout.setSpacing(10)

        spinner = self._build_gif_spinner()
        spinner_label = spinner["label"]
        movie = spinner["movie"]
        show_frame = spinner["show_frame"]

        status_label = QtWidgets.QLabel("Loading run...")
        status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        status_label.setWordWrap(True)

        spinner_row = QtWidgets.QHBoxLayout()
        spinner_row.addStretch(1)
        spinner_row.addWidget(spinner_label)
        spinner_row.addStretch(1)

        layout.addLayout(spinner_row)
        layout.addWidget(status_label)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(container)
        proxy.setParentItem(plot_item.graphicsItem())
        proxy.setZValue(1000)
        proxy.setOpacity(0.0)

        def _apply_theme(_mode: str | None = None) -> None:
            tok = ThemeManager.instance().tokens()
            text_color = tok_css(tok["flat_text"])
            status_label.setStyleSheet(f"font-size: 11pt; font-weight: 600; color: {text_color};")
            spinner["apply_theme"](_mode)

        _apply_theme()
        ThemeManager.instance().themeChanged.connect(_apply_theme)
        movie.start()

        def _center() -> None:
            try:
                full_rect = plot_item.boundingRect()
                pw = proxy.boundingRect().width()
                ph = proxy.boundingRect().height()
                proxy.setPos(
                    full_rect.x() + (full_rect.width() - pw) / 2.0,
                    full_rect.y() + (full_rect.height() - ph) / 2.0,
                )
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(0, _center)
        vb.sigResized.connect(_center)

        fade_anim = self._fade_overlay_opacity([proxy], 1.0)
        fade_anim.valueChanged.connect(_set_frost)

        self._loading_run_overlay = {
            "proxy": proxy,
            "center": _center,
            "theme": _apply_theme,
            "movie": movie,
            "show_frame": show_frame,
            "status_label": status_label,
            "frost": _set_frost,
            "fade": fade_anim,
        }
        self._update_stepper_overlay_visibility()

    def _hide_loading_run_overlay(self, animate: bool = True) -> None:
        """Removes the "loading run" spinner card, if shown.

        Safe to call even if the overlay was never shown or was already
        removed.

        Args:
            animate: If True (default), winds the loader's frames back
                toward its start while fading the card out (rather than
                just cutting the forward loop), before removing it from
                the scene. Callers just clearing a stale instance before
                immediately building a fresh one pass False, which skips
                the wind-down and cuts straight to cleanup.
        """
        overlay = getattr(self, "_loading_run_overlay", None)
        if overlay is None:
            return

        try:
            self.graphWidget.getPlotItem().getViewBox().sigResized.disconnect(overlay["center"])
        except (RuntimeError, TypeError):
            pass
        try:
            ThemeManager.instance().themeChanged.disconnect(overlay["theme"])
        except (RuntimeError, TypeError):
            pass
        prev_fade = overlay.get("fade")
        if prev_fade is not None:
            try:
                prev_fade.stop()
            except RuntimeError:
                pass

        proxy = overlay["proxy"]
        movie, show_frame = overlay["movie"], overlay["show_frame"]
        frost = overlay.get("frost")
        movie.stop()

        def _cleanup() -> None:
            if frost is not None:
                frost(0.0)
            movie.deleteLater()
            try:
                proxy.setParentItem(None)
                scene = proxy.scene()
                if scene is not None:
                    scene.removeItem(proxy)
            except RuntimeError:
                pass

        self._loading_run_overlay = None
        self._update_stepper_overlay_visibility()
        if animate:
            # Step the cached frame index back toward 0 directly (bypassing
            # the movie, which PyQt5's QMovie can't play in reverse) while
            # the card fades out, so the loader winds down instead of just
            # freezing mid-loop.
            reverse_timer = QtCore.QTimer(self)
            reverse_timer.setInterval(33)
            reverse_state = {"frame": movie.currentFrameNumber()}

            def _reverse_tick() -> None:
                reverse_state["frame"] -= 1
                if reverse_state["frame"] < 0:
                    reverse_timer.stop()
                    return
                show_frame(reverse_state["frame"])

            reverse_timer.timeout.connect(_reverse_tick)
            reverse_timer.start()

            def _on_faded() -> None:
                reverse_timer.stop()
                _cleanup()

            unfade_anim = self._fade_overlay_opacity(
                [proxy], 0.0, duration_ms=180, on_finished=_on_faded
            )
            if frost is not None:
                unfade_anim.valueChanged.connect(frost)
        else:
            _cleanup()

    def _set_loading_run_status(self, msg: str) -> None:
        """Updates the loading spinner card's status text, if shown.

        Safe to call even if the overlay isn't currently visible (e.g. a
        model-status signal arriving after the run already finished
        loading) - it's just a no-op in that case.
        """
        overlay = getattr(self, "_loading_run_overlay", None)
        if overlay is not None:
            overlay["status_label"].setText(msg)

    # def hideSelectTool(self, event):
    #     self.AI_SelectTool_Frame.hide()

    def get_results_split_auto_sizes(self, setMinimumWidth=True):
        tableWidget = self.results_split.widget(0).findChild(QtWidgets.QTableWidget)
        full_width = self.results_split.width()
        min_width = tableWidget.verticalHeader().width() + 6  # +6 seems to be needed
        min_width += tableWidget.verticalScrollBar().width()
        for i in range(tableWidget.columnCount()):
            # seems to include gridline
            min_width += tableWidget.columnWidth(i)
            if i == 0 and setMinimumWidth:
                tableWidget.setMinimumWidth(min_width)
        setSizes = [min_width, full_width - min_width]
        return setSizes

    def _visible_poi_vals(self, poi_vals_all6: List[int]) -> List[int]:
        """Projects a full 6-value (one per `self.poi_markers` slot) index
        list down to the 5 user-facing Custom POI values, dropping POI3's
        (index 2) value - see `_VISIBLE_POI_SLOTS`."""
        return [poi_vals_all6[i] for i in self._VISIBLE_POI_SLOTS]

    def update_custom_pois(self):
        new_pois = self.custom_poi_text.text()
        new_pois = (
            new_pois.replace("[", "").replace("]", "").replace(",", "")
        )  # remove array chars: '[],'
        new_pois = np.fromstring(
            new_pois, sep=" "
        ).tolist()  # convert string to numpy array and then to a list
        Log.w(f"Set Custom POIs: {new_pois}")
        # Maps onto the 5 user-facing slots only (POI1, POI2, POI4, POI5,
        # POI6 - see _VISIBLE_POI_SLOTS) - POI3 (poi_markers[2]) is never
        # reachable through this path. Fewer than 5 provided values leaves
        # the trailing visible markers untouched rather than erroring.
        for slot, val in enumerate(new_pois[: len(self._VISIBLE_POI_SLOTS)]):
            marker_idx = self._VISIBLE_POI_SLOTS[slot]
            pm = self.poi_markers[marker_idx]
            try:
                index = self.xs[int(val)]
                if pm.value() != index:
                    Log.d(f"Moving marker {marker_idx} to position {index}")
                    self.detect_change()
                    pm.setValue(index)
                    pm.sigPositionChangeFinished.emit(pm)
                else:
                    Log.d(f"Moving marker {marker_idx} not required. Already there.")
            except Exception as e:
                Log.e(f"Moving marker {marker_idx} failed: {str(e)}")

    def _copy_custom_pois(self) -> None:
        """Copies the current (<=5) Custom POI values to the clipboard as a
        Python-list literal, e.g. "[8723, 8725, 12180]" - same clipboard
        pattern as `run_info_widget.RunInfoWidget.copyText`."""
        values = self._poi_chip_field.values()
        try:
            cb = QtWidgets.QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(str(values), mode=cb.Clipboard)
        except Exception as e:
            Log.e(f"Clipboard error: {e}")
            return
        self._poi_copied_label.setVisible(True)
        QtCore.QTimer.singleShot(2000, lambda: self._poi_copied_label.setVisible(False))

    def showRunsFromAllDevices_clicked(self):
        self.cBox_Devices.setEnabled(not self.showRunsFromAllDevices.isChecked())
        self._style_device_label()
        self.update_run(self.cBox_Devices.currentIndex())

    def _open_run_filter_popover(self) -> None:
        """Opens the "▾ filters" popover anchored to the run search field.

        Reuses the existing cBox_Devices/showRunsFromAllDevices state (owned
        by the Advanced panel's device picker) as the single source of
        truth for device filtering rather than duplicating it - the
        popover's device chip is a thin front-end onto that same state.
        """
        devices = [self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())]
        current_device = self.cBox_Devices.currentText()
        show_all = self.showRunsFromAllDevices.isChecked()

        popover = RunFilterPopover(
            devices=devices,
            current_device=current_device,
            show_all=show_all,
            date_from=self._filter_date_from,
            date_to=self._filter_date_to,
            new_only=self._filter_new_only,
            sort_order=self.sort_order,
            on_device_changed=self._on_filter_device_changed,
            on_date_range_changed=self._on_filter_date_range_changed,
            on_new_only_changed=self._on_filter_new_only_changed,
            on_sort_changed=self._on_filter_sort_changed,
            on_clear=self._on_filter_clear,
        )
        self._run_filter_popover = popover
        popover.closed.connect(lambda: setattr(self, "_run_filter_popover", None))
        popover.show_anchored_to(self.cBox_Runs, main_window=self.parent)

    def _is_filter_active(self) -> bool:
        """True if any run filter (not sort - that reorders, it doesn't
        narrow the list) is currently narrowing cBox_Runs."""
        return (
            not self.showRunsFromAllDevices.isChecked()
            or bool(self._filter_date_from)
            or bool(self._filter_date_to)
            or self._filter_new_only
        )

    def _update_filter_active_indicator(self) -> None:
        """Reflects `_is_filter_active()` onto the action bar's filter
        icon/quick-clear affordance."""
        self.actionbar.set_filter_active(self._is_filter_active())

    def _on_filter_device_changed(self, device: Optional[str]) -> None:
        if device is None:
            self.showRunsFromAllDevices.setChecked(True)
        else:
            self.showRunsFromAllDevices.setChecked(False)
            self.cBox_Devices.setCurrentText(device)
        self.showRunsFromAllDevices_clicked()
        self._update_filter_active_indicator()

    def _on_filter_date_range_changed(
        self, date_from: Optional[str], date_to: Optional[str]
    ) -> None:
        self._filter_date_from = date_from
        self._filter_date_to = date_to
        self._refresh_cbox_runs()
        self._update_filter_active_indicator()

    def _on_filter_new_only_changed(self, new_only: bool) -> None:
        self._filter_new_only = new_only
        self._refresh_cbox_runs()
        self._update_filter_active_indicator()

    def _on_filter_sort_changed(self, sort_order: int) -> None:
        self.sort_order = sort_order
        self._refresh_cbox_runs()

    def _on_filter_clear(self) -> None:
        self.showRunsFromAllDevices.setChecked(True)
        self._filter_date_from = None
        self._filter_date_to = None
        self._filter_new_only = False
        self.sort_order = 1
        self.showRunsFromAllDevices_clicked()
        self._update_filter_active_indicator()

    def _switch_user_for_signature(self) -> Optional[Tuple[str, str]]:
        """Callback passed to `SignatureDialog(on_switch_user=...)`. Performs
        the actual profile switch and pushes the result into the toolbar/
        controls window; returns the new `(username, initials)` on a real
        change so the dialog can refresh its own displayed labels, or `None`
        if the switch failed or the user didn't change."""
        new_username, new_initials, new_userrole = UserProfiles.change(UserRoles.ANALYZE)
        if UserProfiles.check(UserRoles(new_userrole), UserRoles.ANALYZE):
            if self.username != new_username:
                self.username = new_username
                self.initials = new_initials
                self.parent.signature_received = False
                self.parent.signature_required = True

                Log.d("User name changed. Changing sign-in user info.")
                self.parent.controls_window.username.setText(f"User: {new_username}")
                self.parent.controls_window.userrole = UserRoles(new_userrole)
                self.parent.controls_window.signinout.setText("&Sign Out")
                self.parent.controls_window.ui.tool_User.setText(new_username)
                self.parent.analyze_window.ui.tool_User.setText(new_username)
                if self.parent.controls_window.userrole != UserRoles.ADMIN:
                    self.parent.controls_window.manage.setText("&Change Password...")
                return new_username, new_initials
            else:
                Log.d("User switched users to the same user profile. Nothing to change.")
                return None
            # PopUp.warning(self, Constants.app_title, "User has been switched.\n\nPlease sign now.")
        # elif new_username == None and new_initials == None and new_userrole == 0:
        else:
            if new_username == None and not UserProfiles.session_info()[0]:
                Log.d("User session invalidated. Switch users credentials incorrect.")
                self.parent.controls_window.username.setText("User: [NONE]")
                self.parent.controls_window.userrole = UserRoles.NONE
                self.parent.controls_window.signinout.setText("&Sign In")
                self.parent.controls_window.manage.setText("&Manage Users...")
                self.parent.controls_window.ui.tool_User.setText("Anonymous")
                self.parent.analyze_window.ui.tool_User.setText("Anonymous")
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authenticated.",
                )
            if new_username != None and UserProfiles.session_info()[0]:
                Log.d("User name changed. Changing sign-in user info.")
                self.parent.controls_window.username.setText(f"User: {new_username}")
                self.parent.controls_window.userrole = UserRoles(new_userrole)
                self.parent.controls_window.signinout.setText("&Sign Out")
                self.parent.controls_window.ui.tool_User.setText(new_username)
                self.parent.analyze_window.ui.tool_User.setText(new_username)
                if self.parent.controls_window.userrole != UserRoles.ADMIN:
                    self.parent.controls_window.manage.setText("&Change Password...")
                PopUp.warning(
                    self,
                    Constants.app_title,
                    "User has not been switched.\n\nReason: Not authorized.",
                )

            Log.d("User did not authenticate for role to switch users.")
            return None

    def action_cancel(self, exit_batched_processing_mode=False):
        if self.hasUnsavedChanges():
            if not PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to cancel without saving?",
            ):
                return

        data, rows, cols = [
            {
                "A": ["", "", "", ""],
                "B": ["", "", "", ""],
                "C": ["", "", "", ""],
                "D": ["", "", "", ""],
            },
            4,
            4,
        ]
        results_table = TableView(data, rows, cols)
        results_figure = pg.PlotWidget()
        results_figure.setBackground("w")
        plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
        plot_text.setHtml("<span style='font-size: 10pt'><b>No Results To View</b><br/> \
                            Load a run, follow the prompts to select points,<br/> \
                            and press \"Analyze\" action to view results.</span>")
        it = plot_text.textItem
        option = it.document().defaultTextOption()
        option.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        it.document().setDefaultTextOption(option)
        it.setTextWidth(it.boundingRect().width())
        plot_text.setPos(0.5, 0.5)
        results_figure.addItem(plot_text, ignoreBounds=True)

        self.graphStack.setCurrentIndex(1)
        # self.results_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.results_split.replaceWidget(0, results_table)
        self.results_split.replaceWidget(1, results_figure)
        # self.graphStack.setCurrentIndex(0)

        # Clear subset for batched processing
        if hasattr(self, "_batched_runs") and self._batched_runs and exit_batched_processing_mode:
            last_run_in_batch_loaded = False
            if self.cBox_Runs.itemText(self.cBox_Runs.count() - 1) in self._current_run:
                last_run_in_batch_loaded = True
            self._batched_runs = None
            self.showRunsFromAllDevices_clicked()

            if last_run_in_batch_loaded:
                end_reason = "All runs in the batch have been processed."
            else:
                end_reason = "User aborted the batch before it finished."

            PopUp.information(
                self,
                "Batch Processing Mode Ended",
                "You have exited batch processing mode.<br/><br/>" + f"<b>REASON: {end_reason}</b>",
            )
            # details="This is either because you have finished processing all runs in the batch " +
            # "or because you clicked \"Close\" while in the middle of processing the batch.")

        self.clear()  # calls self.enable_buttons()

    def action_back(self):
        try:
            self.step_direction = "backwards"
            self.goBack()
        except Exception as e:
            Log.e(f"An error occurred while moving to the prior step: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        self.enable_buttons()

    def action_next(self):
        try:
            self.step_direction = "forwards"
            self.getPoints()
        except Exception as e:
            Log.e(f"An error occurred while moving to the next step: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        self.enable_buttons()

    def action_modify(self):
        self.allow_modify = self.tool_Modify.isChecked()
        self.enable_buttons()

        if self.tool_Analyze.isEnabled():
            if self.allow_modify:
                self.gotoStepNum(None, 2)  # step 2: select rough points
            else:
                # self.QModel_widget.hide()
                self.gotoStepNum(None, 9)  # summary

    def action_analyze(self):
        if self.parent.signature_required and (self.unsaved_changes or self.model_run_this_load):
            if self.parent.signature_received == False and auto_sign_matches_session():
                Log.w(f"Signing ANALYZE with initials {self.initials} (not asking again)")
                self.parent.signed_at = dt.datetime.now().isoformat()
                self.parent.signature_received = True  # Do not ask again this session
            if not self.parent.signature_received:
                dlg = SignatureDialog(self, on_switch_user=self._switch_user_for_signature)
                if dlg.exec_() != QtWidgets.QDialog.Accepted:
                    return
                self.parent.signed_at = dt.datetime.now().isoformat()
                self.parent.signature_received = True
                if dlg.sign_do_not_ask.isChecked():
                    persist_auto_sign_key()

        try:
            self.moved_markers = [False, False, False, False, False, False]
            self.enable_buttons(False, False)
            results_figure = pg.PlotWidget()
            results_figure.setBackground("w")
            # 'results_split' must be shown prior to replacing
            self.graphStack.setCurrentIndex(1)
            self.results_split.replaceWidget(1, results_figure)
            self.stateStep = 6  # skip to show
            self.getPoints()  # show summary
            self.getPoints()  # show analysis
        except Exception as e:
            Log.e(f"An error occurred while analyzing the selected run: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

    def _refresh_account_button_state(self) -> None:
        """Enable/disable the Account button based on session state.

        Mirrors `UIControls.refresh_user_button_state()`; called once at
        setup and again from `check_user_info()` (run every time Analyze
        mode is opened), which is the only point the sign-in state can have
        changed since Controls/Analyze are never shown side-by-side.
        """
        signed_in = self.parent.controls_window.ui._is_user_signed_in()
        self.tool_User.setEnabled(signed_in)
        self.tool_User.setChecked(False)

        if signed_in:
            return

        popup = getattr(self, "_account_popup", None)
        if popup is not None:
            try:
                popup.close()
            except Exception:
                pass

    def _toggle_account_popup(self) -> None:
        """Toggle the account popup anchored under Analyze's Account button.

        Delegates Manage Users/Preferences/Sign Out to UIControls' existing
        implementations (`self.parent.controls_window.ui`) instead of
        reimplementing session management here - those methods only depend
        on the current UserProfiles session, not on which window's button
        was clicked. See `UIControls._toggle_account_popup` for the
        counterpart this mirrors.
        """
        anchor = self.tool_User
        controls_ui = self.parent.controls_window.ui

        prev = getattr(self, "_account_popup", None)
        if prev is not None:
            if prev.isVisible():
                anchor.setChecked(False)
                prev.close()
                return
            closed_at = getattr(self, "_account_popup_closed_at", 0.0)
            if monotonic() - closed_at < 0.25:
                anchor.setChecked(False)
                prev.deleteLater()
                self._account_popup = None
                return
            try:
                prev._enter_slide.stop()
                prev._enter_fade.stop()
            except Exception:
                pass
            prev.deleteLater()
            self._account_popup = None

        self._account_popup = AccountPopup(
            open_manager_cb=controls_ui._open_user_manager,
            open_preferences_cb=controls_ui._open_user_preferences,
            sign_out_cb=controls_ui._sign_out_current_user,
        )

        def _account_popup_closed():
            self._account_popup_closed_at = monotonic()
            anchor.setChecked(False)

        self._account_popup.closed.connect(_account_popup_closed)
        self._account_popup.destroyed.connect(lambda _=None: anchor.setChecked(False))

        anchor.setChecked(True)
        self._account_popup.show_anchored_to(anchor, main_window=self.parent)

    def _build_advanced_layout(self) -> QtWidgets.QLayout:
        """Assemble the advanced-panel widgets into a clean, sectioned layout.

        Called once during `setup_ui` to produce the `QVBoxLayout` that is
        handed to `AdvancedMainWidget.build_container`. Mirrors
        `UIControls._build_advanced_layout`'s shape (titled sections in two
        side-by-side columns) so both Advanced menus share the same look.

        Layout structure
        ----------------
        **Left column**:

        * Run Selection - `showRunsFromAllDevices` toggle above the
          `text_Devices` + `cBox_Devices` device row (dimmed via
          `_style_device_label` while the toggle is on).
        * Parameters - a Difference Factor sub-group (header row pairing
          the "Difference Factor" caption with the compact
          `difference_factor_optimizer_checkbox` "Auto-calculate" toggle,
          then `tbox_diff_factor` - an `AnimatedDoubleSpinBox` with its own
          built-in +/- chevrons - plus a range/status hint label), a
          Channel Thickness row (`tbox_ch_thick` + unit
          suffix + `h0` info icon), and a Custom POIs chip field
          (`QATCH.ui.components.poi_chip_field.POIChipField`, backed by the
          still-real but non-visible `custom_poi_text` line edit).

        **Right column**:

        * Processing - `option_remove_dups`, `drop_effect_cancelation_checkbox`,
          `partial_fills_checkbox` (the difference-factor auto-calculate
          toggle now lives in Parameters instead - see above).
        * Auto-Fit Model (`cBox_Models`)

        Returns:
            QtWidgets.QLayout: The fully assembled outer `QVBoxLayout` ready
            to be passed to `AdvancedMainWidget.build_container`.
        """

        def section(title, *rows):
            """A titled vertical group: header, hairline, then content rows."""
            col = QtWidgets.QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(6)
            col.addWidget(SectionHeader(title))
            col.addWidget(_hairline())
            for row in rows:
                if isinstance(row, QtWidgets.QLayout):
                    col.addLayout(row)
                else:
                    col.addWidget(row)
            col.addStretch()
            return col

        def hrow(*widgets, spacing=6):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(spacing)
            for w in widgets:
                if isinstance(w, QtWidgets.QLayout):
                    row.addLayout(w)
                else:
                    row.addWidget(w)
            return row

        # Left column - Run Selection + Parameters
        # "Show all available runs" now sits ABOVE the device row (was
        # below); the device caption dims via _style_device_label whenever
        # the toggle disables cBox_Devices.
        device_row = hrow(self.text_Devices, self.cBox_Devices)
        run_selection = section("Run Selection", self.showRunsFromAllDevices, device_row)

        # Small field captions (unlike SectionHeader's group titles) - kept
        # as self.* attributes so _apply_theme's caption loop can re-style
        # them on every theme change, same as footerText_hint/keys.
        self._lbl_diff_factor = QtWidgets.QLabel("Difference Factor")
        self._lbl_ch_thick = QtWidgets.QLabel("Channel Thickness")
        self._lbl_custom_poi = QtWidgets.QLabel("Custom POIs")
        self._lbl_ch_thick_unit = QtWidgets.QLabel("µm")

        # Difference Factor sub-group: header row (caption + compact
        # Auto-calculate toggle) above the value row (stepper field + hint).
        diff_header_row = QtWidgets.QHBoxLayout()
        diff_header_row.setContentsMargins(0, 0, 0, 0)
        diff_header_row.setSpacing(6)
        diff_header_row.addWidget(self._lbl_diff_factor)
        diff_header_row.addStretch()
        diff_header_row.addWidget(self.difference_factor_optimizer_checkbox)

        self._diff_hint_label = QtWidgets.QLabel()
        hint_font = QtGui.QFont()
        hint_font.setItalic(True)
        hint_font.setPixelSize(11)
        self._diff_hint_label.setFont(hint_font)
        self._set_diff_hint_text()

        diff_value_row = hrow(self.tbox_diff_factor, self._diff_hint_label)

        diff_factor_group = QtWidgets.QVBoxLayout()
        diff_factor_group.setContentsMargins(0, 0, 0, 0)
        diff_factor_group.setSpacing(6)
        diff_factor_group.addLayout(diff_header_row)
        diff_factor_group.addLayout(diff_value_row)

        ch_thick_row = hrow(
            self._lbl_ch_thick, self.tbox_ch_thick, self._lbl_ch_thick_unit, self.h0
        )

        # Custom POIs: wrapping removable-chip field, backed by the
        # existing (now non-visible) custom_poi_text line edit - see
        # QATCH.ui.components.poi_chip_field for the full backend contract.
        self._poi_chip_field = POIChipField(self.custom_poi_text, self.update_custom_pois)

        # Header row (caption + "Copied!" flash + Copy button), mirroring
        # diff_header_row's label/stretch/control layout above.
        self._poi_copied_label = QtWidgets.QLabel("Copied!")
        self._poi_copied_label.setVisible(False)
        self._poi_copy_button = QATCHPushButton("Copy", variant="ghost")
        self._poi_copy_button.setToolTip("Copy the Custom POI indices to the clipboard")
        self._poi_copy_button.clicked.connect(self._copy_custom_pois)

        custom_poi_header_row = QtWidgets.QHBoxLayout()
        custom_poi_header_row.setContentsMargins(0, 0, 0, 0)
        custom_poi_header_row.setSpacing(6)
        custom_poi_header_row.addWidget(self._lbl_custom_poi)
        custom_poi_header_row.addStretch()
        custom_poi_header_row.addWidget(self._poi_copied_label)
        custom_poi_header_row.addWidget(self._poi_copy_button)

        custom_poi_group = QtWidgets.QVBoxLayout()
        custom_poi_group.setContentsMargins(0, 0, 0, 0)
        custom_poi_group.setSpacing(6)
        custom_poi_group.addLayout(custom_poi_header_row)
        custom_poi_group.addWidget(self._poi_chip_field)

        parameters = section("Parameters", diff_factor_group, ch_thick_row, custom_poi_group)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(14)
        left_col.addLayout(run_selection)
        left_col.addLayout(parameters)
        left_col.addStretch()

        # Right column - Processing + Auto-Fit Model
        processing = section(
            "Processing",
            self.option_remove_dups,
            self.drop_effect_cancelation_checkbox,
            self.partial_fills_checkbox,
        )
        auto_fit_model = section("Auto-Fit Model", self.cBox_Models)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(14)
        right_col.addLayout(processing)
        right_col.addLayout(auto_fit_model)
        right_col.addStretch()

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(22)
        columns.addLayout(left_col, 1)
        columns.addLayout(right_col, 1)

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(2, 2, 2, 2)
        outer.addLayout(columns)
        return outer

    def action_advanced(self, obj=None) -> None:
        """Toggle the advanced control panel popup.

        Opens or closes the advanced controls widget anchored to the
        toolbar button, mirroring `UIControls.action_advanced` - closes it
        if already open, discards a very-recently-closed instance instead
        of reusing it (debounced via `_advanced_popup_closed_at`), or
        otherwise opens a fresh `AdvancedMainWidget` anchored to
        `tool_Advanced`. Pre-fills `custom_poi_text` from the currently
        loaded run's POI markers, same as the old implementation, but only
        on the "about to open" path rather than unconditionally on every
        click.
        """
        prev = getattr(self, "_advanced_popup", None)
        if prev is not None:
            if prev.isVisible():
                self.tool_Advanced.setChecked(False)
                prev.close()
                return
            closed_at = getattr(self, "_advanced_popup_closed_at", 0.0)
            if monotonic() - closed_at < 0.25:
                self.tool_Advanced.setChecked(False)
                # `prev.deleteLater()` destroys the whole popup subtree,
                # including `_advanced_content_container` if it's still
                # parented inside it (it's built once and reused across
                # every popup open via `AdvancedMainWidget.toggle`, not
                # rebuilt per-popup) - deleting them here would leave that
                # cached attribute pointing at a dead C++ object, raising
                # "wrapped C/C++ object ... has been deleted" the next
                # time it's reparented into a new popup. Detach it first
                # so only the popup shell itself gets torn down.
                content = getattr(self, "_advanced_content_container", None)
                if content is not None:
                    try:
                        content.setParent(None)
                    except RuntimeError:
                        pass
                prev.deleteLater()
                self._advanced_popup = None
                return

        try:
            poi_vals = []
            for pm in self.poi_markers:
                cur_val = pm.value()
                cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                poi_vals.append(cur_idx)
            poi_vals.sort()
            self.custom_poi_text.setText(f"{self._visible_poi_vals(poi_vals)}")
        except Exception as e:
            Log.e(
                "Error: An exception occurred while pre-filling current POIs. Is a run even loaded?"
            )
            Log.e(f"Error Details: {str(e)}")

        popup = AdvancedMainWidget.toggle(
            owner=self,
            anchor=self.tool_Advanced,
            controls_layout=self._advanced_controls_layout,
            main_window=self.parent,
        )

        if popup is None:
            self.tool_Advanced.setChecked(False)
            self._advanced_popup_closed_at = monotonic()
            return

        def _advanced_popup_closed():
            self._advanced_popup_closed_at = monotonic()
            self.tool_Advanced.setChecked(False)

        popup.closed.connect(_advanced_popup_closed)
        popup.destroyed.connect(lambda _=None: self.tool_Advanced.setChecked(False))
        self.tool_Advanced.setChecked(True)
        self.advanced_container = popup.content_container

    def enable_buttons(self, refocus: bool = True, enable: bool = True) -> None:
        """Enables or disables UI buttons based on the current state.

        This function adjusts the availability of various UI buttons based on, the presence of an XML path,
        the selected run, the current step in the state machine, whether modifications are allowed, or Whether the
        system is busy.

        Args:
            refocus (bool, optional): If True, refocuses the UI on `graphWidget2`. Defaults to True.
            enable (bool, optional): If False, disables all buttons (e.g., during processing). Defaults to True.

        Behavior:
        - If `enable` is False, all buttons are disabled.
        - The "Modify" button state is toggled based on whether `enable_cancel` is True and `enable_analyze` is False.
        - Navigation buttons ("Back" and "Next") are disabled if modifications are not allowed.
        - "Advanced" options are only enabled when `enable_cancel` is True.
        """
        # Determine initial button states
        enable_cancel = enable_info = enable_modify = self.xml_path is not None
        enable_back = self.stateStep >= 0
        enable_next = enable_cancel and self.stateStep < 7
        enable_analyze = len(self.poi_markers) > 2

        # If disabled globally (e.g., busy state), disable everything
        if not enable:
            enable_info = enable_cancel = enable_back = enable_next = enable_modify = (
                enable_analyze
            ) = False

        # Handle tool_Modify state
        if enable_cancel and not enable_analyze:
            if not self.tool_Modify.isChecked():
                self.tool_Modify.setChecked(True)
                self.tool_Modify.clicked.emit()
                self.allow_modify = True
        elif not enable_cancel:
            if self.tool_Modify.isChecked():
                self.tool_Modify.setChecked(False)
                self.tool_Modify.clicked.emit()
                self.allow_modify = False

        # If modification is not allowed, disable navigation buttons
        if not self.allow_modify:
            enable_back = enable_next = False

        # Apply button states
        self.tBtn_Info.setEnabled(enable_info)
        self.tool_Cancel.setEnabled(enable_cancel)
        self.tool_Back.setEnabled(enable_back)
        self.tool_Next.setEnabled(enable_next)
        self.tool_Modify.setEnabled(enable_modify)
        self.tool_Analyze.setEnabled(enable_analyze)

        # Handle advanced tool enabling
        self.tool_Advanced.setEnabled(enable_cancel)

        # Handle predict tool enabling
        self.tBtn_Predict.setEnabled(enable_info)

        # Refocus if required
        if refocus:
            self.graphWidget2.setFocus()

    def use_difference_factor_optimizer(self, object):
        """
        Adjusts the difference factor based on the state of the curve optimizer checkbox.

        If the curve optimizer checkbox is not checked, this method resets the
        diff factor to the default value specified in `Constants.default_diff_factor`.
        It then updates the new diff factor value by calling `self.set_new_diff_factor()`.

        Args:
            object (QWidget): The widget or object interacting with this method. Typically,
                this could represent the checkbox or related UI component triggering the event.
        """
        checked = self.difference_factor_optimizer_checkbox.isChecked()
        if not checked:
            self.tbox_diff_factor.setValue(Constants.default_diff_factor)
        # Grey out the manual field while auto-calculate is on - it
        # previously stayed enabled/editable regardless.
        self.tbox_diff_factor.setEnabled(not checked)
        if hasattr(self, "_diff_hint_label"):
            self._set_diff_hint_text()
        self.set_new_diff_factor()

    def use_drop_effect_interpolation(self, object):
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                self.diff_factor = round(self.tbox_diff_factor.value(), 3)
                Log.d(f"Difference Factor = {self.diff_factor}")
                self.load_run()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def use_drop_effect_cancelation(self, object):
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                self.diff_factor = round(self.tbox_diff_factor.value(), 3)
                Log.d(f"Difference Factor = {self.diff_factor}")
                self.load_run()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def set_new_diff_factor(self):
        """
        Sets a new difference factor from `tbox_diff_factor`'s current value.

        `AnimatedDoubleSpinBox` enforces its own [0.5, 2.0] range natively, so
        the value is always valid - no separate acceptable-input check is
        needed here anymore. Confirms any unsaved changes before updating
        `diff_factor` and refreshing the plots via `self.load_run()`.

        Exceptions:
            Logs an error message if setting the new difference factor fails.
        """
        try:
            self.action_cancel()  # ask if they mean it if there are unsaved changes
            if not self.hasUnsavedChanges():  # only proceed if they say yes
                self.diff_factor = round(self.tbox_diff_factor.value(), 3)
                Log.d(f"Difference Factor = {self.diff_factor}")
                self.load_run()  # refresh plots to show new diff factor
        except:
            Log.e("Failed to set new difference factor!")

    def set_new_ch_thick(self):
        try:
            if not self.tbox_ch_thick.hasAcceptableInput():
                Log.e(
                    "Input Error: Channel Thickness must be between {} and {} µm.".format(
                        self.validThickness.bottom(), self.validThickness.top()
                    )
                )
                return

            # Field is in micrometers; Constants.channel_thickness stays in
            # meters (the SI unit analyze_worker.py's formulas expect).
            Constants.channel_thickness = float(self.tbox_ch_thick.text()) * 1e-6
            Log.d(f"Channel thickness = {Constants.channel_thickness} m")
        except:
            Log.e("Failed to set new channel thickness!")

    def set_new_prediction_model(self, text):
        index = len(Constants.list_predict_models) - 1
        default = Constants.list_predict_models[index]
        if text in Constants.list_predict_models:
            index = Constants.list_predict_models.index(text)
        else:
            Log.e(TAG, f"Unknown predict model '{text}', using default '{default}'")
        try:
            # these flags are set above `index` as a fallback option
            Constants.qmodel_tweed_predict = True if index >= 0 else False
            Constants.qmodel_indus_predict = True if index >= 1 else False
            Constants.qmodel_volta_predict = True if index >= 2 else False
            Constants.qmodel_onyx_predict = True if index >= 3 else False
        except:
            Log.e(TAG, "Failed to set new prediction model flags in Constants.py")
        try:
            self.cBox_Models.setCurrentIndex(index)
        except:
            Log.e(TAG, "Failed to set model dropdown menu in Advanced Settings")
        try:
            self.parent.controls_window.qmodel_tweed_version.setChecked(
                True if index == 0 else False
            )
            self.parent.controls_window.qmodel_indus_version.setChecked(
                True if index == 1 else False
            )
            self.parent.controls_window.qmodel_volta_version.setChecked(
                True if index == 2 else False
            )
            self.parent.controls_window.qmodel_onyx_version.setChecked(
                True if index == 3 else False
            )
        except:
            Log.e(TAG, "Failed to check the selected prediction model in the Help menu")

    def _update_progress_value(self, value=0, status=None):
        pct = self.progressBar.value()
        if status != None:
            self.progressBar.setValue(value)
            self.progressBar.setFormat(status)
        elif pct == 0 and self.graphStack.currentIndex() == 0:
            self.progressBar.setFormat("Progress: Not Started")
        elif pct == 100:
            self.progressBar.setFormat("Progress: Finished")
        else:
            if self.analyzer_task.isRunning():
                pass  # see _update_analyze_progress() # self.progressBar.setFormat("Progress: %p%")
            elif self.graphStack.currentIndex() == 1 and self.analyze_work.exitCode() == False:
                self.progressBar.setFormat(
                    "Status: Exception during Analyze Task! (See Console for details)"
                )
                self.progress_value_steps = []
            else:
                self.progressBar.setFormat("Loading...")
        self.progressBar.repaint()

    def _update_analyze_progress(self, value, status):
        if not hasattr(self, "progress_value_scanning"):
            self.progress_value_scanning = False
        if not hasattr(self, "progress_status_step"):
            self.progress_status_step = {}
        if not value > 0:
            self.progress_status_step.clear()
        start = self.progressBar.value() + 1 if value else 0
        stops = min(100, value + 25) if value < 99 else value + 1  # 0-98:+25(100); 99-100:+1
        self.progress_value_steps = list(range(start, stops, 1 if start < stops else -1))
        self.raw_val = value
        if not status in self.progress_status_step:
            self.progress_status_step[value] = status
        if not self.progress_value_scanning:
            self.progress_value_scanning = True
            self._step_to_next_value()
        elif self.analyzer_task.isRunning():
            for key in list(self.progress_status_step.keys())[
                ::-1
            ]:  # iterate keys in reverse order
                # print(f"Check key {key} against value {value}")
                if value >= key:
                    status = self.progress_status_step.get(key)
                    # print(f"Setting status {status} @ value {key}...")
                    self.progressBar.setFormat(f"{status} %p%")
                    break  # stop after finding valid current status label
        self.progressBar.repaint()

    def _step_to_next_value(self):
        if True:
            # NOTE: 'speed' starts slow and ends fast (1, not 0)
            # <-- enter 'found' count when you search for "progress.emit" (including this one)
            total_steps = 9
            speed = max(
                1, int((total_steps - len(self.progress_status_step)) / 2)
            )  # larger numbers mean slower progressBar speed
            go_slow = len(self.progress_value_steps) < 25
            if len(self.progress_value_steps) == 0:
                self.progress_value_steps.append(100)
            if not self.analyzer_task.isRunning():
                # force fast
                speed = 0 if self.progress_value_steps[-1] > 99 else 1
            if self.progress_value_steps[-1] > 99:
                go_slow = False  # force fast
            value = self.progress_value_steps.pop(0)
            # value in self.progress_status_step.keys():
            if not self.analyzer_task.isRunning():
                keys = list(self.progress_status_step.keys())[::-1]  # in reverse order
                status = self.progress_status_step.get(keys[0])  # most recent label # .get(value)
                self.progressFormat.emit(str(f"{status} %p%"))
                # self.progressBar.setFormat(f"{status} %p%")
            if not self.analyzer_task.isRunning():
                if value != 100:
                    value *= 2
                    value = min(99, value)
            else:
                # progress_value_steps[-1]
                value = max(value, self.raw_val - 5)
            self.progressValue.emit(int(value))
            # self.progressBar.setValue(value)
            # Log.w(f"value={self.progressBar.value()}")
            self.progressUpdate.emit()
        if len(self.progress_value_steps):
            wait_time = 0.1 * speed if go_slow else 0.01 * speed
            # if wait_time > 0:
            #     sleep(wait_time)
            QtCore.QTimer.singleShot(int(1000 * wait_time), self._step_to_next_value)
        else:
            self.progress_value_scanning = False

    def hasUnsavedChanges(self):
        if hasattr(self, "unsaved_changes"):
            return self.unsaved_changes
        else:
            return False

    def isBusy(self) -> bool:
        if hasattr(self, "analyze_work"):
            if self.analyze_work.is_running():
                return True
        return False

    def force_full_resync(self):
        """Nuclear-option full rescan: wipes all cached run info and re-walks
        every device from disk. No longer wired to any button (the run list
        is normally kept current automatically by the filesystem watcher -
        see _ensure_watcher_armed/_rearm_watcher) - kept as an internal
        fallback for the watcher's own error-recovery path (e.g. a watched
        path becomes unreachable, or the incremental diff logic repeatedly
        fails) and as a hook for a future manual "force refresh" affordance
        if one is ever needed."""
        if self.hasUnsavedChanges():
            if not PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to refresh without saving?",
            ):
                return

        self.scan_for_most_recent_run = True
        self.reset()

    def reset(self):
        self.cBox_Devices.clear()

        # Rescan device folders from user preference path
        for _, dirs, _ in os.walk(os.path.join(Constants.log_prefer_path)):
            if "_unnamed" in dirs:
                dirs.remove("_unnamed")
            self.parent.data_devices = dirs  # show all available devices in logged data
            break

        self.cBox_Devices.addItems(self.parent.data_devices)
        # self.cBox_Devices.setFixedWidth(self.cBox_Devices.sizeHint().width())

        self.analyzer_task = QtCore.QThread()

        # Clear out any cached run info
        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.run_is_new = {}
        self._device_xml_confirmed_runs = set()

        # find most recent device run
        if self.scan_for_most_recent_run:
            self.scan_for_most_recent_run = False
            # call as timer to allow repaint of Analyze view mode
            QtCore.QTimer.singleShot(500, self.find_most_recent_run)

        self._refresh_cbox_runs()

        self.username = None
        self.initials = None

        self.clear()

    def clear(self):
        # Cheap no-op unless the load-directory preference changed since the
        # watcher was last armed (see _ensure_watcher_armed) - a second
        # safety net alongside the call in MainWindow.analyze_data(), in
        # case something else ever calls clear() directly.
        self._ensure_watcher_armed()

        self.text_Created.clear()
        self.graphWidget.clear()
        self.graphWidget.setTitle(None)
        self.graphWidget.showGrid(x=False, y=False)
        self.graphWidget1.clear()
        self.graphWidget1.setTitle(None)
        self.graphWidget1.showGrid(x=False, y=False)
        self.graphWidget2.clear()
        self.graphWidget2.setTitle(None)
        self.graphWidget2.showGrid(x=False, y=False)
        self.graphWidget3.clear()
        self.graphWidget3.setTitle(None)
        self.graphWidget3.showGrid(x=False, y=False)
        # .clear() can reset axis pens/background on some pyqtgraph versions
        # (see _apply_pg_theme), so without this the four plots would fall
        # back to pyqtgraph's plain white default every time Analyze mode is
        # (re-)entered, instead of the themed PlotsUI-matching background.
        self._apply_pg_theme()
        self.graphStack.setCurrentIndex(0)
        self._set_lower_graphs_visible(False)
        self.btn_Back.setEnabled(False)
        self.btn_Next.setEnabled(False)

        self.progressBar.setValue(0)  # Not started
        # self.QModel_widget.hide()
        self.setDotStepMarkers(0)

        self.stateStep = -1
        self.poi_markers = []
        self.xml_path = None  # used to indicate whether a run is loaded
        self._show_no_run_overlay()
        self.unsaved_changes = False
        self.parent.signed_at = "[NEVER]"
        self.parent.signature_required = True  # secure assumption, set on load
        self.parent.signature_received = False
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"
        self._channel_config_cache = {}

        self.check_user_info()
        self.enable_buttons()

        self.parent.viewTutorialPage([5, 6])  # analyze / prior results

    # ------------------------------------------------------------------
    # Run-list filesystem watcher - auto-maintains cBox_Devices/cBox_Runs
    # from the on-disk contents of Constants.log_prefer_path instead of
    # requiring a manual Rescan or a full rescan on every mode switch. See
    # force_full_resync() for the old/manual full-reset path, kept as an
    # internal fallback.
    # ------------------------------------------------------------------
    def _ensure_watcher_armed(self) -> None:
        """Cheap, idempotent: re-arms the watcher only if the user's load
        directory preference changed `Constants.log_prefer_path` since it
        was last armed. Safe to call on every Analyze-mode entry."""
        if self._watched_load_path == Constants.log_prefer_path:
            return
        self._rearm_watcher()

    def _rearm_watcher(self) -> None:
        """(Re)points the watcher at the current `Constants.log_prefer_path`
        and its device subdirectories, then does one full resync - the one
        case a full rescan is still correct, since the watched root's
        identity just changed and nothing cached can be trusted."""
        watched = self._run_watcher.directories()
        if watched:
            self._run_watcher.removePaths(watched)

        self._watched_load_path = Constants.log_prefer_path

        device_dirs: List[str] = []
        for _, dirs, _ in os.walk(Constants.log_prefer_path):
            device_dirs = [d for d in dirs if d != "_unnamed"]
            break

        paths_to_watch = [Constants.log_prefer_path] + [
            os.path.join(Constants.log_prefer_path, d) for d in device_dirs
        ]
        existing = [p for p in paths_to_watch if os.path.isdir(p)]
        if existing:
            self._run_watcher.addPaths(existing)

        self._full_resync(device_dirs)

    def _full_resync(self, device_dirs: Optional[List[str]] = None) -> None:
        """Rebuilds the device list and launches a full multi-device scan.
        Used only when the watched root itself changes (see
        _rearm_watcher) - everyday updates go through the cheap
        _diff_devices/_diff_runs_for_device path instead. Mirrors reset()'s
        device-list rebuild without the state/graph teardown, which is
        clear()'s separate, per-mode-entry concern."""
        if device_dirs is None:
            device_dirs = []
            for _, dirs, _ in os.walk(Constants.log_prefer_path):
                device_dirs = [d for d in dirs if d != "_unnamed"]
                break

        self.cBox_Devices.clear()
        self.parent.data_devices = device_dirs
        self.cBox_Devices.addItems(device_dirs)

        self.run_timestamps = {}
        self.run_devices = {}
        self.run_names = {}
        self.run_is_new = {}
        self._device_xml_confirmed_runs = set()

        self.find_most_recent_run()

    def _on_watched_dir_changed(self, path: str) -> None:
        """QFileSystemWatcher only reports that a directory changed, never
        what changed inside it - every fire re-lists the directory and
        diffs it (see _process_pending_watch_events), it never trusts the
        signal payload beyond "this path may be stale now"."""
        self._pending_dirty_paths.add(path)
        # Restarts if already running - coalesces bursts (e.g. many rapid
        # writes while a capture is in progress) into one pass.
        self._watch_debounce_timer.start()

    def _process_pending_watch_events(self) -> None:
        dirty = self._pending_dirty_paths
        self._pending_dirty_paths = set()
        if not dirty:
            return

        try:
            root = os.path.normcase(os.path.normpath(Constants.log_prefer_path))

            if any(os.path.normcase(os.path.normpath(p)) == root for p in dirty):
                self._diff_devices()

            for device_dir in dirty:
                if os.path.normcase(os.path.normpath(device_dir)) == root:
                    continue
                device_name = os.path.basename(device_dir)
                self._diff_runs_for_device(device_name)
        except Exception as e:
            Log.e(TAG, f"Error processing filesystem watch events: {e}")

    def _diff_devices(self) -> None:
        """Cheap directory-listing diff for the watched root: adds/removes
        device entries without touching anything that hasn't changed."""
        on_disk: set = set()
        for _, dirs, _ in os.walk(Constants.log_prefer_path):
            on_disk = {d for d in dirs if d != "_unnamed"}
            break

        known = {self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())}
        new_devices = on_disk - known
        removed_devices = known - on_disk

        for dev in sorted(new_devices):
            self.cBox_Devices.addItem(dev)
            device_path = os.path.join(Constants.log_prefer_path, dev)
            if os.path.isdir(device_path) and device_path not in self._run_watcher.directories():
                self._run_watcher.addPath(device_path)
            self._diff_runs_for_device(dev)

        for dev in removed_devices:
            self._remove_device(dev)

    def _remove_device(self, data_device: str) -> None:
        """Prunes a device that no longer exists on disk, unless it backs
        the currently-loaded run - a background watcher event must never
        silently yank an active session out from under the user."""
        if self.xml_path is not None:
            # self.xml_path is built as <log_prefer_path>/<device>/<folder>/<file>.xml
            # (see MainWindow.analyze_data), so its device is two dirnames up.
            loaded_device = os.path.basename(os.path.dirname(os.path.dirname(self.xml_path)))
            if loaded_device == data_device:
                Log.w(
                    TAG,
                    f'Device "{data_device}" was removed from disk, but its run is '
                    "currently loaded - leaving it in the list.",
                )
                return

        device_path = os.path.join(Constants.log_prefer_path, data_device)
        if device_path in self._run_watcher.directories():
            self._run_watcher.removePath(device_path)

        for dict_key in [k for k in self.run_timestamps if k.endswith(f":{data_device}")]:
            self.run_timestamps.pop(dict_key, None)
            self.run_names.pop(dict_key, None)
            self.run_is_new.pop(dict_key, None)
            self._device_xml_confirmed_runs.discard(dict_key)

        idx = self.cBox_Devices.findText(data_device)
        if idx != -1:
            self.cBox_Devices.removeItem(idx)

        self._refresh_cbox_runs()

    def _diff_runs_for_device(self, data_device: str) -> None:
        """Cheap directory-listing diff for one device: only launches a
        `RunScanWorker` (which does the real XML/zip parsing) when the set
        of run folders on disk actually differs from what's cached - the
        vast majority of watcher fires (e.g. a live capture writing into an
        already-known run folder) exit here without touching a thread."""
        on_disk = set(FileStorage.DEV_get_logged_data_folders(data_device))
        on_disk.discard("_unnamed")
        known_keys_for_dev = {k for k in self.run_timestamps if k.endswith(f":{data_device}")}
        known_folders_for_dev = {k.rsplit(":", 1)[0] for k in known_keys_for_dev}

        if on_disk == known_folders_for_dev:
            return

        known_keys: List[str] = list(self.run_timestamps.keys())
        worker = RunScanWorker([data_device], known_keys, parent=self)
        worker.scan_finished.connect(self._on_single_device_scanned)
        self._incremental_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget_incremental_worker(w))
        worker.start()

    def _forget_incremental_worker(self, worker: RunScanWorker) -> None:
        """Keeps self._incremental_workers from growing unbounded once a
        watcher-triggered scan completes. Tracked as a list rather than a
        single attribute (like the foreground single_scan_worker) because
        two devices can legitimately have watcher-triggered scans in
        flight close together."""
        if worker in self._incremental_workers:
            self._incremental_workers.remove(worker)

    def on_run_saved(self, save_root: str, data_device: str, run_directory: str) -> None:
        """Slot for `RenameOutputFilesWorker.run_saved` - fires once a run's
        files are fully renamed AND zipped into capture.zip, i.e. genuinely
        complete on disk. Forces an immediate, targeted re-scan of just this
        run instead of waiting on the filesystem watcher's own next
        incidental re-touch of that device's directory (see
        _diff_runs_for_device, which would otherwise see this run's folder
        as already-known - from the watcher's earlier "folder created"
        event - and skip re-scanning it, leaving it "Undated" indefinitely).
        """
        if os.path.normcase(os.path.normpath(save_root)) != os.path.normcase(
            os.path.normpath(Constants.log_prefer_path)
        ):
            return  # saved under a tree Analyze mode isn't watching (e.g. write path != load path)

        if data_device not in {
            self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())
        }:
            self.cBox_Devices.addItem(data_device)
            device_path = os.path.join(Constants.log_prefer_path, data_device)
            if os.path.isdir(device_path) and device_path not in self._run_watcher.directories():
                self._run_watcher.addPath(device_path)

        # Exclude just this run's key so RunScanWorker treats it as needing
        # a fresh parse, while every other already-known run for this
        # device is still skipped as usual.
        dict_key = f"{run_directory}:{data_device}"
        known_keys: List[str] = [k for k in self.run_timestamps if k != dict_key]
        worker = RunScanWorker([data_device], known_keys, parent=self)
        worker.scan_finished.connect(self._on_single_device_scanned)
        self._incremental_workers.append(worker)
        worker.finished.connect(lambda w=worker: self._forget_incremental_worker(w))
        worker.start()

    def showEvent(self, event):
        # First-show hook for _embed_stepper_overlay: see the comment at its
        # call site removal in setup_ui for why this can't just run there
        # directly. Guarded so re-entering Analyze mode later (this widget
        # gets shown/hidden repeatedly, not recreated) doesn't re-embed.
        if not getattr(self, "_stepper_overlay_embedded", False):
            self._stepper_overlay_embedded = True
            self._embed_stepper_overlay()
        super().showEvent(event)

    def closeEvent(self, event):
        if self.unsaved_changes:
            if not PopUp.question(
                self,
                Constants.app_title,
                "You have unsaved changes!\n\nAre you sure you want to close this window?",
                False,
            ):
                event.ignore()

    def check_user_info(self):
        # get active session info, if available
        active, info = UserProfiles.session_info()
        if active:
            self.parent.signature_required = True
            self.parent.signature_received = False
            self.username, self.initials = info[0], info[1]
        else:
            self.parent.signature_required = False
        self._refresh_account_button_state()

    def _set_saved_state(self, state: str, text: str) -> None:
        """Updates the "Loaded & saved" status pill's dot color and its text
        readout together, so the label always matches what the dot means
        instead of staying frozen on "Loaded & saved" regardless of state."""
        self.saved_state_dot.set_state(state)
        self.saved_state_label.setText(text)

    def detect_change(self):
        if not self.unsaved_changes:
            Log.d("There are unsaved changes detected.")
        if self.parent.signature_received:
            self.parent.signature_received = False
        self.unsaved_changes = True
        self._set_saved_state("unsaved", "Changes pending")

    """
    def AI_Prev_Guess(self):
        min_val = 0 if not self.AI_has_starting_values else -1
        cur_val = self.AI_Guess_Idxs[self.AI_SelectTool_At]
        new_val = max(min_val, cur_val - 1)
        self.AI_Guess_Idxs[self.AI_SelectTool_At] = new_val
        self.ai_guess_write_summary_from_cache()

    def AI_Next_Guess(self):
        min_val = 0 if not self.AI_has_starting_values else -1
        max_val = self.AI_Guess_Maxs[self.AI_SelectTool_At]
        cur_val = self.AI_Guess_Idxs[self.AI_SelectTool_At]
        if cur_val < min_val:
            # manually selected point will be lost
            Log.w("Manually selected point was replaced with an AI guess!")
            # Log.w("You will need to re-select the manual point if you want it back.")
            # TODO: Offer warning dialog, and allow to abort if not wanted
        new_val = min(max_val - 1, cur_val + 1)
        self.AI_Guess_Idxs[self.AI_SelectTool_At] = new_val
        self.ai_guess_write_summary_from_cache()

    def ai_guess_write_summary_from_cache(self):
        error = False
        px = self.AI_SelectTool_At
        try:
            min_val = 0 if not self.AI_has_starting_values else -1
            cur_val = self.AI_Guess_Idxs[px]
            max_val = self.AI_Guess_Maxs[px]
            if cur_val == -1:
                if self.AI_has_starting_values:
                    marker_xs = self.AI_Start_Vals[px]
                else:
                    marker_xs = self.poi_markers[px].value()
            elif (
                self.moved_markers[px] and "manual" not in self.ai_score.text(
                ).lower()
            ):  # custom point
                self.moved_markers[px] = False
                cur_val = -1
                self.AI_Guess_Idxs[px] = -1
                marker_xs = self.poi_markers[px].value()
            else:
                try:
                    marker_xs = self.xs[self.model_candidates[px][0][cur_val]]
                except Exception as e:
                    Log.e("ERROR:", e)
                    Log.e(
                        f"Failed to update prediction for POI{px} to guess #{cur_val}"
                    )
            try:
                t_idx = next(x for x, y in enumerate(
                    self.xs) if y >= marker_xs)
                index = self.xs[int(t_idx)]
                if self.poi_markers[px].value() != index:
                    Log.d(f"Moving marker {px} to position {index}")
                    # self.detect_change() # not needed if calling 'sigPositionChangeFinished' on marker move
                    self.AI_moving_marker = True
                    self.poi_markers[px].setValue(index)
                    self.poi_markers[px].sigPositionChangeFinished.emit(
                        self.poi_markers[px]
                    )
                    self.summaryAt(
                        px
                    )  # will recall this function after moving tool to new marker location
                    self.AI_moving_marker = False
                    return
                else:
                    Log.d(f"Moving marker {px} not required. Already there.")
            except Exception as e:
                Log.e(f"Moving marker {px} failed: {str(e)}")
        except Exception as e:
            error = True
            Log.e("AI ERROR:", e)
            cur_val == -1
            min_val = -1
            max_val = 0
            marker_xs = self.poi_markers[px].value()
        if cur_val == -1:
            if self.AI_has_starting_values:
                # marker_xs = self.AI_Start_Vals[px]
                self.ai_score.setText(
                    "ERROR!" if error else "Loaded from prior run")
            else:
                self.ai_score.setText("Manually selected point")
            self.ai_score.adjustSize()
            self.ai_guess.setText(f"AI has {max_val} guesses")
            self.ai_guess.adjustSize()
        else:
            # marker_xs = self.poi_markers[px].value()
            confidence = int(100 * (self.model_candidates[px][1][cur_val]))
            self.ai_score.setText(f"Confidence Score: {confidence}%")
            self.ai_score.adjustSize()
            self.ai_guess.setText(f"Guess #: {cur_val + 1} of {max_val}")
            self.ai_guess.adjustSize()
        self.ai_label.setText(f"Point {px+1} @ {marker_xs:.1f}s")
        self.ai_label.adjustSize()
        enable_prev = cur_val > min_val
        enable_next = cur_val < max_val - 1
        self.ai_backBtn.setEnabled(enable_prev)  # was 'ai_prev'
        self.ai_nextBtn.setEnabled(enable_next)  # was 'ai_next'
        self.graphWidget2.setFocus()  # allow arrow keys to work immediately

    def summaryBack(self):
        if self.stateStep in range(1, 7):
            self.action_back()
        else:
            self.summaryAt(max(0, self.AI_SelectTool_At - 1))

    def summaryNext(self):
        if self.stateStep in range(1, 7):
            self.action_next()
        else:
            self.summaryAt(min(5, self.AI_SelectTool_At + 1))

    def summaryClick(self, event):
        if self.stateStep in range(1, 7):
            if not self.AI_SelectTool_Frame.isVisible():
                self.summaryAt(self.AI_SelectTool_At)
            else:
                Log.d("Ignoring main graph click while not in 'summary' step.")
            return
        mousePoint = None
        if self.graphWidget.sceneBoundingRect().contains(event._scenePos):
            mousePoint = self.graphWidget.getPlotItem().vb.mapSceneToView(
                event._scenePos
            )
        closest_marker = None
        if mousePoint != None and len(self.poi_markers) > 0:
            index = mousePoint.x()
            Log.d(f"Mouse click @ xs = {index}")
            # find nearest POI by X value, show popup there
            closest_delta_xs = self.poi_markers[-1].value()
            for idx, marker in enumerate(self.poi_markers):
                this_delta_xs = abs(marker.value() - index)
                if this_delta_xs < closest_delta_xs:
                    closest_marker = idx
                    closest_delta_xs = this_delta_xs
        if closest_marker != None:
            Log.d(f"Closest marker @ poi = {closest_marker}")
            if self.AI_has_starting_values and not any(self.AI_Guess_Idxs):
                self.AI_Guess_Idxs = [-1, -1, -1, -1, -1, -1]
                self.AI_Start_Vals = []
                for marker in self.poi_markers:
                    self.AI_Start_Vals.append(marker.value())
                self.AI_Start_Vals.sort()
            self.summaryAt(closest_marker)

    def summaryAt(self, idx):
        if not self.AI_SelectTool_Frame.isVisible():
            self.AI_Guess_Maxs = []

            for candidates, confidences in self.model_candidates:
                self.AI_Guess_Maxs.append(len(candidates))
        self.AI_SelectTool_At = idx
        marker_xs = self.poi_markers[idx].value()
        self.ai_guess_write_summary_from_cache()
        self.AI_SelectTool_Body.adjustSize()
        self.ai_backBtn.setFixedHeight(self.AI_SelectTool_Body.height())
        self.ai_nextBtn.setFixedHeight(self.AI_SelectTool_Body.height())
        # self.AI_SelectTool_Frame.setVisible(True) # per issue #25, keep hidden
        self.AI_SelectTool_Frame.adjustSize()
        scene_pos_x = self.graphWidget.getPlotItem().vb.mapViewToScene(
            QtCore.QPointF(marker_xs, 0)
        ).x() - (self.AI_SelectTool_Frame.width() / 2.5)
        scene_pos_y = 153 + (
            (self.graphWidget.height() - self.AI_SelectTool_Frame.height()) / 2
        )
        self.AI_SelectTool_Frame.move(int(scene_pos_x), int(scene_pos_y))
        # enable_back = (self.AI_SelectTool_At > 0) # repurposed these buttons for prev/next guess
        # self.ai_backBtn.setEnabled(enable_back)
        # enable_next = (self.AI_SelectTool_At < 5)
        # self.ai_nextBtn.setEnabled(enable_next)
    """

    def onClick(self, event):
        """
        Handle a mouse click on any of the three plot widgets and move the current POI marker to the clicked x-position.

        If the click occurs inside one of the three graph widgets, the x-coordinate of the click is used to set the corresponding POI marker's value and emit its finished-move signal. The function accounts for the file's removed/hidden third POI by skipping index 2 when mapping the current stateStep to a marker index.

        Parameters:
            event (QEvent): Mouse click event from the plot scene containing the scene position.

        Side effects:
            - Updates self.poi_markers[...] by calling setValue(...) for the selected marker.
            - Emits sigPositionChangeFinished on the moved marker.
        """
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        mousePoint = None
        if ax1.sceneBoundingRect().contains(event._scenePos):
            mousePoint = ax1.getPlotItem().vb.mapSceneToView(event._scenePos)
        if ax2.sceneBoundingRect().contains(event._scenePos):
            mousePoint = ax2.getPlotItem().vb.mapSceneToView(event._scenePos)
        if ax3.sceneBoundingRect().contains(event._scenePos):
            mousePoint = ax3.getPlotItem().vb.mapSceneToView(event._scenePos)
        if mousePoint != None:
            px = self._current_visible_poi_index()
            if px < 0 or px >= len(self.poi_markers):
                return
            index = mousePoint.x()
            Log.d(f"Mouse click @ xs = {index}")
            self.poi_markers[px].setValue(index)
            self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])

    def keyPressEvent(self, event):
        key = event.key()
        if key in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return, QtCore.Qt.Key_Space]:
            if self.tool_Next.isEnabled():
                self.tool_Next.clicked.emit()
            elif self.tool_Analyze.isEnabled():
                self.tool_Analyze.clicked.emit()
        if key == QtCore.Qt.Key_Escape:
            if self.tool_Back.isEnabled():
                self.tool_Back.clicked.emit()
        elif key == QtCore.Qt.Key_Left:
            self.moveCurrentMarker(-1)
        elif key == QtCore.Qt.Key_Right:
            self.moveCurrentMarker(+1)
        elif key == QtCore.Qt.Key_Up:
            self.zoomFinderPlots(0.5)
        elif key == QtCore.Qt.Key_Down:
            self.zoomFinderPlots(2.0)

    def moveCurrentMarker(self, offset):
        """
        Move the currently selected POI marker by a number of index steps within the data x-axis.

        Parameters:
            offset (int): Number of discrete steps to move the marker; positive moves right, negative moves left. The step size is scaled by the current context width.

        Description:
            Computes which POI marker corresponds to the current step (skipping the hidden POI3 index), finds the nearest data index for that marker on self.xs, applies the offset (bounded to the valid index range), updates the marker position to the new x value, and emits the marker's sigPositionChangeFinished signal to trigger any follow-up updates. If no valid marker exists for the current step, the call does nothing.
        """
        px = self._current_visible_poi_index()
        if px < 0 or px >= len(self.poi_markers):
            return
        # 100 steps per window
        offset *= max(1, int(self.getContextWidth()[0] / 50))
        if px in range(0, len(self.poi_markers)):
            cur_val = self.poi_markers[px].value()
            new_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val) + offset
            if new_idx < 0:
                new_idx = 0
            if new_idx >= len(self.xs):
                new_idx = len(self.xs) - 1
            new_val = self.xs[new_idx]
            self.poi_markers[px].setValue(new_val)
            self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])
        else:
            pass

    def zoomFinderPlots(self, offset):
        """
        Adjust the finder-plot zoom level by a multiplicative offset and refresh the current POI context.

        This updates self.zoomLevel within safe bounds, applies special initial-clipping adjustments based
        on the current step, and emits a position-change-finished signal for the active POI marker so the
        UI refreshes. Note: the method uses the current step (self.stateStep) to select the marker but
        intentionally skips the hidden POI3 (index 2) when mapping step -> marker.

        Parameters:
            offset (float): Multiplicative zoom factor (e.g., >1 to zoom out, <1 to zoom in).

        Side effects:
            - Mutates self.zoomLevel.
            - Emits self.poi_markers[marker_index].sigPositionChangeFinished to trigger UI updates.
            - Logs warnings when zoom attempts hit configured limits or edge conditions.
        """
        if not hasattr(self, "smooth_factor"):
            Log.d("Ignoring arrow key input when no run is loaded.")
            return
        px = self._current_visible_poi_index()
        if px < 0 or px >= len(self.poi_markers):
            return
        if px in range(0, len(self.poi_markers)):
            was_clipped = self.getContextWidth()[1]
            self.zoomLevel = float(self.zoomLevel * offset)
            is_clipped = self.getContextWidth()[1]
            if was_clipped == True and is_clipped == True:
                # Compute visible ordinal 1..5; POI1/POI2 are the only "early" cases now
                visible_ord = (px if px <= 1 else px - 1) + 1
                if visible_ord <= 2:  # start, end of fill
                    self.zoomLevel = 5 * self.getContextWidth()[0] / self.smooth_factor
                else:  # blips
                    self.zoomLevel = self.stateStep * self.getContextWidth()[0] / self.smooth_factor
                Log.d(f"Adjusted initial zoom level to x{self.zoomLevel:2.2f}")
            if was_clipped == False and is_clipped == True:
                # revert to original
                self.zoomLevel = float(self.zoomLevel / offset)
                Log.w("Zoom level at edge limit. Up/down key event ignored.")
            elif self.zoomLevel < 1 / (2**5):
                self.zoomLevel = 1 / (2**5)
                Log.w("Zoom level at lower limit. Up/down key event ignored.")
            elif self.zoomLevel > 1 * (2**5):
                self.zoomLevel = 1 * (2**5)
                Log.w("Zoom level at upper limit. Up/down key event ignored.")
            self.poi_markers[px].sigPositionChangeFinished.emit(self.poi_markers[px])

    def setXmlPath(self, xml_path):
        Log.d(TAG, f"Setting xml filepath to: {xml_path}")
        self.xml_path = xml_path

    def updateDev(self, idx):
        # Disable all toolbar buttons, then selectively re-enable based on state
        self.enable_buttons(False, False)
        if self.xml_path is not None:
            self.tool_Cancel.setEnabled(True)  # enable Cancel
        run = self.cBox_Runs.currentText()
        if len(run.strip()) == 0 or run == "No Runs Found":
            # Don't mutate cBox_Runs here (clear()/addItem() would recurse
            # back into this currentIndexChanged handler) - populating the
            # "No Runs Found" placeholder is _refresh_cbox_runs's job; this
            # only needs to detect the transient/settled empty state and
            # bail without touching the combo itself.
            return
        if self.text_Created.text().endswith(run):
            self.enable_buttons()  # enable ALL buttons
        self.cBox_Runs.setEnabled(True)
        run = run[0 : run.rfind("(") - 1]
        dev = self.run_devices.get(run)
        if dev != None:
            self.cBox_Devices.setCurrentText(dev)
        else:
            Log.w(f"Device not found for run {run}")

    def updateRunOnChange(self, idx):
        if not self.showRunsFromAllDevices.isChecked():
            self.update_run(idx)

    def _on_run_activated(self, idx: int) -> None:
        """Loads a run the moment the user actually picks it from cBox_Runs
        (click or keyboard confirm) - `activated` only fires for genuine
        user interaction, never for the programmatic clear()/addItems()/
        setCurrentText() calls _refresh_cbox_runs and friends make."""
        run = self.cBox_Runs.currentText()
        if not run or run == "No Runs Found":
            return
        if self.text_Created.text().endswith(run):
            return  # already loaded - reselecting the same run is a no-op
        self.load_run()

    @staticmethod
    def _scan_run(data_device: str, data_folder: str, parse_xml: bool = True) -> Dict[str, Any]:
        """Scans a single run to fetch its file list and extract metadata.

        Worker (thread-safe): for one run, fetches its file list and, if
        parse_xml is True, extracts the 'start' metric and 'run_name'
        parameter from its XML metadata.

        This method uses ElementTree (much faster than minidom on small
        documents) and reports warnings or errors via the returned dictionary
        instead of modifying shared state directly.

        Args:
            data_device (str): The name of the device associated with the run.
            data_folder (str): The specific folder name containing the run data.
            parse_xml (bool, optional): Whether to parse the XML file for
                metadata. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the scan results. Keys include:
                - "device" (str): The device name, taken from the run XML's
                    <run_info device="..."> attribute (written at capture
                    time - see QueryRunInfoWidget) when present, so the
                    reported device is guaranteed to match what actually
                    produced the run rather than just the on-disk folder
                    it happens to be scanned from. Falls back to
                    data_device (the folder name) if the XML has no device
                    attribute (e.g. missing/legacy XML) or parse_xml=False.
                - "device_from_xml" (bool): True only if "device" above was
                    actually confirmed from the run's own XML this call -
                    used by _apply_scan_results/_prune_unconfirmed_devices
                    to tell a real device folder (has at least one XML-
                    confirmed run) apart from filesystem cruft (a stray
                    folder under log_prefer_path with no genuine capture).
                - "folder" (str): The folder name.
                - "dict_key" (str): A unique key formatted as "{folder}:{device}".
                - "files" (List[str]): A list of data files (may be empty on error).
                - "timestamp" (Optional[str]): The value of the <metric name="start">.
                - "run_name" (Optional[str]): The value of the <param name="run_name">.
                - "is_new" (bool): True if no analyze-N.zip exists yet for this run
                    (i.e. it has never been analyzed/saved).
                - "warnings" (List[str]): Warnings to be logged on the UI thread.
                - "error" (Optional[str]): Error messages to be logged via Log.e.
        """
        result = {
            "device": data_device,
            "device_from_xml": False,
            "folder": data_folder,
            "dict_key": f"{data_folder}:{data_device}",
            "files": [],
            "timestamp": None,
            "run_name": None,
            "is_new": False,
            "warnings": [],
            "error": None,
        }

        try:
            data_files = FileStorage.DEV_get_logged_data_files(data_device, data_folder)
            result["files"] = data_files or []
            # Cheap - reuses the file listing just fetched above, no extra
            # I/O - and computed unconditionally (even when parse_xml=False)
            # since it doesn't depend on XML parsing at all.
            result["is_new"] = not any(
                f.startswith("analyze-") and f.endswith(".zip") for f in result["files"]
            )

            if not parse_xml:
                return result

            root = None

            # The XML is written/rewritten in place by QueryRunInfoWidget
            # (PARAMS/audit updates re-parse and re-save it after capture),
            # so it is deliberately never bundled into capture.zip - only
            # the CSV/CRC/TEC files get zipped. Every current run on disk
            # therefore keeps its XML as a loose file next to capture.zip,
            # so that must be checked first. "audit.zip" is a legacy
            # back-compat name for pre-existing data that might have
            # actually archived the XML inside it; only fall back to
            # looking inside a zip (audit.zip, then capture.zip) if no
            # loose XML is found, instead of skipping the loose file
            # whenever any zip happens to exist - the previous zip-first
            # order caused every zipped/fully-saved run to permanently
            # show up as "Undated", since the loose XML was never checked.
            run_dir = os.path.join(Constants.log_prefer_path, data_device, data_folder)
            xml_filename = next((x for x in result["files"] if x.endswith(".xml")), None)
            if xml_filename is not None:
                xml_path = os.path.join(
                    Constants.log_prefer_path,
                    data_device,
                    data_folder,
                    xml_filename,
                )
                if os.path.exists(xml_path):
                    root = ET.parse(xml_path).getroot()

            if root is None:
                zn = next(
                    (
                        candidate
                        for candidate in (
                            os.path.join(run_dir, "audit.zip"),
                            os.path.join(run_dir, "capture.zip"),
                        )
                        if FileManager.file_exists(candidate)
                    ),
                    None,
                )
                if zn is not None:
                    with pyzipper.AESZipFile(
                        zn,
                        "r",
                        compression=pyzipper.ZIP_DEFLATED,
                        allowZip64=True,
                        encryption=pyzipper.WZ_AES,
                    ) as zf:
                        # Cheap encryption probe: read the general-purpose bit
                        # flag from the central directory instead of calling
                        # zf.testzip(), which decompresses and CRC-checks every
                        # member of the archive (including the large raw
                        # capture CSV) just to populate the run list.
                        entries = zf.infolist()
                        if entries and (entries[0].flag_bits & 0x1):
                            zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                        files = zf.namelist()
                        zip_xml_filename = next((x for x in files if x.endswith(".xml")), None)
                        if zip_xml_filename is not None:
                            with zf.open(zip_xml_filename, "r") as fh:
                                xml_bytes = fh.read()
                            root = ET.fromstring(xml_bytes)

                if root is None:
                    result["warnings"].append(
                        f'WARNING: XML file not found in data files for run "{data_folder}"'
                    )
                    result["warnings"].append(
                        'Unable to parse "Date" without XML file. Treating as "Undated".'
                    )

            if root is not None:
                for m in root.iter("metric"):
                    if m.get("name") == "start":
                        result["timestamp"] = m.get("value")
                        break
                for p in root.iter("param"):
                    if p.get("name") == "run_name":
                        result["run_name"] = p.get("value")
                        break
                # <run_info device="..."> is the device that actually
                # captured this run (set once at CAPTURE time - see
                # QueryRunInfoWidget). Prefer it over the on-disk folder
                # name so a renamed/misplaced device folder can't silently
                # misattribute a run's device in the UI.
                xml_device = root.get("device")
                if xml_device:
                    result["device"] = xml_device
                    result["device_from_xml"] = True

        except Exception as e:
            result["error"] = str(e)

        return result

    def _apply_scan_results(
        self, scan_results: List[Dict[str, Any]], data_device: str, unchecked_runs: List[str]
    ) -> None:
        """Merges the background scan results into the shared state dictionaries.

        Iterates through the parsed results from the background thread, logs any
        warnings or errors, and updates the internal state trackers (`run_names`,
        `run_timestamps`, and `run_devices`). Also handles cleaning up runs that
        are empty or no longer exist on the filesystem.

        Args:
            scan_results (List[Dict[str, Any]]): A list of dictionaries containing
                the parsed metadata for each run (typically the output from `_scan_run`).
            data_device (str): The name of the device associated with these runs.
            unchecked_runs (List[str]): A list of dict keys (formatted as
                "{folder}:{device}") representing runs that were previously known
                but need verification of their continued existence.
        """
        for r in scan_results:
            data_folder = r["folder"]
            dict_key = r["dict_key"]
            data_files = r["files"]

            for msg in r["warnings"]:
                Log.w(msg)
            if r["error"] is not None:
                Log.e(f'Error getting timestamp from XML for run "{data_folder}"!')
                Log.d(f"Error message: {r['error']}")

            if r.get("device_from_xml"):
                self._device_xml_confirmed_runs.add(dict_key)

            self.run_names[dict_key] = data_folder

            if self.run_timestamps.get(dict_key) is None:
                if r["timestamp"] is not None:
                    self.run_timestamps[dict_key] = r["timestamp"]
                if r["run_name"] is not None:
                    self.run_names[dict_key] = r["run_name"]
                    self.run_devices[r["run_name"]] = r["device"]
                # Only set on first resolution (i.e. when this run was
                # actually XML-parsed - see _scan_run's parse_xml=False
                # early-return). A later incremental rescan of an
                # already-known run skips XML parsing entirely, so r["device"]
                # would just be the raw folder name again; updating
                # unconditionally on every pass would silently discard the
                # XML-resolved device and revert to the file structure.
                self.run_devices[data_folder] = r["device"]

            if len(data_files) > 0:
                if dict_key in unchecked_runs:
                    unchecked_runs.remove(dict_key)
                if self.run_timestamps.get(dict_key) is None:
                    self.run_timestamps[dict_key] = "0 / No Date"
                # Unconditional (unlike timestamp/run_name above): the file
                # listing is always freshly fetched by _scan_run regardless
                # of whether XML re-parsing was skipped, so this cache is
                # always as current as the run's last scan.
                self.run_is_new[dict_key] = r.get("is_new", False)
            else:
                Log.w(f"Removing empty run info ({dict_key})")

        for dict_key in unchecked_runs:
            Log.w(f"Removing missing run info ({dict_key})")
            self.run_timestamps.pop(dict_key, None)
            self.run_is_new.pop(dict_key, None)

    def _prune_unconfirmed_devices(self) -> None:
        """Removes any cBox_Devices entry that has been scanned but has not
        a single run confirmed via that run's own XML `device` attribute
        (see _scan_run's device_from_xml).

        Constants.log_prefer_path can accumulate stray top-level folders
        that aren't real instrument folders at all (e.g. leftover analysis
        output moved/created outside the normal save flow) - since
        cBox_Devices is originally seeded from a plain directory listing
        (see reset/_full_resync), those show up as bogus "devices" with no
        genuine capture inside. A folder only earns removal once it's been
        scanned and found to have zero XML-confirmed runs; a device with no
        scanned runs at all (e.g. brand new, no captures yet) is left alone
        since there's no evidence either way yet.
        """
        for device in [self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())]:
            suffix = f":{device}"
            keys_for_device = {k for k in self.run_timestamps if k.endswith(suffix)}
            if not keys_for_device:
                continue
            if not (keys_for_device & self._device_xml_confirmed_runs):
                self._remove_device(device)

    def _refresh_cbox_runs(self) -> None:
        """Sorts the cached run data and repopulates the UI run selection combobox.

        This method takes the current state of `run_timestamps`, sorts them based on
        the user's selected preference (alphabetical by name or chronological by date),
        and filters them according to the selected device or active batch subsets.
        It then updates the `cBox_Runs` widget's contents (its width is fixed -
        see AnalyzeActionBar - and does not change with the run list).

        The sorting logic follows:
            - `self.sort_order = 0`: Sort by Name (Ascending)
            - `self.sort_order = 1`: Sort by Date (Descending)
            - `self.sort_order = 2`: New (filters to unanalyzed runs, Date Descending)
            - `self.sort_order = 3`: Sort by Date (Ascending)
            - `self.sort_order = 4`: Sort by Name (Descending)
        """
        # Define sorting configuration for readability: {order_index: (sort_key_index, reverse_bool)}
        # item[0] is the dict_key (name-based), item[1] is the timestamp
        sort_config = {
            0: (0, False),  # Name: Ascending
            1: (1, True),  # Date: Descending
            3: (1, False),  # Date: Ascending
            4: (0, True),  # Name: Descending
        }

        key_idx, is_reverse = sort_config.get(self.sort_order, (1, True))

        # Sort the items based on the current UI selection
        self.sorted_runs: List[Tuple[str, str]] = sorted(
            self.run_timestamps.items(), key=lambda item: item[key_idx].lower(), reverse=is_reverse
        )

        display_runs: List[str] = []
        selected_device = self.cBox_Devices.currentText()
        show_all = self.showRunsFromAllDevices.isChecked()

        for dict_key, captured_datetime in self.sorted_runs:
            # Extract device name from f"{folder}:{device}"
            device_name = dict_key.split(":")[-1] if ":" in dict_key else ""
            run_name = self.run_names.get(dict_key, dict_key)

            # Format the date string
            if captured_datetime == "0 / No Date":
                captured_date = "Undated"
            else:
                captured_date = captured_datetime.split("T")[0]

            formatted_display_name = f"{run_name} ({captured_date})"

            # Filter: Check if we are restricted to a specific batch subset
            if (
                hasattr(self, "_batched_runs")
                and self._batched_runs
                and formatted_display_name not in self._batched_runs
            ):
                continue

            # Filter: "New" sort mode (legacy) or the filter popover's
            # New/unanalyzed toggle - either shows only runs with no saved
            # analysis yet (see _scan_run's is_new / run_is_new cache).
            # Default to excluded (False) if a run has never actually been
            # scanned, so nothing shows up here before a scan confirms it.
            if (self.sort_order == 2 or self._filter_new_only) and not self.run_is_new.get(
                dict_key, False
            ):
                continue

            # Filter: date range from the filter popover ("Any" bound = no
            # constraint on that side). Undated runs never match a bounded
            # range - there's nothing to compare - but still show up when
            # no range is set.
            if (self._filter_date_from or self._filter_date_to) and captured_date == "Undated":
                continue
            if self._filter_date_from and captured_date < self._filter_date_from:
                continue
            if self._filter_date_to and captured_date > self._filter_date_to:
                continue

            # Filter: Check device ownership
            if show_all or selected_device == device_name:
                display_runs.append(formatted_display_name)

        # Update UI Components. Preserve the current selection across the
        # rebuild if it's still present - without this, a background
        # (watcher-triggered) refresh would visibly deselect the user's
        # current pick every time, even when its own entry didn't change.
        previous_selection = self.cBox_Runs.currentText()
        self.cBox_Runs.clear()
        if display_runs:
            self.cBox_Runs.addItems(display_runs)
            self.cBox_Runs.setEnabled(True)
            if previous_selection and previous_selection in display_runs:
                self.cBox_Runs.setCurrentText(previous_selection)
        else:
            # Always leave at least one (placeholder) item in place, so
            # updateDev never sees a transiently/truly empty combo outside
            # of this method's own clear()/addItems() sequence.
            self.cBox_Runs.addItem("No Runs Found")
            # Only disable when there's genuinely nothing loaded. Disabling
            # cBox_Runs also disables its embedded search/filter icons (they
            # live inside its QLineEdit, a child widget - Qt's enabled state
            # cascades down regardless of each action's own setEnabled) - if
            # an active filter is what emptied the list, the field must stay
            # enabled or the user is stranded with no way to reach the
            # filter that's hiding everything, for the rest of the session.
            keep_reachable_for_filter = self._is_filter_active()
            self.cBox_Runs.setEnabled(keep_reachable_for_filter)

        # Reset internal lookup caches if they exist
        if hasattr(self, "_run_to_folder_cache"):
            self._run_to_folder_cache = {}

        # cBox_Runs has a permanent fixed width (see AnalyzeActionBar) - it
        # used to be resized here to fit its widest current item, which made
        # the whole task bar visibly jump/reflow every refresh depending on
        # which run happened to be first/selected.

    def find_most_recent_run(self) -> None:
        """Initiates an asynchronous background scan to find the most recent run across all devices.

        This method prepares the UI by disabling buttons and initializing the progress
        bar. It then identifies all available devices from the `cBox_Devices` combobox
        and launches a `RunScanWorker` thread to scan each device's file system
        without freezing the main UI thread.

        The results are processed incrementally via the `_on_find_most_recent_scanned`
        slot as each device scan completes.
        """
        # Prevent user interaction during initial scan
        self.enable_buttons(False, False)
        self._update_progress_value(1, "Loading runs...")
        self.progressBar.setValue(0)

        # Collect device names and existing cache keys for the worker
        devices: List[str] = [
            self.cBox_Devices.itemText(i) for i in range(self.cBox_Devices.count())
        ]
        known_keys: List[str] = list(self.run_timestamps.keys())
        self._async_best_date: str = "0 / No Date"
        self._async_best_dev_idx: int = 0
        self.multi_scan_worker = RunScanWorker(devices, known_keys, parent=self)

        # Connect the signal to the handler that merges results and updates the UI
        self.multi_scan_worker.scan_finished.connect(self._on_find_most_recent_scanned)

        self.multi_scan_worker.start()

    @QtCore.pyqtSlot(list, str, list, bool)
    def _on_find_most_recent_scanned(
        self,
        scan_results: List[Dict[str, Any]],
        data_device: str,
        unchecked_runs: List[str],
        is_last: bool,
    ) -> None:
        """Handles the incremental results from the background most-recent-run scan.

        This slot is triggered whenever the `RunScanWorker` finishes scanning a
        specific device. It updates the internal state with the new results and
        tracks the globally "most recent" run across all processed devices. Once
        the final device is scanned, it finalizes the UI state.

        Args:
            scan_results (List[Dict[str, Any]]): The metadata results for the
                runs found on the scanned device.
            data_device (str): The name of the device that was just scanned.
            unchecked_runs (List[str]): List of run keys to be verified/removed
                if they no longer exist on disk.
            is_last (bool): Flag indicating if this was the last device in the
                scanning queue.
        """
        # Merge results from this device into the main state dictionaries
        self._apply_scan_results(scan_results, data_device, unchecked_runs)
        dev_idx: int = self.cBox_Devices.findText(data_device)

        if dev_idx != -1:
            for r in scan_results:
                timestamp: str = r.get("timestamp") or "0 / No Date"
                if timestamp > self._async_best_date:
                    self._async_best_date = timestamp
                    self._async_best_dev_idx = dev_idx

        if is_last:
            Log.d(
                f"Most recent run detected on device "
                f"{self.cBox_Devices.itemText(self._async_best_dev_idx)} "
                f"from {self._async_best_date}."
            )

            self.cBox_Devices.setCurrentIndex(self._async_best_dev_idx)
            # Only after every device has been scanned and the "most
            # recent" index above has already been consumed - pruning can
            # remove/shift cBox_Devices entries (see _remove_device), which
            # would otherwise invalidate _async_best_dev_idx mid-loop.
            self._prune_unconfirmed_devices()
            self._refresh_cbox_runs()
            self.enable_buttons()

    def update_run(self, idx: int) -> None:
        """Initiates an asynchronous background scan for runs associated with a specific device.

        This method serves as the entry point for refreshing the run list when a user
        selects a new device or a manual refresh is triggered. It disables UI
        interaction, identifies the target device based on the provided index, and
        offloads the file system scanning and XML parsing to a `RunScanWorker` thread.

        The results are processed by the `_on_single_device_scanned` slot once the
        background task completes.

        Args:
            idx (int): The index of the device in the `cBox_Devices` combobox
                to be scanned.
        """
        self.enable_buttons(False, False)
        data_device: str = self.cBox_Devices.itemText(idx)

        # Provide the worker with current cache keys to avoid redundant XML parsing
        known_keys: List[str] = list(self.run_timestamps.keys())
        self.single_scan_worker = RunScanWorker([data_device], known_keys, parent=self)
        self.single_scan_worker.scan_finished.connect(self._on_single_device_scanned)
        self.single_scan_worker.start()

    @QtCore.pyqtSlot(list, str, list, bool)
    def _on_single_device_scanned(
        self,
        scan_results: List[Dict[str, Any]],
        data_device: str,
        unchecked_runs: List[str],
        is_last: bool,
    ) -> None:
        """Handles the results of a background scan for a single device selection.

        This slot is triggered when the `RunScanWorker` completes the scanning
        process for the currently selected device. It updates the internal state
        dictionaries with the new results, refreshes the run selection combobox
        to reflect changes, and re-enables the UI buttons for user interaction.

        Args:
            scan_results (List[Dict[str, Any]]): A list of dictionaries containing
                parsed run metadata (timestamp, name, files, etc.) for the device.
            data_device (str): The name of the device that was scanned.
            unchecked_runs (List[str]): Keys of runs that were previously cached
                but were not found during this scan and should be purged.
            is_last (bool): Flag indicating if this was the final device in the
                worker's queue (always True for single-device updates).
        """
        self._apply_scan_results(scan_results, data_device, unchecked_runs)
        self._prune_unconfirmed_devices()
        self._refresh_cbox_runs()
        self.enable_buttons()

    def _load_qmodels(self) -> None:
        """
        Asynchronously schedules the loading of QModel predictive models.

        This method submits the heavy model-loading routines to a background thread pool
        executor (`_LOAD_EXECUTOR`). It is fully idempotent and safe to call repeatedly
        (e.g., in a polling loop or UI refresh cycle).

        Configuration constraints:
            - Models are only loaded if their respective prediction flags
            (`qmodel_onyx_predict` / `qmodel_volta_predict`) are active in `Constants`.
            - Models are skipped if they are already successfully loaded.
            - Models are skipped if a loading task is already in-flight (tracked via futures).
        """
        # Ensure future tracking attributes exist (fallback if not set in __init__)
        if not hasattr(self, "_qmodel_indus_future"):
            self._qmodel_indus_future = None
        if not hasattr(self, "_qmodel_volta_future"):
            self._qmodel_volta_future = None
        if not hasattr(self, "_qmodel_onyx_future"):
            self._qmodel_onyx_future = None

        # QModel Indus Preload
        try:
            requries_indus = getattr(Constants, "qmodel_indus_predict", False)
            is_indus_pending = self._qmodel_indus_future is not None

            if requries_indus and not self.qmodel_indus_modules_loaded and not is_indus_pending:
                self._qmodel_indus_future = _LOAD_EXECUTOR.submit(self._load_qmodel_indus)

        except Exception as e:
            Log.e("ERROR", f"Failed to schedule 'QModel Indus' preload. Details: {e}")

        # QModel Volta  Preload
        try:
            requires_volta = getattr(Constants, "qmodel_volta_predict", False)
            is_volta_pending = self._qmodel_volta_future is not None

            if requires_volta and not self.QModel_volta_modules_loaded and not is_volta_pending:
                self._qmodel_volta_future = _LOAD_EXECUTOR.submit(self._load_qmodel_volta)

        except Exception as e:
            Log.e("ERROR", f"Failed to schedule 'QModel Volta ' preload. Details: {e}")

        # QModel Onyx (onyx) Preload
        try:
            requires_onyx = getattr(Constants, "qmodel_onyx_predict", False)
            is_onyx_pending = self._qmodel_onyx_future is not None

            if requires_onyx and not self.QModel_onyx_modules_loaded and not is_onyx_pending:
                self._qmodel_onyx_future = _LOAD_EXECUTOR.submit(self._load_qmodel_onyx)

        except Exception as e:
            Log.e("ERROR", f"Failed to schedule 'QModel Onyx' preload. Details: {e}")

    @staticmethod
    def _load_qmodel_indus() -> "QModelIndus":
        """
        Instantiates and loads the QModelIndus prediction model into memory.

        This worker method performs heavy disk I/O to load PyTorch model weights
        (.pth files) for both regressors and classifiers. It is designed to be
        executed safely off the main UI thread via a concurrent futures executor.

        Returns:
            QModelIndus: A fully initialized QModelIndus prediction model ready for inference.
        """
        base_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "QModel",
            "assets",
            "qmodel_indus",
        )

        return QModelIndus(
            reg_path_1=os.path.join(base_path, "poi_model_mini_window_0_1600.pth"),
            reg_path_2=os.path.join(base_path, "poi_model_mini_window_1_1600.pth"),
            clf_path=os.path.join(base_path, "v4_model_pytorch_2100.pth"),
            reg_batch_size=2048,
            clf_batch_size=1024,
        )

    @staticmethod
    def _load_qmodel_volta() -> "QModelVolta":
        """
        Instantiates and loads the Volta prediction model into memory.

        This worker method constructs the asset map and performs heavy disk I/O
        to load the YOLO-based PyTorch weights (.pt files) for various detectors
        and classifiers. It is designed to be executed off the main UI thread.

        Returns:
            QModelVolta: A fully initialized Volta prediction model ready for inference.
        """
        base_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "QModel",
            "assets",
            "qmodel_volta",
        )

        # Map out the required weights files for the model suite
        model_assets = {
            "spacing_prior": os.path.join(base_path, "spacing_prior.json"),
            "fill_classifier": os.path.join(
                base_path, "classifiers", "fill_classifier", "type_cls.pt"
            ),
            "detectors": {
                "init": os.path.join(base_path, "detectors", "init_detector", "init.pt"),
                "ch1": os.path.join(base_path, "detectors", "ch1_detector", "ch1.pt"),
                "ch2": os.path.join(base_path, "detectors", "ch2_detector", "ch2.pt"),
                "ch3": os.path.join(base_path, "detectors", "ch3_detector", "ch3.pt"),
                "poi5_fine": os.path.join(base_path, "detectors", "eof_detector", "eof.pt"),
            },
        }

        return QModelVolta(model_assets=model_assets)

    @staticmethod
    def _load_qmodel_onyx() -> "QModelOnyx":
        """
        Instantiates and loads the QModel Onyx (onyx) prediction model into memory.

        This worker method constructs the asset map and performs heavy disk I/O
        to load the YOLO-based PyTorch weights (.pt files) for various detectors
        and classifiers, plus the onyx-specific spacing-prior decode and zoom
        refinement assets. It is designed to be executed off the main UI thread.

        Returns:
            QModelOnyx: A fully initialized onyx prediction model ready for inference.
        """
        base_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "QModel",
            "assets",
            "qmodel_onyx",
        )

        # Map out the required weights files for the model suite. The
        # zoom-refiner detectors are optional (see QModelOnyx.ZOOM_REFINE_MAP in
        # onyx_yolo.py); absent files simply keep refine_pois a no-op.
        model_assets = {
            "spacing_prior": os.path.join(base_path, "spacing_prior.json"),
            "fill_classifier": os.path.join(
                base_path, "classifiers", "fill_classifier", "type_cls.pt"
            ),
            "detectors": {
                "init": os.path.join(base_path, "detectors", "init_detector", "init.pt"),
                "ch1": os.path.join(base_path, "detectors", "ch1_detector", "ch1.pt"),
                "ch2": os.path.join(base_path, "detectors", "ch2_detector", "ch2.pt"),
                "ch3": os.path.join(base_path, "detectors", "ch3_detector", "ch3.pt"),
                "ch1_zoom": os.path.join(
                    base_path, "detectors", "ch1_zoom_detector", "ch1_zoom.pt"
                ),
                "ch2_zoom": os.path.join(
                    base_path, "detectors", "ch2_zoom_detector", "ch2_zoom.pt"
                ),
                "ch3_zoom": os.path.join(
                    base_path, "detectors", "ch3_zoom_detector", "ch3_zoom.pt"
                ),
            },
        }

        return QModelOnyx(model_assets=model_assets)

    def _await_qmodels(self, timeout: float = 120.0) -> None:
        """
        Blocks the calling thread until in-flight prediction models finish loading.

        This method resolves the asynchronous futures generated by `_load_qmodels()`.
        Once a future completes, it installs the resulting model instance onto the
        class and updates the corresponding load-state flags.

        Args:
            timeout: Maximum seconds to wait for a model load to complete before
                    a TimeoutError is raised. Defaults to 120.0 seconds.

        Notes:
            - Errors encountered during model instantiation are caught and logged
            (not raised), mimicking the original synchronous `try/except` behavior.
            - Future attributes are explicitly nulled out after resolution to
            prevent memory leaks and allow for clean reloading later.
        """
        # Resolve QModel Indus
        v4_fut = getattr(self, "_qmodel_indus_future", None)

        if v4_fut is not None and not getattr(self, "qmodel_indus_modules_loaded", False):
            try:
                # Block and wait for the thread pool to return the loaded V4 model
                self.qmodel_indus_predictor = v4_fut.result(timeout=timeout)
                self.qmodel_indus_modules_loaded = True
                Log.i(TAG, "'QModel Indus' modules loaded successfully.")
            except Exception as e:
                Log.e(TAG, f"Failed to load 'QModel Indus' modules. Details: {e}")
            finally:
                self._qmodel_indus_future = None

        # Resolve QModel Volta
        volta_fut = getattr(self, "_qmodel_volta_future", None)

        if volta_fut is not None and not getattr(self, "QModel_volta_modules_loaded", False):
            try:
                # Block and wait for the thread pool to return the loaded Volta model
                self.QModel_volta_predictor = volta_fut.result(timeout=timeout)
                self.QModel_volta_modules_loaded = True
                Log.i(TAG, "'QModel Volta ' modules loaded successfully.")
            except Exception as e:
                Log.e(TAG, f"Failed to load 'QModel Volta ' modules. Details: {e}")
            finally:
                self._qmodel_volta_future = None

        # Resolve QModel Onyx (onyx)
        onyx_fut = getattr(self, "_qmodel_onyx_future", None)

        if onyx_fut is not None and not getattr(self, "QModel_onyx_modules_loaded", False):
            try:
                # Block and wait for the thread pool to return the loaded onyx model
                self.QModel_onyx_predictor = onyx_fut.result(timeout=timeout)
                self.QModel_onyx_modules_loaded = True
                Log.i(TAG, "'QModel Onyx' modules loaded successfully.")
            except Exception as e:
                Log.e(TAG, f"Failed to load 'QModel Onyx' modules. Details: {e}")
            finally:
                self._qmodel_onyx_future = None

    def _check_dev_mode_cached(self, force_refresh: bool = False) -> bool:
        """
        Retrieves the developer mode status, utilizing a time-to-live (TTL) cache.

        This acts as a high-performance wrapper around `UserProfiles.checkDevMode()`.
        Because checking dev mode requires file system I/O, the result is cached for
        `_DEV_MODE_TTL_SECONDS` to prevent UI bottlenecks during rapid reloads or
        frequent polling.

        Args:
            force_refresh: If True, bypasses the cache, forces a fresh file system
                        read, and resets the TTL timer. Defaults to False.

        Returns:
            bool: True if developer mode is active, False otherwise.
        """
        now = monotonic()

        # Evaluate cache validity
        is_missing = not hasattr(self, "_dev_mode_cache_value")
        is_expired = (now - getattr(self, "_dev_mode_cache_time", 0.0)) > _DEV_MODE_TTL_SECONDS

        if force_refresh or is_missing or is_expired:
            # Cache miss or invalidation: fetch fresh data and reset the timer
            self._dev_mode_cache_value = UserProfiles.checkDevMode()
            self._dev_mode_cache_time = now

        return self._dev_mode_cache_value

    def _compute_folder_from_run(self, run_string: str) -> str:
        """
        Extracts the parent folder identifier from a formatted run string.

        This acts as a cache-miss fallback for resolving a folder name. It parses
        the base folder name by stripping the run suffix (e.g., "(Run 1)"), then
        performs an O(N) reverse lookup in `self.run_names` to find the corresponding
        key prefix.

        Args:
            run_string: The formatted run string (e.g., "Experiment_A (1)").

        Returns:
            str: The extracted key prefix (e.g., "KeyPrefix" from "KeyPrefix:Details"),
                or the raw parsed folder name if the reverse lookup fails.
        """
        # Parse the base folder name by stripping the trailing " (" suffix
        idx = run_string.rfind("(")
        folder_name = run_string[: idx - 1] if idx > 0 else run_string

        try:
            matching_key = next(
                key for key, value in self.run_names.items() if value == folder_name
            )
            return matching_key.split(":", 1)[0]

        except StopIteration:
            return folder_name

    def get_folder_from_run(self, run_string: str) -> str:
        """
        Retrieves the folder identifier for a given run string using a memoized cache.

        This acts as a high-performance wrapper around `_compute_folder_from_run`.
        It prevents redundant O(N) reverse dictionary lookups by caching previously
        resolved run strings in memory, which is especially useful during rapid UI
        refreshes or batch processing.

        Args:
            run_string: The formatted run string (e.g., "Experiment_A (1)").

        Returns:
            str: The extracted folder identifier or key prefix.
        """
        # Lazy-initialize the cache dictionary if it doesn't exist
        if not hasattr(self, "_run_to_folder_cache"):
            self._run_to_folder_cache = {}

        # Return the cached result immediately if it's already been computed
        if run_string in self._run_to_folder_cache:
            return self._run_to_folder_cache[run_string]

        # Cache miss
        folder = self._compute_folder_from_run(run_string)
        self._run_to_folder_cache[run_string] = folder

        return folder

    def load_all_from_folder(self, from_folder: Optional[str] = None) -> None:
        """
        Initiates a batch-loading process for all unanalyzed runs within a selected directory.

        This method scans a target directory (either the root data folder or a specific
        device folder) for runs. It uses a ThreadPoolExecutor to rapidly scan the file
        system for runs that lack an 'analyze-1.zip' file. Unanalyzed runs are queued
        up in the UI, and the first run is automatically loaded.

        Args:
            from_folder: An optional absolute path to bypass the file dialog.
                        Must be a subdirectory of `Constants.log_prefer_path`.
        """
        # Guard: Check for unsaved changes
        self.action_cancel()
        if self.hasUnsavedChanges():
            Log.d("User declined load action. There are unsaved changes.")
            return

        # Determine target directory
        selected_directory = from_folder or QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Directory", Constants.log_prefer_path
        )

        # Guard: Ensure directory is valid and within the allowed working path
        if not selected_directory:
            return
        if not selected_directory.startswith(Constants.log_prefer_path):
            Log.w("User selected an inaccessible directory for batch loading")
            Log.w("NOTE: The load directory must be within the working directory")
            return

        Log.i(f'Batch loading from "{selected_directory}"')

        # Preload ML models early since batch loads will require them
        self._load_qmodels()

        # Parse the directory depth to determine intent (Root vs Device vs Single Run)
        rel_path = os.path.relpath(selected_directory, Constants.log_prefer_path)
        path_parts = [p for p in rel_path.replace("\\", "/").split("/") if p and p != "."]

        all_runs = []

        if len(path_parts) == 0:
            # Root folder selected: gather all runs from all devices
            for dev_name in getattr(self.parent, "data_devices", []):
                for run in FileStorage.DEV_get_logged_data_folders(dev_name):
                    all_runs.append((dev_name, run))

        elif len(path_parts) == 1:
            # Device folder selected: gather all runs for this specific device
            dev_name = path_parts[0]
            for run in FileStorage.DEV_get_logged_data_folders(dev_name):
                all_runs.append((dev_name, run))

        else:
            # Single run folder selected (depth >= 2)
            Log.w("User selected a single run folder for batch loading")
            Log.w("NOTE: Use the normal Load button for single run operation")
            return

        if not all_runs:
            Log.w("No runs found in the selected folder")
            return

        # Filter out unnamed runs before dispatching to threads
        scan_items = [
            (dev, run) for dev, run in all_runs if "_unnamed" not in dev and "_unnamed" not in run
        ]

        # Parallel I/O Scan: Check for missing 'analyze-1.zip' files
        def _scan(dev_run: Tuple[str, str]) -> Tuple[str, str, Optional[List[str]]]:
            dev, run = dev_run
            try:
                files = FileStorage.DEV_get_logged_data_files(dev, run)
                return dev, run, files
            except Exception as e:
                Log.w(f"Error scanning {dev}/{run}: {e}")
                return dev, run, None

        new_runs = []
        if scan_items:
            with ThreadPoolExecutor(
                max_workers=min(8, len(scan_items)), thread_name_prefix="batch-scan"
            ) as ex:
                for dev, run, files in ex.map(_scan, scan_items):
                    if files and "analyze-1.zip" not in files:
                        new_runs.append(run)
                    elif not files:
                        Log.w(f"No files found for run: {dev}/{run}")

        Log.i(f"New runs to be analyzed: {new_runs}")
        if not new_runs:
            Log.w("No new runs require analysis.")
            return

        # Synchronize found runs with the UI Combo Box
        # Build a fast O(1) lookup dictionary: {"RunBaseName": "RunBaseName (idx)"}
        combo_texts = [self.cBox_Runs.itemText(i) for i in range(self.cBox_Runs.count())]
        run_name_to_combo_text = {text.rsplit(" ", 1)[0]: text for text in combo_texts}

        sorted_new_runs = []
        for new_run in new_runs:
            if new_run in run_name_to_combo_text:
                sorted_new_runs.append(run_name_to_combo_text[new_run])
            else:
                Log.e(f"Cannot load missing run (not found in UI): {new_run}")

        # Apply batch state and trigger the first load
        self._batched_runs = sorted_new_runs
        self.showRunsFromAllDevices_clicked()

        # Clear the folder cache since the cBox_Runs items have been entirely replaced
        if hasattr(self, "_run_to_folder_cache"):
            self._run_to_folder_cache.clear()

        Log.i(f"Loading first batch run: {self.cBox_Runs.itemText(0)} (at idx=0)")
        self.cBox_Runs.setCurrentIndex(0)
        self.btn_Load.click()

        # Notify the user
        PopUp.information(
            self,
            "Batch Processing Mode Started",
            f"<b>SUCCESS: {len(sorted_new_runs)} runs found for batch processing.</b><br/><br/>"
            "When finished analyzing a run (or to skip a run), <br/>"
            'click "Load" again to move to the next queued run.<br/>'
            "You'll get another popup when the batch is finished.",
        )

    def load_run(self) -> None:
        """
        Prepares the UI and application state for loading a new analysis run.

        This method acts as a pre-flight coordinator. It ensures unsaved changes
        are handled, verifies Developer Mode compliance (for encrypted results),
        evaluates auto-signing session keys, resets the plotting UI into a "Loading"
        state, and finally queues the actual heavy data-load operation on the event loop.
        """
        # Guard: Check for unsaved changes before proceeding
        self.action_cancel()
        if self.hasUnsavedChanges():
            Log.d("User declined load action. There are unsaved changes.")
            return

        # Kick off background preloading of ML models early
        self._load_qmodels()

        # Developer Mode & Encryption Compliance
        enabled, error, expires = self._check_dev_mode_cached()

        if not enabled and (error or expires):
            PopUp.warning(
                self,
                "Developer Mode Expired",
                "Developer Mode has expired and these analysis results will be encrypted.\n"
                "An admin must renew or disable 'Developer Mode' to suppress this warning.",
            )

        # Reset UI Control States
        self.askForPOIs = True
        self.btn_Next.setText("Next")

        # Clear plotting canvases and show the spinning "loading" overlay
        # over the Signal Overview plot (see _show_loading_run_overlay) -
        # analyze_data() also shows/hides this so every load path (manual
        # click here, or a batch auto-advance that calls action_load_run
        # directly) gets it, but showing it here too means the spinner
        # appears immediately rather than after the queued
        # action_load_run -> analyze_data hop.
        plot_elements = [self.graphWidget, self.graphWidget1, self.graphWidget2, self.graphWidget3]

        for plot_item in plot_elements:
            if plot_item is not None:
                plot_item.clear()
                plot_item.setLimits(yMin=None, yMax=None, minYRange=None, maxYRange=None)
                plot_item.setXRange(min=0, max=1)
                plot_item.setYRange(min=0, max=1)

        self._show_loading_run_overlay()

        # Reset Progress Bar and decouple previous signals
        try:
            self.progressBar.valueChanged.disconnect(self._update_progress_value)
        except Exception:
            # Fails silently if the signal was never connected in the first place
            Log.w("Cannot disconnect non-existent method from ProgressBar.")

        self.enable_buttons(False, False)
        self._update_analyze_progress(0, "Reading Run Data...")
        self._update_analyze_progress(75, "Reading Run Data...")
        QtWidgets.QApplication.processEvents()
        QtCore.QTimer.singleShot(0, self.action_load_run)

    def action_load_run(self) -> None:
        """
        Executes the final stages of loading a run and triggers data analysis.

        This method acts as the execution phase following `loadRun()`. It handles:
        1. Batch Processing Navigation: If in batch mode and the current run is
        already loaded, it auto-increments the UI to the next run in the queue,
        or exits batch mode if the queue is finished.
        2. Model Synchronization: Blocks until background ML models finish loading.
        3. Data Analysis: Dispatches the selected run to the parent coordinator
        for processing and resets UI interaction states.
        """
        try:
            # Handle Batch Processing Queue Navigation
            if getattr(self, "_batched_runs", None):
                current_text = self.cBox_Runs.currentText()
                current_idx = self.cBox_Runs.currentIndex()
                last_idx = self.cBox_Runs.count() - 1
                current_run = getattr(self, "_current_run", "")

                # If the currently selected run is already loaded, "Load" acts as a "Next" button
                if current_run and current_text in current_run:
                    if current_idx < last_idx:
                        Log.i(TAG, "Incrementing batch processing to next file in subset of list.")
                        self.cBox_Runs.setCurrentIndex(current_idx + 1)
                    else:
                        # We reached the end of the batch list
                        Log.w(TAG, "No more runs to batch process. Finished batch processing!")
                        self.action_cancel(exit_batched_processing_mode=True)
                        return

            # Synchronize ML Models
            self._await_qmodels()

            # Reset state and trigger analysis
            self.moved_markers = [False, False, False, False, False, False]

            # Use the memoized folder lookup to prevent redundant file system reads
            folder_name = self.get_folder_from_run(self.cBox_Runs.currentText())
            device_name = self.cBox_Devices.currentText()

            self.parent.analyze_data(device_name, folder_name, None)

            # Re-enable the UI controls post-analysis
            self.enable_buttons()

        except Exception as e:
            Log.e(f"An error occurred while loading the selected run: {e}")
            self.action_cancel()

    def goBack(self):
        """
        Step backwards in the analysis workflow by updating internal step state and UI, skipping the hidden POI3 step.

        If called from the final visible step, re-enables marker movement for all POI markers and decrements the step counter an extra time to bypass the hidden POI3 step. Then advances the internal state two steps backward. If the resulting state falls before the first step, resets the workflow to the initial step and restores any QModel predictions; otherwise, clears the moved-markers flags and re-enters the workflow by calling getPoints().

        Side effects:
        - Modifies self.stateStep.
        - Enables the Next button.
        - May call self._restore_qmodel_predictions() when resetting to the start.
        - Resets self.moved_markers and invokes self.getPoints().
        - Toggles movability on POI markers when stepping back from the final step.
        """
        self.btn_Next.setEnabled(True)
        if self.stateStep == 7:
            for marker in self.poi_markers:
                marker.setMovable(True)
            self.stateStep -= 1  # Skip over step 6 since POI3 is hidden
            # Log.w("State step 7 triggered")
        self.stateStep -= 2
        if self.stateStep < -1:
            self.stateStep = 0
            self._restore_qmodel_predictions()
            # if PopUp.question(
            #     self,
            #     "Are you sure?",
            #     "Any manual points will be lost if you run QModel again.\n\nProceed?",
            # ):
            # self.parent.analyze_data(
            #     self.cBox_Devices.currentText(),
            #     self.get_folder_from_run(self.cBox_Runs.currentText()),
            #     None,
            # )  # force back to step 1 of 6
            # else:
            # self.stateStep = 0
        else:
            self.moved_markers = [False, False, False, False, False, False]
            self.getPoints()

    def getContextWidth(self):
        if not hasattr(self, "smooth_factor"):
            Log.d("Ignoring arrow key input when no run is loaded.")
            return
        clipped = False
        if self.stateStep <= 2:  # start, end of fill, (no longer post point)
            ws = int(self.zoomLevel * self.smooth_factor / 2)  # context width
        else:  # blips
            ws = int(self.zoomLevel * self.smooth_factor * self.stateStep)  # context width
        if ws > len(self.xs) / 2:
            ws = int(len(self.xs) / 20)

        # Use visible-aware px
        px = self._current_visible_poi_index()
        if 0 <= px < len(self.poi_markers):
            tt1 = self.poi_markers[px].value()
        else:
            # self.poi_markers[self.AI_SelectTool_At].value()
            tt1 = self.xs[-1]

        tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
        if tx1 - ws < 0:
            clipped = True
            ws = tx1
        elif tx1 + ws >= len(self.xs):
            clipped = True
            ws = len(self.xs) - 1 - tx1
        elif ws < 10:
            clipped = True
            ws = 10
        return [ws, clipped]

    def _handle_qmodel_progress(self, pct: int, status: Optional[str]) -> None:
        """Routes a QModel predictor's progress-signal tick to the right UI.

        During the initial run load, `_show_loading_run_overlay` is already
        showing (see `analyze_data`) for the whole background-thread
        pipeline, including the QModel auto-fit step inside it - so this
        just forwards the status text onto that existing overlay instead of
        layering a second "Auto-fitting..." card + dim rect on top of it.
        Only when auto-fit runs with no run-load overlay active (e.g. the
        "Run QModel Again" button, after a run is already displayed) does it
        fall back to showing/updating its own overlay.
        """
        if getattr(self, "_loading_run_overlay", None) is not None:
            if status:
                self._set_loading_run_status(status)
            return
        if getattr(self, "_qmodel_overlay", None) is None:
            self._show_qmodel_plot_overlay()
        self._update_qmodel_plot_overlay(pct, status or "")

    def _qmodel_indus_progress_update(self, pct: int, status: Optional[str]):
        self._handle_qmodel_progress(pct, status)

    def _QModel_volta_progress_update(self, pct: int, status: Optional[str]):
        self._handle_qmodel_progress(pct, status)

    def _QModel_onyx_progress_update(self, pct: int, status: Optional[str]):
        self._handle_qmodel_progress(pct, status)

    def _cache_channel_hypotheses(
        self,
        predictor: Any,
        raw_bytes: bytes,
        detected_channels: int,
        detected_poi_vals: List[int],
    ) -> None:
        """Runs `predictor` once more per channel-count hypothesis (0-3) not
        already known - the just-completed detection call already covers
        `detected_channels` - forcing `num_channels` on each additional call
        (only Onyx/Volta support this; Indus/Tweed callers never reach this
        method). Stores the resulting 6-point POI configuration in
        `self._channel_config_cache`, keyed by channel count, so `_apply_
        cached_channel_config` can snap +/- markers straight to a model-
        predicted layout for a new channel count instead of leaving a stale
        position from a different hypothesis or parking at the data edge.

        No-ops if `detected_channels` is the predictor's "no fill at all"
        sentinel (-1) - there's no meaningful 0-3 channel comparison for a
        run with no fill.

        Deliberately silent (no progress_signal) - callers on the main
        thread manage the shared QModel overlay's status text/fade timing
        themselves around this call (see `_restore_qmodel_predictions`);
        the background run-load thread has no overlay to manage.
        """
        if detected_channels is None or detected_channels < 0:
            return
        self._channel_config_cache = {int(detected_channels): [int(v) for v in detected_poi_vals]}
        for ch in (0, 1, 2, 3):
            if ch == detected_channels:
                continue
            try:
                result, _ = predictor.predict(file_buffer=BytesIO(raw_bytes), num_channels=ch)
            except Exception as e:
                Log.w(TAG, f"[Auto-Fit] Could not cache {ch}-channel configuration: {e}")
                continue
            vals = []
            for i in range(6):
                data = result.get(f"POI{i+1}", {})
                indices = data.get("indices", [-1]) or [-1]
                vals.append(int(indices[0]))
            if vals[2] == -1 and vals[1] != -1:
                vals[2] = vals[1] + 2
            self._channel_config_cache[ch] = vals

    def _cache_channel_hypotheses_for_rerun(
        self,
        predictor: Any,
        raw_bytes: bytes,
        detected_channels: int,
        poi_vals: List[int],
    ) -> None:
        """Wraps `_cache_channel_hypotheses` for the main-thread "re-run
        auto-fit" path (`_restore_qmodel_predictions`), which - unlike the
        background run-load path - has a visible QModel overlay whose
        fade-out the detected-channel call's own 100%-progress signal
        already scheduled (see `_update_qmodel_plot_overlay`:
        `progress_signal.emit(100, "Complete!")` fires from inside
        `predictor.predict()` itself, before this method ever runs).
        Cancels that scheduled fade and updates the overlay's status text
        before the extra caching calls run, then re-triggers the normal
        fade once they're done, so the overlay stays up and honest for the
        whole operation instead of disappearing mid-computation while the
        UI is still (synchronously) busy.
        """
        self._qmodel_is_fading = False
        prev_fade = getattr(self, "_qmodel_fade_anim", None)
        if prev_fade is not None:
            try:
                prev_fade.stop()
            except RuntimeError:
                pass
        overlay = getattr(self, "_qmodel_overlay", None)
        if overlay is not None:
            overlay["status_label"].setText("Caching channel configurations…")
            QtCore.QCoreApplication.processEvents()

        self._cache_channel_hypotheses(predictor, raw_bytes, detected_channels, poi_vals)

        # All hypotheses cached - let the overlay's normal fade-out proceed
        # now (no-ops if there's no overlay to begin with).
        self._update_qmodel_plot_overlay(100, "Complete!")

    def _restore_qmodel_predictions(self):
        try:
            if self.model_engine == "None":
                # no run is loaded
                PopUp.information(
                    self,
                    "Auto-Fit Not Available",
                    "Auto-fit cannot be run at this time.\nPlease load a run first.",
                )
                # special exception case: indicates software declined action
                raise ConnectionRefusedError()
            if not PopUp.question(
                self,
                "Are you sure?",
                'Any manual points will be lost if you run "Auto-Fit" again.\n\nProceed?',
            ):
                # special exception case: indicates user declined action
                raise ConnectionAbortedError()

            # Flag used to check when finished (hides progress bar)
            self.prediction_restored = False

            self.timer = QtCore.QTimer()
            self.timer.setInterval(100)
            self.timer.setSingleShot(False)
            self.timer.timeout.connect(self.check_finished)
            self.timer.start()

            # restore QModel predictions
            poi_vals = []
            self.model_result = -1
            self.model_candidates = None
            self.model_engine = "None"
            self._channel_config_cache = {}
            if Constants.qmodel_onyx_predict:
                Log.w("Auto-fitting points with QModel Onyx... (may take a few seconds)")
                QtCore.QCoreApplication.processEvents()
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        raw_bytes = f.read()
                        fh = BytesIO(raw_bytes)
                        predictor = self.QModel_onyx_predictor
                        predict_result, detected_channels = predictor.predict(
                            file_buffer=fh, progress_signal=self.onyx_predict_progress
                        )
                        # Restoring predictions restores the channel count.
                        self.parent.num_channels = detected_channels
                        Log.i(
                            TAG,
                            f"QModel Onyx Inference Complete. Detected Config: {detected_channels} Channel(s)",
                        )
                        predictions = []
                        candidates = []
                        for i in range(6):
                            poi_key = f"POI{i+1}"
                            data = predict_result.get(poi_key, {})
                            indices = data.get("indices", [-1])
                            confidences = data.get("confidences", [-1])
                            if not indices:
                                indices = [-1]
                            if not confidences:
                                confidences = [-1]
                            predictions.append(indices[0])
                            candidates.append((indices, confidences))
                        self.model_run_this_load = True
                        self.model_result = predictions
                        self.model_candidates = candidates
                        self.model_engine = f"Onyx - {detected_channels}ch"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            poi_vals = self.model_result.copy()
                            if poi_vals[2] == -1 and poi_vals[1] != -1:
                                # Correct POST point to End-of-fill + 2
                                poi_vals[2] = poi_vals[1] + 2
                            self._cache_channel_hypotheses_for_rerun(
                                predictor, raw_bytes, detected_channels, poi_vals
                            )
                        else:
                            self.model_result = -1  # Invalid result format

                except Exception as e:
                    import traceback

                    Log.e(TAG, f"Error using 'QModel Onyx': {e}")
                    for line in traceback.format_tb(sys.exc_info()[2]):
                        Log.d(line.strip())
                    self.model_result = -1  # Trigger fallback handling
                    # raise e
            if self.model_result == -1 and Constants.qmodel_volta_predict:
                Log.w("Auto-fitting points with QModel Volta ... (may take a few seconds)")
                QtCore.QCoreApplication.processEvents()
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        raw_bytes = f.read()
                        fh = BytesIO(raw_bytes)
                        predictor = self.QModel_volta_predictor
                        # self._QModel_create_new_progress_dialog()
                        # self.progressBarDiag.setRange(0, 100)
                        predict_result, detected_channels = predictor.predict(
                            file_buffer=fh, progress_signal=self.volta_predict_progress
                        )
                        # Restoring predictions restores the channel count.
                        self.parent.num_channels = detected_channels
                        Log.i(
                            TAG,
                            f"QModel Volta Inference Complete. Detected Config: {detected_channels} Channel(s)",
                        )
                        predictions = []
                        candidates = []
                        for i in range(6):
                            poi_key = f"POI{i+1}"
                            data = predict_result.get(poi_key, {})
                            indices = data.get("indices", [-1])
                            confidences = data.get("confidences", [-1])
                            if not indices:
                                indices = [-1]
                            if not confidences:
                                confidences = [-1]
                            predictions.append(indices[0])
                            candidates.append((indices, confidences))
                        self.model_run_this_load = True
                        self.model_result = predictions
                        self.model_candidates = candidates
                        self.model_engine = f"Volta  - {detected_channels}ch"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            poi_vals = self.model_result.copy()
                            if poi_vals[2] == -1 and poi_vals[1] != -1:
                                # Correct POST point to End-of-fill + 2
                                poi_vals[2] = poi_vals[1] + 2
                            self._cache_channel_hypotheses_for_rerun(
                                predictor, raw_bytes, detected_channels, poi_vals
                            )
                        else:
                            self.model_result = -1  # Invalid result format

                except Exception as e:
                    import traceback

                    Log.e(TAG, f"Error using 'QModel Volta ': {e}")
                    for line in traceback.format_tb(sys.exc_info()[2]):
                        Log.d(line.strip())
                    self.model_result = -1  # Trigger fallback handling
                    # raise e
            if self.model_result == -1 and Constants.qmodel_indus_predict:
                Log.w("Auto-fitting points with QModel Indus... (may take a few seconds)")
                QtCore.QCoreApplication.processEvents()
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        fh = BytesIO(f.read())
                        predictor = self.qmodel_indus_predictor
                        # self._QModel_create_new_progress_dialog()
                        # self.progressBarDiag.setRange(0, 100)  # percentage
                        predict_result = predictor.predict(
                            file_buffer=fh,
                            visualize=False,
                            progress_signal=self.indus_predict_progress,
                            use_partial_fills=self.partial_fills_checkbox.isChecked(),
                        )
                        predictions = []
                        candidates = []
                        for i in range(6):
                            poi_key = f"POI{i+1}"
                            poi_indices = predict_result.get(poi_key, {}).get("indices", [])
                            poi_confidences = predict_result.get(poi_key, {}).get("confidences", [])
                            best_pair = (poi_indices[0], poi_confidences[0])
                            predictions.append(best_pair[0])
                            candidates.append((poi_indices, poi_confidences))
                        self.model_run_this_load = True
                        self.model_result = predictions
                        self.model_candidates = candidates
                        self.model_engine = "Indus"
                        if isinstance(self.model_result, list) and len(self.model_result) == 6:
                            poi_vals = self.model_result.copy()
                            if poi_vals[2] == -1 and poi_vals[1] != -1:
                                # Correct POST point to End-of-fill + 2
                                poi_vals[2] = poi_vals[1] + 2
                        else:
                            self.model_result = -1  # try fallback model
                except Exception as e:
                    limit = None
                    t, v, tb = sys.exc_info()
                    from traceback import format_tb

                    a_list = ["Traceback (most recent call last):"]
                    a_list = a_list + format_tb(tb, limit)
                    a_list.append(f"{t.__name__}: {str(v)}")
                    for line in a_list:
                        Log.d(line)
                    Log.e(e)
                    Log.e(
                        TAG,
                        f"Error using 'QModel Indus'... Using a fallback model for auto-fitting.",
                    )
                    raise e  # debug only
                    self.model_result = -1  # try fallback model

            if self.model_result == -1 and Constants.qmodel_tweed_predict:
                try:
                    with secure_open(self.loaded_datapath, "r", "capture") as f:
                        csv_headers = next(f)

                        if isinstance(csv_headers, bytes):
                            csv_headers = csv_headers.decode()

                        if "Ambient" in csv_headers:
                            csv_cols = (2, 4, 6, 7)
                        else:
                            csv_cols = (2, 3, 5, 6)

                        data = loadtxt(f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols)
                    relative_time = data[:, 0]
                    # temperature = data[:, 1]
                    resonance_frequency = data[:, 2]
                    dissipation = data[:, 3]

                    self.model_run_this_load = True
                    self.model_result = self.qmodel_tweed_predictor.IdentifyPoints(
                        self.loaded_datapath,
                        relative_time,
                        resonance_frequency,
                        dissipation,
                    )
                    self.model_engine = "Tweed"
                    if isinstance(self.model_result, list):
                        poi_vals.clear()
                        # show point with highest confidence for each:
                        self.model_select = []
                        self.model_candidates = []
                        for point in self.model_result:
                            self.model_select.append(0)
                            if isinstance(point, list):
                                self.model_candidates.append(point)
                                select_point = point[self.model_select[-1]]
                                select_index = select_point[0]
                                select_confidence = select_point[1]
                                poi_vals.append(select_index)
                            else:
                                self.model_candidates.append([point])
                                poi_vals.append(point)
                    elif self.model_result == -1:
                        Log.w("Model failed to auto-calculate POIs for this run!")
                        pass
                    else:
                        Log.e(
                            "Model encountered an unexpected response. Please manually select points."
                        )
                        pass
                except:
                    limit = None
                    t, v, tb = sys.exc_info()
                    from traceback import format_tb

                    a_list = ["Traceback (most recent call last):"]
                    a_list = a_list + format_tb(tb, limit)
                    a_list.append(f"{t.__name__}: {str(v)}")
                    for line in a_list:
                        Log.e(line)

            if self.model_result != -1 and len(self.poi_markers) == 6:
                Log.i(f"[Auto-Fit] Auto-fit points with '{self.model_engine}' for this run.")
                for i, pm in enumerate(self.poi_markers):
                    idx = int(poi_vals[i])

                    if idx == -1:
                        # Mark "missing" points at end of data
                        idx = len(self.xs) - 1
                    elif idx <= 0:
                        Log.w(f"[Auto-Fit] Clamped POI{i+1} index {idx} to {1}")
                        idx = 1
                    elif idx >= len(self.xs):
                        Log.w(f"[Auto-Fit] Clamped POI{i+1} index {idx} to {len(self.xs)-1}")
                        idx = len(self.xs) - 1

                    # Update marker position to new index
                    pm.setValue(self.xs[idx])

                self._log_model_confidences()
                self.detect_change()
                # If this (re-)run of auto-fit found real positions for
                # channels beyond what +/- currently shows, reveal those
                # steps/markers rather than leaving them hidden under a
                # stale "not present" assumption - see docstring.
                self._reveal_steps_for_poi_vals(poi_vals)
            else:
                Log.w(
                    "[Auto-Fit] No auto-fit points available for this run. Leaving points unchanged."
                )
        except ConnectionRefusedError:
            Log.d("Attempt to auto-fit with no run loaded. No action taken.")

        except ConnectionAbortedError:
            Log.d("User declined auto-fit restore prompt. No action taken.")

        except Exception as e:
            Log.e(f"Auto-fit restore failed: {str(e)}")

            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            self.prediction_restored = True

    def _current_visible_poi_index(self):
        px = self.stateStep - 1
        return px if px < 2 else px + 1  # skip POI3 (index 2)

    def check_finished(self):
        if self.prediction_restored:
            # finished, but keep the dialog open to retain `wasCanceled()` state
            QtCore.QTimer.singleShot(1000, self._hide_qmodel_plot_overlay)
            self.timer.stop()

    def getPoints(self):
        """
        Advance the analysis workflow by one step, updating UI, markers, plots, and progress state.

        This method drives the point-selection workflow (with POI3 intentionally hidden). It:
        - Advances or clamps the internal step counter and maps it to the visible tutorial/step index (skipping the removed POI3).
        - Updates progress text, enables/disables navigation buttons, and switches the main graph view.
        - On initial step, attempts to auto-populate POI candidates using available prediction engines (QModel v3, QModel v2, QModel Tweed) and creates/mutates vertical POI markers when needed.
        - For intermediate steps it restricts which POI marker is movable, recenters/zooms the context plots around the current POI (skipping POI3), and updates star/current-point indicators and small-context plots.
        - On the summary/analysis step it finalizes marker positions, validates signatures if required, persists POIs and audit information to the run XML when there are unsaved changes, and launches the AnalyzeWorker in a background thread to run the heavy analysis pipeline.
        - Ensures any hidden POI (index 2 / POI3) is not shown or edited and adjusts all marker index calculations accordingly.

        Side effects:
        - Mutates many UI widgets and internal state (stateStep, poi_markers, moved_markers, model_result, model_candidates, model_engine, zoomLevel, gstars/star plots, progress bar, etc.).
        - May write <audit> and <points> entries to the run XML when changes are saved.
        - May start a background AnalyzeWorker thread that performs the full analysis and emits progress signals.

        No return value.
        """
        self.graphStack.setCurrentIndex(0)
        self.btn_Back.setEnabled(True)
        if self.stateStep != 7:
            self.btn_Next.setText("Next")
        self.stateStep += 1  # Increment to next step
        if self.stateStep == 6:  # Skip over hidden dot marker 8
            # Log.w("State step 6 trigger")
            self.stateStep = 7  # straight to Summary
        # Hide POI3 from UI steps: skip step 4 (POI3)
        # There are originally 6 points (POI1-POI6), POI3 is at index 2
        # When stepping, skip index 2
        step_num = self.stateStep + 2
        # Calculate the visible step index, skipping POI3
        visible_step = self._current_visible_poi_index() + 1
        # Only show 5 points to the user
        if step_num < 3 and self.tool_Modify.isChecked():
            self.parent.viewTutorialPage(7)  # analyze (summary)
        elif step_num in range(3, 8 + 1) and self.tool_Modify.isChecked():
            # Show 7.1, 7.2, 7.4, 7.5, 7.6 (skip 7.3)
            tutorial_ids = [round(7 + (visible_step) / 10, 2)]
            if visible_step in range(1, 7):
                tutorial_ids.append(7.7)
            self.parent.viewTutorialPage(tutorial_ids)  # analyze (precise point)
        else:  # "Modify" not checked or step_num > 8
            self.parent.viewTutorialPage([5, 6])  # analyze / prior results
        ax = self.graphWidget  # .plot(hour, temperature)
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        # Only show 5 points (skip POI3)
        w123 = self.stateStep in range(1, 6)
        self._set_lower_graphs_visible(w123)
        self._update_detail_point_cloud_state()
        # was_vis = ax1.isVisible()
        # if w123 and not was_vis:
        #     ax2.setFocus() # allow keyboard shortcuts left/right/up/down to work immediately
        # ax1.setVisible(w123)
        # ax2.setVisible(w123)
        # ax3.setVisible(w123)
        # When stateStep == 0, normal behavior
        if self.stateStep == 0:
            self._update_progress_value(
                12 * (step_num - 1),
                f"Step {step_num - 1} of 6: Select Rough Fill Points",
            )
            ax.setTitle(None)
            ax.setXRange(0, self.xs[-1], padding=0.05)
            ax.setYRange(
                0,
                max(
                    np.amax(self.ys_freq_fit),
                    np.amax(self.ys_fit),
                    np.amax(self.ys_diff_fit),
                ),
                padding=0.05,
            )
            self.fit1.setAlpha(1, False)
            self.fit2.setAlpha(1, False)
            self.fit3.setAlpha(1, False)
            self.scat1.setAlpha(0.01, False)
            self.scat2.setAlpha(0.01, False)
            self.scat3.setAlpha(0.01, False)

            # QModel Tweed
            poi_vals = []
            if len(self.poi_markers) != 6:
                self.model_result = -1
                self.model_candidates = None
                self.model_engine = "None"
                if Constants.qmodel_tweed_predict:
                    Log.w("Auto-fitting points with QModel Onyx... (may take a few seconds)")
                    QtCore.QCoreApplication.processEvents()
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            predictor = self.QModel_onyx_predictor
                            predict_result, detected_channels = predictor.predict(
                                file_buffer=fh, progress_signal=self.onyx_predict_progress
                            )
                            if not self.parent.num_channels:
                                self.parent.num_channels = detected_channels
                            Log.i(
                                TAG,
                                f"QModel Onyx Inference Complete. Detected Config: {detected_channels} Channel(s)",
                            )

                            predictions = []
                            candidates = []
                            for i in range(6):
                                poi_key = f"POI{i+1}"
                                data = predict_result.get(poi_key, {})
                                indices = data.get("indices", [-1])
                                confidences = data.get("confidences", [-1])
                                if not indices:
                                    indices = [-1]
                                if not confidences:
                                    confidences = [-1]
                                predictions.append(indices[0])
                                candidates.append((indices, confidences))
                            self.model_run_this_load = True
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = f"Onyx - {detected_channels}ch"
                            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                                poi_vals = self.model_result.copy()
                                if poi_vals[2] == -1 and poi_vals[1] != -1:
                                    # Correct POST point to End-of-fill + 2
                                    poi_vals[2] = poi_vals[1] + 2
                            else:
                                self.model_result = -1  # Invalid result format

                    except Exception as e:
                        # --- ERROR HANDLING ---
                        import traceback

                        Log.e(TAG, f"Error using 'QModel Onyx': {e}")
                        # Print full stack trace to debug log
                        for line in traceback.format_tb(sys.exc_info()[2]):
                            Log.d(line.strip())
                        self.model_result = -1  # Trigger fallback handling
                        # raise e # Uncomment for strict debugging
                if self.model_result == -1 and Constants.qmodel_volta_predict:
                    Log.w("Auto-fitting points with QModel Volta ... (may take a few seconds)")
                    QtCore.QCoreApplication.processEvents()
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            predictor = self.QModel_volta_predictor
                            # self._QModel_create_new_progress_dialog()
                            # self.progressBarDiag.setRange(0, 100)
                            predict_result, detected_channels = predictor.predict(
                                file_buffer=fh, progress_signal=self.volta_predict_progress
                            )
                            if not self.parent.num_channels:
                                self.parent.num_channels = detected_channels
                            Log.i(
                                TAG,
                                f"QModel Volta Inference Complete. Detected Config: {detected_channels} Channel(s)",
                            )

                            predictions = []
                            candidates = []
                            for i in range(6):
                                poi_key = f"POI{i+1}"
                                data = predict_result.get(poi_key, {})
                                indices = data.get("indices", [-1])
                                confidences = data.get("confidences", [-1])
                                if not indices:
                                    indices = [-1]
                                if not confidences:
                                    confidences = [-1]
                                predictions.append(indices[0])
                                candidates.append((indices, confidences))
                            self.model_run_this_load = True
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = f"Volta  - {detected_channels}ch"
                            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                                poi_vals = self.model_result.copy()
                                if poi_vals[2] == -1 and poi_vals[1] != -1:
                                    # Correct POST point to End-of-fill + 2
                                    poi_vals[2] = poi_vals[1] + 2
                            else:
                                self.model_result = -1  # Invalid result format

                    except Exception as e:
                        # --- ERROR HANDLING ---
                        import traceback

                        Log.e(TAG, f"Error using 'QModel Volta ': {e}")
                        # Print full stack trace to debug log
                        for line in traceback.format_tb(sys.exc_info()[2]):
                            Log.d(line.strip())
                        self.model_result = -1  # Trigger fallback handling
                        # raise e # Uncomment for strict debugging
                if self.model_result == -1 and Constants.qmodel_indus_predict:
                    Log.w("Auto-fitting points with QModel Indus... (may take a few seconds)")
                    QtCore.QCoreApplication.processEvents()
                    try:
                        with secure_open(self.loaded_datapath, "r", "capture") as f:
                            fh = BytesIO(f.read())
                            predictor = self.qmodel_indus_predictor
                            predict_result = predictor.predict(
                                file_buffer=fh,
                                visualize=False,
                                progress_signal=self.indus_predict_progress,
                                use_partial_fills=self.partial_fills_checkbox.isChecked(),
                            )

                            predictions = []
                            candidates = []
                            for i in range(6):
                                poi_key = f"POI{i+1}"
                                poi_indices = predict_result.get(poi_key, {}).get("indices", [])
                                poi_confidences = predict_result.get(poi_key, {}).get(
                                    "confidences", []
                                )
                                best_pair = (poi_indices[0], poi_confidences[0])
                                predictions.append(best_pair[0])
                                candidates.append(best_pair)
                            self.model_result = predictions
                            self.model_candidates = candidates
                            self.model_engine = "Indus"
                            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                                poi_vals = self.model_result.copy()
                                if poi_vals[2] == -1 and poi_vals[1] != -1:
                                    # Correct POST point to End-of-fill + 2
                                    poi_vals[2] = poi_vals[1] + 2
                            else:
                                self.model_result = -1  # try fallback model
                    except Exception as e:
                        limit = None
                        t, v, tb = sys.exc_info()
                        from traceback import format_tb

                        a_list = ["Traceback (most recent call last):"]
                        a_list = a_list + format_tb(tb, limit)
                        a_list.append(f"{t.__name__}: {str(v)}")
                        for line in a_list:
                            Log.d(line)
                        Log.e(e)
                        Log.e(
                            "Error using 'QModel Indus'... Using a fallback model for auto-fitting."
                        )
                        raise e  # debug only
                        self.model_result = -1  # try fallback model

                if self.model_result == -1 and Constants.qmodel_tweed_predict:
                    try:
                        start_time = poi_vals[0] if len(poi_vals) > 0 else 0
                        stop_time = poi_vals[5] if len(poi_vals) > 5 else len(self.xs) - 1
                        model_starting_points = [
                            start_time,
                            None,
                            None,
                            None,
                            None,
                            stop_time,
                        ]
                        self.model_result = self.qmodel_tweed_predictor.IdentifyPoints(
                            data_path=self.loaded_datapath,
                            times=self.data_time,
                            freq=self.data_freq,
                            diss=self.data_diss,
                            start_at=model_starting_points,
                        )
                        self.model_engine = "Tweed"
                        if isinstance(self.model_result, list):
                            poi_vals.clear()
                            # show point with highest confidence for each:
                            self.model_select = []
                            self.model_candidates = []
                            for point in self.model_result:
                                self.model_select.append(0)
                                if isinstance(point, list):
                                    self.model_candidates.append(point)
                                    select_point = point[self.model_select[-1]]
                                    select_index = select_point[0]
                                    select_confidence = select_point[1]
                                    poi_vals.append(select_index)
                                else:
                                    self.model_candidates.append([point])
                                    poi_vals.append(point)
                        elif self.model_result == -1:
                            Log.w("Model failed to auto-calculate POIs for this run!")
                            pass
                        else:
                            Log.e(
                                "Model encountered an unexpected response. Please manually select points."
                            )
                            pass
                    except:
                        limit = None
                        t, v, tb = sys.exc_info()
                        from traceback import format_tb

                        a_list = ["Traceback (most recent call last):"]
                        a_list = a_list + format_tb(tb, limit)
                        a_list.append(f"{t.__name__}: {str(v)}")
                        for line in a_list:
                            Log.e(line)

                try:  # if isinstance(self.model_result, list):
                    poi2_time = self.xs[poi_vals[1]]  # end of fill
                    poi3_time = self.xs[poi_vals[2]]  # post
                    poi4_time = self.xs[poi_vals[3]]  # blip1
                    poi5_time = self.xs[poi_vals[4]]  # blip2
                except:  # else:
                    Log.e("Model returned insufficient points. Please manually select points.")
                    start_time = poi_vals[0] if len(poi_vals) > 0 else 0
                    stop_time = poi_vals[5] if len(poi_vals) > 5 else len(self.xs) - 1
                    fill_time = self.xs[stop_time] - self.xs[start_time]
                    poi2_time = self.xs[start_time] + (fill_time * 0.05)  # end of fill
                    poi3_time = self.xs[start_time] + (fill_time * 0.10)  # post
                    poi4_time = self.xs[start_time] + (fill_time * 0.25)  # blip1
                    poi5_time = self.xs[start_time] + (fill_time * 0.50)  # blip2

                self.moved_markers = [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]  # model can adjust all points on next step

            else:  # all points already set
                for pm in self.poi_markers:
                    cur_val = pm.value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    poi_vals.append(cur_idx)

            if len(self.poi_markers) != 6:
                self.detect_change()

                y0, y1 = self._data_y_range(self.ys, self.ys_freq, self.ys_diff)
                for pt in [poi2_time, poi3_time, poi4_time, poi5_time]:
                    poi_marker = self._make_poi_marker(pt, self.xs, y0, y1)
                    ax.addItem(poi_marker)
                    poi_marker.sigPositionChangeFinished.connect(self.markerMoveFinished)
                    self.poi_markers.insert(-1, poi_marker)
            for idx, marker in enumerate(self.poi_markers):
                marker.setMovable(True)
                self._style_poi_marker(marker, active=True)
                if idx == 2:
                    marker.setVisible(False)
            # self.AI_SelectTool_Frame.setVisible(False)  # Hide AI Tool
        # Only allow steps for POI1, POI2, POI4, POI5, POI6 (skip POI3)
        elif self.stateStep in range(1, 7):
            if self.stateStep + 2 == 3:  # stateStep 1 = Step 3 of 6
                # sort poi_markers by time, in case the user messed up the order moving things around manually in Step 2
                out_of_order = False
                for i in range(1, len(self.poi_markers)):
                    if self.poi_markers[i - 1].value() > self.poi_markers[i].value():
                        Log.d("Detected POI markers are out-of-order... sorting...")
                        out_of_order = True
                        break  # no need to keep searching, the order is wrong, so fix it
                if out_of_order:
                    try:
                        poi_vals = []
                        for pm in self.poi_markers:
                            cur_val = pm.value()
                            cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                            poi_vals.append(cur_idx)
                        poi_vals.sort()
                        self.custom_poi_text.setText(f"{self._visible_poi_vals(poi_vals)}")
                        self.update_custom_pois()  # write POI markers in correct order
                    except Exception as e:
                        Log.e("Error: An exception occurred while sorting POI markers.")
                        Log.e(f"Error Details: {str(e)}")

                poi_vals = []
                for pm in self.poi_markers:  # already sorted
                    cur_val = pm.value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    poi_vals.append(cur_idx)

                if self.model_engine == "Tweed" and Constants.qmodel_tweed_predict:
                    try:
                        # Run Model again, to get an initial automatic fine tuning of points prior to user input
                        model_starting_points = poi_vals.copy()  # NOTE: len(poi_vals) must equal 6
                        self.model_result = self.qmodel_tweed_predictor.IdentifyPoints(
                            data_path=self.loaded_datapath,
                            times=self.data_time,
                            freq=self.data_freq,
                            diss=self.data_diss,
                            start_at=model_starting_points,
                        )
                        self.model_engine = "Tweed"
                        if isinstance(self.model_result, list):
                            poi_vals.clear()
                            # show point with highest confidence for each:
                            self.model_select = []
                            self.model_candidates = []
                            for idx, point in enumerate(self.model_result):
                                if isinstance(point, list):
                                    self.model_candidates.append(point)
                                else:
                                    self.model_candidates.append([point])
                                self.model_select.append(0)
                                if self.moved_markers[idx] == False:
                                    poi_vals.append(model_starting_points[idx])
                                else:
                                    self.moved_markers[idx] = False
                                    if isinstance(point, list):
                                        select_point = point[self.model_select[-1]]
                                        select_index = select_point[0]
                                        select_confidence = select_point[1]
                                        poi_vals.append(select_index)
                                    else:
                                        poi_vals.append(point)
                            # Track which markers have been moved and only update model for those points, otherwise take starting point
                            if poi_vals != model_starting_points:
                                # Updating custom POIs also re-writes the POI markers
                                self.custom_poi_text.setText(f"{self._visible_poi_vals(poi_vals)}")
                                self.update_custom_pois()  # write POI markers in correct order
                                self.moved_markers = [
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                    False,
                                ]
                        elif self.model_result == -1:
                            Log.w("Model failed to auto-calculate POIs for this run!")
                            pass
                        else:
                            Log.e(
                                "Model encountered an unexpected response. Please manually select points."
                            )
                            pass
                    except:
                        Log.e("An error occurred while running the model and organizing markers.")

                    # sort poi_markers one more time, just in case model returned out-of-order points (which should never happen)
                    out_of_order = False
                    for i in range(1, len(self.poi_markers)):
                        if self.poi_markers[i - 1].value() > self.poi_markers[i].value():
                            Log.d("Detected POI markers are out-of-order... sorting...")
                            out_of_order = True
                            break  # no need to keep searching, the order is wrong, so fix it
                    if out_of_order:
                        try:
                            poi_vals = []
                            for pm in self.poi_markers:
                                cur_val = pm.value()
                                cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                                poi_vals.append(cur_idx)
                            poi_vals.sort()
                            self.custom_poi_text.setText(f"{self._visible_poi_vals(poi_vals)}")
                            self.update_custom_pois()  # write POI markers in correct order
                        except Exception as e:
                            Log.e("Error: An exception occurred while sorting POI markers.")
                            Log.e(f"Error Details: {str(e)}")

                else:  # self.model_engine != "Tweed":
                    # do nothing here if "QModel v2" or "None"
                    pass

            # in stateStep 2 thru 6 (Steps 4 thru 8 of 6, skipping POI3)
            elif self.stateStep != 7:
                if self.stateStep == 3:
                    cur_val = self.poi_markers[self.stateStep - 2].value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    new_idx = min(cur_idx + 2, len(self.xs) - 1)
                    if new_idx > cur_idx:
                        self.poi_markers[self.stateStep - 1].setValue(self.xs[int(new_idx)])
                    else:
                        Log.d(
                            "Current marker cannot be bumped forward without exceeding data bounds; leaving as-is."
                        )
                px = self._current_visible_poi_index()
                if px < 0 or px >= len(self.poi_markers):
                    return
                if self.poi_markers[px].value() < self.poi_markers[px - 1].value():
                    cur_val = self.poi_markers[px - 1].value()
                    cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
                    new_idx = min(cur_idx + 2, len(self.xs) - 1)
                    if new_idx > cur_idx:
                        self.poi_markers[self.stateStep - 1].setValue(self.xs[int(new_idx)])
                    else:
                        Log.d(
                            "Current marker cannot be bumped forward without exceeding data bounds; leaving as-is."
                        )
            self.zoomLevel = 1  # reset default zoom level for each point
            show_fits = 1.0 if self.stateStep >= 3 else 0.0
            show_scat = 0.1 if self.stateStep >= 3 else 1.0
            pad = 0.05 if self.stateStep >= 3 else 0.05
            self.fit_1.setAlpha(show_fits, False)
            self.fit_2.setAlpha(show_fits, False)
            self.fit_3.setAlpha(show_fits, False)
            self.scat_1.setAlpha(show_scat, False)
            self.scat_2.setAlpha(show_scat, False)
            self.scat_3.setAlpha(show_scat, False)
            # px is the index in poi_markers, skip POI3 (index 2)
            px = self._current_visible_poi_index()
            visible_ord = (px if px <= 1 else px - 1) + 1  # 1..5
            self._update_progress_value(
                12 * (step_num - 1),
                f"Step {step_num - 1} of 6: Select Precise Fill Point {visible_ord}",
            )
            ax.setTitle(None)
            if px < 0 or px >= len(self.poi_markers):
                return
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt1 = self.poi_markers[px].value()
            tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ws = self.getContextWidth()[0]
            # Calculate safe index boundaries prior to setting ranges
            slice_start, slice_end = [tx1 - ws, tx1 + ws]
            clipped = False
            if slice_start < 0:
                slice_start = 0
                clipped = True
            if slice_end > len(self.xs) - 1:
                slice_end = len(self.xs) - 1
                clipped = True
            if slice_start >= slice_end:
                slice_start = slice_end - 1  # 0
                slice_end = slice_start + 1  # len(self.xs) - 1
                clipped = True
            ax.setXRange(self.xs[tx0], self.xs[tx2], padding=0.12)
            ax1.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax2.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax3.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            # Prevent empty slices
            if tx0 >= tx2:
                tx0 = 0
                tx2 = len(self.xs) - 1
            mn = min(
                np.amin(self.ys_freq_fit[tx0:tx2]),
                np.amin(self.ys_fit[tx0:tx2]),
                np.amin(self.ys_diff_fit[tx0:tx2]),
            )
            mx = max(
                np.amax(self.ys_freq_fit[tx0:tx2]),
                np.amax(self.ys_fit[tx0:tx2]),
                np.amax(self.ys_diff_fit[tx0:tx2]),
            )
            ax.setYRange(mn, mx, padding=pad)
            if self.stateStep >= 3:
                if not clipped:
                    ax1.setYRange(
                        np.min(self.ys_freq_fit[slice_start:slice_end]),
                        np.max(self.ys_freq_fit[slice_start:slice_end]),
                        padding=pad,
                    )
                    ax2.setYRange(
                        np.min(self.ys_diff_fit[slice_start:slice_end]),
                        np.max(self.ys_diff_fit[slice_start:slice_end]),
                        padding=pad,
                    )
                    ax3.setYRange(
                        np.min(self.ys_fit[slice_start:slice_end]),
                        np.max(self.ys_fit[slice_start:slice_end]),
                        padding=pad,
                    )
                else:  # clipped
                    Log.d(
                        "Skipping to next step, due to missing channel in data selection (represented by ValueError exception below):"
                    )
                    # skip to next view
                    Log.w(
                        f"Skipping Step {self.stateStep+2}... User indicated this point is missing from the dataset in Step 2."
                    )
                    if self.step_direction == "backwards":
                        self.action_back()  # repeat last action
                    else:
                        self.action_next()  # repeat last action
                    return  # do not execute remainder of this function, let the above nested 'action_next' call supercede
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq_fit[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff_fit[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys_fit[tx1]))
            else:
                ax1.setYRange(
                    np.min(self.ys_freq[slice_start:slice_end]),
                    np.max(self.ys_freq[slice_start:slice_end]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff[slice_start:slice_end]),
                    np.max(self.ys_diff[slice_start:slice_end]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys[slice_start:slice_end]),
                    np.max(self.ys[slice_start:slice_end]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys[tx1]))
            self.star1.setData(pos=pos1)
            self.star2.setData(pos=pos2)
            self.star3.setData(pos=pos3)
            gstar_idxs = []
            for idx, marker in enumerate(self.poi_markers):
                # Skip POI3 (index 2) for UI
                if idx == 2:
                    continue
                if (
                    idx == px - 1
                ):  # check last point, move this marker if it's out of time sequence from last one
                    if (
                        marker.value() >= self.poi_markers[px].value()
                    ):  # last marker time greater than this marker
                        t_idx = next(x for x, y in enumerate(self.xs) if y >= marker.value())
                        marker.setValue(self.xs[t_idx + 3])
                if idx != px:
                    t_idx = next(x for x, y in enumerate(self.xs) if y >= marker.value())
                    gstar_idxs.append(t_idx)
                marker.setMovable(idx == px)  # only current marker is movable
                self._style_poi_marker(marker, active=(idx == px))
            if self.stateStep >= 3:
                pos1 = np.column_stack((self.xs[gstar_idxs], self.ys_freq_fit[gstar_idxs]))
                pos2 = np.column_stack((self.xs[gstar_idxs], self.ys_diff_fit[gstar_idxs]))
                pos3 = np.column_stack((self.xs[gstar_idxs], self.ys_fit[gstar_idxs]))
            else:
                pos1 = np.column_stack((self.xs[gstar_idxs], self.ys_freq[gstar_idxs]))
                pos2 = np.column_stack((self.xs[gstar_idxs], self.ys_diff[gstar_idxs]))
                pos3 = np.column_stack((self.xs[gstar_idxs], self.ys[gstar_idxs]))
            self.gstars1.setData(pos=pos1)
            self.gstars2.setData(pos=pos2)
            self.gstars3.setData(pos=pos3)
            # # Show AI Tool on current point marker after everything settles:
            # QtCore.QTimer.singleShot(
            #     1, lambda: self.summaryAt(max(0, min(5, self.stateStep - 1)))
            # )
        elif self.stateStep == 7:
            self._update_progress_value(
                100,
                f'Summary: Press "Analyze" to compute results for these selected points',
            )
            # ax.setTitle(f"Summary: All Selected POIs")
            self.fit1.setAlpha(1, False)
            self.fit2.setAlpha(1, False)
            self.fit3.setAlpha(1, False)
            self.scat1.setAlpha(0.01, False)
            self.scat2.setAlpha(0.01, False)
            self.scat3.setAlpha(0.01, False)
            for marker in self.poi_markers:
                marker.setMovable(False)
                self._style_poi_marker(marker, active=True)
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ax.setXRange(self.xs[tx0], self.xs[tx2], padding=0.12)
            # Prevent empty slices
            if tx0 >= tx2:
                tx0 = 0
                tx2 = len(self.xs) - 1
            mn = min(
                np.amin(self.ys_freq_fit[tx0:tx2]),
                np.amin(self.ys_fit[tx0:tx2]),
                np.amin(self.ys_diff_fit[tx0:tx2]),
            )
            mx = max(
                np.amax(self.ys_freq_fit[tx0:tx2]),
                np.amax(self.ys_fit[tx0:tx2]),
                np.amax(self.ys_diff_fit[tx0:tx2]),
            )
            ax.setYRange(mn, mx, padding=0.05)
            for i, marker in enumerate(self.poi_markers):
                Log.d(f"Marker {i} = ", marker.value())
            self.btn_Next.setText("Analyze")
            # self.AI_SelectTool_Frame.setVisible(False)  # Hide AI Tool
        else:
            self.stateStep = 8
            if self.unsaved_changes:
                if self.parent.signature_required and not self.parent.signature_received:
                    Log.e(f"Input Error: Initials do not match current user info ({self.initials})")
                    return
            self.btn_Back.setEnabled(True)
            self.btn_Next.setEnabled(False)
            poi_vals = []
            for marker in self.poi_markers:
                t_idx = next(x for x, y in enumerate(self.xs) if y >= marker.value())
                poi_vals.append(t_idx)
            poi_vals.sort()
            if self.unsaved_changes:
                Log.d("Storing new <points> in XML file")
                self.unsaved_changes = False
                # Optimistic - appendAuditToXml/appendPointsToXml flip this
                # back to "error" (and re-flag unsaved_changes via
                # detect_change()) if the actual XML write fails below.
                self._set_saved_state("saved", "Loaded & saved")
                if self.parent.signature_required:
                    self.appendAuditToXml()
                self.appendPointsToXml(poi_vals)
            # self.showAnalysis(poi_vals)
            # self.analyzer_task = threading.Thread(target=self.showAnalysis, args=(poi_vals,))
            # self.analyzer_task.start()
            allow_start = True
            if hasattr(self, "analyze_work"):
                if self.analyze_work.is_running():
                    Log.w("Double-click detected on Analyze action. Skipping duplicate action.")
                    allow_start = False
            if allow_start:
                self._update_progress_value(1, "Status: Starting...")
                self.graphStack.setCurrentIndex(1)
                self.analyzer_task = QtCore.QThread()
                self.analyze_work = AnalyzeWorker(
                    self,  # pass in parent
                    self.loaded_datapath,
                    self.xml_path,
                    poi_vals,
                    self.diff_factor if hasattr(self, "diff_factor") else None,
                )
                self.analyzer_task.started.connect(self.analyze_work.run)
                self.analyze_work.finished.connect(self.analyzer_task.quit)
                self.analyze_work.progress.connect(self._update_analyze_progress)
                self.analyze_work.finished.connect(self._update_progress_value)
                self.analyze_work.finished.connect(self.enable_buttons)

                # New progress dialog popup instead of run progress bar...
                self._show_analyze_plot_overlay()  # creates figure + overlay on main thread
                self.analyze_work.progress.connect(self._update_analyze_plot_overlay)
                self.analyze_work.finished.connect(self._hide_analyze_plot_overlay)
                # self._create_analyze_progress_dialog()
                # self.analyze_work.progress.connect(self._update_analyze_popup_progress)
                # self.analyze_work.finished.connect(self._close_analyze_progress_dialog)

                self.analyzer_task.start()
        self.setDotStepMarkers(step_num)

        # # Show/Hide QModel re-run button if on Step 2 and run has prior points
        # if step_num == 2 and len(self.poi_markers) == 6:
        #     self._position_floating_widget()
        #     self.QModel_widget.show()
        # elif self.QModel_widget.isVisible():
        #     self.QModel_widget.hide()

    # Maps Stepper index (0..6) -> legacy 1-based step_num used everywhere
    # else in this class (gotoStepNum, stateStep arithmetic, etc).
    #
    # step_num 1 is the status dot (now AnalyzeActionBar's saved-state pill,
    # not a Stepper entry at all) and 8 is permanently hidden ("for POI3
    # removal"), so neither appears here.
    #
    # 5/6/7 used to be captioned "Post"/"Blip 1"/"Blip 2", which was wrong:
    # gotoStepNum(step_num) -> stateStep = step_num - 3, then getPoints()
    # increments it once more and hands the movable marker to
    # _current_visible_poi_index() - working that through for 5/6/7 lands on
    # POI4/POI5/POI6, i.e. the 1st/2nd/3rd channel fill points (confirmed
    # against QATCH.common.tutorials.TutorialPages[7.4/7.5/7.6]), not "Post"
    # (POI3, which is hidden everywhere and never user-editable) or a
    # generic "Blip". Renamed to Channel 1/2/3 to match what they actually
    # jump to.
    #
    # step_num 9 ("Blip 3" in the old 8-dot layout) is intentionally absent
    # too, but unlike 1/8 it's still a value getPoints() hands to
    # setDotStepMarkers() in the ordinary course of reaching the Summary
    # view - see the step_num == 9 special case in setDotStepMarkers, which
    # handles it directly rather than through this table (it was never a
    # sensible *dot* - clicking it as "Blip 3" landed on Summary, not a
    # channel point - but the numeric signal itself is real and needed).
    _STEP_NUMS = [2, 3, 4, 5, 6, 7, 10]

    def _on_stepper_clicked(self, index: int) -> None:
        """Adapter from Stepper.stepClicked(index) to the legacy
        gotoStepNum(obj, step_num) call every other navigation path uses.

        `index` is the stepper's own *actual* clicked position, which can
        be fewer than `_STEP_NUMS`'s 7 slots once some intermediate steps
        are hidden via "-" (see `_INTERMEDIATE_STEPS`) - the clicked pill
        is always either "Load" (position 0, unchanged), a currently-
        visible intermediate step (whose `_STEP_NUMS` slot is numerically
        identical to its stepper position, since hidden steps are always
        the tail *before* "Analyze" - nothing before them ever shifts), or
        "Analyze" itself - always the stepper's actual last position, but
        not necessarily `_STEP_NUMS` index 6 once some are hidden.
        """
        last = self.stepper.step_count() - 1
        step_num = self._STEP_NUMS[-1] if index >= last else self._STEP_NUMS[index]
        self.gotoStepNum(None, step_num)

    def _pillstepper_index_for(self, step_nums_index: int) -> int:
        """Maps an index into the fixed 7-slot `_STEP_NUMS` table to
        `self.stepper`'s own actual current pill index - see
        `_on_stepper_clicked` for why the two can differ once some
        intermediate steps are hidden via "-".
        """
        last = self.stepper.step_count() - 1
        if step_nums_index >= len(self._STEP_NUMS) - 1:
            return last
        return min(step_nums_index, last)

    def setDotStepMarkers(self, step_num):
        if step_num == 0:
            # No run loaded / reset - clear both the status dot and all
            # step progress.
            self._set_saved_state("blank", "No run loaded")
            self.stepper.reset()
            return
        if step_num == 1:
            # Run loaded/saved; wizard hasn't stepped into the numbered
            # steps yet for this load, so clear any prior step progress.
            self._set_saved_state("saved", "Loaded & saved")
            self.stepper.reset()
            return
        if step_num == 9:
            # getPoints() reaches the Summary view (stateStep 7) this way on
            # *every* path that gets there - not just organically stepping
            # through Channel 1/2/3 one at a time, but also the "model found
            # all six POIs on load" shortcut in _advance_analysis_step, which
            # jumps stateStep straight to 6 and calls getPoints() once,
            # skipping 0-5 (and their set_current calls) entirely. Every POI
            # already has a value once you're at Summary, so treat the last
            # dot (Analyze) as reached here too - otherwise, on that
            # auto-advance path, _max_reached never leaves 0 and the pill is
            # unclickable until the user clicks Next once (which reaches
            # step_num 10 and finally bumps it).
            self.stepper.set_current(self.stepper.step_count() - 1)
            return
        try:
            index = self._STEP_NUMS.index(step_num)
        except ValueError:
            Log.w(f"{TAG} setDotStepMarkers: unexpected step_num {step_num}")
            return
        self.stepper.set_current(self._pillstepper_index_for(index))

    def gotoStepNum(self, obj, step_num=1):
        """
        Navigate to a given analysis step using the step-dot controls and update UI state.

        This method interprets a clicked step dot (or the provided step_num) and advances or rewinds the AnalyzeProcess workflow accordingly. It enforces modify-mode rules, sets the step direction, handles the special "finished" step (10), validates prerequisites (loaded run and POIs), and triggers the appropriate actions: restoring QModel predictions for step 1, invoking getPoints() to move into the selected step, toggling modify mode via action_modify(), or emitting the next-button when appropriate. Side effects include updating self.stateStep, self.step_direction, progress bar text/value, enabling/disabling controls, showing/hiding graph panes, and calling other UI handlers (setDotStepMarkers, enable_buttons, _restore_qmodel_predictions, action_modify, getPoints).

        Parameters:
            obj: The UI object that triggered the call (unused - kept for
                signature compatibility with the direct saved-state-dot
                click wiring; step navigation from the numbered stepper
                comes through the _on_stepper_clicked adapter instead).
            step_num (int): Target step dot index (1-based, legacy numbering
                - see _STEP_NUMS).

        Notes:
        - If modify mode is disabled and the target step is less than 9, the method forces modify mode and returns (action_modify will re-enter this method).
        - Step 10 is treated as "Finished" and moves the UI to the results view.
        - Requires a loaded run (self.xml_path) and at least three POI markers to jump to analysis steps; otherwise it logs a warning and no step change occurs.
        """
        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(False)

        if self.allow_modify == False and step_num < 9:
            self.tool_Modify.setChecked(True)
            self.action_modify()  # self.tool_Modify.clicked.emit()
            return  # action_modify() always calls this function again

        # determine step direction
        if step_num < self.stateStep + 2:
            self.step_direction = "backwards"
        else:
            self.step_direction = "forwards"

        if step_num == 10:
            self.progressBar.setValue(100)  # Finished
            self.progressBar.setFormat('Finished: View most recent "Analyze" results')
            self.stateStep = 8
            self.tool_Cancel.setEnabled(True)
            self._set_lower_graphs_visible(False)
            self.graphStack.setCurrentIndex(1)
            self.setDotStepMarkers(step_num)
            return

        enable_cancel = self.xml_path != None
        enable_analyze = len(self.poi_markers) > 2
        if not enable_cancel:
            Log.w("Please load a run prior to using the step jumper dots.")
        elif self.stateStep + 2 == step_num:
            Log.d("User clicked step jumper dot of current step. No action.")
        elif step_num == 1:
            self._restore_qmodel_predictions()
            self.enable_buttons()
            # if PopUp.question(
            #     self,
            #     "Are you sure?",
            #     "Any manual points will be lost if you run QModel again.\n\nProceed?",
            # ):
            # self.parent.analyze_data(
            #     self.cBox_Devices.currentText(),
            #     self.get_folder_from_run(self.cBox_Runs.currentText()),
            #     None,
            # )  # force back to step 1 of 6
            # self.enable_buttons()
        elif enable_analyze:
            self.stateStep = step_num - 3
            self.getPoints()  # increment to next step
            self.enable_buttons()
        elif enable_analyze == False and step_num == 2:
            # special case: allow next action if dot is clicked instead of button
            self.tool_Next.clicked.emit()  # calls enable_buttons()
        else:
            Log.w("Please select begin and end points prior to using the step jumper dots.")

    def appendAuditToXml(self):
        data_path = self.loaded_datapath
        xml_path = data_path[0:-4] + ".xml" if self.xml_path == None else self.xml_path
        xml_params = {}
        if secure_open.file_exists(xml_path, "audit"):
            xml_text = ""
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_text = f.read()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()
            run = minidom.parseString(xml_text)
            xml = run.documentElement

            # create or append new audits element
            try:
                audits = xml.getElementsByTagName("audits")[-1]
            except:
                audits = run.createElement("audits")
                xml.appendChild(audits)

            valid, infos = UserProfiles.session_info()
            if valid:
                Log.d(f"Found valid session: {infos}")
                username = infos[0]
                initials = infos[1]
                salt = UserProfiles.find(username, initials)[1][:-4]
                userrole = infos[2]
            else:
                Log.w(f"Found invalid session: searching for user ({self.initials})")
                username = None  # not known in this context (yet)
                initials = self.initials
                salt = UserProfiles.find(username, initials)[1][:-4]
                userinfo = UserProfiles.get_user_info(f"{salt}.xml")
                username = userinfo[0]
                initials = userinfo[1]
                userrole = userinfo[2]

            audit_action = "ANALYZE"
            timestamp = self.parent.signed_at
            machine = Architecture.get_os_name()
            hash = hashlib.sha256()
            hash.update(salt.encode())  # aka 'profile'
            hash.update(audit_action.encode())
            hash.update(timestamp.encode())
            hash.update(machine.encode())
            hash.update(username.encode())
            hash.update(initials.encode())
            hash.update(userrole.encode())
            signature = hash.hexdigest()

            audit1 = run.createElement("audit")
            audit1.setAttribute("profile", salt)
            audit1.setAttribute("action", audit_action)
            audit1.setAttribute("recorded", timestamp)
            audit1.setAttribute("machine", machine)
            audit1.setAttribute("username", username)
            audit1.setAttribute("initials", initials)
            audit1.setAttribute("role", userrole)
            audit1.setAttribute("signature", signature)
            audits.appendChild(audit1)

            hash = hashlib.sha256()
            a_tags = xml.getElementsByTagName("audit")
            for i, a in enumerate(a_tags):
                if i == len(a_tags) - 1:
                    ref_signature = hash.hexdigest()
                if a.hasAttribute("signature"):
                    hash.update(a.getAttribute("signature").encode())
            audits_signature = hash.hexdigest()
            if audits.hasAttribute("signature"):
                if audits.getAttribute("signature") == ref_signature:
                    audits.setAttribute("signature", audits_signature)
                else:
                    if audits.attributes["signature"].value.find("X") < 0:
                        audits.attributes["signature"].value += "X"
                    Log.e(
                        "Audits signature does not match for this run! Unable to apply new signature."
                    )
            else:
                audits.setAttribute("signature", audits_signature)

            try:
                with open(xml_path, "w", encoding="utf-8") as f:
                    xml_str = run.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore")
                    f.write(xml_str)
                    Log.d(f"Added <audit> to XML file: {xml_path}")
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Filesystem error writing XML: {xml_path}")
                Log.e("Error Details:", ose.strerror)
                self.detect_change()
                self._set_saved_state("error", "Error saving")
                self.saved_state_dot.flash()
            except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                Log.e(f"Unicode error writing XML: {xml_path}")
                Log.e("Error Details:", ue.reason)
                self.detect_change()
                self._set_saved_state("error", "Error saving")
                self.saved_state_dot.flash()

    def appendPointsToXml(self, poi_vals):
        data_path = self.loaded_datapath
        xml_path = data_path[0:-4] + ".xml" if self.xml_path == None else self.xml_path
        xml_params = {}
        if secure_open.file_exists(xml_path, "audit"):
            xml_text = ""
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_text = f.read()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()
            run = minidom.parseString(xml_text)
            xml = run.documentElement

            # create new points element
            recorded_at = (
                self.parent.signed_at
                if self.parent.signature_required
                else dt.datetime.now().isoformat()
            )
            points = run.createElement("points")
            points.setAttribute("recorded", recorded_at)
            xml.appendChild(points)

            for x, y in enumerate(poi_vals):
                point = run.createElement("point")
                point.setAttribute("name", str(x))
                point.setAttribute("value", str(y))
                points.appendChild(point)

            hash = hashlib.sha256()
            for p in points.childNodes:
                for name, value in p.attributes.items():
                    hash.update(name.encode())
                    hash.update(value.encode())
            signature = hash.hexdigest()
            points.setAttribute("signature", signature)

            try:
                with open(xml_path, "w", encoding="utf-8") as f:
                    xml_str = run.toxml(encoding="ascii").decode(encoding="utf-8", errors="ignore")
                    f.write(xml_str)
                    Log.d(f"Added <points> to XML file: {xml_path}")
            except OSError as ose:  # FileNotFoundError
                Log.e(f"Filesystem error writing XML: {xml_path}")
                Log.e("Error Details:", ose.strerror)
                self.detect_change()
                self._set_saved_state("error", "Error saving")
                self.saved_state_dot.flash()
            except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                Log.e(f"Unicode error writing XML: {xml_path}")
                Log.e("Error Details:", ue.reason)
                self.detect_change()
                self._set_saved_state("error", "Error saving")
                self.saved_state_dot.flash()

    def markerMoveFinished(self, marker):
        ax = self.graphWidget
        ax1 = self.graphWidget1
        ax2 = self.graphWidget2
        ax3 = self.graphWidget3
        tt1 = marker.value()
        tx1 = next(x for x, y in enumerate(self.xs) if y >= tt1)
        if abs(self.xs[tx1] - tt1) > abs(self.xs[tx1 - 1] - tt1):
            tx1 -= 1
        marker.setValue(self.xs[tx1])  # snap to nearest point
        marker_idx = -1
        for idx, pm in enumerate(self.poi_markers):
            if pm.value() == marker.value():
                marker_idx = idx
                break
        if self.moved_markers[marker_idx] == False:
            Log.d(f"Marker {marker_idx} has been moved by the user! Flagged for model tuning.")
        # clear flag if it moved from AI directive; only set on manual movement
        # if not self.AI_moving_marker else False
        self.moved_markers[marker_idx] = True
        self.detect_change()
        # setXRange for 'ax' all the time on marker move to keep markers in view (except for Step 2)
        if self.stateStep > 0:
            tt0 = self.poi_markers[0].value()
            tx0 = next(x for x, y in enumerate(self.xs) if y >= tt0)
            tt2 = self.poi_markers[-1].value()
            tx2 = next(x for x, y in enumerate(self.xs) if y >= tt2)
            ax.setXRange(tt0, tt2, padding=0.12)
            # Prevent empty slices
            if tx0 >= tx2:
                tx0 = 0
                tx2 = len(self.xs) - 1
            mn = min(
                np.amin(self.ys_freq_fit[tx0:tx2]),
                np.amin(self.ys_fit[tx0:tx2]),
                np.amin(self.ys_diff_fit[tx0:tx2]),
            )
            mx = max(
                np.amax(self.ys_freq_fit[tx0:tx2]),
                np.amax(self.ys_fit[tx0:tx2]),
                np.amax(self.ys_diff_fit[tx0:tx2]),
            )
            ax.setYRange(mn, mx, padding=0.05)
        if self.stateStep in range(1, 7):
            cur_val = marker.value()
            cur_idx = next(x for x, y in enumerate(self.xs) if y >= cur_val)
            if cur_idx == len(self.xs) - 1:
                return  # do not process skipped points on marker move
            ws = self.getContextWidth()[0]
            pad = 0.05 if self.stateStep >= 3 else 0.05
            # Calculate safe index boundaries prior to setting ranges
            slice_start, slice_end = [tx1 - ws, tx1 + ws]
            if slice_start < 0:
                slice_start = 0
            if slice_end > len(self.xs) - 1:
                slice_end = len(self.xs) - 1
            if slice_start >= slice_end:
                slice_start = 0
                slice_end = len(self.xs) - 1
            ax1.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax2.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            ax3.setXRange(self.xs[slice_start], self.xs[slice_end], padding=0)
            if self.stateStep >= 3:
                ax1.setYRange(
                    np.min(self.ys_freq_fit[slice_start:slice_end]),
                    np.max(self.ys_freq_fit[slice_start:slice_end]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff_fit[slice_start:slice_end]),
                    np.max(self.ys_diff_fit[slice_start:slice_end]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys_fit[slice_start:slice_end]),
                    np.max(self.ys_fit[slice_start:slice_end]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq_fit[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff_fit[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys_fit[tx1]))
            else:
                ax1.setYRange(
                    np.min(self.ys_freq[slice_start:slice_end]),
                    np.max(self.ys_freq[slice_start:slice_end]),
                    padding=pad,
                )
                ax2.setYRange(
                    np.min(self.ys_diff[slice_start:slice_end]),
                    np.max(self.ys_diff[slice_start:slice_end]),
                    padding=pad,
                )
                ax3.setYRange(
                    np.min(self.ys[slice_start:slice_end]),
                    np.max(self.ys[slice_start:slice_end]),
                    padding=pad,
                )
                pos1 = np.column_stack((self.xs[tx1], self.ys_freq[tx1]))
                pos2 = np.column_stack((self.xs[tx1], self.ys_diff[tx1]))
                pos3 = np.column_stack((self.xs[tx1], self.ys[tx1]))
            self.star1.setData(pos=pos1)
            self.star2.setData(pos=pos2)
            self.star3.setData(pos=pos3)
        # if (
        #     self.moved_markers[self.AI_SelectTool_At]
        #     and self.AI_SelectTool_Frame.isVisible()
        # ):
        #     # move AI Tool to new marker location
        #     self.summaryAt(self.AI_SelectTool_At)

    def getRunInfo(self):
        """
        Load and display information about a run from an XML file, initializing
        a GUI to view or edit the run's details.

        This method reads an XML file specified by `self.xml_path` to extract
        attributes such as the run's name, associated CSV file path, ruling
        (e.g., good or bad), and optionally, the username of the parent control.
        It ensures that only one instance of the Run Info GUI is active, and
        manages communication between the main thread and a worker thread for
        GUI display and user interaction.

        If the XML path is invalid or not provided, the method does nothing.

        Attributes:
            self.xml_path (str): Path to the XML file containing the run information.
            self.parent: Reference to the parent object (if any), used to extract the
                username for run metadata.
            self.bThread (QtCore.QThread): Thread handling the Run Info GUI worker.
            self.bWorker (QueryRunInfo): Worker object for the Run Info GUI.

        Raises:
            Exception: If there are issues reading or parsing the XML file, or if
                GUI initialization fails.

        Example:
            self.xml_path = "path/to/run_info.xml"
            self.getRunInfo()
        """
        # Check if the XML path is provided
        if self.xml_path != None:
            Log.d(tag=TAG, msg=f"Loaded xml_path={self.xml_path}")

            # Read the XML file's content.
            xml_text = ""
            with open(self.xml_path, "r", encoding="utf-8") as f:
                xml_text = f.read()

            # Decode if the content is in bytes format.
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode()

            # Parse the XML content and extract attributes from
            # the XML.
            xml = minidom.parseString(xml_text)
            run = xml.documentElement
            run_name = run.getAttribute("name")
            run_path = self.xml_path[0:-4] + ".csv"
            is_good = run.getAttribute("ruling")

            # Get the username from the parent control, if available.
            user_name = (
                None if self.parent == None else self.parent.controls_window.username.text()[6:]
            )
            # check signatures of XML, render a new QueryRunInfo() and allow saving changes
            # (when editing runinfo, append to existing audit, not overwrite as new CAPTURE).
            if hasattr(self, "bThread"):
                if self.bThread.isRunning():
                    Log.w("Run Info GUI already open. Re-showing instead.")
                    self.bWorker.hide()
                    self.bWorker.show()
                    return

            # Initialize the thread and worker for the Run Info GUI.
            self.bThread = QtCore.QThread()
            self.bWorker = QueryRunInfoWidget(
                run_name=run_name,
                run_path=run_path,
                run_ruling=is_good,
                user_name=user_name,
                recall_from=self.xml_path,
                parent=self.parent,
            )  # TODO: more secure to pass user_hash (filename)

            # Configure the Run Info GUI worker.
            self.bWorker.setRuns(1, 0)
            self.bThread.started.connect(self.bWorker.show)
            self.bWorker.finished.connect(self.bThread.quit)
            self.bWorker.finished.connect(self.update_run_names)

            # IPC signal to get the updated path name from the Run Info window on
            # change.
            self.bWorker.updated_run.connect(self.update_current_run_info)
            self.bWorker.updated_xml_path.connect(self.setXmlPath)

            # Start the thread to display the Run Info GUI
            self.bThread.start()

    def update_current_run_info(self, xml_path, new_name, old_name, date):
        """
        Updates the current run information in the combo box and the `run_names` dictionary.

        Args:
            new_name (str): The new name to update in the combo box and dictionary.
            old_name (str): The old name to search for in the combo box and dictionary.
            date (str): The date associated with the run, used to form the complete name.

        Raises:
            None: Logs an error message if the old name with the specified date is not found in the combo box.

        Updates:
            - If the item with the old name exists in the combo box, updates it with the new name.
            - Searches for a key in the `run_names` dictionary that contains the old name followed by a colon (:).
            If found, extracts the part of the key after the colon, removes the old key, and adds a new key with
            the new name and the extracted value.
            - Updates the `text_Created` field to display the new name and date.

        Example:
            If the combo box contains "OldName (2024-11-20)" and the `run_names` dictionary contains:
                {
                    "OldName:Details": "value1"
                }
            Calling `update_current_run_info("NewName", "OldName", "2024-11-20")` will:
            - Update the combo box to "NewName (2024-11-20)"
            - Update the dictionary to:
                {
                    "NewName:Details": "NewName"
                }
            - Set `text_Created` to "NewName (2024-11-20)".
        """
        index = self.cBox_Runs.findText(f"{old_name} ({date})")

        # Check if the old name exists in the combo box
        if index != -1:
            # Update the item with the new name
            self.cBox_Runs.setItemText(index, f"{new_name} ({date})")
        else:
            Log.e(TAG, f"Item with name '{old_name} ({date})' not found in the combo box.")
        for key in list(self.run_names.keys()):  # Use list to avoid runtime changes
            if f"{old_name}:" in key:
                # Extract the part of the key after the ':'
                _, after_colon = key.split(":", 1)  # Split at the first ':'
                # Store the value and remove the entry
                value = self.run_names.pop(key)
                after_colon = after_colon.strip()
                break
        value = self.run_timestamps.pop(key)
        self.run_timestamps[f"{new_name}:{after_colon}"] = value
        self.run_names[f"{new_name}:{after_colon}"] = new_name
        self.text_Created.setText(f"Loaded: {new_name} ({date})")
        if hasattr(self, "_batched_runs") and self._batched_runs:
            self._current_run = self.text_Created.text()
        self.loaded_datapath = xml_path[:-4] + ".csv"

    def update_run_names(self):
        """
        Used as a reciever from QueryRunInfo to update the xml_path name
        to the modified xml_path name.
        """
        if self.bWorker.run_name_changed:
            loaded_idx = self.cBox_Runs.currentIndex()
            devs = FileStorage.DEV_get_all_device_dirs()
            for i, _ in enumerate(devs):
                self.update_run(i)
            self.cBox_Runs.setCurrentIndex(loaded_idx)

    def analyze_data(self, data_path: str) -> None:
        """Load and prepare run data for analysis.

        Reads the CSV at `data_path`, sanitizes backward-time jumps, computes
        smoothed resonance / dissipation / difference curves, restores or predicts
        POIs from a companion XML file or predictive models, applies optional drop-effect
        corrections and difference-factor optimization, then populates all plotting
        widgets and internal state for the analysis workflow.

        Args:
            data_path: Absolute or relative path to the run CSV file.  A companion
                XML file with the same base name may be read to restore prior POIs
                and run parameters.  The CSV must span at least 3 seconds; shorter
                runs are rejected with a logged error and return early.

        Side Effects:
            Mutates many instance attributes including `xs`, `ys`, `ys_freq`,
            `ys_diff`, `ys_fit`, `poi_markers`, `smooth_factor`,
            `loaded_datapath`, `stateStep`, `model_result`, and
            `model_candidates`.  Performs several UI updates (clears/repopulates
            plot widgets, updates progress text, enables/disables navigation
            buttons).

        Note:
            When a QModel finds all six valid POIs, the routine
            auto-advances to the summary step (`stateStep = 6`) and marks the
            dataset as changed.  On any non-fatal failure the method falls back to
            manual POI selection rather than re-raising.

        Threading:
            The actual file I/O, model inference, and signal-processing
            (`_run_analysis_pipeline`) run on a background `_RunLoadThread`
            so this no longer blocks Qt's event loop for the full duration
            of a load. This method still blocks its own *caller* until the
            run is fully loaded (callers such as `action_load_run` assume
            synchronous completion) - it does so via a local `QEventLoop`
            that keeps the UI thread pumping paint/input events while it
            waits, instead of freezing outright.
        """
        self._init_analysis_state(data_path)
        # Every load path (manual Load click, batch auto-advance calling
        # action_load_run directly, etc.) funnels through here, so this is
        # the one place that reliably shows the spinner regardless of how
        # the load was triggered - see _show_loading_run_overlay.
        self._show_loading_run_overlay()

        # Qt widgets may only be read from the thread that owns them, so
        # capture the toggle state the background pipeline needs before
        # dispatching to it.
        drop_effect_enabled = self.drop_effect_cancelation_checkbox.isChecked()
        diff_factor_optimizer_enabled = self.difference_factor_optimizer_checkbox.isChecked()
        partial_fills_enabled = self.partial_fills_checkbox.isChecked()

        worker = _RunLoadThread(
            self,
            data_path,
            drop_effect_enabled,
            diff_factor_optimizer_enabled,
            partial_fills_enabled,
        )
        loop = QtCore.QEventLoop()
        worker.finished.connect(loop.quit)
        worker.start()
        loop.exec_()
        worker.wait()

        result = worker.result or {}
        outcome = result.get("outcome", "error")
        relative_time = result.get("relative_time")
        resonance_frequency = result.get("resonance_frequency")
        dissipation = result.get("dissipation")
        poi_vals = result.get("poi_vals", [])
        curves = result.get("curves", {})
        start_stop = result.get("start_stop", [0, -1])

        if outcome == "success":
            self._update_analyze_progress(100, "Reading Run Data...")

        curves = self._recover_missing_curves(
            curves, relative_time, resonance_frequency, dissipation, savgol_filter
        )
        self._wait_for_progress_bar()
        # Hidden unconditionally here (not left solely to
        # _render_analysis_plots's own call) since the "short" branch right
        # below bypasses that method entirely - leaving the spinner stuck
        # up otherwise.
        self._hide_loading_run_overlay()

        if outcome == "short":
            # Run was under 3 seconds: matches the original early-return -
            # curves were still recovered/waited-on above, but rendering
            # and step-advancement are skipped entirely.
            self._set_saved_state("error", "Error loading")
            self.saved_state_dot.flash()
            return

        self._render_analysis_plots(curves, poi_vals, start_stop)
        self._save_analysis_state(curves, relative_time, resonance_frequency, dissipation)
        self._advance_analysis_step(poi_vals)
        if outcome == "error":
            # _render_analysis_plots (via _setup_graph_axes) already called
            # setDotStepMarkers(1), which sets "saved" - override that here
            # so a genuinely failed load (an exception in
            # _run_analysis_pipeline) still ends up red, not green.
            self._set_saved_state("error", "Error loading")
            self.saved_state_dot.flash()

    def _run_analysis_pipeline(
        self,
        data_path: str,
        drop_effect_enabled: bool,
        diff_factor_optimizer_enabled: bool,
        partial_fills_enabled: bool,
    ) -> Dict[str, Any]:
        """Runs the I/O + inference + signal-processing body of a run load.

        This is the background-thread counterpart of the old, fully
        synchronous `analyze_data` body: identical logic and identical
        exception/fallback handling, just invoked from `_RunLoadThread.run()`
        instead of directly on the UI thread. It must not touch any Qt
        widget directly - the few status-text updates the original inline
        code performed are emitted as signals (`_model_status_changed`,
        `_model_status_cleared`) that are queued onto the main thread by Qt
        automatically, since `self` still lives there.

        Args:
            data_path: Path to the run CSV file.
            drop_effect_enabled: Pre-read state of the drop-effect
                cancelation checkbox (see `analyze_data`).
            diff_factor_optimizer_enabled: Pre-read state of the
                difference-factor optimizer checkbox.
            partial_fills_enabled: Pre-read state of the partial-fills
                checkbox (QModel Indus).

        Returns:
            dict: `outcome` is one of `"success"`, `"short"` (run under 3
            seconds), or `"error"` (an exception was caught), plus the
            `relative_time`/`resonance_frequency`/`dissipation`/`poi_vals`/
            `curves`/`start_stop` values `analyze_data` needs to finish up
            on the main thread.
        """
        relative_time = resonance_frequency = dissipation = None
        poi_vals, start_stop, curves = [], [0, -1], {}
        outcome = "error"

        try:
            Log.i(f"Analysis file = {data_path}")
            # Read the run file exactly once. Every downstream consumer
            # below (QModel predictors, drop-effect correction, the
            # difference-factor optimizer) used to independently re-open
            # and re-read this same file from disk/zip; they all accept an
            # in-memory buffer and seek(0) before use, so a single shared
            # `bytes` object safely replaces up to six redundant reads.
            raw_bytes = self._read_run_file_bytes(data_path)

            relative_time, _, resonance_frequency, dissipation = self._load_run_data(raw_bytes)

            if relative_time[-1] < 3:
                Log.e("ERROR: Data run must be at least 3 seconds in total runtime to analyze.")
                outcome = "short"
            else:
                poi_vals, fill_type = self._load_xml_pois(data_path)
                self._apply_poi_state(poi_vals, fill_type)

                poi_vals = self._run_model_prediction(
                    poi_vals,
                    relative_time,
                    resonance_frequency,
                    dissipation,
                    raw_bytes,
                    partial_fills_enabled,
                )

                self._model_status_cleared.emit()

                dissipation, resonance_frequency = self._apply_signal_corrections(
                    dissipation, resonance_frequency, poi_vals, raw_bytes, drop_effect_enabled
                )

                curves, start_stop = self._compute_signal_curves(
                    relative_time,
                    resonance_frequency,
                    dissipation,
                    poi_vals,
                    savgol_filter,
                    argrelextrema,
                    raw_bytes,
                    diff_factor_optimizer_enabled,
                )

                outcome = "success"

        except Exception:
            self.progress_value_steps.clear()
            self._log_traceback()
            Log.w("An error occurred loading this run! Please manually select points for Analysis.")
            outcome = "error"

        return {
            "outcome": outcome,
            "relative_time": relative_time,
            "resonance_frequency": resonance_frequency,
            "dissipation": dissipation,
            "poi_vals": poi_vals,
            "curves": curves,
            "start_stop": start_stop,
        }

    def _set_model_status_text(self, msg: str) -> None:
        """Main-thread slot: shows a model auto-fitting status message.

        Connected to `_model_status_changed`, which the (possibly
        background-thread) model-prediction methods emit instead of
        touching the loading overlay/`self.graphWidget` directly.
        """
        self._set_loading_run_status(msg)

    def _clear_model_status_text(self) -> None:
        """Main-thread slot: swaps the overlay to the "Showing data..." message.

        Connected to `_model_status_cleared`, emitted once the model
        auto-fitting phase of the run-load pipeline has finished.
        """
        self._set_loading_run_status("Showing data for analysis...")

    def _init_analysis_state(self, data_path):
        """Reset per-run analysis state and configure navigation buttons."""
        self.stateStep = -1
        self.loaded_datapath = data_path
        self.btn_Back.setEnabled(False)
        self.btn_Next.setEnabled(True)

    def _read_run_file_bytes(self, data_path: str) -> bytes:
        """Reads an entire run file exactly once, as raw bytes.

        This is the single I/O read shared by `_load_run_data`, the QModel
        predictors, drop-effect correction, and the difference-factor
        optimizer for a given run load - all of which previously performed
        their own independent `secure_open` read of the same file.

        Args:
            data_path: Path to the run CSV file, openable via `secure_open`.

        Returns:
            bytes: The full contents of the file. `secure_open` returns a
            text-mode handle for loose files still on disk and a binary
            handle for records read out of an (optionally encrypted) zip
            archive; the result is normalized to `bytes` either way so
            every downstream consumer sees a consistent type.
        """
        with secure_open(data_path, "r", "capture") as f:
            content = f.read()
        return content if isinstance(content, bytes) else content.encode()

    def _load_run_data(
        self, raw_bytes: bytes
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse a run CSV and sanitize backward-time rows.

        Detects the presence of an "Ambient" temperature column in the header to
        select the correct column indices, then removes any rows where the relative
        timestamp goes backward using a single vectorized pass.

        Args:
            raw_bytes: The full file contents, as returned by
                `_read_run_file_bytes`.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A four-element
            tuple of `(relative_time, temperature, resonance_frequency,
            dissipation)` as 1-D float arrays, sanitized of time-jump artefacts.
            The original file is never modified.
        """
        text_stream = StringIO(raw_bytes.decode())
        header = next(text_stream)
        csv_cols = (2, 4, 6, 7) if "Ambient" in header else (2, 3, 5, 6)
        # Streaming the remaining lines directly into loadtxt (instead of
        # first materializing them into a Python list via .readlines())
        # avoids an extra full-file pass/allocation on top of removing the
        # redundant disk/zip read this method used to perform on its own.
        data = loadtxt(text_stream, delimiter=",", usecols=csv_cols)

        relative_time = data[:, 0]
        temperature = data[:, 1]
        resonance_frequency = data[:, 2]
        dissipation = data[:, 3]

        # Vectorized time-jump detection(s).
        backward = np.where(np.diff(relative_time) < 0)[0]
        if backward.size:
            Log.w(f"Warning: time jump(s) observed at the following indices: {backward.tolist()}")

            keep = np.ones(len(relative_time), dtype=bool)
            keep[backward] = False
            relative_time = relative_time[keep]
            temperature = temperature[keep]
            resonance_frequency = resonance_frequency[keep]
            dissipation = dissipation[keep]

            Log.w("Time jumps removed from dataset for analysis purposes (original file unchanged)")

        return relative_time, temperature, resonance_frequency, dissipation

    def _load_xml_pois(self, data_path: str):
        """Parse the companion XML file to restore prior POI indices and run params.

        Uses `xml.etree.ElementTree` to extract the most-recent
        `<points>` and `<params>` blocks.  Child elements are iterated directly.

        Args:
            data_path: Path to the run CSV.  The XML file is expected at the same
                location with a `.xml` extension (or at `self.xml_path` when
                that override is set).  If the file does not exist or
                `self.askForPOIs` is `False`, defaults are returned immediately.

        Returns:
            tuple[list[int], int]: A two-element tuple of
            `(poi_vals, fill_type)` where `poi_vals` is a sorted list of
            integer POI indices (empty when none are found) and `fill_type` is
            an integer channel count (-1` when not present in the XML).
        """
        poi_vals = []
        fill_type = -1
        if not self.askForPOIs:
            return poi_vals, fill_type

        xml_path = (data_path[:-4] + ".xml") if self.xml_path is None else self.xml_path
        if not os.path.exists(xml_path):
            Log.w(TAG, f'Missing XML file: Expected at "{xml_path}" for this run.')
            return poi_vals, fill_type

        root = ET.parse(xml_path).getroot()

        # Get POI point values
        all_points = root.findall(".//points")
        if all_points:
            for child in all_points[-1]:  # most-recent <points> block
                raw = child.get("value", "")
                try:
                    poi_vals.append(int(raw))
                except ValueError:
                    Log.e(f'Point value "{raw}" in XML is not an integer.')
            poi_vals.sort()
        else:
            Log.d("No points found in XML file for this run.")

        # Run parameters
        all_params = root.findall(".//params")
        if all_params:
            for child in all_params[-1]:  # most-recent <params> block
                name = child.get("name", "")
                raw = child.get("value", "")
                try:
                    if name == "fill_type":
                        fill_type = int(raw)
                except ValueError:
                    Log.e(f'Param "{name}" in XML is not an integer.')
        else:
            Log.d("No params found in XML file for this run.")

        return poi_vals, fill_type

    def _apply_poi_state(self, poi_vals: List[Any], fill_type: int) -> None:
        """Updates instance state flags based on restored POIs and data fill type.

        This method synchronizes the UI state with data recovered from an XML file.
        If a full set of 6 POIs is found, the workflow skips directly to the summary
        view (Step 6). If partial markers exist, it disables the initial POI prompt
        to allow for manual re-analysis.

        Args:
            poi_vals: A list of POI values retrieved from the XML.
            fill_type: The number of channels detected in the current data set.
        """
        # Reset transient state flags
        self.show_analysis_immediately = False
        self.model_run_this_load = False
        self.prior_points_in_xml = False

        if self.askForPOIs:
            if len(poi_vals) == 6:
                self.askForPOIs = False
                self.prior_points_in_xml = True
                self.stateStep = 6  # Skip straight to summary
                Log.d(f"Found prior POIs from XML file: {poi_vals}")

            elif len(self.poi_markers) > 0:
                # Re-analyse Step 1; prevent auto-advancing to Summary
                self.askForPOIs = False

        # Update channel configuration
        Log.d(f"Number of channels (fill_type): {fill_type}")
        self.parent.num_channels = fill_type

    def _run_model_prediction(
        self,
        poi_vals: List[Any],
        relative_time: np.ndarray,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
        raw_bytes: bytes,
        partial_fills_enabled: bool,
    ) -> List[Any]:
        """Orchestrates the POI auto-fitting fallback chain.

        Attempts to predict POIs the current fallback strategy:
            1. QModel Onyx (onyx)
            2. QModel Volta
            3. QModel Indus
            4. QModel Tweed.

        Args:
            poi_vals: Current list of POI values.
            relative_time: Array of time offsets for the dataset.
            resonance_frequency: Array of frequency shifts.
            dissipation: Array of dissipation values.
            raw_bytes: The full run file contents, shared across every
                model attempted below instead of each one re-reading the
                file from disk/zip independently.
            partial_fills_enabled: Pre-read state of the partial-fills
                checkbox, forwarded to QModel Indus.

        Returns:
            The best available list of POIs. If pre-existing markers exist,
            returns a truncated list containing only [start, end].
        """
        # Exit early if a result already exists from a prior load
        if self.model_result != -1:
            return poi_vals

        # Initialize/Reset model state
        self.model_result = -1
        self.model_candidates = None
        self.model_engine = "None"
        self._channel_config_cache = {}

        poi_vals = self._try_qmodel_onyx(poi_vals, raw_bytes)

        if self.model_result == -1:
            poi_vals = self._try_qmodel_volta(poi_vals, raw_bytes)

        if self.model_result == -1:
            poi_vals = self._try_qmodel_indus(poi_vals, raw_bytes, partial_fills_enabled)

        if self.model_result == -1:
            poi_vals = self._try_model_data(
                poi_vals, relative_time, resonance_frequency, dissipation
            )

        # If partial markers exist, force manual entry for the middle points
        if self.poi_markers:
            return [poi_vals[0], poi_vals[-1]]

        return poi_vals

    def _try_qmodel_onyx(self, poi_vals: List[int], raw_bytes: bytes) -> List[int]:
        """Attempts POI prediction using the QModel Onyx (onyx) engine.

        NOTE: The POST point is not predicted with this model and is simply filled!

        Args:
            poi_vals: The current list of POI indices.
            raw_bytes: The full run file contents (already read once by the
                caller), wrapped fresh into a `BytesIO` here.

        Returns:
            The updated list of POI indices if successful; otherwise, returns
            the original `poi_vals` and sets `self.model_result` to -1 for fallback.
        """
        if not Constants.qmodel_onyx_predict:
            return poi_vals

        # Skip inference if XML already provided a full set of valid points
        if self.prior_points_in_xml:
            self.model_result = poi_vals
            self.model_engine = "Onyx skipped (using prior points)"
            return poi_vals

        msg = "Auto-fitting points with QModel Onyx..."
        Log.w(f"{msg} (may take a few seconds)")
        self._model_status_changed.emit(msg)

        try:
            fh = BytesIO(raw_bytes)

            predict_result, detected_channels = self.QModel_onyx_predictor.predict(
                file_buffer=fh, progress_signal=self.onyx_predict_progress
            )

            if not self.parent.num_channels:
                self.parent.num_channels = detected_channels

            # Extract predictions
            predictions = []
            candidates = []

            for i in range(6):
                data = predict_result.get(f"POI{i + 1}", {})
                indices = data.get("indices", [-1]) or [-1]
                confidences = data.get("confidences", [-1]) or [-1]

                predictions.append(indices[0])
                candidates.append((indices, confidences))

            # Update model state
            self.model_run_this_load = True
            self.model_result = predictions
            self.model_candidates = candidates
            self.model_engine = f"Onyx - {detected_channels}ch"

            # Validate
            # NOTE: Legacy POI3 (POST point) is ignored here!
            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                poi_vals = list(self.model_result)
                if poi_vals[2] == -1 and poi_vals[1] != -1:
                    poi_vals[2] = poi_vals[1] + 2
                self._model_status_changed.emit("Caching channel configurations...")
                self._cache_channel_hypotheses(
                    self.QModel_onyx_predictor, raw_bytes, detected_channels, poi_vals
                )
            else:
                self.model_result = -1

        except Exception as e:
            self._log_traceback(debug=True)
            Log.e(f"Error using 'QModel Onyx': {e}")
            Log.e(TAG, "Falling back to next available model.")
            self.model_result = -1

        return poi_vals

    def _try_qmodel_volta(self, poi_vals: List[int], raw_bytes: bytes) -> List[int]:
        """Attempts POI prediction using the QModel Volta  engine.

        NOTE: The POST point is not predicted with this model and is simply filled!

        Args:
            poi_vals: The current list of POI indices.
            raw_bytes: The full run file contents (already read once by the
                caller), wrapped fresh into a `BytesIO` here.

        Returns:
            The updated list of POI indices if successful; otherwise, returns
            the original `poi_vals` and sets `self.model_result` to -1 for fallback.
        """
        if not Constants.qmodel_volta_predict:
            return poi_vals

        # Skip inference if XML already provided a full set of valid points
        if self.prior_points_in_xml:
            self.model_result = poi_vals
            self.model_engine = "Volta  skipped (using prior points)"
            return poi_vals

        msg = "Auto-fitting points with QModel Volta ..."
        Log.w(f"{msg} (may take a few seconds)")
        self._model_status_changed.emit(msg)

        try:
            fh = BytesIO(raw_bytes)

            # self._QModel_create_new_progress_dialog()
            # self.progressBarDiag.setRange(0, 100)

            predict_result, detected_channels = self.QModel_volta_predictor.predict(
                file_buffer=fh, progress_signal=self.volta_predict_progress
            )
            # QtCore.QTimer.singleShot(1000, self.progressBarDiag.hide)

            if not self.parent.num_channels:
                self.parent.num_channels = detected_channels

            # Extract predictions
            predictions = []
            candidates = []

            for i in range(6):
                data = predict_result.get(f"POI{i + 1}", {})
                indices = data.get("indices", [-1]) or [-1]
                confidences = data.get("confidences", [-1]) or [-1]

                predictions.append(indices[0])
                candidates.append((indices, confidences))

            # Update model state
            self.model_run_this_load = True
            self.model_result = predictions
            self.model_candidates = candidates
            self.model_engine = f"Volta  - {detected_channels}ch"

            # Validate
            # NOTE: Legacy POI3 (POST point) is ignored here!
            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                poi_vals = list(self.model_result)
                if poi_vals[2] == -1 and poi_vals[1] != -1:
                    poi_vals[2] = poi_vals[1] + 2
                self._model_status_changed.emit("Caching channel configurations...")
                self._cache_channel_hypotheses(
                    self.QModel_volta_predictor, raw_bytes, detected_channels, poi_vals
                )
            else:
                self.model_result = -1

        except Exception as e:
            self._log_traceback(debug=True)
            Log.e(f"Error using 'QModel Volta ': {e}")
            Log.e(TAG, "Falling back to next available model.")
            self.model_result = -1

        return poi_vals

    def _try_qmodel_indus(
        self, poi_vals: List[int], raw_bytes: bytes, partial_fills_enabled: bool
    ) -> List[int]:
        """Attempts POI prediction using the QModel Indus engine.

        NOTE: The POST point is not predicted with this model and is simply filled!

        Args:
            poi_vals: The current list of POI indices.
            raw_bytes: The full run file contents (already read once by the
                caller), wrapped fresh into a `BytesIO` here.
            partial_fills_enabled: Pre-read state of the partial-fills
                checkbox (read on the main thread by the caller, since Qt
                widgets cannot be read from a background thread).

        Returns:
            The updated list of POI indices if successful; otherwise, returns
            the original `poi_vals` and sets `self.model_result` to -1.
        """
        if not Constants.qmodel_indus_predict:
            return poi_vals

        msg = "Auto-fitting points with QModel Indus..."
        Log.w(f"{msg} (may take a few seconds)")
        self._model_status_changed.emit(msg)

        try:
            fh = BytesIO(raw_bytes)

            predict_result = self.qmodel_indus_predictor.predict(
                file_buffer=fh,
                visualize=False,
                progress_signal=self.indus_predict_progress,
                use_partial_fills=partial_fills_enabled,
            )

            predictions = []
            candidates = []

            for i in range(6):
                data = predict_result.get(f"POI{i + 1}", {})
                indices = data.get("indices", [-1]) or [-1]
                confidences = data.get("confidences", [-1]) or [-1]
                predictions.append(indices[0])
                candidates.append((indices, confidences))

            # Update model state
            self.model_run_this_load = True
            self.model_result = predictions
            self.model_candidates = candidates
            self.model_engine = "Indus"

            # Validate
            # NOTE: Legacy POI3 (POST point) is ignored here!
            if isinstance(self.model_result, list) and len(self.model_result) == 6:
                poi_vals = list(self.model_result)
                if poi_vals[2] == -1 and poi_vals[1] != -1:
                    poi_vals[2] = poi_vals[1] + 2
            else:
                self.model_result = -1

        except Exception as e:
            self._log_traceback(debug=True)
            Log.e(f"Error using 'QModel Indus': {e}")
            self.model_result = -1

        return poi_vals

    def _try_model_data(
        self,
        poi_vals: List[int],
        relative_time: np.ndarray,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
    ) -> List[int]:
        """Attempts POI prediction using the legacy QModel Tweed predictor.

        This serves as the final algorithmic fallback tier if everything went wrong :)

        NOTE: Unlike the YOLO models, this method directly clears and repopulates
        the provided `poi_vals`!

        Args:
            poi_vals: The current list of POI indices to be updated.
            relative_time: Array of time offsets.
            resonance_frequency: Array of frequency shift data.
            dissipation: Array of dissipation data.

        Returns:
            The updated list of POI indices. Returns the original list if the
            feature is disabled or a critical error occurs.
        """
        if not Constants.qmodel_tweed_predict:
            return poi_vals

        try:
            self.model_run_this_load = True
            # IdentifyPoints returns a list of indices or -1 on failure
            result = self.qmodel_tweed_predictor.IdentifyPoints(
                self.loaded_datapath, relative_time, resonance_frequency, dissipation
            )
            self.model_result = result
            self.model_engine = "Tweed"

            if isinstance(result, list):
                poi_vals.clear()
                self.model_select = []
                self.model_candidates = []

                for point in result:
                    self.model_select.append(0)
                    if isinstance(point, list):
                        self.model_candidates.append(point)
                        # Extract the index (first element of the first candidate)
                        poi_vals.append(point[0][0])
                    else:
                        # Data format: simple index integer
                        self.model_candidates.append([point])
                        poi_vals.append(point)

            elif result == -1:
                Log.w("QModel Tweed failed to auto-calculate points for this run.")
            else:
                Log.e("QModel Tweed returned an unexpected response format.")

        except Exception as e:
            self._log_traceback(debug=False)
            Log.e(f"Legacy QModel Tweed execution failed: {e}")

        return poi_vals

    def _apply_signal_corrections(
        self,
        dissipation: np.ndarray,
        resonance_frequency: np.ndarray,
        poi_vals: List[int],
        raw_bytes: bytes,
        drop_effect_enabled: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies drop-effect correction vectors to signal data.

        If the drop-effect correction UI option is enabled, this method invokes
        a correction algorithm to adjust the dissipation and resonance frequency
        arrays based on the provided POIs.

        Args:
            dissipation: Array of raw dissipation values.
            resonance_frequency: Array of raw resonance frequency values.
            poi_vals: List of indices identifying key POIs.
            raw_bytes: The full run file contents, already read once by the
                caller.
            drop_effect_enabled: Pre-read state of the drop-effect
                cancelation checkbox (read on the main thread by the
                caller, since Qt widgets cannot be read from a background
                thread).

        Returns:
            A tuple of (dissipation, resonance_frequency). Arrays will be the
            corrected versions if the feature is active and valid; otherwise,
            the original input arrays are returned.
        """
        if not drop_effect_enabled:
            return dissipation, resonance_frequency

        corrected_diss, corrected_rf = self._correct_drop_effect(
            self.loaded_datapath, poi_vals, "process", raw_bytes
        )

        if corrected_diss is not None:
            dissipation = corrected_diss

        if corrected_rf is not None:
            resonance_frequency = corrected_rf

        return dissipation, resonance_frequency

    def _compute_smooth_params(self, xs: np.ndarray) -> Tuple[int, int, bool, int]:
        """Derives Savitzky-Golay window sizes and the 90-second split index.

        Calculates smoothing factors based on total runtime. It defines a 'split index'
        at the 90-second mark to allow for different smoothing intensities between
        the initial phase and extended data.

        Args:
            xs: Array of time samples (assumed to be sorted).

        Returns:
            A tuple containing:
                - smooth_factor (int): Primary window size (forced odd, 3-69).
                - t_split (int): Index of the first sample past 90s.
                - extend_data (bool): True if there is sufficient data beyond 90s.
                - extend_smf (int): Coarser window size for extended data (forced odd).
        """
        total_runtime = xs[-1]

        # Compute primary smooth factor
        smooth_factor = int(total_runtime * Constants.smooth_factor_ratio)
        if smooth_factor % 2 == 0:
            smooth_factor += 1
        smooth_factor = max(3, min(69, smooth_factor))

        Log.i(TAG, f"Total run time: {total_runtime} secs")
        Log.d(TAG, f"Smoothing: {smooth_factor} | Applying for first 90s.")

        # np.searchsorted uses binary search O(log n) vs the O(n) generator next()
        t_split = int(np.searchsorted(xs, 90, side="right"))
        extend_data = total_runtime > 90

        # Compute extended smoothing factor (forced to odd)
        extend_smf = int(smooth_factor / 20)
        if extend_smf % 2 == 0:
            extend_smf += 1

        # Check if there is enough "padding" to apply extended smoothing
        if extend_data and len(xs) < (t_split + 2 * extend_smf):
            Log.w(
                "Insufficient points after 90s for effective downsampling. "
                "Disabling extended smoothing."
            )
            t_split = len(xs)
            extend_data = False

        return smooth_factor, t_split, extend_data, extend_smf

    def _apply_savgol(
        self,
        signal: np.ndarray,
        smooth_factor: int,
        t_split: int,
        extend_data: bool,
        extend_smf: int,
        savgol_filter: Callable[..., np.ndarray],
        deriv: int = 0,
    ) -> np.ndarray:
        """Applies a Savitzky-Golay filter with an optional dual-window strategy.

        Smoothes the input signal using a primary window size up to the split point.
        If extended data processing is enabled, it applies a secondary (usually
        coarser) window to the remainder of the signal and concatenates the results.

        Args:
            signal: The input numerical array to be smoothed.
            smooth_factor: Window length for the primary segment (must be odd).
            t_split: The index at which to split the signal for dual-window processing.
            extend_data: Whether to apply a different smoothing factor beyond t_split.
            extend_smf: Target window length for the extended segment.
            savgol_filter: The filter function (e.g., scipy.signal.savgol_filter).
            deriv: The order of the derivative to compute. Defaults to 0 (smoothing).

        Returns:
            A smoothed NumPy array of the same length as the input signal.
        """
        # Process the primary segment (up to t_split)
        primary_window = max(smooth_factor, 3)
        result = savgol_filter(signal[:t_split], primary_window, 1, deriv)

        if not extend_data:
            return result

        # Process the extended segment
        remainder = signal[t_split:]
        rem_len = len(remainder)

        if rem_len > 0:
            # Window size must be odd and: 1 < window_size <= segment_length
            ext_window = min(rem_len, extend_smf)
            if ext_window % 2 == 0:
                ext_window = max(3, ext_window - 1)
            ext_window = max(3, ext_window)

            ext = savgol_filter(remainder, ext_window, 1, deriv)
            return np.concatenate((result, ext))

        return result

    def _compute_signal_curves(
        self,
        relative_time: np.ndarray,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
        poi_vals: list,
        savgol_filter: Callable,
        argrelextrema: Callable,
        raw_bytes: Optional[bytes] = None,
        diff_factor_optimizer_enabled: bool = False,
    ) -> Tuple[dict, list[int, int]]:
        """Compute all smoothed signal curves and derive initial run-boundary candidates.

        Applies Savitzky-Golay smoothing to the raw dissipation, resonance-
        frequency shift, and computed difference signals.  Uses the two highest
        2nd-derivative maxima of the smoothed dissipation to locate the rough
        analysis window, then refines the start/stop indices using a noise-floor
        threshold on the difference curve.

        Args:
            relative_time: 1-D monotonically increasing array of sample timestamps
                in seconds.
            resonance_frequency: 1-D array of raw resonance-frequency measurements
                aligned with `relative_time`.
            dissipation: 1-D array of raw dissipation measurements aligned with
                `relative_time`.
            poi_vals: Current candidate POI indices.  Used only to suppress
                spurious boundary-detection warnings when a model or XML has
                already provided valid points.
            savgol_filter: `scipy.signal.savgol_filter`.
            argrelextrema: `scipy.signal.argrelextrema`.
            raw_bytes: Pre-read file contents forwarded to `_optimize_curve`
                (when the difference-factor optimizer is enabled) to avoid
                re-reading the file from disk/zip.
            diff_factor_optimizer_enabled: Pre-read state of the
                difference-factor optimizer checkbox (read on the main
                thread by the caller, since Qt widgets cannot be read from
                a background thread).

        Returns:
            tuple[dict, list[int, int]]: A two-element tuple `(curves,
            start_stop)` where `curves` is a dict with keys `xs`, `ys`,
            `ys_freq`, `ys_diff`, `ys_fit`, `ys_freq_fit`,
            `ys_diff_fit`, `ys_diss_2ndd`, and `smooth_factor`; and
            `start_stop` is `[begin_idx, end_idx]` representing the estimated
            analysis window in sample-index space.

        NOTE: Several arrays that existed in the original implementation
            (`ys_diss_diff_avg`/ `zeros3`, `minima_*`, `ys_diff_fine`,
            `ys_diff_diff`, `ys_diff_2ndd`) were computed but never used. These have been
            removed!
        """
        xs = relative_time
        n = len(xs)
        smooth_factor, t_split, extend_data, extend_smf = self._compute_smooth_params(xs)

        def sg(sig: np.ndarray, deriv: int = 0) -> np.ndarray:
            """Convenience closure over the smoothing parameters"""
            return self._apply_savgol(
                sig, smooth_factor, t_split, extend_data, extend_smf, savgol_filter, deriv
            )

        # Dissipation smoothing and 2nd derivative
        ys_fit_raw = sg(dissipation)
        ys_diss_diff = sg(ys_fit_raw, deriv=1)
        ys_diss_2ndd = sg(ys_diss_diff, deriv=1)

        # Rough boundary detection via the two highest 2nd-derivative maxima
        maxima_idx = argrelextrema(ys_diss_2ndd, np.greater)[0]
        maxima_val = ys_diss_2ndd[maxima_idx]

        if len(maxima_idx) < 2:
            # Degenerate signal: fall back to a full-width window
            t_start, t_stop = 0, n - 1
        else:
            top2_pos = np.argpartition(maxima_val, -2)[-2:]
            top2_idx = np.sort(maxima_idx[top2_pos])  # ascending by sample position
            t_start = int(top2_idx[0])
            t_stop = int(top2_idx[1]) + 3 * smooth_factor

        no_model_no_poi = not self.model_run_this_load and len(poi_vals) == 0
        if t_stop < n / 2 or t_stop >= n:
            if no_model_no_poi:
                Log.w(f"Stop time index was {t_stop} out of {n} but that seems unlikely!")
                Log.w('Please confirm "End Point" during Step 1 point selection.')
            t_stop = n - 1
        if t_stop - t_start < n / 3 or t_start > n / 2:
            if no_model_no_poi:
                Log.w(f"Start time index was {t_start} out of {n} but that seems unlikely!")
                Log.w('Please confirm "Begin Point" during Step 1 point selection.')
            t_start = 100

        # Baseline window (first ~0.5-2 s of the run)
        if xs[t_start] < 0.5:
            t_0p5 = 0
        else:
            t_0p5 = min(int(np.searchsorted(xs, 0.5, side="right")), n - 1)

        if xs[t_start] < 2.0:
            t_1p0 = t_start
        else:
            t_1p0 = min(int(np.searchsorted(xs, 2.0, side="right")), n - 1)

        if t_0p5 == t_1p0:
            t_1p0 = min(int(np.searchsorted(xs, xs[t_1p0] + 1.5, side="right")), n - 1)
            if t_1p0 == t_0p5:
                # still equal at the array boundary
                t_1p0 = min(t_0p5 + 1, n - 1)

        # Scale dissipation relative to the baseline resonance frequency
        avg = np.mean(resonance_frequency[t_0p5:t_1p0])
        scale = avg / 2
        ys = dissipation * scale
        ys_fit = ys_fit_raw * scale
        offset = np.amin(ys_fit)
        ys -= offset
        ys_fit -= offset

        # Resonance frequency shift and smoothed fit
        ys_freq = avg - resonance_frequency
        ys_freq_fit = sg(ys_freq)

        # Difference factor and difference curve
        diff_factor = Constants.default_diff_factor
        if diff_factor_optimizer_enabled:
            self.diff_factor = self._optimize_curve(self.loaded_datapath, raw_bytes)
        if hasattr(self, "diff_factor"):
            diff_factor = self.diff_factor

        ys_diff = ys_freq - diff_factor * ys

        # Invert when the drop was applied at the outlet (negative initial deltas)
        if np.mean(np.abs(ys_freq_fit)) < np.mean(np.abs(diff_factor * ys_fit)) and abs(
            ys_diff[t_1p0:].min()
        ) > 5 * abs(ys_diff[t_1p0:].max()):
            Log.w("Inverting DIFFERENCE curve due to negative initial fill deltas")
            ys_diff *= -1

        ys_diff_fit = sg(ys_diff)

        Log.d(f"Difference factor: {diff_factor:.3f}x")
        Log.d("Setting diff_factor on Advanced Settings menu")
        self._diff_factor_value_changed.emit(diff_factor)

        # Noise-floor parameters for start/stop thresholding
        eh1 = float(abs(np.amax(ys_diff[t_0p5:t_1p0])))
        em2 = float(np.amax(ys_diff_fit))
        am2 = int(np.argmax(ys_diff_fit))
        eh2 = eh1 * 2  # start threshold: 2x baseline noise

        # Start candidate
        t0 = t_start
        above = np.where(ys_diff_fit[t_1p0 + 1 :] > 5 * eh2)[0]
        if above.size:
            t0 = t_1p0 + 1 + int(above[0])
        elif no_model_no_poi:
            Log.w("Failed to locate rough start point using noise floor approximation.")
            Log.w('Please confirm "Begin Point" during Step 1 point selection.')

        # Walk backward to the true fill start
        going_up = ys[t0] < 5
        while True:
            if not (0 <= t0 < n):
                if no_model_no_poi:
                    Log.w("Hit a limit (start)")
                t0 = 0
                break
            if ys[t0] < 5:
                t0 += 1
                if not going_up:
                    break
            else:
                t0 -= 1
                if going_up:
                    break

        # End candidate.
        t1 = am2
        below = np.where(ys_diff_fit[am2:] < em2 - eh2)[0]
        if below.size:
            t1 = am2 + int(below[0])
        elif no_model_no_poi:
            Log.w("Failed to locate rough end point using noise floor approximation.")
            Log.w('Please confirm "End Point" during Step 1 point selection.')

        # Walk backward to the true end
        while True:
            if t1 - 50 < 0:
                if no_model_no_poi:
                    Log.w("Hit a limit (end)")
                t1 = n - 1
                break
            if ys_diff_fit[t1 - 50] > ys_diff_fit[t1]:
                t1 -= 1
            else:
                break

        curves = {
            "xs": xs,
            "ys": ys,
            "ys_freq": ys_freq,
            "ys_diff": ys_diff,
            "ys_fit": ys_fit,
            "ys_freq_fit": ys_freq_fit,
            "ys_diff_fit": ys_diff_fit,
            "ys_diss_2ndd": ys_diss_2ndd,
            "smooth_factor": smooth_factor,
        }
        return curves, [t0, t1]

    def _recover_missing_curves(
        self,
        curves: dict,
        relative_time: np.ndarray | None,
        resonance_frequency: np.ndarray | None,
        dissipation: np.ndarray | None,
        savgol_filter,
    ) -> dict:
        """Reconstruct any signal arrays missing from curves after a partial failure.

        Called unconditionally from the `finally` block of `analyze_data` so
        that the UI can always fall back to manual POI selection even when signal
        processing threw partway through.  Each key is only written when absent,
        so pre-existing values are always preserved.

        NOTE: Reconstruction order matters: later entries depend on keys built earlier
        in this function (e.g. `ys_diff` needs `ys_freq` and `ys`).

        Args:
            curves: Partially or fully populated signal dict.  Modified in-place
                and returned.  Keys that are already present are never overwritten.
            relative_time: Raw timestamp array, or `None` when data loading
                failed before it could be read.
            resonance_frequency: Raw resonance-frequency array, or `None`.
                Together with `dissipation`, required for any reconstruction;
                if either is `None` the dict is returned unchanged.
            dissipation: Raw dissipation array, or `None`.
            savgol_filter: `scipy.signal.savgol_filter`

        Returns:
            dict: The same curves dict, with any previously absent required keys
            now populated with zero-smoothing fallback values.
        """

        required_keys: frozenset = frozenset(
            {
                "xs",
                "ys",
                "ys_fit",
                "ys_freq",
                "ys_freq_fit",
                "ys_diff",
                "ys_diff_fit",
                "ys_diss_2ndd",
                "smooth_factor",
            }
        )

        if not (required_keys - curves.keys()):
            return curves

        # Without raw sensor data no reconstruction is possible.
        if resonance_frequency is None or dissipation is None:
            return curves

        Log.w("Correcting missing parameters for manual point selection (no smoothing)...")

        # Use the first sample as the baseline resonance value.
        avg = float(resonance_frequency[0])

        if "xs" not in curves and relative_time is not None:
            curves["xs"] = relative_time

        # ys / ys_fit
        if "ys" not in curves or "ys_fit" not in curves:
            scaled = dissipation * (avg / 2)
            offset = float(np.amin(scaled))
            curves["ys"] = scaled - offset
            curves["ys_fit"] = scaled - offset  # independent array; same values

        if "ys_freq" not in curves:
            curves["ys_freq"] = avg - resonance_frequency

        if "ys_freq_fit" not in curves:
            curves["ys_freq_fit"] = curves["ys_freq"]

        if "ys_diff" not in curves:
            curves["ys_diff"] = curves["ys_freq"] - Constants.default_diff_factor * curves["ys"]
        if "ys_diff_fit" not in curves:
            curves["ys_diff_fit"] = curves["ys_diff"]
        if "ys_diss_2ndd" not in curves:
            try:
                ys_diss_diff = savgol_filter(curves["ys_fit"], 2, 1, 1)
                curves["ys_diss_2ndd"] = savgol_filter(ys_diss_diff, 2, 1, 1)
            except Exception:
                Log.e("Unable to calculate 2nd derivative of 'ys' data!")
                curves["ys_diss_2ndd"] = curves["ys_fit"]

        if "smooth_factor" not in curves:
            curves["smooth_factor"] = 3

        return curves

    def _log_traceback(self, debug: bool = False) -> None:
        """Logs the full traceback of the currently handled exception.

        Args:
            debug: If True, logs each line at the DEBUG level (Log.d).
                If False, logs at the ERROR level (Log.e).
        """
        log_func: Callable[[str], None] = Log.d if debug else Log.e
        exc_traceback: str = traceback.format_exc()
        for line in exc_traceback.strip().split("\n"):
            log_func(line)

    def _wait_for_progress_bar(self) -> None:
        """Spins the Qt event loop until the background scan completes or times out.

        This method prevents the UI from freezing during a background scan by
        manually processing pending events. It includes a safety timeout to
        prevent infinite loops. Once finished, it connects the progress bar's
        value change signal to the internal updater.
        """
        timeout_limit = 300
        iterations = 0

        Log.d("Waiting on progress bar to finish background scan...")

        while self.progress_value_scanning and iterations < timeout_limit:
            iterations += 1
            QtCore.QCoreApplication.processEvents()
            time.sleep(0.01)
        try:
            self.progressBar.valueChanged.disconnect(self._update_progress_value)
        except (TypeError, RuntimeError):
            pass
        self.progressBar.valueChanged.connect(self._update_progress_value)
        Log.d(f"Progress bar wait completed after {iterations} iterations. Proceeding...")

    def _render_analysis_plots(
        self, curves: Dict[str, Any], poi_vals: List[int], start_stop: Tuple[int, int]
    ) -> None:
        """Coordinates the rendering of signal data and analysis markers.

        This top-level coordinator manages the visual pipeline by setting up
        graph axes, drawing the primary signal curves, and overlaying channel/POI markers.
        It delegates specific rendering tasks to specialized sub-helper methods.

        Args:
            curves: A dictionary containing signal data. Expected keys include:
                - 'xs': The shared x-axis (time) array.
                - Individual signal keys (e.g., 'df', 'dg') containing y-axis arrays.
            poi_vals: A list of indices representing the detected POI.
            start_stop: A tuple of (start_index, stop_index) defining the active
                analysis window.
        """
        self._hide_no_run_overlay()
        self._hide_loading_run_overlay()
        self._setup_graph_axes(curves)
        self._plot_signal_curves(curves)
        self._add_poi_markers(curves, poi_vals, start_stop)

    def _setup_graph_axes(self, curves: Dict[str, Any]) -> None:
        """Configures titles, labels, ranges, grids, and legends for all four axes.

        This method clears existing plots, initializes the progress UI, sets specific
        titles and colors for four distinct graph widgets, and establishes the
        coordinate ranges based on the provided curve data.

        Args:
            curves: A dictionary containing the data to be plotted.
                Expected keys include:
                - 'xs': The x-axis data (typically time).
                - 'ys': Primary y-axis data.
                - 'ys_freq': Frequency-related y-axis data.
                - 'ys_diff': Difference-related y-axis data.
        """
        ax, ax1, ax2, ax3 = (
            self.graphWidget,
            self.graphWidget1,
            self.graphWidget2,
            self.graphWidget3,
        )
        xs = curves["xs"]
        ys = curves["ys"]
        ys_freq = curves["ys_freq"]
        ys_diff = curves["ys_diff"]

        ax.clear()
        ax1.clear()
        ax2.clear()
        ax3.clear()
        # ax.clear() can reset axis pens on some pyqtgraph versions, so
        # theme colors must be reapplied every time these axes are cleared.
        self._apply_pg_theme()

        self._update_progress_value(1, "Step 1 of 6: Select Begin and End Points")
        self.setDotStepMarkers(1)

        # No in-plot title text on the detail graphs - each one's card
        # header already shows a colored-dot chip with the same name
        # (see DetailPlotCard in analyze_plot_cards.py), so a second label
        # drawn on the plot itself would just be duplicate text.
        ax.setTitle(None)
        ax1.setTitle(None)
        ax2.setTitle(None)
        ax3.setTitle(None)

        # No inline color/font-size style here - PlotsUI's own plot labels
        # don't set one either, letting apply_glass_plot_style's textPen
        # (see _apply_pg_theme) govern tick + label color uniformly.
        ax.showAxis("left")
        ax.setLabel("left", "Frequency (Hz)")
        ax.showAxis("bottom")
        ax.setLabel("bottom", "Time (secs)")

        # pg's native corner autoscale button is redundant now that the
        # overview card header has explicit Zoom/Move-point controls.
        ax.hideButtons()
        ax1.hideButtons()
        ax2.hideButtons()
        ax3.hideButtons()

        # Native pg grid stays off everywhere; gridlines are drawn by the
        # ThemedGridItem overlay instead (see _apply_grid_item), toggled by
        # each card's gear menu and left as-is here - it's added directly to
        # the ViewBox rather than through the PlotItem, so it already
        # survives the ax.clear() calls above and needs no re-creation.
        ax.showGrid(x=False, y=False)
        ax1.showGrid(x=False, y=False)
        ax2.showGrid(x=False, y=False)
        ax3.showGrid(x=False, y=False)

        ax.setXRange(0, xs[-1], padding=0.05)
        ax.setYRange(0, max(np.amax(ys_freq), np.amax(ys), np.amax(ys_diff)), padding=0.05)

        self._set_lower_graphs_visible(False)

    def _animate_curve_reveal(
        self,
        xs: np.ndarray,
        series: List[Tuple[Any, np.ndarray, np.ndarray]],
    ) -> None:
        """Reveals a set of pyqtgraph curves left-to-right instead of
        popping in fully drawn, by animating how much of each curve's data
        `setData()` is given on every tick (same `QVariantAnimation` +
        callback convention as `SavedStateDot`/`_fade_overlay_opacity`,
        rather than `QPropertyAnimation`).

        The revealed slice is chosen by *time* (searching `xs` for the index
        nearest a target x-value that itself advances linearly), not by a
        fixed fraction of the sample *count* - this run's sample rate isn't
        constant (e.g. 20 Hz for the first ~90s of a capture, 10 Hz after),
        so advancing by a constant number-of-samples-per-tick would sweep
        across x (time) at a different apparent speed on each side of that
        boundary, reading as a stutter/kink right where the rate changes.
        Advancing by x instead keeps the wavefront moving across the plot at
        one constant speed regardless of how sample density varies along it.

        `_add_poi_markers`'s `_reveal_poi_markers` (called right after
        this, from `_render_analysis_plots`) rides this same
        `self._curve_reveal_anim` instance to grow each POI marker into
        place as the drawing front reaches that marker's x - see that
        method's docstring.

        Args:
            xs: The shared x-axis (time) array, ascending - both its length
                and its values are used here.
            series: `(curve_item, full_x, full_y)` triples to reveal in
                lockstep; each must share `xs`'s values.
        """
        n = len(xs)
        if n < 2:
            # No reveal for a degenerate run - and no stale animation left
            # behind either, since _reveal_poi_markers (called right after
            # this, from _add_poi_markers) treats a live self._curve_reveal_
            # anim as its cue to ride it; a leftover finished/stale one from
            # a previous, non-degenerate run would otherwise never fire its
            # valueChanged/finished again, stranding this run's markers
            # invisible forever.
            self._curve_reveal_anim = None
            return

        prev = getattr(self, "_curve_reveal_anim", None)
        if prev is not None:
            try:
                prev.stop()
            except RuntimeError:
                pass

        x0, x1 = xs[0], xs[-1]

        def _apply(fraction: float) -> None:
            target_x = x0 + fraction * (x1 - x0)
            idx = max(2, min(n, int(np.searchsorted(xs, target_x, side="right"))))
            for curve, x, y in series:
                try:
                    curve.setData(x[:idx], y[:idx])
                except RuntimeError:
                    pass

        _apply(0.0)  # collapse to the first couple points before the first paint

        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(self._CURVE_REVEAL_MS)
        # Linear, not eased - see the _CURVE_REVEAL_MS comment: this should
        # read as a steady sweep across the plot, not a fast/slow fade.
        anim.setEasingCurve(QtCore.QEasingCurve.Linear)
        anim.valueChanged.connect(_apply)

        def _finish() -> None:
            # Guarantee the exact full curve is shown, regardless of any
            # rounding in the last tick's revealed index.
            for curve, x, y in series:
                try:
                    curve.setData(x, y)
                except RuntimeError:
                    pass

        anim.finished.connect(_finish)
        self._curve_reveal_anim = anim
        anim.start()

    def _plot_signal_curves(self, curves: Dict[str, np.ndarray]) -> None:
        """Adds fit lines, scatter dots, and star highlights to all graph widgets.

        This method populates the main graph and the three specialized sub-graphs
        (Resonance, Difference, Dissipation) with raw data points, fitted curves,
        and interactive star markers representing POI.

        Args:
            curves: A dictionary containing NumPy arrays for plotting.
                Required keys:
                - 'xs': Shared x-axis time values.
                - 'ys', 'ys_freq', 'ys_diff': Raw signal data.
                - 'ys_fit', 'ys_freq_fit', 'ys_diff_fit': Calculated fit line data.
        """
        ax, ax1, ax2, ax3 = (
            self.graphWidget,
            self.graphWidget1,
            self.graphWidget2,
            self.graphWidget3,
        )
        xs = curves["xs"]
        ys = curves["ys"]
        ys_freq = curves["ys_freq"]
        ys_diff = curves["ys_diff"]
        ys_fit = curves["ys_fit"]
        ys_freq_fit = curves["ys_freq_fit"]
        ys_diff_fit = curves["ys_diff_fit"]

        mask = np.arange(0, len(xs), 1)
        # None, not a width=0 QPen: pyqtgraph/Qt renders a 0-width pen as a
        # cosmetic 1px line regardless of width, so the previous `noPen =
        # mkPen(color=(255,255,255), width=0, style=DotLine)` still drew a
        # visible white dotted line connecting every scatter point, despite
        # the name - all but invisible on the overview graph (whose scatter
        # sits at alpha 0.01) but plainly visible on the detail sub-graphs,
        # whose scatter alpha ranges up to 1.0 (see getPoints()'s show_scat)
        # - and, being pure white, it vanished against a light-mode
        # background while still showing on dark. `pen=None` is pyqtgraph's
        # actual "symbols only, no connecting line" - what this always
        # meant to be.
        noPen = None
        # Sourced from self._series_colors (defaults to SIGNAL_COLORS) rather
        # than the SIGNAL_COLORS constant directly, so a color picked via a
        # plot card's gear menu survives the next run load / re-plot.
        res_color = self._series_colors["resonance"]
        diff_color = self._series_colors["difference"]
        diss_color = self._series_colors["dissipation"]

        # Main graph - fit lines. This is the overview graph shown at the
        # run's full time range - by far the worst case for raw point count
        # vs. actual screen pixels. Temporarily wrapped in
        # _enable_adaptive_resolution (clipToView + pg's own auto-downsample)
        # purely so the reveal animation below stays smooth: measured cost of
        # a single full-resolution setData()+repaint on a ~150k-point curve
        # runs ~15ms, and the reveal fires ~90 of those across its six items
        # in well under 1.4s - without downsampling, painting can't keep up
        # with the animation's ticks and the "sweep" degenerates into the
        # curve just popping in. `_settle_to_full_resolution` (below)
        # switches every one of these six items back off of it the moment
        # the reveal finishes, so actually panning/zooming the settled plot
        # is still genuinely full-resolution, undownsampled data - only the
        # one-time entrance animation itself borrows this.
        # Pens match PlotsUI's own "glass" curve style (see
        # MainWindow._get_glass_curve_styles / glass_curve_pen) - a slight
        # translucency (alpha 215/255) at 2px width, antialiased - so the
        # overview's lines read as the same family as PlotsUI's real-time
        # plots rather than plain opaque 1px pyqtgraph defaults.
        self.fit1 = _enable_adaptive_resolution(
            ax.plot(
                xs[mask],
                ys_freq_fit[mask],
                pen=glass_curve_pen(res_color),
                antialias=True,
                name="Resonance",
            )
        )
        self.fit2 = _enable_adaptive_resolution(
            ax.plot(
                xs[mask],
                ys_diff_fit[mask],
                pen=glass_curve_pen(diff_color),
                antialias=True,
                name="Difference",
            )
        )
        self.fit3 = _enable_adaptive_resolution(
            ax.plot(
                xs[mask],
                ys_fit[mask],
                pen=glass_curve_pen(diss_color),
                antialias=True,
                name="Dissipation",
            )
        )

        # Main graph - scatter dots (nearly transparent). symbolPen=None -
        # a flat fill, no outline ring around each dot.
        self.scat1 = _enable_adaptive_resolution(
            ax.plot(
                xs[mask],
                ys_freq[mask],
                pen=noPen,
                symbol="o",
                symbolSize=5,
                symbolBrush=res_color,
                symbolPen=None,
            )
        )
        self.scat2 = _enable_adaptive_resolution(
            ax.plot(
                xs[mask],
                ys_diff[mask],
                pen=noPen,
                symbol="o",
                symbolSize=5,
                symbolBrush=diff_color,
                symbolPen=None,
            )
        )
        self.scat3 = _enable_adaptive_resolution(
            ax.plot(
                xs[mask],
                ys[mask],
                pen=noPen,
                symbol="o",
                symbolSize=5,
                symbolBrush=diss_color,
                symbolPen=None,
            )
        )
        self.scat1.setAlpha(0.01, False)
        self.scat2.setAlpha(0.01, False)
        self.scat3.setAlpha(0.01, False)

        # Left-to-right draw-in reveal for the main overview graph - the one
        # plot visible the instant a run's data appears (the three detail
        # sub-graphs below start hidden via self.lowerGraphs.setVisible(False)
        # in _setup_graph_axes, so they'd have nothing to reveal yet anyway).
        # Only the fit lines are animated - the scatter dot layers above are
        # set to alpha 0.01 (essentially invisible), so collapsing/restoring
        # their data in lockstep bought no visible payoff while still paying
        # a full downsample+repaint pass for each of them on every tick. They
        # keep the full data they were created with instead. _add_poi_markers
        # (called right after this, from _render_analysis_plots) rides this
        # same reveal animation to grow each POI marker into place as the
        # drawing front reaches it - see _reveal_poi_markers.
        self._animate_curve_reveal(
            xs,
            [
                (self.fit1, xs[mask], ys_freq_fit[mask]),
                (self.fit2, xs[mask], ys_diff_fit[mask]),
                (self.fit3, xs[mask], ys_fit[mask]),
            ],
        )

        def _settle_to_full_resolution(
            items=(self.fit1, self.fit2, self.fit3, self.scat1, self.scat2, self.scat3)
        ) -> None:
            """Switches every main-graph curve off of _enable_adaptive_
            resolution's clipToView/auto-downsampling, once the reveal
            animation above no longer needs it for smooth painting - from
            here on these render every raw sample, full resolution, exactly
            as plotted.
            """
            for item in items:
                try:
                    item.setClipToView(False)
                    item.setDownsampling(auto=False)
                except RuntimeError:
                    pass

        if self._curve_reveal_anim is not None:
            self._curve_reveal_anim.finished.connect(_settle_to_full_resolution)
        else:
            _settle_to_full_resolution()

        # Sub-graphs - fit lines. These zoom to a narrow window around the
        # current POI (see getPoints()'s ax1/ax2/ax3.setXRange calls), so
        # clipToView+auto-downsampling is usually a no-op here - applied
        # anyway for consistency and for high-sample-rate runs where even a
        # narrow window still holds more points than screen pixels. Pens
        # match the overview graph's own "glass" curve style (see
        # glass_curve_pen) during Channel 1/2/3 - the only steps where
        # these are actually visible (getPoints()'s show_fits sets them to
        # alpha 0.0 during Fill Start/Fill End, where the sub-graph's raw
        # point cloud is the only real content).
        self.fit_1 = _enable_adaptive_resolution(
            ax1.plot(
                xs[mask],
                ys_freq_fit[mask],
                pen=glass_curve_pen(res_color),
                antialias=True,
                name="Resonance",
            )
        )
        self.fit_2 = _enable_adaptive_resolution(
            ax2.plot(
                xs[mask],
                ys_diff_fit[mask],
                pen=glass_curve_pen(diff_color),
                antialias=True,
                name="Difference",
            )
        )
        self.fit_3 = _enable_adaptive_resolution(
            ax3.plot(
                xs[mask],
                ys_fit[mask],
                pen=glass_curve_pen(diss_color),
                antialias=True,
                name="Dissipation",
            )
        )

        # Sub-graphs - scatter dots. symbolPen=None - a flat fill, no
        # outline ring around each dot, matching the overview graph's own
        # scatter dots (see above).
        self.scat_1 = _enable_adaptive_resolution(
            ax1.plot(
                xs[mask],
                ys_freq[mask],
                pen=noPen,
                symbol="o",
                symbolSize=5,
                symbolBrush=res_color,
                symbolPen=None,
            )
        )
        self.scat_2 = _enable_adaptive_resolution(
            ax2.plot(
                xs[mask],
                ys_diff[mask],
                pen=noPen,
                symbol="o",
                symbolSize=5,
                symbolBrush=diff_color,
                symbolPen=None,
            )
        )
        self.scat_3 = _enable_adaptive_resolution(
            ax3.plot(
                xs[mask],
                ys[mask],
                pen=noPen,
                symbol="o",
                symbolSize=5,
                symbolBrush=diss_color,
                symbolPen=None,
            )
        )

        # Star markers (current-POI highlights)
        pos1 = np.column_stack((xs[0], ys_freq[0]))
        pos2 = np.column_stack((xs[0], ys_diff[0]))
        pos3 = np.column_stack((xs[0], ys[0]))

        # Theme-derived, not the old hardcoded "black"/"gray" - those only
        # ever read correctly in light mode. Same tokens _apply_pg_theme
        # re-applies on every themeChanged, so these stay correct across a
        # live theme switch too, not just at creation.
        bold_color, faint_color = self._target_marker_colors()

        self.star1 = pg.ScatterPlotItem(
            pos=pos1, symbol=_TARGET_SYMBOL, size=25, pen=_target_pen(bold_color), brush=bold_color
        )
        self.star2 = pg.ScatterPlotItem(
            pos=pos2, symbol=_TARGET_SYMBOL, size=25, pen=_target_pen(bold_color), brush=bold_color
        )
        self.star3 = pg.ScatterPlotItem(
            pos=pos3, symbol=_TARGET_SYMBOL, size=25, pen=_target_pen(bold_color), brush=bold_color
        )
        ax1.addItem(self.star1)
        ax2.addItem(self.star2)
        ax3.addItem(self.star3)

        self.gstars1 = pg.ScatterPlotItem(
            pos=pos1,
            symbol=_TARGET_SYMBOL,
            size=10,
            pen=_target_pen(faint_color),
            brush=faint_color,
        )
        self.gstars2 = pg.ScatterPlotItem(
            pos=pos2,
            symbol=_TARGET_SYMBOL,
            size=10,
            pen=_target_pen(faint_color),
            brush=faint_color,
        )
        self.gstars3 = pg.ScatterPlotItem(
            pos=pos3,
            symbol=_TARGET_SYMBOL,
            size=10,
            pen=_target_pen(faint_color),
            brush=faint_color,
        )
        ax1.addItem(self.gstars1)
        ax2.addItem(self.gstars2)
        ax3.addItem(self.gstars3)

        # Re-apply any per-series visibility the user set via a gear menu
        # before this run was (re)loaded - _plot_signal_curves() rebuilds
        # every curve/marker item from scratch, so the previous items'
        # visibility state doesn't carry over on its own.
        for series_key, attrs in self._SERIES_CURVE_ATTRS.items():
            visible = self._series_visible.get(series_key, True)
            if visible:
                continue
            for attr in attrs + self._SERIES_STAR_ATTRS.get(series_key, ()):
                getattr(self, attr).setVisible(False)

        # Re-apply the overview and detail cards' own "Point-to-Point"
        # preferences to this run's freshly-created scatter dot layers -
        # see _apply_overview_point_cloud_visibility /
        # _apply_detail_point_cloud_visibility.
        self._apply_overview_point_cloud_visibility()
        for series_key in self._SERIES_CURVE_ATTRS:
            self._apply_detail_point_cloud_visibility(series_key)

        # Clamp pan/zoom on every graph (main overview + the three POI
        # detail graphs) to this run's own data - see _apply_plot_limits.
        # Each detail graph only shows one signal family, so its limits are
        # sized to that family alone rather than the combined range below.
        self._apply_plot_limits(ax, xs, ys, ys_freq, ys_diff, ys_fit, ys_freq_fit, ys_diff_fit)
        self._apply_plot_limits(ax1, xs, ys_freq, ys_freq_fit)
        self._apply_plot_limits(ax2, xs, ys_diff, ys_diff_fit)
        self._apply_plot_limits(ax3, xs, ys, ys_fit)

        # Guards against a freshly loaded plot occasionally settling into a
        # too-wide/off-center initial view instead of the resting frame just
        # computed above - see _snap_view_into_bounds. A single manual pan/
        # zoom already self-corrects this (via _bounce_view_into_bounds,
        # whose clamp math this reuses), so this just performs that same
        # correction automatically rather than leaving the plot wrong until
        # the user happens to interact with it. Checked a few times over
        # the first ~400ms rather than once - the exact timing of whatever
        # transient condition causes the initial view to drift wasn't
        # pinned down, so this re-checks instead of guessing the one moment
        # it happens; each check is a cheap no-op once the view is already
        # correct.
        for vb in (ax.getViewBox(), ax1.getViewBox(), ax2.getViewBox(), ax3.getViewBox()):
            for delay_ms in (0, 50, 150, 400):
                QtCore.QTimer.singleShot(delay_ms, lambda vb=vb: self._snap_view_into_bounds(vb))

    def _set_overview_point_to_point(self, enabled: bool) -> None:
        """Applies the overview card's "Point-to-Point Rendering" gear-menu
        preference: shows or hides the overview graph's raw-data "point
        cloud" - the near-invisible (alpha 0.01) scatter-dot layers
        (scat1/scat2/scat3) that the solid fit lines are a smoothed average
        through. Doesn't touch the fit lines themselves, or either curve
        type's resolution/downsampling - those are unaffected and always
        settle to full resolution once the entrance reveal finishes (see
        `_plot_signal_curves`), exactly as before this toggle existed.

        A plain user preference (`self._overview_point_to_point`, stored so
        a freshly loaded run applies whichever state the user last chose -
        see `_plot_signal_curves`), not something this code decides on its
        own: this mirrors how the point cloud used to be hidden
        automatically while actively panning/zooming (an earlier,
        gesture-driven approach that was reverted in favor of direct user
        control).
        """
        self._overview_point_to_point = enabled
        self._apply_overview_point_cloud_visibility()

    def _apply_overview_point_cloud_visibility(self) -> None:
        """Shows/hides each of scat1/scat2/scat3 (the overview graph's raw-
        data "point cloud" - see `_set_overview_point_to_point`) using the
        AND of two independent preferences: the overview card's global
        "Point-to-Point" toggle (`self._overview_point_to_point`) and that
        series's own per-series visibility (`self._series_visible`, from
        the eye-icon toggle next to its color swatch - see
        `_on_analyze_section_visibility_changed`). Either one hiding a
        series is enough to hide its point cloud; both must allow it for
        the dots to actually show.

        getattr-guarded since this can run before a run has ever been
        loaded (the gear-menu toggle is clickable as soon as
        SignalOverviewCard exists, before `_plot_signal_curves` has
        created these) - a click in that state just records the
        preference for the first plot to apply once it exists.
        """
        for series_key, scat_attr in self._OVERVIEW_SCATTER_ATTR.items():
            item = getattr(self, scat_attr, None)
            if item is None:
                continue
            visible = self._overview_point_to_point and self._series_visible.get(series_key, True)
            try:
                item.setVisible(visible)
            except RuntimeError:
                pass

    def _set_detail_point_to_point(self, key: str, enabled: bool) -> None:
        """Applies one detail card's own "Point-to-Point Rendering" gear-
        menu preference - see `_set_overview_point_to_point`, its overview-
        card counterpart, for the general idea. Independent per card: the
        Resonance card's toggle never affects Difference's or
        Dissipation's point cloud.

        Args:
            key: One of "resonance"/"difference"/"dissipation" - which
                detail card's toggle fired (bound via `partial` where this
                is connected).
            enabled: The new toggle state.
        """
        self._detail_point_to_point[key] = enabled
        self._apply_detail_point_cloud_visibility(key)

    def _apply_detail_point_cloud_visibility(self, key: str) -> None:
        """Shows/hides one detail sub-graph's raw-data point cloud
        (scat_1/scat_2/scat_3 - see `_DETAIL_SCATTER_ATTR`), using the AND
        of that card's own "Point-to-Point" toggle
        (`self._detail_point_to_point`) and the series's per-series
        visibility (`self._series_visible`) - the same combination
        `_apply_overview_point_cloud_visibility` applies to the overview
        graph's point cloud, just scoped to one series/card at a time.

        Outside the Channel 1/2/3 workflow steps (`self.stateStep < 3` -
        Fill Start/Fill End), the toggle is ignored entirely and the point
        cloud is forced visible (still subject to per-series visibility):
        `getPoints()`'s own `show_fits`/`show_scat` alphas make the fit
        line invisible during those two steps, so the point cloud is the
        only thing actually plotted there - a stale "off" preference left
        over from a Channel step must never black out the whole sub-graph.
        See `_update_detail_point_cloud_state`, which also disables the
        gear-menu row itself outside those steps.
        """
        scat_attr = self._DETAIL_SCATTER_ATTR.get(key)
        if scat_attr is None:
            return
        item = getattr(self, scat_attr, None)
        if item is None:
            return
        point_to_point = self._detail_point_to_point.get(key, True) if self.stateStep >= 3 else True
        visible = point_to_point and self._series_visible.get(key, True)
        try:
            item.setVisible(visible)
        except RuntimeError:
            pass

    def _update_detail_point_cloud_state(self) -> None:
        """Keeps every detail card's "Point-to-Point Rendering" toggle in
        sync with the current workflow step: enabled (and its preference
        actually applied) only during Channel 1/2/3 (`self.stateStep >=
        3`); disabled - and its point cloud forced visible regardless of
        the stored preference - during Fill Start/Fill End, where the fit
        line is invisible and the point cloud is the only content on
        screen (see `_apply_detail_point_cloud_visibility`).

        Called from `getPoints()` (every time the workflow step changes)
        and once at setup, so the toggle starts disabled before any run's
        wizard has reached a Channel step.
        """
        available = self.stateStep >= 3
        for key, card in (
            ("resonance", getattr(self, "resonance_card", None)),
            ("difference", getattr(self, "difference_card", None)),
            ("dissipation", getattr(self, "dissipation_card", None)),
        ):
            if card is not None:
                card.set_point_to_point_available(available)
            self._apply_detail_point_cloud_visibility(key)

    # Debounce window after the last manual pan/zoom tick before checking
    # whether the view needs to bounce back (see _schedule_view_bounce) -
    # short enough to feel responsive, long enough that a continuous drag
    # or scroll (which re-fires sigRangeChangedManually on every tick)
    # never lets the timer actually elapse mid-gesture. Bounce-back
    # animation duration/easing is separate - see _bounce_view_into_bounds.
    _BOUNCE_DEBOUNCE_MS = 220
    _BOUNCE_DURATION_MS = 420

    def _apply_plot_limits(
        self,
        ax: pg.PlotWidget,
        xs: np.ndarray,
        *y_arrays: np.ndarray,
        x_pad_frac: float = 0.05,
        y_pad_frac: float = 0.08,
        max_zoom_out: float = 1.6,
    ) -> None:
        """Lets the user pan/zoom `ax` as far as they like, but softly
        bounces it back to a sane resting frame shortly after they stop -
        rather than a hard `ViewBox.setLimits()` wall, which stops a drag/
        scroll dead the instant it crosses the boundary and reads as
        hitting a wall rather than reaching an edge.

        The resting frame: `x_pad_frac`/`y_pad_frac` give it a small
        margin past the run's actual data on every side (a snap-back to
        *exactly* the data edge would still feel abrupt); `max_zoom_out`
        caps how wide that resting frame's span can be, as a multiple of
        the already-padded data span, so "zoomed out" always settles
        somewhere the curve still reads as more than a sliver.

        Wires `ax`'s ViewBox up to `_schedule_view_bounce` via
        `sigRangeChangedManually` (fired only by actual mouse/wheel
        interaction, never by this file's own programmatic `setXRange`/
        `setYRange` calls - e.g. `getPoints()`'s per-step POI-focus
        zooming - so those never get bounced). Re-applying this (a fresh
        run load) disconnects the previous handler first, since the
        ViewBox itself persists across `ax.clear()`.

        Deliberately doesn't cap zooming *in* - that's exactly what the
        detail graphs are for when placing a POI precisely.
        """
        if xs is None or len(xs) < 2:
            return
        x0, x1 = float(xs[0]), float(xs[-1])
        y0, y1 = self._data_y_range(*y_arrays)

        x_span = max(x1 - x0, 1e-9)
        y_span = max(y1 - y0, 1e-9)
        x_pad = x_span * x_pad_frac
        y_pad = y_span * y_pad_frac

        vb = ax.getViewBox()
        vb._pan_zoom_limits = {
            "xMin": x0 - x_pad,
            "xMax": x1 + x_pad,
            "yMin": y0 - y_pad,
            "yMax": y1 + y_pad,
            "maxXRange": (x_span + 2 * x_pad) * max_zoom_out,
            "maxYRange": (y_span + 2 * y_pad) * max_zoom_out,
        }

        prev_handler = getattr(vb, "_bounce_handler", None)
        if prev_handler is not None:
            try:
                vb.sigRangeChangedManually.disconnect(prev_handler)
            except (TypeError, RuntimeError):
                pass

        def _handler(_mask, vb=vb):
            self._schedule_view_bounce(vb)

        vb.sigRangeChangedManually.connect(_handler)
        vb._bounce_handler = _handler

    def _schedule_view_bounce(self, vb: pg.ViewBox) -> None:
        """(Re)starts the debounce timer that checks `vb` against its soft
        pan/zoom bounds shortly after the user stops interacting with it.

        `sigRangeChangedManually` fires on every tick of a drag/scroll, not
        just once it ends - restarting a short single-shot timer on every
        call, and only acting once it actually elapses, is what makes the
        correction land after the gesture settles instead of fighting it
        mid-drag. If a bounce-back animation is already in flight, this
        manual change means the user grabbed the view again - stop
        correcting and let them.
        """
        anim = getattr(vb, "_bounce_anim", None)
        if anim is not None:
            try:
                anim.stop()
            except RuntimeError:
                pass
            vb._bounce_anim = None

        timer = getattr(vb, "_bounce_timer", None)
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda vb=vb: self._bounce_view_into_bounds(vb))
            vb._bounce_timer = timer
        timer.start(self._BOUNCE_DEBOUNCE_MS)

    def _bounce_view_into_bounds(self, vb: pg.ViewBox) -> None:
        """Eases `vb`'s current view range back within its soft pan/zoom
        bounds (see `_apply_plot_limits`) if it's currently outside them -
        a no-op otherwise. `OutBack` easing gives the settle a slight
        overshoot past the target before it comes to rest, reading as an
        actual bounce rather than a plain slide-back.
        """
        limits = getattr(vb, "_pan_zoom_limits", None)
        if limits is None:
            return
        try:
            (x0, x1), (y0, y1) = vb.viewRange()
        except RuntimeError:
            return

        tx0, tx1 = self._clamp_axis_range(
            x0, x1, limits["xMin"], limits["xMax"], limits["maxXRange"]
        )
        ty0, ty1 = self._clamp_axis_range(
            y0, y1, limits["yMin"], limits["yMax"], limits["maxYRange"]
        )

        if (tx0, tx1) == (x0, x1) and (ty0, ty1) == (y0, y1):
            return

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self._BOUNCE_DURATION_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)

        def _apply(t: float) -> None:
            try:
                vb.setRange(
                    xRange=(x0 + (tx0 - x0) * t, x1 + (tx1 - x1) * t),
                    yRange=(y0 + (ty0 - y0) * t, y1 + (ty1 - y1) * t),
                    padding=0,
                )
            except RuntimeError:
                pass

        anim.valueChanged.connect(_apply)

        def _on_finished(vb=vb) -> None:
            if getattr(vb, "_bounce_anim", None) is anim:
                vb._bounce_anim = None

        anim.finished.connect(_on_finished)
        vb._bounce_anim = anim
        anim.start()

    def _snap_view_into_bounds(self, vb: pg.ViewBox) -> None:
        """Instantly clamps `vb`'s current view range back within its soft
        pan/zoom bounds (see `_apply_plot_limits`) if it's outside them -
        the same `_clamp_axis_range` math `_bounce_view_into_bounds` uses,
        without that method's animation. Called a few times shortly after
        a fresh plot loads (see `_plot_signal_curves`) to silently correct
        an occasionally-too-wide initial view before the user ever sees it.
        """
        limits = getattr(vb, "_pan_zoom_limits", None)
        if limits is None:
            return
        try:
            (x0, x1), (y0, y1) = vb.viewRange()
        except RuntimeError:
            return

        tx0, tx1 = self._clamp_axis_range(
            x0, x1, limits["xMin"], limits["xMax"], limits["maxXRange"]
        )
        ty0, ty1 = self._clamp_axis_range(
            y0, y1, limits["yMin"], limits["yMax"], limits["maxYRange"]
        )

        if (tx0, tx1) == (x0, x1) and (ty0, ty1) == (y0, y1):
            return

        try:
            vb.setRange(xRange=(tx0, tx1), yRange=(ty0, ty1), padding=0)
        except RuntimeError:
            pass

    @staticmethod
    def _clamp_axis_range(
        lo: float, hi: float, min_bound: float, max_bound: float, max_span: float
    ) -> Tuple[float, float]:
        """Nearest in-bounds `[lo, hi]` to the given range: first shrinks
        it (around its own center) to fit within `max_span`, then slides
        it to fit within `[min_bound, max_bound]`.
        """
        span = hi - lo
        if span > max_span:
            center = (lo + hi) / 2.0
            lo, hi = center - max_span / 2.0, center + max_span / 2.0
        if lo < min_bound:
            hi += min_bound - lo
            lo = min_bound
        if hi > max_bound:
            lo -= hi - max_bound
            hi = max_bound
        return max(lo, min_bound), min(hi, max_bound)

    @staticmethod
    def _data_y_range(*y_arrays: np.ndarray) -> Tuple[float, float]:
        """Real min/max across the given plotted-data arrays (ignoring
        NaN/inf). Used both to size POI markers' finite vertical extent
        (see `POIMarker.setDataRange`) and to compute each plot's pan/zoom
        limits (see `_apply_plot_limits`) - both want the run's actual data
        extent, not the current (zoomable) view. Falls back to (0.0, 1.0)
        if nothing usable is given.
        """
        parts = [np.asarray(a).ravel() for a in y_arrays if a is not None and len(a)]
        if not parts:
            return 0.0, 1.0
        finite = np.concatenate(parts)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(finite.min()), float(finite.max())

    def _style_poi_marker(self, marker: "POIMarker", active: bool = True) -> None:
        """Applies theme-driven colors to a POI marker.

        `active=False` renders a muted tone - used by `getPoints()` to
        distinguish the single currently-draggable marker (per wizard
        step) from the rest. `POIMarker` always shows its handle (see
        `POIMarker.paint`), unlike the old `InfiniteLine` arrow glyph that
        `getPoints()` used to add/clear per-step as the sole "this one's
        active" cue - color now carries that distinction instead.

        `active` is stored on the marker itself (rather than re-derived
        from `setMovable`) so `_apply_pg_theme` can restore the same look
        on a theme switch - the two don't always agree: every marker is
        `setMovable(False)` at the step-7 summary, but all of them should
        still read as fully "active"/confirmed there, not muted.
        """
        marker._active_style = active
        tok = ThemeManager.instance().tokens()
        line_color = QtGui.QColor(*(tok["accent"] if active else tok["plot_text_dim"]))
        marker.setPen(pg.mkPen(line_color, width=2))
        marker.setHoverPen(pg.mkPen(QtGui.QColor(*tok["flat_accent_hover"]), width=2.5))
        marker.setHandleOutlineColor(QtGui.QColor(*tok["plot_glass_rim"]))

    def _target_marker_colors(self) -> Tuple[QtGui.QColor, QtGui.QColor]:
        """Returns (bold, faint) colors for the detail sub-graphs'
        current-POI target markers (star1/2/3, gstars1/2/3) - theme
        tokens rather than the original hardcoded "black"/"gray", which
        only ever read correctly in light mode. `plot_text_normal`/
        `plot_text_dim` are the same muted-text token family used
        elsewhere on these plots (see `_apply_pg_theme`'s axis text,
        `_style_poi_marker`'s muted marker state) precisely because
        they're already built to read against either theme's background.
        """
        tok = ThemeManager.instance().tokens()
        return QtGui.QColor(*tok["plot_text_normal"]), QtGui.QColor(*tok["plot_text_dim"])

    def _style_target_markers(self) -> None:
        """Re-applies `_target_marker_colors` to whichever of star1/2/3 /
        gstars1/2/3 currently exist - called on init and on every
        `themeChanged` (see `_apply_pg_theme`), the same way
        `_style_poi_marker` keeps POI markers correct across a live theme
        switch.
        """
        bold_color, faint_color = self._target_marker_colors()
        for attrs, color in (
            (("star1", "star2", "star3"), bold_color),
            (("gstars1", "gstars2", "gstars3"), faint_color),
        ):
            for attr in attrs:
                item = getattr(self, attr, None)
                if item is None:
                    continue
                item.setBrush(color)
                item.setPen(_target_pen(color))

    def _make_poi_marker(self, x: float, xs: np.ndarray, y0: float, y1: float) -> "POIMarker":
        """Builds one themed, finite-extent `POIMarker` at data-x `x`,
        bounded to `[xs[0], xs[-1]]` and vertically sized to `[y0, y1]`
        (see `POIMarker.setDataRange`/`_data_y_range`).
        """
        marker = POIMarker(pos=x, angle=90, movable=True, bounds=[xs[0], xs[-1]])
        marker.setDataRange(y0, y1)
        self._style_poi_marker(marker, active=True)
        return marker

    def _add_poi_markers(
        self, curves: Dict[str, Any], poi_vals: List[int], start_stop: List[int]
    ) -> None:
        """Places movable POIMarker markers on the main graph.

        This method initializes vertical markers (POI) on the
        graphWidget. If `poi_vals` is provided, it undergoes validation to ensure
        indices are within the bounds of the `xs` array. Validated `poi_vals`
        will take precedence over the `start_stop` values.

        Args:
            curves: The signal-data dict produced by the run-load pipeline
                (see `_render_analysis_plots`). Only `xs`/`ys`/`ys_freq`/
                `ys_diff` are read here - the latter three purely to size
                each marker's finite vertical extent via
                `_data_y_range`.
            poi_vals: A list of integer indices representing predefined POIs.
                Indices that are out of bounds (except -1) are reset to -1 and logged.
            start_stop: A fallback list of integer indices used if `poi_vals`
                is not provided or to define the initial marker set.
        """
        ax = self.graphWidget
        xs = curves["xs"]
        y0, y1 = self._data_y_range(curves["ys"], curves["ys_freq"], curves["ys_diff"])

        if poi_vals:
            for i, pt in enumerate(poi_vals):
                if pt != -1 and not 0 <= pt < len(xs):
                    Log.w(f"Model point {pt} cannot be used. Skipping point {i + 1}.")
                    poi_vals[i] = -1
            start_stop = poi_vals

        self.poi_markers = []
        marker_targets: List[Tuple["POIMarker", float]] = []
        for idx, pt in enumerate(start_stop):
            marker = self._make_poi_marker(xs[pt], xs, y0, y1)
            if idx == 2:
                marker.setVisible(False)
            ax.addItem(marker)
            marker.sigPositionChangeFinished.connect(self.markerMoveFinished)
            self.poi_markers.append(marker)
            marker_targets.append((marker, xs[pt]))

        self._reveal_poi_markers(xs[0], xs[-1], marker_targets)
        self._reset_step_visibility()

    def _reset_step_visibility(self) -> None:
        """Resets the +/- step-visibility state to "everything shown" -
        called once per freshly (re)loaded run, right after
        `_add_poi_markers` rebuilds all 6 markers from scratch (unhidden,
        every time - see that method). If the user had reduced
        `active_count` during a *previous* run this session, the stepper
        itself doesn't otherwise know to grow back on its own, since
        nothing else re-syncs it to a freshly loaded run's own markers -
        this keeps the two in lockstep.
        """
        self._parked_marker_values = {}
        while self.stepper.step_count() - 2 < len(self._INTERMEDIATE_STEPS):
            label, _ = self._INTERMEDIATE_STEPS[self.stepper.step_count() - 2]
            self.stepper.add_step(label)
        self.active_count = len(self._INTERMEDIATE_STEPS)
        self._update_step_buttons_enabled()

    def _animate_marker_reveal(self, marker: "POIMarker", start: float, end: float) -> None:
        """Animates one `POIMarker`'s entrance/exit via its existing
        `setRevealProgress` primitive (see `POIMarker`) - used by
        `_on_remove_step_requested`/`_on_add_step_requested` when a
        stepper +/- click shows/hides one marker. Deliberately a small,
        standalone `QVariantAnimation` rather than `_reveal_poi_markers`'s
        sweep machinery below, which is a one-shot, whole-plot, left-to-
        right entrance reveal tied to initial run load and long finished
        by the time a user clicks a stepper button.

        Stashed on the marker itself (mirroring `vb._bounce_anim`-style
        state elsewhere in this file) so a second call for the same
        marker - e.g. a fast add/remove/add sequence - stops whichever
        animation was already in flight rather than fighting it.
        """
        prev = getattr(marker, "_step_vis_anim", None)
        if prev is not None:
            prev.stop()
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self._STEP_VIS_ANIM_MS)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(lambda v, m=marker: m.setRevealProgress(v))
        marker._step_vis_anim = anim
        anim.start()

    def _animate_marker_removal(self, marker: "POIMarker", target_value: float) -> None:
        """Animates one `POIMarker`'s removal via +/- "-": turns it red and
        slides its position from wherever it currently sits to
        `target_value` (`xs[-1]`, the existing "channel not present"
        convention - see `_on_remove_step_requested`) while shrinking it
        away (`setRevealProgress` 1.0->0.0), all together in one animation
        instead of an instant position jump (the old `marker.setValue(...)`
        call) followed by a separate in-place fade - reads as the marker
        actively closing rather than teleporting then vanishing.

        Mirrors `_animate_marker_reveal`'s stash-on-marker convention
        (`marker._step_vis_anim`) so a fast remove/add/remove sequence
        stops whichever animation was already in flight rather than
        fighting it. `_on_add_step_requested` resets the marker's color
        back to its normal active tint (`_style_poi_marker`) before fading
        it back in, so the red tint never lingers past one removal.
        """
        prev = getattr(marker, "_step_vis_anim", None)
        if prev is not None:
            prev.stop()
        start_value = marker.value()
        start_color = QtGui.QColor(marker.pen.color())
        end_color = QtGui.QColor("#DA2E2E")

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self._STEP_VIS_ANIM_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        def _tick(t: float, m=marker) -> None:
            m.setRevealProgress(1.0 - t)
            m.setValue(start_value + (target_value - start_value) * t)
            m.setPen(
                pg.mkPen(
                    QtGui.QColor(
                        round(start_color.red() + (end_color.red() - start_color.red()) * t),
                        round(start_color.green() + (end_color.green() - start_color.green()) * t),
                        round(start_color.blue() + (end_color.blue() - start_color.blue()) * t),
                    ),
                    width=2,
                )
            )

        anim.valueChanged.connect(_tick)
        marker._step_vis_anim = anim
        anim.start()

    def _on_remove_step_requested(self) -> None:
        """Handles the stepper's "-" button: hides the last currently-
        shown intermediate step (Channel 3 first, working back to Fill
        Start - see `_INTERMEDIATE_STEPS`), turning its `POIMarker` red and
        sliding it to `xs[-1]` while it shrinks away (see
        `_animate_marker_removal`).

        Deliberately view-only: the marker is *not* removed from
        `self.poi_markers`/the scene, and `self.stateStep`/`_STEP_NUMS`/
        `_current_visible_poi_index()` are untouched. Ending at `xs[-1]` is
        exactly the existing "channel not present in this run" convention
        (see `getContextWidth`/`getPoints()`'s clipped-context skip) - it's
        what makes Back/Next already glide straight past this step without
        any further changes needed here. A real, user-set position is
        remembered (see `_parked_marker_values`) so a later "+" can restore
        it instead of forcing the user to re-drag from the edge.
        """
        if self.active_count <= 0:
            return
        _, marker_idx = self._INTERMEDIATE_STEPS[self.active_count - 1]
        markers = getattr(self, "poi_markers", None)
        xs = getattr(self, "xs", None)
        if markers is not None and xs is not None and marker_idx < len(markers):
            marker = markers[marker_idx]
            current_val = marker.value()
            if abs(current_val - xs[-1]) > 1e-9:
                self._parked_marker_values[marker_idx] = current_val
            marker.setMovable(False)
            self._animate_marker_removal(marker, xs[-1])
        self.active_count -= 1
        self.stepper.remove_step()
        self._update_step_buttons_enabled()
        self._apply_cached_channel_config()

    def _on_add_step_requested(self, restore_position: bool = True) -> None:
        """Handles the stepper's "+" button - the mirror image of
        `_on_remove_step_requested`: reveals the next hidden intermediate
        step (in fixed order - see `_INTERMEDIATE_STEPS`), resets its
        `POIMarker`'s color back from the removal animation's red tint,
        and fades it back in.

        `restore_position=True` (the default, used for an actual "+"
        click) restores the marker's last real position if one was
        remembered (see `_parked_marker_values`), or leaves it parked at
        `xs[-1]` (same "drag it into place" convention a freshly-
        undetected channel already uses) if not. `restore_position=False`
        (used by `_reveal_steps_for_poi_vals`, when auto-fit itself just
        set the marker to a fresh predicted position) skips that restore
        entirely, so a stale remembered position can't clobber the new
        one - the parked value is still discarded either way, so it can't
        linger and apply incorrectly on some later reveal.

        Also marks the newly-revealed pill as "reached" (see
        `PillStepper.mark_reached`) so it's immediately clickable, rather
        than requiring the user to click "Next" repeatedly until Back/
        Next's normal forward progress happens to reach it.
        """
        if self.active_count >= len(self._INTERMEDIATE_STEPS):
            return
        label, marker_idx = self._INTERMEDIATE_STEPS[self.active_count]
        self.active_count += 1
        markers = getattr(self, "poi_markers", None)
        if markers is not None and marker_idx < len(markers):
            marker = markers[marker_idx]
            restored = self._parked_marker_values.pop(marker_idx, None)
            if restore_position and restored is not None:
                marker.setValue(restored)
            marker.setMovable(True)
            # Undo _animate_marker_removal's red tint before fading back in,
            # so it never lingers past one removal.
            self._style_poi_marker(marker, active=True)
            self._animate_marker_reveal(marker, 0.0, 1.0)
        self.stepper.add_step(label)
        self.stepper.mark_reached(self.stepper.step_count() - 2)
        self._update_step_buttons_enabled()
        if restore_position:
            self._apply_cached_channel_config()

    def _apply_cached_channel_config(self) -> None:
        """After `active_count` changes via +/-, if a cached 0-3-channel
        auto-fit hypothesis exists for the resulting channel count (see
        `_cache_channel_hypotheses`), snap every currently-visible
        intermediate marker straight to it instead of leaving stale
        positions from a different channel-count hypothesis.

        No-ops below the Fill Start/Fill End floor (`active_count < 2`,
        i.e. no meaningful channel count in play) or when no cache exists
        for this run/engine (Indus/Tweed don't support forcing a channel
        count, or a given hypothesis failed to compute) - falls back to
        the existing parking/restore behavior untouched in that case.

        Deliberately overrides any restored `_parked_marker_values`
        position for markers a cache hit covers: the cache reflects what
        the model actually predicts for this exact channel count, which is
        strictly more useful than a stale hand-dragged position from a
        different one.
        """
        if self.active_count < 2:
            return
        cached = self._channel_config_cache.get(self.active_count - 2)
        if cached is None:
            return
        markers = getattr(self, "poi_markers", None)
        xs = getattr(self, "xs", None)
        if markers is None or xs is None:
            return
        for step_idx in range(self.active_count):
            _, marker_idx = self._INTERMEDIATE_STEPS[step_idx]
            if marker_idx >= len(markers) or marker_idx >= len(cached):
                continue
            target = int(cached[marker_idx])
            if target == -1:
                continue
            target = max(0, min(len(xs) - 1, target))
            markers[marker_idx].setValue(xs[target])
            self._parked_marker_values.pop(marker_idx, None)
            # POI3 (index 2, permanently hidden) rides along with Fill End
            # (index 1) everywhere else in this file - keep it consistent.
            if marker_idx == 1 and 2 < len(markers) and 2 < len(cached):
                post = max(0, min(len(xs) - 1, int(cached[2])))
                markers[2].setValue(xs[post])
        self.detect_change()

    def _reveal_steps_for_poi_vals(self, poi_vals: List[int]) -> None:
        """After auto-fit (re-)runs (see `_restore_qmodel_predictions`) and
        produces a fresh full 6-point `poi_vals`, reveals any additional
        intermediate steps the model found real positions for beyond
        what's currently shown via +/- (see `_INTERMEDIATE_STEPS`/
        `active_count`) - e.g. if the user had previously hidden Channel 3
        and auto-fit is re-run on a run that does have a third channel,
        that step's marker/pill reappear automatically instead of staying
        hidden with a stale "missing" assumption.

        Grows the shown prefix one step at a time (see
        `_on_add_step_requested`, called with `restore_position=False`
        since the position was just freshly set by the caller, not a
        stale remembered one), stopping at the first still-undetected
        (-1) point, since the active set is always a contiguous prefix of
        `_INTERMEDIATE_STEPS`.
        """
        markers = getattr(self, "poi_markers", None)
        if markers is None or len(poi_vals) != len(markers):
            return
        while self.active_count < len(self._INTERMEDIATE_STEPS):
            _, marker_idx = self._INTERMEDIATE_STEPS[self.active_count]
            if int(poi_vals[marker_idx]) == -1:
                break
            self._on_add_step_requested(restore_position=False)

    # Duration (as a fraction of _animate_curve_reveal's own total duration)
    # of one POI marker's own entrance effect - short relative to the whole
    # reveal, so it plays as a brief "pop, then unfurl" right as the drawing
    # front passes, not something that visibly lags behind it.
    _MARKER_REVEAL_FRAC = 0.16

    def _reveal_poi_markers(
        self, x0: float, x1: float, marker_targets: List[Tuple["POIMarker", float]]
    ) -> None:
        """Grows each freshly created POI marker into place - handle first,
        then the vertical line unfurling out from it (see `POIMarker.
        setRevealProgress`) - right as `_animate_curve_reveal`'s left-to-
        right curve draw-in reaches that marker's x position, instead of
        having every marker appear at full size the instant the sweep
        starts.

        Markers stay put at their real x the whole time - only their own
        size animates - so each one's entrance is independent of every
        other's: `_animate_curve_reveal` advances its drawing front
        linearly from `x0` to `x1` over the animation's whole duration, so
        a marker at x-fraction `f` of the way across `[x0, x1]` starts
        growing once the overall reveal progress reaches `f`, and finishes
        `_MARKER_REVEAL_FRAC` of the total duration later.

        Rides the *same* `QVariantAnimation` instance that method already
        started for the main graph's fit lines (`self._curve_reveal_anim`)
        by connecting an additional slot to its existing `valueChanged`/
        `finished` signals, rather than threading marker state through that
        method's own parameters - this is called from `_add_poi_markers`,
        which `_render_analysis_plots` always runs after `_plot_signal_
        curves` (so markers land on top of the curves in z-order), by which
        point that animation is already running.

        Falls back to placing every marker directly at full size (no
        animation) if there's no reveal animation to ride - e.g. a
        degenerate run too short for `_animate_curve_reveal` to bother
        with (see its own `n < 2` guard).
        """
        anim = getattr(self, "_curve_reveal_anim", None)
        if anim is None or not marker_targets:
            for marker, _target_x in marker_targets:
                marker.setRevealProgress(1.0)
            return

        span = x1 - x0
        windows = []  # (start_frac, end_frac, marker)
        for marker, target_x in marker_targets:
            start_frac = (target_x - x0) / span if span else 1.0
            start_frac = max(0.0, min(1.0, start_frac))
            end_frac = min(1.0, start_frac + self._MARKER_REVEAL_FRAC)
            windows.append((start_frac, end_frac, marker))

        for _start_frac, _end_frac, marker in windows:
            marker.setRevealProgress(0.0)

        def _apply(fraction: float) -> None:
            for start_frac, end_frac, marker in windows:
                if fraction <= start_frac:
                    progress = 0.0
                elif fraction >= end_frac:
                    progress = 1.0
                else:
                    progress = (fraction - start_frac) / (end_frac - start_frac)
                try:
                    marker.setRevealProgress(progress)
                except RuntimeError:
                    pass

        def _finish() -> None:
            for _start_frac, _end_frac, marker in windows:
                try:
                    marker.setRevealProgress(1.0)
                except RuntimeError:
                    pass

        anim.valueChanged.connect(_apply)
        anim.finished.connect(_finish)
        _apply(0.0)

    def _save_analysis_state(
        self,
        curves: Optional[Dict[str, Any]],
        relative_time: Union[np.ndarray, List[float]],
        resonance_frequency: Union[np.ndarray, List[float]],
        dissipation: Union[np.ndarray, List[float]],
    ) -> None:
        """Persists computed signal arrays and raw sensor data to instance attributes.

        This method acts as the primary data synchronization point, moving processed
        results from local function scope into the class instance attributes. This
        allows downstream analysis steps, exports, or UI updates to access the
        most recent calculation results.

        Args:
            curves: A dictionary containing processed signal arrays and metadata.
                If None or empty, the save operation is aborted.
                Expected keys include:
                - 'xs', 'ys', 'ys_freq', 'ys_diff': Raw/Processed signal data.
                - 'ys_fit', 'ys_freq_fit', 'ys_diff_fit': Fit results.
                - 'ys_diss_2ndd': Second derivative or dissipation data.
                - 'smooth_factor': The value used for signal smoothing.
            relative_time: Array of time values relative to the experiment start.
            resonance_frequency: Array of calculated resonance frequency values.
            dissipation: Array of calculated dissipation values.
        """
        if not curves:
            return  # Nothing to save i.e., a very early failure occurred

        self.xs = curves["xs"]
        self.ys = curves["ys"]
        self.ys_freq = curves["ys_freq"]
        self.ys_diff = curves["ys_diff"]
        self.ys_fit = curves["ys_fit"]
        self.ys_freq_fit = curves["ys_freq_fit"]
        self.ys_diff_fit = curves["ys_diff_fit"]
        self.ys_diss_2ndd = curves["ys_diss_2ndd"]
        self.smooth_factor = curves["smooth_factor"]
        self.data_time = relative_time
        self.data_freq = resonance_frequency
        self.data_diss = dissipation

    def _advance_analysis_step(self, poi_vals: List[int]) -> None:
        """Determines the next UI state based on model results and POI completeness.

        This method acts as a workflow controller. It evaluates whether the
        automated model successfully identified all six POIs.
        If successful, it advances the application to the summary step
        (Step 6); otherwise, it prompts the user for manual intervention.

        Args:
            poi_vals: A list of indices representing the POI
                identified by the model or loaded from a previous session.
                A complete set must contain exactly six points to trigger
                automated advancement.
        """
        if self.model_run_this_load and self.stateStep != 6:
            # Model produced a guess and there is no prior analysis to show
            if len(poi_vals) == 6:
                Log.i("Model successfully calculated POIs for this dataset.")
                Log.d(f"Model Result = {self.model_engine}: {self.model_result}")
                self.stateStep = 6
                self._log_model_confidences()
                self.detect_change()
            else:
                Log.e("Please manually select POIs to Analyze this dataset.")
        else:
            # Model was not run this load
            if self.stateStep == 6:
                Log.i("Loaded POIs from a prior run of Analyze tool.")
            else:
                Log.e("Please manually select POIs to Analyze this dataset.")

        if self.stateStep == 6:
            Log.d("Skipping to summary step")
            self.getPoints()  # show summary when all six points are already known

        if self.show_analysis_immediately:
            Log.d("Showing analysis immediately")
            self.getPoints()  # confirm and run full analysis for prior-result view

    # def _position_floating_widget(self):
    #     pos_X = 20 + self.parent.mode_window.pos().x() + self.parent.mode_window.ui0.modemenu.width() + \
    #         (self.width() - self.QModel_widget.width()) // 2
    #     pos_Y = self.parent.mode_window.pos().y() + 250
    #     self.QModel_widget.move(pos_X, pos_Y)

    def _log_model_confidences(self):
        """Logs the confidence scores for the model's candidate points.

        Iterates through `self.model_candidates` to log the confidence percentage
        of the primary prediction for specific named points ("start", "end_fill",
        "ch1", "ch2", "ch3"). The log severity (Info, Warning, Error) scales
        dynamically based on the confidence level. The "post" point is explicitly ignored.

        Note:
            The "Tweed" model engine does not support confidence logging. If the
            engine is set to "Tweed", this method will log an informational message
            and exit early.

        Raises:
            Exception: Captures and logs any errors encountered while parsing or
                logging the confidences from the model response.
        """
        if self.model_engine == "Tweed":
            Log.i(
                tag=f"[{self.model_engine}]", msg="Confidence logging is not available for Tweed."
            )
            return

        if self.model_candidates is None:
            Log.w(
                tag=f"[{self.model_engine}]",
                msg="No model candidates available to log confidences!",
            )
            return

        def get_logger_for_confidence(confidence):
            logger = Log.e  # less than 33%
            if confidence > 66:
                logger = Log.i  # greater than 66%
            elif confidence > 33:
                logger = Log.w  # from 33% to 66%
            return logger

        try:
            point_names = ["start", "end_fill", "post", "ch1", "ch2", "ch3"]
            for i, (_, confidences) in enumerate(self.model_candidates):
                if i == 2:
                    # do not print confidence of "post" point, it doesn't matter
                    continue

                point_name = point_names[i]

                # issue: QModel is returning a single `float` instead of a `list`
                if type(confidences) is float:
                    confidences = [confidences]

                confidence = 100 * confidences[0] if len(confidences) > 0 else 0
                num_spaces = len(point_names[1]) - len(point_name) + 1

                get_logger_for_confidence(confidence)(
                    tag=f"[{self.model_engine}]",
                    msg=f"Confidence @ {point_name}:{' '*num_spaces}{confidence:2.0f}%",
                )
        except Exception:
            Log.e(
                tag=f"[{self.model_engine}]", msg="Error logging confidences from QModel response."
            )

    def resizeEvent(self, event):
        # # Position relative to main window
        # if self.QModel_widget.isVisible():
        #     QtCore.QTimer.singleShot(100, self._position_floating_widget)
        # if self.AI_SelectTool_Frame.isVisible():
        #     self.AI_SelectTool_Frame.setVisible(
        #         False
        #     )  # require re-click to show popup tool incorrect position
        pass

    def _optimize_curve(self, data_path: str, raw_bytes: Optional[bytes] = None) -> float:
        """
        Optimizes the difference factor for a given data file.

        This method reads a data file securely, processes its header, and runs a
        curve optimization algorithm to determine the optimal difference factor
        and its associated score. If an optimal factor is found, it is returned.
        Otherwise, the default difference factor is used.

        Args:
            data_path (str): Path to the data file to be optimized.
            raw_bytes (bytes, optional): Pre-read file contents. When
                provided (the `analyze_data` load path already reads the
                file once), this skips the redundant `secure_open` read
                below; when omitted (e.g. the AnalyzeWorker context), the
                file is read here as before.

        Returns:
            float: The optimal difference factor if found; otherwise, the default
            difference factor (`Constants.default_diff_factor`).

        Raises:
            Any exception during the secure file operation or optimization process
            will propagate and should be handled by the caller.

        Example:
            optimal_factor = self._optimize_curve("path/to/data/file")
        """
        try:
            optimal_factor = None
            if raw_bytes is None:
                raw_bytes = self._read_run_file_bytes(data_path)
            file_header = BytesIO(raw_bytes)
            optimizer = DifferenceFactorOptimizer(data_path, file_header)
            optimal_factor, lb, rb = optimizer.optimize()
            Log.i(
                TAG,
                f"Using difference factor {optimal_factor} optimized between {lb}s and {rb}s.",
            )

            if optimal_factor is not None:
                Log.d(TAG, f"Reporting difference factor of {optimal_factor}.")
                return optimal_factor
            else:
                Log.d(
                    TAG,
                    f"No optimal difference factor found, reporting default of {Constants.default_diff_factor}.",
                )
                return Constants.default_diff_factor
        except Exception as e:
            Log.e(
                TAG,
                f"Difference factor optimizer failed due to error. Using default factor.",
            )
            Log.e(TAG, f"Error Details: {str(e)}")
            return Constants.default_diff_factor

    def _correct_drop_effect(
        self,
        file_path: str,
        poi_vals: list,
        context: str = "process",
        raw_bytes: Optional[bytes] = None,
    ) -> tuple:
        """
        Corrects the dissipation and resonance drop effect in the provided file.

        This method reads the contents of the file specified by `file_path`,
        applies a drop effect correction algorithm using the specified
        difference factor, and returns the corrected data if successful.

        Args:
            file_path (str): Path to the file containing the data to be corrected.
            poi_vals (list): List of points-of-interest passed from QModel or user-input.
            context (str): Indicate the context of the call. Values: 'process', 'worker'.
            raw_bytes (bytes, optional): Pre-read file contents. When provided
                (the `analyze_data` load path already reads the file once),
                this skips the redundant `secure_open` read below; when
                omitted (e.g. the AnalyzeWorker "worker" context, which has
                no such buffer handy), the file is read here as before.

        Returns:
            tuple or None: The corrected data if the correction is successful;
            otherwise, returns None and logs that the original data will be used.

        Logs:
            - Debug: Indicates the start of the drop effect cancellation process with the difference factor.
            - Info: Indicates the drop effect result when successful.
            - Warning: Indicates the drop effect result when not result was returned.
            - Error: Indicates the drop effect result when an unhandled error occurred.

        Raises:
            IOError: If there is an issue opening or reading the file.
            Exception: For any unexpected errors during the correction process.
        """
        try:
            if raw_bytes is None:
                raw_bytes = self._read_run_file_bytes(file_path)
            file_buffer = BytesIO(raw_bytes)
            if hasattr(self, "diff_factor"):
                diff_factor = self.diff_factor
            else:
                diff_factor = 2.0

            Log.d(
                TAG,
                f"Performing drop effect cancelation with difference factor {diff_factor}.",
            )

            dec = DropEffectCorrection(
                file_path=file_path,
                file_buffer=file_buffer,
                initial_diff_factor=diff_factor,
                bounds=poi_vals,
            )
            corrected_data = dec.correct_drop_effects(
                save_corrections=True if context == "worker" else False
            )

            if corrected_data is not None:
                Log.i(TAG, f"Drop effect cancelation successful.")
                return corrected_data
            else:
                Log.w(TAG, f"Drop effect cancelation failed. Using original data.")
                return [None, None]
        except Exception as e:
            Log.e(
                TAG,
                f"Drop effect cancelation failed due to error. Using original data.",
            )
            Log.e(TAG, f"Error Details: {str(e)}")
            return [None, None]


class _RunLoadThread(QtCore.QThread):
    """Runs `UIAnalyze._run_analysis_pipeline` off the Qt UI thread.

    `analyze_data` still blocks its caller until the run finishes loading
    (see the "Threading" note on that method), but the actual file I/O,
    QModel inference, and signal-processing happen here instead of on the
    UI thread, so the app stays responsive - repaints, window moves, and
    other input keep working - while a run loads.

    Intentionally separate from `AnalyzeWorker` (QATCH.ui.workers.analyze_worker):
    that class drives the later "finalize analysis" step and is left untouched.
    """

    def __init__(
        self,
        ui: "UIAnalyze",
        data_path: str,
        drop_effect_enabled: bool,
        diff_factor_optimizer_enabled: bool,
        partial_fills_enabled: bool,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._ui = ui
        self._data_path = data_path
        self._drop_effect_enabled = drop_effect_enabled
        self._diff_factor_optimizer_enabled = diff_factor_optimizer_enabled
        self._partial_fills_enabled = partial_fills_enabled
        self.result: Optional[Dict[str, Any]] = None

    def run(self) -> None:
        self.result = self._ui._run_analysis_pipeline(
            self._data_path,
            self._drop_effect_enabled,
            self._diff_factor_optimizer_enabled,
            self._partial_fills_enabled,
        )


class ResizeFilter(QtCore.QObject):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self._draw_pending = False
        self._draw_delay = 250  # ms

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Resize:
            self._resize_time = monotonic()
            if not self._draw_pending:
                self._draw_pending = True
                QtCore.QTimer.singleShot(self._draw_delay, self._draw_idle)
        return super().eventFilter(obj, event)

    def _draw_idle(self):
        # convert secs -> ms: compare ms to ms
        if (monotonic() - self._resize_time) * 1000 < self._draw_delay:
            # resize event still occurring, try again later
            QtCore.QTimer.singleShot(self._draw_delay, self._draw_idle)
        else:
            # resize event finished, hysteresis elapsed: redraw!
            self.worker.place_text_avoiding_data()
            self._draw_pending = False
