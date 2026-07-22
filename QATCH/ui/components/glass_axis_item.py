"""QATCH.ui.components.glass_axis_item

The "glass" pyqtgraph axis look (no visible spine, no tick marks, muted
theme-derived text) shared by PlotsUI (`QATCH.ui.main_window`) and AnalyzeUI
(`QATCH.ui.interfaces.ui_analyze`), so both windows' plots read as the same
family.
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtGui
from pyqtgraph import AxisItem


class GlassAxisItem(AxisItem):
    """An `AxisItem` with no visible spine or tick marks, a small fixed
    tick font, and scientific-notation suppressed for small values (e.g.
    Dissipation numbers like 0.000193, which pyqtgraph would otherwise
    render as "1.93e-4")."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        f = QtGui.QFont("Segoe UI")
        f.setPixelSize(10)
        self.setTickFont(f)
        self.setStyle(
            tickLength=0,
            tickTextOffset=3,
            autoExpandTextSpace=False,
        )
        self.setPen(pg.mkPen(None))
        # Disable pyqtgraph's SI prefix mechanism; scale factors (if any)
        # are managed directly by the caller instead.
        self.enableAutoSIPrefix(False)

    def paint(self, p, opt, widget) -> None:
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing)
        super().paint(p, opt, widget)

    def tickStrings(self, values, scale, spacing):
        strs = super().tickStrings(values, scale, spacing)
        clean_strs = []
        for s, v in zip(strs, values):
            val = v * scale
            abs_v = abs(val)
            # Suppress scientific notation for small float values like Dissipation (e.g., 0.000193)
            if 0 < abs_v < 0.01 or "e" in s.lower():
                clean_strs.append(f"{val:.6f}")
            else:
                clean_strs.append(s)
        return clean_strs


def suppress_axis_ticks(axis: AxisItem | None) -> None:
    """Monkey-patches `axis` so it never emits tick-mark line specs.

    `tickLength=0` alone doesn't reliably prevent a stray mark at each tick
    position: confirmed empirically by inspecting pyqtgraph's own
    `generateDrawSpecs()` output directly - every tick spec really is
    zero-length with a fully `NoPen` pen (style 0), which by itself should
    paint nothing, yet a 1px artifact still showed up at every *major* tick
    position in an actual rendered frame (most visible in dark mode, since
    the artifact's fixed light-gray color stands out against a dark
    background - it isn't theme-derived at all). Rather than depend on
    pyqtgraph's tick-line geometry/pen happening to be invisible, this
    removes the tick specs before `drawPicture()` ever sees them, so
    nothing can be drawn regardless of the exact underlying rendering
    quirk. Neither PlotsUI nor AnalyzeUI ever wants visible tick marks by
    design (only text labels, and optionally a separate `ThemedGridItem`
    grid overlay), so there's no loss.

    Safe to call on any `AxisItem` (a `GlassAxisItem`, `GlassDateAxis`, or a
    plain auto-created one, e.g. a linked "right" axis) and safe to call
    more than once on the same instance.

    Args:
        axis: The `pg.AxisItem` (or subclass) instance to patch.
    """
    if axis is None or getattr(axis, "_ticks_suppressed", False):
        return
    original = axis.generateDrawSpecs

    def _no_ticks(p, _original=original):
        specs = _original(p)
        if specs is None:
            return specs
        axis_spec, _tick_specs, text_specs = specs
        return axis_spec, [], text_specs

    axis.generateDrawSpecs = _no_ticks
    axis._ticks_suppressed = True
    axis.picture = None
    axis.update()


def apply_glass_plot_style(plot_item, text_pen) -> None:
    """Apply minimal glass axis styling - no spines, no ticks, floating labels.

    Args:
        plot_item: The `pg.PlotItem` (or `PlotWidget`'s `.getPlotItem()`) to style.
        text_pen: Pen (or color) used for axis tick/label text - typically the
            theme's muted text color, so it matches whichever window called this.
    """
    for name in ("bottom", "left", "right", "top"):
        ax = plot_item.getAxis(name)
        if ax is not None:
            ax.setPen(pg.mkPen(None))  # remove spine on every axis
            ax.setTextPen(text_pen)
            ax.setGrid(False)  # grids start off; controlled by settings menu
            suppress_axis_ticks(ax)

    plot_item.getViewBox().setBorder(pg.mkPen(None))
    plot_item.getViewBox().setDefaultPadding(0.02)
    plot_item.hideButtons()
    plot_item.showGrid(x=False, y=False)  # grids start off; controlled by settings menu
