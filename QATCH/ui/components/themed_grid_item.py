"""QATCH.ui.components.themed_grid_item

A `pg.GridItem` subclass that draws grid lines aligned to a plot's real
axis tick positions, with a fixed caller-specified alpha per level. Shared
by PlotsUI (`QATCH.ui.main_window`) and AnalyzeUI
(`QATCH.ui.interfaces.ui_analyze`) for their gear-menu grid toggles.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as pg_fn
from PyQt5 import QtCore, QtGui
from pyqtgraph import AxisItem


class ThemedGridItem(pg.GridItem):
    """A `pg.GridItem` that draws at the same tick positions as a pair of
    real AxisItems, with a fixed, caller-specified line alpha per level.

    Two problems with the stock `pg.GridItem` this replaces:

    1. It computes grid-line spacing independently of the axes it sits
       next to (its own `10**i`-based "nice round number" heuristic), so
       lines routinely land at different positions than the axis's own
       tick labels - lines don't reach/align with every labeled tick, and
       depending on the view range the computed spacing can undershoot the
       visible span, leaving the grid not fully covering the plot area.
    2. `generatePicture()` computes its own alpha for each level from
       on-screen line density and overwrites the pen's color in place -
       `self.opts['pen']` is mutated directly, not copied - which silently
       discards whatever alpha was configured on the constructor's pen.
       Confirmed empirically: a pen built with alpha=155 read back as
       alpha=50 after a single repaint, and since every level shares that
       one mutated pen object, major/minor never reliably read as visually
       distinct either.

    Fix: instead of computing tick spacing itself, this item asks the
    actual `x_axis`/`y_axis` AxisItems for their current `tickValues()` -
    the exact values they use to place their own labels - and draws lines
    at those. "Major" (`include_minor_ticks=False`) draws only at the
    axis's top-level (labeled) ticks; "Minor" (`include_minor_ticks=True`)
    draws at the labeled ticks *plus* the axis's next-finer tick level, so
    minor lines are guaranteed to include every labeled position rather
    than an independently-computed set that may not line up with them.
    """

    def __init__(
        self,
        pen,
        alpha: int,
        x_axis: AxisItem | None = None,
        y_axis: AxisItem | None = None,
        include_minor_ticks: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(pen=pen, **kwargs)
        self._fixed_alpha = alpha
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._include_minor_ticks = include_minor_ticks

    def _axis_tick_positions(self, axis: AxisItem | None, size: float) -> list:
        """Returns the tick values `axis` is currently placing labels at
        (plus its next-finer level too, if `_include_minor_ticks`), plus
        the axis's own range boundaries.

        Tick values are quantized to "nice round numbers" within the
        range, which are very unlikely to land exactly on the range's own
        start/end (e.g. a live time axis whose left edge is whatever
        timestamp the oldest visible sample happens to have, not a round
        number) - leaving a visible gap between the true edge of the plot
        and the nearest round-number gridline. Explicitly including the
        boundary values guarantees the grid always reaches the plot's
        edges exactly, regardless of where the round-number ticks fall.
        """
        if axis is None or size <= 0:
            return []
        try:
            rng = axis.range
            levels = axis.tickValues(rng[0], rng[1], size)
        except Exception:
            return []
        if not levels:
            return []
        n_levels = 2 if self._include_minor_ticks else 1
        values: set = {rng[0], rng[1]}
        for _, vals in levels[:n_levels]:
            values.update(vals)
        return sorted(values)

    def generatePicture(self) -> None:
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter()
        p.begin(self.picture)

        lvr = self.boundingRect()
        ul = np.array([lvr.left(), lvr.top()])
        br = np.array([lvr.right(), lvr.bottom()])
        if ul[1] > br[1]:
            ul[1], br[1] = br[1], ul[1]
        x_lo, x_hi = min(ul[0], br[0]), max(ul[0], br[0])
        y_lo, y_hi = min(ul[1], br[1]), max(ul[1], br[1])

        base_color = QtGui.QColor(self.opts["pen"].color())
        base_color.setAlpha(self._fixed_alpha)
        line_pen = QtGui.QPen(self.opts["pen"])
        line_pen.setColor(base_color)
        line_pen.setCosmetic(True)
        p.setPen(line_pen)

        x_geom = self._x_axis.geometry() if self._x_axis is not None else None
        x_size = x_geom.width() if x_geom is not None else 0.0
        for xv in self._axis_tick_positions(self._x_axis, x_size):
            if xv < x_lo or xv > x_hi:
                continue
            p.drawLine(QtCore.QPointF(xv, ul[1]), QtCore.QPointF(xv, br[1]))

        y_geom = self._y_axis.geometry() if self._y_axis is not None else None
        y_size = y_geom.height() if y_geom is not None else 0.0
        for yv in self._axis_tick_positions(self._y_axis, y_size):
            if yv < y_lo or yv > y_hi:
                continue
            p.drawLine(QtCore.QPointF(ul[0], yv), QtCore.QPointF(br[0], yv))

        tr = self.deviceTransform()
        p.setWorldTransform(pg_fn.invertQTransform(tr))
        p.end()
