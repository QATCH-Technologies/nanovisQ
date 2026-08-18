"""
QATCH.ui.components.plot_grid_item.py

A :class:`pyqtgraph.GridItem` subclass that draws grid lines aligned to a plot's real
axis tick positions, with a fixed caller-specified alpha per level. Shared
by PlotsUI (`QATCH.ui.main_window`) and AnalyzeUI
(`QATCH.ui.interfaces.ui_analyze`) for their gear-menu grid toggles.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as pg_fn
from PyQt5 import QtCore, QtGui
from pyqtgraph import AxisItem

from QATCH.common.logger import Logger as Log

TAG = "[PlotGridItem]"


class PlotGridItem(pg.GridItem):
    """A :class:`pyqtgraph.GridItem` that draws grid lines at real axis tick positions.

    This subclass overrides stock :class:`pyqtgraph.GridItem` behavior to align grid lines
    directly with the tick positions of provided :class:`~pyqtgraph.AxisItem` instances,
    using a fixed, caller-specified line alpha per level.

    Args:
        pen (QtGui.QPen | QColor | str): Pen or color used to configure grid line aesthetics.
        alpha (int): Fixed opacity value (0-255) applied to all grid lines.
        x_axis (AxisItem | None, optional): The horizontal axis used to retrieve X tick positions.
            Defaults to `None`.
        y_axis (AxisItem | None, optional): The vertical axis used to retrieve Y tick positions.
            Defaults to `None`.
        include_minor_ticks (bool, optional): If `True`, renders minor tick grid lines in addition
            to major tick lines. Defaults to `False`.
        **kwargs: Additional keyword arguments passed to the parent :class:`pyqtgraph.GridItem`.
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
        """Calculate sorted tick positions including range boundaries for a given axis.

        Retrieves the current tick values calculated by :meth:`~pyqtgraph.AxisItem.tickValues`
        for the given `axis`. Explicitly appends the axis range boundaries to ensure grid lines
        reach the outer edges of the plot area regardless of tick quantization.

        Args:
            axis (AxisItem | None): Target axis to fetch tick values from.
            size (float): The current width or height dimension of the axis geometry in pixels.

        Returns:
            list[float]: A sorted list of unique floating-point coordinate positions for
            rendering grid lines. Returns an empty list if `axis` is `None` or `size <= 0`.
        """
        super().__init__(pen=pen, **kwargs)
        self._fixed_alpha = alpha
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._include_minor_ticks = include_minor_ticks

    def _axis_tick_positions(self, axis: AxisItem | None, size: float) -> list:
        """Calculate sorted tick positions including range boundaries for a given axis.

        Retrieves the current tick values calculated by :meth:`~pyqtgraph.AxisItem.tickValues`
        for the given `axis`. Explicitly appends the axis range boundaries to ensure grid lines
        reach the outer edges of the plot area regardless of tick quantization.

        Args:
            axis (AxisItem | None): Target axis to fetch tick values from.
            size (float): The current width or height dimension of the axis geometry in pixels.

        Returns:
            list[float]: A sorted list of unique floating-point coordinate positions for
            rendering grid lines. Returns an empty list if `axis` is `None` or `size <= 0`.
        """
        if axis is None or size <= 0:
            return []
        try:
            rng = axis.range
            levels = axis.tickValues(rng[0], rng[1], size)
        except Exception as e:  # noqa: BLE001
            Log.w(TAG, f"Error occurred while fetching tick values for axis: {e}")
            return []
        if not levels:
            return []
        n_levels = 2 if self._include_minor_ticks else 1
        values: set = {rng[0], rng[1]}
        for _, vals in levels[:n_levels]:
            values.update(vals)
        return sorted(values)

    def generatePicture(self) -> None:
        """Generate and cache the :class:`QtGui.QPicture` containing drawn grid lines.

        Overrides :meth:`pyqtgraph.GridItem.generatePicture`. Draws vertical and horizontal grid
        lines aligned with tick positions obtained from :meth:`_axis_tick_positions`. Enforces
        `_fixed_alpha` onto a non-mutating copy of the active pen.
        """
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
