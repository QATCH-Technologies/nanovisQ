"""Themed plot-card widgets for AnalyzeUI.

Wraps AnalyzeUI's pyqtgraph plot widgets in the same rounded-card chrome
used by PlotsUI (`QATCH.ui.interfaces.ui_plots.PlotContainer`), plus a
small colored-dot "legend chip" widget used both in the Signal Overview
card's header legend and each detail plot card's title.
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.interfaces.ui_plots import GridMenuRow, PlotContainer
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode
from QATCH.ui.styles.tokens import PALETTES

# The three analyzed signal series get one fixed color each, used
# consistently for: legend chips, detail-plot-card title dots, and the
# pyqtgraph curve pens/axis titles drawn in ui_analyze.py. These are
# data-semantic (not theme chrome), so they stay literal rather than
# deriving from light/dark tokens.
SIGNAL_COLORS = {
    "resonance": QtGui.QColor("#2e9e46"),
    "difference": QtGui.QColor("#2f7fd1"),
    "dissipation": QtGui.QColor("#d43f3f"),
}


class _ColorDot(QtWidgets.QWidget):
    """Fixed-size solid-filled circle used as a legend swatch."""

    def __init__(self, color: QtGui.QColor, size: int, parent=None) -> None:
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(self.rect())
        painter.end()

    def set_color(self, color: QtGui.QColor) -> None:
        self._color = QtGui.QColor(color)
        self.update()


class LegendChip(QtWidgets.QWidget):
    """A small colored dot + label, e.g. "● Resonance"."""

    def __init__(self, text: str, color: QtGui.QColor, dot_size: int = 8, parent=None) -> None:
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self._dot_size = dot_size

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self._dot = _ColorDot(self._color, dot_size)
        layout.addWidget(self._dot, 0, QtCore.Qt.AlignVCenter)

        self._label = QtWidgets.QLabel(text)
        self._label.setObjectName("PlotGlassTitle")
        layout.addWidget(self._label, 0, QtCore.Qt.AlignVCenter)

    def set_color(self, color: QtGui.QColor) -> None:
        self._color = QtGui.QColor(color)
        self._dot.set_color(self._color)


class _DualArrowControl(QtWidgets.QWidget):
    """Two small adjacent icon buttons sharing one control footprint, e.g.
    a "Zoom" control (up/down) or a "Move point" control (left/right).

    The mockup shows one bidirectional icon per control; since a single
    QToolButton can't have two independently-clickable halves, this pairs
    two small buttons (btn_a/btn_b) so each direction is still a distinct,
    unambiguous click while reading visually as one compact unit.
    """

    def __init__(
        self,
        icon_a: str,
        icon_b: str,
        orientation: str,
        tooltip_a: str,
        tooltip_b: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._icon_a = icon_a
        self._icon_b = icon_b

        vertical = orientation == "vertical"
        layout = (QtWidgets.QVBoxLayout if vertical else QtWidgets.QHBoxLayout)(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.btn_a = QtWidgets.QToolButton()
        self.btn_b = QtWidgets.QToolButton()
        for btn, tooltip in ((self.btn_a, tooltip_a), (self.btn_b, tooltip_b)):
            btn.setFixedSize(16, 14) if vertical else btn.setFixedSize(14, 16)
            btn.setIconSize(QtCore.QSize(10, 10))
            btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
            btn.setToolTip(tooltip)
            btn.setObjectName("PlotIconBtn")
            layout.addWidget(btn)

    def set_icon_color(self, color: QtGui.QColor) -> None:
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self.btn_a.setIcon(
            PlotContainer._tinted_icon(os.path.join(icons_dir, self._icon_a), color, 10)
        )
        self.btn_b.setIcon(
            PlotContainer._tinted_icon(os.path.join(icons_dir, self._icon_b), color, 10)
        )


class SignalOverviewCard(PlotContainer):
    """The large "Signal Overview" plot card: title + inline colored
    legend + Zoom/Move-point controls in the header, wrapping the same
    overview pg.PlotWidget AnalyzeUI already owns.

    Exposes btn_zoom_in/btn_zoom_out/btn_move_left/btn_move_right for the
    caller to wire to zoomFinderPlots/moveCurrentMarker (those callbacks
    live on UIAnalyze, not here).

    Attributes:
        point_to_point_toggled (QtCore.pyqtSignal): Emitted when the gear
            menu's "Point-to-Point Rendering" row is toggled. Controls the
            raw-data "point cloud" - the near-invisible scatter dots the
            solid fit lines are a smoothed average through - not the fit
            lines themselves. True shows it, False hides it. UIAnalyze owns
            applying this to the actual plotted curves.
    """

    point_to_point_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, plot_widget: QtWidgets.QWidget, parent=None) -> None:
        super().__init__(
            plot_widget,
            title="Signal Overview",
            show_menu=True,
            sections=[
                ("resonance", "Resonance", SIGNAL_COLORS["resonance"]),
                ("difference", "Difference", SIGNAL_COLORS["difference"]),
                ("dissipation", "Dissipation", SIGNAL_COLORS["dissipation"]),
            ],
            parent=parent,
        )
        self._add_legend_and_controls()
        self._refresh_control_icons()

    def _add_legend_and_controls(self) -> None:
        header_layout = self.header.layout()
        title_label = header_layout.itemAt(0).widget()
        header_layout.setStretchFactor(title_label, 0)

        self._legend_chips: dict[str, LegendChip] = {}
        header_layout.addSpacing(12)
        for key, label in (
            ("resonance", "Resonance"),
            ("difference", "Difference"),
            ("dissipation", "Dissipation"),
        ):
            chip = LegendChip(label, SIGNAL_COLORS[key])
            self._legend_chips[key] = chip
            header_layout.addWidget(chip)
            header_layout.addSpacing(10)

        header_layout.addStretch(1)

        self.zoom_control = _DualArrowControl(
            "up-arrow.svg", "down-arrow.svg", "vertical", "Zoom In", "Zoom Out"
        )
        self.move_control = _DualArrowControl(
            "left-arrow.svg", "right-arrow.svg", "horizontal", "Move Point Left", "Move Point Right"
        )
        self.btn_zoom_in = self.zoom_control.btn_a
        self.btn_zoom_out = self.zoom_control.btn_b
        self.btn_move_left = self.move_control.btn_a
        self.btn_move_right = self.move_control.btn_b

        header_layout.addWidget(self.zoom_control)
        header_layout.addSpacing(6)
        header_layout.addWidget(self.move_control)

        # PlotContainer._create_header() (show_menu=True) already placed
        # btn_fs/gear right after the title, before any of the legend/zoom/
        # move content just added above - re-append them so fullscreen+gear
        # end up rightmost, matching PlotContainer's own header order.
        for ctrl in (getattr(self, "btn_fs", None), getattr(self, "_menu_btn", None)):
            if ctrl is not None:
                header_layout.removeWidget(ctrl)
                header_layout.addWidget(ctrl)

    def set_section_color(self, key: str, color: QtGui.QColor) -> None:
        """Recolors this card's legend chip dot for `key`, if it has one.

        Called for every plot card whenever any card's gear menu changes a
        series color, so the Signal Overview legend and the matching detail
        card's title dot (see DetailPlotCard.set_section_color) stay in sync
        regardless of which card's menu was actually used.
        """
        chip = self._legend_chips.get(key)
        if chip is not None:
            chip.set_color(color)

    def _refresh_control_icons(self, _mode: str | None = None) -> None:
        if not hasattr(self, "zoom_control"):
            # PlotContainer.__init__ calls _apply_icon_theme() (which
            # dispatches here) before _add_legend_and_controls() has run.
            return
        dark = ThemeManager.instance().mode() == ThemeMode.DARK
        tint = QtGui.QColor(*PALETTES["dark" if dark else "light"]["plot_text_normal"][:3])
        self.zoom_control.set_icon_color(tint)
        self.move_control.set_icon_color(tint)

    def _apply_icon_theme(self, _mode: str | None = None) -> None:
        super()._apply_icon_theme(_mode)
        self._refresh_control_icons(_mode)

    def _build_extra_menu_rows(self, menu: QtWidgets.QMenu) -> None:
        """Adds a "Point-to-Point Rendering" toggle to this card's gear
        menu - the only card with a raw-data point cloud to show/hide in
        the first place. Reuses `GridMenuRow` (same compact checkbox-row
        widget as the Major/Minor gridline toggles above it) rather than
        introducing a new row type for a single boolean.

        Starts unchecked (point cloud hidden by default) - the overview
        graph plots a whole run's raw sample count at once, so its point
        cloud is the more expensive one to leave on by default; the detail
        cards' equivalent toggle (see `DetailPlotCard._build_extra_menu_
        rows`) starts checked instead, since each only covers one series'
        narrower POI window.
        """
        menu.addSeparator()
        row = GridMenuRow("point_to_point", "Point-to-Point Rendering", checked=False)
        row.toggled.connect(lambda _key, checked: self.point_to_point_toggled.emit(checked))

        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(row)
        menu.addAction(wa)


class DetailPlotCard(PlotContainer):
    """One of the three small detail plot cards (Resonance/Difference/
    Dissipation) - a colored-dot LegendChip header instead of a plain
    title, wrapping one of AnalyzeUI's graphWidget1/2/3.

    Attributes:
        point_to_point_toggled (QtCore.pyqtSignal): Emitted when the gear
            menu's "Point-to-Point Rendering" row is toggled. Controls this
            card's own raw-data point cloud, independent of the overview
            card's equivalent toggle. UIAnalyze owns applying this to the
            actual plotted curves.
    """

    point_to_point_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, plot_widget: QtWidgets.QWidget, label: str, color_key: str, parent=None) -> None:
        super().__init__(
            plot_widget,
            title=label,
            show_menu=True,
            sections=[(color_key, label, SIGNAL_COLORS[color_key])],
            parent=parent,
        )
        self._color_key = color_key
        self._title_chip: LegendChip | None = None
        self._replace_title_with_chip(label, SIGNAL_COLORS[color_key])

    def _replace_title_with_chip(self, label: str, color: QtGui.QColor) -> None:
        header_layout = self.header.layout()
        old_label = header_layout.itemAt(0).widget()
        header_layout.removeWidget(old_label)
        old_label.deleteLater()
        chip = LegendChip(label, color)
        self._title_chip = chip
        header_layout.insertWidget(0, chip, 1)

    def set_section_color(self, key: str, color: QtGui.QColor) -> None:
        """Recolors this card's title chip dot if `key` matches its own
        series - see SignalOverviewCard.set_section_color for why every
        card gets called regardless of which one's menu changed the color.
        """
        if key == self._color_key and self._title_chip is not None:
            self._title_chip.set_color(color)

    def _build_extra_menu_rows(self, menu: QtWidgets.QMenu) -> None:
        """Adds this card's own "Point-to-Point Rendering" toggle - see
        `SignalOverviewCard._build_extra_menu_rows`, its counterpart there.

        Starts checked (point cloud shown) - unlike the overview card's
        toggle, which starts unchecked: this card only ever plots one
        series' data within a narrow POI window, not a whole run's raw
        sample count, so there's much less reason to default it off.

        Kept as `self._point_to_point_row` so `set_point_to_point_available`
        can disable it outside the Channel 1/2/3 workflow steps - see that
        method.
        """
        menu.addSeparator()
        row = GridMenuRow("point_to_point", "Point-to-Point Rendering", checked=True)
        row.toggled.connect(lambda _key, checked: self.point_to_point_toggled.emit(checked))
        self._point_to_point_row = row

        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(row)
        menu.addAction(wa)

    def set_point_to_point_available(self, available: bool) -> None:
        """Enables/disables this card's "Point-to-Point Rendering" row -
        UIAnalyze calls this with `available=False` outside the Channel
        1/2/3 workflow steps, where this sub-graph's fit line is invisible
        and its raw-data point cloud is the only thing actually plotted
        (see `getPoints()`'s `show_fits`/`show_scat`), so hiding the point
        cloud there would leave nothing to look at. Doesn't touch the
        row's checked state - re-enabling it later restores whatever it
        was last set to.
        """
        self._point_to_point_row.setEnabled(available)
