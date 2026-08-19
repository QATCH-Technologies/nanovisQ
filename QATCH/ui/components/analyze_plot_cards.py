"""
QATCH.ui.components.analyze_plot_cards.py

Analyze plot-card widgets for AnalyzeUI.

Wraps AnalyzeUI's pyqtgraph plot widgets in the same rounded-card chrome
used by PlotsUI (`QATCH.ui.interfaces.ui_plots.PlotContainer`), plus a
small colored-dot "legend chip" widget used both in the Signal Overview
card's header legend and each detail plot card's title.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.interfaces.ui_plots import GridMenuRow, PlotContainer
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode
from QATCH.ui.styles.tokens import PALETTES

SIGNAL_COLORS = {
    "resonance": QtGui.QColor("#2e9e46"),
    "difference": QtGui.QColor("#2f7fd1"),
    "dissipation": QtGui.QColor("#d43f3f"),
}


class _ColorDot(QtWidgets.QWidget):
    """Display a fixed-size solid-colored circular legend swatch.

    The widget renders a single anti-aliased filled circle using the supplied
    color. It is intended for use as a compact visual indicator alongside
    signal names in legends and plot-card headers.
    """

    def __init__(
        self,
        color: QtGui.QColor,
        size: int,
        parent=None,
    ) -> None:
        """Initialize the colored-dot widget.

        Args:
            color: Initial color used to fill the dot.
            size: Width and height of the widget in pixels.
            parent: Optional parent Qt widget.
        """
        super().__init__(parent)
        self._color = QtGui.QColor(color)
        self.setFixedSize(size, size)

    def paintEvent(self, _event) -> None:
        """Paint the colored circular swatch.

        The dot is rendered with anti-aliasing and without an outline so that
        it appears as a clean, solid legend marker.

        Args:
            _event: Qt paint event supplied by the widget system.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(self._color)
        painter.drawEllipse(self.rect())
        painter.end()

    def set_color(self, color: QtGui.QColor) -> None:
        """Update the dot color and schedule the widget for repainting.

        Args:
            color: New color to use when rendering the dot.
        """
        self._color = QtGui.QColor(color)
        self.update()


class LegendChip(QtWidgets.QWidget):
    """Display a compact colored signal indicator with a text label.

    The widget combines a small circular color swatch with a vertically
    centered label, making it suitable for plot legends and plot-card
    headers.

    Args:
        text: Text displayed next to the colored dot.
        color: Initial color of the legend dot.
        dot_size: Diameter of the circular dot in pixels.
        parent: Optional parent Qt widget.
    """

    def __init__(
        self,
        text: str,
        color: QtGui.QColor,
        dot_size: int = 8,
        parent=None,
    ) -> None:
        """Initialize the legend chip.

        Args:
            text: Text displayed next to the colored dot.
            color: Initial color of the legend dot.
            dot_size: Diameter of the circular dot in pixels.
            parent: Optional parent Qt widget.
        """
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
        """Update the legend dot color.

        Args:
            color: New color to use for the legend dot.
        """
        self._color = QtGui.QColor(color)
        self._dot.set_color(self._color)


class _DualArrowControl(QtWidgets.QWidget):
    """Display two compact directional buttons as a single control.

    The control combines two small :class:`QToolButton` instances into a
    shared visual footprint. It is useful for bidirectional controls such as
    zooming or moving a point, where each direction needs to remain an
    independently clickable action while visually appearing as one compact
    control.

    The buttons can be arranged horizontally or vertically and use themed
    icons tinted to the requested color.

    Attributes:
        btn_a: First directional tool button.
        btn_b: Second directional tool button.
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
        """Initialize the dual directional control.

        Args:
            icon_a: Filename of the icon displayed on the first button.
            icon_b: Filename of the icon displayed on the second button.
            orientation: Layout orientation. Use `"vertical"` to stack the
                buttons vertically; any other value creates a horizontal
                layout.
            tooltip_a: Tooltip displayed for the first button.
            tooltip_b: Tooltip displayed for the second button.
            parent: Optional parent Qt widget.
        """
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
        """Apply a tint color to both directional button icons.

        Icons are loaded from the application's shared icon directory and
        tinted using :meth:`PlotContainer._tinted_icon`.

        Args:
            color: Color used to tint both button icons.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self.btn_a.setIcon(
            PlotContainer._tinted_icon(os.path.join(icons_dir, self._icon_a), color, 10)
        )
        self.btn_b.setIcon(
            PlotContainer._tinted_icon(os.path.join(icons_dir, self._icon_b), color, 10)
        )


class SignalOverviewCard(PlotContainer):
    """Display the AnalyzeUI signal overview plot in a themed card.

    Extends :class:`PlotContainer` with an inline signal legend and compact
    zoom and marker-navigation controls. The card wraps the existing
    AnalyzeUI `pyqtgraph` plot widget rather than creating or owning the
    underlying plot.

    The directional controls are exposed individually so the owning
    `UIAnalyze` instance can connect them to its existing plot-navigation
    callbacks.

    Attributes:
        point_to_point_toggled: Signal emitted when the gear menu's
            `"Point-to-Point Rendering"` option is toggled. The emitted
            boolean indicates whether raw-data point rendering should be
            enabled. The signal does not directly modify the plotted curves;
            `UIAnalyze` is responsible for applying the requested state.
        zoom_control: Vertical dual-arrow control containing the zoom-in and
            zoom-out buttons.
        move_control: Horizontal dual-arrow control containing the
            left and right marker-navigation buttons.
        btn_zoom_in: Button used to request zooming in.
        btn_zoom_out: Button used to request zooming out.
        btn_move_left: Button used to move the current point to the left.
        btn_move_right: Button used to move the current point to the right.
    """

    point_to_point_toggled = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        plot_widget: QtWidgets.QWidget,
        parent=None,
    ) -> None:
        """Initialize the signal overview card.

        Args:
            plot_widget: Existing AnalyzeUI plot widget to wrap in the card.
            parent: Optional parent Qt widget.
        """
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
        """Add the signal legend and navigation controls to the header.

        Creates a legend chip for each supported signal and adds compact
        controls for zooming and moving the current plot marker. The
        fullscreen and gear-menu controls created by the base
        :class:`PlotContainer` are moved to the end of the header so they
        remain right-aligned.
        """
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

        for ctrl in (getattr(self, "btn_fs", None), getattr(self, "_menu_btn", None)):
            if ctrl is not None:
                header_layout.removeWidget(ctrl)
                header_layout.addWidget(ctrl)

    def set_section_color(self, key: str, color: QtGui.QColor) -> None:
        """Update the legend color for a signal section.

        If a legend chip exists for the specified section key, its color is
        updated to keep the Signal Overview legend synchronized with the
        corresponding detail plot card.

        Args:
            key: Signal section identifier whose legend color should be
                updated.
            color: New color for the signal's legend indicator.
        """
        chip = self._legend_chips.get(key)
        if chip is not None:
            chip.set_color(color)

    def _refresh_control_icons(self, _mode: str | None = None) -> None:
        """Refresh navigation-control icon colors for the active theme.

        Determines the current application theme and applies the corresponding
        plot-text color to the zoom and marker-navigation controls.

        This method may be invoked by :class:`PlotContainer` during
        initialization before the custom controls have been created. In that
        case, it safely returns without attempting to update them.

        Args:
            _mode: Optional theme-mode value supplied by the theme-change
                callback. The current theme is queried directly from
                :class:`ThemeManager`, so this argument is intentionally
                unused.
        """
        if not hasattr(self, "zoom_control"):
            return
        dark = ThemeManager.instance().mode() == ThemeMode.DARK
        tint = QtGui.QColor(*PALETTES["dark" if dark else "light"]["plot_text_normal"][:3])
        self.zoom_control.set_icon_color(tint)
        self.move_control.set_icon_color(tint)

    def _apply_icon_theme(self, _mode: str | None = None) -> None:
        """Apply the base and custom icon theme to the overview card.

        Delegates the standard icon-theme update to :class:`PlotContainer`
        and then refreshes the colors of the Signal Overview zoom and
        marker-navigation controls.

        Args:
            _mode: Optional theme-mode value supplied by the theme-change
                callback.
        """
        super()._apply_icon_theme(_mode)
        self._refresh_control_icons(_mode)

    def _build_extra_menu_rows(self, menu: QtWidgets.QMenu) -> None:
        """Add the point-to-point rendering toggle to the gear menu.

        Adds a compact checkbox row for controlling visibility of the raw-data
        point cloud in the Signal Overview plot. The row reuses
        :class:`GridMenuRow` to maintain the same appearance and interaction
        behavior as the major and minor gridline controls.

        Point-to-point rendering is disabled by default for the overview plot
        because it can contain the full raw sample count for an entire run,
        making the point cloud more expensive to render. Detail plot cards use
        a separate implementation with the toggle enabled by default because
        they display narrower POI-specific windows.

        When the toggle changes, the emitted state is forwarded through
        :attr:`point_to_point_toggled`. The card does not directly modify the
        plotted data; the owning `UIAnalyze` instance is responsible for
        applying the setting.

        Args:
            menu: Gear-menu instance to which the toggle row should be added.
        """
        menu.addSeparator()
        row = GridMenuRow("point_to_point", "Point-to-Point Rendering", checked=False)
        row.toggled.connect(lambda _key, checked: self.point_to_point_toggled.emit(checked))

        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(row)
        menu.addAction(wa)


class DetailPlotCard(PlotContainer):
    """Display an individual AnalyzeUI signal as a themed detail plot card.

    Represents one of the three detail plot cards for Resonance, Difference,
    or Dissipation. The card wraps one of AnalyzeUI's existing
    `graphWidget1`, `graphWidget2`, or `graphWidget3` instances and
    replaces the standard title label with a :class:`LegendChip` containing
    the signal's color indicator.

    Attributes:
        point_to_point_toggled: Signal emitted when the gear menu's
            `"Point-to-Point Rendering"` option is toggled. The emitted
            boolean controls visibility of this card's raw-data point cloud
            independently of the Signal Overview card. The owning
            `UIAnalyze` instance is responsible for applying the setting
            to the plotted curves.
        _color_key: Signal identifier associated with this detail plot.
        _title_chip: Legend chip displayed in place of the standard plot
            title, or `None` before the chip has been created.
    """

    point_to_point_toggled = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        plot_widget: QtWidgets.QWidget,
        label: str,
        color_key: str,
        parent=None,
    ) -> None:
        """Initialize a detail plot card.

        Args:
            plot_widget: Existing AnalyzeUI plot widget to wrap in the card.
            label: Human-readable signal name displayed in the card header.
            color_key: Signal identifier used to select the signal color.
            parent: Optional parent Qt widget.
        """
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
        """Replace the standard header title with a colored legend chip.

        Removes the title label created by :class:`PlotContainer` and inserts
        a :class:`LegendChip` in its original position. The resulting chip
        provides both the signal name and its associated color indicator.

        Args:
            label: Text displayed by the replacement legend chip.
            color: Color used for the chip's signal indicator.
        """
        header_layout = self.header.layout()
        old_label = header_layout.itemAt(0).widget()
        header_layout.removeWidget(old_label)
        old_label.deleteLater()
        chip = LegendChip(label, color)
        self._title_chip = chip
        header_layout.insertWidget(0, chip, 1)

    def set_section_color(self, key: str, color: QtGui.QColor) -> None:
        """Update this card's title color when its signal color changes.

        Only updates the title chip when `key` matches the signal associated
        with this detail card. The method may be called on every plot card
        when a color is changed from any card's gear menu, allowing all
        corresponding visual indicators to remain synchronized.

        Args:
            key: Signal identifier whose color was changed.
            color: New color for the signal indicator.
        """
        if key == self._color_key and self._title_chip is not None:
            self._title_chip.set_color(color)

    def _build_extra_menu_rows(self, menu: QtWidgets.QMenu) -> None:
        """Add this card's point-to-point rendering toggle to the gear menu.

        Adds a :class:`GridMenuRow` for controlling visibility of this card's
        raw-data point cloud. The toggle is enabled by default because detail
        cards display only a single signal over a relatively narrow POI window,
        making the point cloud substantially less expensive to render than the
        full-run point cloud shown by the Signal Overview card.

        The row is stored as `_point_to_point_row` so that
        :meth:`set_point_to_point_available` can enable or disable the control
        based on the current AnalyzeUI workflow state.

        When the toggle changes, the resulting state is forwarded through
        :attr:`point_to_point_toggled`. The card itself does not modify the
        plotted curves; the owning `UIAnalyze` instance is responsible for
        applying the setting.

        Args:
            menu: Gear-menu instance to which the toggle row should be added.
        """
        menu.addSeparator()
        row = GridMenuRow("point_to_point", "Point-to-Point Rendering", checked=True)
        row.toggled.connect(lambda _key, checked: self.point_to_point_toggled.emit(checked))
        self._point_to_point_row = row

        wa = QtWidgets.QWidgetAction(menu)
        wa.setDefaultWidget(row)
        menu.addAction(wa)

    def set_point_to_point_available(self, available: bool) -> None:
        """Enable or disable the point-to-point rendering control.

        Disables the rendering toggle outside the Channel 1/2/3 workflow steps,
        where the associated fit line is not visible and the raw-data point cloud
        may be the only plotted content. Disabling the control prevents the user
        from hiding that content when it is needed for the current workflow.

        The checked state is preserved when the control is disabled. If the
        control is subsequently re-enabled, its previous checked state is
        restored.

        Args:
            available: `True` to enable the point-to-point rendering control;
                `False` to disable it.
        """
        self._point_to_point_row.setEnabled(available)
