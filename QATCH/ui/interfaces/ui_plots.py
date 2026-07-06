"""QATCH.ui.interfaces.ui_plots.py

This module builds the plotting window's UI plot containers,
their gear menus (per-section color/visibility rows and grid-line toggles),
a tabbed container for switching between multiple device plots, and the
`UIPlots` class that assembles the Dissipation/Resonance, Amplitude, and
Temperature panes into the main plots window layout.

Author(s)
    Alexander Ross  (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-05-11
"""

from __future__ import annotations

import math
import os
import time
from functools import partial
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import GraphicsLayoutWidget

from QATCH.common.architecture import Architecture
from QATCH.common.fwUpdater import FW_UPDATE
from QATCH.core.constants import Constants
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.widgets.update_status_badge import UpdateStatusIcon
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode, tok_css
from QATCH.ui.styles.tokens import PALETTES


class PlotMenuRow(QtWidgets.QWidget):
    """A compact interactive row widget for plot options.

    This widget represents a single row in a plot's option menu, providing controls
    to change the plot's color, view its label, and toggle its visibility. It is
    typically used as a payload inside a QWidgetAction.

    Attributes:
        color_changed (QtCore.pyqtSignal): Signal emitted when the color is changed.
            Provides the plot key (str) and the new color (QtGui.QColor).
        visibility_changed (QtCore.pyqtSignal): Signal emitted when visibility is toggled.
            Provides the plot key (str) and the new visibility state (bool).
    """

    color_changed = QtCore.pyqtSignal(str, QtGui.QColor)
    visibility_changed = QtCore.pyqtSignal(str, bool)

    def __init__(
        self,
        key: str,
        label: str,
        color: QtGui.QColor,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initializes the PlotMenuRow.

        Args:
            key (str): The unique identifier for the plot associated with this row.
            label (str): The display name of the plot.
            color (QColor): The initial color of the plot.
            parent (QWidget): The parent widget, if any.
        """
        super().__init__(parent)
        self._key = key
        self._color = QtGui.QColor(color)
        self._visible = True
        self._icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        self._setup_ui(label)
        self._connect_signals()

    def _setup_ui(self, label: str) -> None:
        """Configures the widget's layout and internal components.

        Args:
            label: The text to display for the plot's name.
        """
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("PlotMenuRow")
        self.setMinimumWidth(190)
        self.setFixedHeight(34)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        # Color swatch
        self._swatch = QtWidgets.QToolButton()
        self._swatch.setFixedSize(20, 20)
        self._swatch.setIconSize(QtCore.QSize(16, 16))
        self._swatch.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._swatch.setToolTip("Change color")
        self._apply_swatch_style()

        # Label
        self._lbl = QtWidgets.QLabel(label)
        self._lbl.setObjectName("PlotMenuItemLabel")

        # Show / Hide toggle
        self._eye = QtWidgets.QToolButton()
        self._eye.setFixedSize(22, 22)
        self._eye.setIconSize(QtCore.QSize(16, 16))
        self._eye.setCheckable(True)
        self._eye.setChecked(True)
        self._eye.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self._eye.setToolTip("Show / Hide")
        self._apply_eye_style()

        layout.addWidget(self._swatch)
        layout.addWidget(self._lbl, 1)
        layout.addWidget(self._eye)

    def _connect_signals(self) -> None:
        """Connects UI and theme manager signals to their respective slots."""
        self._swatch.clicked.connect(self._pick_color)
        self._eye.clicked.connect(self._toggle_visibility)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _apply_swatch_style(self) -> None:
        """Applies dynamic styling and tinted icons to the color swatch button."""
        path = os.path.join(self._icons_dir, "color-mode.svg")
        self._swatch.setIcon(PlotContainer._tinted_icon(path, self._color, size=16))

        tok = ThemeManager.instance().tokens()
        br = tok["plot_swatch_border"]
        border = tok_css(br)
        border_full = tok_css((*br[:3], 255))

        self._swatch.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1.5px solid {border};
                border-radius: 8px;
            }}
            QToolButton:hover {{
                border: 2px solid {border_full};
            }}
        """)

    def _apply_eye_style(self) -> None:
        """Applies dynamic styling and tinted icons to the visibility toggle button."""
        icon_name = "eye-on.svg" if self._visible else "eye-off.svg"
        path = os.path.join(self._icons_dir, icon_name)

        tok = ThemeManager.instance().tokens()
        r, g, b, _ = tok["plot_text_normal"]
        tint = QtGui.QColor(r, g, b, 200 if self._visible else 80)
        self._eye.setIcon(PlotContainer._tinted_icon(path, tint, size=16))

        hover_bg = tok_css(tok["plot_tab_bg_hover"])
        self._eye.setStyleSheet(f"""
            QToolButton {{
                background: transparent; border: none;
                border-radius: 4px;
            }}
            QToolButton:hover {{ background: {hover_bg}; }}
        """)

    def _on_theme_changed(self, mode: str) -> None:
        """Updates widget styling when the application theme changes.

        Args:
            mode: The new theme mode (e.g., 'dark' or 'light').
        """
        self._apply_swatch_style()
        self._apply_eye_style()

    def _pick_color(self) -> None:
        """Opens a color dialog to select a new plot color.

        Temporarily closes the parent menu to ensure the color dialog is
        displayed prominently without overlay issues. Emits the `color_changed`
        signal if a valid color is chosen.
        """
        menu = self._find_parent_menu()
        if menu:
            menu.close()

        color = QtWidgets.QColorDialog.getColor(self._color, None, "Choose Color")
        if color.isValid():
            self._color = color
            self._apply_swatch_style()
            self.color_changed.emit(self._key, self._color)

    def _toggle_visibility(self, checked: bool) -> None:
        """Toggles the plot visibility state.

        Args:
            checked: The new toggled state of the visibility button.
        """
        self._visible = checked
        self._apply_eye_style()
        self.visibility_changed.emit(self._key, checked)

    def _find_parent_menu(self) -> QtWidgets.QMenu | None:
        """Traverses the widget hierarchy to find the parent QMenu.

        Returns:
            The parent QMenu instance if found, otherwise None.
        """
        parent = self.parentWidget()
        while parent:
            if isinstance(parent, QtWidgets.QMenu):
                return parent
            parent = parent.parentWidget()
        return None


class GridMenuRow(QtWidgets.QWidget):
    """A labeled checkbox widget for toggling a grid-line axis.

    Typically used inside a plot's gear menu to allow the user to dynamically
    show or hide specific grid lines (e.g., X-axis or Y-axis grids).

    Attributes:
        toggled (QtCore.pyqtSignal): Signal emitted when the checkbox state changes.
            Provides the axis key (str) and the new visibility state (bool).
    """

    toggled = QtCore.pyqtSignal(str, bool)

    def __init__(
        self,
        key: str,
        label: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initializes the GridMenuRow.

        Args:
            key (str): The unique identifier for the grid axis (e.g., "x", "y").
            label (label): The display text for the checkbox.
            parent (QWidget): The parent widget, if any.
        """
        super().__init__(parent)
        self._key = key

        self._setup_ui(label)
        self._connect_signals()
        self._apply_style()

    def _setup_ui(self, label: str) -> None:
        """Configures the widget's layout and internal components.

        Args:
            label: The display text for the checkbox.
        """
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("PlotMenuRow")
        self.setMinimumWidth(190)
        self.setFixedHeight(34)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(8)

        self._checkbox = QtWidgets.QCheckBox(label)
        self._checkbox.setObjectName("PlotMenuItemLabel")
        self._checkbox.setChecked(False)  # Off by default
        self._checkbox.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        layout.addWidget(self._checkbox, 1)

    def _connect_signals(self) -> None:
        """Connects UI and theme manager signals to their respective slots."""
        self._checkbox.toggled.connect(self._on_checkbox_toggled)
        ThemeManager.instance().themeChanged.connect(self._apply_style)

    def _on_checkbox_toggled(self, checked: bool) -> None:
        """Handles internal checkbox toggles and emits the public signal.

        Args:
            checked: The new checked state of the checkbox.
        """
        self.toggled.emit(self._key, checked)

    def _apply_style(self, _mode: str | None = None) -> None:
        """Applies dynamic CSS styling based on the current theme tokens.

        Args:
            _mode: Optional theme mode string passed by the themeChanged signal.
        """
        tok = ThemeManager.instance().tokens()

        # External dependencies (assumed to be available in module scope)
        text_css = tok_css(tok["plot_text_normal"])
        border_css = tok_css(tok["plot_swatch_border"])
        checked_css = tok_css(tok["plot_data_primary"])

        self._checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {text_css};
                spacing: 6px;
            }}
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
                border-radius: 3px;
                border: 1.5px solid {border_css};
                background: transparent;
            }}
            QCheckBox::indicator:checked {{
                background: {checked_css};
                border-color: {checked_css};
            }}
        """)


class PlotContainer(QtWidgets.QWidget):
    """A standard card wrapping a pyqtgraph GraphicsLayoutWidget.

    This container provides a stylized background, an optional title header,
    fullscreen toggling, and a dynamic settings menu for plot visualization options.

    Attributes:
        color_requested (QtCore.pyqtSignal): Legacy signal emitted when color change is requested.
        fullscreen_requested (QtCore.pyqtSignal): Emitted when the fullscreen toggle is clicked.
        section_color_changed (QtCore.pyqtSignal): Emitted when a specific plot section changes
            color. Provides the section key (str) and new color (QtGui.QColor).
        section_visibility_changed (QtCore.pyqtSignal): Emitted when a plot section's visibility is
            toggled. Provides the section key (str) and visibility state (bool).
        grid_changed (QtCore.pyqtSignal): Emitted when a grid line type is toggled.
            Provides the grid key (str) and visibility state (bool).
    """

    # Legacy signals
    color_requested = QtCore.pyqtSignal()
    fullscreen_requested = QtCore.pyqtSignal()

    # Per-section signals
    section_color_changed = QtCore.pyqtSignal(str, QtGui.QColor)
    section_visibility_changed = QtCore.pyqtSignal(str, bool)
    grid_changed = QtCore.pyqtSignal(str, bool)

    # UI Constants
    _R = 12.0
    _M = 3
    _HEADER_H = 28

    def __init__(
        self,
        plot_widget: QtWidgets.QWidget,
        title: str | None = None,
        accent_color: QtGui.QColor | None = None,
        parent: QtWidgets.QWidget | None = None,
        show_menu: bool = True,
        sections: list[tuple[str, str, QtGui.QColor]] | None = None,
    ) -> None:
        """Initializes the PlotContainer.

        Args:
            plot_widget (QWidget): The main plotting widget to wrap.
            title (str): An optional title string to display in the header.
            accent_color (QColor): An optional accent color (currently reserved for future styling).
            parent (QWidget): The parent widget, if any.
            show_menu (bool): Whether to display the gear menu and fullscreen controls.
            sections (list): A list of plot data configurations structured as
                (key, label, default_color).
        """
        super().__init__(parent)
        self._sections = sections or []
        self._is_fullscreen = False
        self.has_header = title is not None or show_menu

        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self._preload_icons()
        self._setup_ui(plot_widget, title, show_menu)
        self._connect_signals()
        self._apply_icon_theme()

    def _setup_ui(self, plot_widget: QtWidgets.QWidget, title: str | None, show_menu: bool) -> None:
        """Configures the main layout and embeds the header and plot widget.

        Sets up a vertical box layout with minimal margins, conditionally builds
        and adds a custom header (containing the title and tool buttons), and
        embeds the primary plotting widget below it.

        Args:
            plot_widget (QtWidgets.QWidget): The primary visualization widget
                (typically a pyqtgraph.GraphicsLayoutWidget) to be displayed.
            title (str | None): The title text to display in the header. If None,
                no title is rendered.
            show_menu (bool): Whether to include the interactive gear menu and
                fullscreen toggle button in the header.
        """
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(self._M, self._M, self._M, self._M)
        layout.setSpacing(0)

        if self.has_header:
            self.header = self._create_header(title, show_menu)
            layout.addWidget(self.header)

        layout.addWidget(plot_widget, 1)

    def _create_header(self, title: str | None, show_menu: bool) -> QtWidgets.QWidget:
        """Builds the top header widget for the plot container.

        Constructs a horizontally laid-out widget that displays an optional title
        label on the left. If the menu is enabled, it aligns interactive tool
        buttons (fullscreen toggle and a gear options menu) to the right.

        Args:
            title (str | None): The text to display as the plot's title. If None,
                the title area is left blank, and layout spacing is maintained
                using a layout stretch.
            show_menu (bool): Whether to instantiate and display the interactive
                control buttons (fullscreen and plot options) on the right side.

        Returns:
            QtWidgets.QWidget: The fully constructed and configured header widget,
            ready to be embedded into the main container layout.
        """
        header = QtWidgets.QWidget()
        header.setFixedHeight(self._HEADER_H)

        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(8, 0, 4, 0)
        h_layout.setSpacing(4)

        if title:
            lbl = QtWidgets.QLabel(title)
            lbl.setObjectName("PlotGlassTitle")
            h_layout.addWidget(lbl, 1)
        else:
            h_layout.addStretch(1)

        if show_menu:
            # Fullscreen button
            self.btn_fs = self._make_icon_button("fullscreen.svg", "Toggle Fullscreen")
            self.btn_fs.clicked.connect(self.fullscreen_requested.emit)
            h_layout.addWidget(self.btn_fs)

            # Gear / options button
            h_layout.addWidget(self._build_menu())

        return header

    def _connect_signals(self) -> None:
        """Connects global theme signals to local update slots.

        Ensures that whenever the application-wide theme changes, the widget
        triggers a repaint (via `self.update`) and refreshes its icons to
        match the new theme context.
        """
        theme_manager = ThemeManager.instance()
        theme_manager.themeChanged.connect(self.update)
        theme_manager.themeChanged.connect(self._apply_icon_theme)

    @staticmethod
    def _tinted_icon(path: str, color: QtGui.QColor, size: int = 13) -> QtGui.QIcon:
        """Creates a color-tinted QIcon from an SVG path.

        Uses a composition mode to apply a solid color overlay to the alpha
        channel of the source SVG, effectively tinting the icon while
        preserving its original shape and transparency.

        Args:
            path (str): Absolute file path to the SVG asset.
            color (QtGui.QColor): The target color to tint the icon with.
            size (int): The base resolution (width and height in pixels) for
                the pixmap mapping. Defaults to 13.

        Returns:
            QtGui.QIcon: A new icon with the specified color tint applied.
        """
        src = QtGui.QIcon(path).pixmap(size, size)
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(dst)
        painter.drawPixmap(0, 0, src)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        painter.fillRect(dst.rect(), color)
        painter.end()

        return QtGui.QIcon(dst)

    def _preload_icons(self) -> None:
        """Preloads and caches standard and tinted icons to optimize rendering.

        Loads SVG assets from the disk once during initialization. It generates both
        the standard icon and a "lit" (tinted) version for dark mode, attaching them
        dynamically as attributes to the instance (e.g., `self._icon_gear` and
        `self._icon_gear_lit`). This prevents expensive disk I/O and image processing
        during rapid UI updates or theme switches.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        tint = QtGui.QColor(*PALETTES["dark"]["plot_text_normal"][:3])

        for stem in ("fullscreen", "fullscreen-exit", "gear"):
            path = os.path.join(icons_dir, f"{stem}.svg")
            attr = "_icon_" + stem.replace("-", "_")
            setattr(self, attr, QtGui.QIcon(path))
            setattr(self, f"{attr}_lit", self._tinted_icon(path, tint))

    def _apply_icon_theme(self, _mode: str | None = None) -> None:
        """Updates the tool button icons based on the current theme and state.

        Checks the application's active theme mode and applies the corresponding
        cached icon (standard for light mode, "lit" for dark mode) to the gear
        menu and fullscreen toggle buttons. It also factors in the current
        fullscreen state to choose the correct icon stem.

        Args:
            _mode (str | None): Optional theme mode string (e.g., 'dark' or 'light')
                typically provided when triggered by the `themeChanged` signal.
                Defaults to None.
        """
        dark = ThemeManager.instance().mode() == ThemeMode.DARK

        if hasattr(self, "_menu_btn"):
            self._menu_btn.setIcon(self._icon_gear_lit if dark else self._icon_gear)

        if hasattr(self, "btn_fs"):
            if self._is_fullscreen:
                icon = self._icon_fullscreen_exit_lit if dark else self._icon_fullscreen_exit
            else:
                icon = self._icon_fullscreen_lit if dark else self._icon_fullscreen

            self.btn_fs.setIcon(icon)

    def _make_icon_button(self, icon_name: str, tooltip: str) -> QtWidgets.QToolButton:
        """Creates a styled, fixed-size circular tool button.

        Args:
            icon_name: Filename of the icon.
            tooltip: Text to display on hover.

        Returns:
            A configured QToolButton instance.
        """
        btn = QtWidgets.QToolButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)
        btn.setIcon(QtGui.QIcon(icon_path))
        btn.setIconSize(QtCore.QSize(13, 13))
        btn.setFixedSize(24, 24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.setToolTip(tooltip)
        btn.setObjectName("PlotIconBtn")
        return btn

    def _build_menu(self) -> QtWidgets.QToolButton:
        """Builds the gear icon button and attaches the frosted glass options menu.

        Instantiates a customized tool button using the gear icon, configures it
        for instant popup behavior, and attaches the specialized translucent QMenu
        containing the plot's interactive display options.

        Returns:
            QtWidgets.QToolButton: The configured gear button with the attached
            popup menu, ready to be added to the header layout.
        """
        self._menu_btn = self._make_icon_button("gear.svg", "Plot Options")
        self._menu_btn.setObjectName("PlotMenuBtn")
        self._menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)

        self.plot_menu = self._style_menu(self._menu_btn)
        self._menu_btn.setMenu(self.plot_menu)
        return self._menu_btn

    def _style_menu(self, parent_widget: QtWidgets.QWidget) -> QtWidgets.QMenu:
        """Constructs the frameless translucent QMenu and populates its actions.

        Builds a custom QMenu configured with a frameless, transparent background
        to support custom styling (like a frosted glass effect). It populates the
        menu with `PlotMenuRow` widgets for dynamic data sections (allowing color
        and visibility toggles) and appends standard `GridMenuRow` toggles for
        plot gridlines.

        Args:
            parent_widget (QtWidgets.QWidget): The widget (typically the gear tool
                button) to anchor the dropdown menu to.

        Returns:
            QtWidgets.QMenu: The fully populated and styled menu instance containing
            all interactive plot options.
        """
        menu = QtWidgets.QMenu(parent_widget)
        menu.setObjectName("PlotGlassMenu")
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        menu.setWindowFlags(
            menu.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )

        # Plot data sections
        if self._sections:
            for i, (key, label, color) in enumerate(self._sections):
                row = PlotMenuRow(key, label, color)
                row.color_changed.connect(self.section_color_changed)
                row.visibility_changed.connect(self.section_visibility_changed)

                wa = QtWidgets.QWidgetAction(menu)
                wa.setDefaultWidget(row)
                menu.addAction(wa)

                if i < len(self._sections) - 1:
                    menu.addSeparator()
        else:
            action_color = menu.addAction("Change Line Colors…")
            assert action_color is not None
            action_color.triggered.connect(self.color_requested.emit)

        # Grid line toggles (always present)
        menu.addSeparator()
        for grid_key, grid_label in (
            ("grid_major", "Major Gridlines"),
            ("grid_minor", "Minor Gridlines"),
        ):
            grid_row = GridMenuRow(grid_key, grid_label)
            grid_row.toggled.connect(self.grid_changed)

            gwa = QtWidgets.QWidgetAction(menu)
            gwa.setDefaultWidget(grid_row)
            menu.addAction(gwa)

        return menu

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat card surface (fill + border) behind the plot.

        Uses the same `surface`/`surface_border` tokens as the Mode sidebar
        and Log console (see QATCH.ui.components.flat_paint) so plot cards
        read as part of the same panel family rather than the old frosted
        glass look.

        Args:
            ev (QtGui.QPaintEvent): The paint event parameters provided by the
                Qt framework.
        """
        tok = ThemeManager.instance().tokens()
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._R,
            fill=QtGui.QColor(*tok["surface"]),
            border=QtGui.QColor(*tok["surface_border"]),
            painter=painter,
        )

        # Header separator line
        if self.has_header:
            painter.setPen(QtGui.QPen(QtGui.QColor(*tok["surface_border"]), 1.0))
            y_line = self._HEADER_H + self._M
            painter.drawLine(0, y_line, self.width(), y_line)

        painter.end()


class PlotTabContainer(PlotContainer):
    """Specialized frosted glass container managing multiple device plots via tabs.

    Extends `PlotContainer` to support a `QStackedWidget` for toggling between
    multiple device plots. It integrates a custom header featuring exclusive tab
    buttons with animated activity indicators (dots) and a global firmware
    update status icon.

    Attributes:
        device_color_requested (QtCore.pyqtSignal): Legacy signal emitted to
            request a color change for a device, providing the active device index (int).
    """

    device_color_requested = QtCore.pyqtSignal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initializes the PlotTabContainer.

        Args:
            parent (QWidget | None): The parent widget, if any.
        """
        dummy = QtWidgets.QWidget()
        tok = ThemeManager.instance().tokens()

        main_sections = [
            ("dissipation", "Dissipation", QtGui.QColor(*tok["plot_data_temperature"][:3])),
            ("resonance_freq", "Resonance Frequency", QtGui.QColor(*tok["plot_data_primary"][:3])),
        ]
        super().__init__(
            plot_widget=dummy,
            title=None,
            parent=parent,
            show_menu=False,
            sections=main_sections,
        )

        self.has_header = True
        layout = self.layout()
        assert layout is not None and isinstance(layout, QtWidgets.QVBoxLayout)
        layout.removeWidget(dummy)
        dummy.deleteLater()

        # Header
        self.header = QtWidgets.QWidget()
        self.header.setFixedHeight(self._HEADER_H)

        h_layout = QtWidgets.QHBoxLayout(self.header)
        h_layout.setContentsMargins(4, 0, 4, 0)
        h_layout.setSpacing(2)

        self.btn_group = QtWidgets.QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.btn_group.idClicked.connect(self._on_tab_clicked)

        self.tabs_layout = QtWidgets.QHBoxLayout()
        self.tabs_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs_layout.setSpacing(4)

        h_layout.addLayout(self.tabs_layout)
        h_layout.addStretch(1)

        # Firmware update status icon
        _fw_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "fw-update.svg")
        self._fw_status_icon = UpdateStatusIcon(_fw_icon_path, size=18)
        h_layout.addWidget(self._fw_status_icon)

        # Fullscreen button
        self.btn_fs = self._make_icon_button("fullscreen.svg", "Toggle Fullscreen")
        self.btn_fs.clicked.connect(self.fullscreen_requested.emit)
        h_layout.addWidget(self.btn_fs)

        # Gear menu (uses self._sections = main_sections)
        h_layout.addWidget(self._build_menu())

        layout.insertWidget(0, self.header)

        # Stack for multiple plot widgets
        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._apply_icon_theme()

        # Activity indicator state management
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._animate_icons)

        self._device_states: dict[int, str] = {}
        self._valid_states = {"idle", "error", "init", "success", "recording"}

        # Per-port firmware states for the status icon
        self._fw_port_states: dict[str, int] = {}

    def _style_menu(self, parent_widget: QtWidgets.QWidget) -> QtWidgets.QMenu:
        """Overrides parent to create a translucent menu with section-specific grids.

        Args:
            parent_widget (QWidget): The widget to anchor the menu to.

        Returns:
            The populated QMenu instance.
        """
        menu = QtWidgets.QMenu(parent_widget)
        menu.setObjectName("PlotGlassMenu")
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        menu.setWindowFlags(
            menu.windowFlags()
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )

        # Section-specific grid key mapping
        _grid_keys = {
            "dissipation": ("grid_diss_major", "grid_diss_minor"),
            "resonance_freq": ("grid_rf_major", "grid_rf_minor"),
        }

        for i, (key, label, color) in enumerate(self._sections):
            row = PlotMenuRow(key, label, color)
            row.color_changed.connect(self.section_color_changed)
            row.visibility_changed.connect(self.section_visibility_changed)

            wa = QtWidgets.QWidgetAction(menu)
            wa.setDefaultWidget(row)
            menu.addAction(wa)

            major_key, minor_key = _grid_keys.get(key, (f"grid_{key}_major", f"grid_{key}_minor"))
            for grid_key, grid_label in (
                (major_key, "Major Gridlines"),
                (minor_key, "Minor Gridlines"),
            ):
                grid_row = GridMenuRow(grid_key, grid_label)
                grid_row.toggled.connect(self.grid_changed)

                gwa = QtWidgets.QWidgetAction(menu)
                gwa.setDefaultWidget(grid_row)
                menu.addAction(gwa)

            if i < len(self._sections) - 1:
                menu.addSeparator()

        return menu

    def add_device(
        self,
        name: str,
        plot_widget: QtWidgets.QWidget,
        accent_color: QtGui.QColor | None = None,
    ) -> int:
        """Adds a new device tab and embeds its associated plot widget.

        Args:
            name (str): The display name for the device tab.
            plot_widget (QWidget): The visual plot widget to add to the QStackedWidget.
            accent_color (QColor): Optional color to theme the device tab.

        Returns:
            int: The stacked widget index of the newly added device.
        """
        index = self.stack.addWidget(plot_widget)

        btn = QtWidgets.QPushButton(name)
        btn.setCheckable(True)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.setFixedHeight(20)
        btn.setObjectName("PlotDeviceTab")

        self.btn_group.addButton(btn, index)
        self.tabs_layout.addWidget(btn)

        if index == 0:
            btn.setChecked(True)

        self._device_states[index] = "idle"
        self._update_button_icon(index)

        return index

    def set_device_state(self, index: int, state: str) -> None:
        """Sets the activity state for a specific device tab.

        Updates the device state dictionary and controls the animation timer.
        If any device enters an animated state ("init" or "recording"), the
        timer starts.

        Args:
            index (int): The device index to update.
            state (str): The new state string ('idle', 'error', 'init', 'success', 'recording').
        """
        if index not in self._device_states or state not in self._valid_states:
            return

        self._device_states[index] = state
        needs_anim = any(s in ("init", "recording") for s in self._device_states.values())

        if needs_anim and not self._anim_timer.isActive():
            self._anim_timer.start(50)
        elif not needs_anim and self._anim_timer.isActive():
            self._anim_timer.stop()

        self._update_button_icon(index)

    def _animate_icons(self) -> None:
        """Timer callback that triggers repaints for animated states."""
        for idx, state in self._device_states.items():
            if state in ("init", "recording"):
                self._update_button_icon(idx)

    def _update_button_icon(self, index: int) -> None:
        """Updates the status dot icon for a device tab based on its state.

        Calculates dynamic colors and alpha transparency for blinking effects
        ("init" creates a hard blink, "recording" creates a smooth sine pulse).

        Args:
            index (int): The device index specifying which button to update.
        """
        state = self._device_states.get(index, "idle")
        btn = self.btn_group.button(index)
        if not btn:
            return

        tok = ThemeManager.instance().tokens()
        state_tok = {
            "idle": tok["danger"],
            "error": tok["danger"],
            "init": tok["warning"],
            "success": tok["success"],
            "recording": tok["success"],
        }
        color = QtGui.QColor(*state_tok.get(state, tok["danger"])[:3])

        if state == "init":
            alpha = 255 if (int(time.time() * 2) % 2 == 0) else 60
            color.setAlpha(alpha)
        elif state == "recording":
            alpha = int(140 + 115 * math.sin(time.time() * 3.5))
            color.setAlpha(alpha)

        btn.setIcon(self._create_dot_icon(color))

    def _create_dot_icon(self, color: QtGui.QColor) -> QtGui.QIcon:
        """Draws a solid, colored circular dot into a QIcon.

        Args:
            color (color): The QColor to fill the circle with.

        Returns:
            QIcon: A QIcon containing the generated circular indicator.
        """
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setBrush(color)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 8, 8)
        painter.end()

        return QtGui.QIcon(pixmap)

    def _on_tab_clicked(self, index: int) -> None:
        """Handles tab clicks by bringing the associated plot widget to the front.

        Args:
            index (int): The device index to switch a tab to.
        """
        self.stack.setCurrentIndex(index)

    def on_fw_status_changed(self, port: str, fw_value: int) -> None:
        """Updates the firmware status icon based on a per-port check result.

        Tracks each port's FW_UPDATE value and displays the worst-case state
        across all connected devices as a single icon indicator.

        Args:
            port (str): Serial port string (e.g., "COM3").
            fw_value (int): Integer value of FW_UPDATE enum from fwUpdater.
        """

        self._fw_port_states[port] = fw_value

        # Compute worst-case state across all known ports
        worst = UpdateStatusIcon.State.UP_TO_DATE
        detail_lines = []

        for p, v in self._fw_port_states.items():
            if v == int(FW_UPDATE.RESULT_REQUIRED):
                worst = UpdateStatusIcon.State.MANDATORY
                detail_lines.append(f"{p}: incompatible firmware")
            elif v in (int(FW_UPDATE.RESULT_OUTDATED), int(FW_UPDATE.RESULT_UNKNOWN)):
                if worst != UpdateStatusIcon.State.MANDATORY:
                    worst = UpdateStatusIcon.State.OPTIONAL
                label = (
                    "update available" if v == int(FW_UPDATE.RESULT_OUTDATED) else "unknown version"
                )
                detail_lines.append(f"{p}: {label}")
            elif v == int(FW_UPDATE.RESULT_FAILED):
                if worst == UpdateStatusIcon.State.UP_TO_DATE:
                    worst = UpdateStatusIcon.State.UNKNOWN
                detail_lines.append(f"{p}: check failed")

        self._fw_status_icon.setState(worst, "\n".join(detail_lines))


def _make_plot_widget(parent: QtWidgets.QWidget | None = None) -> GraphicsLayoutWidget:
    """Creates and configures a frameless, transparent pyqtgraph widget.

    Instantiates a `GraphicsLayoutWidget` and intentionally strips away its
    default backgrounds, frames, and shadows. This configuration
    ensures the plot can seamlessly overlay onto custom UI elements (like
    the frosted glass containers) without introducing visual artifacts or
    clashing backgrounds.

    Args:
        parent (QtWidgets.QWidget | None): The parent widget to own this layout.
            Defaults to None.

    Returns:
        GraphicsLayoutWidget: A configured, completely transparent plotting
        widget ready to accept plot items and data.
    """
    w = GraphicsLayoutWidget(parent)

    # Strip background to allow underlying custom paint events to show through
    w.setAutoFillBackground(False)
    w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    w.setBackground(None)

    # Remove all borders and frames
    w.setFrameShape(QtWidgets.QFrame.NoFrame)
    w.setFrameShadow(QtWidgets.QFrame.Plain)
    w.setLineWidth(0)

    # Ensure it doesn't collapse entirely when empty
    w.setMinimumSize(80, 60)

    return w


class UIPlots:
    """Main UI construction class for the plotting window."""

    def setup_ui(self, plots_window: QtWidgets.QMainWindow) -> None:
        """Constructs and configures the main plotting user interface.

        Builds the entire UI hierarchy, consisting of a central split layout.
        The left side holds a specialized tabbed plot container (for device
        specific metrics like Dissipation and Resonance Frequency), while the
        right side is vertically split into two standard glass containers
        (Amplitude and Temperature).

        Args:
            plots_window (QtWidgets.QMainWindow): The primary window instance
                to attach the generated UI elements to.
        """
        # Window Configuration
        screen_width = QtWidgets.QDesktopWidget().availableGeometry().width()
        use_fullscreen = screen_width == 2880

        plots_window.setObjectName("plotsWindow")
        plots_window.setMinimumSize(QtCore.QSize(1000, 250))
        plots_window.setTabShape(QtWidgets.QTabWidget.Rounded)

        if use_fullscreen:
            plots_window.resize(1701, 1435)
            plots_window.move(0, 0)
        else:
            plots_window.move(692, 0)

        # State tracker
        self._fullscreen_active_widget: QtWidgets.QWidget | None = None

        # Central Layout
        self.centralwidget = QtWidgets.QWidget(plots_window)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        root_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        root_layout.setContentsMargins(0, 8, 0, 0)
        root_layout.setSpacing(6)

        tok = ThemeManager.instance().tokens()

        # Main horizontal splitter
        self.main_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Horizontal, self.centralwidget
        )
        self.main_splitter.setHandleWidth(6)

        # Tabs
        self.left_pane = PlotTabContainer(parent=self.main_splitter)

        self.pltB = _make_plot_widget(self.left_pane)
        self.pltB.setObjectName("pltB")

        device_color = QtGui.QColor(*tok["plot_data_device_accent"][:3])
        self.left_pane.add_device("Device 1", self.pltB, device_color)

        # Stacked Plots
        self.right_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical, self.main_splitter
        )
        self.right_splitter.setHandleWidth(6)

        # Amplitude Plot
        self.plt = _make_plot_widget(self.right_splitter)
        self.plt.setObjectName("plt")

        amp_color = QtGui.QColor(220, 68, 80)
        self.amp_glass = PlotContainer(
            plot_widget=self.plt,
            title="Amplitude (dB)",
            accent_color=amp_color,
            parent=self.right_splitter,
            sections=[("amplitude", "Amplitude", amp_color)],
        )
        self.right_splitter.addWidget(self.amp_glass)

        # Temperature Plot
        self.plt_temp = _make_plot_widget(self.right_splitter)
        self.plt_temp.setObjectName("plt_temp")

        temp_color = QtGui.QColor(148, 99, 210)
        self.temp_glass = PlotContainer(
            plot_widget=self.plt_temp,
            title="Temperature °C",
            accent_color=temp_color,
            parent=self.right_splitter,
            sections=[("temperature", "Temperature", temp_color)],
        )
        self.right_splitter.addWidget(self.temp_glass)

        # Fullscreen Signal Routing
        self.left_pane.fullscreen_requested.connect(
            partial(self._toggle_fullscreen, self.left_pane)
        )
        self.amp_glass.fullscreen_requested.connect(
            partial(self._toggle_fullscreen, self.amp_glass)
        )
        self.temp_glass.fullscreen_requested.connect(
            partial(self._toggle_fullscreen, self.temp_glass)
        )

        # Splitter Ratios
        root_layout.addWidget(self.main_splitter)
        plots_window.setCentralWidget(self.centralwidget)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        self.right_splitter.setStretchFactor(0, 1)
        self.right_splitter.setStretchFactor(1, 1)

        # Translation
        self.Layout_graphs = QtWidgets.QWidget()
        self.btnCollapse = QtWidgets.QToolButton()
        self.btnExpand = QtWidgets.QToolButton()

        if hasattr(self, "retranslateUi"):
            self.retranslateUi(plots_window)

        QtCore.QMetaObject.connectSlotsByName(plots_window)

    def _toggle_fullscreen(self, target_widget: QtWidgets.QWidget) -> None:
        """Toggle a pane between fullscreen and normal splitter layouts.

        Expands the selected pane to occupy all available space within the
        splitter hierarchy. If the selected pane is already fullscreen, the
        original splitter sizes are restored. Transitions are animated using
        a timer-driven quartic ease-in-out interpolation for smooth resizing.

        During the animation, plot updates are temporarily disabled to prevent
        rendering stutter caused by frequent splitter resize events. Updates are
        restored automatically when the animation completes.

        Args:
            target_widget: Pane widget whose fullscreen state should be toggled.

        Returns:
            None
        """
        # Determine the target splitter sizes.
        if getattr(self, "_fullscreen_active_widget", None) == target_widget:
            target_main_sizes = self._normal_main_sizes
            target_right_sizes = self._normal_right_sizes

            target_widget._is_fullscreen = False
            target_widget._apply_icon_theme()
            target_widget.btn_fs.setToolTip("Toggle Fullscreen")

            self._fullscreen_active_widget = None
        else:
            previous_widget = getattr(self, "_fullscreen_active_widget", None)

            if previous_widget is None:
                self._normal_main_sizes = self.main_splitter.sizes()
                self._normal_right_sizes = self.right_splitter.sizes()
            else:
                previous_widget._is_fullscreen = False
                previous_widget._apply_icon_theme()

            self._fullscreen_active_widget = target_widget

            target_widget._is_fullscreen = True
            target_widget._apply_icon_theme()
            target_widget.btn_fs.setToolTip("Restore Size")

            total_main = sum(self.main_splitter.sizes())
            total_right = sum(self.right_splitter.sizes())

            fullscreen_sizes = {
                self.left_pane: (
                    [total_main, 0],
                    self._normal_right_sizes,
                ),
                self.amp_glass: (
                    [0, total_main],
                    [total_right, 0],
                ),
                self.temp_glass: (
                    [0, total_main],
                    [0, total_right],
                ),
            }

            if target_widget not in fullscreen_sizes:
                return

            target_main_sizes, target_right_sizes = fullscreen_sizes[target_widget]

        # Stop any active animation before starting a new one.
        if hasattr(self, "_fs_timer") and self._fs_timer.isActive():
            self._fs_timer.stop()
            self._freeze_plots(False)

        self._freeze_plots(True)

        duration_ms = 380
        interval_ms = 16  # ~60 FPS
        total_steps = max(1, duration_ms // interval_ms)

        start_main_sizes = list(self.main_splitter.sizes())
        start_right_sizes = list(self.right_splitter.sizes())
        step_count = [0]

        def _ease_in_out_quart(progress: float) -> float:
            """Return a quartic ease-in-out interpolation value.

            Args:
                progress: Normalized animation progress in the range [0.0, 1.0].

            Returns:
                Eased interpolation value in the range [0.0, 1.0].
            """
            if progress < 0.5:
                return 8.0 * progress**4

            progress -= 1.0
            return 1.0 - 8.0 * progress**4

        def _tick() -> None:
            """Advance the splitter animation by one frame."""
            step_count[0] += 1

            progress = min(step_count[0] / total_steps, 1.0)
            eased_progress = _ease_in_out_quart(progress)

            main_sizes = [
                int(start + (end - start) * eased_progress)
                for start, end in zip(start_main_sizes, target_main_sizes)
            ]
            right_sizes = [
                int(start + (end - start) * eased_progress)
                for start, end in zip(start_right_sizes, target_right_sizes)
            ]

            self.main_splitter.setSizes(main_sizes)
            self.right_splitter.setSizes(right_sizes)

            if progress >= 1.0:
                self._fs_timer.stop()
                self._freeze_plots(False)

        self._fs_timer = QtCore.QTimer(self.centralwidget)
        self._fs_timer.setInterval(interval_ms)
        self._fs_timer.timeout.connect(_tick)
        self._fs_timer.start()

    def _freeze_plots(self, freeze: bool) -> None:
        """Enable or suppress plot redraws during splitter animations.

        Temporarily disables viewport updates on the application's
        pyqtgraph widgets to reduce rendering overhead while splitter
        sizes are being animated. When updates are re-enabled, a repaint
        is requested to ensure the plots reflect their final layout.

        GraphicsLayoutWidget inherits from `QGraphicsView`, allowing its
        viewport update mode to be adjusted to avoid expensive redraws on
        every resize event.

        Args:
            freeze: If `True`, disables viewport updates. If `False`,
                restores normal update behavior and triggers a repaint.

        Returns:
            None
        """
        from PyQt5.QtWidgets import QGraphicsView

        update_mode = (
            QGraphicsView.NoViewportUpdate if freeze else QGraphicsView.MinimalViewportUpdate
        )

        for plot_name in ("plt", "plt_temp", "pltB"):
            plot_widget = getattr(self, plot_name, None)
            if plot_widget is None:
                continue

            try:
                plot_widget.setViewportUpdateMode(update_mode)

                if not freeze:
                    plot_widget.viewport().update()
            except Exception:
                # Ignore widgets that do not expose the expected
                # QGraphicsView interface.
                continue

    def retranslateUi(self, plots_window: QtWidgets.QMainWindow) -> None:  # noqa: N802
        """Apply translated text and window resources to the plots window.

        Sets the application icon and updates the window title using Qt's
        translation framework. This method is typically called during UI
        initialization and whenever the application's language changes.

        Args:
            plots_window (QWidget): Main window instance that hosts the plots UI.
        """
        translate = QtCore.QCoreApplication.translate

        icon_path = os.path.join(
            Architecture.get_path(),
            "QATCH",
            "icons",
            "qatch-icon.png",
        )

        plots_window.setWindowIcon(QtGui.QIcon(icon_path))
        plots_window.setWindowTitle(
            translate(
                "plotsWindow",
                f"{Constants.app_title} {Constants.app_version} - Plots",
            )
        )
