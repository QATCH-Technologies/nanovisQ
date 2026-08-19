"""
QATCH.ui.widgets.data_management_widget.py

Data management overlay and supporting panel components.

Provides the main :class:`DataManagementWidget` overlay used to access the
application's data-management modes. The widget manages the shared
`DataServices` instance, overlay lifecycle, fullscreen transitions, mode
navigation, and switching between the available mode views through a
`TabbedRailPanel`.

The module also defines the private :class:`_Panel` widget used as the main
overlay container. It renders its background, border, and corner radius
directly during painting so these properties can be animated efficiently
without repeatedly applying style sheets to the panel and its child widgets.

The five data-management mode widgets are created by
:class:`DataManagementWidget` and receive the shared services instance for
operations that span multiple modes.

Compatibility methods such as `showNormal` and `open_mode` are retained
to support existing callers that previously interacted with the legacy
`export_widget.Ui_Export` interface.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-19
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.common.data_service import DataServices
from QATCH.ui.components import TabbedRailPanel
from QATCH.ui.components.overlay_shell import (
    FULLSCREEN_ANIM_EASING,
    OverlayLifecycleMixin,
    rebuild_fullscreen_icons,
    run_variant_animation,
)
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.widgets.data_mode_advanced import AdvancedMode
from QATCH.ui.widgets.data_mode_export import ExportMode
from QATCH.ui.widgets.data_mode_history import HistoryMode
from QATCH.ui.widgets.data_mode_import import ImportMode
from QATCH.ui.widgets.data_mode_recover import RecoverMode

TAG = "[DataManagement]"

MODE_CLASSES = [ImportMode, ExportMode, RecoverMode, AdvancedMode, HistoryMode]
_TAB_TO_KEY = {0: "import", 1: "export", 2: "recover", 3: "advanced", 4: "history"}


class _Panel(QtWidgets.QFrame):
    """Main container panel for the overlay.

    Renders its background, border, and corner radius directly during the
    paint event rather than relying on style sheets. This allows appearance
    properties to be animated efficiently without repeatedly triggering
    style recalculation and repolishing of the panel's child widgets.

    The panel also maintains a rounded widget mask so that the corners outside
    the painted area remain transparent both during normal display and when
    the widget is captured or rendered to an image.

    Attributes:
        _bg_alpha (int | float): Opacity of the panel background.
        _border_width (int | float): Width of the panel border.
        _radius (int | float): Corner radius of the panel.
    """

    def __init__(self, parent=None):
        """Initialize the panel.

        Args:
            parent (QtWidgets.QWidget, optional): Parent widget for the panel.
                Defaults to None.
        """
        super().__init__(parent)
        self._bg_alpha = 235
        self._border_width = 1.5
        self._radius = 12.0
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def set_appearance(self, alpha, border_width, radius):
        """Update the panel's appearance properties.

        Args:
            alpha (int | float): Background opacity.
            border_width (int | float): Border width in pixels.
            radius (int | float): Corner radius in pixels.
        """
        self._bg_alpha = alpha
        self._border_width = border_width
        self._radius = radius
        self._update_mask()
        self.update()

    def resizeEvent(self, event):
        """Update the panel mask after the widget is resized.

        Args:
            event (QtGui.QResizeEvent): Resize event containing the widget's
                new dimensions.
        """
        super().resizeEvent(event)
        self._update_mask()

    def _update_mask(self) -> None:
        """Update the widget mask to match the current corner radius.

        Applies a rounded-rectangle mask so that pixels outside the panel's
        rounded shape are excluded from rendering. This ensures transparent
        corners during normal display as well as when the widget is captured
        or rendered to an image.

        If the configured corner radius is zero or less, the widget mask is
        cleared and the entire rectangular widget area remains available for
        rendering.
        """
        if self._radius <= 0:
            self.clearMask()
            return
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(self.rect()), self._radius, self._radius)
        self.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))

    def paintEvent(self, event):
        """Paint the panel background and border.

        Draws the panel using the active theme's background and border colors,
        applying the configured opacity, border width, and corner radius.

        Args:
            event (QtGui.QPaintEvent): Paint event requesting the widget to
                redraw.
        """
        tok = ThemeManager.instance().tokens()
        base = tok["plot_glass_base"]
        rim = tok["plot_glass_rim"]

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        bw = self._border_width
        rect = QtCore.QRectF(self.rect()).adjusted(bw / 2, bw / 2, -bw / 2, -bw / 2)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(base[0], base[1], base[2], int(self._bg_alpha)))
        if self._radius > 0:
            painter.drawRoundedRect(rect, self._radius, self._radius)
        else:
            painter.drawRect(rect)

        if bw > 0:
            painter.setPen(QtGui.QPen(QtGui.QColor(rim[0], rim[1], rim[2], 230), bw))
            painter.setBrush(QtCore.Qt.NoBrush)
            if self._radius > 0:
                painter.drawRoundedRect(rect, self._radius, self._radius)
            else:
                painter.drawRect(rect)
        painter.end()


class DataManagementWidget(OverlayLifecycleMixin, QtWidgets.QWidget):
    """Overlay widget for managing application data.

    Provides a shared overlay container for data-management modes, including
    the navigation rail and active mode content. Common overlay behavior such
    as the scrim, panel geometry, opacity handling, and parent resize
    tracking is provided by `OverlayLifecycleMixin`.

    Attributes:
        ICON_MAIN (str): Path to the primary data management icon.
        ICON_EXPAND (str): Path to the icon used to expand a navigation item.
        ICON_COLLAPSE (str): Path to the icon used to collapse a navigation
            item.
        services (DataServices): Shared data-management services used by the
            available modes.
        _current_key (str | None): Key identifying the currently active
            management mode.
        _close_fade_proxy (QtWidgets.QWidget | None): Temporary static
            representation used while animating the overlay close operation.
        _modes (dict): Mapping of mode keys to their corresponding widgets.
        panel (QtWidgets.QWidget): Main panel containing the navigation rail
            and active mode content.
    """

    def __init__(self, parent=None) -> None:
        """Initialize the data management overlay.

        Args:
            parent (QtWidgets.QWidget, optional): Parent widget over which
                the overlay is displayed. Defaults to None.
        """
        super().__init__(parent)
        self.ICON_MAIN = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "import-export.svg"
        )
        self.ICON_EXPAND = os.path.join(Architecture.get_path(), "QATCH", "icons", "expand.svg")
        self.ICON_COLLAPSE = os.path.join(Architecture.get_path(), "QATCH", "icons", "collapse.svg")

        # Shared machinery, injected into every mode.
        self.services = DataServices(self)

        self._current_key = None  # active mode (container-tracked, not the rail)
        self._close_fade_proxy = None  # static-pixmap stand-in animated during close
        glass_frame = _Panel(self)
        glass_frame.setObjectName("dmview")
        self._init_overlay_shell(
            parent,
            "dmview",
            panel_alpha=235,
            margin_pct=0.175,
            content_margins=(20, 14, 20, 20),
            content_spacing=14,
            glass_frame=glass_frame,
        )

        self._modes = {}  # key -> mode widget
        self._build_panel()

        # Header across the top
        self.main_layout.addLayout(self.header_layout)
        body_row = QtWidgets.QHBoxLayout()
        body_row.setSpacing(0)
        body_row.addWidget(self.panel, 1)
        self.main_layout.addLayout(body_row, 1)
        self._finish_overlay_shell()

    def _build_panel(self) -> None:
        """Build the overlay header, mode navigation, and content panel.

        Creates the overlay header and initializes the navigation rail and content
        stack used to switch between data-management modes. Each mode receives
        the shared :class:`DataServices` instance and is registered with the
        panel using its mode key.

        The method also configures the mode icons, connects mode-selection
        changes to the corresponding handler, and applies the current theme
        before subscribing to future theme changes.
        """
        self.header_layout = self._build_overlay_header(
            self.ICON_MAIN, "Data Management", fullscreen=True
        )

        # Vertical rail of modes beside the content stack
        _icon_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        _mode_icons = {
            "import": os.path.join(_icon_dir, "import.svg"),
            "export": os.path.join(_icon_dir, "export.svg"),
            "recover": os.path.join(_icon_dir, "recover.svg"),
            "advanced": os.path.join(_icon_dir, "gear.svg"),
            "history": os.path.join(_icon_dir, "history.svg"),
        }
        modes = [
            (cls.MODE_KEY, cls.MODE_LABEL, _mode_icons.get(cls.MODE_KEY)) for cls in MODE_CLASSES
        ]
        self.panel = TabbedRailPanel(modes, content_radius=11.0)
        self.panel.currentChanged.connect(self._on_mode_changed)

        # Instantiate each mode with the shared services and register it.
        for cls in MODE_CLASSES:
            mode = cls(self.services, parent=self.panel.stack)
            self._modes[cls.MODE_KEY] = mode
            self.panel.add_page(cls.MODE_KEY, mode)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Handle a theme change by refreshing theme-dependent styling.

        Args:
            _mode (str): Identifier for the newly activated theme mode. The value
                is not used directly because the current theme is obtained from
                the theme manager.
        """

        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme to the overlay's theme-dependent elements.

        Refreshes the header styling. The content panel and navigation rail
        handle their own theme-dependent rendering, so no additional styling is
        required here.
        """
        self._refresh_header_theme()

    def _rebuild_fs_icons(self) -> None:
        """Rebuild the fullscreen control icons for the active state.

        Regenerates the expand and collapse icons used by the fullscreen control
        so they reflect the current theme and fullscreen state.
        """
        rebuild_fullscreen_icons(self, self.ICON_EXPAND, self.ICON_COLLAPSE)

    def _on_mode_changed(self, key):
        """Handle a change to the active data-management mode.

        Performs lifecycle housekeeping when the navigation panel switches to a
        different mode. The previously active mode receives `on_leave()` and
        the newly selected mode becomes the current mode. If the overlay is
        already visible, the new mode is immediately initialized with
        `on_enter()`. Initialization is deferred when the overlay is still
        opening to avoid blocking its reveal animation.

        Args:
            key (str): Key identifying the newly selected data-management mode.
        """
        prev_key = self._current_key
        mode = self._modes.get(key)
        if mode is None or key == prev_key:
            return

        prev_mode = self._modes.get(prev_key) if prev_key else None
        if prev_mode is not None:
            prev_mode.on_leave()

        self._current_key = key
        if self._revealed and self.isVisible():
            mode.on_enter()

    def open_mode(self, key) -> None:
        """Open the overlay and activate the specified mode.

        Args:
            key (str): Key identifying the data-management mode to activate.
        """
        self.panel.set_active(key)
        self.setVisible(True)

    def showNormal(self, tab_idx=0) -> None:
        """Open the overlay using a legacy tab index.

        Provides backwards compatibility for callers that previously selected
        data-management modes using numeric tab indices.

        Args:
            tab_idx (int, optional): Legacy tab index used to determine which
                mode to activate. Defaults to 0.
        """
        self.open_mode(_TAB_TO_KEY.get(tab_idx, "import"))

    def _apply_shadow(
        self,
        widget,
        blur_radius=15,
        alpha=40,
        offset=(0, 4),
    ) -> None:
        """Apply a drop shadow effect to a widget.

        Args:
            widget (QtWidgets.QWidget): Widget that receives the shadow effect.
            blur_radius (int, optional): Blur radius of the shadow. Defaults to
                15.
            alpha (int, optional): Alpha value of the shadow color. Defaults to
                40.
            offset (tuple[int, int], optional): Horizontal and vertical shadow
                offsets in pixels. Defaults to `(0, 4)`.
        """
        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(blur_radius)
        shadow.setColor(QtGui.QColor(0, 0, 0, alpha))
        shadow.setOffset(offset[0], offset[1])
        widget.setGraphicsEffect(shadow)

    def _apply_panel_appearance(self, frac: float) -> None:
        """Update the panel appearance for an overlay animation state.

        Interpolates the panel's background opacity, border width, and corner
        radius based on the current inset fraction. This keeps the panel
        appearance synchronized with its animated geometry during fullscreen
        transitions.

        Args:
            frac (float): Current inset amount used to calculate the normalized
                interpolation factor. A value of `0` represents the fullscreen
                state, while `_default_margin_pct` represents the fully inset
                state.
        """
        p = 0.0 if self._default_margin_pct <= 0 else min(1.0, frac / self._default_margin_pct)
        alpha = int(255 + (235 - 255) * p)
        border = 1.5 * p
        radius = 12.0 * p
        self.glass_frame.set_appearance(alpha, border, radius)

    def toggle_fullscreen(self) -> None:
        """Toggle the overlay between fullscreen and inset states.

        Starts an animated transition between the configured default margins and
        the fullscreen state. The fullscreen control icons are rebuilt before
        the animation begins so the control reflects the resulting action.
        """
        self._is_fullscreen = not self._is_fullscreen
        self._rebuild_fs_icons()

        target = 0.0 if self._is_fullscreen else self._default_margin_pct
        start = self._default_margin_pct if self._is_fullscreen else 0.0

        def _step(t):
            frac = start + (target - start) * t
            self._apply_margin_frac(frac)

        run_variant_animation(
            self,
            "_fs_anim",
            duration=240,
            easing=FULLSCREEN_ANIM_EASING,
            on_step=_step,
        )

    def _on_before_reveal(self) -> None:
        """Prepare shared services before revealing the overlay.

        Starts the shared data-management services required while the overlay is
        open.
        """
        self.services.start()  # spin up the shared USB loop on open

    def _on_after_reveal(self) -> None:
        """Initialize the active mode after the overlay reveal begins.

        Defers the active mode's `on_enter()` call until a later event-loop
        iteration so potentially expensive initialization does not block the
        overlay reveal animation.
        """
        key = self.panel.active_key()

        def _populate():
            mode = self._modes.get(key)
            if mode is not None:
                self._current_key = key
                mode.on_enter()

        QtCore.QTimer.singleShot(16, _populate)

    def resizeEvent(self, event):
        """Handle resizing of the overlay widget.

        Recalculates the overlay geometry when the widget is resized while
        visible, ensuring the panel continues to track the available parent
        dimensions.

        Args:
            event (QtGui.QResizeEvent): Resize event containing the widget's new
                dimensions.
        """
        super().resizeEvent(event)
        if self.isVisible():
            self._refit_to_parent()

    def _animate_close(self) -> None:
        """Animate the overlay closing.

        Captures the current panel appearance into a temporary proxy widget,
        hides the live panel, and fades the proxy and overlay scrim to
        transparency. The proxy preserves the panel's current appearance while
        the close animation runs.

        The close and fullscreen controls are kept above the proxy so they remain
        visible during the transition.
        """
        self._teardown_close_fade_proxy()
        cur_op = self._glass_opacity.opacity() if self._glass_opacity else 1.0
        geo = self.glass_frame.geometry()
        pix = self.glass_frame.grab()
        self.glass_frame.hide()

        proxy = QtWidgets.QLabel(self)
        proxy.setObjectName("closeFadeProxy")
        proxy.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        proxy.setPixmap(pix)
        proxy.setGeometry(geo)
        proxy_radius = self.glass_frame._radius
        proxy_path = QtGui.QPainterPath()
        proxy_path.addRoundedRect(QtCore.QRectF(proxy.rect()), proxy_radius, proxy_radius)
        proxy.setMask(QtGui.QRegion(proxy_path.toFillPolygon().toPolygon()))
        proxy.show()
        proxy.raise_()

        # Keep the window-control buttons
        self.btn_close.raise_()
        self.btn_fullscreen.raise_()
        self._close_fade_proxy = proxy

        proxy_opacity = QtWidgets.QGraphicsOpacityEffect(proxy)
        proxy_opacity.setOpacity(cur_op)
        proxy.setGraphicsEffect(proxy_opacity)

        self._run_fade(
            scrim_from=self._scrim_alpha,
            scrim_to=0,
            op_from=cur_op,
            op_to=0.0,
            duration=180,
            easing=QtCore.QEasingCurve.InQuad,
            on_done=self._do_close,
            opacity_effect=proxy_opacity,
        )

    def _teardown_close_fade_proxy(self) -> None:
        """Remove the temporary close-animation proxy.

        Hides and schedules the proxy widget for deletion when one exists, then
        clears the stored proxy reference. Runtime errors caused by an already
        deleted Qt object are safely ignored.
        """
        proxy = self._close_fade_proxy
        if proxy is not None:
            try:
                proxy.hide()
                proxy.setParent(None)
                proxy.deleteLater()
            except RuntimeError:
                pass
            self._close_fade_proxy = None

    def _do_close(self) -> None:
        """Finalize the overlay close operation.

        Stops shared data-management services, removes the temporary close-fade
        proxy, restores the default fullscreen state and control icon, resets
        panel state, and delegates the remaining overlay cleanup to the base
        lifecycle implementation.
        """
        self.services.stop()  # tear down the shared USB loop on close
        self._teardown_close_fade_proxy()
        self._is_fullscreen = False
        # Restore the expand icon for the next open.
        self._rebuild_fs_icons()
        self._panel_alpha = 235
        self._current_key = None
        self.panel.stack.show()
        super()._do_close()
