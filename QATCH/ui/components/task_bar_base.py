"""Shared themed chrome and button construction for task bars.

Provides the common Python-side styling and behavior used by the application's
`CtrlToolBar`-styled task bars, including the Analyze and Controls action
bars. The module centralizes toolbar construction, tool-button configuration,
icon management, checked-state repaint behavior, layout constants, and the
rounded task-bar surface so individual task bars do not need to duplicate
these details.

The shared implementation complements the application's
`app_theme.qss` rules for `QToolBar#CtrlToolBar` and its
`QToolButton` children. QSS remains responsible for state-dependent button
styling such as hover, pressed, checked, and disabled states, while this
module handles the Python-side configuration required for those rules to
behave consistently.

Subclasses are expected to construct their own toolbar zones and layouts while
using :meth:`TaskBarBase._make_toolbar` and the corresponding tool-button
helper for shared controls. Subclasses that maintain additional
theme-dependent state should handle their own `themeChanged` processing and
call the provided icon-retinting helper as appropriate.

The module intentionally keeps task-bar-specific layout and behavior in the
individual action-bar subclasses. `TaskBarBase` supplies only the shared
visual chrome and infrastructure needed to keep those bars consistent.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, ThemeMode, tok_css


def _icon_path(icon_name: str) -> str:
    return os.path.join(Architecture.get_path(), "QATCH", "icons", icon_name)


class TaskBarBase(QtWidgets.QWidget):
    """Base widget providing shared chrome for themed task bars.

    Supplies the common configuration and infrastructure used by the
    application's top-level task bars, including toolbar construction,
    tool-button sizing, icon tracking, theme-aware background rendering, and
    shared layout constants.

    Subclasses are responsible for constructing their own toolbar zones and
    layouts. They should use :meth:`_make_toolbar` for every toolbar and the
    class's tool-button helper for every tool button so icon sizing, button
    dimensions, cursor behavior, and checked-state repaint handling remain
    consistent across task bars.

    Subclasses should use :attr:`OUTER_MARGINS` and :attr:`ZONE_SPACING`
    when assembling their layouts rather than duplicating the corresponding
    values. If a subclass has its own `themeChanged` handler, it should
    explicitly call :meth:`retint_icon_buttons` as needed because this base
    class does not automatically retint tracked icons on theme changes.
    Background repainting is handled here because it is common to every
    subclass.

    Attributes:
        _icon_buttons: List of `(button, icon_name)` pairs registered by
            the tool-button construction helper. These are used for batch icon
            retinting when the theme changes.
        _bg_cache: Cached rendered task-bar background pixmap, or `None` if
            no cached background is currently available.
        _bg_cache_mode: Theme mode corresponding to `_bg_cache`, or
            `None` when no background cache exists.

    Class Attributes:
        ICON_SIZE: Standard icon size applied to task-bar toolbars.
        BUTTON_HEIGHT: Standard height of task-bar tool buttons.
        _CARD_RADIUS: Corner radius of the task-bar background surface.
        OUTER_MARGINS: Left, top, right, and bottom margins used when
            assembling task-bar layouts.
        ZONE_SPACING: Horizontal spacing between task-bar zones.
        DIVIDER_INSET: Vertical inset applied to zone dividers so they read as
            floating accents rather than full-height rules.
    """

    ICON_SIZE = QtCore.QSize(50, 30)
    BUTTON_HEIGHT = 60
    _CARD_RADIUS = 12.0
    OUTER_MARGINS = (10, 1, 10, 1)
    ZONE_SPACING = 14
    DIVIDER_INSET = 16

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the shared task-bar state.

        Disables automatic background filling so the task bar can render its
        own themed surface, initializes icon tracking and background caching,
        and connects theme changes to a background repaint.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        # (button, icon_name) pairs built via _tool_button(), retinted as a
        # batch by retint_icon_buttons().
        self._icon_buttons: list[tuple[QtWidgets.QAbstractButton, str]] = []
        self._bg_cache: QtGui.QPixmap | None = None
        self._bg_cache_mode: ThemeMode | None = None
        ThemeManager.instance().themeChanged.connect(lambda _: self.update())

    def _make_toolbar(self) -> QtWidgets.QToolBar:
        """Create a toolbar configured for the task-bar visual system.

        Creates a `QToolBar` with the `CtrlToolBar` object name expected
        by the application's theme stylesheet and applies the shared task-bar
        icon size.

        Returns:
            A configured :class:`QToolBar` ready to be added to a task-bar
            layout.
        """
        bar = QtWidgets.QToolBar()
        bar.setObjectName("CtrlToolBar")
        bar.setIconSize(self.ICON_SIZE)
        return bar

    def _make_divider(self, toolbar_h: int) -> TaskBarDivider:
        """Create a standardized divider between task-bar zones.

        Sizes the divider from the supplied toolbar height while applying the
        shared :attr:`DIVIDER_INSET` so the divider appears as a floating accent
        rather than spanning the full height of the task bar.

        Args:
            toolbar_h: Height of the toolbar row used as the divider's reference
                height.

        Returns:
            A :class:`TaskBarDivider` sized according to the shared task-bar
            divider geometry.
        """
        return TaskBarDivider(max(toolbar_h - self.DIVIDER_INSET, 1))

    def _tool_button(
        self,
        text: str,
        icon_name: str | None = None,
        checkable: bool = False,
    ) -> QtWidgets.QToolButton:
        """Create a consistently configured task-bar tool button.

        Configures the button with the shared text-under-icon presentation,
        standard button height, pointing-hand cursor, and optional checkable
        behavior. Checkable buttons are automatically repolished when their
        checked state changes so programmatic state changes immediately trigger
        the corresponding QSS styling.

        When an icon name is supplied, the button and icon name are registered in
        the task bar's icon-button collection for later batch retinting.

        Args:
            text: Text displayed beneath the button icon.
            icon_name: Optional filename of the icon associated with the button.
                When provided, the button is registered for theme-aware icon
                retinting.
            checkable: Whether the button should maintain a checked state.

        Returns:
            A configured :class:`QToolButton`.
        """
        btn = QtWidgets.QToolButton()
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        btn.setText(text)
        btn.setFixedHeight(self.BUTTON_HEIGHT)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        btn.setCheckable(checkable)
        if checkable:
            btn.toggled.connect(lambda _checked, b=btn: self._repolish(b))
        if icon_name:
            self._icon_buttons.append((btn, icon_name))
        return btn

    @staticmethod
    def _repolish(widget: QtWidgets.QWidget) -> None:
        """Force Qt to immediately re-evaluate a widget's stylesheet state.

        Explicitly unpolishes and reapplies the widget's style before requesting
        a repaint. This ensures QSS pseudo-states such as `:checked` and
        `:disabled` are refreshed immediately when their state changes
        programmatically rather than through a native user interaction.

        Args:
            widget: Widget whose active Qt style should be reapplied.
        """
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def retint_icon_buttons(self) -> None:
        """Retint all registered task-bar icons using the active theme.

        Applies the current theme's `flat_text` color to every icon registered
        through :meth:`_tool_button`. This method is intentionally public so
        task-bar subclasses or their owning UI controllers can invoke it from
        existing theme-change handling without requiring `TaskBarBase` to
        subscribe independently to the theme signal.

        """
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["flat_text"])
        for btn, icon_name in self._icon_buttons:
            btn.setIcon(tinted_icon(_icon_path(icon_name), color, self.ICON_SIZE.height()))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the task bar's shared rounded-card background.

        Renders a cached themed background pixmap behind the task-bar contents.
        The background is regenerated only when the widget size or active theme
        changes; subsequent paint events simply blit the cached pixmap.

        Caching avoids repeatedly rendering the rounded fill and border during
        high-frequency UI updates, such as the approximately 100 ms refreshes
        used by live temperature and PID readouts.

        Args:
            event: Qt paint event generated when the task bar requires repainting.
        """
        mode = ThemeManager.instance().mode()
        size = self.size()
        if self._bg_cache is None or self._bg_cache.size() != size or self._bg_cache_mode != mode:
            self._bg_cache = self._render_background(size)
            self._bg_cache_mode = mode

        p = QtGui.QPainter(self)
        p.drawPixmap(0, 0, self._bg_cache)
        p.end()

    def _render_background(self, size: QtCore.QSize) -> QtGui.QPixmap:
        """Render the task bar's themed rounded background into a pixmap.

        Creates a transparent offscreen pixmap and paints the shared flat surface
        recipe into it using the active theme's surface and border tokens. The
        resulting pixmap is cached by :meth:`paintEvent` for reuse until the task
        bar is resized or the theme changes.

        Args:
            size: Dimensions of the background pixmap to render.

        Returns:
            A transparent pixmap containing the fully rendered task-bar
            background.
        """
        tok = ThemeManager.instance().tokens()
        pm = QtGui.QPixmap(size)
        pm.fill(QtCore.Qt.GlobalColor.transparent)

        p = QtGui.QPainter(pm)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._CARD_RADIUS,
            fill=QtGui.QColor(*tok["surface"]),
            border=QtGui.QColor(*tok["surface_border"]),
            painter=p,
        )
        p.end()
        return pm


class TaskBarDivider(QtWidgets.QFrame):
    """Themed vertical divider separating task-bar zones.

    Renders as a one-pixel-wide vertical rule whose height is explicitly
    controlled by the caller. The divider uses the active theme's
    `flat_border` token and automatically updates when the application
    theme changes.

    The intended height is the task-bar toolbar row height with the shared
    :attr:`TaskBarBase.DIVIDER_INSET` applied. Callers should add the divider
    with `Qt.AlignVCenter` so it reads as a floating accent between zones
    rather than a full-height edge-to-edge rule.

    Args:
        height: Height of the divider in pixels.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        height: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the divider with the requested height.

        Args:
            height: Vertical height of the divider in pixels.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setFixedWidth(1)
        self.setFixedHeight(height)
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the divider styling after a theme change.

        Args:
            _mode: Theme mode supplied by the ``themeChanged`` signal.
                The value is not used directly because the current theme
                tokens are retrieved from :class:`ThemeManager`.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme's divider color.

        Uses the ``flat_border`` theme token as the divider's background
        color and removes the frame border so the widget renders as a
        single-pixel flat rule.
        """
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(f"background-color: {tok_css(tok['flat_border'])}; border: none;")
