"""
QATCH.ui.compoinents.qatch_push_button.py

Flat-styled push button controls for the QATCH application.

Provides a `QPushButton` subclass that follows the application's flat
control system, as defined by `QATCH.ui.components.flat_paint`. The module
consolidates the functionality formerly provided by the separate
`GlassPushButton` and `BorderlessActionButton` classes into a single
button family with configurable visual variants.

All button surface colors, borders, focus rings, hover states, and interaction
states are resolved from the application's `flat_*` theme tokens. This
allows buttons to remain synchronized with light and dark themes without
maintaining separate theme-specific color definitions.

Variants:
    primary: Solid accent-colored fill for primary call-to-action controls
        such as Initialize and Save.
    secondary: Transparent fill with a strong border for standard actions
        such as Advanced, Refresh, Cancel, and Back. This is the default
        variant and the fallback for unrecognized variant names.
    ghost: Transparent button with accent-colored text and no border for
        quiet inline actions such as Forgot Password? and per-field
        Save/Reset/Default controls.
    destructive: Solid error-colored fill with white text for destructive
        actions requiring explicit confirmation, such as permanent deletion.
    destructive_outline: Transparent button with an error-colored border and
        text for softer destructive actions shown before confirmation.
    ghost_danger: Transparent button with error-colored text and no border
        for quiet destructive actions such as Sign Out.
    icon_toolbar: Vertical icon-above-label layout intended for toolbar
        controls. The variant is available for future toolbar use but is not
        currently wired to application call sites.

Legacy variant aliases are supported for backwards compatibility. The old
`default`, `neutral`, `danger`, and `danger_confirm` names resolve to
their corresponding canonical variants. The obsolete `warning` variant is
not supported. Any unrecognized variant name falls back to `secondary`.

The module also supports specialized button layouts, including left-aligned
icons with centered text and full-width menu-row buttons with a leading icon.
These layouts are rendered manually where Qt's standard button label
pipeline cannot provide the required positioning reliably.

Example:
    Create a primary action button::

        btn = QATCHPushButton("Add", variant="primary")
        btn.setIcon(QtGui.QIcon(path))
        btn.setIconSize(QtCore.QSize(18, 18))
        btn.setFixedHeight(34)

    Create a quiet inline action using the `ghost` variant::

        btn_link = QATCHPushButton("Reset", variant="ghost")

    Change a destructive action into a confirmation state at runtime::

        btn_del.set_variant("destructive")
        btn_del.set_variant("destructive_outline")

    Configure a button with a leading icon while keeping its text centered::

        btn.set_icon_left(True)

    Configure a button as a full-width menu row::

        btn.set_menu_row(True)

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.styles.typography import FONT_SANS_STACK, make_qfont

_RADIUS = 7.0
_RADIUS_ICON_TOOLBAR = 8.0

# Old variant name -> canonical variant name.
_VARIANT_ALIASES: dict[str, str] = {
    "default": "secondary",
    "neutral": "secondary",
    "danger": "destructive_outline",
    "danger_confirm": "destructive",
}

_CANONICAL_VARIANTS = frozenset(
    {
        "primary",
        "secondary",
        "ghost",
        "ghost_danger",
        "destructive",
        "destructive_outline",
        "icon_toolbar",
    }
)


class QATCHPushButton(QtWidgets.QPushButton):
    """Provide a themed push button with token-driven rendering.

    Extends `QPushButton` with custom flat-control rendering that matches
    the application's shared flat control system. Fill, border, and focus-ring
    colors are resolved from the active `flat_*` theme tokens at paint time,
    allowing the button to respond automatically to light and dark theme
    changes without maintaining separate palette definitions.

    A minimal stylesheet is used only for layout-related properties such as
    padding and font configuration. Qt's native icon and text rendering
    pipeline remains responsible for drawing the button content.

    The button supports configurable visual variants and tracks hover and
    pressed states internally so that the appropriate surface styling can be
    rendered during `paintEvent`.

    Attributes:
        _variant: Requested button variant name. May be an alias or canonical
            variant name recognized by the button's styling system.
        _hovered: Whether the mouse cursor is currently inside the widget.
        _pressed_state: Whether a mouse button is currently held over the
            button.
        _border_visible: Whether the button's border stroke should be drawn.
        _icon_left: Whether the button icon should be positioned on the left
            side of its content.
        _menu_row: Whether the button is being rendered as a menu-row control.
    """

    def __init__(
        self,
        text: str = "",
        parent: QtWidgets.QWidget | None = None,
        *,
        variant: str = "secondary",
    ) -> None:
        """Initialize the flat-styled push button.

        Args:
            text: Text to display on the button.
            parent: Optional parent widget.
            variant: Visual variant used to determine the button's theme and
                interaction styling. Defaults to `"secondary"`.

        Returns:
            None.
        """
        super().__init__(text, parent)
        self._variant: str = variant
        self._hovered: bool = False
        self._pressed_state: bool = False
        self._border_visible: bool = True
        self._icon_left: bool = False
        self._menu_row: bool = False

        self._apply_qss(text)

        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the button after the application theme changes.

        Schedules the widget for repainting so its fill, border, focus ring, and
        other token-driven visual properties are rendered using the newly active
        theme.

        Args:
            _mode: Theme mode identifier emitted by the `themeChanged` signal.
                The value is not otherwise used by this handler.

        Returns:
            None.
        """
        self.update()

    def set_icon_left(self, on: bool = True) -> None:
        """Configure the button to display its icon on the left.

        The icon is positioned toward the leading side of the button while the
        label text remains centered. Additional padding is applied by the button's
        layout styling to keep the icon clear of the rounded corners.

        Args:
            on: `True` to enable left-aligned icon placement, or `False` to
                restore the default icon positioning.

        Returns:
            None.
        """
        self._icon_left = on
        self.update()

    def set_menu_row(self, on: bool = True) -> None:
        """Configure the button as a full-width, left-aligned menu row.

        Menu-row mode positions the icon at a fixed leading inset and places the
        text immediately after it rather than centering the text. This layout is
        intended for borderless popup menu items such as account or navigation
        actions.

        Enabling menu-row mode also enables left-aligned icon placement. The
        button's stylesheet is reapplied so the layout changes take effect
        immediately.

        Args:
            on: `True` to enable menu-row layout, or `False` to disable it.

        Returns:
            None.
        """
        self._menu_row = on
        if on:
            self._icon_left = True
        self._apply_qss(self.text())
        self.update()

    def set_variant(self, variant: str) -> None:
        """Change the button's visual variant at runtime.

        Updates the requested variant and schedules the button for repainting so
        the new variant styling is applied immediately. This can be used for
        transient state changes where a button changes appearance based on the
        current action or workflow state.

        Example:
            Change a delete action into a destructive confirmation state::

                btn.set_variant("destructive")
                btn.set_variant("destructive_outline")

        Args:
            variant: Name of the visual variant to apply.

        Returns:
            None.
        """
        self._variant = variant
        self.update()

    def set_border_visible(self, visible: bool) -> None:
        """Control whether the button's border and focus ring are displayed.

        When the border is hidden, keyboard focus is also disabled to prevent the
        button from drawing a focus ring that would otherwise extend beyond the
        borderless control. This is useful for compact actions placed inside
        another bordered container, where an additional button border would be
        visually redundant.

        Args:
            visible: `True` to display the button border, or `False` to hide
                the border and disable keyboard focus.

        Returns:
            None.
        """
        self._border_visible = visible
        if not visible:
            self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.update()

    def setEnabled(self, enabled: bool) -> None:
        """Set the button's enabled state and update its cursor.

        Disabled buttons use a forbidden cursor to communicate that the control
        cannot currently be activated. Enabled buttons restore the standard
        pointing-hand cursor. The button is then scheduled for repainting so its
        disabled or enabled visual styling is updated.

        Args:
            enabled: `True` to enable the button, or `False` to disable it.

        Returns:
            None.
        """
        super().setEnabled(enabled)
        self.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.ForbiddenCursor)
            if not enabled
            else QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.update()

    def _canonical_variant(self) -> str:
        """Resolve the configured variant to a canonical variant name.

        Variant aliases are translated using `_VARIANT_ALIASES`. If the
        resulting name is not recognized as a canonical variant, `"secondary"`
        is returned as the safe default.

        Returns:
            The canonical variant name used by the button's rendering and
            stylesheet logic.
        """
        v = _VARIANT_ALIASES.get(self._variant, self._variant)
        return v if v in _CANONICAL_VARIANTS else "secondary"

    def _apply_qss(self, text: str) -> None:
        """Apply the minimal stylesheet required for button content layout.

        Configures transparent backgrounds and borders while providing
        variant-aware padding, font family, font size, and font weight. The
        button's fill, border, focus ring, and other visual chrome are rendered
        separately by `paintEvent`.

        Icon-toolbar buttons use reduced horizontal padding. Other buttons use
        standard padding when text is present and remove padding for textless
        controls.

        Args:
            text: Current button text, used to determine whether content padding
                should be applied.

        Returns:
            None.
        """
        variant = self._canonical_variant()
        if variant == "icon_toolbar":
            h_pad, v_pad = 4, 8
        else:
            h_pad = 18 if text.strip() else 0
            v_pad = 9 if text.strip() else 0
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                padding: {v_pad}px {h_pad}px;
                font-family: {FONT_SANS_STACK};
                font-size: 13px;
                font-weight: 600;
            }}
        """)

    @staticmethod
    def _lighten(color: QtGui.QColor, percent: int) -> QtGui.QColor:
        """Lighten a color by the specified percentage.

        Uses Qt's `QColor.lighter` operation to increase the color's lightness
        relative to its original value.

        Args:
            color: Base color to lighten.
            percent: Percentage by which to increase the color's lightness.
                A value of `0` leaves the color unchanged.

        Returns:
            A new `QColor` with the requested increase in lightness.
        """
        return color.lighter(100 + percent)

    def _resolve_colors(self) -> dict:
        """Resolve colors and rendering properties for the current button state.

        Determines the button's fill, text, border, focus-ring, border width, and
        shadow configuration from the active theme tokens. The result is based on
        the canonical button variant and the current interaction state, including
        hover, pressed, enabled, and keyboard-focus states.

        The color configuration is resolved at paint time so theme changes are
        reflected immediately without maintaining separate light and dark color
        tables.

        Supported variants include `"primary"`, `"destructive"`,
        `"destructive_outline"`, `"ghost"`, `"ghost_danger"`,
        `"icon_toolbar"`, and `"secondary"`. Unrecognized variants fall back
        to the `"secondary"` styling.

        Returns:
            A dictionary containing the resolved rendering properties:

            * `fill`: `QColor` used for the button background.
            * `text`: `QColor` used for button text and content.
            * `border`: `QColor` used for the button border.
            * `border_width`: Width of the button border in pixels.
            * `ring`: Optional `QColor` used for the keyboard focus ring.
            * `shadow`: Whether the button should render a drop shadow.
        """
        tok = ThemeManager.instance().tokens()
        variant = self._canonical_variant()
        hovered = self._hovered
        pressed = self._pressed_state
        transparent = QtGui.QColor(0, 0, 0, 0)

        def c(key: str) -> QtGui.QColor:
            return QtGui.QColor(*tok[key])

        if not self.isEnabled():
            return {
                "fill": c("flat_surface2"),
                "text": c("flat_text_muted"),
                "border": c("flat_border"),
                "border_width": 1.0,
                "ring": None,
                "shadow": False,
            }

        focus_ring = self.hasFocus() and self._border_visible

        if variant == "primary":
            if pressed:
                fill = c("flat_accent_active")
            elif hovered:
                fill = c("flat_accent_hover")
            else:
                fill = c("flat_accent")
            return {
                "fill": fill,
                "text": c("flat_on_accent"),
                "border": fill,
                "border_width": 0.0,
                "ring": c("flat_accent_ring") if focus_ring else None,
                "shadow": True,
            }

        if variant == "destructive":
            base = c("flat_error")
            fill = base.darker(125) if pressed else (base.darker(110) if hovered else base)
            return {
                "fill": fill,
                "text": QtGui.QColor(255, 255, 255),
                "border": fill,
                "border_width": 0.0,
                "ring": c("flat_error_ring") if focus_ring else None,
                "shadow": True,
            }

        if variant == "destructive_outline":
            if pressed:
                fill = c("flat_error_weak").darker(105)
            elif hovered:
                fill = c("flat_error_weak")
            else:
                fill = transparent
            return {
                "fill": fill,
                "text": c("flat_error"),
                "border": c("flat_error"),
                "border_width": 1.0,
                "ring": c("flat_error_ring") if focus_ring else None,
                "shadow": False,
            }

        if variant == "ghost":
            if pressed:
                fill = c("flat_accent_weak").darker(105)
            elif hovered:
                fill = c("flat_accent_weak")
            else:
                fill = transparent
            return {
                "fill": fill,
                "text": c("flat_accent"),
                "border": transparent,
                "border_width": 0.0,
                "ring": c("flat_accent_ring") if focus_ring else None,
                "shadow": False,
            }

        if variant == "ghost_danger":
            if pressed:
                fill = c("flat_error_weak").darker(105)
            elif hovered:
                fill = c("flat_error_weak")
            else:
                fill = transparent
            return {
                "fill": fill,
                "text": c("flat_error"),
                "border": transparent,
                "border_width": 0.0,
                "ring": c("flat_error_ring") if focus_ring else None,
                "shadow": False,
            }

        if variant == "icon_toolbar":
            if hovered or pressed:
                fill = c("flat_accent_weak")
                text = c("flat_accent")
            else:
                fill = transparent
                text = c("flat_text_muted")
            return {
                "fill": fill,
                "text": text,
                "border": transparent,
                "border_width": 0.0,
                "ring": c("flat_accent_ring") if focus_ring else None,
                "shadow": False,
            }

        # "secondary" - also the fallback for any unrecognized variant name.
        if pressed:
            fill = c("flat_surface2").darker(105)
        elif hovered:
            fill = c("flat_surface2")
        else:
            fill = transparent
        return {
            "fill": fill,
            "text": c("flat_text"),
            "border": c("flat_border_strong"),
            "border_width": 1.0,
            "ring": c("flat_accent_ring") if focus_ring else None,
            "shadow": False,
        }

    def enterEvent(self, event) -> None:
        """Handle the mouse cursor entering the button.

        Updates the internal hover state and schedules the button for repainting
        so the hover-specific visual styling can be rendered.

        Args:
            event: Qt event generated when the mouse cursor enters the widget.

        Returns:
            None.
        """
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        """Handle the mouse cursor leaving the button.

        Clears the internal hover state and schedules the button for repainting so
        the normal visual styling can be restored.

        Args:
            event: Qt event generated when the mouse cursor leaves the widget.

        Returns:
            None.
        """
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def mousePressEvent(self, event) -> None:
        """Handle a mouse press on the button.

        Delegates the event to `QPushButton` and records the pressed state so
        the appropriate pressed-state styling can be rendered.

        Args:
            event: Mouse event generated when a mouse button is pressed over the
                widget.

        Returns:
            None.
        """
        super().mousePressEvent(event)
        self._pressed_state = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        """Handle a mouse release on the button.

        Delegates the event to `QPushButton` and clears the internal pressed
        state before scheduling a repaint.

        Args:
            event: Mouse event generated when a mouse button is released.

        Returns:
            None.
        """
        super().mouseReleaseEvent(event)
        self._pressed_state = False
        self.update()

    def focusInEvent(self, event) -> None:
        """Handle the button receiving keyboard focus.

        Delegates focus handling to the base class and schedules the button for
        repainting so its focus-ring styling can be displayed.

        Args:
            event: Qt focus event generated when the widget receives focus.

        Returns:
            None.
        """
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        """Handle the button losing keyboard focus.

        Delegates focus handling to the base class and schedules the button for
        repainting so any active focus-ring styling is removed.

        Args:
            event: Qt focus event generated when the widget loses focus.

        Returns:
            None.
        """
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the button surface and delegate content rendering to Qt.

        Renders the button's flat fill, border, focus ring, and optional drop
        shadow using the resolved variant and interaction-state colors. Standard
        Qt text and icon rendering is then used for normal buttons, while
        specialized layouts are handled manually for icon-toolbar and menu-row
        variants.

        Solid-fill variants such as `primary` and `destructive` receive a
        manually offset translucent shadow. A graphics effect is intentionally
        avoided because continuously repainted widgets can exhibit pixmap-cache
        ghosting when wrapped in `QGraphicsDropShadowEffect`.

        The method also handles buttons that temporarily have negligible height,
        such as controls during size animations, by delegating directly to the
        base implementation when the height is below the minimum rendering
        threshold.

        For buttons configured with `set_icon_left(True)`, the icon is rendered
        manually at a leading inset while Qt renders the text centered across the
        full button width.

        Args:
            event: Qt paint event provided when the widget needs to be repainted.
                The event is passed to the base implementation when the widget is
                too small to render its custom surface.

        Returns:
            None.
        """
        w, h = self.width(), self.height()
        if h < 4:
            super().paintEvent(event)
            return

        variant = self._canonical_variant()
        colors = self._resolve_colors()
        radius = _RADIUS_ICON_TOOLBAR if variant == "icon_toolbar" else _RADIUS

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # Subtle drop shadow under solid-fill variants
        if colors["shadow"]:
            tok = ThemeManager.instance().tokens()
            shadow_rect = QtCore.QRectF(0.0, 1.0, float(w), float(h))
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QBrush(QtGui.QColor(*tok["flat_shadow"])))
            p.drawRoundedRect(shadow_rect, radius, radius)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        border_width = colors["border_width"] if self._border_visible else 0.0
        paint_flat_surface(
            self,
            radius=radius,
            fill=colors["fill"],
            border=colors["border"],
            border_width=border_width,
            ring=colors["ring"] if self._border_visible else None,
            painter=p,
        )
        p.end()

        # Apply the resolved text color to the palette
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ButtonText, colors["text"])
        self.setPalette(pal)

        if variant == "icon_toolbar":
            self._paint_icon_toolbar_label(colors["text"])
            return

        if self._menu_row:
            self._paint_menu_row_label(colors["text"])
            return

        if self._icon_left and not self.icon().isNull():
            # Left-aligned icon, centered text.
            isz = self.iconSize()
            icon_pad = max(int(radius * 0.6), 12)
            icon_y = (h - isz.height()) // 2
            pm = self.icon().pixmap(isz)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            if not self.isEnabled():
                painter.setOpacity(0.45)
            painter.drawPixmap(icon_pad, icon_y, pm)
            painter.end()

            sp = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionButton()
            self.initStyleOption(opt)
            opt.icon = QtGui.QIcon()  # suppress the style's own icon draw
            opt.iconSize = QtCore.QSize(0, 0)
            sp.drawControl(QtWidgets.QStyle.CE_PushButtonLabel, opt)
            sp.end()
        else:
            sp = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionButton()
            self.initStyleOption(opt)
            sp.drawControl(QtWidgets.QStyle.CE_PushButtonLabel, opt)
            sp.end()

    def _paint_icon_toolbar_label(self, text_color: QtGui.QColor) -> None:
        """Paint the icon-toolbar button's vertically stacked content.

        Draws the button icon above its text label to provide the vertical
        icon-over-label layout required by the `"icon_toolbar"` variant. This
        layout is rendered manually because Qt's standard
        `CE_PushButtonLabel` style draws button icons and text horizontally.

        The icon is centered horizontally with a fixed top inset. The text is
        positioned below the icon with a small configurable gap. If no icon is
        configured, the label is drawn using the same top inset without reserving
        space for an icon.

        Disabled buttons are rendered with reduced opacity.

        Args:
            text_color: `QColor` used to render the button's text label.

        Returns:
            None.
        """
        w = self.width()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        if not self.isEnabled():
            p.setOpacity(0.45)

        icon = self.icon()
        label_h = p.fontMetrics().height()
        gap = 5

        if not icon.isNull():
            isz = self.iconSize()
            icon_x = (w - isz.width()) // 2
            icon_y = 8
            p.drawPixmap(icon_x, icon_y, icon.pixmap(isz))
            text_y = icon_y + isz.height() + gap
        else:
            text_y = 8

        p.setPen(QtGui.QPen(text_color))
        text_rect = QtCore.QRect(0, text_y, w, label_h)
        p.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignHCenter, self.text())
        p.end()

    def _paint_menu_row_label(self, text_color: QtGui.QColor) -> None:
        """Paint the icon and text for a left-aligned menu row.

        Renders the menu-row content manually rather than delegating to Qt's
        `CE_PushButtonLabel` pipeline. Direct painting provides precise control
        over the leading icon inset and text position, avoiding inconsistencies
        caused by the global stylesheet's handling of padding and icon-suppressed
        button labels.

        The icon is vertically centered at a fixed leading inset, followed by the
        label text with a defined gap. The text is vertically centered and
        left-aligned within the remaining button width.

        Disabled buttons are rendered with reduced opacity.

        Args:
            text_color: `QColor` used to render the menu-row text.

        Returns:
            None.
        """
        w, h = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        if not self.isEnabled():
            p.setOpacity(0.45)

        icon = self.icon()
        icon_pad = 12
        gap = 10
        text_x = icon_pad
        if not icon.isNull():
            isz = self.iconSize()
            icon_y = (h - isz.height()) // 2
            p.drawPixmap(icon_pad, icon_y, icon.pixmap(isz))
            text_x = icon_pad + isz.width() + gap

        p.setFont(make_qfont(pixel_size=13, weight=QtGui.QFont.DemiBold))
        p.setPen(QtGui.QPen(text_color))
        text_rect = QtCore.QRect(text_x, 0, w - text_x - 12, h)
        p.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )
        p.end()
