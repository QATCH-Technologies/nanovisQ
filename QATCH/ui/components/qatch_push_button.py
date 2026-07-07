"""
glass_push_button.py

A QPushButton subclass matching the app's flat control system (see
QATCH.ui.components.flat_paint). Also the single home for what used to be
two separate classes - the old `GlassPushButton` (glass-morphism, pill
radius) and `BorderlessActionButton` (flat text-link buttons) - since the
new design spec defines one button family with a `ghost` variant covering
the old borderless-link use case.

Variants
--------
  "primary"             Solid accent fill - primary CTA (Initialize, Save).
  "secondary"           Transparent fill, border_strong outline - default
                         action (Advanced, Refresh, Cancel, Back...).
  "ghost"                Transparent, accent-colored text, no border - quiet
                         inline links (Forgot Password?, per-field
                         Save/Reset/Default actions).
  "destructive"          Solid error fill, white text - hard-confirm delete.
  "destructive_outline"  Transparent, error-colored text + border - softer
                         labelled destructive action before confirmation.
  "ghost_danger"          Transparent, error-colored text, no border - quiet
                         destructive links (Sign Out).
  "icon_toolbar"          Vertical icon-above-label stack (not yet wired to
                         any call site - available for future toolbar work).

Old variant names ("default", "neutral", "danger", "danger_confirm") are
accepted as aliases and resolve to the canonical names above, so existing
call sites keep working unchanged. "warning" (unused anywhere in the app)
is not carried forward; an unrecognized variant name falls back to
"secondary".

Usage
-----
    btn = GlassPushButton(" Add", variant="primary")
    btn.setIcon(QtGui.QIcon(path))
    btn.setIconSize(QtCore.QSize(18, 18))
    btn.setFixedHeight(34)

    btn_link = GlassPushButton("Reset", variant="ghost")   # was BorderlessActionButton

    # Switch state at runtime (e.g. delete -> confirmation mode):
    btn_del.set_variant("destructive")
    btn_del.set_variant("destructive_outline")   # restore
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.fonts import FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager

_RADIUS = 7.0
_RADIUS_ICON_TOOLBAR = 8.0

# Old variant name -> canonical variant name. Unknown names (including the
# retired "warning") fall through to "secondary" at resolve time.
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
    """QPushButton with flat-design rendering, matching the app's flat
    control system.

    All fill, border, and focus-ring colors are resolved fresh at paint
    time from the active `flat_*` tokens (light/dark aware automatically -
    no separate light/dark palette table needed). A minimal QSS string
    (padding and font only) is applied so Qt's standard icon/text pipeline
    can render on top unimpeded.

    Attributes:
        _variant (str): Requested variant name (alias or canonical).
        _hovered (bool): True while the cursor is inside the widget.
        _pressed_state (bool): True while a mouse button is held down.
        _border_visible (bool): False suppresses the border stroke.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        variant: str = "secondary",
    ) -> None:
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
        self.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_icon_left(self, on: bool = True) -> None:
        """Left-align the icon while keeping the label text centered.

        Padding accounts for the corner radius so the icon clears the
        curve. Use for labelled control buttons that want a leading glyph.
        """
        self._icon_left = on
        self.update()

    def set_menu_row(self, on: bool = True) -> None:
        """Lays the button out as a full-width, left-aligned menu row: icon
        at a fixed inset, text immediately after it (not centered) - for
        borderless popup menu items (e.g. the account dropdown's "Manage
        Users" / "Sign Out" rows). Implies `set_icon_left(True)`.
        """
        self._menu_row = on
        if on:
            self._icon_left = True
        self._apply_qss(self.text())
        self.update()

    def set_variant(self, variant: str) -> None:
        """Switch the visual variant at runtime.

        Useful for transient state changes such as the delete button morphing
        into a confirmation state::

            btn.set_variant("destructive")
            btn.set_variant("destructive_outline")   # restore
        """
        self._variant = variant
        self.update()

    def set_border_visible(self, visible: bool) -> None:
        """Hide the border entirely while keeping the fill + hover effect -
        for compact actions that sit inside an already-bordered container
        (e.g. the USB picker box), where drawing a border around the button
        too would be one border too many.

        Also drops keyboard focus when borderless: the focus ring this
        widget would otherwise draw needs a border to sit outside of, and a
        borderless button living inside another bordered container should
        not introduce its own focus outline."""
        self._border_visible = visible
        if not visible:
            self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.update()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self.setCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.ForbiddenCursor)
            if not enabled
            else QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        )
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _canonical_variant(self) -> str:
        v = _VARIANT_ALIASES.get(self._variant, self._variant)
        return v if v in _CANONICAL_VARIANTS else "secondary"

    def _apply_qss(self, text: str) -> None:
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
                font-family: '{FONT_SANS_SEMIBOLD}';
                font-size: 13px;
            }}
        """)

    @staticmethod
    def _lighten(color: QtGui.QColor, percent: int) -> QtGui.QColor:
        return color.lighter(100 + percent)

    def _resolve_colors(self) -> dict:
        """Resolves fill/text/border/ring colors for the current variant and
        interaction state, read fresh from the active theme's flat_* tokens."""
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

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------
    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self._pressed_state = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._pressed_state = False
        self.update()

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat fill/border/focus-ring, then delegates icon/text
        rendering to Qt's standard CE_PushButtonLabel pipeline (or a manual
        vertical icon-above-label layout for the icon_toolbar variant).

        Guard against zero-height during size animations (cancel buttons
        start at 0x0 and expand).
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

        # Subtle drop shadow under solid-fill variants (primary/destructive),
        # drawn as a manually-offset translucent duplicate shape rather than
        # a QGraphicsDropShadowEffect - this widget is hover-repainted, and
        # wrapping it in a graphics effect risks the pixmap-caching ghosting
        # documented for other continuously-repainted custom widgets in this
        # app (see ui_controls.py's _PerspectiveAnimator docstring).
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

        # Apply the resolved text color to the palette so Qt's own label
        # drawing (below) picks it up.
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
            # Left-aligned icon, centered text. The icon is inset from the
            # left edge by a padding that accounts for the corner radius so
            # it doesn't collide with the curve; the text stays centered
            # across the full button width.
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
        """Manual vertical icon-above-label layout for variant="icon_toolbar".

        Qt's native CE_PushButtonLabel lays icon and text out horizontally;
        the flat spec's toolbar buttons stack them vertically, so this is
        drawn by hand instead of delegated.
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
        """Left-aligned icon + left-aligned text for variant=set_menu_row(True).

        Hand-painted rather than delegated to CE_PushButtonLabel + QSS
        text-align: Qt's stylesheet engine does not reliably resolve a
        shorthand `padding` against a differing effective left inset for
        icon-suppressed text the way this needs, so direct drawing is used
        instead - the same reasoning as `_paint_icon_toolbar_label`.
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

        font = QtGui.QFont(FONT_SANS_SEMIBOLD)
        font.setPixelSize(13)
        p.setFont(font)
        p.setPen(QtGui.QPen(text_color))
        text_rect = QtCore.QRect(text_x, 0, w - text_x - 12, h)
        p.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )
        p.end()
