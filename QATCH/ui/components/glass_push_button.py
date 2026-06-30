"""
glass_push_button.py

A QPushButton subclass with a glass-morphism aesthetic and shimmer hover
animation, intentionally mirroring the visual language of GlassLineEdit.

Visual mapping
--------------
  GlassLineEdit             ->   GlassPushButton
  ─────────────────────────────────────────────
  focusInEvent              ->   enterEvent
  focusOutEvent             ->   leaveEvent
  _focused fill (alpha 100)     ->   _hovered fill (alpha 100)
  resting fill  (alpha 58)      ->   resting fill  (alpha 58)
  shimmer sweep on focus    ->   shimmer sweep on hover
  settled border (alpha 130)    ->   settled border (alpha 130)

Variants
--------
  "default"        Translucent white glass - secondary actions (Audit, Refresh, Back).
  "primary"        Vertical blue gradient - primary CTA (Add / confirm).
  "danger"         Red-tinted glass - destructive labelled actions (Delete).
  "danger_confirm" Solid red - delete-button confirmation state.
  "warning"        Amber-tinted glass - cautionary icon actions (Reset Password).
  "neutral"        Grey-tinted glass - cancel / row audit icon actions.

Usage
-----
    btn = GlassPushButton(" Add", variant="primary")
    btn.setIcon(QtGui.QIcon(path))
    btn.setIconSize(QtCore.QSize(18, 18))
    btn.setFixedHeight(34)               # height drives the pill radius

    btn_del = GlassPushButton("", variant="danger")
    btn_del.setFixedSize(28, 28)         # icon-only circle

    # Switch state at runtime (e.g. delete -> confirmation mode):
    btn_del.set_variant("danger_confirm")
    btn_del.set_variant("danger")        # restore
"""

from __future__ import annotations
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager

# ---------------------------------------------------------------------------
# Colour palettes
# All colour data stored as plain RGBA tuples to avoid constructing QColor
# objects at class-definition time (before any QApplication exists).
#
# Nested one level by theme mode ("light" / "dark") - see
# QATCH.ui.styles.tokens for the app-wide palette this mirrors. A variant's
# `text_role` name is mode-agnostic; what color it actually resolves to is
# looked up in _TEXT_COLORS[mode], so most variants below keep identical
# fill/border/shimmer values across modes (translucent overlays already read
# fine on either background) - only "default"/"neutral"/"ghost" (low-alpha
# washes that need a touch more presence on a near-black surface) and
# "primary" (brightened slightly for vibrancy) actually differ.
#
# fills:      (normal, hover, pressed) - background RGBA tuples.
#             None for "primary", which uses a vertical linear gradient.
# grad:       ((top, bot) normal, hover, pressed) - used only by "primary".
# border:     Resting border RGBA.
# sh_accent:  Shimmer gradient wing colour RGBA  (matches GlassLineEdit).
# sh_peak:    Shimmer gradient centre peak RGBA  (matches GlassLineEdit).
# text_role:  Key into _TEXT_COLORS[mode] for QPalette.ButtonText.
# ---------------------------------------------------------------------------
_LIGHT_PALETTES: dict[str, dict] = {
    "default": dict(
        fills=(
            (255, 255, 255, 58),  # normal
            (255, 255, 255, 100),  # hover
            (255, 255, 255, 140),  # pressed
        ),
        border=(255, 255, 255, 105),
        sh_accent=(185, 218, 248, 115),
        sh_peak=(255, 255, 255, 240),
        text_role="dark",
    ),
    # Primary: blue gradient fill, white shimmer.
    "primary": dict(
        fills=None,  # drawn as a vertical gradient
        grad=(
            ((45, 165, 250, 210), (15, 125, 210, 190)),  # normal
            ((65, 185, 255, 240), (25, 145, 230, 220)),  # hover
            ((15, 115, 200, 220), (5, 95, 160, 200)),  # pressed
        ),
        border=(255, 255, 255, 100),
        sh_accent=(185, 218, 248, 160),
        sh_peak=(255, 255, 255, 240),
        text_role="light",
    ),
    # Danger (labelled Delete button - restores here after confirmation).
    "danger": dict(
        fills=(
            (220, 53, 69, 26),  # ~0.10
            (220, 53, 69, 64),  # ~0.25
            (220, 53, 69, 102),  # ~0.40
        ),
        border=(220, 53, 69, 77),  # ~0.30
        sh_accent=(220, 53, 69, 160),
        sh_peak=(255, 160, 160, 240),
        text_role="danger",  # #B02A37
    ),
    # Danger-confirm (delete button while awaiting confirmation - solid red).
    "danger_confirm": dict(
        fills=(
            (220, 53, 69, 102),  # ~0.40  normal
            (220, 53, 69, 153),  # ~0.60  hover
            (220, 53, 69, 204),  # ~0.80  pressed
        ),
        border=(220, 53, 69, 204),
        sh_accent=(255, 160, 160, 200),
        sh_peak=(255, 220, 220, 255),
        text_role="light",
    ),
    # Warning (amber - Reset Password icon button).
    "warning": dict(
        fills=(
            (255, 193, 7, 31),  # ~0.12
            (255, 193, 7, 71),  # ~0.28
            (255, 193, 7, 115),  # ~0.45
        ),
        border=(255, 193, 7, 128),  # ~0.50
        sh_accent=(255, 193, 7, 160),
        sh_peak=(255, 240, 150, 240),
        text_role="dark",
    ),
    # Neutral (grey - Cancel / row-Audit icon button).
    "neutral": dict(
        fills=(
            (108, 117, 125, 31),  # ~0.12
            (108, 117, 125, 71),  # ~0.28
            (108, 117, 125, 115),  # ~0.45
        ),
        border=(108, 117, 125, 115),
        sh_accent=(108, 117, 125, 160),
        sh_peak=(200, 210, 220, 240),
        text_role="dark",
    ),
    # Ghost: near-invisible at rest, blue-accent wash on hover - for compact
    # actions packed inside an already-bordered container (set_border_visible
    # off), where the app's blue accent reads as "interactive" more clearly
    # than a generic white opacity bump.
    "ghost": dict(
        fills=(
            (10, 163, 230, 18),  # resting - barely-there tint
            (10, 163, 230, 60),  # hover
            (10, 163, 230, 100),  # pressed
        ),
        border=(10, 163, 230, 150),
        sh_accent=(185, 218, 248, 140),
        sh_peak=(255, 255, 255, 240),
        text_role="dark",
    ),
}

# Dark-mode variants. Saturated washes (danger/danger_confirm/warning) carry
# enough presence on their own and are reused as-is from _LIGHT_PALETTES
# below; only the low-alpha neutral-tone washes ("default"/"neutral"/
# "ghost") get a touch more fill/border alpha so they stay perceivable
# against a near-black surface, and "primary"'s gradient is brightened
# slightly for vibrancy. text_role names are unchanged - see _TEXT_COLORS.
_DARK_PALETTES: dict[str, dict] = {
    "default": dict(
        fills=(
            (255, 255, 255, 70),  # normal
            (255, 255, 255, 115),  # hover
            (255, 255, 255, 155),  # pressed
        ),
        border=(255, 255, 255, 130),
        sh_accent=(185, 218, 248, 130),
        sh_peak=(255, 255, 255, 245),
        text_role="dark",
    ),
    "primary": dict(
        fills=None,
        grad=(
            ((55, 175, 255, 220), (20, 130, 215, 200)),  # normal
            ((75, 195, 255, 245), (30, 150, 235, 230)),  # hover
            ((20, 120, 205, 230), (10, 100, 165, 210)),  # pressed
        ),
        border=(255, 255, 255, 120),
        sh_accent=(185, 218, 248, 170),
        sh_peak=(255, 255, 255, 245),
        text_role="light",
    ),
    "danger": _LIGHT_PALETTES["danger"],
    "danger_confirm": _LIGHT_PALETTES["danger_confirm"],
    "warning": _LIGHT_PALETTES["warning"],
    "neutral": dict(
        fills=(
            (140, 148, 156, 40),  # normal
            (140, 148, 156, 85),  # hover
            (140, 148, 156, 130),  # pressed
        ),
        border=(140, 148, 156, 135),
        sh_accent=(140, 148, 156, 175),
        sh_peak=(210, 218, 226, 245),
        text_role="dark",
    ),
    "ghost": dict(
        fills=(
            (45, 175, 240, 24),  # resting - barely-there tint
            (45, 175, 240, 70),  # hover
            (45, 175, 240, 115),  # pressed
        ),
        border=(45, 175, 240, 165),
        sh_accent=(185, 218, 248, 155),
        sh_peak=(255, 255, 255, 245),
        text_role="dark",
    ),
}

_PALETTES: dict[str, dict[str, dict]] = {"light": _LIGHT_PALETTES, "dark": _DARK_PALETTES}

_TEXT_COLORS: dict[str, dict[str, QtGui.QColor]] = {
    "light": {
        "light": QtGui.QColor(255, 255, 255),
        "dark": QtGui.QColor(51, 51, 51),  # #333
        "danger": QtGui.QColor(176, 42, 55),  # #B02A37
    },
    "dark": {
        "light": QtGui.QColor(255, 255, 255),
        # "dark" role here means "not light, not danger" - on a dark
        # surface that has to be a light color too, not literal dark text.
        "dark": QtGui.QColor(225, 230, 235),
        "danger": QtGui.QColor(255, 140, 140),
    },
}


class GlassPushButton(QtWidgets.QPushButton):
    """QPushButton with glass-morphism rendering and a shimmer sweep on hover.

    All fill, border, and shimmer effects are painted manually in paintEvent -
    no QSS is used for backgrounds or borders.  A minimal QSS string (padding
    and font-weight only) is applied so Qt's standard icon / text pipeline can
    render on top unimpeded.

    The shimmer algorithm is a direct port of GlassLineEdit's focus shimmer:
    same 12 ms tick, same 0.022 step, same 0.30 spread, same accent / peak
    colours for the "default" variant.  Hover maps to focus in the analogy.

    Attributes:
        _variant (str):       Active palette name.
        _shimmer_t (float):   Sweep progress 0.0 -> 1.0, reset on each hover entry.
        _hovered (bool):      True while the cursor is inside the widget.
        _pressed_state (bool):True while a mouse button is held down.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        variant: str = "default",
    ) -> None:
        super().__init__(text, parent)
        self._variant: str = variant
        self._shimmer_t: float = 0.0
        self._hovered: bool = False
        self._pressed_state: bool = False
        self._border_visible: bool = True

        # 12 ms / ~0.022 step -> matches GlassLineEdit tick exactly
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(12)
        self._timer.timeout.connect(self._tick)

        self._apply_text_color(variant)

        # Derive horizontal padding from whether the button has label text.
        # Icon-only buttons (text == "") get zero padding so the icon centres
        # naturally inside the widget's fixed size.
        h_pad = 14 if text.strip() else 0
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                border: none;
                font-weight: bold;
                padding: 0px {h_pad}px;
            }}
        """)

        # WA_Hover ensures enterEvent / leaveEvent fire correctly on all platforms
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Re-applies text color for the now-active mode and repaints with
        the matching light/dark fill/border/shimmer palette."""
        self._apply_text_color(self._variant)
        self.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_icon_left(self, on: bool = True) -> None:
        """Left-align the icon while keeping the label text centered.

        Padding accounts for the pill's rounded corner so the icon clears the
        curve. Use for labelled control buttons that want a leading glyph.
        """
        self._icon_left = on
        self.update()

    def set_variant(self, variant: str) -> None:
        """Switch the visual variant at runtime.

        Useful for transient state changes such as the delete button morphing
        into a confirmation circle::

            btn.set_variant("danger_confirm")
            btn.set_variant("danger")   # restore
        """
        self._variant = variant
        self._apply_text_color(variant)
        self.update()

    def set_border_visible(self, visible: bool) -> None:
        """Hide the glass border entirely while keeping the fill + shimmer
        hover effect - for compact actions that sit inside an already
        bordered container (e.g. the USB picker box), where drawing a border
        around the button too would be one border too many.

        Also drops keyboard focus when borderless: the native focus
        rectangle some styles draw around a focused QPushButton would look
        like a stray grey border reappearing on click, with nothing (no
        custom border) to frame it."""
        self._border_visible = visible
        if not visible:
            self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.update()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _active_palettes() -> dict:
        """Returns the variant->palette dict for the currently active
        light/dark theme."""
        return _PALETTES[ThemeManager.instance().mode().value]

    def _apply_text_color(self, variant: str) -> None:
        mode = ThemeManager.instance().mode().value
        palettes = self._active_palettes()
        role = palettes.get(variant, palettes["default"]).get("text_role", "dark")
        text_colors = _TEXT_COLORS[mode]
        color = text_colors.get(role, text_colors["dark"])
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ButtonText, color)
        self.setPalette(pal)

    @staticmethod
    def _c(rgba: tuple) -> QtGui.QColor:
        """Construct a QColor from a plain (r, g, b, a) tuple."""
        return QtGui.QColor(*rgba)

    # ------------------------------------------------------------------
    # Timer (shimmer animation)
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        """Increments shimmer progress and triggers a repaint - identical to
        GlassLineEdit._tick."""
        self._shimmer_t = min(1.0, self._shimmer_t + 0.022)
        self.update()
        if self._shimmer_t >= 1.0:
            self._timer.stop()

    # ------------------------------------------------------------------
    # Event overrides
    # ------------------------------------------------------------------
    def enterEvent(self, event) -> None:
        """Mouse enters: reset and start shimmer sweep - analogous to
        GlassLineEdit.focusInEvent."""
        super().enterEvent(event)
        self._hovered = True
        self._shimmer_t = 0.0
        self._timer.start()
        self.update()

    def leaveEvent(self, event) -> None:
        """Mouse leaves: stop sweep - analogous to GlassLineEdit.focusOutEvent."""
        super().leaveEvent(event)
        self._hovered = False
        self._timer.stop()
        self.update()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self._pressed_state = True
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        self._pressed_state = False
        self.update()

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the glass fill and animated border, then delegates icon /
        text rendering to Qt's standard CE_PushButtonLabel pipeline.

        Guard against zero-height during size animations (cancel buttons
        start at 0x0 and expand).
        """
        w, h = self.width(), self.height()
        if h < 4:
            # Widget is still animating open from 0x0 - nothing to draw yet.
            super().paintEvent(event)
            return

        palettes = self._active_palettes()
        pal = palettes.get(self._variant, palettes["default"])
        # Pill radius: height/2 − 1 px inset, identical to GlassLineEdit
        r = (h / 2.0) - 1.0
        rect = QtCore.QRectF(1.0, 1.0, w - 2.0, h - 2.0)

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtCore.Qt.NoPen)

        # ── Fill ─────────────────────────────────────────────────────
        state_idx = 2 if self._pressed_state else (1 if self._hovered else 0)

        if pal["fills"] is None:
            # Primary variant: vertical linear gradient
            stops = pal["grad"][state_idx]
            gf = QtGui.QLinearGradient(0, 0, 0, h)
            gf.setColorAt(0.0, self._c(stops[0]))
            gf.setColorAt(1.0, self._c(stops[1]))
            p.setBrush(QtGui.QBrush(gf))
        else:
            p.setBrush(QtGui.QBrush(self._c(pal["fills"][state_idx])))

        p.drawRoundedRect(rect, r, r)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        # ── Border / shimmer ─────────────────────────────────────────
        # Algorithm mirrors GlassLineEdit paintEvent exactly:
        # sweeping phase -> spread / peak gradient; settled phase -> solid accent.
        if self._border_visible:
            t = self._shimmer_t
            sh_accent = self._c(pal["sh_accent"])
            sh_peak = self._c(pal["sh_peak"])

            if self._hovered and t < 1.0:
                # Sweeping shimmer: bright peak chases t across the border
                spread = 0.30
                grad = QtGui.QLinearGradient(0.0, 0.0, float(w), 0.0)
                grad.setColorAt(0.0, sh_accent)

                pre = max(0.0, t - spread)
                if pre > 0.0:
                    grad.setColorAt(pre, sh_accent)

                grad.setColorAt(max(0.0, t - spread * 0.12), sh_peak)
                grad.setColorAt(min(1.0, t + spread * 0.12), sh_peak)

                post = min(1.0, t + spread)
                if post < 1.0:
                    grad.setColorAt(post, sh_accent)

                grad.setColorAt(1.0, sh_accent)
                p.setPen(QtGui.QPen(QtGui.QBrush(grad), 1.5))

            elif self._hovered:
                # Settled: accent colour at slightly elevated alpha (mirrors
                # GlassLineEdit's settled_color alpha 130 logic)
                settled = QtGui.QColor(sh_accent)
                settled.setAlpha(min(255, sh_accent.alpha() + 15))
                p.setPen(QtGui.QPen(settled, 1.5))

            else:
                # Resting border - same as GlassLineEdit's unfocused pen
                p.setPen(QtGui.QPen(self._c(pal["border"]), 1.0))

            p.drawRoundedRect(rect, r, r)
        p.end()

        # ── Icon + text rendering ────────────────────────────────────
        if getattr(self, "_icon_left", False) and not self.icon().isNull():
            # Left-aligned icon, centered text. The icon is inset from the left
            # edge by a padding that accounts for the pill's rounded corner so
            # it doesn't collide with the curve; the text stays centered across
            # the full button width (it may visually overlap the icon zone only
            # if the label is very long, which these control labels are not).
            isz = self.iconSize()
            icon_pad = max(int(r * 0.6), 12)  # clear the rounded corner
            icon_y = (h - isz.height()) // 2
            pm = self.icon().pixmap(isz)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            if not self.isEnabled():
                painter.setOpacity(0.45)
            painter.drawPixmap(icon_pad, icon_y, pm)
            painter.end()

            # Centered text via the style pipeline, with the icon suppressed so
            # it isn't drawn a second time (centered).
            sp = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionButton()
            self.initStyleOption(opt)
            opt.icon = QtGui.QIcon()  # suppress the style's own icon draw
            opt.iconSize = QtCore.QSize(0, 0)
            sp.drawControl(QtWidgets.QStyle.CE_PushButtonLabel, opt)
            sp.end()
        else:
            # ── Icon + text via Qt's standard rendering pipeline ─────────
            # CE_PushButtonLabel draws only the icon and text - no background or
            # border - so our custom-painted glass surface is not overwritten.
            sp = QtWidgets.QStylePainter(self)
            opt = QtWidgets.QStyleOptionButton()
            self.initStyleOption(opt)
            sp.drawControl(QtWidgets.QStyle.CE_PushButtonLabel, opt)
            sp.end()
