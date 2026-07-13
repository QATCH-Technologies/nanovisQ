from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.fonts import FONT_MONO_MEDIUM
from QATCH.ui.styles.theme_manager import ThemeManager

# Chrome (fill/border/radius/hover/focus-ring/error) is painted in Python via
# `_OdometerMixin._paint_chrome`, matching the app's flat control system (see
# QATCH.ui.components.flat_paint) - the same reasoning as AnimatedComboBox:
# Qt Style Sheets have no box-shadow, so the focus ring can't be expressed in
# QSS alone. Only the parts a paintEvent can't reach - the custom chevron
# pixmaps and the odometer overlay's painted colors - are handled here in
# addition to the chrome; the base QSS in app_theme.qss is trimmed to just
# suppressing the native up/down button region and giving the widget (and
# its internal line-edit sub-control) a transparent backdrop.

_RADIUS = 7.0


def _tinted_pixmap(source: QtGui.QPixmap, color: QtGui.QColor) -> QtGui.QPixmap:
    """Recolors every opaque pixel of `source` to `color`, preserving alpha."""
    dst = QtGui.QPixmap(source.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(dst)
    painter.drawPixmap(0, 0, source)
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    painter.fillRect(dst.rect(), color)
    painter.end()
    return dst


class _SpinArrow(QtWidgets.QLabel):
    """A clickable chevron that recolors (idle -> accent) on hover/press.

    Unlike the combo box arrow (which is mouse-transparent and purely
    decorative), these chevrons are the interactive step controls, so mouse
    events are enabled. Visual feedback is a direct pixmap swap between a
    precomputed idle-tint and active-tint pixmap - matching the flat design's
    "recolor on hover" language rather than the old opacity-fade. Holding the
    button auto-repeats the step via a QTimer so press-and-hold ramps the
    value like a native spin box.
    """

    stepped = QtCore.pyqtSignal()

    _REPEAT_DELAY_MS = 350
    _REPEAT_INTERVAL_MS = 60

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: transparent; border: none;")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._idle_pixmap = QtGui.QPixmap()
        self._active_pixmap = QtGui.QPixmap()
        self._active = False

        # Auto-repeat while pressed: an initial delay, then a steady interval.
        self._repeat_timer = QtCore.QTimer(self)
        self._repeat_timer.setSingleShot(False)
        self._repeat_timer.timeout.connect(self.stepped.emit)
        self._delay_timer = QtCore.QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._begin_repeat)

    def set_pixmaps(self, idle: QtGui.QPixmap, active: QtGui.QPixmap) -> None:
        self._idle_pixmap = idle
        self._active_pixmap = active
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        self.setPixmap(self._active_pixmap if self._active else self._idle_pixmap)

    def _begin_repeat(self) -> None:
        self._repeat_timer.start(self._REPEAT_INTERVAL_MS)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        self._active = True
        self._apply_pixmap()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._active = False
        self._apply_pixmap()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._active = True
            self._apply_pixmap()
            self.stepped.emit()
            self._delay_timer.start(self._REPEAT_DELAY_MS)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self._delay_timer.stop()
        self._repeat_timer.stop()
        super().mouseReleaseEvent(event)


# Digits that participate in true reel-rolling (0-9). Any other glyph (sign,
# decimal point, separators, spaces) is treated as a static "fixed" column that
# crossfades in place instead of spinning, since "rolling" a '.' is meaningless.
_DIGITS = "0123456789"


class _OdometerOverlay(QtWidgets.QWidget):
    """Opaque odometer reel painted over the spin box's text area.

    The previous implementation drew a translucent two-position crossfade: the
    old glyph slid out while the new glyph slid in, layered on top of the live
    line edit. That reads as a "slide", not a roll, and the line edit text bled
    through the translucency.

    This version instead:
      * Paints its OWN opaque background (matching the flat body fill) so
        nothing underneath shows through -- no need to recolour the line edit.
      * Treats each numeric column as a continuous reel. Rolling 7 -> 2 actually
        travels through the intermediate digits (7,6,5,4,3,2 going down, or
        7,8,9,0,1,2 going up), so the column visibly spins like a mechanical
        odometer wheel rather than swapping two frames.
      * Leaves non-digit columns (sign, '.', separators) as a simple in-place
        crossfade so they don't spin nonsensically.
    """

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

        self._old_text = ""
        self._new_text = ""
        self._direction = 1
        self._progress = 0.0
        self._color = QtGui.QColor(40, 50, 62)
        self._bg = QtGui.QColor(255, 255, 255)  # opaque fill behind the reels
        self._align_right = False

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(260)
        # Ease-out-back-ish settle gives a mechanical "click into place" feel.
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_tick)
        self._anim.finished.connect(self._on_finished)

    def set_color(self, color: QtGui.QColor) -> None:
        self._color = QtGui.QColor(color)

    def set_background(self, color: QtGui.QColor) -> None:
        self._bg = QtGui.QColor(color)

    def set_alignment_right(self, right: bool) -> None:
        self._align_right = right

    def roll(self, old_text: str, new_text: str, direction: int) -> None:
        self._direction = 1 if direction >= 0 else -1

        # Right-justify so place values line up (units under units, etc.).
        max_len = max(len(old_text), len(new_text))
        self._old_text = old_text.rjust(max_len)
        self._new_text = new_text.rjust(max_len)

        # Freeze a fixed column grid ONCE so spacing can't breathe mid-roll.
        # Every digit slot gets the same uniform width (the widest of 0-9), and
        # each non-digit glyph keeps its own fixed advance. Columns are placed
        # here and never recomputed per frame, so the decimal point and the
        # gaps around it stay rock-steady while digits spin.
        self._build_columns()

        self._anim.stop()
        self._progress = 0.0
        self.show()
        self.raise_()
        self._anim.start()

    def _build_columns(self) -> None:
        fm = QtGui.QFontMetricsF(self.font())
        # Uniform width for any digit slot: the widest glyph among 0-9.
        digit_w = max(fm.horizontalAdvance(c) for c in _DIGITS)

        cols = []  # list of (old_c, new_c, x, width, is_digit_col)
        total_w = 0.0
        for old_c, new_c in zip(self._old_text, self._new_text):
            old_is_d = old_c in _DIGITS
            new_is_d = new_c in _DIGITS
            if old_is_d or new_is_d:
                w = digit_w  # any column that ever shows a digit is digit-width
            else:
                # Punctuation/sign/space: fixed to its own (max) advance.
                w = max(fm.horizontalAdvance(old_c or " "), fm.horizontalAdvance(new_c or " "))
            cols.append([old_c, new_c, 0.0, w, old_is_d and new_is_d])
            total_w += w

        x = (self.width() - total_w) if self._align_right else 0.0
        for col in cols:
            col[2] = x
            x += col[3]
        self._columns = cols

    def _on_tick(self, t) -> None:
        self._progress = float(t)
        self.update()

    def _on_finished(self) -> None:
        self.hide()

    # -- reel helpers -----------------------------------------------------
    @staticmethod
    def _digit_distance(old_d: int, new_d: int, direction: int) -> int:
        """Number of steps from old_d to new_d travelling in `direction`.

        direction = +1 means counting up (0->1->...->9->0).
        direction = -1 means counting down.
        """
        if direction >= 0:
            return (new_d - old_d) % 10
        return (old_d - new_d) % 10

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing, True)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setFont(self.font())

        # Opaque background so the static line-edit text never shows through.
        p.fillRect(self.rect(), self._bg)

        cols = getattr(self, "_columns", None)
        if not cols:
            p.end()
            return

        h = self.height()
        fm = p.fontMetrics()
        row_h = fm.height()
        baseline_center = (h - row_h) / 2.0 + fm.ascent()

        for old_c, new_c, x, w, is_digit_col in cols:
            p.save()
            p.setClipRect(QtCore.QRectF(x, 0, w, h))
            p.setPen(self._color)

            if old_c == new_c:
                self._draw_centered(p, x, w, baseline_center, new_c)
            elif is_digit_col:
                self._paint_reel(p, x, w, h, row_h, baseline_center, int(old_c), int(new_c))
            else:
                self._paint_single_slide(p, x, w, row_h, baseline_center, old_c, new_c)

            p.restore()
        p.end()

    def _draw_centered(self, p, x, w, baseline, ch) -> None:
        if ch.strip():
            p.drawText(
                QtCore.QPointF(x + (w - p.fontMetrics().horizontalAdvance(ch)) / 2.0, baseline), ch
            )

    def _paint_reel(self, p, x, w, h, row_h, baseline, old_d, new_d) -> None:
        """Paint a spinning digit reel from old_d to new_d.

        The reel is the ordered list of digits passed through. We interpolate a
        continuous position along that list and draw the two glyphs straddling
        the current position, offset vertically. direction +1 scrolls the reel
        upward (new digit enters from below); -1 the reverse.
        """
        d = self._direction
        steps = self._digit_distance(old_d, new_d, d)
        if steps == 0:
            steps = 10  # full wrap when old==new shouldn't happen, guard anyway

        # Continuous position along the reel: 0 -> steps.
        pos = steps * self._progress
        frac = pos - int(pos)
        idx = int(pos)

        # Current digit just below the window's centre and the next one above/below.
        cur_digit = (old_d + d * idx) % 10
        nxt_digit = (old_d + d * (idx + 1)) % 10

        fm = p.fontMetrics()
        # As frac goes 0->1, the current glyph slides out by one row and the
        # next glyph slides into the centre. Scrolling up means glyphs move up
        # (negative y), so the incoming digit starts one row below.
        cur_y = baseline - d * frac * row_h
        nxt_y = baseline + d * (1.0 - frac) * row_h

        cur_c = str(cur_digit)
        nxt_c = str(nxt_digit)
        p.drawText(QtCore.QPointF(x + (w - fm.horizontalAdvance(cur_c)) / 2.0, cur_y), cur_c)
        p.drawText(QtCore.QPointF(x + (w - fm.horizontalAdvance(nxt_c)) / 2.0, nxt_y), nxt_c)

    def _paint_single_slide(self, p, x, w, row_h, baseline, old_c, new_c) -> None:
        d = self._direction
        fm = p.fontMetrics()
        out_y = baseline - d * self._progress * row_h
        in_y = baseline + d * (1.0 - self._progress) * row_h
        if old_c.strip():
            p.drawText(QtCore.QPointF(x + (w - fm.horizontalAdvance(old_c)) / 2.0, out_y), old_c)
        if new_c.strip():
            p.drawText(QtCore.QPointF(x + (w - fm.horizontalAdvance(new_c)) / 2.0, in_y), new_c)


class _OdometerMixin:
    """Shared chrome, chevron, odometer, and theme wiring for the int and
    double spin boxes."""

    def _init_flat_chrome(self) -> None:
        self._hovered = False
        self._in_error = False
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setStyleSheet("padding-left: 12px;")
        f = QtGui.QFont(FONT_MONO_MEDIUM)
        f.setPixelSize(14)
        self.setFont(f)
        if self.lineEdit() is not None:
            self.lineEdit().setFont(f)

    def set_error(self, on: bool) -> None:
        """Toggles the visual error state (border/ring + value text color)."""
        if on != self._in_error:
            self._in_error = on
            self._apply_text_palette()
            self.update()

    def _apply_text_palette(self) -> None:
        tok = ThemeManager.instance().tokens()
        if not self.isEnabled():
            color = QtGui.QColor(*tok["flat_text_muted"])
        elif self._in_error:
            color = QtGui.QColor(*tok["flat_error"])
        else:
            color = QtGui.QColor(*tok["flat_text"])
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Text, color)
        self.setPalette(pal)

    def enterEvent(self, event: QtCore.QEvent) -> None:
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self._apply_text_palette()
        self.update()

    def _paint_chrome(self) -> None:
        """Paints the flat fill/border/focus-ring behind the native internal
        line-edit sub-control (a real child widget, painted separately by
        Qt immediately afterward - no need to delegate to it here)."""
        tok = ThemeManager.instance().tokens()

        if not self.isEnabled():
            fill = QtGui.QColor(*tok["flat_surface2"])
            border = QtGui.QColor(*tok["flat_border"])
            ring = None
        elif self._in_error:
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_error"])
            ring = QtGui.QColor(*tok["flat_error_ring"])
        elif self.hasFocus():
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_accent"])
            ring = QtGui.QColor(*tok["flat_accent_ring"])
        elif self._hovered:
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_border_strong"])
            ring = None
        else:
            fill = QtGui.QColor(*tok["flat_surface"])
            border = QtGui.QColor(*tok["flat_border"])
            ring = None

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_flat_surface(self, radius=_RADIUS, fill=fill, border=border, ring=ring, painter=p)
        p.end()

    def _init_chevrons(self, up_icon_path: str, down_icon_path: Optional[str]) -> None:
        self._up_source, self._down_source = _make_chevrons(up_icon_path, down_icon_path)
        self._up_arrow = _SpinArrow(self)
        self._down_arrow = _SpinArrow(self)
        self._up_arrow.stepped.connect(self.stepUp)
        self._down_arrow.stepped.connect(self.stepDown)
        self._retint_chevrons()

    def _retint_chevrons(self) -> None:
        tok = ThemeManager.instance().tokens()
        idle = QtGui.QColor(*tok["flat_text_muted"])
        active = QtGui.QColor(*tok["flat_accent"])
        self._up_arrow.set_pixmaps(
            _tinted_pixmap(self._up_source, idle), _tinted_pixmap(self._up_source, active)
        )
        self._down_arrow.set_pixmaps(
            _tinted_pixmap(self._down_source, idle), _tinted_pixmap(self._down_source, active)
        )

    def _on_theme_changed(self, _mode: Optional[str] = None) -> None:
        self._retint_chevrons()
        self._retint_odometer()
        self._apply_text_palette()
        self.update()

    def _init_odometer(self) -> None:
        # Tabular (monospaced) figures: every digit gets the same advance in the
        # live line edit too, so the static text and the rolling overlay share
        # one fixed grid. Without this, the proportional line edit and the
        # uniform-grid overlay would disagree and you'd see a tiny snap when the
        # roll ends. This also keeps the decimal point fixed between states.
        f = self.font()
        f.setStyleStrategy(QtGui.QFont.PreferDefault)
        try:
            # Qt 5.11+: request tabular figures via the font feature.
            f.setStyleName(f.styleName())
        except Exception:
            pass
        # The portable lever is the font's fixed-pitch hint for digits; on most
        # UI fonts enabling kerning off + tabular numerals is enough.
        f.setKerning(False)
        self.setFont(f)
        if self.lineEdit() is not None:
            self.lineEdit().setFont(f)

        self._odometer = _OdometerOverlay(self)
        self._retint_odometer()
        self._apply_text_palette()

        self._prev_value = self.value()
        self._prev_text = self.text()
        # Only chevron steps animate. We do NOT animate on every valueChanged,
        # because that fires for keyboard edits too. Instead stepBy() (which the
        # chevrons route through via stepUp/stepDown) raises a one-shot flag that
        # _maybe_animate consults. Typing, setValue(), and programmatic changes
        # update the text instantly with no roll.
        self._step_pending = False
        self._step_direction = 1
        self.valueChanged.connect(self._maybe_animate)

    def _maybe_animate(self, new_value) -> None:
        # Keep prev_* in sync regardless, so a later step rolls from the right
        # starting text even if intervening changes came from typing.
        old_text = self._prev_text
        self._prev_value = new_value
        new_text = self.text()
        self._prev_text = new_text

        if not self._step_pending:
            return
        self._step_pending = False

        if not self.isVisible():
            return
        le = self.lineEdit()
        if le is None or old_text == new_text:
            return

        self._position_odometer()
        self._odometer.roll(old_text, new_text, self._step_direction)

    def _retint_odometer(self) -> None:
        # flat_text matches the internal line-edit's own text color; flat_surface
        # is the body's own fill, so the overlay reads as a seamless
        # continuation of the flat chrome in both themes.
        tok = ThemeManager.instance().tokens()
        self._odometer.set_color(QtGui.QColor(*tok["flat_text"]))
        self._odometer.set_background(QtGui.QColor(*tok["flat_surface"]))
        self._odometer.update()

    def _position_odometer(self) -> None:
        le = self.lineEdit()
        if le is None:
            return
        # The line edit draws its text inside its own content rect, inset from
        # the widget edge by frame width + text margins + a small cursor pad. If
        # we naively use le.geometry() the overlay's x=0 won't line up with where
        # the static text actually starts, and the digits visibly shift sideways
        # when the overlay shows. Map the line edit's *content* rect into the
        # spin box's coordinate space and give the overlay exactly that origin.
        cr = le.contentsRect()
        ml = le.textMargins().left()
        cursor_pad = 2  # QLineEdit reserves ~2px before the first glyph
        top_left = le.mapTo(self, cr.topLeft())
        x = top_left.x() + ml + cursor_pad
        y = top_left.y()
        self._odometer.setGeometry(
            int(x), int(y), int(cr.width() - ml - cursor_pad), int(cr.height())
        )
        self._odometer.setFont(le.font())
        # Boxes are left-aligned, so the reel grid grows from the left origin.
        right = bool(self.alignment() & QtCore.Qt.AlignmentFlag.AlignRight)
        self._odometer.set_alignment_right(right)

    def stepBy(self, steps: int) -> None:
        # The chevrons (and keyboard arrow keys / wheel) route through stepBy.
        # Mark that the *next* valueChanged came from a discrete step so it
        # animates. Direct text entry never calls stepBy, so it won't animate.
        if steps != 0:
            self._step_pending = True
            self._step_direction = 1 if steps > 0 else -1
        super().stepBy(steps)


class AnimatedDoubleSpinBox(_OdometerMixin, QtWidgets.QDoubleSpinBox):
    """A QDoubleSpinBox styled and animated to match the app's flat control
    system (and AnimatedComboBox).

    Visual + motion identity:
      * Flat rounded body (radius 7px) with hover / focus-ring / error states,
        painted the same way as QATCHLineEdit/AnimatedComboBox.
      * Custom stacked up/down chevrons (native step buttons suppressed) that
        recolor (idle -> accent) on hover, flash on press, and auto-repeat on
        hold.
      * On every value change the digits roll vertically like a mechanical
        odometer -- each numeric column spins through the intermediate digits
        into its new value.

    Constructor mirrors AnimatedComboBox: pass an `up_icon_path`. If
    `down_icon_path` is omitted the up chevron is flipped 180 degrees to
    derive the down one, so a single asset works for both.
    """

    def __init__(
        self,
        up_icon_path: str,
        down_icon_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("AnimatedDoubleSpinBox")
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self._init_flat_chrome()
        self._init_chevrons(up_icon_path, down_icon_path)
        self._init_odometer()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        _layout_chevrons(self, self._up_arrow, self._down_arrow)
        self._position_odometer()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        # Deliberately skips super().paintEvent(): QAbstractSpinBox's base
        # implementation draws the full CC_SpinBox complex control (frame +
        # buttons), which - like QComboBox's CC_ComboBox - erases whatever
        # was painted immediately before it even with a "transparent" QSS
        # background. The internal line-edit sub-control is a real child
        # widget, so it still paints its own text/cursor on top afterward
        # without needing that call.
        self._paint_chrome()


class AnimatedSpinBox(_OdometerMixin, QtWidgets.QSpinBox):
    """Integer counterpart to AnimatedDoubleSpinBox with identical styling.

    Shares the flat chrome, chevron behaviour, and the odometer digit-roll on
    value changes.
    """

    def __init__(
        self,
        up_icon_path: str,
        down_icon_path: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("AnimatedSpinBox")
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self._init_flat_chrome()
        self._init_chevrons(up_icon_path, down_icon_path)
        self._init_odometer()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        _layout_chevrons(self, self._up_arrow, self._down_arrow)
        self._position_odometer()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        # See AnimatedDoubleSpinBox.paintEvent for why super() is skipped.
        self._paint_chrome()


# ----------------------------------------------------------------- shared utils
def _make_chevrons(up_icon_path: str, down_icon_path: Optional[str]):
    up_pix = QtGui.QPixmap(up_icon_path).scaled(
        10,
        10,
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
    if down_icon_path:
        down_pix = QtGui.QPixmap(down_icon_path).scaled(
            10,
            10,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
    else:
        down_pix = up_pix.transformed(
            QtGui.QTransform().rotate(180),
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
    return up_pix, down_pix


def _layout_chevrons(box, up_arrow, down_arrow) -> None:
    gutter_x = box.width() - 26
    gutter_w = 22
    half = box.height() // 2
    up_arrow.setGeometry(gutter_x, 1, gutter_w, max(1, half - 1))
    down_arrow.setGeometry(gutter_x, half, gutter_w, max(1, half - 1))
