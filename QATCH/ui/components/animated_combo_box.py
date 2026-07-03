from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.fonts import FONT_SANS, FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css

_RADIUS = 7.0


class _FlatItemDelegate(QtWidgets.QStyledItemDelegate):
    """Fixed-height row delegate that paints the flat design's per-item
    states directly: hover wash, selected wash + accent text + checkmark.

    A full `paint()` override (not QSS) is used because Qt Style Sheets
    can't conditionally draw an extra glyph (the selected-row checkmark),
    and this repo's flat control system already migrates chrome that QSS
    can't express (focus rings, etc.) to paintEvent-style drawing elsewhere
    - this keeps the same approach for the popup rows.
    """

    _ROW_RADIUS = 6.0

    def __init__(
        self, row_height: int, combo: "AnimatedComboBox", parent: Optional[QtCore.QObject] = None
    ) -> None:
        super().__init__(parent)
        self._row_height = row_height
        self._combo = combo

    def sizeHint(
        self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex
    ) -> QtCore.QSize:
        size = super().sizeHint(option, index)
        size.setHeight(self._row_height)
        return size

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        tok = ThemeManager.instance().tokens()
        rect = QtCore.QRectF(option.rect)
        is_selected = index.row() == self._combo.currentIndex()
        is_hover = bool(option.state & QtWidgets.QStyle.State_MouseOver)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)

        row_rect = rect.adjusted(0.0, 1.0, 0.0, -1.0)
        if is_selected:
            painter.setBrush(QtGui.QColor(*tok["flat_accent_weak"]))
            painter.drawRoundedRect(row_rect, self._ROW_RADIUS, self._ROW_RADIUS)
            text_color = QtGui.QColor(*tok["flat_accent"])
            font = QtGui.QFont(FONT_SANS_SEMIBOLD)
        elif is_hover:
            painter.setBrush(QtGui.QColor(*tok["flat_surface2"]))
            painter.drawRoundedRect(row_rect, self._ROW_RADIUS, self._ROW_RADIUS)
            text_color = QtGui.QColor(*tok["flat_text"])
            font = QtGui.QFont(FONT_SANS)
        else:
            text_color = QtGui.QColor(*tok["flat_text"])
            font = QtGui.QFont(FONT_SANS)
        font.setPixelSize(13)
        painter.setFont(font)

        text_right_pad = 26 if is_selected else 10
        text_rect = rect.adjusted(10.0, 0.0, -text_right_pad, 0.0)
        painter.setPen(text_color)
        painter.drawText(
            text_rect, QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            index.data(),
        )

        if is_selected:
            check_pen = QtGui.QPen(text_color, 1.8)
            check_pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
            check_pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
            painter.setPen(check_pen)
            cx = rect.right() - 18.0
            cy = rect.center().y()
            path = QtGui.QPainterPath()
            path.moveTo(cx - 4.0, cy)
            path.lineTo(cx - 1.0, cy + 3.0)
            path.lineTo(cx + 5.0, cy - 4.0)
            painter.drawPath(path)

        painter.restore()


class _RoundedPopup(QtWidgets.QFrame):
    """A frameless, translucent, rounded popup that hosts the combo's list view.

    Hosted in a widget we own (not Qt's `QComboBoxPrivateContainer`) so we can
    translucent + round + animate it safely. The slide is driven by animating the
    window `mask` (a growing rounded-rect `QRegion`): this clips the real
    top-level popup window frame-by-frame and forces repaints, which animating a
    height property on a `Qt::Popup` does not do reliably. Styling is scoped by
    objectName because a Qt Style Sheet *type* selector matches the Qt class name,
    not a Python subclass name.
    """

    closed = QtCore.pyqtSignal()

    def __init__(self, radius: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._radius = radius
        self._reveal = 1.0  # 0..1 fraction of height currently revealed
        self._allow_hide = False  # gate for hideEvent veto (see close_animated)
        self._closing_now = False  # True while owner runs the slide-close
        self.setObjectName("roundedComboPopup")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(0)

        self._view = QtWidgets.QListView(self)
        self._view.setObjectName("roundedComboView")
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setResizeMode(QtWidgets.QListView.Adjust)
        self._view.setUniformItemSizes(True)
        self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        # ::item:hover only fires with mouse tracking enabled (native combo
        # popups turn this on internally; an owned view does not). Enable it on
        # both the view and its viewport, and show the pointing hand.
        self._view.setMouseTracking(True)
        self._view.viewport().setMouseTracking(True)
        self._view.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._view.viewport().setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self._view)
        self._lay = lay

    def size_to_rows(self, rows: int, row_h: int) -> int:
        """Size the popup (and pin the view) to exactly `rows` of `row_h`.

        The view is given a fixed height equal to its content so the surrounding
        layout cannot hand it a slack row; the window height is that plus the
        view's frame/viewport margins and the layout margins. Returns the
        total window height so the owner can place + animate it.
        """
        view = self._view
        content_h = rows * row_h
        # The viewport can sit inside a small frame/margin even with NoFrame;
        # account for the difference between the view's outer size and its
        # viewport so we neither clip nor pad.
        margins = view.contentsMargins()
        extra = margins.top() + margins.bottom()
        view.setFixedHeight(content_h + extra)
        # Layout margins add the inner breathing room around the rows.
        lm = self._lay.contentsMargins()
        total = content_h + extra + lm.top() + lm.bottom()
        self.setFixedHeight(total)
        return total

    def view(self) -> QtWidgets.QListView:
        return self._view

    def apply_theme(self, font: QtGui.QFont, row_h: int) -> None:
        tok = ThemeManager.instance().tokens()
        bg = tok_css(tok["flat_surface"])
        border = tok_css(tok["flat_border"])
        r = self._radius
        inner = max(r - 2, 0)
        self._view.setFont(font)
        self.setStyleSheet(f"""
            QFrame#roundedComboPopup {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: {r}px;
            }}
            QListView#roundedComboView {{
                background-color: {bg};
                border: none;
                outline: none;
                border-radius: {inner}px;
                padding: 0px;
            }}
        """)

    # ---- reveal (mask animation) -------------------------------------------

    def get_reveal(self) -> float:
        return self._reveal

    def set_reveal(self, value: float) -> None:
        self._reveal = max(0.0, min(1.0, value))
        self._apply_mask()

    reveal = QtCore.pyqtProperty(float, fget=get_reveal, fset=set_reveal)

    def _apply_mask(self) -> None:
        w = self.width()
        full_h = self.height()
        h = max(1, int(round(full_h * self._reveal)))
        path = QtGui.QPainterPath()
        # Round only within the currently revealed slice so the growing edge
        # stays crisp; corners round once fully open.
        path.addRoundedRect(QtCore.QRectF(0, 0, w, h), self._radius, self._radius)
        self.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_mask()

    # ---- dismissal --------------------------------------------------------
    # A Qt::Popup hides itself the instant it loses focus (outside click) or an
    # item is clicked, which would kill the slide-close before it runs. We veto
    # that hide and emit `dismiss` so the owner can run the animation and hide
    # us itself. `allow_hide` is set True only for that final animated hide.

    dismiss = QtCore.pyqtSignal()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        if not self._allow_hide:
            # Veto: keep the window up so the owner can animate the close. Only
            # re-show if we're not already sliding shut (avoids a full-reveal
            # flash when an item click both closes us and auto-hides the popup).
            event.ignore()
            if not self._closing_now:
                QtCore.QTimer.singleShot(0, self.show)
            self.dismiss.emit()
            return
        self.closed.emit()
        super().hideEvent(event)

    def set_closing(self, closing: bool) -> None:
        """Owner tells us a slide-close is in progress so hideEvent won't
        re-show us mid-animation."""
        self._closing_now = closing

    def close_animated(self) -> None:
        """Called by the owner once the slide-close has finished."""
        self._allow_hide = True
        self.hide()
        self._allow_hide = False
        self._closing_now = False


class AnimatedComboBox(QtWidgets.QComboBox):
    """A QComboBox that slides its rounded drop-down open/closed and spins its
    chevron, styled to match the app's flat control system. The list is
    hosted in a `_RoundedPopup` we own; the slide is a mask reveal (0..1) so
    it actually repaints on a `Qt::Popup` window.
    """

    _POPUP_RADIUS = 8
    _ROW_HEIGHT = 30  # tweak to match the combo button's own item height

    def __init__(self, icon_path: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnimatedComboBox")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self._hovered = False

        # --- Custom popup (owned by us) ---
        self._popup = _RoundedPopup(self._POPUP_RADIUS, self)
        self._popup.view().setModel(self.model())
        self._row_delegate = _FlatItemDelegate(self._ROW_HEIGHT, self, self._popup.view())
        self._popup.view().setItemDelegate(self._row_delegate)
        self._popup.view().clicked.connect(self._on_item_clicked)
        self._popup.closed.connect(self._on_popup_closed)
        self._popup.dismiss.connect(self._on_dismiss_requested)

        self._popup_open = False
        self._closing = False
        self._just_closed_ms = 0

        # --- Arrow Icon Setup ---
        self.arrow_lbl = QtWidgets.QLabel(self)
        self.arrow_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.arrow_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.arrow_lbl.setStyleSheet("background: transparent; border: none;")

        self._icon_source = QtGui.QPixmap(icon_path).scaled(
            15,
            15,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._arrow_override: Optional[QtGui.QColor] = None
        self._current_angle = 0.0
        self._base_pixmap = self._tinted_pixmap(self._icon_source, self._current_arrow_color())
        self.arrow_lbl.setPixmap(self._base_pixmap)

        # --- Chevron spin animation ---
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._on_spin_frame)

        # --- Popup slide animation (mask reveal 0..1) ---
        self._slide = QtCore.QPropertyAnimation(self._popup, b"reveal", self)
        self._slide.setDuration(180)
        self._slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._slide.finished.connect(self._on_slide_finished)

        self._apply_text_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    # ---- public API ---------------------------------------------------------

    def set_arrow_color(self, color: Optional[QtGui.QColor]) -> None:
        self._arrow_override = QtGui.QColor(color) if color is not None else None
        self._retint_arrow()

    # ---- text / font ----------------------------------------------------------

    def _apply_text_qss(self) -> None:
        tok = ThemeManager.instance().tokens()
        text_color = tok["flat_text_muted"] if not self.isEnabled() else tok["flat_text"]
        r, g, b, a = text_color
        self.setStyleSheet(f"""
            QComboBox#AnimatedComboBox {{
                color: rgba({r}, {g}, {b}, {a});
                font-family: '{FONT_SANS}';
                font-size: 13px;
                padding-left: 12px;
            }}
        """)

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802
        super().setEnabled(enabled)
        self._apply_text_qss()
        self.update()

    # ---- arrow tinting ------------------------------------------------------

    @staticmethod
    def _tinted_pixmap(source: QtGui.QPixmap, color: QtGui.QColor) -> QtGui.QPixmap:
        dst = QtGui.QPixmap(source.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(dst)
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        painter.fillRect(dst.rect(), color)
        painter.end()
        return dst

    def _current_arrow_color(self) -> QtGui.QColor:
        if self._arrow_override is not None:
            return self._arrow_override
        tok = ThemeManager.instance().tokens()
        key = "flat_accent" if self._popup_open else "flat_text_muted"
        return QtGui.QColor(*tok[key])

    def _retint_arrow(self) -> None:
        self._base_pixmap = self._tinted_pixmap(self._icon_source, self._current_arrow_color())
        self._on_spin_frame(self._current_angle)

    def _on_theme_changed(self, _mode: Optional[str] = None) -> None:
        self._apply_text_qss()
        self._retint_arrow()
        self._popup.apply_theme(self.font(), self._ROW_HEIGHT)
        self.update()

    # ---- geometry helpers ---------------------------------------------------

    def _visible_rows(self) -> int:
        return min(max(self.count(), 1), self.maxVisibleItems())

    def _place_popup(self, full_h: int) -> None:
        # Align the popup's left edge to the combo and make it at least as wide
        # as the combo box (the 1px frame border would otherwise make it read
        # slightly narrower than the box).
        top_left = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        width = self.width() + 2  # cover the combo's own border on both sides
        self._popup.setGeometry(top_left.x() - 1, top_left.y(), width, full_h)

    # ---- open / close -------------------------------------------------------

    def showPopup(self) -> None:  # noqa: N802 (Qt override)
        if self._popup_open:
            return
        if QtCore.QDateTime.currentMSecsSinceEpoch() - self._just_closed_ms < 200:
            return

        self._popup_open = True
        self._closing = False
        self._retint_arrow()
        self._popup.apply_theme(self.font(), self._ROW_HEIGHT)

        # Deterministic: size to N rows of the delegate's known height, pinning
        # the view so the layout cannot add a slack row at the bottom.
        full_h = self._popup.size_to_rows(self._visible_rows(), self._ROW_HEIGHT)
        self._place_popup(full_h)
        self._popup.set_reveal(0.0)
        self._popup.show()

        idx = self.model().index(self.currentIndex(), self.modelColumn())
        self._popup.view().setCurrentIndex(idx)

        self._slide.stop()
        self._slide.setStartValue(0.0)
        self._slide.setEndValue(1.0)
        self._slide.start()
        self._spin_to(180.0)
        self.update()

    def hidePopup(self) -> None:  # noqa: N802 (Qt override)
        if not self._popup_open or self._closing:
            return
        self._closing = True
        self._popup.set_closing(True)
        self._slide.stop()
        self._slide.setStartValue(self._popup.get_reveal())
        self._slide.setEndValue(0.0)
        self._slide.start()
        self._spin_to(0.0)

    def _on_slide_finished(self) -> None:
        if self._closing:
            self._closing = False
            self._popup_open = False
            self._just_closed_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
            self._popup.close_animated()  # the only path allowed to truly hide
            self._popup.set_reveal(1.0)  # reset for next open
            self._retint_arrow()
            self.update()

    def _on_dismiss_requested(self) -> None:
        # Popup tried to auto-hide (outside click / item click). Run the
        # animated close instead; hidePopup no-ops if we're already closing.
        self.hidePopup()

    def _on_popup_closed(self) -> None:
        # Emitted only on the final (allowed) hide now, so just keep the arrow
        # in sync as a safety net.
        if self._popup_open and not self._closing:
            self._popup_open = False
            self._just_closed_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
            self._spin_to(0.0)
            self._popup.set_reveal(1.0)
            self._retint_arrow()
            self.update()

    def _on_item_clicked(self, index: QtCore.QModelIndex) -> None:
        self.setCurrentIndex(index.row())
        self.activated.emit(index.row())
        self.hidePopup()

    # ---- chevron frame ------------------------------------------------------

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.arrow_lbl.setGeometry(self.width() - 28, 0, 24, self.height())

    def _spin_to(self, target_angle: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._current_angle)
        self._anim.setEndValue(target_angle)
        self._anim.start()

    def _on_spin_frame(self, angle: float) -> None:
        self._current_angle = angle
        transform = QtGui.QTransform().rotate(angle)
        rotated = self._base_pixmap.transformed(
            transform, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.arrow_lbl.setPixmap(rotated)

    # ---- event overrides ------------------------------------------------------

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    # ---- painting ------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat fill/border/focus-ring, then delegates the
        current-item label to Qt's native CE_ComboBoxLabel pipeline.

        Deliberately does NOT call `super().paintEvent(event)`: QComboBox's
        base implementation draws the full CC_ComboBox complex control
        (panel + arrow + label), and even with QSS "background: transparent"
        that panel-drawing step erases whatever was painted immediately
        before it (a QStyleSheetStyle quirk - the "transparent" background is
        applied with a replacing, not blending, composition step). Drawing
        only CE_ComboBoxLabel directly - the same technique GlassPushButton
        uses for CE_PushButtonLabel - sidesteps the panel/arrow drawing
        (and its erase) entirely.
        """
        tok = ThemeManager.instance().tokens()

        if not self.isEnabled():
            fill = QtGui.QColor(*tok["flat_surface2"])
            border = QtGui.QColor(*tok["flat_border"])
            ring = None
        elif self._popup_open or self.hasFocus():
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

        opt = QtWidgets.QStyleOptionComboBox()
        self.initStyleOption(opt)
        sp = QtWidgets.QStylePainter(self)
        sp.drawControl(QtWidgets.QStyle.CE_ComboBoxLabel, opt)
