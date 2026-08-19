"""
QATCH.ui.components.animated_combo_box.py

Themed and animated combo box and popup components.

Provides the custom combo box implementation and supporting widgets used by
the application's flat control system. The module replaces Qt's native
combo-box popup with an application-owned rounded popup so its appearance,
interaction states, and animations can be controlled consistently across
platforms.

The main public widget is :class:`AnimatedComboBox`. The remaining helper
classes are implementation details used to render its popup, suggestion
rows, and animated controls.

The module intentionally relies on direct `QPainter` rendering for
color-critical surfaces and item states. Qt Style Sheets remain useful for
simple child-widget styling, but are avoided where platform-specific native
styles can interfere with custom rendering.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-19
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_SANS_STACK, make_qfont

_RADIUS = 7.0


class _FlatItemDelegate(QtWidgets.QStyledItemDelegate):
    """Paint fixed-height combo-box popup rows using the flat design system.

    Provides custom rendering for individual combo-box items, including
    hover and selected states, accent-colored selected text, and a checkmark
    for the currently selected item.

    A custom :meth:`paint` implementation is used instead of Qt Style Sheets
    because QSS cannot conditionally render the additional checkmark glyph.
    This also keeps popup-row rendering consistent with the rest of the
    flat control system, which uses explicit painting for visual elements
    that cannot be reliably expressed through QSS.

    Attributes:
        _row_height: Fixed height assigned to each popup row.
        _combo: Combo box whose current index determines the selected row.
    """

    _ROW_RADIUS = 6.0

    def __init__(
        self,
        row_height: int,
        combo: AnimatedComboBox,
        parent: QtCore.QObject | None = None,
    ) -> None:
        """Initialize the flat combo-box item delegate.

        Args:
            row_height: Fixed height of each popup row, in pixels.
            combo: Combo box whose current selection is used when rendering
                selected rows.
            parent: Optional parent Qt object.
        """
        super().__init__(parent)
        self._row_height = row_height
        self._combo = combo

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        """Return the fixed size hint for a popup row.

        The width is inherited from the base delegate while the height is
        replaced with the configured fixed row height.

        Args:
            option: Style options describing the item being measured.
            index: Model index for the item being measured.

        Returns:
            A size hint with the standard item width and configured row
            height.
        """
        size = super().sizeHint(option, index)
        size.setHeight(self._row_height)
        return size

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        """Paint a combo-box popup row according to its current state.

        Selected rows receive an accent-colored background, accent-colored
        semibold text, and a checkmark. Hovered but unselected rows receive
        the secondary flat-surface background. Normal rows use the standard
        flat text color without a background treatment.

        Args:
            painter: Painter used to render the item.
            option: Style options containing the item's geometry and current
                interaction state.
            index: Model index containing the item's display data.
        """
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
            font = make_qfont(weight=QtGui.QFont.DemiBold)
        elif is_hover:
            painter.setBrush(QtGui.QColor(*tok["flat_surface2"]))
            painter.drawRoundedRect(row_rect, self._ROW_RADIUS, self._ROW_RADIUS)
            text_color = QtGui.QColor(*tok["flat_text"])
            font = make_qfont()
        else:
            text_color = QtGui.QColor(*tok["flat_text"])
            font = make_qfont()
        font.setPixelSize(13)
        painter.setFont(font)

        text_right_pad = 26 if is_selected else 10
        text_rect = rect.adjusted(10.0, 0.0, -text_right_pad, 0.0)
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
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


class _CompleterRowDelegate(QtWidgets.QStyledItemDelegate):
    """Paint QCompleter suggestion rows using the flat control styling.

    Renders completer suggestions with the same hover and selected-state
    backgrounds and typography used by :class:`_FlatItemDelegate` for
    combo-box popup rows. Unlike combo-box rows, completer suggestions do
    not display a checkmark because they represent live suggestions rather
    than the currently selected value.

    A custom :meth:`paint` implementation is used instead of QSS item-view
    selection rules. On Windows, the native style can ignore stylesheet
    `::item` and `::item:selected` pseudo-states, causing selection to
    use the operating system's native highlight color rather than the
    application's flat design tokens. Direct painting avoids this
    platform-specific behavior.

    Attributes:
        _ROW_HEIGHT: Fixed height of each completer suggestion row.
        _ROW_RADIUS: Corner radius used for hover and selected backgrounds.
    """

    _ROW_HEIGHT = 30
    _ROW_RADIUS = 6.0

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        """Return the fixed size hint for a completer suggestion row.

        The width is inherited from the base delegate while the height is
        fixed to :attr:`_ROW_HEIGHT`.

        Args:
            option: Style options describing the item being measured.
            index: Model index for the item being measured.

        Returns:
            A size hint using the standard item width and fixed row height.
        """
        size = super().sizeHint(option, index)
        size.setHeight(self._ROW_HEIGHT)
        return size

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        """Paint a completer suggestion according to its interaction state.

        Selected rows use the flat accent background and semibold accent text.
        Hovered rows use the secondary flat surface and standard text color.
        Normal rows use only the standard flat text color.

        Args:
            painter: Painter used to render the suggestion row.
            option: Style options containing the row geometry and interaction
                state.
            index: Model index containing the suggestion text.
        """
        tok = ThemeManager.instance().tokens()
        rect = QtCore.QRectF(option.rect)
        is_selected = bool(option.state & QtWidgets.QStyle.State_Selected)
        is_hover = bool(option.state & QtWidgets.QStyle.State_MouseOver)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)

        row_rect = rect.adjusted(2.0, 1.0, -2.0, -1.0)
        if is_selected:
            painter.setBrush(QtGui.QColor(*tok["flat_accent_weak"]))
            painter.drawRoundedRect(row_rect, self._ROW_RADIUS, self._ROW_RADIUS)
            text_color = QtGui.QColor(*tok["flat_accent"])
            font = make_qfont(weight=QtGui.QFont.DemiBold)
        elif is_hover:
            painter.setBrush(QtGui.QColor(*tok["flat_surface2"]))
            painter.drawRoundedRect(row_rect, self._ROW_RADIUS, self._ROW_RADIUS)
            text_color = QtGui.QColor(*tok["flat_text"])
            font = make_qfont()
        else:
            text_color = QtGui.QColor(*tok["flat_text"])
            font = make_qfont()
        font.setPixelSize(13)
        painter.setFont(font)

        text_rect = rect.adjusted(10.0, 0.0, -10.0, 0.0)
        painter.setPen(text_color)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            index.data(),
        )
        painter.restore()


class _RoundedPopup(QtWidgets.QFrame):
    """Display a themed, frameless, rounded popup containing a list view.

    Provides a custom top-level popup for combo-box suggestions or selections,
    avoiding Qt's internal `QComboBoxPrivateContainer` so that the popup's
    rounded geometry, translucency, styling, and animation can be controlled
    explicitly.

    The popup reveal animation is implemented by animating the window mask
    rather than its height. Animating a `Qt.Popup` height does not reliably
    force the platform window to repaint frame-by-frame, whereas a growing
    rounded-rectangle mask clips the actual top-level window and triggers the
    required repaints.

    The popup's background and border are painted directly by
    :meth:`paintEvent` using theme colors rather than relying on QSS. This is
    intentional because translucent, frameless popup windows do not
    consistently composite QSS background and border styles across Windows
    configurations. Direct painting writes the themed chrome into the
    widget's raster buffer while `WA_TranslucentBackground` remains useful
    for anti-aliased rounded corners on platforms that support it.

    Styling is scoped using explicit object names because Qt Style Sheet type
    selectors resolve against the underlying Qt class name rather than the
    Python subclass name.

    Attributes:
        closed: Signal emitted when the popup has completed closing.
        _radius: Corner radius used for the popup's rounded geometry.
        _reveal: Current vertical reveal fraction, ranging from `0.0` to
            `1.0`.
        _allow_hide: Internal flag controlling whether a hide event is allowed
            to proceed during animated closing.
        _closing_now: Indicates that the popup is currently performing its
            close animation.
        _bg_color: Current popup background color.
        _border_color: Current popup border color.
        _view: List view displaying the popup's items.
        _lay: Layout containing the list view.
    """

    closed = QtCore.pyqtSignal()

    def __init__(
        self,
        radius: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the rounded popup and its list view.

        Configures the popup as a frameless, transient `Qt.Popup` window and
        creates an embedded :class:`QListView` with compact, borderless
        presentation and mouse tracking enabled for custom item hover
        rendering.

        Args:
            radius: Corner radius of the popup in pixels.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._radius = radius
        self._reveal = 1.0  # 0..1 fraction of height currently revealed
        self._allow_hide = False  # gate for hideEvent veto (see close_animated)
        self._closing_now = False  # True while owner runs the slide-close
        self._bg_color = QtGui.QColor(255, 255, 255)
        self._border_color = QtGui.QColor(200, 203, 206)
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
        """Size the popup to exactly fit a specified number of rows.

        The list view receives a fixed height corresponding to the requested
        number of rows. View margins and layout margins are included in the
        resulting popup height so that rows are neither clipped nor padded
        with unintended extra space.

        Args:
            rows: Number of visible rows the popup should contain.
            row_h: Height of each row in pixels.

        Returns:
            Total popup height in pixels after accounting for view and layout
            margins.
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
        """Return the list view hosted by the popup.

        Returns:
            The popup's embedded :class:`QListView`.
        """
        return self._view

    def apply_theme(self, font: QtGui.QFont, row_h: int) -> None:
        """Apply the current theme and typography to the popup.

        Updates the popup's background and border colors from the active theme,
        applies the supplied font to the list view, and styles the child
        `QListView` with the themed surface color.

        The popup border intentionally uses the `popup_border` token rather than
        `flat_border` because the popup is rendered as a top-level floating
        window and needs sufficient contrast against arbitrary content behind it.

        Row text and selection colors are rendered by the item delegate rather
        than QSS. The viewport is therefore explicitly updated after the theme
        change to ensure already-rendered rows are repainted immediately.

        Args:
            font: Font to apply to the popup's list view.
            row_h: Standard row height associated with the popup. The value is
                retained by the caller for sizing; it is not directly used by
                this method.
        """
        tok = ThemeManager.instance().tokens()
        bg = tok_css(tok["flat_surface"])
        # popup_border, not flat_border
        self._bg_color = QtGui.QColor(*tok["flat_surface"])
        self._border_color = QtGui.QColor(*tok["popup_border"])
        inner = max(self._radius - 2, 0)
        self._view.setFont(font)
        # Frame fill/border are painted in paintEvent.
        self.setStyleSheet(f"""
            QListView#roundedComboView {{
                background-color: {bg};
                border: none;
                outline: none;
                border-radius: {inner}px;
                padding: 0px;
            }}
        """)
        self.update()
        # The row text/highlight colors come from _FlatItemDelegate.paint(),
        # not QSS, so changing this stylesheet alone doesn't repaint them
        self._view.viewport().update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the popup's rounded background and border.

        Draws the popup frame directly using the currently configured background
        and border colors. Anti-aliasing is enabled to produce smooth rounded
        corners and a clean one-pixel border.

        Args:
            event: Qt paint event generated when the widget requires repainting.
        """
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setBrush(QtGui.QBrush(self._bg_color))
        p.setPen(QtGui.QPen(self._border_color, 1.0))
        p.drawRoundedRect(rect, self._radius, self._radius)
        p.end()

    def get_reveal(self) -> float:
        """Return the popup's current reveal progress.

        Returns:
            Reveal fraction in the range `0.0` to `1.0`, where `0.0` is
            fully hidden and `1.0` is fully revealed.
        """
        return self._reveal

    def set_reveal(self, value: float) -> None:
        """Set the popup's reveal progress and update its window mask.

        The supplied value is clamped to the valid `0.0`-`1.0` range before
        the popup mask is regenerated.

        Args:
            value: Desired reveal fraction, where `0.0` is fully hidden and
                `1.0` is fully revealed.
        """
        self._reveal = max(0.0, min(1.0, value))
        self._apply_mask()

    reveal = QtCore.pyqtProperty(float, fget=get_reveal, fset=set_reveal)

    def _apply_mask(self) -> None:
        """Apply a rounded mask corresponding to the current reveal progress.

        The mask exposes only the currently revealed portion of the popup. The
        rounded corners are applied to the visible slice so the leading edge of
        the reveal remains crisp while the popup is animating.

        The mask is regenerated whenever the reveal fraction or popup dimensions
        change.
        """
        w = self.width()
        full_h = self.height()
        h = max(1, int(round(full_h * self._reveal)))
        path = QtGui.QPainterPath()
        # Round only within the currently revealed slice so the growing edge
        # stays crisp; corners round once fully open.
        path.addRoundedRect(QtCore.QRectF(0, 0, w, h), self._radius, self._radius)
        self.setMask(QtGui.QRegion(path.toFillPolygon().toPolygon()))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Update the popup mask after its dimensions change.

        Recalculates the visible mask using the current reveal fraction so that
        the popup remains correctly clipped after a resize.

        Args:
            event: Qt resize event generated after the popup dimensions change.
        """
        super().resizeEvent(event)
        self._apply_mask()

    dismiss = QtCore.pyqtSignal()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        """Handle popup hide requests and coordinate animated dismissal.

        Prevents the popup from being hidden immediately when a normal hide is
        requested so the owner can perform the slide-close animation first. When
        hiding is not explicitly allowed, the event is ignored and the popup is
        scheduled to be shown again unless a close animation is already in
        progress.

        When the owner has explicitly allowed the popup to hide, the
        :attr:`closed` signal is emitted and the normal Qt hide-event processing
        is allowed to continue.

        Args:
            event: Qt hide event generated when the popup is being hidden.
        """
        if not self._allow_hide:
            # Veto: keep the window up so the owner can animate the close. Only
            # re-show if we're not already sliding shut.
            event.ignore()
            if not self._closing_now:
                QtCore.QTimer.singleShot(0, self.show)
            self.dismiss.emit()
            return
        self.closed.emit()
        super().hideEvent(event)

    def set_closing(self, closing: bool) -> None:
        """Set whether the popup is currently performing a close animation.

        While a close animation is active, :meth:`hideEvent` does not attempt to
        re-show the popup after a hide request. This prevents the popup from
        briefly flashing back to its fully revealed state during animated
        dismissal.

        Args:
            closing: `True` when a slide-close animation is in progress;
                `False` when the popup is no longer closing.
        """
        self._closing_now = closing

    def close_animated(self) -> None:
        """Allow the popup to complete its hide operation after closing.

        Temporarily enables normal hiding so that the owner can finish the
        slide-close animation without :meth:`hideEvent` vetoing the final
        `hide()` call. Internal closing state is reset after the popup has been
        hidden.
        """
        self._allow_hide = True
        self.hide()
        self._allow_hide = False
        self._closing_now = False


class AnimatedComboBox(QtWidgets.QComboBox):
    """Provide an animated, theme-aware combo box with a custom popup.

    Extends :class:`QComboBox` with a rounded drop-down popup that animates
    open and closed using a mask reveal, plus an animated chevron that rotates
    with the popup state.

    The drop-down list is hosted in an application-owned
    :class:`_RoundedPopup` rather than Qt's internal combo-box popup. This
    allows the popup's rounded geometry, direct-painted chrome, hover and
    selection rendering, and open/close animation to be controlled
    consistently across platforms.

    The popup reveal is animated through the popup's `reveal` property
    rather than its window height. This forces reliable repainting of the
    top-level `Qt.Popup` window during the animation.

    Attributes:
        arrow_lbl: Label used to display the animated combo-box arrow icon.
        _popup: Custom rounded popup containing the combo-box list view.
        _row_delegate: Delegate responsible for rendering popup rows.
        _popup_open: Whether the custom popup is currently considered open.
        _closing: Whether the popup is currently performing a close
            animation.
        _just_closed_ms: Timestamp or state marker used to suppress
            immediately repeated popup interactions after closing.
        _icon_source: Original arrow icon pixmap.
        _arrow_override: Optional explicit arrow color override.
        _current_angle: Current rotation angle of the arrow icon in degrees.
        _base_pixmap: Current unrotated, tinted arrow pixmap.
        _anim: Animation controlling the chevron rotation.
        _slide: Property animation controlling the popup's reveal mask.

    Class Attributes:
        _POPUP_RADIUS: Corner radius of the custom popup in pixels.
        _ROW_HEIGHT: Height of each popup item in pixels.
    """

    _POPUP_RADIUS = 8
    _ROW_HEIGHT = 30

    def __init__(
        self,
        icon_path: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the animated combo box.

        Creates the custom popup and its item delegate, configures the
        interactive arrow label, loads and tints the source icon, and
        initializes the chevron and popup slide animations.

        Args:
            icon_path: Path to the source arrow/chevron icon.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("AnimatedComboBox")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self._hovered = False
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

        # Arrow Icon Setup
        self.arrow_lbl = QtWidgets.QLabel(self)
        self.arrow_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.arrow_lbl.setStyleSheet("background: transparent; border: none;")
        # Not transparent-for-mouse-events, and explicitly wired to toggle
        # the popup itself: once `setEditable(True)` is on, Qt's internal
        # QLineEdit child covers nearly the whole box and swallows clicks
        # before they ever reach QComboBox's own arrow-subcontrol hit test,
        # so the native "click anywhere opens the popup" behavior silently
        # stops working. Handling it here instead makes the arrow a
        # reliable, editable-or-not click target for showPopup/hidePopup.
        self.arrow_lbl.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.arrow_lbl.mousePressEvent = self._on_arrow_clicked

        self._icon_source = QtGui.QPixmap(icon_path).scaled(
            15,
            15,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._arrow_override: QtGui.QColor | None = None
        self._current_angle = 0.0
        self._base_pixmap = self._tinted_pixmap(self._icon_source, self._current_arrow_color())
        self.arrow_lbl.setPixmap(self._base_pixmap)

        # Chevron spin animation
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._on_spin_frame)

        # Popup slide animation
        self._slide = QtCore.QPropertyAnimation(self._popup, b"reveal", self)
        self._slide.setDuration(180)
        self._slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._slide.finished.connect(self._on_slide_finished)

        self._apply_text_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def set_arrow_color(self, color: Optional[QtGui.QColor]) -> None:
        self._arrow_override = QtGui.QColor(color) if color is not None else None
        self._retint_arrow()

    def _apply_text_qss(self) -> None:
        """Apply theme-aware text styling to the combo box and editor.

        Updates the combo box text color, font, and left padding using the active
        theme tokens. Disabled controls use the muted text color while enabled
        controls use the standard flat text color.

        When the combo box is editable, the displayed value is rendered by its
        internal :class:`QLineEdit` rather than by the combo box itself. The line
        edit is therefore explicitly styled so entered text remains readable
        across theme changes, particularly in dark mode. Its selection colors and
        placeholder palette role are also synchronized with the active theme.

        The placeholder color is configured through `QPalette.PlaceholderText`
        instead of the Qt Style Sheet `::placeholder` pseudo-element because
        the latter is not consistently supported across Qt 5 versions.
        """
        tok = ThemeManager.instance().tokens()
        text_color = tok["flat_text_muted"] if not self.isEnabled() else tok["flat_text"]
        r, g, b, a = text_color
        self.setStyleSheet(f"""
            QComboBox#AnimatedComboBox {{
                color: rgba({r}, {g}, {b}, {a});
                font-family: {FONT_SANS_STACK};
                font-size: 13px;
                padding-left: 12px;
            }}
        """)
        if self.isEditable():
            line_edit = self.lineEdit()
            if line_edit is not None:
                line_edit.setStyleSheet(f"""
                    QLineEdit {{
                        background: transparent;
                        border: none;
                        color: rgba({r}, {g}, {b}, {a});
                        selection-background-color: {tok_css(tok["flat_accent_weak"])};
                        selection-color: {tok_css(tok["flat_accent"])};
                    }}
                """)
                pr, pg, pb, pa = tok["flat_text_muted"]
                palette = line_edit.palette()
                palette.setColor(QtGui.QPalette.PlaceholderText, QtGui.QColor(pr, pg, pb, pa))
                line_edit.setPalette(palette)

    def setEnabled(self, enabled: bool) -> None:
        """Enable or disable the combo box and refresh its text styling.

        Reapplies theme-aware text colors after the enabled state changes so
        disabled controls immediately use the muted text color.

        Args:
            enabled: `True` to enable the combo box; `False` to disable it.
        """
        super().setEnabled(enabled)
        self._apply_text_qss()
        self.update()

    def setEditable(self, editable: bool) -> None:
        """Enable or disable text editing and refresh editor styling.

        Reapplies the combo box theme immediately after the editable state
        changes. This is necessary because enabling editability can create or
        replace the internal :class:`QLineEdit`, which must be themed as soon as
        it becomes available.

        Args:
            editable: `True` to make the combo box editable; `False` to use
                selection-only behavior.
        """
        super().setEditable(editable)
        self._apply_text_qss()

    @staticmethod
    def _tinted_pixmap(
        source: QtGui.QPixmap,
        color: QtGui.QColor,
    ) -> QtGui.QPixmap:
        """Create a color-tinted copy of a pixmap while preserving its alpha.

        Uses `SourceAtop` composition so the supplied color is applied only
        where the source pixmap contains visible pixels, preserving the original
        icon's transparency.

        Args:
            source: Source pixmap whose visible pixels should be tinted.
            color: Color applied to the visible pixels.

        Returns:
            A new pixmap with the same dimensions as `source` and the requested
            tint color.
        """
        dst = QtGui.QPixmap(source.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(dst)
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        painter.fillRect(dst.rect(), color)
        painter.end()
        return dst

    def _current_arrow_color(self) -> QtGui.QColor:
        """Return the color that should currently be applied to the arrow icon.

        An explicitly configured arrow-color override takes precedence over the
        active theme. Otherwise, the arrow uses the accent color while the popup
        is open and the muted text color while it is closed.

        Returns:
            Current arrow tint color.
        """
        if self._arrow_override is not None:
            return self._arrow_override
        tok = ThemeManager.instance().tokens()
        key = "flat_accent" if self._popup_open else "flat_text_muted"
        return QtGui.QColor(*tok[key])

    def _retint_arrow(self) -> None:
        """Regenerate the arrow pixmap using the current arrow color.

        Re-tints the original source icon rather than the previously tinted
        pixmap, preventing successive theme changes from accumulating color
        transformations. The current rotation angle is then reapplied.
        """
        self._base_pixmap = self._tinted_pixmap(self._icon_source, self._current_arrow_color())
        self._on_spin_frame(self._current_angle)

    def _on_theme_changed(self, _mode: str | None = None) -> None:
        """Refresh combo-box visuals after the application theme changes.

        Updates the combo-box text styling, arrow tint, custom rounded popup,
        completer popup, and widget rendering so all associated controls
        immediately reflect the newly active theme.

        Args:
            _mode: Optional theme-mode value supplied by the theme-change signal.
                The current theme is resolved by the individual update methods.
        """
        self._apply_text_qss()
        self._retint_arrow()
        self._popup.apply_theme(self.font(), self._ROW_HEIGHT)
        self._apply_completer_popup_theme()
        self.update()

    def style_completer_popup(self, completer: QtWidgets.QCompleter) -> None:
        """Apply the combo box's flat styling to a completer suggestion popup.

        Creates a dedicated :class:`QListView` for the completer and configures
        it with the same flat-token surface, typography, hover treatment, and
        selection rendering used by the combo box's custom dropdown.

        The custom delegate is installed after calling :meth:`QCompleter.setPopup`
        because `setPopup()` replaces the view's existing item delegate with
        its own default delegate. Mouse tracking is enabled so the delegate can
        detect hovered rows and render the appropriate hover background.

        Args:
            completer: Completer whose suggestion popup should be styled.
        """
        view = QtWidgets.QListView()
        view.setObjectName("animatedComboCompleterPopup")
        view.setUniformItemSizes(True)
        view.setMouseTracking(True)  # required for the delegate's hover wash
        completer.setPopup(view)
        view.setItemDelegate(_CompleterRowDelegate(view))
        self._completer_popup_view = view
        self._apply_completer_popup_theme()

    def _apply_completer_popup_theme(self) -> None:
        """Apply the current theme to the completer suggestion popup.

        Updates the completer popup's surface, border, corner radius, and padding
        using the active theme tokens. Per-item hover and selection colors remain
        the responsibility of :class:`_CompleterRowDelegate`.

        If no completer popup has been configured, this method returns without
        making any changes.

        """
        view = getattr(self, "_completer_popup_view", None)
        if view is None:
            return
        view.viewport().setMouseTracking(True)
        tok = ThemeManager.instance().tokens()
        view.setStyleSheet(f"""
            QListView#animatedComboCompleterPopup {{
                background: {tok_css(tok["flat_surface"])};
                border: 1px solid {tok_css(tok["popup_border"])};
                border-radius: 8px;
                padding: 5px;
                outline: none;
            }}
        """)

    def _visible_rows(self) -> int:
        """Return the number of combo-box rows visible in the custom popup.

        The result is constrained to at least one row and at most the combo
        box's configured maximum number of visible items.

        Returns:
            Number of rows that should be displayed by the popup.
        """
        return min(max(self.count(), 1), self.maxVisibleItems())

    def _place_popup(self, full_h: int) -> None:
        """Position and size the custom popup directly below the combo box.

        Aligns the popup's left edge with the combo box and makes it slightly
        wider than the combo box so the popup visually covers the combo's border
        on both sides.

        Args:
            full_h: Total height of the popup in pixels.
        """
        top_left = self.mapToGlobal(QtCore.QPoint(0, self.height()))
        width = self.width() + 2  # cover the combo's own border on both sides
        self._popup.setGeometry(top_left.x() - 1, top_left.y(), width, full_h)

    def showPopup(self) -> None:
        """Open the custom combo-box popup with a slide and chevron animation.

        Prevents duplicate opens and suppresses immediate reopening shortly after
        a close. When opening, the popup is themed, sized to the number of visible
        rows, positioned below the combo box, and initialized with a fully hidden
        reveal mask.

        The popup then animates from a reveal value of `0.0` to `1.0` while
        the chevron rotates to indicate the open state. The current combo-box
        selection is synchronized with the popup's list view before the animation
        begins.
        """
        if self._popup_open:
            return
        if QtCore.QDateTime.currentMSecsSinceEpoch() - self._just_closed_ms < 200:
            return

        self._popup_open = True
        self._closing = False
        self._retint_arrow()
        self._popup.apply_theme(self.font(), self._ROW_HEIGHT)

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

    def hidePopup(self) -> None:
        """Close the custom popup using the reverse slide animation.

        Prevents duplicate close requests while a close is already in progress.
        The popup's reveal animation runs from its current value down to
        `0.0` while the chevron rotates back to its closed orientation.

        The popup is not actually hidden until :meth:`_on_slide_finished`
        completes the animation and explicitly permits the popup to hide.
        """
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
        """Finalize the popup state after a slide animation completes.

        When closing, resets the popup's internal state, records the close time,
        permits the popup to perform its actual hide operation, and restores the
        reveal value to `1.0` so the next open animation can begin from a known
        fully hidden state.

        The arrow is also re-tinted to reflect the closed popup state.
        """
        if self._closing:
            self._closing = False
            self._popup_open = False
            self._just_closed_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
            self._popup.close_animated()  # the only path allowed to truly hide
            self._popup.set_reveal(1.0)  # reset for next open
            self._retint_arrow()
            self.update()

    def _on_dismiss_requested(self) -> None:
        """Animate the popup closed after an automatic dismiss request.

        The custom popup can request dismissal when the user clicks outside it or
        selects an item. Rather than allowing the popup to disappear immediately,
        route the request through :meth:`hidePopup` so the normal close animation
        is used.

        If a close animation is already in progress, :meth:`hidePopup` safely
        ignores the duplicate request.
        """
        self.hidePopup()

    def _on_popup_closed(self) -> None:
        """Synchronize combo-box state after the popup is actually hidden.

        The `closed` signal is emitted only after the popup has been explicitly
        permitted to hide. This method therefore acts as a safety net to keep the
        popup state, chevron orientation, reveal value, and arrow tint consistent
        if the popup closes through a path other than the normal animated-close
        sequence.
        """
        if self._popup_open and not self._closing:
            self._popup_open = False
            self._just_closed_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
            self._spin_to(0.0)
            self._popup.set_reveal(1.0)
            self._retint_arrow()
            self.update()

    def _on_item_clicked(self, index: QtCore.QModelIndex) -> None:
        """Handle selection of an item from the custom popup.

        Updates the combo box's current index, emits the standard Qt
        `activated` signal for the selected row, and begins the animated popup
        close sequence.

        Args:
            index: Model index of the item selected in the popup.
        """
        self.setCurrentIndex(index.row())
        self.activated.emit(index.row())
        self.hidePopup()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Update the arrow hit area after the combo box is resized.

        Repositions the arrow label against the right edge of the combo box and
        raises it above the internal editor widget when necessary. Editable
        combo boxes can create their internal :class:`QLineEdit` lazily and place
        it above the arrow label, so raising the arrow during every relayout
        ensures the arrow remains an active click target.

        Args:
            event: Qt resize event generated after the combo box dimensions
                change.
        """
        super().resizeEvent(event)
        self.arrow_lbl.setGeometry(self.width() - 28, 0, 24, self.height())
        self.arrow_lbl.raise_()

    def _on_arrow_clicked(self, _event: QtGui.QMouseEvent) -> None:
        """Toggle the combo-box popup in response to an arrow click.

        Closes the popup when it is currently open; otherwise opens it using the
        normal animated popup sequence.

        Args:
            _event: Mouse event generated by clicking the arrow label. The event
                is intentionally unused because the click location does not
                affect the toggle behavior.
        """
        if self._popup_open:
            self.hidePopup()
        else:
            self.showPopup()

    def _spin_to(self, target_angle: float) -> None:
        """Animate the arrow chevron toward a target rotation angle.

        Stops any currently running chevron animation and starts a new animation
        from the arrow's current angle to the requested target angle.

        Args:
            target_angle: Target rotation angle in degrees.
        """
        self._anim.stop()
        self._anim.setStartValue(self._current_angle)
        self._anim.setEndValue(target_angle)
        self._anim.start()

    def _on_spin_frame(self, angle: float) -> None:
        """Render an intermediate frame of the chevron rotation animation.

        Updates the current rotation angle, applies the corresponding transform
        to the tinted base pixmap, and displays the rotated icon in the arrow
        label.

        Args:
            angle: Rotation angle in degrees for the current animation frame.
        """
        self._current_angle = angle
        transform = QtGui.QTransform().rotate(angle)
        rotated = self._base_pixmap.transformed(
            transform, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.arrow_lbl.setPixmap(rotated)

    def enterEvent(self, event) -> None:
        """Handle pointer entry by enabling the hover visual state.

        Records that the pointer is currently over the combo box and schedules a
        repaint so the flat control surface can update its border appearance.

        Args:
            event: Qt enter event generated when the pointer enters the widget.
        """
        super().enterEvent(event)
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        """Handle pointer exit by disabling the hover visual state.

        Records that the pointer has left the combo box and schedules a repaint
        so the control returns to its normal or focus-dependent appearance.

        Args:
            event: Qt leave event generated when the pointer leaves the widget.
        """
        super().leaveEvent(event)
        self._hovered = False
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the themed combo-box surface and current-item label.

        Renders the flat fill, border, and optional focus ring using the active
        theme tokens and the shared :func:`paint_flat_surface` recipe. The visual
        state depends on whether the combo box is disabled, focused or open,
        hovered, or in its normal state.

        The method intentionally does not call `QComboBox.paintEvent()`.
        Qt's default implementation paints the complete `CC_ComboBox` complex
        control, including its panel, arrow, and label. With a transparent QSS
        background, that panel can still erase custom painting performed before
        it. Instead, only `CE_ComboBoxLabel` is rendered through
        :class:`QStylePainter`, allowing Qt's native style to handle the current
        item text while leaving the custom surface and arrow untouched.

        Args:
            event: Qt paint event generated when the combo box requires repainting.
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
