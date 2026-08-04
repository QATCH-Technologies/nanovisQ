"""
avanced_main_widget.py

This module contains the `AdvancedMainWidget` and its supporting UI components
(`_InfoIcon`, `_AdvancedInnerPanel`). It is designed to render an elegant,
translucent dropdown panel that can anchor to a specific UI element and house
advanced configuration controls.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-06-19
"""

import os
import contextlib
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_SANS_STACK


def _tinted_pixmap(src: QtGui.QPixmap, color: QtGui.QColor) -> QtGui.QPixmap:
    """Returns a copy of `src` fully painted in `color`, preserving alpha.

    Uses SourceAtop composition so the tint respects the original alpha
    channel - transparent SVG areas stay transparent. Mirrors the
    established `_tinted_icon` pattern in account_popup.py.
    """
    if src.isNull():
        return src
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return dst


class _InfoIcon(QtWidgets.QLabel):
    """Info icon (SVG) that brightens on hover and shows a tooltip.

    Loads an SVG from the provided path. The icon is tinted with the
    `flat_text_muted` token at rest and `flat_accent` on hover, refreshing
    automatically on light/dark theme changes.

    Attributes:
        _DISPLAY_SIZE (int): The display size (width and height) of the icon in pixels.
    """

    _DISPLAY_SIZE: int = 16

    def __init__(self, icon_path: str, tooltip: str = "", parent=None) -> None:
        """Initializes the _InfoIcon.

        Args:
            icon_path (str): The file path to the SVG or image asset.
            tooltip (str, optional): The tooltip text to display on hover. Defaults to "".
            parent (QtWidgets.QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setFixedSize(self._DISPLAY_SIZE, self._DISPLAY_SIZE)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_Hover, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent; border: none;")
        self.setToolTip(tooltip)

        self._src = QtGui.QPixmap(icon_path)
        if not self._src.isNull():
            self._src = self._src.scaled(
                self._DISPLAY_SIZE,
                self._DISPLAY_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        self._hovered = False
        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
        if self._src.isNull():
            return
        tok = ThemeManager.instance().tokens()
        color = tok["flat_accent"] if self._hovered else tok["flat_text_muted"]
        self.setPixmap(_tinted_pixmap(self._src, QtGui.QColor(*color)))

    def enterEvent(self, event) -> None:  # noqa: N802
        """Handles the mouse enter event to brighten the icon.

        Args:
            event (QtCore.QEvent): The triggering hover event.
        """
        self._hovered = True
        self._apply_theme()

    def leaveEvent(self, event) -> None:  # noqa: N802
        """Handles the mouse leave event to dim the icon back to rest state.

        Args:
            event (QtCore.QEvent): The triggering hover leave event.
        """
        self._hovered = False
        self._apply_theme()


class _AdvancedInnerPanel(QtWidgets.QWidget):
    """Inner panel for the advanced settings popup.

    Styled as a flat card (see QATCH.ui.components.flat_paint) - a solid
    `flat_surface` fill with a 1px `flat_border` stroke - matching the
    account popup and the rest of the flat control system, rather than the
    old frosted-glass recipe. Repaints automatically on light/dark theme
    changes.

    Attributes:
        _RADIUS (float): The corner radius of the panel.
    """

    _RADIUS: float = 12.0

    def __init__(self, parent=None) -> None:
        """Initializes the _AdvancedInnerPanel.

        Args:
            parent (QtWidgets.QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the flat card surface (fill + border) on the widget.

        Args:
            event (QtGui.QPaintEvent): The paint event parameters provided by Qt.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_flat_surface(
            self,
            radius=self._RADIUS,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            painter=p,
        )
        p.end()


class _PerspectiveStage(QtWidgets.QWidget):
    """A clipped, horizontally-scrolling viewport that hosts two perspectives.

    Both the "advanced" perspective (index 0) and the "device" perspective
    (index 1) live side-by-side inside a single inner strip that is twice the
    viewport width. Switching perspectives animates the strip's horizontal
    offset so the advanced content slides left and reveals the device content
    as part of the SAME surface - not as a separate popup window sliding in as
    an overlay.

    The viewport sizes itself to the *currently active* perspective's size hint
    (the inactive one is laid out but does not inflate the viewport), so the
    popup footprint matches whichever view is showing rather than the larger of
    the two.

    Attributes:
        _DURATION (int): Slide animation duration in milliseconds.
    """

    _DURATION: int = 260

    transitionFinished = QtCore.pyqtSignal(int)  # emits the settled index

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        # The strip holds the two perspectives left-to-right. It is moved
        # horizontally inside this (clipping) viewport via setGeometry.
        self._strip = QtWidgets.QWidget(self)
        self._strip.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._strip.setStyleSheet("background: transparent;")

        self._pages: list[QtWidgets.QWidget | None] = [None, None]
        self._index = 0
        self._offset = 0.0  # 0.0 == advanced fully shown, 1.0 == device fully shown

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(self._DURATION)
        self._anim.valueChanged.connect(self._on_anim_value)
        self._anim.finished.connect(self._on_anim_finished)

    # -- page management ---------------------------------------------------
    def set_page(self, index: int, widget: QtWidgets.QWidget) -> None:
        """Adopt `widget` as the perspective at `index` (0=advanced, 1=device)."""
        old = self._pages[index]
        if old is widget:
            return
        if old is not None:
            old.setParent(None)
        self._pages[index] = widget
        widget.setParent(self._strip)
        widget.show()
        self._relayout()

    def current_index(self) -> int:
        return self._index

    # -- sizing ------------------------------------------------------------
    def _active_page(self) -> QtWidgets.QWidget | None:
        # During a transition the viewport sizes to the TALLER of the two pages
        # so neither clips mid-slide; at rest it tracks the active page so the
        # popup footprint matches the visible perspective.
        if self._anim.state() == QtCore.QAbstractAnimation.State.Running:
            a, b = self._pages
            if a is not None and b is not None:
                return a if self._page_h(a) >= self._page_h(b) else b
        return self._pages[self._index]

    def _page_w(self, p: QtWidgets.QWidget) -> int:
        """A page's effective width: the larger of its hint and its minimum."""
        return max(p.sizeHint().width(), p.minimumSizeHint().width(), p.minimumWidth())

    def _page_h(self, p: QtWidgets.QWidget) -> int:
        return max(p.sizeHint().height(), p.minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        page = self._active_page()
        if page is None:
            return QtCore.QSize(440, 320)
        # Width tracks the widest page so horizontal travel is consistent and
        # neither page overflows its slot.
        w = 0
        for p in self._pages:
            if p is not None:
                w = max(w, self._page_w(p))
        return QtCore.QSize(w, self._page_h(page))

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self) -> None:
        """Lay both pages side-by-side and position the strip per the offset."""
        vw = self.width()
        vh = self.height()
        if vw <= 0 or vh <= 0:
            return
        # Strip is two viewport-widths wide; each page occupies one slot.
        self._strip.setGeometry(self._strip_x(), 0, vw * 2, vh)
        for i, page in enumerate(self._pages):
            if page is not None:
                page.setGeometry(i * vw, 0, vw, vh)

    def _strip_x(self) -> int:
        return int(round(-self._offset * self.width()))

    # -- transitions -------------------------------------------------------
    def slide_to(self, index: int, animated: bool = True) -> None:
        """Slide to perspective `index` (0=advanced, 1=device)."""
        index = 1 if index else 0
        target = float(index)
        if not animated or not self.isVisible():
            self._index = index
            self._offset = target
            self.updateGeometry()
            self._relayout()
            self.transitionFinished.emit(index)
            return
        if abs(self._offset - target) < 1e-3 and self._index == index:
            return
        self._index = index  # active (for final sizing) is the destination
        self._anim.stop()
        self._anim.setStartValue(float(self._offset))
        self._anim.setEndValue(target)
        self._anim.start()

    def _on_anim_value(self, v) -> None:
        self._offset = float(v)
        # Re-assert width/height each frame so the viewport can grow/shrink
        # smoothly as the taller/shorter page comes into view.
        self.updateGeometry()
        self._strip.move(self._strip_x(), 0)

    def _on_anim_finished(self) -> None:
        self._offset = float(self._index)
        self.updateGeometry()
        self._relayout()
        self.transitionFinished.emit(self._index)


class _PerspectiveAnimator(QtCore.QObject):
    """Plays a gentle entrance on a perspective container each time it is shown.

    IMPORTANT: this deliberately does NOT use QGraphicsOpacityEffect. Wrapping a
    container that holds custom-painted children (combos, toggles,
    buttons) in a graphics effect caches them into an offscreen pixmap, which
    causes ghosting, duplicated section labels, and widgets vanishing on hover.
    Instead the fade is applied to the top-level popup window via
    setWindowOpacity (no pixmap caching), paired with a brief top-margin slide.

    Attributes:
        _container (QtWidgets.QWidget): The target widget container to animate.
        _slide (QtCore.QVariantAnimation): Animation for the vertical sliding movement.
        _fade (QtCore.QVariantAnimation): Animation for the window opacity transition.
    """

    def __init__(self, container: QtWidgets.QWidget) -> None:
        """Initializes the animator and attaches an event filter to the container.

        Args:
            container: The widget container whose window will be animated.
        """
        super().__init__(container)
        self._container = container

        # Guard flag: ensures only one _begin_slide is ever scheduled per show
        # event cycle.  Multiple ShowEvents can fire on the container during a
        # single open (e.g. from set_page's show() and set_advanced_perspective's
        # show()), particularly on the first open when the container transitions
        # from a hidden top-level widget.  Without this guard the animation
        # starts twice, producing the "doubly renders and animates" glitch.
        self._start_pending: bool = False

        # Slide the whole popup window DOWN into place (start 12px above the
        # final position), matching the account menu. Animating the inner
        # top-margin instead made content rise up from the bottom, which read
        # as a bottom-up animation.
        self._slide = QtCore.QVariantAnimation(self)
        self._slide.setDuration(220)
        self._slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._slide.setStartValue(0.0)
        self._slide.setEndValue(1.0)
        self._slide.valueChanged.connect(self._apply_slide)
        self._slide_from = None
        self._slide_to = None
        self._slide_offset = 12

        self._fade = QtCore.QVariantAnimation(self)
        self._fade.setDuration(200)
        self._fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.valueChanged.connect(self._apply_fade)
        self._fade.finished.connect(self._finish_fade)

        container.installEventFilter(self)

    def _apply_slide(self, t: float) -> None:
        """Calculates and applies the window position during the slide animation.

        Args:
            t (float): Normalized time value (0.0 to 1.0).
        """
        win = self._container.window()
        if win is None or self._slide_to is None:
            return
        slide_from = self._slide_from
        if slide_from is not None:
            x = self._slide_to.x()
            y = int(slide_from.y() + (self._slide_to.y() - slide_from.y()) * float(t))
        else:
            y = self._slide_to.y()
        win.move(x, y)

    def _apply_fade(self, v: float) -> None:
        """Applies the window opacity during the fade animation.

        Args:
            v (float): Opacity value (0.0 to 1.0).
        """
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(float(v))

    def _finish_fade(self) -> None:
        """Ensures the window is fully opaque and at the final position upon finish."""
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(1.0)
            if self._slide_to is not None:
                win.move(self._slide_to)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """Filters show events to trigger animations after the widget is visible.

        Args:
            obj: The object the event is sent to.
            event: The QEvent object.

        Returns:
            True if the event was handled, otherwise the base class result.
        """
        if obj is self._container and event.type() == QtCore.QEvent.Type.Show:
            # Deduplicate: only schedule _begin_slide once per open cycle.
            # The fade is NOT started here - _begin_slide starts both animations
            # together after the popup window is confirmed visible, preventing
            # the fade from running ahead of the slide (and ahead of show()).
            if not self._start_pending:
                self._start_pending = True
                QtCore.QTimer.singleShot(0, self._begin_slide)
        return super().eventFilter(obj, event)

    def _begin_slide(self) -> None:
        """Calculates start/end positions and starts both fade and slide animations.

        Called once per open cycle via a zero-delay singleShot, after all
        queued ShowEvents have been processed.  Resets the _start_pending
        guard so the next open cycle can arm again.
        """
        self._start_pending = False
        win = self._container.window()
        if win is None or not win.isVisible():
            return
        final_pos = win.pos()
        self._slide_to = QtCore.QPoint(final_pos)
        self._slide_from = QtCore.QPoint(final_pos.x(), final_pos.y() - self._slide_offset)
        win.move(self._slide_from)
        self._fade.stop()
        self._fade.start()
        self._slide.stop()
        self._slide.start()


class AdvancedMainWidget(QtWidgets.QWidget):
    """Frosted-glass dropdown panel for the Advanced Settings.

    This widget owns the entire advanced-settings surface: the frosted popup
    shell, the orange warning banner, and the container that hosts the controls.
    Callers build the individual controls (combo boxes, buttons, etc.) in their
    own grid layout and hand that layout to `build_content`; this widget
    wraps it inside the owned `content_container`.

    Attributes:
        content_container (QtWidgets.QWidget | None): The lazily-built container holding
            the dynamically injected content.
        content_layout (QtWidgets.QVBoxLayout): The internal layout managing the panel items.
    """

    closed = QtCore.pyqtSignal()

    _SHADOW_MARGIN_L = 22
    _SHADOW_MARGIN_T = 18
    _SHADOW_MARGIN_R = 22
    _SHADOW_MARGIN_B = 26

    _INFO_TEXT = (
        "These are advanced settings. Changes here affect device operation \u2014 "
        "adjust them only if you know what they do."
    )

    def __init__(self, parent=None) -> None:
        flags = (
            QtCore.Qt.WindowType.Popup
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.NoDropShadowWindowHint
        )
        super().__init__(parent, QtCore.Qt.WindowFlags(flags))  # type: ignore
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._main_window = None
        self._anchor = None
        self.content_container = None  # built lazily by build_content()

        # Outer container
        self._panel = _AdvancedInnerPanel(self)
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(
            self._SHADOW_MARGIN_L,
            self._SHADOW_MARGIN_T,
            self._SHADOW_MARGIN_R,
            self._SHADOW_MARGIN_B,
        )
        outer_layout.setSpacing(0)
        outer_layout.addWidget(self._panel)

        # Drop shadow
        shadow = QtWidgets.QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QtGui.QColor(0, 20, 40, 110))
        self._panel.setGraphicsEffect(shadow)

        # Inner layout
        self.content_layout = QtWidgets.QVBoxLayout(self._panel)
        self.content_layout.setContentsMargins(14, 14, 14, 14)

        # Perspective stage: hosts the advanced + device views side-by-side and
        # slides horizontally between them inside this single popup surface.
        self.stage = _PerspectiveStage(self._panel)
        self.content_layout.addWidget(self.stage)
        self.stage.transitionFinished.connect(self._on_stage_transition_finished)
        self._on_device_back = None  # optional callback after sliding back

    @staticmethod
    def build_container(controls_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        """Creates the advanced container (title header + controls).

        The header is a title row in the top-left: a gear icon, the
        "Advanced Options" title, and an info icon that reveals the advanced
        usage message on hover. `controls_layout` holds the actual control
        widgets (created/wired by the caller). The resulting container is
        hidden by default.

        Args:
            controls_layout (QtWidgets.QLayout): The layout containing the actual
                setting widgets.

        Returns:
            QtWidgets.QWidget: The wrapper widget containing the header and controls.
        """
        icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        container = QtWidgets.QWidget()
        container.setWhatsThis(AdvancedMainWidget._INFO_TEXT)

        # Title header (top-left)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(2, 0, 2, 0)
        header.setSpacing(8)

        gear = QtWidgets.QLabel()
        gear.setFixedSize(18, 18)
        gear.setScaledContents(True)
        gear.setStyleSheet("background: transparent; border: none;")
        _gear_pix = QtGui.QPixmap(os.path.join(icons_dir, "gear.svg"))

        title = QtWidgets.QLabel("Advanced Options")
        header.addWidget(gear, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(title, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        def _restyle_header(_mode: str = "") -> None:
            tok = ThemeManager.instance().tokens()
            if not _gear_pix.isNull():
                gear.setPixmap(_tinted_pixmap(_gear_pix, QtGui.QColor(*tok["flat_text_muted"])))
            title.setStyleSheet(
                f"QLabel {{ color: {tok_css(tok['flat_text'])}; "
                f"font-family: {FONT_SANS_STACK}; font-size: 14px; font-weight: 600; "
                "background: transparent; border: none; }"
            )

        _restyle_header()
        ThemeManager.instance().themeChanged.connect(_restyle_header)

        # Info icon
        info = _InfoIcon(
            os.path.join(icons_dir, "warning-circle.svg"),
            tooltip=AdvancedMainWidget._INFO_TEXT,
        )
        header.addWidget(info, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        header.addStretch()

        wrap = QtWidgets.QVBoxLayout(container)
        wrap.setSpacing(10)
        wrap.addLayout(header)
        wrap.addLayout(controls_layout)

        container.hide()
        return container

    @staticmethod
    def install_entrance_animation(owner: object, container: QtWidgets.QWidget) -> None:
        """Attaches a `_PerspectiveAnimator` entrance animation to `container`.

        Call once, right after a content container is first built (e.g. in
        the owning interface's `setup_ui`), so every subsequent popup open
        (see `toggle`, which reparents the same cached container into a
        fresh popup instance each time) replays the slide/fade entrance.

        A strong reference is kept on `owner` (as `owner._perspective_animators`)
        because `_PerspectiveAnimator` is only parented to `container` on the
        Qt/C++ side - that keeps the underlying C++ object alive, but PyQt can
        still garbage-collect the Python wrapper without an explicit Python-side
        reference, which would silently drop the animation.

        Args:
            owner (object): The object (typically a UI controller instance)
                that should hold the animator's strong reference.
            container (QtWidgets.QWidget): The content container to animate on
                each show.
        """
        animator = _PerspectiveAnimator(container)
        if not hasattr(owner, "_perspective_animators"):
            owner._perspective_animators = []
        owner._perspective_animators.append(animator)
        container.installEventFilter(animator)

    def build_content(self, controls_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        """Builds and adopts a fresh advanced container around `controls_layout`.

        Args:
            controls_layout (QtWidgets.QLayout): The layout containing the user controls.

        Returns:
            QtWidgets.QWidget: The assigned content container.
        """
        self.content_container = self.build_container(controls_layout)
        return self.content_container

    def set_content_widget(self, widget: QtWidgets.QWidget) -> None:
        """Injects an already-built container as the advanced perspective.

        Retained for backward compatibility or parity with the device-info
        popup. Prefer `build_content` for the advanced settings.

        Args:
            widget (QtWidgets.QWidget): The fully built widget to insert.
        """
        self.content_container = widget
        self.stage.set_page(0, widget)
        widget.show()

    # -- perspective hosting --------------------------------------------------
    def set_advanced_perspective(self, widget: QtWidgets.QWidget) -> None:
        """Registers `widget` as the advanced (index 0) perspective."""
        self.content_container = widget
        self.stage.set_page(0, widget)
        widget.show()

    def set_device_perspective(self, widget: QtWidgets.QWidget) -> None:
        """Registers `widget` as the device-config (index 1) perspective."""
        self.device_container = widget
        self.stage.set_page(1, widget)
        widget.show()

    def show_device_perspective(self, animated: bool = True) -> None:
        """Slide the panel left to reveal the device-config perspective."""
        self.stage.slide_to(1, animated=animated)

    def show_advanced_perspective(self, animated: bool = True, on_finished=None) -> None:
        """Slide the panel right, back to the advanced perspective.

        Args:
            animated (bool): Whether to animate the slide.
            on_finished (callable, optional): Invoked once the slide settles on
                the advanced view (used by the device "Back" button).
        """
        self._on_device_back = on_finished
        self.stage.slide_to(0, animated=animated)

    def _on_stage_transition_finished(self, index: int) -> None:
        # Keep the popup tightly sized to whichever perspective settled, then
        # re-anchor so it doesn't drift off the anchor as the height changes.
        self.adjustSize()
        self._reanchor()
        if index == 0 and self._on_device_back is not None:
            cb, self._on_device_back = self._on_device_back, None
            cb()

    @classmethod
    def toggle(cls, owner, anchor, controls_layout, main_window=None, attr="_advanced_popup"):
        """Opens or closes the advanced popup, owning the full lifecycle.

        `owner` stores the popup instance (typically the UIControls instance).
        The container is taken from `owner._advanced_content_container` if it
        was pre-built; otherwise it is built from `controls_layout` and cached
        there. If a popup is already visible it is closed.

        Args:
            owner (object): The parent object or controller that stores the popup state.
            anchor (QtWidgets.QWidget): The UI element this popup should anchor to.
            controls_layout (QtWidgets.QLayout): The layout containing the inner controls.
            main_window (QtWidgets.QWidget, optional): The application main window.
                Defaults to None.
            attr (str, optional): The attribute name on `owner` where the popup is stored.
                Defaults to "_advanced_popup".

        Returns:
            AdvancedMainWidget | None: The active popup instance if opened, or None if closed.
        """
        existing = getattr(owner, attr, None)
        if existing is not None and existing.isVisible():
            existing.close()
            return None

        popup = cls()

        cache_attr = "_advanced_content_container"
        container = getattr(owner, cache_attr, None)
        if container is None:
            container = popup.build_content(controls_layout)
            setattr(owner, cache_attr, container)
        else:
            popup.content_container = container

        popup.set_advanced_perspective(container)

        # If the owner has a pre-built device-config perspective, register it as
        # the second page so the two views can slide within this one popup.
        device_container = getattr(owner, "device_info_container", None)
        if device_container is not None:
            popup.set_device_perspective(device_container)

        # Let the owner reach the popup to drive perspective transitions.
        owner._advanced_popup = popup
        popup.stage.slide_to(0, animated=False)

        setattr(owner, attr, popup)
        popup.show_anchored_to(anchor, main_window=main_window)
        return popup

    def show_anchored_to(self, anchor: QtWidgets.QWidget, main_window=None) -> None:
        """Shows the popup pinned to the anchor, clamped to the main window bounds.

        Calculates geometry to ensure the dropdown renders neatly beneath or near
        the anchor while avoiding rendering off-screen.

        Args:
            anchor (QtWidgets.QWidget): The widget this popup stems from.
            main_window (QtWidgets.QWidget, optional): The top-level window used for
                boundary constraints. Defaults to None.
        """
        self._main_window = main_window
        self._anchor = anchor
        self.adjustSize()

        x, y = self._compute_anchored_pos(anchor)
        self.move(x, y)
        # Pre-hide so the window is invisible until the fade animation begins.
        # Without this the window flashes at full opacity for one frame before
        # _PerspectiveAnimator._begin_slide gets its zero-delay singleShot callback.
        self.setWindowOpacity(0.0)
        if self._main_window is not None:
            self._main_window.installEventFilter(self)
        self.show()

    def _compute_anchored_pos(self, anchor: QtWidgets.QWidget) -> tuple:
        """Computes the clamped top-left position for the current popup size."""
        popup_w, popup_h = self.width(), self.height()
        anchor_br = anchor.mapToGlobal(QtCore.QPoint(anchor.width(), anchor.height()))

        x = anchor_br.x() + self._SHADOW_MARGIN_R - popup_w
        y = anchor_br.y() + 2 - self._SHADOW_MARGIN_T

        top_level = anchor.window() if anchor is not None else None
        bounds = (
            top_level.geometry()
            if top_level
            else (self._main_window.geometry() if self._main_window else QtCore.QRect())
        )

        if not bounds.isNull():
            visible = QtCore.QRect(
                x + self._SHADOW_MARGIN_L,
                y + self._SHADOW_MARGIN_T,
                popup_w - self._SHADOW_MARGIN_L - self._SHADOW_MARGIN_R,
                popup_h - self._SHADOW_MARGIN_T - self._SHADOW_MARGIN_B,
            )
            if visible.right() > bounds.right():
                x -= visible.right() - bounds.right()
            if visible.left() < bounds.left():
                x += bounds.left() - visible.left()
            if visible.bottom() > bounds.bottom():
                anchor_top = anchor.mapToGlobal(QtCore.QPoint(0, 0)).y()
                y_above = anchor_top - 2 - popup_h + self._SHADOW_MARGIN_B
                if (y_above + self._SHADOW_MARGIN_T) >= bounds.top():
                    y = y_above
                else:
                    y -= visible.bottom() - bounds.bottom()
        return x, y

    def _reanchor(self) -> None:
        """Re-pin the popup to its anchor after its size changes."""
        if getattr(self, "_anchor", None) is None:
            return
        x, y = self._compute_anchored_pos(self._anchor)
        self.move(x, y)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Filters events on the main window to automatically close the popup.

        Triggers closure on main window movements or resizes to prevent floating UI.

        Args:
            watched (QtCore.QObject): The object being watched.
            event (QtCore.QEvent): The intercepted event.

        Returns:
            bool: Always returns the base class eventFilter result.
        """
        # Auto-close if the main window moves or resizes
        if watched is self._main_window and event.type() in (
            QtCore.QEvent.Type.Resize,
            QtCore.QEvent.Type.Move,
            QtCore.QEvent.Type.WindowStateChange,
        ):
            self.close()
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Cleans up event filters when the popup is closed.

        Args:
            event (QtGui.QCloseEvent): The close event.
        """
        if self._main_window is not None:
            with contextlib.suppress(Exception):
                self._main_window.removeEventFilter(self)
            self._main_window = None
        super().closeEvent(event)
        self.closed.emit()
