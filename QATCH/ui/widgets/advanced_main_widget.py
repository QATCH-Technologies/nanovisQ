"""
QATCH.ui.widgets.advanced_main_widget.py

Advanced settings popup components for QATCH.

Provides the widgets and supporting utilities used to display and interact
with the application's Advanced Settings popup. The module includes the
flat-painted popup panel, informational SVG icons, animated perspective
transitions, and the main popup container used to host advanced and
device-configuration controls.

The popup is implemented as a translucent, frameless Qt popup with a
flat-surface inner panel and a soft drop shadow. Advanced and device
configuration views are hosted side-by-side by :class:`_PerspectiveStage`
and transition horizontally within the same popup surface.

Entrance animations are handled by :class:`_PerspectiveAnimator`, which
animates the top-level popup window using `setWindowOpacity` and positional
movement rather than `QGraphicsOpacityEffect`. This avoids offscreen
pixmap caching artifacts that can occur when custom-painted child widgets
are rendered through a graphics effect.

The module also provides role-independent informational icon tinting and
theme-aware painting so the popup remains visually consistent across light
and dark application themes.

Classes:
    _InfoIcon: Theme-aware informational icon that changes tint on hover.
    _AdvancedInnerPanel: Flat-painted inner surface of the advanced settings
        popup.
    _PerspectiveStage: Clipped viewport that hosts and animates the advanced
        and device perspectives.
    _PerspectiveAnimator: Entrance animation controller for the popup window.
    AdvancedMainWidget: Main advanced-settings popup container.

Functions:
    _tinted_pixmap: Apply a color tint to a pixmap while preserving its alpha
        channel.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

import contextlib
import os

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_SANS_STACK


def _tinted_pixmap(src: QtGui.QPixmap, color: QtGui.QColor) -> QtGui.QPixmap:
    """Return a copy of a pixmap tinted with the specified color.

    The source pixmap's alpha channel is preserved by using
    `CompositionMode_SourceAtop` when applying the tint. This ensures
    transparent regions remain transparent while all visible pixels are
    replaced by the requested color.

    Args:
        src: Source pixmap to tint.
        color: Color applied to the visible pixels of the source pixmap.

    Returns:
        A tinted copy of `src` with its original alpha channel preserved.
        If `src` is null, the original pixmap is returned unchanged.
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
    """Display a themed informational SVG icon with hover highlighting.

    Loads an icon from the specified path and renders it at a fixed display
    size. The icon uses the active theme's muted text color in its normal
    state and switches to the theme accent color while hovered. The icon is
    automatically refreshed when the application theme changes.

    A tooltip can optionally be displayed when the user hovers over the icon.
    The source pixmap is tinted while preserving its original alpha channel.

    Attributes:
        _DISPLAY_SIZE: Width and height of the rendered icon in pixels.
        _src: Scaled source pixmap used as the basis for tinted rendering.
        _hovered: Whether the mouse is currently hovering over the icon.
    """

    _DISPLAY_SIZE: int = 16

    def __init__(self, icon_path: str, tooltip: str = "", parent=None) -> None:
        """Initialize the informational icon.

        Args:
            icon_path: File path to the SVG or image asset used by the icon.
            tooltip: Optional tooltip text displayed when the icon is hovered.
            parent: Optional parent widget.
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
        """Refresh the icon tint when the application theme changes.

        Args:
            _mode: Theme mode identifier emitted by `ThemeManager`. The
                value is not used directly because the current theme tokens
                are retrieved by :meth:`_apply_theme`.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the current theme color to the icon.

        Uses the theme's accent color while the icon is hovered and the muted
        text color otherwise. If the source pixmap could not be loaded, no
        update is performed.
        """
        if self._src.isNull():
            return
        tok = ThemeManager.instance().tokens()
        color = tok["flat_accent"] if self._hovered else tok["flat_text_muted"]
        self.setPixmap(_tinted_pixmap(self._src, QtGui.QColor(*color)))

    def enterEvent(self, event) -> None:
        """Highlight the icon when the mouse enters its bounds.

        Args:
            event: Qt event generated when the mouse enters the widget.
        """
        self._hovered = True
        self._apply_theme()

    def leaveEvent(self, event) -> None:
        """Restore the icon's muted tint when the mouse leaves.

        Args:
            event: Qt event generated when the mouse leaves the widget.
        """
        self._hovered = False
        self._apply_theme()


class _AdvancedInnerPanel(QtWidgets.QWidget):
    """Flat-styled inner panel for the advanced settings popup.

    Renders the popup's content surface using the application's flat control
    system. The panel consists of a solid `flat_surface` fill with a
    one-pixel `flat_border` stroke and rounded corners, matching the visual
    treatment used by other flat popup panels.

    The panel listens for application theme changes and repaints itself so
    that its fill and border colors remain synchronized with the active theme.

    Attributes:
        _RADIUS: Corner radius of the painted panel surface, in pixels.
    """

    _RADIUS: float = 12.0

    def __init__(self, parent=None) -> None:
        """Initialize the advanced settings inner panel.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the panel when the application theme changes.

        Args:
            _mode: Theme mode identifier emitted by `ThemeManager`. The
                value is not used directly because the current theme tokens
                are retrieved during painting.
        """
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the themed flat card surface.

        Draws the panel using the active theme's surface and border tokens
        through the shared `paint_flat_surface` rendering recipe.

        Args:
            event: Qt paint event describing the region that requires
                repainting.
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
    """Clipped horizontal viewport for animated perspective transitions.

    Hosts the advanced and device perspectives side-by-side within a single
    inner strip that is twice the width of the viewport. Changing perspectives
    animates the strip horizontally so the outgoing perspective slides away
    while the incoming perspective is revealed as part of the same surface,
    rather than appearing as a separate popup overlay.

    The viewport tracks the size of the currently active perspective when
    stationary. During a transition, it temporarily sizes itself according to
    the taller perspective so that neither page is clipped while sliding.
    This allows the popup footprint to match the visible perspective at rest
    while preserving a clean transition between differently sized views.

    Signals:
        transitionFinished: Emitted with the settled perspective index when a
            slide transition completes or when an immediate perspective change
            is performed.

    Attributes:
        _DURATION: Duration of perspective slide animations, in milliseconds.
        _strip: Transparent container holding both perspective widgets
            side-by-side.
        _pages: Two-element list containing the advanced perspective at index
            0 and the device perspective at index 1.
        _index: Index of the currently active or destination perspective.
        _offset: Fractional horizontal transition offset, where `0.0` shows
            the advanced perspective and `1.0` shows the device perspective.
        _anim: Animation controlling the horizontal perspective transition.
    """

    _DURATION: int = 260

    transitionFinished = QtCore.pyqtSignal(int)  # emits the settled index

    def __init__(self, parent=None) -> None:
        """Initialize the perspective viewport.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")

        # The strip holds the two perspectives left-to-right
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

    #  Page management
    def set_page(self, index: int, widget: QtWidgets.QWidget) -> None:
        """Set a widget as one of the available perspectives.

        Existing content at the specified index is detached before the new
        widget is adopted by the internal strip.

        Args:
            index: Perspective index. `0` represents the advanced
                perspective and `1` represents the device perspective.
            widget: Widget to display for the selected perspective.
        """
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
        """Return the index of the currently active perspective.

        Returns:
            `0` for the advanced perspective or `1` for the device
            perspective.
        """
        return self._index

    def _active_page(self) -> QtWidgets.QWidget | None:
        """Return the page used to determine the current viewport height.

        During an active transition, the taller of the two available pages is
        used so that neither page is clipped while sliding. When stationary,
        the currently active page determines the viewport height.

        Returns:
            The page whose height should currently determine the viewport
            size, or `None` if no page has been installed.
        """
        if self._anim.state() == QtCore.QAbstractAnimation.State.Running:
            a, b = self._pages
            if a is not None and b is not None:
                return a if self._page_h(a) >= self._page_h(b) else b
        return self._pages[self._index]

    def _page_w(self, p: QtWidgets.QWidget) -> int:
        """Return a page's effective width.

        The effective width is the largest value among the page's size hint,
        minimum size hint, and explicit minimum width.

        Args:
            p: Page widget whose effective width should be calculated.

        Returns:
            Effective page width in pixels.
        """
        return max(p.sizeHint().width(), p.minimumSizeHint().width(), p.minimumWidth())

    def _page_h(self, p: QtWidgets.QWidget) -> int:
        """Return a page's effective height.

        Args:
            p: Page widget whose effective height should be calculated.

        Returns:
            Effective page height in pixels.
        """
        return max(p.sizeHint().height(), p.minimumSizeHint().height())

    def sizeHint(self) -> QtCore.QSize:
        """Return the preferred viewport size for the active perspective.

        The width is based on the widest installed page so that each
        perspective occupies a consistent horizontal slot. The height is
        determined by the page selected by :meth:`_active_page`.

        Returns:
            Preferred viewport size in pixels. If no page is installed, a
            default size of `440 x 320` is returned.
        """
        page = self._active_page()
        if page is None:
            return QtCore.QSize(440, 320)
        w = 0
        for p in self._pages:
            if p is not None:
                w = max(w, self._page_w(p))
        return QtCore.QSize(w, self._page_h(page))

    def minimumSizeHint(self) -> QtCore.QSize:
        """Return the minimum size required by the active perspective.

        Returns:
            The same size returned by :meth:`sizeHint`.
        """
        return self.sizeHint()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Relayout the perspective strip when the viewport is resized.

        Args:
            event: Qt resize event generated by Qt.
        """
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self) -> None:
        """Lay out both perspectives and position the transition strip.

        Each installed page occupies one viewport-width slot in the internal
        strip. The strip itself is positioned according to the current
        fractional transition offset.
        """
        vw = self.width()
        vh = self.height()
        if vw <= 0 or vh <= 0:
            return
        self._strip.setGeometry(self._strip_x(), 0, vw * 2, vh)
        for i, page in enumerate(self._pages):
            if page is not None:
                page.setGeometry(i * vw, 0, vw, vh)

    def _strip_x(self) -> int:
        """Return the strip's horizontal position for the current offset.

        Returns:
            X-coordinate, in pixels, at which the internal strip should be
            positioned within the viewport.
        """
        return int(round(-self._offset * self.width()))

    def slide_to(self, index: int, animated: bool = True) -> None:
        """Transition to the requested perspective.

        An animated transition slides the internal strip horizontally. When
        animation is disabled or the viewport is not visible, the destination
        perspective is applied immediately.

        Args:
            index: Target perspective index. Values are normalized to `0`
                for the advanced perspective and `1` for the device
                perspective.
            animated: Whether to animate the transition. Defaults to `True`.
        """
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
        """Update the strip position as the transition animation progresses.

        Args:
            v: Current animation value representing the fractional horizontal
                transition offset between the two perspectives.
        """
        self._offset = float(v)
        self.updateGeometry()
        self._strip.move(self._strip_x(), 0)

    def _on_anim_finished(self) -> None:
        """Finalize the perspective transition and emit its completion signal.

        Snaps the strip to the destination perspective, updates the viewport
        geometry, and notifies listeners that the transition has settled.
        """
        self._offset = float(self._index)
        self.updateGeometry()
        self._relayout()
        self.transitionFinished.emit(self._index)


class _PerspectiveAnimator(QtCore.QObject):
    """Animate the entrance of a perspective container.

    Provides a coordinated fade and downward slide whenever the associated
    container is shown. The animation is applied to the container's top-level
    window using `setWindowOpacity` rather than `QGraphicsOpacityEffect`.

    Avoiding a graphics opacity effect is intentional because containers may
    contain custom-painted Qt widgets such as combo boxes, toggles, and
    buttons. Graphics effects can cache such widgets into an offscreen pixmap,
    causing rendering artifacts including ghosted content, duplicated labels,
    and widgets disappearing during hover interactions.

    The animator listens for `Show` events on the container and schedules a
    single entrance animation for each show cycle. A guard prevents multiple
    `Show` events generated during a single popup opening from starting
    duplicate animations.

    Attributes:
        _container: Widget whose top-level window is animated.
        _start_pending: Whether an entrance animation has already been
            scheduled for the current show cycle.
        _slide: Animation controlling the window's vertical movement.
        _slide_from: Starting position of the current slide animation.
        _slide_to: Final position of the current slide animation.
        _slide_offset: Number of pixels above the final position from which the
            window begins its entrance.
        _fade: Animation controlling the top-level window opacity.
    """

    def __init__(self, container: QtWidgets.QWidget) -> None:
        """Initialize the perspective animator.

        Installs an event filter on `container` so that its show events can
        trigger the entrance animation.

        Args:
            container: Widget whose top-level window should be animated when
                the container becomes visible.
        """
        super().__init__(container)
        self._container = container

        # Guard flag: ensures only one _begin_slide is ever scheduled per show
        # event cycle.
        self._start_pending: bool = False

        # Slide the whole popup window down into place
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
        """Apply the current vertical position of the slide animation.

        Interpolates between the starting and final window positions using the
        supplied normalized animation progress.

        Args:
            t: Normalized animation progress, typically ranging from `0.0`
                to `1.0`.
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
        """Apply the current opacity value to the top-level window.

        Args:
            v: Opacity value, normally ranging from `0.0` to `1.0`.
        """
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(float(v))

    def _finish_fade(self) -> None:
        """Finalize the entrance animation at full opacity and final position.

        Ensures the top-level window is fully opaque and positioned exactly at
        the destination coordinates after the fade animation completes.
        """
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(1.0)
            if self._slide_to is not None:
                win.move(self._slide_to)

    def eventFilter(self, obj, event) -> bool:
        """Monitor the container for show events that start the animation.

        A zero-delay timer is used to defer animation startup until queued
        `Show` events have been processed. A guard prevents multiple show
        events during a single opening cycle from scheduling duplicate
        animations.

        Args:
            obj: Object whose event is being filtered.
            event: Qt event being processed.

        Returns:
            The result returned by the base class event-filter
            implementation.
        """
        if obj is self._container and event.type() == QtCore.QEvent.Type.Show:
            if not self._start_pending:
                self._start_pending = True
                QtCore.QTimer.singleShot(0, self._begin_slide)
        return super().eventFilter(obj, event)

    def _begin_slide(self) -> None:
        """Prepare and start the coordinated entrance animations.

        Determines the window's final position, places it slightly above that
        position, and starts the fade and slide animations together. The
        pending-start guard is cleared so a subsequent show cycle can schedule
        another entrance animation.

        If the container's top-level window is unavailable or no longer
        visible when the deferred callback executes, no animation is started.
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
    """Dropdown popup containing the application's advanced settings surface.

    Owns the complete advanced-settings popup, including its translucent
    top-level shell, flat inner panel, drop shadow, perspective transition
    stage, warning content, and dynamically injected controls.

    Callers construct the individual advanced-settings controls and provide
    their layout through :meth:`build_content`. The widget wraps that content
    in its own container while retaining ownership of the overall popup
    surface and its presentation behavior.

    The popup supports multiple perspectives through
    :class:`_PerspectiveStage`, allowing the advanced and device views to
    slide horizontally within the same surface rather than appearing as
    separate popup windows.

    Signals:
        closed: Emitted when the advanced-settings popup is closed.

    Attributes:
        content_container: Lazily-created container holding dynamically
            injected advanced-settings content.
        content_layout: Layout managing the popup's main content, including
            the perspective stage.
        stage: Perspective viewport containing the advanced and device
            settings views.
        _main_window: Main application window associated with the popup.
        _anchor: Widget used to anchor the popup when it is displayed.
        _panel: Inner painted panel containing the popup content.
        _on_device_back: Optional callback invoked after returning from the
            device perspective.

    Class Attributes:
        _SHADOW_MARGIN_L: Left margin reserved around the inner panel for the
            drop shadow.
        _SHADOW_MARGIN_T: Top margin reserved around the inner panel for the
            drop shadow.
        _SHADOW_MARGIN_R: Right margin reserved around the inner panel for the
            drop shadow.
        _SHADOW_MARGIN_B: Bottom margin reserved around the inner panel for
            the drop shadow and its positive Y offset.
        _INFO_TEXT: Warning text displayed to users when viewing advanced
            settings.
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
        """Initialize the advanced-settings popup.

        Creates the translucent popup shell, inner panel, drop shadow, content
        layout, and perspective stage. The individual advanced-settings
        controls are added later through the content-building interface.

        Args:
            parent: Optional parent widget.
        """
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

        # Perspective stage
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

    def set_advanced_perspective(self, widget: QtWidgets.QWidget) -> None:
        """Set the widget used for the advanced-settings perspective.

        Registers `widget` as perspective index `0` in the internal
        :class:`_PerspectiveStage` and makes it the current advanced-settings
        content container.

        Args:
            widget: Widget containing the advanced-settings controls.
        """
        self.content_container = widget
        self.stage.set_page(0, widget)
        widget.show()

    def set_device_perspective(self, widget: QtWidgets.QWidget) -> None:
        """Set the widget used for the device-configuration perspective.

        Registers `widget` as perspective index `1` in the internal
        :class:`_PerspectiveStage` and stores it as the device configuration
        container.

        Args:
            widget: Widget containing the device-configuration controls.
        """
        self.device_container = widget
        self.stage.set_page(1, widget)
        widget.show()

    def show_device_perspective(self, animated: bool = True) -> None:
        """Switch to the device-configuration perspective.

        Slides the perspective stage horizontally to reveal the device
        configuration view.

        Args:
            animated: Whether to animate the transition. Defaults to `True`.
        """
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
        """Finalize the popup after a perspective transition completes.

        Resizes the popup to match the settled perspective and re-anchors it to
        the original anchor so changes in perspective height do not cause the
        popup to drift. When returning to the advanced perspective, invokes the
        pending device-back callback once and clears it.

        Args:
            index: Index of the perspective that has finished transitioning.
                `0` represents the advanced perspective and `1` represents
                the device-configuration perspective.
        """
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
        """Compute the clamped top-left position for the current popup size.

        Positions the popup relative to the bottom-right corner of `anchor`,
        accounting for the transparent shadow margins around the inner panel.
        The resulting position is clamped to the containing top-level window when
        necessary. If the popup would extend below the window, it is repositioned
        above the anchor when sufficient space is available.

        Args:
            anchor: Widget to which the popup should be anchored.

        Returns:
            A `(x, y)` tuple containing the popup's top-left position in global
            screen coordinates.
        """
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
        """Reposition the popup relative to its anchor after a size change.

        Does nothing when no anchor has been assigned. Otherwise, recalculates the
        clamped popup position using the current popup dimensions and moves the
        popup to the resulting global coordinates.
        """
        if getattr(self, "_anchor", None) is None:
            return
        x, y = self._compute_anchored_pos(self._anchor)
        self.move(x, y)

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
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
