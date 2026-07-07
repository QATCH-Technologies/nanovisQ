"""login_central_widget.py

This module defines two classes that together implement the frosted-glass, blurred-backdrop
login overlay for the nanovisQ application:

- `PopAnimation`: A custom `QGraphicsEffect` that applies synchronized scale and opacity
  transforms to a widget without affecting its layout, used to animate the login card with a
  "pop-in / pop-out" transition.

- `LoginCentralWidget`: A full-coverage overlay widget that composites a live snapshot of
  the application window (blurred and dimmed) behind the login card. It manages all animation
  lifecycle (backdrop blur/dim, card pop, card border fade) and exposes a clean API
  (`reveal_signed_out`, `dismiss_for_signin`) for orchestrating sign-in transitions.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-07-01

"""

from typing import TYPE_CHECKING, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.styles.tokens import ColorTokens

if TYPE_CHECKING:
    from QATCH.ui.components import QATCHCard


class PopAnimation(QtWidgets.QGraphicsEffect):
    """A combined scale and opacity graphics effect for animated "pop" UI cards.

    This effect is designed for login/landing UI cards that need a smooth
    zoom-in/zoom-out ("pop") animation without affecting layout geometry.

    It works by rendering the source widget into a pixmap and then applying:
      - opacity (fade in/out)
      - uniform scale (zoom in/out)

    The underlying widget layout is not modified; only the rendered output
    is transformed.

    Attributes:
        _scale: Uniform scale factor applied to the rendered pixmap.
            Values > 1.0 enlarge, < 1.0 shrink.
        _opacity: Opacity applied during rendering in the range [0.0, 1.0].
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        """Initializes the PopAnimation with default scale and opacity.

        Args:
            parent: Optional Qt parent object for memory management.
        """
        super().__init__(parent)
        self._scale: float = 1.0
        self._opacity: float = 1.0

    def setScale(self, scale: float) -> None:
        """Sets the uniform scale factor for the effect.

        Args:
            scale: Scale multiplier applied to the rendered widget.
                Typical range is 0.8-1.2 for "pop" animations.

        Notes:
            Values are not strictly clamped; extreme values may cause
            clipping depending on boundingRectFor().
        """
        self._scale = scale
        self.update()

    def scale(self) -> float:
        """Returns the current scale factor.

        Returns:
            Current uniform scale multiplier.
        """
        return self._scale

    def setOpacity(self, opacity: float) -> None:  # noqa: N802
        """Sets the opacity of the rendered widget.

        Args:
            opacity: Opacity value in the range [0.0, 1.0].

        Notes:
            Values are clamped to [0.0, 1.0].
        """
        self._opacity = max(0.0, min(1.0, opacity))
        self.update()

    def opacity(self) -> float:
        """Returns the current opacity value.

        Returns:
            Float in the range [0.0, 1.0].
        """
        return self._opacity

    def boundingRectFor(self, source_rect: QtCore.QRectF) -> QtCore.QRectF:  # noqa: N802
        """Returns an expanded bounding rectangle to accommodate scaling.

        This prevents clipping when the effect applies overshoot animations
        (e.g., scale > 1.0 during "pop" in/out transitions).

        Args:
            source_rect: Original bounding rectangle of the source widget.

        Returns:
            Expanded rectangle that accounts for potential scale overshoot.
        """
        margin = max(source_rect.width(), source_rect.height()) * 0.15
        return source_rect.adjusted(-margin, -margin, margin, margin)

    def draw(self, painter: QtGui.QPainter) -> None:
        """Renders the source widget with applied opacity and scale transform.

        The widget is first rendered into a pixmap, then:
          1. Opacity is applied.
          2. Painter is translated to the pixmap center.
          3. Uniform scaling is applied.
          4. The pixmap is drawn back at the original offset.

        Args:
            painter: Active QPainter used by Qt's graphics effect system.

        NOTE:
            Early returns if the source pixmap is null.
        """
        pixmap, offset = self.sourcePixmap(QtCore.Qt.CoordinateSystem.LogicalCoordinates)
        if pixmap.isNull():
            return

        painter.save()
        try:
            painter.setOpacity(self._opacity)

            rect = QtCore.QRectF(offset, QtCore.QSizeF(pixmap.size()))
            center = rect.center()

            painter.translate(center)
            painter.scale(self._scale, self._scale)
            painter.translate(-center)

            painter.drawPixmap(offset, pixmap)
        finally:
            painter.restore()


class LoginCentralWidget(QtWidgets.QWidget):
    """Central widget providing a blurred, frosted backdrop for the login card.

    This widget manages the frosted glass background effect. It can either
    capture a live snapshot of the main application window or load a static
    branded asset. The backdrop is pre-blurred and cached as a QPixmap to
    ensure high-performance rendering during window resizes and animations.

    The visual stack consists of:
        1. A blurred source Pixmap (Wall-to-wall).
        2. A neutral light-gray overlay tint (~24% opacity)'.
        3. The login card and UI elements (added as children).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the central widget and configures backdrop and card state.

        This widget manages a custom "Deep Focus" backdrop system and a separate
        animated login card. It deliberately avoids relying on Qt's built-in
        opacity/graphics effects for backdrop transitions due to inconsistencies
        in cascading effects through complex native widget hierarchies on Windows.

        Instead, all backdrop dim/blur compositing is performed manually in
        paintEvent using QPainter.
        """
        super().__init__(parent)

        self._blurred: Optional[QtGui.QPixmap] = None
        self._raw_snapshot: Optional[QtGui.QPixmap] = None

        # Backdrop animation state
        self._blur_frac: float = 0.0
        self._dim_frac: float = 0.0
        self._backdrop_anim: Optional[QtCore.QVariantAnimation] = None

        # Animated login card state.
        self._card: Optional["QATCHCard"] = None
        self._card_effect: Optional[PopAnimation] = None
        self._card_anim: Optional[QtCore.QVariantAnimation] = None
        self._border_anim: Optional[QtCore.QVariantAnimation] = None

        # Disable Qt's automatic background filling so paintEvent has full control
        # over compositing the live parent content beneath translucency effects.
        #
        # WA_OpaquePaintEvent is intentionally NOT used because it may cause Qt to
        # skip repainting underlying content, which breaks blended transition
        # frames during animated dim/blur states.
        self.setAutoFillBackground(False)

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Handles theme changes by invalidating backdrop and child widgets.

        When the theme changes, both the backdrop (blur/dim layer) and the card
        UI must be repainted because they sample theme-dependent tokens directly
        in their paint events rather than inheriting palette changes automatically.

        Args:
            _mode: Theme mode identifier (unused directly; triggers repaint).
        """
        self.update()

        # Ensure all child widgets refresh their cached theme-dependent rendering.
        for child in self.findChildren(QtWidgets.QWidget):
            child.update()

    def register_dismissable_card(self, card: "QATCHCard") -> None:
        """Registers a login card for coordinated backdrop and pop animations.

        This method attaches a dedicated PopAnimation to the provided card so it
        can participate in synchronized "sign-in / sign-out" transitions alongside
        the backdrop (blur/dim) animations.

        The card's visual state is controlled independently from
        layout, ensuring animations do not affect widget geometry.

        Args:
            card (GlassCard): The GlassCard instance to be animated during authentication
                transitions (e.g., reveal_signed_out, dismiss_for_signin).
        """
        self._card = card

        self._card_effect = PopAnimation(card)
        self._card_effect.setScale(1.0)
        self._card_effect.setOpacity(1.0)

        card.setGraphicsEffect(self._card_effect)

    def attach_to(self, host: QtWidgets.QWidget) -> None:
        """Attaches this widget as a full-coverage overlay on top of a host widget.

        The overlay is reparented onto the given host and configured to track both
        the host widget and its top-level window so it remains perfectly aligned
        during resize, move, or window state changes.

        The overlay starts hidden and must be explicitly shown when needed.

        Args:
            host (QtWidgets.QWidget): The QWidget that this overlay should cover.
        """
        self.setParent(host)

        host.installEventFilter(self)

        top = host.window()
        if top is not host:
            top.installEventFilter(self)

        self._refit_to_parent()
        self.hide()

    def _refit_to_parent(self) -> None:
        """Resizes the overlay to exactly match its parent widget's geometry.

        This ensures the overlay always fully covers the parent widget,
        regardless of resize or layout changes.

        Notes:
            - If no parent widget is set, this method is a no-op.
            - Uses the parent's local rect (not global coordinates), since the
            overlay is reparented to the host widget.
        """
        parent = self.parentWidget()
        if parent is None:
            return

        self.setGeometry(parent.rect())

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Keeps the overlay geometry synchronized with its host window.

        This event filter listens for geometry-related changes on the host widget
        and its top-level window. When a resize or move event occurs, the overlay
        is refitted to fully cover the host.

        Args:
            obj: The QObject being watched (host widget or top-level window).
            event: The Qt event being processed.

        Returns:
            True if the event is fully handled by this filter; otherwise delegates
            to the base implementation.
        """
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move):
            self._refit_to_parent()

        return super().eventFilter(obj, event)

    def capture_backdrop(self, source: QtWidgets.QWidget) -> None:
        """Captures and prepares a blurred snapshot of a source widget for the backdrop.

        This method creates a visual "live UI" backdrop by capturing the current
        contents of the application window, extracting the region occupied by the
        given source widget, and generating a blurred version for use during
        transitions (e.g., login / sign-out states).

        The captured image is cached in both raw and blurred forms to avoid
        repeated expensive window grabs during subsequent resize or animation
        updates.

        Args:
            source: The widget whose window region should be captured for the
                backdrop.
        """
        # Capture the top-level window so that any ancestor-painted backgrounds
        # (e.g., central widget gradients) are included in the snapshot.
        was_visible = self.isVisible()

        if was_visible:
            self.hide()
            QtWidgets.QApplication.processEvents(
                QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
            )

        window = source.window()
        assert window is not None
        window_pixmap = window.grab()

        source_rect = QtCore.QRect(
            source.mapTo(window, QtCore.QPoint(0, 0)),
            source.size(),
        )

        raw = window_pixmap.copy(source_rect)

        if was_visible:
            self.show()
            self.raise_()

        # If the overlay already has a valid size, normalize snapshot to match it.
        if not self.size().isEmpty() and not raw.isNull():
            raw = raw.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

        self._raw_snapshot = raw
        self._blurred = self._apply_blur(raw, radius=22)

        # Reset transition state: a fresh capture invalidates any previous animation.
        self._stop_backdrop_anim()
        self._blur_frac = 0.0
        self._dim_frac = 0.0

        self.update()

    def refresh_backdrop_instant(self, source: QtWidgets.QWidget) -> None:
        """Refreshes the backdrop snapshot without changing the current visual state.

        This is used to replace the underlying image while preserving the existing
        "fully signed-out" visual presentation (i.e., fully blurred and dimmed).

        It is primarily needed for cold-start scenarios where the UI is not yet
        fully realized at the time of the initial `reveal_signed_out(instant=True)`
        call. In that case, a fallback gradient may be captured instead of the
        actual dashboard. Once the UI is properly laid out, this method re-captures
        the correct content and swaps it in seamlessly.

        Args:
            source: The widget whose current rendered state should be captured
                and used as the backdrop source.

        Notes:
            - Does not animate or modify transition state.
            - Forces blur and dim fractions to 1.0 to preserve the "fully frosted"
            signed-out appearance.
            - Intended to be called after the UI has reached a stable geometry.
        """
        self.capture_backdrop(source)

        # Keep the UI in the fully "signed-out" visual state.
        self._blur_frac = 1.0
        self._dim_frac = 1.0

        self.update()

    def reveal_signed_out(
        self,
        backdrop_source: Optional[QtWidgets.QWidget] = None,
        blur_duration: int = 400,
        card_duration: int = 500,
        border_delay: int = 100,
        instant: bool = False,
    ) -> None:
        """Transitions the UI into the "signed-out / login focus" state.

        This orchestrates the full "Deep Focus" effect:

        1. The dashboard backdrop is captured, then blurred and dimmed.
        2. The login card is animated into view with a pop.
        3. The card border fades in shortly after the main pop animation begins.

        The result is a layered visual transition where the underlying application
        recedes into a soft, frosted background while the login card becomes the
        primary focus.

        Args:
            backdrop_source: Widget used to capture the live dashboard snapshot.
                If None, the previously cached snapshot or fallback gradient is used.

                Ignored when `instant=True`.

            blur_duration: Duration (ms) of backdrop blur/dim animation.

            card_duration: Duration (ms) of the card pop animation.

            border_delay: Delay (ms) before starting the card border fade-in,
                relative to the start of the card animation.

            instant: If True, skips all animations and transitions immediately into
                the fully "signed-out" state. Used during cold start to avoid a
                single-frame flash of the signed-in dashboard.
        """
        if instant:
            self._stop_backdrop_anim()
            self._stop_card_anim()
            self._stop_border_anim()
            self._refit_to_parent()

            self._blur_frac = 1.0
            self._dim_frac = 1.0

            if self._card_effect is not None:
                self._card_effect.setScale(1.0)
                self._card_effect.setOpacity(1.0)

            if self._card is not None:
                self._card.set_border_frac(1.0)

            self.update()
            self.show()
            self.raise_()
            return

        if backdrop_source is not None:
            self.capture_backdrop(backdrop_source)

        self._stop_backdrop_anim()
        self._stop_card_anim()
        self._stop_border_anim()

        self._blur_frac = 0.0
        self._dim_frac = 0.0

        self._refit_to_parent()

        if self._card_effect is not None:
            self._card_effect.setScale(0.9)
            self._card_effect.setOpacity(0.0)

        if self._card is not None:
            self._card.set_border_frac(0.0)

        self.update()
        self.show()
        self.raise_()

        backdrop_anim = QtCore.QVariantAnimation(self)
        backdrop_anim.setStartValue(0.0)
        backdrop_anim.setEndValue(1.0)
        backdrop_anim.setDuration(blur_duration)
        backdrop_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        backdrop_anim.valueChanged.connect(self._set_backdrop_frac)
        self._backdrop_anim = backdrop_anim
        backdrop_anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

        card_anim = QtCore.QVariantAnimation(self)
        card_anim.setStartValue(0.0)
        card_anim.setEndValue(1.0)
        card_anim.setDuration(card_duration)
        card_anim.setEasingCurve(QtCore.QEasingCurve.OutBack)
        card_anim.valueChanged.connect(self._set_card_pop_frac)
        self._card_anim = card_anim
        card_anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

        QtCore.QTimer.singleShot(
            border_delay,
            lambda: self._start_border_anim(max(card_duration - border_delay, 1)),
        )

    def dismiss_for_signin(
        self,
        card_duration: int = 250,
        blur_duration: int = 400,
    ) -> None:
        """Transitions the UI from the "signed-out/login focus" state back to the app.

        This is the inverse of `reveal_signed_out()` and restores the application
        to its normal, sharp, and bright "signed-in" presentation.

        1. The login card rapidly dismisses.
        2. The blurred/dimmed dashboard backdrop gradually returns to full
            clarity.

        Args:
            card_duration (int): Duration (ms) of the card dismissal animation.
            blur_duration (int): Duration (ms) for restoring the backdrop to full clarity.
        """
        self._stop_card_anim()
        self._stop_border_anim()
        self._stop_backdrop_anim()

        if self._card is not None:
            self._card.set_border_frac(0.0)

        card_anim = QtCore.QVariantAnimation(self)
        card_anim.setStartValue(1.0)
        card_anim.setEndValue(0.0)
        card_anim.setDuration(card_duration)
        card_anim.setEasingCurve(QtCore.QEasingCurve.InQuad)
        card_anim.valueChanged.connect(self._set_card_pop_frac)

        self._card_anim = card_anim
        card_anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

        backdrop_anim = QtCore.QVariantAnimation(self)
        backdrop_anim.setStartValue(self._blur_frac)
        backdrop_anim.setEndValue(0.0)
        backdrop_anim.setDuration(blur_duration)
        backdrop_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        backdrop_anim.valueChanged.connect(self._set_backdrop_frac)
        backdrop_anim.finished.connect(self._finish_dismiss)

        self._backdrop_anim = backdrop_anim
        backdrop_anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _set_backdrop_frac(self, value: float) -> None:
        """Updates the backdrop transition state and triggers a repaint.

        This method drives the synchronized backdrop animation by
        updating both blur and dim intensity using a single normalized progress
        value.

        Args:
            value (float): Normalized animation progress for backdrop transition.
        """
        self._blur_frac = value
        self._dim_frac = value

        self.update()

        # Ensure dependent widgets that sample backdrop-derived visuals stay in sync.
        # Some children render using cached pixmaps in paintEvent and will not
        # automatically repaint on parent updates alone.
        for child in self.findChildren(QtWidgets.QWidget):
            child.update()

    def _set_card_pop_frac(self, t: float) -> None:
        """Updates the login card's pop animation state.

        This method is driven by a QVariantAnimation and maps a normalized
        progress value into visual properties on the PopAnimation.

        The animation produces a "pop-in" feel where the card becomes fully
        visible while subtly scaling from slightly smaller to full size.

        Args:
            t (float): Normalized animation progress.
                - 0.0 = hidden state (slightly reduced scale, fully transparent)
                - 1.0 = fully visible state (full scale, full opacity)
                - May exceed 1.0 during overshoot easing (e.g. OutBack), which
                produces a bounce effect.
        """
        if self._card_effect is None:
            return

        self._card_effect.setScale(0.9 + 0.1 * t)
        self._card_effect.setOpacity(t)

    def _start_border_anim(self, duration: int) -> None:
        """Starts the card emphasis-border fade-in animation.

        This animation is typically delayed relative to the main card pop to
        create a layered visual effect, where the card first appears and then
        gains its emphasis border.

        Args:
            duration (int): Duration (ms) of the border fade-in animation.

        Notes:
            - No-op if no card has been registered.
            - Any existing border animation is stopped before starting a new one.
            - The animation drives `GlassCard.set_border_frac()` from 0.0 to 1.0.
        """
        if self._card is None:
            return

        self._stop_border_anim()

        anim = QtCore.QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(duration)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(self._card.set_border_frac)

        self._border_anim = anim
        anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

    def _finish_dismiss(self) -> None:
        """Hides the overlay and releases its cached backdrop pixmaps.

        Called once the background finishes resolving to sharp/bright.
        Beyond hiding, this drops the (potentially full-screen-sized)
        blurred/raw snapshots so no stale frame and no image memory linger
        while signed in - the next `reveal_signed_out()` always starts from
        a fresh `capture_backdrop()`.
        """
        self.hide()
        self._blurred = None
        self._raw_snapshot = None
        self._blur_frac = 0.0
        self._dim_frac = 0.0

    def _stop_backdrop_anim(self) -> None:
        """Stops and detaches any running backdrop (blur/dim) animation.

        This ensures that no QVariantAnimation continues to drive backdrop state
        updates after a transition has been cancelled or replaced.
        """
        if self._backdrop_anim is not None:
            try:
                self._backdrop_anim.stop()
                self._backdrop_anim.valueChanged.disconnect()
                self._backdrop_anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._backdrop_anim = None

    def _stop_card_anim(self) -> None:
        """Stops and detaches any running card pop animation.

        This prevents stale animations from continuing to modify card scale or
        opacity after a new transition begins.
        """
        if self._card_anim is not None:
            try:
                self._card_anim.stop()
                self._card_anim.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._card_anim = None

    def _stop_border_anim(self) -> None:
        """Stops and detaches any running card border animation.

        Ensures the emphasis border animation does not continue updating the card
        after a transition has been interrupted or replaced.

        Notes:
            - Safe against repeated disconnect calls and Qt object teardown timing.
        """
        if self._border_anim is not None:
            try:
                self._border_anim.stop()
                self._border_anim.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._border_anim = None

    @staticmethod
    def _apply_blur(source: QtGui.QPixmap, radius: int = 22) -> QtGui.QPixmap:
        """Applies a Gaussian-style blur to a QPixmap using a graphics scene pipeline.

        This method renders the input pixmap through a QGraphicsScene with a
        QGraphicsBlurEffect applied, producing a high-quality blurred version of
        the image suitable for backdrop transitions.

        Args:
            source (QtGui.QPixmap): The input pixmap to blur.
            radius (int): Blur radius controlling intensity. Higher values produce a
                softer, more diffused image (default is 22).

        Returns:
            QtGui.QPixmap: A new QPixmap containing the blurred result.
        """
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(source)

        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(radius)
        blur.setBlurHints(QtWidgets.QGraphicsBlurEffect.BlurHint.QualityHint)
        item.setGraphicsEffect(blur)

        scene.addItem(item)

        out = QtGui.QPixmap(source.size())
        out.fill(QtCore.Qt.GlobalColor.transparent)

        p = QtGui.QPainter(out)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        scene.render(p, source=QtCore.QRectF(item.boundingRect()))
        p.end()

        return out

    @staticmethod
    def _compose_backdrop(
        p: QtGui.QPainter,
        rect: QtCore.QRect,
        sharp: Optional[QtGui.QPixmap],
        blurred: Optional[QtGui.QPixmap],
        blur_frac: float,
        dim_frac: float,
        tokens: ColorTokens,
    ) -> None:
        """Composes the backdrop layer.

        The layer composition consists of:
        1. Sharp-to-blurred crossfade of the captured UI snapshot.
        2. A constant overlay that defines the glass aesthetic.
        3. A dynamic dimming overlay that increases as focus shifts to the card.

        Args:
            p (QtGui.QPainter): Active QPainter used for rendering.
            rect (QtCore.QRect): Target rectangle in local widget coordinates.
            sharp (QtGui.QPixmap): Optional unblurred snapshot of the UI.
            blurred (QtGui.QPixmap): Optional pre-blurred snapshot of the UI.
            blur_frac (float): Crossfade factor between sharp (0.0) and blurred (1.0).
            dim_frac (float): Dimming intensity from 0.0 (bright) to 1.0 (fully dimmed).
            tokens (ColorTokens): Color palette defining backdrop gradient, frost, and dim colors.
        """
        blur_frac = max(0.0, min(1.0, blur_frac))
        dim_frac = max(0.0, min(1.0, dim_frac))

        if sharp is not None and not sharp.isNull():
            p.drawPixmap(rect, sharp, sharp.rect())

            if blurred is not None and not blurred.isNull() and blur_frac > 0.0:
                p.save()
                p.setOpacity(p.opacity() * blur_frac)
                p.drawPixmap(rect, blurred, blurred.rect())
                p.restore()

        elif blurred is not None and not blurred.isNull():
            p.drawPixmap(rect, blurred, blurred.rect())

        else:
            # Fallback gradient used during initial capture or missing snapshot state
            grad = QtGui.QLinearGradient(0, 0, rect.width(), rect.height())
            grad.setColorAt(
                0.0,
                QtGui.QColor(*tokens["backdrop_fallback_start"]),
            )
            grad.setColorAt(
                1.0,
                QtGui.QColor(*tokens["backdrop_fallback_end"]),
            )
            p.fillRect(rect, QtGui.QBrush(grad))

        # Frost layer: constant visual identity of the glass effect.
        p.fillRect(rect, QtGui.QColor(*tokens["backdrop_frost"]))

        # Dim layer: increases with focus shift toward the login card.
        dim_color = QtGui.QColor(*tokens["backdrop_dim"])
        dim_alpha = int(dim_color.alpha() * dim_frac)

        if dim_alpha > 0:
            dim_color.setAlpha(dim_alpha)
            p.fillRect(rect, dim_color)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paints the "Deep Focus" backdrop for this widget.

        Args:
            event (QtGui.QPaintEvent): Qt paint event triggered when the widget requires repainting.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        self._compose_backdrop(
            p,
            self.rect(),
            self._raw_snapshot,
            self._blurred,
            self._blur_frac,
            self._dim_frac,
            ThemeManager.instance().tokens(),
        )

        p.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        """Handles widget resizing by rescaling and refreshing the cached backdrop.

        Args:
            event: Qt resize event triggered when the widget geometry changes.
        """
        super().resizeEvent(event)

        if self._raw_snapshot is not None and not self.size().isEmpty():
            scaled = self._raw_snapshot.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

            self._blurred = self._apply_blur(scaled, radius=22)
            self.update()
