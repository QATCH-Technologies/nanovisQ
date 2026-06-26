"""
ui_login.py

Authentication and user session management interface for nanovisQ.

This module implements a modern, glass-morphism inspired login system. It
orchestrates the user's entry into the application by managing:
    - Secure credential input via custom animated GlassLineEdit widgets.
    - User authentication and role-based access control (RBAC) via UserProfiles.
    - Password recovery workflows and interface navigation via a sliding panel.
    - Persistent user preferences (e.g., 'Remember Me') utilizing QSettings.
    - Hardware status monitoring, specifically tracking and displaying the
      system Caps Lock state.

The module is designed to operate as a standalone modal or a primary entry
point, providing fluid transitions and high-fidelity visual feedback through
manual painting and property-based animations.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-05-05
"""

import os
from typing import Callable, Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, UserRoles
from QATCH.ui.popUp import PopUp
from QATCH.ui.widgets.floating_message_badge_widget import FloatingMessageBadgeWidget
from QATCH.ui.components.glass_line_edit import GlassLineEdit

_CARD_W: int = 320
_INPUT_H: int = 34
_BTN_H: int = 32
_PAGE_H: int = 400

_P_SIGNIN = 0
_P_RECOVER = 1


class LoginWindow(QtWidgets.QMainWindow):
    """Main window for handling user login events.

    This class provides a login window that manages user interactions, and the window close event.
    It initializes the login UI and processes events to either authenticate the user,
    clear the login form, or update the UI state (e.g., toggling the Caps Lock indicator).

    Attributes:
        ui5 (Ui_Login): An instance of the login UI class used to set up and manage
            the login interface view.
    """

    def __init__(self, parent: QtWidgets.QMainWindow) -> None:
        """Initializes the LoginWindow with the given parent window.

        This method sets up the user interface for the login window by creating an instance
        of the UI class and initializing it with the current window and parent window.

        Args:
            parent (QtWidgets.QMainWindow): The parent widget for this login window.
        """
        super().__init__()
        self.ui5 = UILogin()
        self.ui5.setup_ui(self, parent)

    def eventFilter(self, obj, event: QtCore.QEvent) -> bool:
        """Intercepts and processes key events for the login window.

        This method handles `KeyPress` events to facilitate:
          - **Enter/Return**: Focuses the password field if empty; otherwise, signs in.
          - **Escape**: Clears the login form.

        This method handles `KeyRelease` events to facilitate:
          - **Caps Lock**: Toggles the global Caps Lock indicator state.

        This method handles `FocusIn` events to facilitate:
          - **Global Focus**: Syncs the Caps Lock indicator with the OS state when returning to the app.
        """
        if event.type() == QtCore.QEvent.KeyPress:
            # Handles focus for user password field and sign-in action.
            if event.key() in [QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return]:
                if len(self.ui5.user_password.text()) == 0:
                    self.ui5.user_password.setFocus()
                else:
                    self.ui5.action_sign_in()
                return True  # Stop the event from triggering returnPressed

            # Handles clearing the login form on EscapeKey press.
            if event.key() == QtCore.Qt.Key_Escape:
                self.ui5.clear_form()
                return True

        # Listen for KeyRelease instead of KeyPress for CapsLock.
        # This gives the OS a fraction of a second to update its internal state
        # before we read it using windll_is_caps_lock_on().
        if event.type() == QtCore.QEvent.KeyRelease:
            if event.key() == QtCore.Qt.Key_CapsLock:
                self.ui5.caps_lock_on = Constants.windll_is_caps_lock_on()
                self.ui5.update_caps_lock_state(self.ui5.caps_lock_on)

        # Handles focus in events for the window.
        if event.type() == QtCore.QEvent.FocusIn:
            # Sync the UI indicator with the actual OS state whenever the app gets focus back
            self.ui5.caps_lock_on = Constants.windll_is_caps_lock_on()
            self.ui5.update_caps_lock_state(self.ui5.caps_lock_on)

        # Always process default event handling, too.
        return super().eventFilter(obj, event)

    def closeEvent(self, event: QtCore.QEvent) -> None:
        """Handles the window close event by prompting the user for confirmation.

        When a close event occurs, this method displays a confirmation dialog asking the user
        whether they wish to quit the application. If the user confirms, the application quits;
        otherwise, the event is ignored, and the window remains open.

        Args:
            event (QtCore.QEvent): The close event triggered when the user attempts to close the window.

        Returns:
            None
        """
        res = PopUp.question(
            self,
            Constants.app_title,
            "Are you sure you want to quit QATCH Q-1 application now?",
            True,
        )
        if res:
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


class CardPopEffect(QtWidgets.QGraphicsEffect):
    """Combined scale + opacity effect for the login card's "pop" in/out.

    A single custom QGraphicsEffect rather than QGraphicsOpacityEffect plus
    some separate scale mechanism, so the card can visually zoom (the "Deep
    Focus" pop/push) without reflowing its real child widgets — QLineEdit,
    QPushButton, etc. keep their normal layout-driven geometry; only the
    rendered pixmap is transformed.
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._scale: float = 1.0
        self._opacity: float = 1.0

    def setScale(self, scale: float) -> None:
        self._scale = scale
        self.update()

    def scale(self) -> float:
        return self._scale

    def setOpacity(self, opacity: float) -> None:
        self._opacity = max(0.0, min(1.0, opacity))
        self.update()

    def opacity(self) -> float:
        return self._opacity

    def boundingRectFor(self, source_rect: QtCore.QRectF) -> QtCore.QRectF:
        # Margin so an overshoot scale (>1.0, e.g. the OutBack pop-in) isn't clipped.
        margin = max(source_rect.width(), source_rect.height()) * 0.15
        return source_rect.adjusted(-margin, -margin, margin, margin)

    def draw(self, painter: QtGui.QPainter) -> None:
        pixmap, offset = self.sourcePixmap(QtCore.Qt.LogicalCoordinates)
        if pixmap.isNull():
            return
        painter.save()
        painter.setOpacity(self._opacity)
        rect = QtCore.QRectF(offset, QtCore.QSizeF(pixmap.size()))
        center = rect.center()
        painter.translate(center)
        painter.scale(self._scale, self._scale)
        painter.translate(-center)
        painter.drawPixmap(offset, pixmap)
        painter.restore()


class LoginCentralWidget(QtWidgets.QWidget):
    """Central widget providing a blurred, frosted backdrop for the login card.

    This widget manages the 'frosted glass' background effect. It can either
    capture a live snapshot of the main application window or load a static
    branded asset. The backdrop is pre-blurred and cached as a QPixmap to
    ensure high-performance rendering during window resizes and animations.

    The visual stack consists of:
        1. A blurred source Pixmap (Wall-to-wall).
        2. A neutral light-gray overlay tint (~24% opacity) to provide 'frost'.
        3. The login card and UI elements (added as children).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the central widget and configures paint attributes."""
        super().__init__(parent)
        self._blurred: Optional[QtGui.QPixmap] = None
        self._raw_snapshot: Optional[QtGui.QPixmap] = None

        # "Deep Focus" backdrop state: both 0.0 (dashboard sharp/bright, the
        # signed-in look) -> 1.0 (dashboard blurred/dimmed, the signed-out
        # look). Applied via QPainter in paintEvent — deliberately NOT a
        # QGraphicsEffect. An ancestor-level QGraphicsOpacityEffect was tried
        # first, but it doesn't reliably cascade through the card's deep,
        # real-widget subtree (native inputs, WA_TranslucentBackground
        # children) on the real Windows backend. Manual paint-time
        # compositing has no such ambiguity.
        self._blur_frac: float = 0.0
        self._dim_frac: float = 0.0
        self._backdrop_anim: Optional[QtCore.QVariantAnimation] = None

        # The card fades/scales independently via its own CardPopEffect —
        # see register_dismissable_card().
        self._card: Optional["GlassCard"] = None
        self._card_effect: Optional[CardPopEffect] = None
        self._card_anim: Optional[QtCore.QVariantAnimation] = None
        self._border_anim: Optional[QtCore.QVariantAnimation] = None

        # NOTE: deliberately NOT WA_OpaquePaintEvent. That hint tells Qt this
        # widget always fully overwrites its area, so Qt skips repainting
        # whatever is underneath first — fine at rest, but once paintEvent
        # blends with reduced opacity during a transition, the blend needs
        # the *live* parent content underneath, not a stale, skipped-repaint
        # backing store.
        self.setAutoFillBackground(False)

    def register_dismissable_card(self, card: "GlassCard") -> None:
        """Gives `card` its own pop effect (scale + opacity) and tracks it
        for the synchronized sign-out reveal / sign-in dismiss sequences.

        Args:
            card (GlassCard): The login card to animate alongside this
                widget's backdrop during reveal_signed_out()/dismiss_for_signin().
        """
        self._card = card
        self._card_effect = CardPopEffect(card)
        self._card_effect.setScale(1.0)
        self._card_effect.setOpacity(1.0)
        card.setGraphicsEffect(self._card_effect)

    def attach_to(self, host: QtWidgets.QWidget) -> None:
        """Mounts this widget as a full-coverage overlay on top of `host`.

        Reparents onto `host`, tracks its resizes/moves (and those of its
        top-level window) to stay perfectly fitted, and starts out hidden.

        Args:
            host (QtWidgets.QWidget): The widget this overlay should cover.
        """
        self.setParent(host)
        host.installEventFilter(self)
        top = host.window()
        if top is not host:
            top.installEventFilter(self)
        self._refit_to_parent()
        self.hide()

    def _refit_to_parent(self) -> None:
        """Resizes this overlay to exactly cover its parent widget."""
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Keeps the overlay's geometry in sync with the watched host/window."""
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move):
            self._refit_to_parent()
        return super().eventFilter(obj, event)

    def capture_backdrop(self, source: QtWidgets.QWidget) -> None:
        """Captures a live snapshot of the provided widget, blurs it, and repaints.

        This allows the login screen to feel integrated into the current app state
        by using the actual UI as the background. The pre-blur snapshot is cached
        so subsequent resizes can rescale/reblur without re-grabbing live content.

        Args:
            source (QtWidgets.QWidget): The widget to grab for the backdrop.
        """
        raw: QtGui.QPixmap = source.grab()

        if not self.size().isEmpty():
            raw = raw.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )

        self._raw_snapshot = raw
        self._blurred = self._apply_blur(raw, radius=22)
        # A fresh capture always starts fully sharp/bright; any prior
        # transition's progress is no longer meaningful.
        self._stop_backdrop_anim()
        self._blur_frac = 0.0
        self._dim_frac = 0.0
        self.update()

    def reveal_signed_out(
        self,
        backdrop_source: Optional[QtWidgets.QWidget] = None,
        blur_duration: int = 400,
        card_duration: int = 500,
        border_delay: int = 100,
    ) -> None:
        """Plays the "Deep Focus" sign-out reveal.

        The dashboard pulls out of focus — blurring and dimming in behind
        the glass — while the login card pops into the foreground with a
        slight overshoot, its emphasis border catching up shortly after.

        Args:
            backdrop_source (QtWidgets.QWidget, optional): If given, a fresh
                `capture_backdrop()` is taken from this widget before showing.
            blur_duration (int, optional): Background blur/dim duration (ms).
            card_duration (int, optional): Card pop duration (ms).
            border_delay (int, optional): Delay before the card border starts
                fading in (ms), relative to the start of the card pop.
        """
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

    def dismiss_for_signin(self, card_duration: int = 250, blur_duration: int = 400) -> None:
        """Plays the "Deep Focus" sign-in dismissal.

        The login card snaps back and fades quickly — as if pushed past the
        lens — while the dashboard pulls back into sharp, bright focus over
        a slightly longer beat. The overlay fully tears down once the
        background finishes resolving.

        Args:
            card_duration (int, optional): Card dismissal duration (ms).
            blur_duration (int, optional): Background blur/dim-out duration (ms).
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
        """Applies the current blur/dim progress and repaints."""
        self._blur_frac = value
        self._dim_frac = value
        self.update()
        # GlassCard samples our pixmaps directly in its own paintEvent rather
        # than inheriting our repaint, so it must be nudged explicitly to stay
        # in sync frame-by-frame.
        for child in self.findChildren(QtWidgets.QWidget):
            child.update()

    def _set_card_pop_frac(self, t: float) -> None:
        """Applies the card's scale + opacity for the current pop progress.

        `t` ranges 0.0 (hidden, 90% size) -> 1.0 (shown, 100% size), and may
        briefly exceed 1.0 during an OutBack overshoot — CardPopEffect clamps
        opacity but lets scale overshoot, which is the intended bounce.
        """
        if self._card_effect is None:
            return
        self._card_effect.setScale(0.9 + 0.1 * t)
        self._card_effect.setOpacity(t)

    def _start_border_anim(self, duration: int) -> None:
        """Starts the card's delayed emphasis-border fade-in."""
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
        while signed in — the next `reveal_signed_out()` always starts from
        a fresh `capture_backdrop()`.
        """
        self.hide()
        self._blurred = None
        self._raw_snapshot = None
        self._blur_frac = 0.0
        self._dim_frac = 0.0

    def _stop_backdrop_anim(self) -> None:
        """Stops and detaches any in-flight blur/dim animation."""
        if self._backdrop_anim is not None:
            try:
                self._backdrop_anim.stop()
                self._backdrop_anim.valueChanged.disconnect()
                self._backdrop_anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._backdrop_anim = None

    def _stop_card_anim(self) -> None:
        """Stops and detaches any in-flight card pop animation."""
        if self._card_anim is not None:
            try:
                self._card_anim.stop()
                self._card_anim.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._card_anim = None

    def _stop_border_anim(self) -> None:
        """Stops and detaches any in-flight card-border animation."""
        if self._border_anim is not None:
            try:
                self._border_anim.stop()
                self._border_anim.valueChanged.disconnect()
            except (TypeError, RuntimeError):
                pass
            self._border_anim = None

    @staticmethod
    def _apply_blur(source: QtGui.QPixmap, radius: int = 22) -> QtGui.QPixmap:
        """Applies a Gaussian-style blur to a QPixmap.

        Utilizes QGraphicsScene and QGraphicsBlurEffect to perform a
        high-quality blur of the source image.

        Args:
            source (QtGui.QPixmap): The raw pixmap to blur.
            radius (int): The intensity of the blur. Defaults to 22.

        Returns:
            QtGui.QPixmap: The processed, blurred pixmap.
        """
        scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(source)

        blur = QtWidgets.QGraphicsBlurEffect()
        blur.setBlurRadius(radius)
        blur.setBlurHints(QtWidgets.QGraphicsBlurEffect.QualityHint)
        item.setGraphicsEffect(blur)

        scene.addItem(item)

        out = QtGui.QPixmap(source.size())
        out.fill(QtCore.Qt.transparent)

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
    ) -> None:
        """Paints the "Deep Focus" backdrop into `rect`: a sharp/blurred
        crossfade plus a light glass frost and a dark dimming overlay.

        Shared by LoginCentralWidget's own paint and GlassCard's sampled
        slice so both always show the exact same blur/dim progress.

        Args:
            p (QtGui.QPainter): The active painter.
            rect (QtCore.QRect): The local rect to paint into.
            sharp (QtGui.QPixmap, optional): The unblurred snapshot.
            blurred (QtGui.QPixmap, optional): The pre-blurred snapshot.
            blur_frac (float): 0.0 (sharp) -> 1.0 (fully blurred).
            dim_frac (float): 0.0 (bright) -> 1.0 (max dim).
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
            # Fallback gradient shown during initial load/capture
            grad = QtGui.QLinearGradient(0, 0, rect.width(), rect.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD8, 0xE6, 0xF0))
            grad.setColorAt(1.0, QtGui.QColor(0xEE, 0xF4, 0xF8))
            p.fillRect(rect, QtGui.QBrush(grad))

        # Constant light glass frost — the identity of the card, independent
        # of how dimmed the scene currently is.
        p.fillRect(rect, QtGui.QColor(238, 243, 247, 62))

        # Dark dimming overlay: 0 -> ~0.3 alpha as the camera pulls focus
        # toward the foreground card.
        dim_alpha = int(76 * dim_frac)
        if dim_alpha > 0:
            p.fillRect(rect, QtGui.QColor(0, 0, 0, dim_alpha))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Renders the "Deep Focus" backdrop: a blur/dim crossfade.

        Args:
            event (QtGui.QPaintEvent): The paint event triggered by Qt.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self._compose_backdrop(
            p, self.rect(), self._raw_snapshot, self._blurred, self._blur_frac, self._dim_frac
        )
        p.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handles widget resizing by rescaling and re-blurring the cached snapshot.

        Args:
            event (QtGui.QResizeEvent): The resize event triggered by Qt.
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


class GlassCard(QtWidgets.QFrame):
    """A custom frame that renders a glassmorphism effect via backdrop sampling.

    This widget creates the illusion of translucent glass by sampling the blurred
    pixmap from a LoginCentralWidget. It maps its local coordinates to the
    backdrop's coordinate space to 'slice' the background perfectly.

    The rendering pipeline follows these steps:
        1. Create a rounded-rectangle clip path.
        2. Sample and translate the blurred backdrop slice.
        3. Apply a neutral 'frost' tint and a faint cool blue identifier tint.
        4. Render a top-down white shimmer gradient.
        5. Draw a multi-layered border (muted outer stroke + inner highlight rim).

    Attributes:
        _RADIUS (float): The corner radius for the rounded rectangle.
        _backdrop (LoginCentralWidget): Reference to the widget providing the
            blurred source image.
    """

    _RADIUS: float = 22.0

    def __init__(
        self,
        backdrop: LoginCentralWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initializes the glass card and configures transparency attributes.

        Args:
            backdrop (LoginCentralWidget): The source widget for background blur.
            parent (QtWidgets.QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self._backdrop = backdrop
        self._border_frac: float = 0.0

        # Prevent the base class from painting opaque backgrounds
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def set_border_frac(self, value: float) -> None:
        """Sets the emphasis-border progress (0.0 hidden -> 1.0 fully shown)."""
        self._border_frac = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Executes the custom glassmorphism painting pipeline.

        This method manually handles all background and border rendering. It
        specifically avoids calling super().paintEvent() to prevent Qt Style
        Sheets from overwriting the translucent effects with opaque colors.
        """
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)
        blurred = getattr(self._backdrop, "_blurred", None)
        sharp = getattr(self._backdrop, "_raw_snapshot", None)
        if (blurred is not None and not blurred.isNull()) or (
            sharp is not None and not sharp.isNull()
        ):
            origin = self.mapTo(self._backdrop, QtCore.QPoint(0, 0))
            p.save()
            p.translate(-origin.x(), -origin.y())
            # Mirror the backdrop's blur/dim progress so the card's sampled
            # slice matches the area around it exactly during transitions.
            blur_frac = getattr(self._backdrop, "_blur_frac", 0.0)
            dim_frac = getattr(self._backdrop, "_dim_frac", 0.0)
            LoginCentralWidget._compose_backdrop(
                p, self._backdrop.rect(), sharp, blurred, blur_frac, dim_frac
            )
            p.restore()
        else:
            grad = QtGui.QLinearGradient(0, 0, 0, self.height())
            grad.setColorAt(0.0, QtGui.QColor(0xD4, 0xE6, 0xF4))
            grad.setColorAt(1.0, QtGui.QColor(0xE6, 0xF2, 0xFA))
            p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Primary white glass tint
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 70))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 35))
        shimmer = QtGui.QLinearGradient(0, 0, 0, 80)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 60))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))
        p.setClipping(False)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 135), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 175), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        # Emphasis border: a crisp 1px white edge that catches up shortly
        # after the card pops in, like glass catching the light.
        if self._border_frac > 0.0:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, int(210 * self._border_frac)), 1.0))
            p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)

        p.end()


class SlidingPanel(QtWidgets.QWidget):
    """A horizontal viewport that slides between multiple UI pages.

    This widget acts as a clipping container. It maintains a fixed width (the
    viewport) and hosts an inner container that can be significantly wider.
    By animating the horizontal position of the inner container, the panel
    reveals different pages (e.g., Sign In, Password Recovery) with a fluid
    sliding transition.

    Attributes:
        _pw (int): The fixed width of a single page/viewport.
        _pages (list[QtWidgets.QWidget]): Ordered list of widgets added as pages.
        _inner (QtWidgets.QWidget): The horizontal host for all page widgets.
        _anim (Optional[QtCore.QPropertyAnimation]): The current active transition.
    """

    def __init__(self, page_width: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the slider with a fixed viewport width.

        Args:
            page_width (int): The width in pixels for the visible area.
            parent (QtWidgets.QWidget, optional): The parent widget.
        """
        super().__init__(parent)
        self._pw = page_width
        self._anim: Optional[QtCore.QPropertyAnimation] = None
        self._pages: List[QtWidgets.QWidget] = []
        self.setFixedWidth(page_width)
        self.setContentsMargins(0, 0, 0, 0)
        self.setAutoFillBackground(False)
        self._inner = QtWidgets.QWidget(self)
        self._inner.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._inner.setAutoFillBackground(False)
        self._inner.move(0, 0)

    def add_page(self, widget: QtWidgets.QWidget) -> int:
        """Adds a widget as a new page in the slider.

        Args:
            widget (QtWidgets.QWidget): The widget to be used as a page.

        Returns:
            int: The index of the added page.
        """
        idx = len(self._pages)
        widget.setParent(self._inner)
        widget.setFixedWidth(self._pw)
        widget.move(idx * self._pw, 0)
        self._pages.append(widget)
        return idx

    def finalize(self, fallback_height: int = 300) -> None:
        """Synchronizes the geometry of the inner container and all child pages.

        This should be called once after all pages have been added and the
        main window layout has settled. It ensures the inner host is tall
        enough to prevent vertical clipping.

        Args:
            fallback_height (int, optional): Height to use if the current
                widget height is zero. Defaults to 300.
        """
        if not self._pages:
            return

        h = self.height() if self.height() > 0 else fallback_height

        # Ensure every page fills the vertical space
        for i, p in enumerate(self._pages):
            p.setFixedSize(self._pw, h)
            p.move(i * self._pw, 0)

        # Expand inner host to accommodate the total number of pages
        self._inner.setFixedSize(len(self._pages) * self._pw, h)

    def slide_to(self, page_idx: int, duration: int = 360) -> None:
        """Animates the panel to reveal the page at the specified index.

        Args:
            page_idx (int): The index of the page to show.
            duration (int, optional): Animation length in milliseconds.
                Defaults to 360.
        """
        if page_idx >= len(self._pages) or page_idx < 0:
            return

        target_x = -page_idx * self._pw

        # Stop existing animation to prevent conflicting positional updates.
        # DeleteWhenStopped (below) deletes the underlying C++ object once a
        # previous animation finishes naturally, but leaves this Python
        # reference dangling — touching it then raises RuntimeError.
        if self._anim is not None:
            try:
                if self._anim.state() == QtCore.QPropertyAnimation.Running:
                    self._anim.stop()
            except RuntimeError:
                pass
            self._anim = None

        self._anim = QtCore.QPropertyAnimation(self._inner, b"pos")
        self._anim.setStartValue(self._inner.pos())
        self._anim.setEndValue(QtCore.QPoint(target_x, 0))
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)

        # DeleteWhenStopped helps with memory management in long-running apps
        self._anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


class _UserInfoProxy:
    """Compatibility shim: routes legacy user_info label calls to the floating badge."""

    def __init__(self, badge: FloatingMessageBadgeWidget, card) -> None:
        self._badge = badge
        self._card = card
        self.suppress = False  # Add a suppression flag

    def clear(self) -> None:
        self._badge.clear()

    def setText(self, text: str) -> None:
        if self.suppress:
            return  # Ignore legacy text updates during auth success

        if text and text.strip():
            self._badge.show_message(text.strip(), is_error=False, parent_widget=self._card)
        else:
            self._badge.clear()

    def text(self) -> str:
        return ""

    def __getattr__(self, name):
        return lambda *a, **kw: None


class UILogin:
    """Manages the lifecycle and state of the nanovisQ login interface.

    This class is responsible for constructing the glass-morphism UI, handling
    user authentication flows, and managing transitions between the login card
    and the password recovery interface. It utilizes a sliding panel architecture
    to provide a fluid, single-window user experience.

    Attributes:
        parent (QtWidgets.QMainWindow): The main application window used for
            post-login role assignment and UI state transitions.
        caps_lock_on (bool): Internal tracker for the hardware Caps Lock state.
        password_shown (bool): Toggle state for password masking (echo mode).
        floating_badge (FloatingMessageBadge): A frameless overlay used to
            display non-blocking alerts, errors, and status notifications.
        user_info (_UserInfoProxy): A compatibility shim that routes legacy
            text updates to the modern floating badge system.
        remember_me_cb (QtWidgets.QCheckBox): Toggle for persisting the
            username across application sessions via QSettings.

    Primary Interactions:
        - Authentication: Validates input, calls the UserProfiles auth backend,
          and configures the application mode (Run vs. Analyze) upon success.
        - Navigation: Handles horizontal sliding transitions between the
          Sign In and Password Recovery pages.
        - Persistence: Reads and writes user preferences to the system registry
          using the QSettings framework.
        - Feedback: Provides real-time hardware status (Caps Lock) and
          interactive error feedback (shake animations and floating badges).
    """

    def setup_ui(
        self,
        MainWindow5: QtWidgets.QMainWindow,
        parent: QtWidgets.QMainWindow,
    ) -> None:
        """Initializes all UI elements for the simplified login window.

        Constructs the main window properties, central widget, and the sliding panel
        pages (Sign In and Password Recovery). Connects all necessary UI signals to
        their respective slots and sets up session management.

        Args:
            MainWindow5 (QtWidgets.QMainWindow): The primary window object to build the UI onto.
            parent (QtWidgets.QMainWindow): The parent application window, used for state transitions
                and role assignments upon successful authentication.
        """
        self.parent = parent
        self.caps_lock_on = False

        global _CARD_W, _INPUT_H, _PAGE_H, _BTN_H
        _CARD_W = 320
        _INPUT_H = 34
        _BTN_H = 32
        _PAGE_H = 400

        # Window
        MainWindow5.setObjectName("MainWindow5")
        MainWindow5.setMinimumSize(QtCore.QSize(800, 500))
        MainWindow5.resize(800, 500)
        MainWindow5.setTabShape(QtWidgets.QTabWidget.Rounded)

        # Central widget
        self.centralwidget = LoginCentralWidget(MainWindow5)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        MainWindow5.setCentralWidget(self.centralwidget)

        MainWindow5.setStyleSheet("""
            QWidget {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            }
            #centralwidget { background: transparent; border: none; }
            #slidingPanel  { background: transparent; border: none; }
            #loginCard     { background: transparent; border: none; }
        """)

        # Sign In
        signInPage = QtWidgets.QWidget()
        signInPage.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        si = QtWidgets.QVBoxLayout(signInPage)
        si.setContentsMargins(28, 26, 28, 22)
        si.setSpacing(10)
        si.setAlignment(QtCore.Qt.AlignTop)

        # Logo
        logoLabel = QtWidgets.QLabel()
        logoLabel.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        logoLabel.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        logo_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        logo_pm = QtGui.QPixmap(logo_path)

        if not logo_pm.isNull():
            logo_pm = logo_pm.scaled(
                54,
                54,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            logoLabel.setPixmap(logo_pm)
            logoLabel.setFixedSize(54, 54)

        si.addWidget(logoLabel, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        si.addSpacing(2)

        # Title
        siTitle = QtWidgets.QLabel("Sign In")
        siTitle.setObjectName("cardTitle")
        siTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        siTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        si.addWidget(siTitle)
        si.addSpacing(4)

        self.user_username = GlassLineEdit()
        self.user_username.setObjectName("user_username")
        self.user_username.setFixedHeight(_INPUT_H)
        self.user_username.setPlaceholderText("Username")
        self.user_username.textChanged.connect(lambda _: self.user_username.set_error(False))
        # NOTE: Until "username" is used for something other than "initials",
        #       normalize all user input to uppercase as they type characters
        self.user_username.textEdited.connect(
            lambda _: self.user_username.setText(self.user_username.text().upper())
        )
        si.addWidget(self.user_username)
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda _: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_username.returnPressed.connect(self.user_password.setFocus)
        self.user_initials = self.user_username

        # Password field
        self.user_password = GlassLineEdit()
        self.user_password.setObjectName("user_password")
        self.user_password.setFixedHeight(_INPUT_H)
        self.user_password.setPlaceholderText("Password")
        self.user_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.user_password.textChanged.connect(lambda _: self.user_password.set_error(False))
        self.user_password.returnPressed.connect(self.action_sign_in)
        self.user_password.installEventFilter(MainWindow5)
        self.user_username.installEventFilter(MainWindow5)
        MainWindow5.installEventFilter(MainWindow5)

        # Eye action
        self.visibleIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye-on.svg")
        )
        self.hiddenIcon = QtGui.QIcon(
            os.path.join(Architecture.get_path(), "QATCH", "icons", "eye-off.svg")
        )
        self.password_shown = False
        self.togglepasswordAction = self.user_password.addAction(
            self.visibleIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.togglepasswordAction.triggered.connect(self.on_toggle_password_Action)
        si.addWidget(self.user_password)

        # Caps Lock indicator — always occupies its row; text is blank when off
        self.caps_indicator = QtWidgets.QLabel("")
        self.caps_indicator.setObjectName("capsIndicator")
        self.caps_indicator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.caps_indicator.setFixedHeight(16)
        self.caps_indicator.setStyleSheet(
            "color: rgba(200, 130, 30, 235); font-size: 7.5pt; font-weight: 600;"
        )
        si.addWidget(self.caps_indicator, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        # Remember Me Toggle
        self.remember_me_cb = QtWidgets.QCheckBox("Remember me")
        self.remember_me_cb.setObjectName("rememberMe")
        self.remember_me_cb.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.remember_me_cb.setStyleSheet("""
            QCheckBox {
                color: rgba(60, 70, 80, 220);
                font-size: 9pt;
                font-weight: 500;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 4px;
                border: 1px solid rgba(150, 160, 170, 180);
                background: rgba(255, 255, 255, 120);
            }
            QCheckBox::indicator:hover {
                border: 1px solid rgba(10, 163, 230, 150);
            }
            QCheckBox::indicator:checked {
                background: rgba(10, 163, 230, 210);
                border: 1px solid rgba(10, 150, 210, 255);
            }
        """)

        # Wrap in a horizontal layout to keep it left-aligned cleanly
        rm_layout = QtWidgets.QHBoxLayout()
        rm_layout.setContentsMargins(4, 0, 0, 0)
        rm_layout.addWidget(self.remember_me_cb)
        rm_layout.addStretch()
        si.addLayout(rm_layout)
        si.addSpacing(6)

        # Sign In button
        self.sign_in_btn = QtWidgets.QPushButton("Sign In")
        self.sign_in_btn.setObjectName("signInBtn")
        self.sign_in_btn.setFixedHeight(_BTN_H)
        self.sign_in_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.sign_in_btn.setStyleSheet(self._make_primary_btn_style())
        self.sign_in_btn.clicked.connect(self.action_sign_in)
        si.addWidget(self.sign_in_btn)

        si.addStretch()

        # Forgot Password link
        forgotPasswordLbl = QtWidgets.QLabel("Forgot Password?")
        forgotPasswordLbl.setObjectName("forgotPassword")
        forgotPasswordLbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        forgotPasswordLbl.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        forgotPasswordLbl.setStyleSheet("""
            QLabel {
                color: rgba(100, 110, 120, 180);
                font-size: 9pt; font-weight: 500;
            }
            QLabel:hover { color: rgba(60, 60, 60, 220); text-decoration: underline; }
        """)
        forgotPasswordLbl.mousePressEvent = lambda _e: self._slide_to(_P_RECOVER)
        si.addWidget(forgotPasswordLbl)

        # Forgot Password
        recoverPage = QtWidgets.QWidget()
        recoverPage.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        rec = QtWidgets.QVBoxLayout(recoverPage)
        rec.setContentsMargins(28, 18, 28, 18)
        rec.setSpacing(10)
        rec.setAlignment(QtCore.Qt.AlignTop)

        back_icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg")

        rec.addWidget(
            self._make_back_btn(
                "Back to Sign In",
                lambda: self._slide_to(_P_SIGNIN),
                align="left",
                icon_path=back_icon_path,
            ),
            alignment=QtCore.Qt.AlignLeft,
        )

        recTitle = QtWidgets.QLabel("Reset Password")
        recTitle.setObjectName("recoverTitle")
        recTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        recTitle.setStyleSheet("color: rgba(50, 55, 65, 220); font-size: 12pt; font-weight: 700;")
        rec.addWidget(recTitle)

        recInfo = QtWidgets.QLabel("Contact your administrator to reset your password.")
        recInfo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        recInfo.setWordWrap(True)
        recInfo.setStyleSheet("color: rgba(100, 110, 120, 220); font-size: 8.5pt;")
        rec.addWidget(recInfo)
        # TODO: Currently placeholder workflow to contact admin to reset
        self.recoverEmail = GlassLineEdit()
        self.recoverEmail.setObjectName("recoverEmail")
        self.recoverEmail.setPlaceholderText("Email Address")
        self.recoverEmail.setFixedHeight(_INPUT_H)
        self.recoverEmail.setVisible(False)
        rec.addWidget(self.recoverEmail)

        self.sendResetBtn = QtWidgets.QPushButton("Send Reset Link")
        self.sendResetBtn.setObjectName("sendResetBtn")
        self.sendResetBtn.setFixedHeight(_BTN_H)
        self.sendResetBtn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.sendResetBtn.setStyleSheet(self._make_primary_btn_style())
        self.sendResetBtn.clicked.connect(self._on_send_reset)
        self.sendResetBtn.setVisible(False)
        rec.addWidget(self.sendResetBtn)

        self.recoverStatus = QtWidgets.QLabel("")
        self.recoverStatus.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.recoverStatus.setWordWrap(True)
        self.recoverStatus.setFixedHeight(34)
        self.recoverStatus.setStyleSheet(
            "color: rgba(46, 139, 87, 220); font-size: 8.5pt; font-weight: 500;"
        )
        rec.addWidget(self.recoverStatus)
        rec.addStretch()

        # Sliding panel
        self._slider = SlidingPanel(_CARD_W)
        self._slider.setObjectName("slidingPanel")
        self._slider.setStyleSheet(
            "QWidget#slidingPanel { background: transparent; border: none; }"
        )

        # Page layout: [0=SignIn, 1=Recover]
        self._slider.add_page(signInPage)
        self._slider.add_page(recoverPage)
        self._slider.setFixedHeight(_PAGE_H)

        def _init_slider() -> None:
            self._slider.finalize(_PAGE_H)
            # Position instantly at Sign In (page 0) with no animation
            self._slider._inner.move(0, 0)

        QtCore.QTimer.singleShot(0, _init_slider)

        self.loginCard = GlassCard(self.centralwidget)
        self.loginCard.setObjectName("loginCard")
        self.loginCard.setAttribute(QtCore.Qt.WA_StyledBackground, False)
        self.loginCard.setContentsMargins(0, 0, 0, 0)
        # NOTE: intentionally no QGraphicsDropShadowEffect here — the card
        # gets its own CardPopEffect below (via register_dismissable_card,
        # for the sign-out/sign-in pop), and Qt doesn't compose two graphics
        # effects on the same widget. The card's own manual border painting
        # (see GlassCard.paintEvent) provides sufficient visual separation
        # in place of the shadow.
        self.centralwidget.register_dismissable_card(self.loginCard)

        card_vbox = QtWidgets.QVBoxLayout(self.loginCard)
        card_vbox.setContentsMargins(0, 0, 0, 0)
        card_vbox.setSpacing(0)
        card_vbox.addWidget(self._slider)

        v_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        v_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        v_layout.addStretch(2)
        v_layout.addSpacing(20)
        v_layout.addWidget(self.loginCard, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        v_layout.addStretch(3)

        # Floating badge for error / info messages
        self.floating_badge = FloatingMessageBadgeWidget(
            MainWindow5,
            os.path.join(Architecture.get_path(), "QATCH", "icons", "clear.svg"),
        )

        # user_info proxy — legacy code calls .user_info.clear() / .setText(); route to badge
        self.user_info = _UserInfoProxy(self.floating_badge, self.loginCard)

        self._sessionTimer = QtCore.QTimer()
        self._sessionTimer.setSingleShot(True)
        self._sessionTimer.timeout.connect(self.check_user_session)
        self._sessionTimer.setInterval(1000 * 60 * 60)

        # Initialize settings and load remembered user
        self._settings = QtCore.QSettings("QATCH", "nanovisQ")
        self._load_remembered_user()

    def _make_input_style(self, error: bool = False) -> str:
        """Generates the QSS stylesheet for the GlassLineEdit components.

        Provides a translucent, frosted glass aesthetic for text inputs. If the
        error flag is set, it shifts the color palette to a high-visibility red
        to indicate invalid or missing user input.

        Args:
            error (bool, optional): Whether to generate the error-state styling.
                Defaults to False.

        Returns:
            str: The formatted Qt Style Sheet (QSS) string.
        """
        r = _INPUT_H // 2

        if error:
            return f"""
                QLineEdit {{
                    background-color: rgba(255, 230, 230, 160);
                    border: 1.5px solid rgba(230, 50, 50, 200);
                    border-radius: {r}px;
                    padding: 0px 15px;
                    color: rgba(200, 30, 30, 255);
                    font-size: 10pt;
                    selection-background-color: rgba(230, 50, 50, 80);
                }}
                QLineEdit:focus {{
                    background-color: rgba(255, 245, 245, 255);
                    border: 2px solid rgba(255, 50, 50, 255);
                }}
                QLineEdit QToolButton {{ 
                    background: transparent; 
                    border: none; 
                }}
                QLineEdit QToolButton:hover {{ 
                    background: rgba(230, 50, 50, 20); 
                    border-radius: 12px; 
                }}
            """

        return f"""
            QLineEdit {{
                background-color: rgba(255, 255, 255, 72);
                border: 1px solid rgba(255, 255, 255, 130);
                border-radius: {r}px;
                padding: 0px 15px;
                color: rgba(40, 50, 60, 230);
                font-size: 10pt;
                selection-background-color: rgba(10, 163, 230, 80);
            }}
            QLineEdit:hover {{
                background-color: rgba(255, 255, 255, 110);
                border-color: rgba(255, 255, 255, 200);
            }}
            QLineEdit:focus {{
                background-color: rgba(255, 255, 255, 145);
                border: 1.5px solid rgba(10, 163, 230, 140);
            }}
            QLineEdit QToolButton {{ 
                background: transparent; 
                border: none; 
            }}
            QLineEdit QToolButton:hover {{ 
                background: rgba(255, 255, 255, 60); 
                border-radius: 12px; 
            }}
        """

    def _make_primary_btn_style(self) -> str:
        """Generates the QSS stylesheet for the primary action buttons.

        Creates a rich, semi-transparent blue gradient button with custom borders
        to simulate physical depth and lighting. Includes states for hover,
        pressed, and disabled interactions.

        Returns:
            str: The formatted Qt Style Sheet (QSS) string.
        """
        r = _BTN_H // 2
        return f"""
            QPushButton {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(45, 165, 250, 210),
                    stop:1 rgba(15, 125, 210, 190)
                );
                border-top:    1px solid rgba(255, 255, 255, 100);
                border-left:   1px solid rgba(255, 255, 255, 50);
                border-right:  1px solid rgba(0, 80, 150, 50);
                border-bottom: 1px solid rgba(0, 80, 150, 80);
                border-radius: {r}px;
                color: rgba(255, 255, 255, 255);
                font-size: 9.5pt;
                font-weight: 600;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(65, 185, 255, 240),
                    stop:1 rgba(25, 145, 230, 220)
                );
                border-top: 1px solid rgba(255, 255, 255, 140);
            }}
            QPushButton:pressed {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(15, 115, 200, 220),
                    stop:1 rgba(5, 95, 160, 200)
                );
                border-top: 1px solid rgba(255, 255, 255, 40);
                border-bottom: 1px solid rgba(255, 255, 255, 20);
            }}
            QPushButton:disabled {{
                background: rgba(150, 170, 190, 100);
                color: rgba(255, 255, 255, 150);
                border: 1px solid rgba(255, 255, 255, 40);
            }}
        """

    @staticmethod
    def _make_back_btn(
        text: str, callback: Callable, align: str = "left", icon_path: str = ""
    ) -> QtWidgets.QPushButton:
        """Creates a styled, transparent text button used for backward navigation.

        Args:
            text (str): The label to display on the button (e.g., "← Back to Sign In").
            callback (typing.Callable): The function or method to execute when the button is clicked.
            align (str, optional): The text alignment within the button's bounding box
                ("left", "right", or "center"). Defaults to "left".

        Returns:
            QtWidgets.QPushButton: The configured, styled navigation button instance.
        """
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("backBtn")
        btn.setFixedHeight(24)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        # Add the icon if a path is provided
        if icon_path and os.path.exists(icon_path):
            btn.setIcon(QtGui.QIcon(icon_path))
            btn.setIconSize(QtCore.QSize(14, 14))

        btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: rgba(100, 110, 120, 200);
                border: none;
                font-size: 8.5pt;
                font-weight: 600;
                text-align: {align};
                padding-left: 2px;
            }}
            QPushButton:hover {{ 
                color: rgba(60, 60, 60, 220); 
                text-decoration: underline; 
            }}
        """)
        btn.clicked.connect(callback)
        return btn

    def _slide_to(self, page_idx: int) -> None:
        """Animates the sliding panel to a specified page and cleans up stale state.

        Transitions the UI to the requested layout index. If the user is returning to the
        primary Sign In page, this method automatically clears any text inputs, disables
        any active error states, or resets status messages left behind on the secondary
        pages (e.g., the Password Recovery page).

        Args:
            page_idx (int): The index of the target page to display (e.g., `_P_SIGNIN` or `_P_RECOVER`).
        """
        self._slider.slide_to(page_idx)

        if page_idx == _P_SIGNIN:
            # Reset the password recovery page to its default state
            self.recoverEmail.clear()
            self.recoverStatus.clear()
            self.sendResetBtn.setEnabled(True)

    def on_toggle_password_Action(self) -> None:
        """Toggles the visibility of the password text.

        Switches the password field between 'Password' mode (masked) and 'Normal'
        mode (plain text). This method also updates the trailing action icon
        to reflect the current visibility state.
        """
        # Determine new state
        self.password_shown = not self.password_shown

        # Select mode and icon based on new state
        mode = QtWidgets.QLineEdit.Normal if self.password_shown else QtWidgets.QLineEdit.Password
        icon = self.hiddenIcon if self.password_shown else self.visibleIcon

        # Apply changes
        self.user_password.setEchoMode(mode)
        self.togglepasswordAction.setIcon(icon)

    def update_caps_lock_state(self, caps_lock_on: bool) -> None:
        """Updates the UI indicator based on the system Caps Lock state.

        This method is typically invoked by the event filter. It updates the
        internal state tracking and modifies the visibility of the warning
        label to provide immediate feedback to the user.

        Args:
            caps_lock_on (bool): The current physical state of the Caps Lock key.
        """
        self.caps_lock_on = caps_lock_on
        self.caps_indicator.setText("Caps Lock is On" if caps_lock_on else "")

    def _on_send_reset(self) -> None:
        """Validates the recovery email and triggers the reset log/action.

        Checks for a non-empty email address. If valid, it provides positive
        feedback to the user and disables the submission button to prevent
        duplicate requests. If invalid, it displays a localized error message
        in red.
        """
        email = self.recoverEmail.text().strip()

        # Style constants for status messaging
        COLOR_ERR = "rgba(200, 30, 30, 230)"
        COLOR_SUC = "rgba(46, 139, 87, 220)"
        BASE_STYLE = "font-size: 8.5pt; font-weight: 500;"

        if not email:
            self.recoverStatus.setStyleSheet(f"color: {COLOR_ERR}; {BASE_STYLE}")
            self.recoverStatus.setText("Please enter your email address.")
            return

        # Log request and provide success feedback
        Log.i(f"Password reset requested for: {email}")

        self.recoverStatus.setStyleSheet(f"color: {COLOR_SUC}; {BASE_STYLE}")
        self.recoverStatus.setText(
            "If an account exists for that address,\na reset link has been sent."
        )
        self.sendResetBtn.setEnabled(False)

    def action_sign_in(self) -> None:
        """Executes the authentication flow and manages the transition to the main application.

        This method validates the input fields, performs a credential check against the
        UserProfiles database, and handles the post-login state. If successful, it:
          1. Synchronizes the 'Remember me' preference.
          2. Configures the parent window's user roles and UI labels.
          3. Determines the initial software mode (Run vs. Analyze) based on permissions.
          4. Starts the session inactivity timer.
          5. Clears the username field if 'Remember me' preference not selected.

        If authentication fails, it triggers error animations and displays a
        'Password invalid' message.
        """
        username = self.user_username.text().strip()

        if not username:
            self._shake_widget(self.user_username)
            self.user_username.set_error(True)
            self.floating_badge.show_message(
                "Please enter your username", is_error=True, parent_widget=self.loginCard
            )
            return

        if not self.user_password.text():
            self._shake_widget(self.user_password)
            self.user_password.set_error(True)
            self.floating_badge.show_message(
                "Password required", is_error=True, parent_widget=self.loginCard
            )
            return
        pwd = self.user_password.text()
        authenticated, filename, params = UserProfiles.auth(username, pwd, UserRoles.ANY)

        if authenticated:
            # Animated, not an instant clear() — slides/fades out in sync
            # with the card's "Deep Focus" dismissal below.
            self.floating_badge.slide_out()
            self.user_info.suppress = True
            # Persistence
            self._save_remembered_user(username)

            Log.i(f"Welcome, {params[0]}! Role: {params[2].name}.")
            name, init, role = params[0], params[1], params[2].value
            self._sessionTimer.start()
        else:
            name, init, role = None, None, 0

        self._clear_not_remembered_user()
        self._clear_credentials()

        if name is not None:
            controls = self.parent.ControlsWin
            controls.username.setText(f"User: {name}")
            controls.userrole = UserRoles(role)
            controls.signinout.setText("&Sign Out")
            controls.ui1.tool_User.setText(name)
            self.parent.AnalyzeProc.tool_User.setText(name)
            controls.set_signed_in_menu_state(True)
            self.parent.MainWin.ui0.mark_signed_in()

            # Restrict password management for non-admins
            if controls.userrole != UserRoles.ADMIN:
                controls.manage.setText("&Change Password...")
            has_capture_perm = UserProfiles().check(controls.userrole, UserRoles.CAPTURE)
            if has_capture_perm:
                self.parent.MainWin.ui0._set_run_mode(None)
            else:
                self.parent.MainWin.ui0._set_analyze_mode(None)

            # Dismiss the login overlay, revealing the now-current mode
            # underneath. Decoupled from the mode-switch calls above since
            # those may no-op (e.g. already on the matching mode at cold start).
            self.centralwidget.dismiss_for_signin()

            # Cleanup temporary download attributes
            if hasattr(self.parent, "url_download"):
                delattr(self.parent, "url_download")

            self.user_info.suppress = False
        else:
            self.user_info.suppress = False
            self.error_invalid()

    def _clear_credentials(self) -> None:
        """Securely clears sensitive input fields and resets visibility.

        Wipes the password field text and ensures the masking (echo mode)
        is restored if the user had previously toggled the password visibility 'on'.
        """
        self.user_password.clear()
        if self.password_shown:
            self.on_toggle_password_Action()

    def clear_form(self) -> None:
        """Performs a full reset of the login interface to its initial state.

        This is the primary cleanup method called during application startup,
        on Escape key presses, or upon user sign-out. It clears all input
        fields, resets error states, hides the Caps Lock indicator, and
        slides the panel back to the Sign In page.
        """
        self.loginCard.show()

        # Clear inputs and error styling
        self.user_username.clear()
        self.user_username.set_error(False)
        self.user_password.set_error(False)
        self._clear_credentials()

        # Reset indicators and navigation
        self.update_caps_lock_state(False)
        self._slide_to(_P_SIGNIN)

    def error_invalid(self, message: str = "Invalid Credentials") -> None:
        """Triggers the UI sequence for a failed authentication attempt.

        Applies red error styling to the password field, executes a shake
        animation, clears the invalid input, and displays a floating
        error badge with the provided message.

        Args:
            message (str, optional): The error text to display in the floating badge.
                Defaults to "Invalid Credentials".
        """
        self.user_password.set_error(True)
        self._shake_widget(self.user_password)
        self.user_password.clear()
        self.user_password.setFocus()
        self.floating_badge.show_message(message, is_error=True, parent_widget=self.loginCard)

    def error_loggedout(self) -> None:
        """Displays a logout notification and resets the password field state."""
        self.user_password.clear()
        self.user_password.set_error(False)
        self.show_signout_message()

    def error_expired(self) -> None:
        """Displays a session expiration warning and resets the password field state."""
        self.user_password.clear()
        self.user_password.set_error(False)
        self.floating_badge.show_message(
            "Your session has expired", is_error=True, parent_widget=self.loginCard
        )

    def show_signout_message(self) -> None:
        """Triggers a floating badge notification confirming the user has signed out.

        Drops in with a spring overshoot timed to land as the login card
        settles, per the "Deep Focus" sign-out reveal.
        """
        self.floating_badge.show_message(
            "You have been signed out.",
            is_error=True,
            parent_widget=self.loginCard,
            drop_in=True,
        )

    def check_user_session(self) -> None:
        valid, _ = UserProfiles().session_info()
        if not valid:
            if self.parent.ControlsWin.userrole != UserRoles.NONE:
                Log.w("User session has expired.")
                self.parent.ControlsWin.set_user_profile()
        else:
            Log.d("User session valid at hourly check.")
            self._sessionTimer.start()

    def _clear_not_remembered_user(self) -> None:
        """Clears the not remembered username from login form on sign in action.

        Retrieves the 'Remember me' toggle state from checkbox widget. If not enabled,
        it clears the entered username value in the field to be ready for the next sign
        in and to make sure their username is not shown again when this session ends.
        """
        not_remembered = not self.remember_me_cb.isChecked()

        if not_remembered:
            self.user_username.clear()

    def _load_remembered_user(self) -> None:
        """Loads the remembered username from system settings on startup.

        Retrieves the 'Remember me' toggle state from QSettings. If previously enabled,
        it restores the saved username and automatically shifts keyboard focus to the
        password input field for a quicker sign-in experience.
        """
        remembered = self._settings.value("login/remember_me", False, type=bool)
        self.remember_me_cb.setChecked(remembered)

        if remembered:
            saved_user = self._settings.value("login/username", "", type=str)
            if saved_user:
                self.user_username.setText(saved_user)
                self.user_password.setFocus()

    def _save_remembered_user(self, username: str) -> None:
        """Saves or clears the remembered username based on the toggle state.

        Synchronizes the current state of the 'Remember me' checkbox with QSettings.
        If checked, the username is stored for future sessions. If unchecked,
        any previously stored username is wiped from the system's settings registry.

        Args:
            username (str): The authenticated username to store if the toggle is active.
        """
        is_remembered = self.remember_me_cb.isChecked()
        self._settings.setValue("login/remember_me", is_remembered)

        if is_remembered:
            self._settings.setValue("login/username", username)
        else:
            self._settings.remove("login/username")

    def retranslateUi(self, MainWindow5: QtWidgets.QMainWindow) -> None:
        """Sets the window titles and localizable strings for the interface.

        Configures the primary window title using application constants and
        loads the software icon from the localized asset path.

        Args:
            MainWindow5 (QtWidgets.QMainWindow): The window instance to update.
        """
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        MainWindow5.setWindowIcon(QtGui.QIcon(icon_path))
        title_text = f"{Constants.app_title} {Constants.app_version} - Login"
        MainWindow5.setWindowTitle(_translate("MainWindow5", title_text))

    def _shake_widget(self, widget: QtWidgets.QWidget) -> None:
        """Executes a rapid horizontal jiggle animation to provide error feedback.

        Uses a QPropertyAnimation to manipulate the 'pos' property of the widget.
        The animation uses a series of decaying keyframes to simulate a physical
        shake that settles back at the original position.

        Args:
            widget (QtWidgets.QWidget): The target widget (e.g., LineEdit) to animate.
        """
        if not widget or not widget.isVisible():
            return

        self._shake_anim = QtCore.QPropertyAnimation(widget, b"pos")
        self._shake_anim.setDuration(380)

        base = widget.pos()
        self._shake_anim.setKeyValueAt(0.0, base)
        self._shake_anim.setKeyValueAt(0.1, base + QtCore.QPoint(-6, 0))
        self._shake_anim.setKeyValueAt(0.3, base + QtCore.QPoint(6, 0))
        self._shake_anim.setKeyValueAt(0.5, base + QtCore.QPoint(-4, 0))
        self._shake_anim.setKeyValueAt(0.7, base + QtCore.QPoint(4, 0))
        self._shake_anim.setKeyValueAt(0.9, base + QtCore.QPoint(-2, 0))
        self._shake_anim.setKeyValueAt(1.0, base)

        self._shake_anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
