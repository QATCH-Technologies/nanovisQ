"""
QATCH.ui.components.overlay_shell

Shared infrastructure for constructing, styling, positioning, animating, and
managing application overlays.

This module centralizes the common visual and lifecycle behavior used by
QATCH overlay widgets, including themed headers, floating corner controls,
fullscreen icons, fade transitions, scrim rendering, parent-relative geometry,
and reveal/close sequencing. The shared helpers ensure that overlays use
consistent sizing, spacing, theme colors, button placement, and animation
behavior without duplicating implementation details across individual
widgets.

The module is intentionally focused on reusable overlay mechanics rather than
specific application content. Individual overlay widgets are responsible for
building their own content layouts and, where necessary, overriding the
provided hooks for custom panel appearance, fullscreen transitions, button
placement, and close-time teardown.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    close_button_qss,
    glass_panel_qss,
    tok_css,
)

CORNER_BUTTON_SIZE = 28
CORNER_ICON_SIZE = QtCore.QSize(14, 14)
CORNER_BUTTON_PADDING = 14
CORNER_BUTTON_SPACING = 6

DEFAULT_MARGIN_PCT = 0.175
DEFAULT_PANEL_ALPHA = 215

OPEN_FADE_DURATION = 200
OPEN_FADE_EASING = QtCore.QEasingCurve.OutQuad
CLOSE_FADE_DURATION = 180
CLOSE_FADE_EASING = QtCore.QEasingCurve.InQuad
SCRIM_MAX_ALPHA = 65

FULLSCREEN_ANIM_EASING = QtCore.QEasingCurve.InOutCubic


def build_overlay_title(
    icon_path: str,
    title_text: str,
    *,
    icon_size: int = 16,
) -> tuple:
    """Build a themed icon and title pair for an overlay header.

    Creates a `QLabel` containing a theme-tinted icon and a bold title
    label using the standard overlay header styling. The returned
    refresh callback can be invoked when the application theme changes to
    update both the icon tint and title color.

    Args:
        icon_path: Path to the icon resource used for the overlay header.
        title_text: Text displayed in the title label.
        icon_size: Width and height of the icon in pixels.

    Returns:
        tuple: A three-element tuple containing:
            - `icon_label`: `QLabel` displaying the tinted header icon.
            - `title_label`: `QLabel` displaying the styled title text.
            - `refresh_fn`: Callable that reapplies the current theme to
              the icon and title. This should be invoked by the host when
              the application switches between light and dark themes.

    Note:
        The initial theme is applied before the function returns. The
        refresh callback obtains the current theme tokens from
        `ThemeManager` each time it is called, ensuring the header remains
        synchronized with the active application theme.
    """
    icon_label = QtWidgets.QLabel()
    icon_label.setFixedSize(icon_size, icon_size)
    title_label = QtWidgets.QLabel(title_text)

    def _refresh() -> None:
        tok = ThemeManager.instance().tokens()
        icon_label.setPixmap(
            tinted_icon(
                icon_path,
                QtGui.QColor(*tok["flat_text"]),
                size=icon_size,
            ).pixmap(icon_size, icon_size)
        )
        title_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-weight: bold; "
            "font-size: 13px; background: transparent; }}"
        )

    _refresh()
    return icon_label, title_label, _refresh


def build_corner_button(
    parent: QtWidgets.QWidget,
    text: str,
    tooltip: str,
    *,
    size: int = CORNER_BUTTON_SIZE,
) -> QtWidgets.QPushButton:
    """Build a square button for an overlay's floating corner controls.

    Creates a minimally configured `QPushButton` intended for controls
    such as fullscreen and close buttons positioned in the top-right corner
    of an overlay. The button is deliberately left without icon, stylesheet,
    or click-handler configuration so callers can apply their own behavior
    and visual treatment.

    Args:
        parent: Parent widget that owns the button.
        text: Text assigned to the button. Typically empty when the button
            will display an icon.
        tooltip: Tooltip displayed when the cursor hovers over the button.
        size: Width and height of the square button, in pixels. Defaults to
            `CORNER_BUTTON_SIZE`.

    Returns:
        QtWidgets.QPushButton: A fixed-size, pointer-cursor button configured
        for use as an overlay corner control.
    """
    btn = QtWidgets.QPushButton(text, parent)
    btn.setFixedSize(size, size)
    btn.setToolTip(tooltip)
    btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
    return btn


def position_corner_buttons(
    width: int,
    mx: int,
    my: int,
    buttons,
    *,
    padding: int = CORNER_BUTTON_PADDING,
    spacing: int = CORNER_BUTTON_SPACING,
    size: int = CORNER_BUTTON_SIZE,
) -> None:
    """Position overlay corner buttons in a right-to-left button row.

    Arranges same-sized square buttons inset from the panel's top-right
    corner. Buttons are positioned from right to left, with the first
    non-`None` button placed closest to the corner. This provides the
    standard floating corner-button layout shared by overlay widgets such
    as `DataManagementWidget` and `UserProfilesManagerWidget`.

    `None` entries are ignored, allowing callers to conditionally omit
    controls while preserving the intended ordering of the remaining
    buttons.

    Args:
        width: Width of the containing panel, in pixels.
        mx: Horizontal offset of the panel's content or margin origin, in
            pixels.
        my: Vertical offset of the panel's content or margin origin, in
            pixels.
        buttons: Iterable of buttons to position. Buttons should be ordered
            outermost-first, meaning the first button is placed closest to
            the panel's top-right corner. `None` entries are skipped.
        padding: Distance between the button row and the panel's top and
            right edges, in pixels. Defaults to `CORNER_BUTTON_PADDING`.
        spacing: Horizontal spacing between adjacent buttons, in pixels.
            Defaults to `CORNER_BUTTON_SPACING`.
        size: Width and height of each button, in pixels. Defaults to
            `CORNER_BUTTON_SIZE`.

    Returns:
        None.

    Example:
        Passing `[btn_close, btn_fullscreen]` places the close button
        nearest the top-right corner and the fullscreen button immediately
        to its left. Passing `[btn_close, None]` omits the fullscreen
        button.
    """
    x = width - mx - padding - size
    y = my + padding

    for btn in buttons:
        if btn is None:
            continue

        btn.setGeometry(x, y, size, size)
        btn.raise_()
        x -= size + spacing


def rebuild_fullscreen_icons(
    host,
    icon_expand_path: str,
    icon_collapse_path: str,
    *,
    size: int = 14,
) -> None:
    """Rebuild the themed icons used by an overlay fullscreen button.

    Generates normal and hover variants of the fullscreen toggle icon using
    the active theme colors. The appropriate expand or collapse icon is
    selected based on the host's current fullscreen state. If the host has a
    fullscreen button, its displayed icon is immediately refreshed while
    preserving its current hover state.

    This helper provides a shared icon-generation recipe for overlay widgets,
    ensuring that fullscreen controls consistently use the muted theme color
    in their normal state and the full text color when hovered.

    Args:
        host: Overlay widget containing the fullscreen state and optional
            fullscreen button. The host is expected to expose
            `_is_fullscreen` and may expose `btn_fullscreen`,
            `_fs_normal_icon`, and `_fs_hover_icon` attributes.
        icon_expand_path: Path to the icon displayed when the host is not in
            fullscreen mode.
        icon_collapse_path: Path to the icon displayed when the host is
            currently in fullscreen mode.
        size: Width and height, in pixels, used when generating the tinted
            icons.

    Returns:
        None.
    """
    tok = ThemeManager.instance().tokens()

    normal_color = QtGui.QColor(*tok["flat_text_muted"])
    hover_color = QtGui.QColor(*tok["flat_text"])

    icon_path = icon_collapse_path if getattr(host, "_is_fullscreen", False) else icon_expand_path

    host._fs_normal_icon = tinted_icon(icon_path, normal_color, size=size)
    host._fs_hover_icon = tinted_icon(icon_path, hover_color, size=size)

    btn = getattr(host, "btn_fullscreen", None)
    if btn is not None:
        btn.setIcon(host._fs_hover_icon if btn.underMouse() else host._fs_normal_icon)


def run_variant_animation(
    host,
    anim_attr: str,
    *,
    duration: int,
    easing,
    on_step,
    on_finish=None,
) -> None:
    """Run a reusable `QVariantAnimation` with lifecycle management.

    Creates and starts a `QVariantAnimation` that progresses from `0.0`
    to `1.0` using the specified duration and easing curve. Any existing
    animation stored on `host` under `anim_attr` is stopped, disconnected,
    and scheduled for deletion before the new animation is created.

    The helper manages the common animation lifecycle while leaving the
    actual interpolation logic to the caller. This allows different overlay
    animations to reuse the same machinery for properties such as margins,
    opacity, border width, or corner radius.

    Args:
        host: Object that owns the animation. The animation is also created
            with `host` as its Qt parent.
        anim_attr: Attribute name on `host` used to store the active
            `QVariantAnimation` instance.
        duration: Animation duration, in milliseconds.
        easing: Qt easing curve passed to
            `QVariantAnimation.setEasingCurve()`.
        on_step: Callback invoked on every animation tick with the current
            eased progress as a `float` in the range `[0.0, 1.0]`.
        on_finish: Optional callback invoked once after the final
            `on_step(1.0)` call and after the animation has been cleared
            from `host`.

    Returns:
        None.

    Note:
        Starting a new animation automatically replaces any previous
        animation stored under `anim_attr`. The final progress value of
        `1.0` is explicitly emitted from the `finished` handler to ensure
        that the target state is always applied, even if the animation's
        final `valueChanged` signal does not provide the exact endpoint.
    """
    old = getattr(host, anim_attr, None)
    if old is not None:
        try:
            old.stop()
            old.valueChanged.disconnect()
        except (TypeError, RuntimeError):
            pass

        old.deleteLater()
        setattr(host, anim_attr, None)

    anim = QtCore.QVariantAnimation(host)
    anim.setDuration(duration)
    anim.setEasingCurve(easing)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.valueChanged.connect(lambda t: on_step(float(t)))

    def _done():
        on_step(1.0)

        done = getattr(host, anim_attr, None)
        setattr(host, anim_attr, None)

        if done is not None:
            done.deleteLater()

        if on_finish is not None:
            on_finish()

    anim.finished.connect(_done)
    setattr(host, anim_attr, anim)
    anim.start()


class OverlayFadeMixin:
    """Provide reusable fade transitions and default overlay close behavior.

    This mixin implements the shared scrim, panel, and lifecycle
    behavior used by overlay widgets. It provides the animation teardown
    required to safely interrupt an active fade and defines extension points
    for overlays that need custom open or close transitions.

    Host requirements:
        The host class must initialize the following attributes, typically
        through `_init_overlay_shell` provided by
        `OverlayLifecycleMixin`:

        - `_scrim_alpha`: Current opacity of the overlay scrim.
        - `_panel_alpha`: Current opacity of the overlay panel.
        - `_glass_opacity`: `QGraphicsOpacityEffect` applied to
          `glass_frame`.
        - `_fade_anim`: Currently active fade animation, or `None`.
        - `_closing`: Whether the overlay is currently closing.
        - `_close_in_progress`: Whether close processing is underway.
        - `glass_frame`: The overlay's glass-panel widget.

    Override points:
        `_animate_open`:
            Override to customize the fade-in animation endpoints or timing.

        `_animate_close`:
            Override to customize the fade-out animation endpoints or timing.

        `_do_close`:
            Override to perform additional close-time teardown, such as
            stopping a shared service or terminating a running slide
            transition. Implementations should call `super()._do_close()`
            to preserve the shared reset-and-refit cleanup sequence.

    Note:
        Animation lifecycle management is centralized here so derived
        overlays can safely interrupt and replace fade transitions without
        leaving active signal connections or deferred animations behind.
    """

    def _stop_anim(self) -> None:
        """Stop and fully tear down the active fade animation.

        Safely stops the animation stored in `_fade_anim`, disconnects its
        `valueChanged` and `finished` signals, schedules it for deletion,
        and clears the host's animation reference. This prevents a previously
        started fade from continuing to update the overlay after the
        transition has been interrupted.

        Returns:
            None.

        Note:
            Signal disconnection errors are intentionally ignored because a
            Qt signal may already have been disconnected or its underlying
            object may already have been deleted.
        """
        anim = getattr(self, "_fade_anim", None)
        if anim is not None:
            try:
                anim.stop()
                anim.valueChanged.disconnect()
                anim.finished.disconnect()
            except (TypeError, RuntimeError):
                pass

            anim.deleteLater()
            self._fade_anim = None

    def _set_glass_opacity(self, frac: float) -> None:
        """Set the frame opacity to a clamped fractional value.

        Updates the opacity of the host's `QGraphicsOpacityEffect` when one
        is available. Values outside the valid opacity range are clamped to
        `[0.0, 1.0]`.

        Args:
            frac: Desired opacity as a fractional value, where `0.0` is
                fully transparent and `1.0` is fully opaque.

        Returns:
            None.
        """
        frac = max(0.0, min(1.0, float(frac)))
        effect = getattr(self, "_glass_opacity", None)
        if effect is not None:
            effect.setOpacity(frac)

    def _run_fade(
        self,
        scrim_from,
        scrim_to,
        op_from,
        op_to,
        duration,
        easing,
        on_done=None,
        opacity_effect=None,
    ) -> None:
        """Animate the overlay scrim and opacity between fixed endpoints.

        Runs a `QVariantAnimation` that simultaneously interpolates the
        scrim alpha and the opacity of a `QGraphicsOpacityEffect`. The
        scrim is updated through the host's `paintEvent` using
        `_scrim_alpha`, avoiding per-frame stylesheet changes and their
        associated rendering overhead.

        Any existing fade animation is stopped and cleaned up before the new
        animation begins. The opacity effect defaults to the host's
        `_glass_opacity` effect, but callers may provide an alternate
        effect when a different visual element, such as a static pixmap
        proxy, needs to participate in the transition.

        Args:
            scrim_from: Starting scrim alpha value.
            scrim_to: Ending scrim alpha value.
            op_from: Starting opacity-effect value in the range `[0.0, 1.0]`.
            op_to: Ending opacity-effect value in the range `[0.0, 1.0]`.
            duration: Animation duration, in milliseconds.
            easing: Qt easing curve used to interpolate the transition.
            on_done: Optional callback invoked after the fade animation
                completes.
            opacity_effect: Optional `QGraphicsOpacityEffect` to animate.
                When `None`, `self._glass_opacity` is used.

        Returns:
            None.
        """
        self._stop_anim()
        effect = opacity_effect if opacity_effect is not None else self._glass_opacity
        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(duration)
        anim.setEasingCurve(easing)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def _set_opacity(frac):
            if effect is not None:
                effect.setOpacity(max(0.0, min(1.0, float(frac))))

        def _step(t):
            self._scrim_alpha = int(scrim_from + (scrim_to - scrim_from) * t)
            _set_opacity(op_from + (op_to - op_from) * t)
            self.update()  # repaint the scrim only

        def _settle():
            self._scrim_alpha = int(scrim_to)
            _set_opacity(op_to)
            self.update()
            done_anim = self._fade_anim
            self._fade_anim = None
            if done_anim is not None:
                done_anim.deleteLater()
            if on_done is not None:
                on_done()

        anim.valueChanged.connect(_step)
        anim.finished.connect(_settle)
        self._fade_anim = anim
        anim.start()

    def _animate_open(self) -> None:
        """Animate the overlay from its hidden state to full visibility.

        Resets the scrim and panel opacity to fully transparent, forces
        an immediate repaint, and then starts the standard fade-in transition
        to the configured maximum scrim alpha and full panel opacity.

        Returns:
            None.
        """
        self._scrim_alpha = 0
        self._set_glass_opacity(0.0)
        self.update()

        self._run_fade(
            scrim_from=0,
            scrim_to=SCRIM_MAX_ALPHA,
            op_from=0.0,
            op_to=1.0,
            duration=OPEN_FADE_DURATION,
            easing=OPEN_FADE_EASING,
        )

    def _animate_close(self) -> None:
        """Animate the overlay from its current state to fully hidden.

        Fades the scrim to transparent while simultaneously fading the
        panel to zero opacity. The current opacity is used as the
        starting point so an interrupted or partially completed transition
        can continue smoothly. Once the fade completes, `_do_close` is
        invoked to perform the overlay's close-time teardown.

        Returns:
            None.
        """
        cur_op = self._glass_opacity.opacity() if self._glass_opacity else 1.0

        self._run_fade(
            scrim_from=self._scrim_alpha,
            scrim_to=0,
            op_from=cur_op,
            op_to=0.0,
            duration=CLOSE_FADE_DURATION,
            easing=CLOSE_FADE_EASING,
            on_done=self._do_close,
        )

    def _do_close(self) -> None:
        """Finalize the overlay close after the fade-out has completed.

        Stops any remaining fade animation, marks the overlay as being in its
        internal close phase, and invokes the normal Qt `close()` operation.
        The `_closing` flag allows `closeEvent` to accept this second
        close request instead of starting another fade transition.

        After the widget has been closed, the method restores the overlay's
        reusable visual state by clearing the close-in-progress flag,
        displaying the frame, restoring full opacity, and
        refitting the overlay to its parent.

        Subclasses that require additional close-time teardown should
        override this method and call `super()._do_close()` as part of their
        implementation.

        Returns:
            None.
        """
        self._stop_anim()
        self._closing = True
        self.close()  # closeEvent sees _closing=True -> accepts -> Qt calls hide()
        self._closing = False
        self._close_in_progress = False
        self.glass_frame.show()
        self._set_glass_opacity(1.0)
        self._refit_to_parent()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Intercept a close request and replace it with a fade-out.

        Normal close requests are initially ignored so the overlay can run
        its fade-out animation before actually closing. Once the fade has
        completed, `_do_close` sets `_closing` and invokes `close()`
        again, allowing this handler to accept the event normally.

        Repeated close requests received while a close animation is already
        in progress are ignored to prevent multiple close animations from
        being started concurrently.

        Args:
            event: Qt close event being handled.

        Returns:
            None.

        Note:
            The `_closing` flag distinguishes the internal close operation
            from an external close request. This allows the final close event
            to pass through without recursively triggering another fade-out.
        """
        if self._closing:
            event.accept()
            return
        anim = getattr(self, "_fade_anim", None)
        if (
            anim is not None
            and self._close_in_progress
            and anim.state() == QtCore.QAbstractAnimation.State.Running
        ):
            event.ignore()
            return
        event.ignore()
        self._close_in_progress = True
        self._animate_close()


class OverlayLifecycleMixin(OverlayFadeMixin):
    """Provide shared overlay construction, reveal, geometry, and lifecycle behavior.

    Extends `OverlayFadeMixin` with the common infrastructure required by
    overlay widgets, including shell construction, reveal-on-open behavior,
    parent-tracking geometry, and shared Qt event handling.

    The mixin is designed to provide sensible defaults while allowing
    individual overlays to customize specific aspects of their appearance,
    positioning, and reveal lifecycle.

    Note:
        Header construction is centralized here so all overlay widgets share
        consistent icon sizing, title styling, button placement, and
        fullscreen-control behavior.
    """

    def _build_overlay_header(
        self, icon_path: str, title_text: str, *, fullscreen: bool = False, icon_size: int = 16
    ) -> QtWidgets.QHBoxLayout:
        """Build the shared overlay header and its corner controls.

        Creates a left-aligned header containing a theme-aware icon and bold
        title. The close button and optional fullscreen button are created
        separately as floating controls in the panel's top-right corner and
        are positioned later by `_position_overlay_buttons`.

        The method stores the created widgets on the host as
        `window_icon_label`, `window_title_label`, `btn_close`, and
        `btn_fullscreen`. It also stores the title refresh callback as
        `_refresh_title` for use by the host's theme-change handler.

        When fullscreen support is enabled, the host's `_rebuild_fs_icons`
        method is called to initialize the fullscreen icons. The host must
        also provide a `toggle_fullscreen()` method.

        Args:
            icon_path: Path to the icon displayed beside the overlay title.
            title_text: Text displayed in the overlay header.
            fullscreen: Whether to create a fullscreen toggle button.
            icon_size: Width and height of the header icon, in pixels.

        Returns:
            QtWidgets.QHBoxLayout: Header layout containing the icon, title,
            and stretch used to left-align the header content.

        Raises:
            AttributeError: If `fullscreen` is `True` and the host does
                not provide the required `_rebuild_fs_icons` or
                `toggle_fullscreen` methods when the corresponding
                operations are invoked.
        """
        self.window_icon_label, self.window_title_label, self._refresh_title = build_overlay_title(
            icon_path, title_text, icon_size=icon_size
        )
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(self.window_icon_label)
        header_layout.addWidget(self.window_title_label)
        header_layout.addStretch()

        if fullscreen:
            self._rebuild_fs_icons()
            self.btn_fullscreen = build_corner_button(self, "", "Toggle Fullscreen")
            self.btn_fullscreen.setIcon(self._fs_normal_icon)
            self.btn_fullscreen.setIconSize(CORNER_ICON_SIZE)
            self.btn_fullscreen.setStyleSheet(
                "QPushButton { background: transparent; border: none; }"
            )
            self.btn_fullscreen.installEventFilter(self)
            self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        else:
            self.btn_fullscreen = None

        self.btn_close = build_corner_button(self, "x", "Close")
        self.btn_close.setStyleSheet(close_button_qss())
        self.btn_close.clicked.connect(self.close)

        return header_layout

    def _refresh_header_theme(self) -> None:
        """Refresh the overlay header to reflect the active application theme.

        Reapplies the theme-dependent icon tint and title styling, refreshes the
        close-button stylesheet, and rebuilds the fullscreen icons when a
        fullscreen toggle is present.

        This method should be called by the host widget's theme-change handler
        whenever the application's active theme changes.

        Returns:
            None.
        """
        self._refresh_title()
        self.btn_close.setStyleSheet(close_button_qss())
        if self.btn_fullscreen is not None:
            self._rebuild_fs_icons()

    def _init_overlay_shell(
        self,
        parent,
        object_name: str,
        *,
        panel_alpha: int = DEFAULT_PANEL_ALPHA,
        margin_pct: float = DEFAULT_MARGIN_PCT,
        content_margins: tuple = (20, 14, 20, 20),
        content_spacing: int = 14,
        glass_frame: QtWidgets.QFrame = None,
        glass_qss: bool = True,
    ) -> None:
        """Initialize the shared overlay shell and rendering infrastructure.

        Configures the overlay's scrim and panel state, creates the base and
        content layouts, establishes the composited panel opacity effect,
        and installs event filters used to track parent and top-level window
        resizing.

        This method should be called exactly once immediately after
        `super().__init__(parent)`. The derived widget should populate its
        content using `self.main_layout` and then call
        `_finish_overlay_shell()` to complete the overlay initialization.

        Args:
            parent: Parent widget that owns the overlay. When provided, resize
                event filters are installed on both the parent and its top-level
                window when they are different objects.
            object_name: Qt object name assigned to a newly created frame
                and used by the default panel stylesheet.
            panel_alpha: Default alpha value used by the panel appearance.
            margin_pct: Default fractional margin used when fitting the overlay
                panel within its parent.
            content_margins: Four-element sequence containing the left, top,
                right, and bottom margins for `main_layout`, in pixels.
            content_spacing: Vertical spacing between widgets in
                `main_layout`, in pixels.
            glass_frame: Optional pre-built `QFrame` used as the overlay's
                panel. This is useful for custom-painted panels such as
                `DataManagementWidget`'s `_GlassPanel`. When provided,
                `glass_qss` is ignored, and the frame is assumed to already be
                parented to `self` and have its object name configured.
            glass_qss: Whether to apply the standard `glass_panel_qss` styling
                when creating the default frame. Ignored when
                `glass_frame` is provided.

        Returns:
            None.

        Note:
            The overlay uses a child-widget scrim rather than
            `WA_TranslucentBackground` because that attribute is intended for
            top-level windows. The scrim is instead rendered directly by the
            overlay's `paintEvent` over the parent's backing store.

            Opacity is animated through a `QGraphicsOpacityEffect` rather
            than by repeatedly modifying the panel stylesheet. This avoids
            reparsing and repolishing the entire widget subtree on every
            animation frame.
        """
        self.parent = parent
        self._overlay_object_name = object_name

        # Child-widget scrim: WA_TranslucentBackground is top-level-only, so
        # disable Qt's auto-fill instead and let paintEvent draw the scrim
        # directly over the parent's backing store.
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        if parent is not None:
            parent.installEventFilter(self)
            top = parent.window()
            if top is not parent:
                top.installEventFilter(self)

        self._scrim_alpha = 0
        self._panel_alpha = panel_alpha
        self._closing = False
        self._close_in_progress = False
        self._revealed = False
        self._fade_anim = None
        self._default_margin_pct = margin_pct
        self._is_fullscreen = False
        self._fs_anim = None

        self.base_layout = QtWidgets.QVBoxLayout(self)
        self.base_layout.setContentsMargins(0, 0, 0, 0)

        if glass_frame is not None:
            self.glass_frame = glass_frame
        else:
            self.glass_frame = QtWidgets.QFrame(self)
            self.glass_frame.setObjectName(object_name)
            if glass_qss:
                self.glass_frame.setStyleSheet(glass_panel_qss(object_name, panel_alpha, 1.5, 12))

        # Fade via a composited opacity effect
        self._glass_opacity = QtWidgets.QGraphicsOpacityEffect(self.glass_frame)
        self._glass_opacity.setOpacity(0.0)
        self.glass_frame.setGraphicsEffect(self._glass_opacity)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(*content_margins)
        self.main_layout.setSpacing(content_spacing)

        self.base_layout.addWidget(self.glass_frame)

    def _finish_overlay_shell(self) -> None:
        """Finalize the overlay shell after its content has been constructed.

        Initializes the overlay in a hidden state, clears the scrim, hides the
        glass frame, and fits the overlay to its parent's content area. Performing
        the initial geometry calculation before the first `show()` prevents the
        overlay from briefly appearing at an incorrect or undersized geometry
        during its first reveal.

        This method should be called once after the host widget has finished
        constructing its content and configuring its layout.

        Returns:
            None.
        """
        self.hide()
        self._scrim_alpha = 0
        self.glass_frame.hide()
        self._refit_to_parent()

    def _current_margin_frac(self) -> float:
        """Return the current overlay panel margin fraction.

        Uses a zero margin when the overlay is in fullscreen mode; otherwise,
        returns the configured default margin percentage.

        Returns:
            float: Margin fraction applied when fitting the overlay panel to its
            parent. Returns `0.0` in fullscreen mode and
            `self._default_margin_pct` otherwise.
        """
        return 0.0 if getattr(self, "_is_fullscreen", False) else self._default_margin_pct

    def _apply_panel_appearance(self, frac: float) -> None:
        """Apply panel-specific appearance changes for an inset transition.

        Hook invoked when the panel's inset fraction changes, such as during a
        fullscreen-toggle animation. The default implementation intentionally
        performs no work because overlays using a static `glass_panel_qss`
        stylesheet do not require per-frame appearance updates.

        Subclasses may override this method to animate visual properties such as
        panel opacity, border, radius, or other styling characteristics based on
        the current inset fraction.

        Args:
            frac: Current panel inset fraction used by the active transition.
                The expected range is typically `[0.0, 1.0]`.

        Returns:
            None.
        """

    def _position_overlay_buttons(self, mx: int, my: int) -> None:
        """Position the overlay's floating corner buttons.

        Hook used to position the header's close and optional fullscreen buttons
        within the overlay's top-right corner. The default implementation uses
        the shared `position_corner_buttons` layout helper and safely ignores
        buttons that have not been created.

        Subclasses may override this method when their close control is managed
        by a conventional header layout rather than the shared floating-button
        arrangement.

        Args:
            mx: Horizontal panel margin or inset, in pixels.
            my: Vertical panel margin or inset, in pixels.

        Returns:
            None.
        """
        position_corner_buttons(
            QtWidgets.QWidget.width(self),
            mx,
            my,
            [
                getattr(self, "btn_close", None),
                getattr(self, "btn_fullscreen", None),
            ],
        )

    def _apply_margin_frac(self, frac: float) -> None:
        """Apply a proportional inset to the overlay panel.

        Calculates horizontal and vertical margins from the overlay's current
        dimensions and the supplied fractional inset, then applies the margins
        symmetrically to `base_layout`. The panel appearance and floating
        corner-button positions are updated afterward so they remain synchronized
        with the current inset.

        Args:
            frac: Fraction of the overlay's width and height to use as the
                surrounding inset. A value of `0.0` produces no inset, while
                larger values move the panel farther from the overlay edges.

        Returns:
            None.
        """
        w, h = QtWidgets.QWidget.width(self), QtWidgets.QWidget.height(self)
        mx = int(w * frac)
        my = int(h * frac)
        self.base_layout.setContentsMargins(mx, my, mx, my)
        self._apply_panel_appearance(frac)
        self._position_overlay_buttons(mx, my)

    def _refit_to_parent(self) -> None:
        """Fit the overlay to its parent and apply the current panel inset.

        Resizes the overlay to match its parent's content rectangle and then
        applies the current margin fraction to inset the panel within the
        overlay. When no parent is available, the primary screen's available
        geometry is used as the fallback.

        If a fullscreen-toggle animation is currently running, the method leaves
        the panel margins unchanged. This prevents parent resize or move events
        from interrupting the animation by prematurely snapping the panel to its
        normal or fullscreen margin.

        Returns:
            None.

        Note:
            Overlay button movement can cause additional geometry-related events
            to be processed through the event filter. During a fullscreen
            transition, the active `_fs_anim` therefore owns the panel margins
            until the animation completes.
        """
        if self.parent is None:
            geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        else:
            geo = self.parent.rect()
        self.setGeometry(geo)

        # A running fullscreen-toggle animation owns the margins for its
        # duration
        fs_anim = getattr(self, "_fs_anim", None)
        if fs_anim is not None and fs_anim.state() == QtCore.QAbstractAnimation.Running:
            return

        self._apply_margin_frac(self._current_margin_frac())

    def _on_before_reveal(self) -> None:
        """Hook for work that must run immediately before the overlay is shown.

        Subclasses may override this method to perform synchronous preparation
        immediately before the overlay becomes visible. The default
        implementation intentionally performs no work.

        Returns:
            None.
        """

    def _on_after_reveal(self) -> None:
        """Hook for work that runs after the overlay reveal begins.

        Subclasses may override this method to perform additional initialization
        or UI updates after the panel has been shown and the opening fade
        animation has started. The default implementation intentionally performs
        no work.

        Returns:
            None.
        """

    def _raise_overlay_buttons(self) -> None:
        """Raise the overlay's floating corner buttons above the panel.

        Raises the close and fullscreen buttons, when present, so they remain
        visually above the overlay's frame and other child widgets.

        Returns:
            None.
        """
        for btn in (getattr(self, "btn_close", None), getattr(self, "btn_fullscreen", None)):
            if btn is not None:
                btn.raise_()

    def setVisible(self, visible: bool) -> None:
        """Show or hide the overlay while preventing an unstyled reveal flash.

        Overrides the standard Qt visibility operation to coordinate the
        overlay's geometry, glass-panel visibility, layout activation, and
        opening animation. When showing the overlay, the panel remains hidden
        until the next event-loop tick so all geometry and layout calculations
        have settled before the overlay becomes visible.

        The reveal is performed atomically on the deferred event-loop callback:
        the overlay is made visible, its panel is shown and raised, corner
        buttons are raised, and the opening fade is started without exposing an
        intermediate frame containing an incorrectly sized or blank panel.

        When hiding the overlay, visibility is delegated directly to the base
        `QWidget` implementation.

        Args:
            visible: `True` to reveal the overlay or `False` to hide it.

        Returns:
            None.
        """
        if visible and not self.isVisible():
            self._on_before_reveal()
            self._scrim_alpha = 0
            self._panel_alpha = 0
            self.glass_frame.hide()
            self._refit_to_parent()

            def _reveal():
                self._refit_to_parent()
                if self.layout() is not None:
                    self.layout().activate()
                self._revealed = True
                self._set_glass_opacity(0.0)
                QtWidgets.QWidget.setVisible(self, True)
                self._refit_to_parent()
                if self.layout() is not None:
                    self.layout().activate()
                self.glass_frame.show()
                self.glass_frame.raise_()
                self._raise_overlay_buttons()
                self.update()
                self._animate_open()
                self._on_after_reveal()

            QtCore.QTimer.singleShot(0, _reveal)
            return
        QtWidgets.QWidget.setVisible(self, visible)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Handle the overlay becoming visible.

        Refits the overlay to its parent before completing the show event and
        raises the overlay above its sibling widgets. This ensures the overlay
        covers the correct content area whenever it becomes visible.

        Args:
            event: Qt show event being handled.

        Returns:
            None.
        """
        self._refit_to_parent()
        self.raise_()
        super().showEvent(event)

    def eventFilter(self, obj, event: QtCore.QEvent) -> bool:
        """Track overlay geometry changes and fullscreen-button hover state.

        Keeps the visible overlay fitted to its parent when the parent or its
        containing window is resized or moved. The filter also swaps the
        fullscreen button between its normal and hover icons when the pointer
        enters or leaves the button.

        Args:
            obj: Object that generated the event.
            event: Qt event being filtered.

        Returns:
            bool: Result returned by the superclass event filter after the
            overlay-specific event handling has been performed.
        """
        btn_fs = getattr(self, "btn_fullscreen", None)
        if btn_fs is not None and obj is btn_fs:
            if event.type() == QtCore.QEvent.Enter:
                btn_fs.setIcon(self._fs_hover_icon)
            elif event.type() == QtCore.QEvent.Leave:
                btn_fs.setIcon(self._fs_normal_icon)
        if event.type() in (QtCore.QEvent.Type.Resize, QtCore.QEvent.Type.Move):
            if self.isVisible():
                self._refit_to_parent()
        return super().eventFilter(obj, event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the overlay's semi-transparent scrim.

        Draws a dark, semi-transparent rectangle across the entire overlay using
        the current `_scrim_alpha` value. Because the alpha value is updated by
        the fade animation, the scrim participates directly in the overlay's
        open and close transitions without requiring stylesheet changes.

        Args:
            event: Qt paint event requesting the overlay to be repainted.

        Returns:
            None.
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._scrim_alpha))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Dismiss the overlay when the user clicks outside the glass panel.

        Mouse presses within the glass panel are passed to the superclass so
        child widgets and normal panel interaction continue to work. A mouse
        press outside the panel initiates the overlay's normal close sequence.

        Args:
            event: Qt mouse event containing the position of the mouse press.

        Returns:
            None.
        """
        if not self.glass_frame.geometry().contains(event.pos()):
            self.close()
        else:
            super().mousePressEvent(event)
