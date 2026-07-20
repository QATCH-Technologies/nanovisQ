"""
QATCH.ui.components.overlay_shell

Shared plumbing for the app's full-window glassmorphic overlay widgets -
`DataManagementWidget`, `UserProfilesManagerWidget`, `UserPreferencesWidget`.
Each of those is a child `QWidget` reparented over the app's central widget
that presents a dimmed scrim behind an inset "glass" panel, with a header
(icon, title, optional fullscreen toggle, close button), a fade in/out on
open/close, an optional fullscreen-toggle animation, and click-outside-to-
dismiss. Before this module existed, all of that machinery - dimensions,
corner-button positions, fade timings/easing curves, the fullscreen
interpolation animation - was copy-pasted across the three widgets and had
begun to drift (e.g. mismatched corner-button padding between two of them).

This module centralizes the parts that really are identical in shape:

  - Shared dimensions/timings: corner-button size, panel inset percentage,
    fade durations/easing curves.
  - `OverlayFadeMixin`: the reusable scrim+glass fade-in/fade-out animation
    engine (`_run_fade`/`_stop_anim`) and the default open/close sequence
    (`_animate_open`/`_animate_close`/`_do_close`/`closeEvent`).
  - `OverlayLifecycleMixin` (extends `OverlayFadeMixin`): the "hidden until
    laid out, then reveal atomically" `setVisible`/`showEvent` sequence, the
    parent-tracking `eventFilter`, the scrim `paintEvent`, and
    click-outside-to-dismiss `mousePressEvent`.
  - `build_overlay_title()` / `build_corner_button()` /
    `position_corner_buttons()`: small header-chrome builders.
  - `rebuild_fullscreen_icons()` / `run_variant_animation()`: the fullscreen
    toggle's icon-swap and generic 0->1 animation-with-cleanup helpers.
  - `run_stack_slide()` / `teardown_stack_slide()`: the cross-slide page
    transition (grab both pages as pixmaps, animate them past each other,
    swap the live `QStackedWidget` back in when done) shared by every
    overlay with a sidebar/chip nav switching between pages -
    `DataManagementWidget`'s mode switch and `UserPreferencesWidget`'s
    section switch. Axis-agnostic (`"x"` or `"y"`) so a horizontal chip bar
    and a vertical sidebar both animate along their own nav direction.

Widget-specific visuals (a custom-painted vs. QSS-styled glass frame, a
horizontal chip bar vs. a vertical sidebar vs. a search/action taskbar,
whether there's a fullscreen toggle at all) stay in each widget - this
module only owns the behavior that is genuinely identical across them.

Usage: a widget class mixes in `OverlayLifecycleMixin` alongside
`QtWidgets.QWidget`, calls `_init_overlay_shell(...)` early in `__init__`
(after `super().__init__(parent)`), builds its own content into
`self.main_layout`, and calls `_finish_overlay_shell()` at the end of
`__init__`. See any of the three widgets above for a concrete example.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.icon_utils import tinted_icon
from QATCH.ui.styles.theme_manager import ThemeManager, close_button_qss, glass_panel_qss, tok_css

# ----------------------------------------------------------------------
#  Shared dimensions / timings
# ----------------------------------------------------------------------
CORNER_BUTTON_SIZE = 28
CORNER_ICON_SIZE = QtCore.QSize(14, 14)
CORNER_BUTTON_PADDING = 14  # inset from the panel's top-right corner
CORNER_BUTTON_SPACING = 6  # gap between adjacent corner buttons

DEFAULT_MARGIN_PCT = 0.175  # glass panel inset, as a fraction of each dimension
DEFAULT_PANEL_ALPHA = 215

OPEN_FADE_DURATION = 200
OPEN_FADE_EASING = QtCore.QEasingCurve.OutQuad
CLOSE_FADE_DURATION = 180
CLOSE_FADE_EASING = QtCore.QEasingCurve.InQuad
SCRIM_MAX_ALPHA = 65

FULLSCREEN_ANIM_EASING = QtCore.QEasingCurve.InOutCubic


# ----------------------------------------------------------------------
#  Header / button builders
# ----------------------------------------------------------------------
def build_overlay_title(
    icon_path: str, title_text: str, *, icon_size: int = 16
) -> tuple:
    """Builds a themed (icon_label, title_label) pair for an overlay header:
    a tinted 16px icon followed by a bold 13px title, matching every glass
    overlay's header treatment.

    Returns `(icon_label, title_label, refresh_fn)` - call `refresh_fn()`
    from the host's theme-change handler to re-tint/re-color both on a
    light/dark switch.
    """
    icon_label = QtWidgets.QLabel()
    icon_label.setFixedSize(icon_size, icon_size)
    title_label = QtWidgets.QLabel(title_text)

    def _refresh() -> None:
        tok = ThemeManager.instance().tokens()
        icon_label.setPixmap(
            tinted_icon(icon_path, QtGui.QColor(*tok["flat_text"]), size=icon_size).pixmap(
                icon_size, icon_size
            )
        )
        title_label.setStyleSheet(
            f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-weight: bold; "
            "font-size: 13px; background: transparent; }"
        )

    _refresh()
    return icon_label, title_label, _refresh


def build_corner_button(
    parent: QtWidgets.QWidget, text: str, tooltip: str, *, size: int = CORNER_BUTTON_SIZE
) -> QtWidgets.QPushButton:
    """Builds a bare square button (transparent, no border/text styling of
    its own) sized and cursor-configured for the overlay's floating
    top-right corner row (fullscreen/close). Callers still set their own
    icon/stylesheet/click handler."""
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
    """Lays `buttons` out right-to-left as a row of same-size square buttons
    inset from the panel's top-right corner - the floating close/fullscreen
    button convention shared by `DataManagementWidget` and
    `UserProfilesManagerWidget`.

    `buttons` is ordered outermost-first (i.e. `[btn_close, btn_fullscreen]`
    puts close nearest the corner). `None` entries are skipped, so a widget
    with no fullscreen button can pass `[btn_close, None]`.
    """
    x = width - mx - padding - size
    y = my + padding
    for btn in buttons:
        if btn is None:
            continue
        btn.setGeometry(x, y, size, size)
        btn.raise_()
        x -= size + spacing


def rebuild_fullscreen_icons(host, icon_expand_path: str, icon_collapse_path: str, *, size: int = 14) -> None:
    """Rebuilds `host._fs_normal_icon`/`host._fs_hover_icon` from the active
    theme's tokens, picking expand vs. collapse based on `host._is_fullscreen`,
    and refreshes `host.btn_fullscreen`'s current icon to match. Shared by
    every overlay with a fullscreen toggle so the icon recipe (muted normal /
    full-text hover tint) can't drift between them."""
    tok = ThemeManager.instance().tokens()
    normal_color = QtGui.QColor(*tok["flat_text_muted"])
    hover_color = QtGui.QColor(*tok["flat_text"])
    icon_path = icon_collapse_path if getattr(host, "_is_fullscreen", False) else icon_expand_path
    host._fs_normal_icon = tinted_icon(icon_path, normal_color, size=size)
    host._fs_hover_icon = tinted_icon(icon_path, hover_color, size=size)
    btn = getattr(host, "btn_fullscreen", None)
    if btn is not None:
        btn.setIcon(host._fs_hover_icon if btn.underMouse() else host._fs_normal_icon)


def run_variant_animation(host, anim_attr: str, *, duration: int, easing, on_step, on_finish=None) -> None:
    """Runs a single reusable `QVariantAnimation` from 0.0 to 1.0, storing it
    on `host` as `anim_attr` and tearing down any prior animation stored
    there first. `on_step(t)` fires every tick with eased progress in
    [0.0, 1.0]; `on_finish()` fires once, after the final `on_step(1.0)`.

    This is the generic "create/stop/cleanup a QVariantAnimation" boilerplate
    shared by every fullscreen-toggle animation; the actual interpolation
    (margins, alpha, border, radius, ...) stays with the caller since it
    differs per overlay.
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


def run_stack_slide(
    host,
    stack: QtWidgets.QStackedWidget,
    old_widget: QtWidgets.QWidget,
    new_widget: QtWidgets.QWidget,
    *,
    axis: str = "x",
    forward: bool = True,
    duration: int = 260,
    easing=QtCore.QEasingCurve.OutQuint,
    before_show_new=None,
    on_finished=None,
) -> None:
    """Cross-slides `old_widget` -> `new_widget` inside `stack` along `axis`
    ("x" for a horizontal chip bar, "y" for a vertical sidebar), pixmap-grab
    style: both pages are grabbed as static pixmaps, animated past each
    other in a clip frame overlaid on the live stack, then the live stack is
    swapped back in. This avoids animating the real widget trees (cheap,
    since a tick is just "move two pixmaps" instead of re-laying-out and
    repainting whatever the page actually contains).

    `before_show_new` (optional) fires right after `new_widget` becomes the
    stack's current widget and is resized to its final size, but before it's
    grabbed - use it to populate the page (e.g. a mode's `on_enter()`) so
    the grabbed pixmap already reflects the populated content.

    `on_finished` (optional) fires once the slide completes and the live
    stack is showing again - use it to record the new current key/index and
    re-raise any floating chrome (close/fullscreen buttons) that sits above
    the stack.

    Any slide already running on `host` is torn down first (see
    `teardown_stack_slide`). Falls back to an instant swap (no animation) if
    `stack` doesn't have a settled size yet (e.g. the overlay isn't visible).

    Host requirements: `host.glass_frame` (the clip frame's parent) and the
    slide-state slots `host._slide_group` / `host._slide_clip` /
    `host._slide_stack`, all pre-set to `None` before the first call.
    """
    teardown_stack_slide(host)

    geo = stack.geometry()
    size = geo.size()
    if size.width() <= 0 or size.height() <= 0:
        stack.setCurrentWidget(new_widget)
        if before_show_new is not None:
            before_show_new()
        if on_finished is not None:
            on_finished()
        return

    old_widget.resize(size)
    old_pix = old_widget.grab()
    # Make the incoming page current first so it's laid out, then grab it.
    stack.setCurrentWidget(new_widget)
    new_widget.resize(size)
    if before_show_new is not None:
        before_show_new()
    new_pix = new_widget.grab()

    # Clip wrapper over the stack region so the moving labels are masked.
    clip = QtWidgets.QFrame(host.glass_frame)
    clip.setObjectName("slideClip")
    clip.setStyleSheet("QFrame#slideClip { background: transparent; border: none; }")
    clip.setGeometry(geo)
    clip.show()
    clip.raise_()
    host._slide_clip = clip
    host._slide_stack = stack

    rest = QtCore.QPoint(0, 0)
    if axis == "y":
        h = size.height()
        if forward:
            old_end = QtCore.QPoint(0, -h)  # old exits up
            new_start = QtCore.QPoint(0, h)  # new enters from below
        else:
            old_end = QtCore.QPoint(0, h)  # old exits down
            new_start = QtCore.QPoint(0, -h)  # new enters from above
    else:
        w = size.width()
        if forward:
            old_end = QtCore.QPoint(-w, 0)  # old exits left
            new_start = QtCore.QPoint(w, 0)  # new enters from the right
        else:
            old_end = QtCore.QPoint(w, 0)  # old exits right
            new_start = QtCore.QPoint(-w, 0)  # new enters from the left

    old_lbl = QtWidgets.QLabel(clip)
    old_lbl.setPixmap(old_pix)
    old_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    old_lbl.setGeometry(QtCore.QRect(rest, size))
    old_lbl.show()

    new_lbl = QtWidgets.QLabel(clip)
    new_lbl.setPixmap(new_pix)
    new_lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    new_lbl.setGeometry(QtCore.QRect(new_start, size))
    new_lbl.show()
    new_lbl.raise_()

    # Hide the live stack during the slide so only the proxies show.
    stack.hide()

    anim_old = QtCore.QPropertyAnimation(old_lbl, b"pos", host)
    anim_old.setDuration(duration)
    anim_old.setEasingCurve(easing)
    anim_old.setStartValue(rest)
    anim_old.setEndValue(old_end)

    anim_new = QtCore.QPropertyAnimation(new_lbl, b"pos", host)
    anim_new.setDuration(duration)
    anim_new.setEasingCurve(easing)
    anim_new.setStartValue(new_start)
    anim_new.setEndValue(rest)

    group = QtCore.QParallelAnimationGroup(host)
    group.addAnimation(anim_old)
    group.addAnimation(anim_new)

    def _finish():
        stack.show()
        stack.raise_()
        clip.hide()
        clip.setParent(None)
        clip.deleteLater()
        host._slide_clip = None
        host._slide_group = None
        host._slide_stack = None
        if on_finished is not None:
            on_finished()

    group.finished.connect(_finish)
    host._slide_group = group
    group.start()


def teardown_stack_slide(host) -> None:
    """Stops and fully tears down any slide `run_stack_slide` has in flight
    on `host`, re-showing the live stack it hid.

    `QAbstractAnimation.stop()` does NOT emit `finished()`, so an
    interrupted slide's `_finish` callback never runs on its own - without
    this, the proxy clip (and its captured page pixmaps) would stay
    parented to `host.glass_frame` and keep painting over the live stack
    for the rest of the session (the "ghost of the previous page" bug).
    Safe to call even when no slide is running.
    """
    group = getattr(host, "_slide_group", None)
    if group is not None:
        try:
            group.stop()
            group.deleteLater()
        except RuntimeError:
            pass
        host._slide_group = None
    clip = getattr(host, "_slide_clip", None)
    if clip is not None:
        try:
            clip.hide()
            clip.setParent(None)
            clip.deleteLater()
        except RuntimeError:
            pass
        host._slide_clip = None
    stack = getattr(host, "_slide_stack", None)
    if stack is not None:
        stack.show()
        host._slide_stack = None


# ----------------------------------------------------------------------
#  Fade engine
# ----------------------------------------------------------------------
class OverlayFadeMixin:
    """The reusable scrim+glass fade-in/fade-out engine, plus the default
    open/close sequence built on top of it.

    Host requirements (set by `_init_overlay_shell`, see
    `OverlayLifecycleMixin`): `self._scrim_alpha`, `self._panel_alpha`,
    `self._glass_opacity` (a `QGraphicsOpacityEffect` on `self.glass_frame`),
    `self._fade_anim = None`, `self._closing = False`,
    `self._close_in_progress = False`, and a `self.glass_frame`.

    Override points:
      - `_animate_open`/`_animate_close` - change fade endpoints/timing.
      - `_do_close` - extra close-time teardown (stop a shared service,
        tear down a running slide transition, ...); call `super()._do_close()`
        to still get the shared reset-and-refit tail.
    """

    def _stop_anim(self) -> None:
        """Stop and fully tear down any running fade so it can't keep firing."""
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
        frac = max(0.0, min(1.0, float(frac)))
        effect = getattr(self, "_glass_opacity", None)
        if effect is not None:
            effect.setOpacity(frac)

    def _run_fade(
        self, scrim_from, scrim_to, op_from, op_to, duration, easing, on_done=None, opacity_effect=None
    ) -> None:
        """Animates the scrim alpha (cheap paintEvent) and an opacity effect
        (composited) from fixed endpoints - no per-frame stylesheet.

        `opacity_effect` defaults to `self._glass_opacity` (the open fade);
        a close fade that needs to animate a static pixmap proxy instead
        (see `DataManagementWidget._animate_close`) passes its own effect.
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
        """Performs the real close once the fade-out completes. Subclasses
        with extra close-time teardown should override this, call
        `super()._do_close()`, and add their own steps before/after."""
        self._stop_anim()
        self._closing = True
        self.close()  # closeEvent sees _closing=True -> accepts -> Qt calls hide()
        self._closing = False
        self._close_in_progress = False
        self.glass_frame.show()
        self._set_glass_opacity(1.0)
        self._refit_to_parent()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Intercepts close to run a fade-out first; re-enters with
        `_closing=True` to actually accept."""
        if self._closing:
            event.accept()
            return
        anim = getattr(self, "_fade_anim", None)
        if anim is not None and anim.state() == QtCore.QAbstractAnimation.Running:
            if self._close_in_progress:
                event.ignore()
                return
        event.ignore()
        self._close_in_progress = True
        self._animate_close()


# ----------------------------------------------------------------------
#  Full lifecycle: init scaffolding, reveal sequence, geometry, events
# ----------------------------------------------------------------------
class OverlayLifecycleMixin(OverlayFadeMixin):
    """Adds shell construction, the reveal-on-open sequence, parent-tracking
    geometry fit, and the shared event handlers on top of `OverlayFadeMixin`.

    Override points (all have working no-op-ish defaults):
      - `_current_margin_frac()` - return 0.0 while fullscreen, else
        `self._default_margin_pct` (already the default).
      - `_apply_panel_appearance(frac)` - how the glass panel's own alpha/
        border/radius track `frac` as it animates; default does nothing
        (a fixed `glass_panel_qss` stays as constructed).
      - `_position_overlay_buttons(mx, my)` - where the corner buttons go;
        default is the shared `position_corner_buttons` floating-button row.
        A widget whose close button lives in a normal header layout instead
        (no floating buttons) overrides this to a no-op.
      - `_on_before_reveal()` / `_on_after_reveal()` - extra work right
        before/after the overlay becomes visible.
    """

    # -- header --------------------------------------------------------
    def _build_overlay_header(
        self, icon_path: str, title_text: str, *, fullscreen: bool = False, icon_size: int = 16
    ) -> QtWidgets.QHBoxLayout:
        """Builds the shared header row - a tinted icon + bold title,
        left-aligned - and the close button (plus, if `fullscreen`, the
        fullscreen toggle), which float in the panel's top-right corner via
        `_position_overlay_buttons` rather than living in this layout. This
        is the one chrome treatment every overlay uses (originally
        `DataManagementWidget`'s), so title/icon sizing, position, and the
        close/fullscreen buttons can't drift between overlays again.

        Sets `self.window_icon_label`, `self.window_title_label`,
        `self.btn_close`, and `self.btn_fullscreen` (`None` if not
        `fullscreen`). A widget with a fullscreen toggle must define
        `toggle_fullscreen()` and `_rebuild_fs_icons()` (see
        `rebuild_fullscreen_icons`) before calling this.

        Call `_refresh_header_theme()` from the widget's theme-change
        handler to keep it in sync. Returns the QHBoxLayout to add to the
        widget's own top-level layout.
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
            self.btn_fullscreen.setStyleSheet("QPushButton { background: transparent; border: none; }")
            self.btn_fullscreen.installEventFilter(self)
            self.btn_fullscreen.clicked.connect(self.toggle_fullscreen)
        else:
            self.btn_fullscreen = None

        self.btn_close = build_corner_button(self, "x", "Close")
        self.btn_close.setStyleSheet(close_button_qss())
        self.btn_close.clicked.connect(self.close)

        return header_layout

    def _refresh_header_theme(self) -> None:
        """Re-applies the header's icon/title tint and close-button style -
        call from the widget's theme-change handler."""
        self._refresh_title()
        self.btn_close.setStyleSheet(close_button_qss())
        if self.btn_fullscreen is not None:
            self._rebuild_fs_icons()

    # -- construction ----------------------------------------------------
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
        """Sets up the overlay's scrim/panel state, `base_layout` +
        `glass_frame` + composited opacity effect + `main_layout`, and
        installs the parent-resize event filter. Call once, right after
        `super().__init__(parent)`; build the widget's own content into
        `self.main_layout` afterward, then call `_finish_overlay_shell()`.

        `glass_frame`: pass a pre-built custom-painted frame (e.g.
        `DataManagementWidget`'s `_GlassPanel`) to use instead of a plain
        QSS-styled `QFrame`; when given, `glass_qss` is ignored (the caller
        owns that frame's appearance) and it's assumed already parented to
        `self` with its object name set.
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

        # Fade via a composited opacity effect (animating the property is a
        # paint-time multiply) instead of re-setting the stylesheet each
        # frame, which would reparse + repolish the entire glass subtree.
        self._glass_opacity = QtWidgets.QGraphicsOpacityEffect(self.glass_frame)
        self._glass_opacity.setOpacity(0.0)
        self.glass_frame.setGraphicsEffect(self._glass_opacity)

        self.main_layout = QtWidgets.QVBoxLayout(self.glass_frame)
        self.main_layout.setContentsMargins(*content_margins)
        self.main_layout.setSpacing(content_spacing)

        self.base_layout.addWidget(self.glass_frame)

    def _finish_overlay_shell(self) -> None:
        """Starts hidden and pre-fitted to the parent's content area, so the
        first-ever show doesn't flash a tiny unstyled dialog. Call this once
        the widget's own content has been fully built."""
        self.hide()
        self._scrim_alpha = 0
        self.glass_frame.hide()
        self._refit_to_parent()

    # -- geometry ----------------------------------------------------
    def _current_margin_frac(self) -> float:
        return 0.0 if getattr(self, "_is_fullscreen", False) else self._default_margin_pct

    def _apply_panel_appearance(self, frac: float) -> None:
        """Hook: react to the panel inset fraction changing (e.g. during a
        fullscreen-toggle animation). Default: nothing - a static
        `glass_panel_qss` call is enough for overlays with no fullscreen
        toggle."""

    def _position_overlay_buttons(self, mx: int, my: int) -> None:
        """Hook: position the header's floating corner buttons. Default is
        the shared `[btn_close, btn_fullscreen]` top-right row; override to
        a no-op for a widget whose close button lives in a normal header
        layout instead."""
        position_corner_buttons(self.width(), mx, my, [
            getattr(self, "btn_close", None),
            getattr(self, "btn_fullscreen", None),
        ])

    def _apply_margin_frac(self, frac: float) -> None:
        w, h = self.width(), self.height()
        mx = int(w * frac)
        my = int(h * frac)
        self.base_layout.setContentsMargins(mx, my, mx, my)
        self._apply_panel_appearance(frac)
        self._position_overlay_buttons(mx, my)

    def _refit_to_parent(self) -> None:
        """Resizes `self` to fill the parent's content area, then insets the
        glass panel by the current margin fraction of each dimension."""
        if self.parent is None:
            geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        else:
            geo = self.parent.rect()
        self.setGeometry(geo)

        # A running fullscreen-toggle animation owns the margins for its
        # duration; a resize/move event mid-animation (each button move
        # re-enters here via the eventFilter) must not snap them early.
        fs_anim = getattr(self, "_fs_anim", None)
        if fs_anim is not None and fs_anim.state() == QtCore.QAbstractAnimation.Running:
            return

        self._apply_margin_frac(self._current_margin_frac())

    # -- reveal / show / hide ----------------------------------------------------
    def _on_before_reveal(self) -> None:
        """Hook: runs synchronously, before the overlay becomes visible at all."""

    def _on_after_reveal(self) -> None:
        """Hook: runs once the panel is shown and the open fade has started."""

    def _raise_overlay_buttons(self) -> None:
        for btn in (getattr(self, "btn_close", None), getattr(self, "btn_fullscreen", None)):
            if btn is not None:
                btn.raise_()

    def setVisible(self, visible: bool) -> None:
        """Fits to the parent and keeps the glass panel hidden until layout
        settles, then shows the overlay and panel atomically on the next
        event-loop tick - this is what kills the "tiny blank panel" flash
        on open (no intermediate frame where the overlay is visible without
        its panel)."""
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
        self._refit_to_parent()
        self.raise_()
        super().showEvent(event)

    # -- events ----------------------------------------------------
    def eventFilter(self, obj, event: QtCore.QEvent) -> bool:
        """Tracks parent/window resize so the overlay always covers the
        content area, and swaps the fullscreen button's hover icon."""
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
        """Draws a semi-transparent dark scrim over the whole overlay area,
        driven by `_scrim_alpha` so it participates in open/close animations."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, self._scrim_alpha))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Clicking outside the glass panel dismisses the overlay."""
        if not self.glass_frame.geometry().contains(event.pos()):
            self.close()
        else:
            super().mousePressEvent(event)
