"""
QATCH.ui.styles.tokens

Semantic color tokens for the app's light and dark themes.

Each token is a plain (r, g, b, a) tuple - alpha 0-255 - mirroring the
existing convention in QATCH.ui.components.glass_push_button._PALETTES and
QATCH.ui.widgets.saved_state_dot._COLORS, so a token is a drop-in
QtGui.QColor(*token) the same way those call sites already build colors.

Design note
-----------
Both palettes are *derived*, not hand-listed. A small set of base anchors
(one accent hue, one neutral ink/paper axis per mode, and one hue per
semantic role) is fed through a handful of ramp/mix/alpha helpers, and every
one of the ~150 tokens is expressed as a role on those ramps. `LIGHT` and
`DARK` are produced by the *same* builder with different anchors, so a given
token means the same thing in both modes - e.g. every accent-tinted control
resolves to the same step of the accent ramp, and every hairline/border to
the same step of the neutral ramp. This is what keeps the two themes feeling
like one product instead of two separately tuned colorways.

To retune the whole app, change an anchor (e.g. `_ACCENT`) or a ramp stop -
not 40 individual tokens. To retune a single control, override its line in
`_build()`.
"""

from __future__ import annotations

from typing import Tuple, TypedDict

RGBA = Tuple[int, int, int, int]
RGB = Tuple[int, int, int]


# =====================================================================
# Base anchors
# =====================================================================
# Hues are locked here; lightness/darkness comes from the ramps below, so
# every tint of a hue stays on-hue instead of drifting the way the old
# hand-entered values did.

_ACCENT: RGB = (10, 163, 230)  # brand cyan-blue
_DANGER: RGB = (216, 50, 60)  # error / destructive
_WARNING: RGB = (240, 168, 20)  # amber caution / heating
_SUCCESS: RGB = (52, 190, 120)  # ready / ok
_CAUTION: RGB = (235, 120, 10)  # deep orange (cooling / hot readouts)
_INFO: RGB = (0, 150, 170)  # teal (log timestamps)
_PURPLE: RGB = (150, 60, 180)  # log location tag

# Plot data-line colors are *content*, not chrome, so they are identical in
# both modes on purpose - a trace must read the same regardless of theme.
_PLOT_PRIMARY: RGB = (46, 155, 218)
_PLOT_SECONDARY: RGB = (240, 100, 53)
_PLOT_TEMPERATURE: RGB = (240, 156, 53)
_PLOT_DEVICE: RGB = (72, 190, 120)

# Per-mode neutral axis endpoints (ink = text-ward, paper = background-ward)
# and the mid neutral used for borders/hairlines. One hue drives all greys.
_LIGHT_INK: RGB = (38, 46, 58)
_LIGHT_PAPER: RGB = (255, 255, 255)
_DARK_INK: RGB = (232, 236, 240)
_DARK_PAPER: RGB = (31, 37, 44)


# =====================================================================
# Helpers
# =====================================================================
def _clamp(x: float) -> int:
    return max(0, min(255, int(round(x))))


def _mix(c1: RGB, c2: RGB, t: float) -> RGB:
    """Linear blend from c1 (t=0) to c2 (t=1) in RGB."""
    return (
        _clamp(c1[0] + (c2[0] - c1[0]) * t),
        _clamp(c1[1] + (c2[1] - c1[1]) * t),
        _clamp(c1[2] + (c2[2] - c1[2]) * t),
    )


def _shade(base: RGB, s: float) -> RGB:
    """Darken (s<0) or lighten (s>0) a hue while keeping it on-hue.

    s in roughly [-1, 1]; 0 returns the base hue unchanged.
    """
    if s < 0:
        return _mix(base, (0, 0, 0), -s)
    return _mix(base, (255, 255, 255), s)


def _a(c: RGB, alpha: int) -> RGBA:
    """Attach an alpha channel to an RGB triple."""
    return (c[0], c[1], c[2], _clamp(alpha))


class ColorTokens(TypedDict):
    bg_gradient_start: RGBA
    bg_gradient_end: RGBA
    surface: RGBA
    surface_border: RGBA
    popup_border: RGBA
    menu_item_hover: RGBA
    text_primary: RGBA
    text_secondary: RGBA
    accent: RGBA
    accent_translucent: RGBA
    scrollbar_handle: RGBA
    scrollbar_handle_hover: RGBA
    overlay_dim: RGBA
    danger: RGBA
    warning: RGBA
    success: RGBA
    backdrop_fallback_start: RGBA
    backdrop_fallback_end: RGBA
    backdrop_frost: RGBA
    backdrop_dim: RGBA
    # Log console - container / controls
    log_surface: RGBA
    log_surface_border: RGBA
    log_control_bg: RGBA
    log_control_bg_hover: RGBA
    log_control_bg_focus: RGBA
    log_control_border: RGBA
    log_control_border_hover: RGBA
    log_dropdown_bg: RGBA
    log_dropdown_border: RGBA
    log_separator: RGBA
    log_btn_hover: RGBA
    log_btn_pressed: RGBA
    log_text: RGBA
    log_match_highlight: RGBA
    # Log console - per-level text colors (used in inline HTML spans)
    log_time: RGBA
    log_location: RGBA
    log_debug: RGBA
    log_info: RGBA
    log_warning: RGBA
    log_error: RGBA
    log_default: RGBA
    # Plot glass cards - paintEvent fill colors
    plot_glass_base: RGBA
    plot_glass_overlay: RGBA
    plot_glass_shimmer_top: RGBA
    plot_glass_shimmer_mid: RGBA
    plot_glass_vignette_end: RGBA
    plot_glass_rim: RGBA
    plot_glass_inset: RGBA
    plot_glass_header_line: RGBA
    # Plot glass cards - text
    plot_text_normal: RGBA
    plot_text_bright: RGBA
    plot_text_muted: RGBA
    plot_text_dim: RGBA
    # Plot glass cards - icon buttons
    plot_icon_btn_hover_bg: RGBA
    plot_icon_btn_hover_border: RGBA
    plot_icon_btn_pressed_bg: RGBA
    # Plot glass cards - dropdown menu
    plot_menu_bg: RGBA
    plot_menu_border: RGBA
    plot_menu_separator: RGBA
    plot_menu_row_hover: RGBA
    plot_swatch_border: RGBA
    # Plot glass cards - device tabs (pill buttons)
    plot_tab_bg: RGBA
    plot_tab_border: RGBA
    plot_tab_bg_hover: RGBA
    plot_tab_bg_active: RGBA
    plot_tab_border_active: RGBA
    plot_tab_bg_active_hover: RGBA
    # Plot data line default colors
    plot_data_primary: RGBA
    plot_data_secondary: RGBA
    plot_data_temperature: RGBA
    plot_data_device_accent: RGBA
    # Controls UI - toolbar
    ctrl_toolbar_btn_disabled_text: RGBA
    ctrl_toolbar_btn_pressed_bg: RGBA
    ctrl_toolbar_separator: RGBA
    # Controls UI - progress bar
    ctrl_progress_border: RGBA
    ctrl_progress_chunk_start: RGBA
    ctrl_progress_chunk_end: RGBA
    # Controls UI - temperature controller
    ctrl_temp_ctrl_bg: RGBA
    ctrl_temp_pid_header_text: RGBA
    ctrl_temp_status_offline_bg: RGBA
    ctrl_temp_status_offline_text: RGBA
    ctrl_temp_status_border: RGBA
    # Controls UI - sliders (shared: temp controller + range sliders)
    ctrl_slider_groove: RGBA
    ctrl_slider_handle_border: RGBA
    ctrl_slider_track: RGBA
    ctrl_slider_disabled_handle: RGBA
    ctrl_slider_handle_hover: RGBA
    # Controls UI - hairline divider
    ctrl_hairline: RGBA
    # Controls UI - toggle row labels
    ctrl_toggle_label_text: RGBA
    # Controls UI - device config back button
    ctrl_back_btn_bg: RGBA
    ctrl_back_btn_border: RGBA
    ctrl_back_btn_hover_bg: RGBA
    ctrl_back_btn_hover_border: RGBA
    # Controls UI - glass input fields
    ctrl_input_bg: RGBA
    ctrl_input_border: RGBA
    ctrl_input_text: RGBA
    ctrl_input_focus_bg: RGBA
    ctrl_input_focus_border: RGBA
    # Controls UI - animated combo box (AnimatedComboBox)
    combo_bg: RGBA
    combo_border: RGBA
    combo_bg_hover: RGBA
    combo_border_hover: RGBA
    combo_bg_focus: RGBA
    combo_border_focus: RGBA
    combo_text: RGBA
    combo_popup_bg: RGBA
    combo_popup_border: RGBA
    combo_selection_bg: RGBA
    combo_selection_text: RGBA
    # Controls UI - temperature status dynamic states
    ctrl_temp_ready_bg: RGBA
    ctrl_temp_ready_text: RGBA
    ctrl_temp_heating_bg: RGBA
    ctrl_temp_heating_text: RGBA
    ctrl_temp_cooling_bg: RGBA
    ctrl_temp_cooling_text: RGBA
    # Controls UI - amber pulse border (unsaved field highlight)
    ctrl_pulse_border: RGBA
    # Controls UI - infobar readout label
    ctrl_infobar_text: RGBA
    # Controls window - menu bar signed-in (normal) state
    menubar_bg: RGBA
    menubar_text: RGBA
    menubar_item_hover_bg: RGBA
    menubar_item_disabled_text: RGBA
    menubar_border: RGBA
    menubar_separator: RGBA
    # Controls window - menu bar signed-out (dimmed) state
    menubar_dim_bg: RGBA
    menubar_dim_text: RGBA
    menubar_dim_item_hover_bg: RGBA
    menubar_dim_item_disabled_text: RGBA
    menubar_dim_border: RGBA
    menubar_dim_separator: RGBA
    # Login UI - titles, body text, links
    login_title_text: RGBA
    login_body_text: RGBA
    login_link_text: RGBA
    login_link_text_hover: RGBA
    login_caps_warning: RGBA
    # Login UI - checkbox
    login_checkbox_text: RGBA
    login_checkbox_border: RGBA
    login_checkbox_bg: RGBA
    login_checkbox_hover_border: RGBA
    login_checkbox_checked_bg: RGBA
    login_checkbox_checked_border: RGBA
    # Login UI - recover status feedback
    login_recover_success: RGBA
    login_recover_error: RGBA
    # Login UI - primary action buttons
    login_primary_btn_top: RGBA
    login_primary_btn_bottom: RGBA
    login_primary_btn_hover_top: RGBA
    login_primary_btn_hover_bottom: RGBA
    login_primary_btn_pressed_top: RGBA
    login_primary_btn_pressed_bottom: RGBA
    login_primary_btn_disabled_bg: RGBA
    # Login UI - back navigation button
    login_back_btn_text: RGBA
    login_back_btn_text_hover: RGBA
    # Mode window - footer copyright label
    mode_footer_text: RGBA
    mode_footer_dim_text: RGBA
    # Mode window - log-console toggle button
    mode_toggle_btn_hover: RGBA
    mode_toggle_btn_pressed: RGBA
    # Mode window - log-splitter handle hover
    mode_splitter_handle_hover: RGBA
    mode_splitter_dim_handle_hover: RGBA
    # Selectable option card (GlassOptionCard)
    option_card_bg: RGBA
    option_card_border: RGBA
    option_card_hover_bg: RGBA
    option_card_hover_border: RGBA
    option_card_title: RGBA
    option_card_desc: RGBA
    option_card_radio_bg: RGBA
    option_card_radio_border: RGBA
    option_card_checked_bg: RGBA
    option_card_checked_border: RGBA
    option_card_checked_title: RGBA
    option_card_radio_checked: RGBA
    # Glass input field (GlassLineEdit) - paint-time fills/borders/shimmer
    input_glass_text: RGBA
    input_glass_selection_bg: RGBA
    input_glass_fill: RGBA
    input_glass_fill_focus: RGBA
    input_glass_fill_error: RGBA
    input_glass_border: RGBA
    input_glass_border_error: RGBA
    input_glass_shimmer_accent: RGBA
    input_glass_shimmer_peak: RGBA

    # Account popup - avatar badge (brand mark, identical both modes)
    account_avatar_grad_start: RGBA
    account_avatar_grad_end: RGBA
    account_avatar_ring: RGBA
    account_avatar_shimmer: RGBA
    account_avatar_text: RGBA
    # Account popup - role badge chip
    account_role_admin_bg: RGBA
    account_role_admin_text: RGBA
    account_role_operate_bg: RGBA
    account_role_operate_text: RGBA
    account_role_analyze_bg: RGBA
    account_role_analyze_text: RGBA
    account_role_capture_bg: RGBA
    account_role_capture_text: RGBA
    account_role_default_bg: RGBA
    account_role_default_text: RGBA

    # User management widget - role wash chips (role combo + role badge).
    # A soft-tint pill (translucent bg + saturated text), distinct from the
    # account popup's solid role badge above - same role->hue mapping isn't
    # reused because the two are different visual treatments.
    user_role_admin_bg: RGBA
    user_role_admin_bg_hover: RGBA
    user_role_admin_border: RGBA
    user_role_admin_text: RGBA
    user_role_operate_bg: RGBA
    user_role_operate_bg_hover: RGBA
    user_role_operate_border: RGBA
    user_role_operate_text: RGBA
    user_role_capture_bg: RGBA
    user_role_capture_bg_hover: RGBA
    user_role_capture_border: RGBA
    user_role_capture_text: RGBA
    user_role_analyze_bg: RGBA
    user_role_analyze_bg_hover: RGBA
    user_role_analyze_border: RGBA
    user_role_analyze_text: RGBA
    user_role_default_bg: RGBA
    user_role_default_bg_hover: RGBA
    user_role_default_border: RGBA
    user_role_default_text: RGBA

    # ---- Flat control system (Pushbutton, Line Edit, Combo Box, Spin Box,
    # Toggle, Option Card) - literal values from the app's flat design spec.
    # Unlike the rest of this file, these are NOT derived through the
    # fg()/surf()/line() ramps: they are fixed external spec colors (already
    # converted from the spec's OKLCH values to sRGB) that don't mirror
    # between light/dark, so deriving them would drift from the spec. See
    # _FLAT_LIGHT / _FLAT_DARK below.
    flat_text: RGBA
    flat_text_muted: RGBA
    flat_surface: RGBA
    flat_surface2: RGBA
    flat_border: RGBA
    flat_border_strong: RGBA
    flat_accent: RGBA
    flat_accent_hover: RGBA
    flat_accent_active: RGBA
    flat_accent_weak: RGBA
    flat_accent_ring: RGBA
    flat_on_accent: RGBA
    flat_error: RGBA
    flat_error_weak: RGBA
    flat_error_ring: RGBA
    flat_success: RGBA
    flat_success_weak: RGBA
    flat_success_ring: RGBA
    flat_warning: RGBA
    flat_warning_weak: RGBA
    flat_warning_ring: RGBA
    flat_track: RGBA
    flat_knob: RGBA
    flat_shadow: RGBA
    flat_menu_shadow: RGBA


def _build(mode: str) -> ColorTokens:
    """Derives a full ColorTokens palette for `mode` ('light' | 'dark').

    Both modes run through this one function so that each token is the *same
    semantic role* on the *same ramps* in both - only the anchors differ.
    """
    dark = mode == "dark"

    ink = _DARK_INK if dark else _LIGHT_INK
    paper = _DARK_PAPER if dark else _LIGHT_PAPER

    # Two role-based neutral ramps. Both are authored on the *same* semantic
    # scale in both modes (t has the same meaning), but each resolves toward
    # the correct end of that mode's ink/paper axis - which is why surfaces
    # come out light in light mode and dark in dark mode from one line.

    def fg(t: float) -> RGB:
        """Foreground/text ramp. t=0 = strongest (ink), t=1 = faintest.

        Fades ink toward the surface tone, so faint text stays legible in
        both modes instead of washing to pure white/black.
        """
        return _mix(ink, paper, t)

    def surf(t: float) -> RGB:
        """Surface/chrome ramp. t=0 = base panel tone, t=1 = most raised.

        Light mode climbs toward white; dark mode climbs from the base panel
        up to a lighter raised grey. Same t -> same perceived elevation.
        """
        if dark:
            base = paper  # (31,37,44) base panel
            top = _mix(paper, ink, 0.16)  # gently raised grey, still dark/cool
        else:
            base = _mix(paper, ink, 0.06)  # faintly-tinted off-white
            top = (255, 255, 255)
        return _mix(base, top, t)

    def tint() -> RGB:
        """Contrast overlay tint: white over dark surfaces, black over light.

        For hairlines/grooves that are painted *as a translucent tint on top
        of* a surface and must read against it in either mode.
        """
        return (255, 255, 255) if dark else (0, 0, 0)

    def line(t: float) -> RGB:
        """Chrome border/separator/handle ramp. t=0 subtlest, t=1 strongest.

        A border is 'the surface, nudged toward more contrast'. In dark mode
        that means slightly *lighter* than the panel; in light mode slightly
        *darker*. Same t -> same perceived contrast against the surface in
        either mode, so borders stop drifting between the two palettes.
        """
        if dark:
            base = _mix(paper, ink, 0.30)  # subtle: just above the panel
            top = _mix(paper, ink, 0.52)  # strong-ish cool grey line
        else:
            base = _mix(paper, ink, 0.36)  # subtle grey on white
            top = _mix(paper, ink, 0.58)  # stronger grey line
        return _mix(base, top, t)

    # Accent, brightened slightly on dark so it holds contrast on dark glass.
    accent = _shade(_ACCENT, 0.12) if dark else _ACCENT
    # Solid raised-panel fill (opaque white in light, raised grey in dark).
    glass = surf(1.0)

    def sem(base: RGB, s: float = 0.0) -> RGB:
        return _shade(base, s)

    t: ColorTokens = {}  # type: ignore[assignment]

    # ---- Core surfaces / text / accent ----
    if dark:
        t["bg_gradient_start"] = _a((27, 32, 38), 255)
        t["bg_gradient_end"] = _a((35, 41, 48), 255)
    else:
        t["bg_gradient_start"] = _a((228, 235, 241), 255)
        t["bg_gradient_end"] = _a((244, 247, 249), 255)
    t["surface"] = _a(glass, 170 if dark else 160)
    t["surface_border"] = _a(surf(0.9), 160 if dark else 220)
    # Border for popups/dropdowns that float as their own top-level window
    # (AnimatedComboBox's rounded popup, the generic QComboBox/QMenu
    # fallback chrome) - deliberately its own token rather than reusing
    # flat_border_strong, which is tuned for small in-line hover affordances
    # and reads as too heavy once stretched around a whole popup's edge in
    # dark mode (light text/lines on a dark surface perceptually "irradiate"
    # wider than the same contrast the other way around, at equal pixel
    # width). fg(0.60) in light mode matches flat_border_strong's weight
    # (no complaints there); dark backs off to fg(0.72) for the same
    # perceived weight instead of matching contrast numerically.
    t["popup_border"] = _a(fg(0.72) if dark else fg(0.60), 255)
    t["menu_item_hover"] = _a(surf(0.55) if dark else fg(0.90), 150)
    t["text_primary"] = _a(fg(0.0), 255)
    t["text_secondary"] = _a(fg(0.30), 200 if dark else 180)
    t["accent"] = _a(accent, 255)
    t["accent_translucent"] = _a(accent, 46 if dark else 38)
    t["scrollbar_handle"] = _a(line(0.45), 120 if dark else 100)
    t["scrollbar_handle_hover"] = _a(line(0.70), 190 if dark else 180)
    t["overlay_dim"] = _a(surf(0.0) if dark else (164, 168, 172), 255)
    t["danger"] = _a(sem(_DANGER, 0.06 if dark else 0.0), 255)
    t["warning"] = _a(sem(_WARNING, 0.10 if dark else 0.0), 255)
    t["success"] = _a(sem(_SUCCESS, 0.06 if dark else 0.0), 255)
    if dark:
        t["backdrop_fallback_start"] = _a((32, 38, 46), 255)
        t["backdrop_fallback_end"] = _a((20, 24, 30), 255)
        t["backdrop_frost"] = _a(surf(0.7), 70)
        t["backdrop_dim"] = (0, 0, 0, 110)
    else:
        t["backdrop_fallback_start"] = _a((216, 230, 240), 255)
        t["backdrop_fallback_end"] = _a((238, 244, 248), 255)
        t["backdrop_frost"] = _a((238, 243, 247), 62)
        t["backdrop_dim"] = (0, 0, 0, 76)

    # ---- Log console - container / controls ----
    t["log_surface"] = _a(glass, 140 if dark else 120)
    t["log_surface_border"] = _a(surf(0.9), 180 if dark else 200)
    t["log_control_bg"] = _a(surf(0.45) if dark else surf(1.0), 160)
    t["log_control_bg_hover"] = _a(surf(0.60) if dark else surf(1.0), 200)
    t["log_control_bg_focus"] = _a(surf(0.68) if dark else surf(1.0), 220)
    t["log_control_border"] = _a(line(0.45), 160)
    t["log_control_border_hover"] = _a(line(0.70), 200)
    t["log_dropdown_bg"] = _a(surf(0.30) if dark else fg(0.96), 255)
    # Same weight as every other floating popup border - see popup_border.
    t["log_dropdown_border"] = t["popup_border"]
    t["log_separator"] = _a(line(0.40), 110)
    t["log_btn_hover"] = _a(surf(0.55) if dark else surf(1.0), 180)
    t["log_btn_pressed"] = _a(surf(0.68) if dark else surf(1.0), 220)
    t["log_text"] = _a(fg(0.16) if dark else fg(0.12), 200)
    t["log_match_highlight"] = _a(accent, 90)
    # Per-level text colors - semantic hues, brightened on dark for legibility.
    t["log_time"] = _a(sem(_INFO, 0.30 if dark else -0.02), 255)
    t["log_location"] = _a(sem(_PURPLE, 0.42 if dark else -0.02), 255)
    t["log_debug"] = _a(fg(0.58) if dark else fg(0.52), 255)
    t["log_info"] = _a(sem(_SUCCESS, 0.18 if dark else -0.10), 255)
    t["log_warning"] = _a(sem(_CAUTION, 0.06 if dark else -0.08), 255)
    t["log_error"] = _a(sem(_DANGER, 0.10 if dark else -0.10), 255)
    t["log_default"] = _a(fg(0.20) if dark else fg(0.0), 255)

    # ---- Plot glass cards - paintEvent fills ----
    t["plot_glass_base"] = _a(surf(0.35) if dark else surf(1.0), 160)
    t["plot_glass_overlay"] = _a(surf(0.55) if dark else (228, 235, 241), 18)
    t["plot_glass_shimmer_top"] = _a(surf(0.9) if dark else surf(1.0), 60 if dark else 100)
    t["plot_glass_shimmer_mid"] = _a(surf(0.9) if dark else surf(1.0), 15 if dark else 20)
    t["plot_glass_vignette_end"] = _a((10, 14, 22) if dark else (200, 218, 240), 30 if dark else 18)
    t["plot_glass_rim"] = _a(surf(1.0) if dark else surf(1.0), 180 if dark else 230)
    t["plot_glass_inset"] = _a(surf(0.7) if dark else (190, 210, 235), 70)
    t["plot_glass_header_line"] = _a(surf(0.65) if dark else (195, 215, 238), 70)
    # Plot text
    t["plot_text_normal"] = _a(fg(0.14) if dark else fg(0.12), 200)
    t["plot_text_bright"] = _a(fg(0.10) if dark else fg(0.12), 235)
    t["plot_text_muted"] = _a(fg(0.28) if dark else fg(0.12), 155)
    t["plot_text_dim"] = _a(fg(0.34) if dark else fg(0.12), 130 if dark else 110)
    # Plot icon buttons
    t["plot_icon_btn_hover_bg"] = _a(surf(0.55) if dark else surf(1.0), 160)
    t["plot_icon_btn_hover_border"] = _a(surf(1.0) if dark else surf(1.0), 180 if dark else 200)
    t["plot_icon_btn_pressed_bg"] = _a(sem(_ACCENT, -0.18) if dark else (180, 215, 255), 190)
    # Plot dropdown menu - neutral surface tone (matches `surface`/
    # `surface_border`, the rest of the app's flat-card language) rather
    # than the old frosted-glass family's blue tint, which read as
    # inconsistent with the rest of the app in light mode.
    t["plot_menu_bg"] = _a(surf(0.35) if dark else surf(1.0), 248)
    # Same weight as every other floating popup border - see popup_border.
    t["plot_menu_border"] = t["popup_border"]
    t["plot_menu_separator"] = _a(surf(0.9) if dark else surf(0.85), 90)
    t["plot_menu_row_hover"] = _a(accent, 35 if dark else 28)
    t["plot_swatch_border"] = _a(surf(1.0) if dark else surf(1.0), 180 if dark else 210)
    # Device tabs (pill buttons)
    t["plot_tab_bg"] = _a(surf(0.5) if dark else surf(1.0), 55)
    t["plot_tab_border"] = _a(surf(0.9) if dark else surf(1.0), 110)
    t["plot_tab_bg_hover"] = _a(surf(0.6) if dark else surf(1.0), 130)
    t["plot_tab_bg_active"] = _a(surf(0.7) if dark else surf(1.0), 215)
    t["plot_tab_border_active"] = _a(surf(1.0) if dark else surf(1.0), 255)
    t["plot_tab_bg_active_hover"] = _a(surf(0.8) if dark else surf(1.0), 240)
    # Plot data-line colors (content, identical both modes)
    t["plot_data_primary"] = _a(_PLOT_PRIMARY, 255)
    t["plot_data_secondary"] = _a(_PLOT_SECONDARY, 255)
    t["plot_data_temperature"] = _a(_PLOT_TEMPERATURE, 255)
    t["plot_data_device_accent"] = _a(_PLOT_DEVICE, 255)

    # ---- Controls UI - toolbar ----
    t["ctrl_toolbar_btn_disabled_text"] = _a(fg(0.14) if dark else fg(0.12), 90)
    t["ctrl_toolbar_btn_pressed_bg"] = _a(surf(0.55) if dark else fg(0.90), 200)
    t["ctrl_toolbar_separator"] = _a(tint(), 22)
    # Progress bar
    t["ctrl_progress_border"] = _a(tint(), 25)
    t["ctrl_progress_chunk_start"] = _a(accent, 130)
    t["ctrl_progress_chunk_end"] = _a(accent, 90)
    # Temperature controller
    t["ctrl_temp_ctrl_bg"] = _a(surf(0.4) if dark else fg(0.90), 80)
    t["ctrl_temp_pid_header_text"] = _a(sem(accent, -0.06) if dark else sem(_ACCENT, -0.30), 220)
    t["ctrl_temp_status_offline_bg"] = _a(surf(0.7) if dark else (150, 155, 160), 120)
    t["ctrl_temp_status_offline_text"] = _a(fg(0.50) if dark else fg(0.12), 160)
    t["ctrl_temp_status_border"] = _a(surf(1.0) if dark else surf(1.0), 160)
    # Sliders
    t["ctrl_slider_groove"] = _a(tint(), 30)
    t["ctrl_slider_handle_border"] = _a(sem(_ACCENT, -0.12), 200)
    t["ctrl_slider_track"] = _a(accent, 120)
    t["ctrl_slider_disabled_handle"] = _a(line(0.55) if dark else (150, 170, 190), 140)
    t["ctrl_slider_handle_hover"] = _a(sem(_ACCENT, 0.24), 255)
    # Hairline divider
    t["ctrl_hairline"] = _a(surf(0.7) if dark else (200, 210, 220), 130)
    # Toggle row labels
    t["ctrl_toggle_label_text"] = _a(fg(0.14) if dark else fg(0.12), 215)
    # Device config back button
    t["ctrl_back_btn_bg"] = _a(surf(0.55) if dark else surf(1.0), 40)
    t["ctrl_back_btn_border"] = _a(surf(0.9) if dark else surf(1.0), 100)
    t["ctrl_back_btn_hover_bg"] = _a(surf(0.7) if dark else surf(1.0), 80)
    t["ctrl_back_btn_hover_border"] = _a(surf(1.0) if dark else surf(1.0), 150)
    # Glass input fields
    t["ctrl_input_bg"] = _a(surf(0.5) if dark else surf(1.0), 60)
    t["ctrl_input_border"] = _a(surf(0.9) if dark else surf(1.0), 120)
    t["ctrl_input_text"] = _a(fg(0.14) if dark else fg(0.12), 220)
    t["ctrl_input_focus_bg"] = _a(surf(0.7) if dark else surf(1.0), 180)
    t["ctrl_input_focus_border"] = _a(sem(_ACCENT, -0.06), 180)
    # Animated combo box
    t["combo_bg"] = _a(surf(0.5) if dark else surf(1.0), 150)
    t["combo_border"] = _a(line(0.50), 150)
    t["combo_bg_hover"] = _a(surf(0.6) if dark else surf(1.0), 200)
    t["combo_border_hover"] = _a(line(0.72), 190)
    t["combo_bg_focus"] = _a(surf(0.68) if dark else surf(1.0), 225)
    t["combo_border_focus"] = _a(accent, 200)
    t["combo_text"] = _a(fg(0.10) if dark else fg(0.06), 255)
    t["combo_popup_bg"] = _a(surf(0.3) if dark else fg(0.96), 255)
    t["combo_popup_border"] = _a(surf(0.9) if dark else fg(0.74), 220)
    t["combo_selection_bg"] = _a(accent, 50 if dark else 40)
    t["combo_selection_text"] = _a(sem(_ACCENT, 0.24) if dark else _ACCENT, 255)
    # Temperature status dynamic states
    t["ctrl_temp_ready_bg"] = _a(sem(_SUCCESS, 0.0), 220)
    t["ctrl_temp_ready_text"] = _a((255, 255, 255), 230)
    t["ctrl_temp_heating_bg"] = _a(sem(_WARNING, -0.14), 220)
    t["ctrl_temp_heating_text"] = _a((255, 240, 180) if dark else (30, 20, 0), 200)
    t["ctrl_temp_cooling_bg"] = _a(sem(_CAUTION, -0.06), 220)
    t["ctrl_temp_cooling_text"] = _a((255, 220, 170) if dark else (30, 20, 0), 200)
    # Amber pulse border
    t["ctrl_pulse_border"] = _a(sem(_WARNING, 0.10), 230)
    # Infobar readout
    t["ctrl_infobar_text"] = _a(fg(0.40) if dark else fg(0.30), 190)

    # ---- Controls window - menu bar (signed-in) ----
    t["menubar_bg"] = _a(surf(0.3) if dark else (233, 239, 244), 255)
    t["menubar_text"] = _a(fg(0.20) if dark else fg(0.18), 230)
    t["menubar_item_hover_bg"] = _a(accent, 60)
    t["menubar_item_disabled_text"] = _a(fg(0.42), 140)
    t["menubar_border"] = _a(surf(0.9) if dark else surf(1.0), 200 if dark else 230)
    t["menubar_separator"] = _a(line(0.35), 80 if dark else 70)
    # Menu bar (signed-out / dimmed)
    t["menubar_dim_bg"] = _a(surf(0.5) if dark else (163, 167, 171), 255)
    t["menubar_dim_text"] = _a(fg(0.56) if dark else fg(0.12), 235)
    t["menubar_dim_item_hover_bg"] = _a(fg(0.42) if dark else surf(1.0), 60)
    t["menubar_dim_item_disabled_text"] = _a(fg(0.36) if dark else fg(0.34), 150)
    t["menubar_dim_border"] = _a(surf(0.9) if dark else surf(1.0), 90)
    t["menubar_dim_separator"] = _a(line(0.40) if dark else surf(1.0), 80)

    # ---- Login UI ----
    t["login_title_text"] = _a(fg(0.10) if dark else fg(0.18), 220)
    t["login_body_text"] = _a(fg(0.34) if dark else fg(0.40), 220)
    t["login_link_text"] = _a(fg(0.40) if dark else fg(0.40), 180)
    t["login_link_text_hover"] = _a(fg(0.16) if dark else fg(0.10), 220)
    t["login_caps_warning"] = _a(sem(_WARNING, -0.04 if dark else -0.20), 235)
    # Checkbox
    t["login_checkbox_text"] = _a(fg(0.26) if dark else fg(0.30), 220)
    t["login_checkbox_border"] = _a(line(0.50) if dark else (150, 160, 170), 180)
    t["login_checkbox_bg"] = _a(surf(0.3) if dark else surf(1.0), 120)
    t["login_checkbox_hover_border"] = _a(accent, 150)
    t["login_checkbox_checked_bg"] = _a(sem(_ACCENT, -0.10), 210)
    t["login_checkbox_checked_border"] = _a(sem(_ACCENT, -0.18), 255)
    # Recover status feedback
    t["login_recover_success"] = _a(sem(_SUCCESS, -0.10 if not dark else 0.0), 220)
    t["login_recover_error"] = _a(sem(_DANGER, -0.06 if not dark else 0.06), 230)
    # Primary action buttons (top/bottom gradient stops)
    t["login_primary_btn_top"] = _a(sem(_ACCENT, 0.14), 210)
    t["login_primary_btn_bottom"] = _a(sem(_ACCENT, -0.10), 190)
    t["login_primary_btn_hover_top"] = _a(sem(_ACCENT, 0.26), 240)
    t["login_primary_btn_hover_bottom"] = _a(sem(_ACCENT, 0.0), 220)
    t["login_primary_btn_pressed_top"] = _a(sem(_ACCENT, -0.14), 220)
    t["login_primary_btn_pressed_bottom"] = _a(sem(_ACCENT, -0.34), 200)
    t["login_primary_btn_disabled_bg"] = _a(fg(0.40) if dark else (150, 170, 190), 100)
    # Back navigation button
    t["login_back_btn_text"] = _a(fg(0.42) if dark else fg(0.40), 200)
    t["login_back_btn_text_hover"] = _a(fg(0.16) if dark else fg(0.10), 220)

    # ---- Mode window ----
    t["mode_footer_text"] = _a(fg(0.36), 140)
    t["mode_footer_dim_text"] = _a(fg(0.36) if dark else fg(0.14), 200)
    t["mode_toggle_btn_hover"] = _a(fg(0.40) if dark else fg(0.52), 60 if dark else 45)
    t["mode_toggle_btn_pressed"] = _a(fg(0.40) if dark else fg(0.52), 100 if dark else 80)
    t["mode_splitter_handle_hover"] = _a(line(0.45), 80 if dark else 60)
    t["mode_splitter_dim_handle_hover"] = _a(line(0.35) if dark else fg(0.56), 255)

    # ---- Selectable option card (GlassOptionCard) ----
    # Unchecked = a raised glass tile; checked = accent-tinted with a strong
    # accent rim. Titles use the standard text ramp; checked title deepens to
    # a dark accent (light) / brightens (dark) for emphasis.
    t["option_card_bg"] = _a(surf(1.0), 130 if not dark else 150)
    t["option_card_border"] = _a(line(0.30) if dark else (212, 219, 228), 190)
    t["option_card_hover_bg"] = _a(surf(1.0), 180 if not dark else 200)
    t["option_card_hover_border"] = _a(line(0.55), 210)
    t["option_card_title"] = _a(fg(0.06), 230)
    t["option_card_desc"] = _a(fg(0.30), 200)
    t["option_card_radio_bg"] = _a(surf(1.0), 200)
    t["option_card_radio_border"] = _a(line(0.50), 180)
    t["option_card_checked_bg"] = _a(accent, 44 if dark else 35)
    t["option_card_checked_border"] = _a(sem(_ACCENT, -0.28 if not dark else -0.02), 200)
    t["option_card_checked_title"] = _a(
        sem(_ACCENT, -0.46) if not dark else sem(_ACCENT, 0.30), 245
    )
    t["option_card_radio_checked"] = _a(sem(_ACCENT, -0.28 if not dark else 0.0), 235)

    # ---- Glass input field (GlassLineEdit) ----
    # Fills are neutral glass; error uses the danger hue at low alpha; the
    # focus shimmer sweeps from a soft accent tint up to a bright peak.
    t["input_glass_text"] = _a(fg(0.06), 230)
    t["input_glass_selection_bg"] = _a(accent, 60)
    t["input_glass_fill"] = _a(surf(1.0), 58)
    t["input_glass_fill_focus"] = _a(surf(1.0), 100)
    t["input_glass_fill_error"] = _a(sem(_DANGER, 0.62 if not dark else 0.30), 68)
    t["input_glass_border"] = _a(surf(1.0), 105)
    t["input_glass_border_error"] = _a(sem(_DANGER, 0.04 if not dark else 0.14), 150)
    t["input_glass_shimmer_accent"] = _a(sem(_ACCENT, 0.58 if not dark else 0.30), 118)
    t["input_glass_shimmer_peak"] = _a(surf(1.0), 240)

    # ---- Account popup ----
    # Avatar gradient/ring/shimmer/text are the brand mark - content, not
    # chrome, so identical in both modes (same reasoning as plot_data_*).
    t["account_avatar_grad_start"] = _a((0, 158, 210), 255)
    t["account_avatar_grad_end"] = _a((0, 100, 160), 255)
    t["account_avatar_ring"] = _a((255, 255, 255), 90)
    t["account_avatar_shimmer"] = _a((255, 255, 255), 55)
    t["account_avatar_text"] = _a((255, 255, 255), 235)
    # Role badge chip - saturated fills read fine on either surface, so only
    # the "no role" fallback (a neutral grey with ink-colored text) needs to
    # flip its text between modes.
    t["account_role_admin_bg"] = _a((0, 118, 174), 215)
    t["account_role_admin_text"] = _a((255, 255, 255), 255)
    t["account_role_operate_bg"] = _a((40, 155, 75), 200)
    t["account_role_operate_text"] = _a((255, 255, 255), 255)
    t["account_role_analyze_bg"] = _a((130, 80, 200), 200)
    t["account_role_analyze_text"] = _a((255, 255, 255), 255)
    t["account_role_capture_bg"] = _a((200, 125, 0), 200)
    t["account_role_capture_text"] = _a((255, 255, 255), 255)
    t["account_role_default_bg"] = _a((140, 150, 160), 160)
    t["account_role_default_text"] = _a(fg(0.0), 180)

    return t


LIGHT: ColorTokens = _build("light")
DARK: ColorTokens = _build("dark")

# =====================================================================
# Flat control system overlay
# =====================================================================
# Literal RGBA values from the flat design spec (already converted from the
# spec's OKLCH custom properties to sRGB). Applied as an overlay on top of
# the derived palettes above rather than folded into _build(), since these
# are fixed target colors for a specific control family, not anchors meant
# to participate in the app-wide ramp system.
_FLAT_LIGHT: dict = {
    "flat_text": (45, 52, 56, 255),
    "flat_text_muted": (109, 114, 119, 255),
    "flat_surface": (253, 253, 254, 255),
    "flat_surface2": (238, 240, 242, 255),
    "flat_border": (206, 209, 212, 255),
    "flat_border_strong": (165, 172, 177, 255),
    "flat_accent": (0, 138, 195, 255),
    "flat_accent_hover": (0, 120, 175, 255),
    "flat_accent_active": (0, 102, 156, 255),
    "flat_accent_weak": (217, 243, 255, 255),
    "flat_accent_ring": (0, 138, 195, 71),
    "flat_on_accent": (255, 255, 255, 255),
    "flat_error": (201, 47, 51, 255),
    "flat_error_weak": (255, 235, 232, 255),
    "flat_error_ring": (201, 47, 51, 61),
    "flat_success": (22, 132, 74, 255),
    "flat_success_weak": (226, 247, 235, 255),
    "flat_success_ring": (22, 132, 74, 61),
    "flat_warning": (178, 111, 0, 255),
    "flat_warning_weak": (255, 244, 224, 255),
    "flat_warning_ring": (178, 111, 0, 61),
    "flat_track": (196, 200, 203, 255),
    "flat_knob": (255, 255, 255, 255),
    "flat_shadow": (20, 30, 45, 26),
    "flat_menu_shadow": (20, 30, 45, 41),
}
_FLAT_DARK: dict = {
    "flat_text": (212, 216, 219, 255),
    "flat_text_muted": (129, 135, 140, 255),
    "flat_surface": (42, 46, 49, 255),
    "flat_surface2": (31, 35, 38, 255),
    "flat_border": (67, 73, 77, 255),
    "flat_border_strong": (99, 106, 111, 255),
    "flat_accent": (30, 151, 208, 255),
    "flat_accent_hover": (74, 172, 226, 255),
    "flat_accent_active": (0, 132, 188, 255),
    "flat_accent_weak": (36, 68, 87, 255),
    "flat_accent_ring": (30, 151, 208, 115),
    "flat_on_accent": (249, 252, 255, 255),
    "flat_error": (236, 91, 87, 255),
    "flat_error_weak": (87, 40, 37, 255),
    "flat_error_ring": (236, 91, 87, 89),
    "flat_success": (110, 199, 145, 255),
    "flat_success_weak": (30, 58, 44, 255),
    "flat_success_ring": (110, 199, 145, 89),
    "flat_warning": (231, 168, 68, 255),
    "flat_warning_weak": (74, 55, 20, 255),
    "flat_warning_ring": (231, 168, 68, 89),
    "flat_track": (72, 78, 82, 255),
    "flat_knob": (239, 242, 244, 255),
    "flat_shadow": (0, 0, 0, 89),
    "flat_menu_shadow": (0, 0, 0, 140),
}
# User management widget - role wash chips. Literal spec values (like the
# flat control system above) rather than derived: each role is a fixed,
# immediately-recognizable hue, and light/dark only need the text tint
# brightened and the wash alpha bumped for contrast against a dark surface -
# not a full re-derivation through the ramp helpers.
_USER_ROLE_LIGHT: dict = {
    "user_role_admin_bg": (220, 53, 69, 31),
    "user_role_admin_bg_hover": (220, 53, 69, 56),
    "user_role_admin_border": (220, 53, 69, 115),
    "user_role_admin_text": (200, 35, 51, 255),
    "user_role_operate_bg": (40, 167, 69, 31),
    "user_role_operate_bg_hover": (40, 167, 69, 56),
    "user_role_operate_border": (40, 167, 69, 115),
    "user_role_operate_text": (30, 126, 52, 255),
    "user_role_capture_bg": (255, 193, 7, 31),
    "user_role_capture_bg_hover": (255, 193, 7, 56),
    "user_role_capture_border": (255, 193, 7, 128),
    "user_role_capture_text": (179, 134, 0, 255),
    "user_role_analyze_bg": (111, 66, 193, 31),
    "user_role_analyze_bg_hover": (111, 66, 193, 56),
    "user_role_analyze_border": (111, 66, 193, 115),
    "user_role_analyze_text": (111, 66, 193, 255),
    "user_role_default_bg": (108, 117, 125, 31),
    "user_role_default_bg_hover": (108, 117, 125, 56),
    "user_role_default_border": (108, 117, 125, 115),
    "user_role_default_text": (73, 80, 87, 255),
}
_USER_ROLE_DARK: dict = {
    "user_role_admin_bg": (220, 70, 80, 50),
    "user_role_admin_bg_hover": (220, 70, 80, 80),
    "user_role_admin_border": (220, 70, 80, 140),
    "user_role_admin_text": (255, 130, 130, 255),
    "user_role_operate_bg": (60, 180, 90, 50),
    "user_role_operate_bg_hover": (60, 180, 90, 80),
    "user_role_operate_border": (60, 180, 90, 140),
    "user_role_operate_text": (120, 220, 150, 255),
    "user_role_capture_bg": (230, 175, 40, 50),
    "user_role_capture_bg_hover": (230, 175, 40, 80),
    "user_role_capture_border": (230, 175, 40, 140),
    "user_role_capture_text": (240, 200, 90, 255),
    "user_role_analyze_bg": (140, 100, 210, 50),
    "user_role_analyze_bg_hover": (140, 100, 210, 80),
    "user_role_analyze_border": (140, 100, 210, 140),
    "user_role_analyze_text": (185, 155, 235, 255),
    "user_role_default_bg": (140, 148, 155, 45),
    "user_role_default_bg_hover": (140, 148, 155, 75),
    "user_role_default_border": (140, 148, 155, 130),
    "user_role_default_text": (205, 210, 215, 255),
}

LIGHT.update(_FLAT_LIGHT)  # type: ignore[typeddict-item]
DARK.update(_FLAT_DARK)  # type: ignore[typeddict-item]
LIGHT.update(_USER_ROLE_LIGHT)  # type: ignore[typeddict-item]
DARK.update(_USER_ROLE_DARK)  # type: ignore[typeddict-item]

PALETTES: dict[str, ColorTokens] = {"light": LIGHT, "dark": DARK}
