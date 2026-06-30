"""
QATCH.ui.styles.tokens

Semantic color tokens for the app's light and dark themes.

Each token is a plain (r, g, b, a) tuple - alpha 0-255 - mirroring the
existing convention in QATCH.ui.components.glass_push_button._PALETTES and
QATCH.ui.widgets.saved_state_dot._COLORS, so a token is a drop-in
QtGui.QColor(*token) the same way those call sites already build colors.

`DARK` is a real first-pass dark palette, not placeholder values - it has
not been through a dedicated contrast/design review yet.
"""

from __future__ import annotations

from typing import Tuple, TypedDict

RGBA = Tuple[int, int, int, int]


class ColorTokens(TypedDict):
    bg_gradient_start: RGBA
    bg_gradient_end: RGBA
    surface: RGBA
    surface_border: RGBA
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
    # Log console — container / controls
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
    # Log console — per-level text colors (used in inline HTML spans)
    log_time: RGBA
    log_location: RGBA
    log_debug: RGBA
    log_info: RGBA
    log_warning: RGBA
    log_error: RGBA
    log_default: RGBA
    # Plot glass cards — paintEvent fill colors
    plot_glass_base: RGBA
    plot_glass_overlay: RGBA
    plot_glass_shimmer_top: RGBA
    plot_glass_shimmer_mid: RGBA
    plot_glass_vignette_end: RGBA
    plot_glass_rim: RGBA
    plot_glass_inset: RGBA
    plot_glass_header_line: RGBA
    # Plot glass cards — text
    plot_text_normal: RGBA
    plot_text_bright: RGBA
    plot_text_muted: RGBA
    plot_text_dim: RGBA
    # Plot glass cards — icon buttons
    plot_icon_btn_hover_bg: RGBA
    plot_icon_btn_hover_border: RGBA
    plot_icon_btn_pressed_bg: RGBA
    # Plot glass cards — dropdown menu
    plot_menu_bg: RGBA
    plot_menu_border: RGBA
    plot_menu_separator: RGBA
    plot_menu_row_hover: RGBA
    plot_swatch_border: RGBA
    # Plot glass cards — device tabs (pill buttons)
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
    # Controls UI — toolbar
    ctrl_toolbar_btn_disabled_text: RGBA
    ctrl_toolbar_btn_pressed_bg: RGBA
    ctrl_toolbar_separator: RGBA
    # Controls UI — progress bar
    ctrl_progress_border: RGBA
    ctrl_progress_chunk_start: RGBA
    ctrl_progress_chunk_end: RGBA
    # Controls UI — temperature controller
    ctrl_temp_ctrl_bg: RGBA
    ctrl_temp_pid_header_text: RGBA
    ctrl_temp_status_offline_bg: RGBA
    ctrl_temp_status_offline_text: RGBA
    ctrl_temp_status_border: RGBA
    # Controls UI — sliders (shared: temp controller + range sliders)
    ctrl_slider_groove: RGBA
    ctrl_slider_handle_border: RGBA
    ctrl_slider_track: RGBA
    ctrl_slider_disabled_handle: RGBA
    ctrl_slider_handle_hover: RGBA
    # Controls UI — hairline divider
    ctrl_hairline: RGBA
    # Controls UI — toggle row labels
    ctrl_toggle_label_text: RGBA
    # Controls UI — device config back button
    ctrl_back_btn_bg: RGBA
    ctrl_back_btn_border: RGBA
    ctrl_back_btn_hover_bg: RGBA
    ctrl_back_btn_hover_border: RGBA
    # Controls UI — glass input fields
    ctrl_input_bg: RGBA
    ctrl_input_border: RGBA
    ctrl_input_text: RGBA
    ctrl_input_focus_bg: RGBA
    ctrl_input_focus_border: RGBA
    # Controls UI — temperature status dynamic states
    ctrl_temp_ready_bg: RGBA
    ctrl_temp_ready_text: RGBA
    ctrl_temp_heating_bg: RGBA
    ctrl_temp_heating_text: RGBA
    ctrl_temp_cooling_bg: RGBA
    ctrl_temp_cooling_text: RGBA
    # Controls UI — amber pulse border (unsaved field highlight)
    ctrl_pulse_border: RGBA
    # Controls UI — infobar readout label
    ctrl_infobar_text: RGBA
    # Controls window — menu bar signed-in (normal) state
    menubar_bg: RGBA
    menubar_text: RGBA
    menubar_item_hover_bg: RGBA
    menubar_item_disabled_text: RGBA
    menubar_border: RGBA
    menubar_separator: RGBA
    # Controls window — menu bar signed-out (dimmed) state
    menubar_dim_bg: RGBA
    menubar_dim_text: RGBA
    menubar_dim_item_hover_bg: RGBA
    menubar_dim_item_disabled_text: RGBA
    menubar_dim_border: RGBA
    menubar_dim_separator: RGBA


# Lifted directly from QATCH/ui/styles/app_theme.qss (formerly ui_main_theme.qss)
# and the "default" variant of glass_push_button._PALETTES.
LIGHT: ColorTokens = {
    "bg_gradient_start": (0xE4, 0xEB, 0xF1, 255),
    "bg_gradient_end": (0xF4, 0xF7, 0xF9, 255),
    "surface": (255, 255, 255, 160),
    "surface_border": (255, 255, 255, 220),
    "menu_item_hover": (229, 229, 229, 150),
    "text_primary": (51, 51, 51, 255),
    "text_secondary": (60, 60, 60, 180),
    "accent": (10, 163, 230, 255),
    "accent_translucent": (10, 163, 230, 38),
    "scrollbar_handle": (130, 130, 130, 100),
    "scrollbar_handle_hover": (130, 130, 130, 180),
    "overlay_dim": (164, 168, 172, 255),
    "danger": (220, 53, 69, 255),
    "warning": (255, 193, 7, 255),
    "success": (60, 190, 120, 255),
    "backdrop_fallback_start": (0xD8, 0xE6, 0xF0, 255),
    "backdrop_fallback_end": (0xEE, 0xF4, 0xF8, 255),
    "backdrop_frost": (238, 243, 247, 62),
    "backdrop_dim": (0, 0, 0, 76),
    # Log console — container / controls
    "log_surface": (255, 255, 255, 120),
    "log_surface_border": (255, 255, 255, 200),
    "log_control_bg": (255, 255, 255, 160),
    "log_control_bg_hover": (255, 255, 255, 200),
    "log_control_bg_focus": (255, 255, 255, 220),
    "log_control_border": (120, 130, 145, 160),
    "log_control_border_hover": (90, 100, 115, 200),
    "log_dropdown_bg": (245, 247, 250, 255),
    "log_dropdown_border": (200, 200, 200, 180),
    "log_separator": (120, 130, 145, 110),
    "log_btn_hover": (255, 255, 255, 180),
    "log_btn_pressed": (255, 255, 255, 220),
    "log_text": (30, 40, 55, 200),
    "log_match_highlight": (10, 163, 230, 90),
    # Log console — per-level text colors
    "log_time": (0, 131, 143, 255),
    "log_location": (142, 36, 170, 255),
    "log_debug": (120, 144, 156, 255),
    "log_info": (46, 125, 50, 255),
    "log_warning": (230, 81, 0, 255),
    "log_error": (198, 40, 40, 255),
    "log_default": (51, 51, 51, 255),
    # Plot glass cards — paintEvent fill colors
    "plot_glass_base": (255, 255, 255, 160),
    "plot_glass_overlay": (228, 235, 241, 18),
    "plot_glass_shimmer_top": (255, 255, 255, 100),
    "plot_glass_shimmer_mid": (255, 255, 255, 20),
    "plot_glass_vignette_end": (200, 218, 240, 18),
    "plot_glass_rim": (255, 255, 255, 230),
    "plot_glass_inset": (190, 210, 235, 70),
    "plot_glass_header_line": (195, 215, 238, 70),
    # Plot glass cards — text
    "plot_text_normal": (30, 40, 55, 200),
    "plot_text_bright": (30, 40, 55, 235),
    "plot_text_muted": (30, 40, 55, 155),
    "plot_text_dim": (30, 40, 55, 110),
    # Plot glass cards — icon buttons
    "plot_icon_btn_hover_bg": (255, 255, 255, 160),
    "plot_icon_btn_hover_border": (255, 255, 255, 200),
    "plot_icon_btn_pressed_bg": (180, 215, 255, 190),
    # Plot glass cards — dropdown menu
    "plot_menu_bg": (232, 242, 252, 248),
    "plot_menu_border": (255, 255, 255, 230),
    "plot_menu_separator": (175, 200, 228, 90),
    "plot_menu_row_hover": (46, 155, 218, 28),
    "plot_swatch_border": (255, 255, 255, 210),
    # Plot glass cards — device tabs (pill buttons)
    "plot_tab_bg": (255, 255, 255, 55),
    "plot_tab_border": (255, 255, 255, 110),
    "plot_tab_bg_hover": (255, 255, 255, 130),
    "plot_tab_bg_active": (255, 255, 255, 215),
    "plot_tab_border_active": (255, 255, 255, 255),
    "plot_tab_bg_active_hover": (255, 255, 255, 240),
    # Plot data line default colors (same in both themes — content, not chrome)
    "plot_data_primary": (46, 155, 218, 255),
    "plot_data_secondary": (240, 100, 53, 255),
    "plot_data_temperature": (240, 156, 53, 255),
    "plot_data_device_accent": (72, 190, 120, 255),
    # Controls UI — toolbar
    "ctrl_toolbar_btn_disabled_text": (30, 40, 55, 90),
    "ctrl_toolbar_btn_pressed_bg": (229, 229, 229, 200),
    "ctrl_toolbar_separator": (0, 0, 0, 22),
    # Controls UI — progress bar
    "ctrl_progress_border": (0, 0, 0, 25),
    "ctrl_progress_chunk_start": (10, 163, 230, 130),
    "ctrl_progress_chunk_end": (10, 163, 230, 90),
    # Controls UI — temperature controller
    "ctrl_temp_ctrl_bg": (229, 229, 229, 80),
    "ctrl_temp_pid_header_text": (0, 118, 174, 220),
    "ctrl_temp_status_offline_bg": (150, 155, 160, 120),
    "ctrl_temp_status_offline_text": (30, 40, 55, 160),
    "ctrl_temp_status_border": (255, 255, 255, 160),
    # Controls UI — sliders
    "ctrl_slider_groove": (0, 0, 0, 30),
    "ctrl_slider_handle_border": (0, 130, 200, 200),
    "ctrl_slider_track": (10, 163, 230, 120),
    "ctrl_slider_disabled_handle": (150, 170, 190, 140),
    "ctrl_slider_handle_hover": (31, 179, 240, 255),
    # Controls UI — hairline divider
    "ctrl_hairline": (200, 210, 220, 130),
    # Controls UI — toggle row labels
    "ctrl_toggle_label_text": (30, 40, 55, 215),
    # Controls UI — device config back button
    "ctrl_back_btn_bg": (255, 255, 255, 40),
    "ctrl_back_btn_border": (255, 255, 255, 100),
    "ctrl_back_btn_hover_bg": (255, 255, 255, 80),
    "ctrl_back_btn_hover_border": (255, 255, 255, 150),
    # Controls UI — glass input fields
    "ctrl_input_bg": (255, 255, 255, 60),
    "ctrl_input_border": (255, 255, 255, 120),
    "ctrl_input_text": (30, 40, 55, 220),
    "ctrl_input_focus_bg": (255, 255, 255, 180),
    "ctrl_input_focus_border": (0, 120, 215, 150),
    # Controls UI — temperature status dynamic states
    "ctrl_temp_ready_bg": (60, 200, 90, 220),
    "ctrl_temp_ready_text": (255, 255, 255, 230),
    "ctrl_temp_heating_bg": (240, 190, 0, 220),
    "ctrl_temp_heating_text": (30, 20, 0, 200),
    "ctrl_temp_cooling_bg": (240, 140, 0, 220),
    "ctrl_temp_cooling_text": (30, 20, 0, 200),
    # Controls UI — amber pulse border
    "ctrl_pulse_border": (240, 170, 50, 230),
    # Controls UI — infobar readout
    "ctrl_infobar_text": (70, 90, 110, 190),
    # Controls window — menu bar signed-in (normal) state
    "menubar_bg": (233, 239, 244, 255),
    "menubar_text": (50, 60, 70, 230),
    "menubar_item_hover_bg": (10, 163, 230, 60),
    "menubar_item_disabled_text": (120, 130, 140, 140),
    "menubar_border": (255, 255, 255, 230),
    "menubar_separator": (120, 130, 140, 70),
    # Controls window — menu bar signed-out (dimmed) state
    "menubar_dim_bg": (163, 167, 171, 255),
    "menubar_dim_text": (40, 48, 56, 235),
    "menubar_dim_item_hover_bg": (255, 255, 255, 60),
    "menubar_dim_item_disabled_text": (90, 98, 106, 150),
    "menubar_dim_border": (255, 255, 255, 90),
    "menubar_dim_separator": (255, 255, 255, 80),
}

# First-pass dark palette: dark surfaces, lightened accent/text for contrast.
DARK: ColorTokens = {
    "bg_gradient_start": (0x1B, 0x20, 0x26, 255),
    "bg_gradient_end": (0x23, 0x29, 0x30, 255),
    "surface": (40, 46, 54, 170),
    "surface_border": (70, 78, 88, 160),
    "menu_item_hover": (60, 68, 78, 150),
    "text_primary": (230, 234, 238, 255),
    "text_secondary": (170, 178, 188, 200),
    "accent": (45, 175, 240, 255),
    "accent_translucent": (45, 175, 240, 46),
    "scrollbar_handle": (110, 116, 124, 140),
    "scrollbar_handle_hover": (140, 146, 156, 190),
    "overlay_dim": (30, 34, 40, 255),
    "danger": (235, 90, 90, 255),
    "warning": (255, 200, 80, 255),
    "success": (80, 210, 140, 255),
    "backdrop_fallback_start": (0x20, 0x26, 0x2E, 255),
    "backdrop_fallback_end": (0x14, 0x18, 0x1E, 255),
    "backdrop_frost": (60, 68, 78, 70),
    "backdrop_dim": (0, 0, 0, 110),
    # Log console — container / controls
    "log_surface": (40, 46, 54, 140),
    "log_surface_border": (70, 78, 88, 180),
    "log_control_bg": (50, 58, 68, 160),
    "log_control_bg_hover": (60, 68, 80, 200),
    "log_control_bg_focus": (65, 74, 86, 220),
    "log_control_border": (80, 90, 105, 160),
    "log_control_border_hover": (100, 110, 125, 200),
    "log_dropdown_bg": (35, 40, 50, 255),
    "log_dropdown_border": (70, 78, 88, 180),
    "log_separator": (80, 88, 100, 110),
    "log_btn_hover": (55, 63, 75, 180),
    "log_btn_pressed": (65, 73, 85, 220),
    "log_text": (200, 210, 220, 200),
    "log_match_highlight": (45, 175, 240, 90),
    # Log console — per-level text colors
    "log_time": (0, 188, 212, 255),
    "log_location": (206, 147, 216, 255),
    "log_debug": (144, 164, 174, 255),
    "log_info": (102, 187, 106, 255),
    "log_warning": (255, 152, 0, 255),
    "log_error": (239, 83, 80, 255),
    "log_default": (204, 204, 204, 255),
    # Plot glass cards — paintEvent fill colors
    "plot_glass_base": (45, 52, 62, 160),
    "plot_glass_overlay": (60, 68, 80, 18),
    "plot_glass_shimmer_top": (80, 90, 105, 60),
    "plot_glass_shimmer_mid": (80, 90, 105, 15),
    "plot_glass_vignette_end": (10, 14, 22, 30),
    "plot_glass_rim": (90, 100, 115, 180),
    "plot_glass_inset": (60, 70, 90, 70),
    "plot_glass_header_line": (55, 65, 85, 70),
    # Plot glass cards — text
    "plot_text_normal": (210, 218, 230, 200),
    "plot_text_bright": (220, 228, 238, 235),
    "plot_text_muted": (170, 180, 195, 155),
    "plot_text_dim": (160, 170, 185, 130),
    # Plot glass cards — icon buttons
    "plot_icon_btn_hover_bg": (55, 63, 75, 160),
    "plot_icon_btn_hover_border": (90, 100, 120, 180),
    "plot_icon_btn_pressed_bg": (30, 80, 150, 190),
    # Plot glass cards — dropdown menu
    "plot_menu_bg": (38, 45, 56, 248),
    "plot_menu_border": (70, 80, 95, 200),
    "plot_menu_separator": (70, 80, 100, 90),
    "plot_menu_row_hover": (45, 175, 240, 35),
    "plot_swatch_border": (90, 100, 120, 180),
    # Plot glass cards — device tabs (pill buttons)
    "plot_tab_bg": (50, 58, 70, 55),
    "plot_tab_border": (70, 80, 98, 110),
    "plot_tab_bg_hover": (58, 67, 82, 130),
    "plot_tab_bg_active": (65, 75, 90, 215),
    "plot_tab_border_active": (95, 108, 125, 255),
    "plot_tab_bg_active_hover": (72, 83, 98, 240),
    # Plot data line default colors (same in both themes — content, not chrome)
    "plot_data_primary": (46, 155, 218, 255),
    "plot_data_secondary": (240, 100, 53, 255),
    "plot_data_temperature": (240, 156, 53, 255),
    "plot_data_device_accent": (72, 190, 120, 255),
    # Controls UI — toolbar
    "ctrl_toolbar_btn_disabled_text": (210, 218, 230, 90),
    "ctrl_toolbar_btn_pressed_bg": (60, 68, 78, 200),
    "ctrl_toolbar_separator": (255, 255, 255, 22),
    # Controls UI — progress bar
    "ctrl_progress_border": (255, 255, 255, 25),
    "ctrl_progress_chunk_start": (45, 175, 240, 130),
    "ctrl_progress_chunk_end": (45, 175, 240, 90),
    # Controls UI — temperature controller
    "ctrl_temp_ctrl_bg": (40, 46, 54, 80),
    "ctrl_temp_pid_header_text": (45, 175, 240, 220),
    "ctrl_temp_status_offline_bg": (60, 65, 70, 120),
    "ctrl_temp_status_offline_text": (180, 190, 200, 160),
    "ctrl_temp_status_border": (90, 100, 115, 160),
    # Controls UI — sliders
    "ctrl_slider_groove": (255, 255, 255, 30),
    "ctrl_slider_handle_border": (30, 150, 220, 200),
    "ctrl_slider_track": (45, 175, 240, 120),
    "ctrl_slider_disabled_handle": (80, 90, 100, 140),
    "ctrl_slider_handle_hover": (70, 195, 255, 255),
    # Controls UI — hairline divider
    "ctrl_hairline": (60, 70, 80, 130),
    # Controls UI — toggle row labels
    "ctrl_toggle_label_text": (210, 218, 230, 215),
    # Controls UI — device config back button
    "ctrl_back_btn_bg": (50, 60, 75, 40),
    "ctrl_back_btn_border": (80, 90, 110, 100),
    "ctrl_back_btn_hover_bg": (65, 75, 95, 80),
    "ctrl_back_btn_hover_border": (100, 115, 140, 150),
    # Controls UI — glass input fields
    "ctrl_input_bg": (50, 58, 70, 60),
    "ctrl_input_border": (70, 80, 98, 120),
    "ctrl_input_text": (210, 218, 230, 220),
    "ctrl_input_focus_bg": (65, 75, 90, 180),
    "ctrl_input_focus_border": (40, 130, 240, 180),
    # Controls UI — temperature status dynamic states
    "ctrl_temp_ready_bg": (50, 190, 100, 220),
    "ctrl_temp_ready_text": (255, 255, 255, 230),
    "ctrl_temp_heating_bg": (220, 170, 0, 220),
    "ctrl_temp_heating_text": (255, 240, 180, 200),
    "ctrl_temp_cooling_bg": (220, 120, 0, 220),
    "ctrl_temp_cooling_text": (255, 220, 170, 200),
    # Controls UI — amber pulse border
    "ctrl_pulse_border": (255, 190, 70, 230),
    # Controls UI — infobar readout
    "ctrl_infobar_text": (150, 165, 180, 190),
    # Controls window — menu bar signed-in (normal) state
    "menubar_bg": (35, 40, 48, 255),
    "menubar_text": (200, 210, 220, 230),
    "menubar_item_hover_bg": (45, 175, 240, 60),
    "menubar_item_disabled_text": (90, 100, 110, 140),
    "menubar_border": (70, 78, 88, 200),
    "menubar_separator": (70, 80, 95, 80),
    # Controls window — menu bar signed-out (dimmed) state
    "menubar_dim_bg": (45, 50, 58, 255),
    "menubar_dim_text": (150, 160, 170, 235),
    "menubar_dim_item_hover_bg": (80, 90, 105, 60),
    "menubar_dim_item_disabled_text": (80, 90, 100, 140),
    "menubar_dim_border": (60, 68, 78, 90),
    "menubar_dim_separator": (80, 88, 100, 80),
}

PALETTES: dict[str, ColorTokens] = {"light": LIGHT, "dark": DARK}
