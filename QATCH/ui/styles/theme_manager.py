"""
QATCH.ui.styles.theme_manager

Centralized light/dark theme state for the whole app. `ThemeManager` is a
process-wide singleton that holds the active `ColorTokens` palette, persists
the chosen mode to its own small JSON file (so the login screen - which has
no user session yet - also respects it), and applies the resulting
stylesheet across the whole `QApplication`.

Widgets that paint colors directly in `paintEvent` (rather than through QSS)
should connect to `themeChanged` and re-paint from
`ThemeManager.instance().tokens()` / `.color()` rather than caching a
palette reference - see QATCH.ui.components.glass_push_button and
QATCH.ui.widgets.saved_state_dot for the established pattern.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.ui.styles.fonts import FONT_MONO
from QATCH.ui.styles.native_titlebar import apply_dark_titlebar_to_all_windows
from QATCH.ui.styles.style_loader import StyleLoader
from QATCH.ui.styles.tokens import PALETTES, ColorTokens

TAG = "[ThemeManager]"

_PREFS_FILENAME = "theme_preferences.json"


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


def _css_rgba(rgba) -> str:
    """Formats an (r, g, b, a) tuple as a literal CSS rgba(...) string."""
    r, g, b, a = rgba
    return f"rgba({r}, {g}, {b}, {a})"


def tok_css(rgba: Tuple[int, int, int, int]) -> str:
    """Converts an RGBA token tuple to a valid CSS color string.

    Args:
        rgba (Tuple[int, int, int, int]): A tuple representing Red, Green, Blue,
            and Alpha values (0-255).

    Returns:
        str: A CSS-compatible color string (either HEX or rgba).
    """
    r, g, b, a = rgba
    if a == 255:
        return f"#{r:02X}{g:02X}{b:02X}"
    return f"rgba({r}, {g}, {b}, {a})"


def desc_label_qss() -> str:
    """Shared QSS for a small muted description/subtitle QLabel, resolved
    fresh from the active theme's `flat_text_muted` token.

    Consolidates the several near-identical `_desc_qss()` copies that used
    to be hand-defined (with hardcoded, light-only colors) inside each of
    the data-management mode widgets.
    """
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 12px; "
        "background: transparent; }"
    )


def caption_label_qss() -> str:
    """Shared QSS for a small uppercase caption QLabel, resolved fresh from
    the active theme's `flat_text_muted` token.

    Consolidates the several near-identical `_caption_qss()` copies that
    used to be hand-defined (with hardcoded, light-only colors) inside each
    of the data-management mode widgets.
    """
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 10px; "
        "font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; "
        "background: transparent; }"
    )


def dialog_title_qss() -> str:
    """Shared QSS for a glass-dialog header title QLabel, resolved fresh
    from the active theme's `plot_text_bright` token.

    Used by every modal built on `QATCH.ui.components.qatch_dialog
    .GlassDialogBase` (QATCHDialog, SignatureDialog, ...) so their header
    titles render identically.
    """
    tok = ThemeManager.instance().tokens()
    return f"QLabel {{ color: {tok_css(tok['plot_text_bright'])}; font-size: 14px; font-weight: 700; }}"


def dialog_message_qss() -> str:
    """Shared QSS for a glass-dialog body QLabel, resolved fresh from the
    active theme's `plot_text_normal` token. See `dialog_title_qss`."""
    tok = ThemeManager.instance().tokens()
    return f"QLabel {{ color: {tok_css(tok['plot_text_normal'])}; font-size: 13px; }}"


def hairline_qss() -> str:
    """Shared QSS for a themed `QFrame.HLine` divider, resolved fresh from
    the active theme's `ctrl_hairline` token."""
    tok = ThemeManager.instance().tokens()
    return f"QFrame {{ border: none; background: {tok_css(tok['ctrl_hairline'])}; max-height: 1px; }}"


def close_button_qss() -> str:
    """Shared QSS for the borderless 'x' close button used by glass overlay
    panels (DataManagementWidget, UserProfilesManagerWidget, ...)."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QPushButton {{ background: transparent; border: none; "
        f"color: {tok_css(tok['flat_text_muted'])}; font-size: 18px; "
        f"font-weight: bold; padding-bottom: 2px; }} "
        f"QPushButton:hover {{ color: {tok_css(tok['flat_error'])}; }} "
        f"QPushButton:pressed {{ color: {tok_css(tok['flat_error_weak'])}; }}"
    )


def glass_panel_qss(object_name: str, alpha: int, border_width: float, radius: float) -> str:
    """QSS for an animated frosted-glass panel background/border/radius,
    resolved fresh from the active theme's `plot_glass_base`/`plot_glass_rim`
    tokens.

    alpha/border_width/radius are per-frame animation parameters (not baked
    into the token) - for overlay panels that drive their fullscreen-toggle/
    open-close motion via repeated setStyleSheet() calls rather than a
    custom paintEvent (see data_management_widget._GlassPanel for the
    paint-based alternative, used where the extra per-frame QSS reparse
    cost actually matters).
    """
    tok = ThemeManager.instance().tokens()
    base = tok["plot_glass_base"]
    rim = tok["plot_glass_rim"]
    return (
        f"QFrame#{object_name} {{ "
        f"background: rgba({base[0]}, {base[1]}, {base[2]}, {int(alpha)}); "
        f"border: {border_width:.1f}px solid {tok_css(rim)}; "
        f"border-radius: {int(radius)}px; }}"
    )


def taskbar_pill_qss(object_name: str, radius: int = 27) -> str:
    """QSS for a static rounded 'pill' toolbar strip (flat_surface2 fill +
    flat_border outline) - the top-control-bar chrome shared by overlay
    panels."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QFrame#{object_name} {{ background: {tok_css(tok['flat_surface2'])}; "
        f"border: 1px solid {tok_css(tok['flat_border'])}; "
        f"border-radius: {radius}px; }}"
    )


def surface_panel_qss(object_name: str, radius: int = 10) -> str:
    """QSS for a static raised content well (flat_surface fill + flat_border
    outline) - table/list container chrome."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QFrame#{object_name} {{ background: {tok_css(tok['flat_surface'])}; "
        f"border: 1px solid {tok_css(tok['flat_border'])}; "
        f"border-radius: {radius}px; }}"
    )


def themed_table_qss() -> str:
    """QSS for a QTableWidget + QHeaderView pairing (row/section fills,
    selection/alternating tints, hover), resolved fresh from the active
    theme's flat_* tokens."""
    tok = ThemeManager.instance().tokens()
    return f"""
        QTableWidget {{
            background-color: transparent;
            border: none;
            gridline-color: transparent;
        }}
        QTableWidget::item {{
            padding: 6px;
            border-bottom: 1px solid {tok_css(tok['flat_border'])};
            color: {tok_css(tok['flat_text'])};
        }}
        QTableWidget::item:selected {{
            background-color: {tok_css(tok['flat_accent_weak'])};
            color: {tok_css(tok['flat_text'])};
        }}
        QTableWidget::item:alternate {{
            background-color: {tok_css(tok['flat_surface2'])};
        }}
        QTableWidget QLineEdit {{
            background-color: {tok_css(tok['flat_surface'])};
            color: {tok_css(tok['flat_text'])};
            border: 1px solid {tok_css(tok['flat_accent'])};
            padding: 2px 4px;
            selection-background-color: {tok_css(tok['flat_accent_weak'])};
        }}
        QHeaderView {{
            background: transparent;
            border-top-left-radius: 9px;
            border-top-right-radius: 9px;
            border: none;
        }}
        QHeaderView::section {{
            background-color: {tok_css(tok['flat_surface2'])};
            padding: 10px;
            border: none;
            border-bottom: 1px solid {tok_css(tok['flat_border_strong'])};
            border-right: 1px solid {tok_css(tok['flat_border'])};
            font-weight: bold;
            color: {tok_css(tok['flat_text'])};
        }}
        QHeaderView::section:first {{ border-top-left-radius: 9px; }}
        QHeaderView::section:last {{ border-top-right-radius: 9px; border-right: none; }}
        QHeaderView::section:hover {{ background-color: {tok_css(tok['flat_border'])}; }}
    """


def role_wash_tokens(role_text: str):
    """Maps a role name to its (bg, bg_hover, border, text) `user_role_*`
    tokens - shared by role_badge_qss and role_combo_qss below so the two
    role-chip treatments (label badge, combo box) always agree on color."""
    tok = ThemeManager.instance().tokens()
    role_upper = str(role_text).upper()
    if "ADMIN" in role_upper:
        prefix = "user_role_admin"
    elif "OPERATE" in role_upper:
        prefix = "user_role_operate"
    elif "CAPTURE" in role_upper:
        prefix = "user_role_capture"
    elif "ANALYZE" in role_upper:
        prefix = "user_role_analyze"
    else:
        prefix = "user_role_default"
    return (
        tok[f"{prefix}_bg"],
        tok[f"{prefix}_bg_hover"],
        tok[f"{prefix}_border"],
        tok[f"{prefix}_text"],
    )


def role_badge_qss(role_text: str) -> str:
    """QSS for a role-name badge chip (QLabel), colored by role via the
    `user_role_*` wash tokens."""
    bg, _hover, border, text = role_wash_tokens(role_text)
    return (
        f"QLabel {{ background-color: {tok_css(bg)}; border: 1px solid {tok_css(border)}; "
        f"color: {tok_css(text)}; border-radius: 10px; padding: 2px 10px; "
        f"font-weight: bold; font-size: 11px; }}"
    )


def role_combo_qss(role_text: str) -> str:
    """QSS for the editable role QComboBox (closed pill + dropdown popup),
    colored by role via the `user_role_*` wash tokens; the popup itself uses
    the shared `combo_popup_*`/`combo_text`/`combo_selection_bg` tokens so it
    matches every other themed dropdown in the app."""
    tok = ThemeManager.instance().tokens()
    bg, hover_bg, border, text = role_wash_tokens(role_text)
    popup_bg = tok["combo_popup_bg"]
    item_text = tok["combo_text"]
    item_hover = tok["combo_selection_bg"]
    return f"""
        QComboBox {{
            background-color: {tok_css(bg)};
            color: {tok_css(text)};
            border: 1px solid {tok_css(border)};
            border-radius: 10px;
            padding: 2px 28px 2px 10px;
            font-weight: bold;
            font-size: 11px;
        }}
        QComboBox:hover {{
            background-color: {tok_css(hover_bg)};
            border: 1px solid {tok_css(text)};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border: none;
            border-left: 1px solid {tok_css(border)};
            border-top-right-radius: 10px;
            border-bottom-right-radius: 10px;
        }}
        QComboBox::down-arrow {{ image: none; }}
        QComboBox QAbstractItemView {{
            background-color: {tok_css(popup_bg)};
            border: 1px solid {tok_css(border)};
            border-radius: 6px;
            outline: none;
            selection-background-color: transparent;
            selection-color: {tok_css(item_text)};
            color: {tok_css(item_text)};
            padding: 4px;
        }}
        QComboBox QAbstractItemView::viewport {{
            background: transparent;
            border-radius: 6px;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 4px 10px;
            min-height: 20px;
            color: {tok_css(item_text)};
            background-color: transparent;
            border-radius: 4px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {tok_css(item_hover)};
            color: {tok_css(item_text)};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: transparent;
            color: {tok_css(item_text)};
        }}
    """


def dev_mode_status_qss(state: str, expiry: bool = False) -> str:
    """QSS for the developer-mode toggle's status label, colored by `state`
    ('error' | 'enabled' | 'disabled'). Set `expiry=True` for the smaller
    inline expiry sub-label instead of the main label."""
    tok = ThemeManager.instance().tokens()
    if state == "error":
        color = tok["flat_error"]
    elif state == "enabled":
        color = tok["flat_success"]
    else:
        color = tok["flat_text_muted"]
    if expiry:
        return (
            f"QLabel {{ color: {tok_css(color)}; font-size: 10px; "
            f"background: transparent; padding-left: 6px; }}"
        )
    return (
        f"QLabel {{ color: {tok_css(color)}; font-weight: bold; font-size: 11px; "
        f"background: transparent; }}"
    )


# =====================================================================
# Auth-card overlays (CreateUserWidget, ResetPasswordWidget) - the two
# widgets are near-identical glass cards ("Structure mirrors
# create_user_widget.py exactly", per the latter's docstring), so their
# chrome is centralized here rather than duplicated per file.
# =====================================================================
def auth_card_qss(object_name: str) -> str:
    """QSS for the centered card background these overlays float in front of
    their parent's scrim - static (no open/close alpha/radius animation,
    unlike glass_panel_qss's callers), so a fixed alpha/border/radius is
    baked in rather than left as parameters. Fully opaque (alpha 255) - the
    Add/Reset-Password cards read better solid than semi-transparent over
    a busy table underneath, especially in dark mode."""
    return glass_panel_qss(object_name, 255, 1.5, 18)


def auth_shadow_color() -> QtGui.QColor:
    """Drop-shadow color for the auth card, from the active theme's
    `flat_shadow` token (a neutral shadow tuned per-mode, rather than a
    fixed light-mode-only blue-black)."""
    tok = ThemeManager.instance().tokens()
    return QtGui.QColor(*tok["flat_shadow"])


def card_title_qss() -> str:
    """QSS for an auth card's large centered header title ('Create User',
    'Reset Password')."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 14pt; "
        f"font-weight: 700; background: transparent; }}"
    )


def error_label_qss() -> str:
    """QSS for a compact inline field-error QLabel, resolved fresh from the
    active theme's `flat_error` token."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_error'])}; font-size: 8.5pt; "
        f"font-weight: 600; background: transparent; padding-left: 6px; }}"
    )


def gradient_button_qss() -> str:
    """QSS for the circular accent-gradient submit button (auth cards'
    arrow-icon confirm action), with hover/pressed states derived from the
    active theme's `flat_accent` family instead of a fixed light-blue
    gradient that stayed identical (and low-contrast against a dark card)
    regardless of theme."""
    tok = ThemeManager.instance().tokens()
    base = QtGui.QColor(*tok["flat_accent"])
    hover = QtGui.QColor(*tok["flat_accent_hover"])
    pressed = QtGui.QColor(*tok["flat_accent_active"])

    def grad(top: QtGui.QColor, bottom: QtGui.QColor, top_a: int, bot_a: int) -> str:
        return (
            f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
            f"stop:0 rgba({top.red()}, {top.green()}, {top.blue()}, {top_a}), "
            f"stop:1 rgba({bottom.red()}, {bottom.green()}, {bottom.blue()}, {bot_a}))"
        )

    return f"""
        QPushButton {{
            background: {grad(base.lighter(115), base, 220, 200)};
            border-top: 1px solid rgba(255, 255, 255, 100);
            border-left: 1px solid rgba(255, 255, 255, 50);
            border-right: 1px solid rgba(0, 0, 0, 50);
            border-bottom: 1px solid rgba(0, 0, 0, 80);
            border-radius: 20px;
            color: {tok_css(tok['flat_on_accent'])};
            font-size: 17px;
            font-weight: bold;
            padding-left: 2px;
        }}
        QPushButton:hover {{
            background: {grad(hover.lighter(115), hover, 240, 220)};
            border-top: 1px solid rgba(255, 255, 255, 140);
        }}
        QPushButton:pressed {{
            background: {grad(pressed.lighter(115), pressed, 220, 200)};
        }}
    """


def glass_combo_qss(*, error: bool = False) -> str:
    """QSS for a plain (non role-colored) glass-style QComboBox, e.g. the
    role picker on the Create User card - distinct from role_combo_qss,
    which tints per selected role."""
    tok = ThemeManager.instance().tokens()
    input_h = 34
    if error:
        bg = tok_css(tok["input_glass_fill_error"])
        border = f"1.5px solid {tok_css(tok['input_glass_border_error'])}"
        color = tok_css(tok["flat_error"])
    else:
        bg = tok_css(tok["combo_bg"])
        border = f"1px solid {tok_css(tok['combo_border'])}"
        color = tok_css(tok["combo_text"])
    return f"""
        QComboBox {{
            background: {bg};
            border: {border};
            border-radius: {input_h // 2}px;
            padding: 0px 14px;
            color: {color};
            font-size: 10pt;
            min-height: {input_h}px;
        }}
        QComboBox:hover {{
            background: {tok_css(tok['combo_bg_hover'])};
            border: 1px solid {tok_css(tok['combo_border_hover'])};
        }}
        QComboBox:on {{
            background: {tok_css(tok['combo_bg_focus'])};
            border: 1.5px solid {tok_css(tok['combo_border_focus'])};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 32px;
            border: none;
        }}
        QComboBox::down-arrow {{ image: none; }}
        QComboBox QAbstractItemView {{
            background: {tok_css(tok['combo_popup_bg'])};
            border: 1px solid {tok_css(tok['combo_popup_border'])};
            border-radius: 10px;
            selection-background-color: transparent;
            selection-color: {tok_css(tok['combo_text'])};
            color: {tok_css(tok['combo_text'])};
            font-size: 10pt;
            padding: 4px;
            outline: none;
        }}
        QComboBox QAbstractItemView::viewport {{
            background: transparent;
            border-radius: 10px;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 4px 10px;
            min-height: 20px;
            border-radius: 5px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background: {tok_css(tok['combo_selection_bg'])};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background: {tok_css(tok['combo_selection_bg'])};
        }}
    """


def info_wash_card_qss(object_name: Optional[str] = None) -> str:
    """QSS for a translucent inset info card (ResetPasswordWidget's target-
    user summary), a lighter wash than the main auth card it sits inside."""
    tok = ThemeManager.instance().tokens()
    selector = f"QFrame#{object_name}" if object_name else "QFrame"
    return (
        f"{selector} {{ background: {tok_css(tok['flat_surface2'])}; "
        f"border: 1px solid {tok_css(tok['flat_border'])}; border-radius: 12px; }}"
    )


def accent_avatar_qss() -> str:
    """QSS for a round initials-avatar badge, accent-tinted."""
    tok = ThemeManager.instance().tokens()
    return f"""
        QLabel {{
            background: {tok_css(tok['flat_accent_weak'])};
            color: {tok_css(tok['flat_accent'])};
            border: 1.5px solid {tok_css(tok['flat_accent_ring'])};
            border-radius: 23px;
            font-weight: 700;
            font-size: 13pt;
        }}
    """


def dim_field_qss(object_name: Optional[str] = None) -> str:
    """QSS for a disabled/'coming soon' placeholder field pill, dimmed
    relative to the active inputs beside it."""
    tok = ThemeManager.instance().tokens()
    selector = f"QFrame#{object_name}" if object_name else "QFrame"
    return (
        f"{selector} {{ background: {tok_css(tok['flat_surface'])}; "
        f"border: 1px solid {tok_css(tok['flat_border'])}; border-radius: 17px; }}"
    )


def dim_text_qss(*, italic: bool = False) -> str:
    """QSS for the muted label/placeholder text inside a dim_field_qss pill."""
    tok = ThemeManager.instance().tokens()
    style = "font-style: italic; " if italic else ""
    return (
        f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; font-size: 10pt; "
        f"{style}background: transparent; }}"
    )


def dim_badge_qss() -> str:
    """QSS for the small 'coming soon' chip inside a dim_field_qss pill."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_text_muted'])}; "
        f"background: {tok_css(tok['flat_surface2'])}; "
        f"border: 1px solid {tok_css(tok['flat_border'])}; border-radius: 9px; "
        f"padding: 1px 8px; font-size: 8pt; }}"
    )


def auth_separator_qss() -> str:
    """QSS for a 1px horizontal divider rule inside an auth card."""
    tok = ThemeManager.instance().tokens()
    return f"QWidget {{ background: {tok_css(tok['flat_border'])}; }}"


def field_label_qss() -> str:
    """QSS for a compact semibold field-name QLabel that precedes an inline
    control (e.g. "Date Format" ahead of a QComboBox) - full `flat_text`
    color and a fixed weight, distinct from the muted `desc_label_qss`."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_text'])}; font-size: 12.5px; "
        "font-weight: 600; background: transparent; }"
    )


def mono_preview_qss() -> str:
    """QSS for an accent-colored monospace preview readout (e.g. a
    generated file-name/date-time pattern), resolved fresh from the active
    theme's `flat_accent` token."""
    tok = ThemeManager.instance().tokens()
    return (
        f"QLabel {{ color: {tok_css(tok['flat_accent'])}; "
        f"font-family: '{FONT_MONO}'; font-size: 12.5px; background: transparent; }}"
    )


def window_canvas_qss(object_name: str) -> str:
    """QSS for a standalone top-level QWidget's own background fill
    (`flat_surface2`) - for windows that keep their native OS title bar
    (no frameless/translucent chrome) but still need a theme-correct canvas
    instead of falling back to the OS's default (light-only) widget
    background."""
    tok = ThemeManager.instance().tokens()
    return f"QWidget#{object_name} {{ background: {tok_css(tok['flat_surface2'])}; }}"


class ThemeManager(QtCore.QObject):
    """Process-wide singleton owning the active light/dark palette.

    Not constructed directly - use `ThemeManager.instance()`. A `QObject`
    subclass purely so it can own the `themeChanged` signal; it does not
    need a parent and outlives every widget, so dependents connecting to
    its signal don't need to manage disconnection themselves.
    """

    themeChanged = QtCore.pyqtSignal(str)  # emits the new ThemeMode.value

    _instance: Optional["ThemeManager"] = None

    def __init__(self) -> None:
        super().__init__()
        self._mode: ThemeMode = self._load_persisted_mode()
        self._loader = StyleLoader(
            base_dir=os.path.dirname(os.path.abspath(__file__)),
            theme_file="app_theme.qss",
            styles_subdir="",
        )

    @classmethod
    def instance(cls) -> "ThemeManager":
        """Returns the process-wide ThemeManager, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def mode(self) -> ThemeMode:
        """Returns the currently active ThemeMode."""
        return self._mode

    def tokens(self) -> ColorTokens:
        """Returns the active palette's ColorTokens dict."""
        return PALETTES[self._mode.value]

    def color(self, key: str) -> QtGui.QColor:
        """Returns the active palette's token at `key` as a QColor."""
        return QtGui.QColor(*self.tokens()[key])

    def set_mode(self, mode: ThemeMode, persist: bool = True) -> None:
        """Switches the active theme, re-applies the app stylesheet, and
        notifies paint-time widgets via `themeChanged`.

        Args:
            mode (ThemeMode): The theme to switch to.
            persist (bool, optional): If True (default), writes the choice
                to disk so it's restored on the next launch - including
                before any user signs in.
        """
        if mode == self._mode:
            return
        self._mode = mode

        app = QtWidgets.QApplication.instance()
        if app is not None:
            self.apply_app_stylesheet(app)

        if persist:
            self._save_persisted_mode(mode)

        self.themeChanged.emit(mode.value)

    def apply_app_stylesheet(self, app: QtWidgets.QApplication) -> None:
        """Applies `app_theme.qss` (with the active palette substituted in)
        across the whole `QApplication`.

        Args:
            app (QtWidgets.QApplication): The running application instance.
        """
        token_placeholders = {key.upper(): _css_rgba(value) for key, value in self.tokens().items()}
        self._loader.set_tokens(token_placeholders)
        try:
            app.setStyleSheet(self._loader.get_stylesheet(use_cache=False))
        except FileNotFoundError as exc:
            Log.e(TAG, f"Could not apply theme stylesheet: {exc}")
            return

        self._repolish_dropdowns()
        # QSS only reaches content Qt itself paints, not the OS-drawn title
        # bar/frame - Windows' native dark title bar is a separate DWM call
        # (see native_titlebar.py) applied here to every live top-level
        # window so it stays in sync with the rest of the app.
        apply_dark_titlebar_to_all_windows(self._mode == ThemeMode.DARK)

    @staticmethod
    def _repolish_dropdowns() -> None:
        """Forces an immediate style repolish on every live QComboBox/QMenu
        (and combo popup view).

        `QApplication.setStyleSheet()` is supposed to refresh every widget,
        but "complex control" widgets whose popup renders as a separate
        top-level window - QComboBox chief among them - don't reliably
        pick up the new stylesheet until Qt recomputes their style for some
        other reason (e.g. the popup being shown). Symptom: switching theme
        left closed dropdowns showing the *previous* theme's colors until
        the user opened one, at which point the colors visibly "swapped".
        Explicitly unpolishing+polishing closes that gap immediately
        instead of waiting for an incidental interaction to trigger it.
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        for widget in app.allWidgets():
            if isinstance(widget, (QtWidgets.QComboBox, QtWidgets.QMenu)):
                style = widget.style()
                style.unpolish(widget)
                style.polish(widget)
                widget.update()
                if isinstance(widget, QtWidgets.QComboBox):
                    view = widget.view()
                    if view is not None:
                        view.style().unpolish(view)
                        view.style().polish(view)
                        # unpolish()/polish() alone doesn't reliably force
                        # an already-painted viewport to actually redraw -
                        # explicitly invalidate it so item delegates that
                        # read colors straight from ThemeManager (not QSS)
                        # repaint immediately instead of showing the
                        # previous theme's colors until the popup happens
                        # to be shown again.
                        view.viewport().update()

    def _prefs_path(self) -> str:
        return os.path.join(Constants.local_app_data_path, _PREFS_FILENAME)

    def _load_persisted_mode(self) -> ThemeMode:
        path = self._prefs_path()
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return ThemeMode(data.get("mode", ThemeMode.LIGHT.value))
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            Log.d(TAG, f"No persisted theme preference found ({exc}); defaulting to light.")
            return ThemeMode.LIGHT

    def _save_persisted_mode(self, mode: ThemeMode) -> None:
        path = self._prefs_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump({"mode": mode.value}, f)
        except OSError as exc:
            Log.e(TAG, f"Could not persist theme preference: {exc}")
