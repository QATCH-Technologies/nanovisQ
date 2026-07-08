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
