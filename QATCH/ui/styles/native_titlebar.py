"""
QATCH.ui.styles.native_titlebar

Applies Windows' native dark/light title bar to top-level windows via the
DWM (Desktop Window Manager) API. Qt has no cross-platform way to theme a
window's own OS-drawn title bar/frame - QSS only reaches content Qt itself
paints - so this reaches past Qt into `ctypes.windll.dwmapi` directly.

No-ops (and never raises) on any non-Windows platform, or if the DWM call
itself isn't available/fails for any reason - the title bar just keeps its
default OS-drawn look in that case, which is a safe fallback, not a broken
one.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

import ctypes
import sys

from PyQt5 import QtWidgets

from QATCH.common.logger import Logger as Log

TAG = "[native_titlebar]"

# DWMWA_USE_IMMERSIVE_DARK_MODE: 20 on Windows 10 20H1 (build 19041)+ and
# Windows 11. Some earlier Windows 10 builds (1809-1909) used 19 for the
# same attribute before it was assigned a stable id - try the modern value
# first and fall back to the legacy one if it's rejected.
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY = 19

_SWP_NOMOVE = 0x0002
_SWP_NOSIZE = 0x0001
_SWP_NOZORDER = 0x0004
_SWP_FRAMECHANGED = 0x0020

_IS_WINDOWS = sys.platform == "win32"


def set_window_dark_titlebar(widget: QtWidgets.QWidget, dark: bool) -> None:
    """Sets `widget`'s native OS title bar to dark or light mode (Windows only).

    Safe to call on any top-level widget at any time, before or after
    `show()` - reading `winId()` forces native window creation if it
    hasn't happened yet, which is a normal, supported thing to do. A no-op
    on non-Windows platforms or if the DWM call fails for any reason.

    Args:
        widget: The top-level widget whose title bar to theme.
        dark: True for a dark title bar, False for the default light one.
    """
    if not _IS_WINDOWS:
        return
    try:
        hwnd = int(widget.winId())
        value = ctypes.c_int(1 if dark else 0)
        dwmapi = ctypes.windll.dwmapi
        result = dwmapi.DwmSetWindowAttribute(
            hwnd,
            _DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
        if result != 0:  # non-zero HRESULT: the modern attribute id was rejected
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                _DWMWA_USE_IMMERSIVE_DARK_MODE_LEGACY,
                ctypes.byref(value),
                ctypes.sizeof(value),
            )
        # Force the non-client area (title bar) to redraw immediately - on
        # some Windows builds it otherwise doesn't repaint until the window
        # is next resized or moved.
        ctypes.windll.user32.SetWindowPos(
            hwnd,
            None,
            0,
            0,
            0,
            0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_FRAMECHANGED,
        )
    except Exception as exc:
        Log.d(TAG, f"Could not set native title bar theme: {exc}")


def apply_dark_titlebar_to_all_windows(dark: bool) -> None:
    """Applies `set_window_dark_titlebar` to every currently-live top-level
    widget in the application (Windows only; no-op elsewhere).

    Args:
        dark: True for dark title bars, False for the default light ones.
    """
    if not _IS_WINDOWS:
        return
    app = QtWidgets.QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        if isinstance(widget, QtWidgets.QWidget):
            set_window_dark_titlebar(widget, dark)
