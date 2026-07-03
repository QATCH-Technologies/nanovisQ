"""
QATCH.ui.styles.fonts

Registers the bundled IBM Plex Sans / IBM Plex Mono weights used by the
flat control system (Pushbutton, Line Edit, Combo Box, Spin Box, Toggle,
Option Card - see QATCH.ui.components). Scoped to those components only;
this does NOT change the app's default font.

Font files live under QATCH/resources/fonts/ (SIL Open Font License,
sourced from the official IBM Plex GitHub releases - see
LICENSE-IBMPlex.txt alongside them). That folder is already bundled by
the PyInstaller spec via its recursive "QATCH/resources" rule, so no
packaging changes are needed.

Usage
-----
    from QATCH.ui.styles.fonts import register_app_fonts
    register_app_fonts()   # call once, after QApplication() exists

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

import os

from PyQt5 import QtGui

from QATCH.common.logger import Logger as Log

TAG = "[fonts]"

FONT_SANS = "IBM Plex Sans"
FONT_MONO = "IBM Plex Mono"

# Qt's font database registers non-Regular/Bold static weights (Medium,
# SemiBold) as their own family name rather than as a weight variant of the
# base family - `QFont(FONT_SANS).setWeight(...)` does not reliably switch
# to them across platforms. Use these exact, verified family names instead
# (confirmed via QFontDatabase().families() after registration).
FONT_SANS_MEDIUM = "IBM Plex Sans Medm"
FONT_SANS_SEMIBOLD = "IBM Plex Sans SmBld"
FONT_MONO_MEDIUM = "IBM Plex Mono Medm"

_FONT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "fonts"
)

_FONT_FILES = [
    "IBMPlexSans-Regular.ttf",
    "IBMPlexSans-Medium.ttf",
    "IBMPlexSans-SemiBold.ttf",
    "IBMPlexSans-Bold.ttf",
    "IBMPlexMono-Regular.ttf",
    "IBMPlexMono-Medium.ttf",
]

_registered = False


def register_app_fonts() -> None:
    """Loads the bundled IBM Plex font files into the application font database.

    Safe to call more than once (a no-op after the first call). Never raises:
    a missing or unreadable font file is logged and skipped, and any
    component that requests "IBM Plex Sans"/"IBM Plex Mono" by family name
    will silently fall back to Qt's default font substitution if the family
    didn't register - degrading gracefully rather than crashing.
    """
    global _registered
    if _registered:
        return
    _registered = True

    for fname in _FONT_FILES:
        path = os.path.join(_FONT_DIR, fname)
        if not os.path.isfile(path):
            Log.w(TAG, f"Bundled font not found, skipping: {path}")
            continue
        font_id = QtGui.QFontDatabase.addApplicationFont(path)
        if font_id == -1:
            Log.w(TAG, f"Failed to register bundled font: {fname}")
