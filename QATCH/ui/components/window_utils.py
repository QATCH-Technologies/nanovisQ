"""
QATCH.ui.components.window_utils

Shared helper for locating the app's true top-level window - the one
whole-app shell that dim-overlay dialogs and floating badges/popups should
anchor and constrain themselves to.

`QApplication.activeWindow()` isn't reliable for this: several parts of the
app (Controls/Plots/Logger) start life as their own `QMainWindow` before
`ui_mode.py` reparents their central widget elsewhere, leaving a hidden
top-level widget behind - depending on focus timing, `activeWindow()` can
resolve to one of those instead of the actual visible app shell. Screen
geometry isn't the right bound either: a floating popup clamped only to the
screen can drift past the app window's own edge onto the desktop.
"""

from __future__ import annotations

from typing import Optional, Tuple, Type

from PyQt5 import QtCore, QtWidgets


def find_app_window(
    exclude_types: Tuple[Type[QtWidgets.QWidget], ...] = (),
) -> Optional[QtWidgets.QWidget]:
    """Returns the largest visible top-level widget - the true whole-app
    window - instead of whatever `QApplication.activeWindow()` reports.

    Picking the widget with the largest visible area is a simple, focus-
    timing-independent way to find the actual app shell: any hidden
    QMainWindow left behind by the reparenting pattern above is excluded by
    the visibility check, and any other genuinely-separate window (a small
    dialog, a floating badge) is smaller and so never wins.

    Args:
        exclude_types: Widget types to skip entirely (e.g. dialog classes
            that shouldn't anchor to themselves or to one another).

    Returns:
        The largest visible top-level widget, or None if nothing currently
        qualifies.
    """
    best: Optional[QtWidgets.QWidget] = None
    best_area = -1
    for tlw in QtWidgets.QApplication.topLevelWidgets():
        if not isinstance(tlw, QtWidgets.QWidget) or not tlw.isVisible():
            continue
        if exclude_types and isinstance(tlw, exclude_types):
            continue
        area = tlw.width() * tlw.height()
        if area > best_area:
            best_area = area
            best = tlw
    return best


def app_window_bounds_global(window: QtWidgets.QWidget) -> QtCore.QRect:
    """Returns `window`'s geometry in global (screen) coordinates.

    Works whether `window` is a genuine top-level widget (whose own
    `geometry()` is already screen-relative) or not, by mapping its
    top-left corner explicitly rather than assuming either case.
    """
    top_left = window.mapToGlobal(QtCore.QPoint(0, 0))
    return QtCore.QRect(top_left.x(), top_left.y(), window.width(), window.height())
