"""
QATCH.ui.components.window_utils.py

Shared helper for locating the app's true top-level window.

Finds the one whole-app shell that dim-overlay dialogs and floating badges/popups
should anchor and constrain themselves to.

:meth:`QApplication.activeWindow` isn't reliable for this: several parts of the
app (Controls/Plots/Logger) start life as their own :class:`QMainWindow` before
`ui_mode.py` reparents their central widget elsewhere, leaving a hidden
top-level widget behind - depending on focus timing, :meth:`activeWindow` can
resolve to one of those instead of the actual visible app shell. Screen
geometry isn't the right bound either: a floating popup clamped only to the
screen can drift past the app window's own edge onto the desktop.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2024-08-04
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets


def find_app_window(
    exclude_types: tuple[type[QtWidgets.QWidget], ...] = (),
) -> QtWidgets.QWidget | None:
    """Returns the largest visible top-level widget.

    Finds the true whole-app window instead of whatever :meth:`QApplication.activeWindow`
    reports. Picking the widget with the largest visible area is a simple,
    focus-timing-independent way to find the actual app shell: any hidden
    :class:`QMainWindow` left behind by the reparenting pattern above is excluded
    by the visibility check, and any other genuinely-separate window (a small
    dialog, a floating badge) is smaller and so never wins.

    Args:
        exclude_types (tuple[type[QtWidgets.QWidget], ...], optional): Widget types
            to skip entirely (e.g. dialog classes that shouldn't anchor to
            themselves or to one another). Defaults to ().

    Returns:
        QtWidgets.QWidget | None: The largest visible top-level widget, or None
        if nothing currently qualifies.
    """
    best: QtWidgets.QWidget | None = None
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
    """Returns the window's geometry in global (screen) coordinates.

    Works whether `window` is a genuine top-level widget (whose own
    :meth:`QWidget.geometry` is already screen-relative) or not, by mapping its
    top-left corner explicitly rather than assuming either case.

    Args:
        window (QtWidgets.QWidget): The widget to calculate global boundaries for.

    Returns:
        QtCore.QRect: A rectangle representing the widget's boundaries in
        global screen coordinates.
    """
    top_left = window.mapToGlobal(QtCore.QPoint(0, 0))
    return QtCore.QRect(top_left.x(), top_left.y(), window.width(), window.height())
