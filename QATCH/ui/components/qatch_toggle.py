"""
qatch_toggle.py

Animated toggle switch matching the app's flat control system (see
QATCH.ui.components.flat_paint).

Track color interpolates from a neutral grey (off) to the accent color
(on) - both driven by the "flat_*" tokens in QATCH.ui.styles.tokens so the
toggle stays in sync with light/dark theme changes. A knob slides left /
right with an OutCubic easing over 150 ms.

Usage
-----
    toggle = QATCHToggle(parent)
    toggle.setChecked(True)           # set initial state (no animation)
    toggle.toggled.connect(handler)   # bool signal, same as QCheckBox

    # For silent initialisation (avoid triggering the handler):
    toggle.setChecked(value)          # connect signal AFTER this call
    toggle.toggled.connect(handler)
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager


class QATCHToggle(QtWidgets.QAbstractButton):
    """Pill-shaped animated toggle switch.

    Inherits `toggled(bool)` from `QAbstractButton` - a drop-in
    replacement for `QCheckBox` wherever only the checked state matters.

    Attributes:
        _anim_t (float): Thumb position 0.0 (off / left) → 1.0 (on / right).
    """

    # ── Geometry ──────────────────────────────────────────────────────
    _TRACK_W: int = 42
    _TRACK_H: int = 23
    _THUMB_D: int = 18  # diameter; margin = (_TRACK_H - _THUMB_D) / 2 = 2.5 px

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(self._TRACK_W, self._TRACK_H)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

        self._anim_t: float = 0.0

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)

        # toggled fires after the internal checked state flips, so
        # _anim_t correctly approaches the new target.
        self.toggled.connect(self._start_anim)

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def _start_anim(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(float(self._anim_t))
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def _on_anim_step(self, v: float) -> None:
        self._anim_t = v
        self.update()

    # ------------------------------------------------------------------
    # Override setChecked to snap the thumb without animation when the
    # initial state is set programmatically (before any signal fires).
    # ------------------------------------------------------------------
    def setChecked(self, checked: bool) -> None:
        # Snap anim_t so the thumb appears in the correct position
        # immediately - avoids a jarring mid-paint initial frame.
        self._anim_t = 1.0 if checked else 0.0
        super().setChecked(checked)

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    @staticmethod
    def _lerp(a: int, b: int, t: float) -> int:
        return int(a + (b - a) * t)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        t = self._anim_t
        w, h = self.width(), self.height()
        r = h / 2.0  # track corner radius - full pill

        tok = ThemeManager.instance().tokens()

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        if not self.isEnabled():
            p.setOpacity(0.45)

        # ── Track fill (interpolated colour, no border per spec) ──────
        lo, hi = tok["flat_track"], tok["flat_accent"]
        track_color = QtGui.QColor(
            self._lerp(lo[0], hi[0], t),
            self._lerp(lo[1], hi[1], t),
            self._lerp(lo[2], hi[2], t),
            self._lerp(lo[3], hi[3], t),
        )
        ring = QtGui.QColor(*tok["flat_accent_ring"]) if self.hasFocus() else None
        paint_flat_surface(
            self,
            radius=r,
            fill=track_color,
            border=track_color,
            border_width=0.0,
            ring=ring,
            painter=p,
        )

        # ── Thumb ────────────────────────────────────────────────────
        margin = (h - self._THUMB_D) / 2.0
        x_left = margin
        x_right = w - margin - self._THUMB_D
        thumb_x = x_left + (x_right - x_left) * t
        thumb = QtCore.QRectF(thumb_x, margin, self._THUMB_D, self._THUMB_D)

        # Soft drop shadow (offset 1 px down, flat_shadow token)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(*tok["flat_shadow"])))
        p.drawEllipse(thumb.adjusted(0.0, 1.0, 0.0, 1.0))

        # Thumb face: knob token when off, literal white when on (spec's
        # "on" knob is always white in both themes; the "off" knob follows
        # the flat_knob token, which differs subtly between themes).
        knob_color = (
            QtGui.QColor(255, 255, 255) if self.isChecked() else QtGui.QColor(*tok["flat_knob"])
        )
        p.setBrush(QtGui.QBrush(knob_color))
        p.drawEllipse(thumb)

        p.end()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._TRACK_W, self._TRACK_H)
