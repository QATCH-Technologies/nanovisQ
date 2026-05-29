"""
glass_toggle.py

Animated toggle switch that matches the glass-morphism aesthetic of
GlassLineEdit and GlassPushButton.

Track colour interpolates from a soft grey (off) to the primary blue
accent (on) — the same blue used by GlassPushButton's "primary" variant
(rgba 45, 165, 250).  A white thumb slides left / right with an
OutCubic easing over 150 ms.

Usage
-----
    toggle = GlassToggle(parent)
    toggle.setChecked(True)           # set initial state (no animation)
    toggle.toggled.connect(handler)   # bool signal, same as QCheckBox

    # For silent initialisation (avoid triggering the handler):
    toggle.setChecked(value)          # connect signal AFTER this call
    toggle.toggled.connect(handler)
"""

from __future__ import annotations
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets


class GlassToggle(QtWidgets.QAbstractButton):
    """Pill-shaped animated toggle switch.

    Inherits ``toggled(bool)`` from ``QAbstractButton`` — a drop-in
    replacement for ``QCheckBox`` wherever only the checked state matters.

    Attributes:
        _anim_t (float): Thumb position 0.0 (off / left) → 1.0 (on / right).
    """

    # ── Geometry ──────────────────────────────────────────────────────
    _TRACK_W: int = 40
    _TRACK_H: int = 22
    _THUMB_D: int = 16  # diameter; margin = (_TRACK_H - _THUMB_D) / 2 = 3 px

    # ── Colours (plain tuples — QColor constructed at paint time) ─────
    # Track: off = muted glass grey, on = primary-button blue
    _TRACK_OFF = (180, 185, 195, 140)
    _TRACK_ON = (45, 165, 250, 200)

    # Track border: off and on states
    _BORDER_OFF = (160, 165, 175, 80)
    _BORDER_ON = (30, 140, 220, 120)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(self._TRACK_W, self._TRACK_H)
        self.setCursor(QtCore.Qt.PointingHandCursor)

        self._anim_t: float = 0.0

        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim_step)

        # toggled fires after the internal checked state flips, so
        # _anim_t correctly approaches the new target.
        self.toggled.connect(self._start_anim)

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
        # immediately — avoids a jarring mid-paint initial frame.
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
        r = h / 2.0  # track corner radius — full pill

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # ── Track fill (interpolated colour) ─────────────────────────
        lo, hi = self._TRACK_OFF, self._TRACK_ON
        track_color = QtGui.QColor(
            self._lerp(lo[0], hi[0], t),
            self._lerp(lo[1], hi[1], t),
            self._lerp(lo[2], hi[2], t),
            self._lerp(lo[3], hi[3], t),
        )
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(track_color))
        p.drawRoundedRect(QtCore.QRectF(0.0, 0.0, w, h), r, r)

        # ── Track border ─────────────────────────────────────────────
        lo_b, hi_b = self._BORDER_OFF, self._BORDER_ON
        border_color = QtGui.QColor(
            self._lerp(lo_b[0], hi_b[0], t),
            self._lerp(lo_b[1], hi_b[1], t),
            self._lerp(lo_b[2], hi_b[2], t),
            self._lerp(lo_b[3], hi_b[3], t),
        )
        p.setPen(QtGui.QPen(border_color, 1.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(QtCore.QRectF(0.5, 0.5, w - 1.0, h - 1.0), r, r)

        # ── Thumb ────────────────────────────────────────────────────
        margin = (h - self._THUMB_D) / 2.0
        x_left = margin
        x_right = w - margin - self._THUMB_D
        thumb_x = x_left + (x_right - x_left) * t
        thumb = QtCore.QRectF(thumb_x, margin, self._THUMB_D, self._THUMB_D)

        # Soft drop shadow (offset 1 px down, low alpha)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 35)))
        p.drawEllipse(thumb.adjusted(0.0, 1.0, 0.0, 1.0))

        # Thumb face
        p.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 245)))
        p.drawEllipse(thumb)

        p.end()

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._TRACK_W, self._TRACK_H)
