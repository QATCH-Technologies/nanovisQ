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

    # Smaller footprint (34x19 track / 15px thumb instead of the default
    # 42x23 / 18px) for inline compact rows, e.g. a sub-row's
    # "Auto-calculate" affordance sitting next to its own field:
    small = QATCHToggle(parent, compact=True)
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

    # Smaller footprint used when `compact=True` is passed to __init__ -
    # shadows the class-level geometry above with instance attributes of
    # the same name, so paintEvent()/sizeHint() (which already read
    # self._TRACK_W/_TRACK_H/_THUMB_D dynamically) need no changes at all.
    _COMPACT_TRACK_W: int = 34
    _COMPACT_TRACK_H: int = 19
    _COMPACT_THUMB_D: int = 15

    def __init__(
        self, parent: Optional[QtWidgets.QWidget] = None, *, compact: bool = False
    ) -> None:
        """Initializes the toggle.

        Args:
            parent: The parent widget, if any.
            compact: If True, uses the smaller `_COMPACT_*` geometry instead
                of the default `_TRACK_W`/`_TRACK_H`/`_THUMB_D` - for inline
                sub-row affordances that need to read as visually secondary
                to the main pill toggles. Every existing call site keeps its
                current (non-compact) size since this defaults to False.
        """
        super().__init__(parent)
        if compact:
            self._TRACK_W = self._COMPACT_TRACK_W
            self._TRACK_H = self._COMPACT_TRACK_H
            self._THUMB_D = self._COMPACT_THUMB_D
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


class LabeledToggle(QtWidgets.QWidget):
    """A QATCHToggle paired with a text label in a horizontal row.

    Exposes the subset of the QCheckBox API used by the rest of the app
    (`isChecked`, `setChecked`, `setEnabled`, `setText`, `toggled`,
    `clicked`) so it can stand in for a checkbox without touching call
    sites.

    Attributes:
        toggled (pyqtSignal): A signal forwarded from the internal QATCHToggle
            that is emitted when the toggle state changes.
        clicked (pyqtSignal): A signal forwarded from the internal QATCHToggle
            that is emitted on every user click (same as QCheckBox.clicked).
    """

    def __init__(
        self,
        text: str = "",
        parent=None,
        *,
        label_left: bool = False,
        compact: bool = False,
    ) -> None:
        """Initializes the LabeledToggle.

        Args:
            text: The label text to display next to the toggle.
            parent: The parent widget, if any.
            label_left: If True, positions the label to the left of the toggle;
                otherwise, positions it to the right.
            compact: If True, uses a smaller `QATCHToggle(compact=True)` and a
                slightly smaller label font - for inline sub-row affordances
                (e.g. "Auto-calculate" beside its own field) that should read
                as visually secondary to the main pill toggles. Defaults to
                False so every existing call site is unaffected.
        """
        super().__init__(parent)
        self.toggle = QATCHToggle(self, compact=compact)
        self.label = QtWidgets.QLabel(text, self)
        self.label.setObjectName("CtrlToggleLabel")
        self.label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        if compact:
            self.label.setStyleSheet("QLabel#CtrlToggleLabel { font-size: 11px; }")

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6 if compact else 8)
        if label_left:
            lay.addWidget(self.label)
            lay.addWidget(self.toggle)
            lay.addStretch()
        else:
            lay.addWidget(self.toggle)
            lay.addWidget(self.label)
            lay.addStretch()
        self.toggled = self.toggle.toggled
        self.clicked = self.toggle.clicked

    def isChecked(self) -> bool:
        """Returns the current checked state of the toggle."""
        return self.toggle.isChecked()

    def setChecked(self, checked: bool) -> None:
        """Sets the checked state of the toggle.

        Args:
            checked: The boolean state to apply.
        """
        self.toggle.setChecked(checked)

    def setText(self, text: str) -> None:
        """Sets the text for the label.

        Args:
            text: The new label string.
        """
        self.label.setText(text)

    def text(self) -> str:
        """Returns the current label text."""
        return self.label.text()

    def setEnabled(self, enabled: bool) -> None:
        """Sets the enabled state for the widget and its children.

        Args:
            enabled: The boolean state to apply to the entire widget and children.
        """
        super().setEnabled(enabled)
        self.toggle.setEnabled(enabled)
        self.label.setEnabled(enabled)
