"""Small pill-shaped stepper meant to float over the top edge of a plot.

Same numbered-step/click-to-jump contract as
`QATCH.ui.components.stepper.Stepper` (`stepClicked` signal, `set_current
(index)`, `reset()`), so it's a drop-in swap at a call site expecting that
interface - but visually different: one row of small circles connected by
thin lines, where only the *current* step expands into a wider pill
revealing its caption text; every other step stays a bare number. Advancing
animates the previously-current pill collapsing back to a circle and the new
current circle expanding into a pill.

`Stepper` itself is left alone (still used by the Export wizard as a full
two-row header) since the two layouts are different enough - dynamic
per-cell width vs. a fixed grid - that folding this into `Stepper` would
mean two unrelated layout algorithms living in one class.

Meant to be embedded as a `QtWidgets.QGraphicsProxyWidget` floating over a
pyqtgraph plot (see `UIAnalyze._show_no_run_overlay` for the same embedding
technique), where a full-size `Stepper` would be too tall/wide to float
compactly.
"""

from __future__ import annotations

from typing import List, Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class PillStepper(QtWidgets.QWidget):
    """Horizontal numbered-step indicator; the current step expands to a pill.

    Circles connected by thin rule lines, same color language as `Stepper`
    (current = filled solid, reached-but-not-current = outlined/tinted,
    future = plain gray). Clicking a step the user has already reached jumps
    back to it. The whole row sits on its own pill-shaped card background
    (radius = half the widget's height, so it's a stadium shape regardless
    of how wide the currently-expanded cell makes the row).
    """

    stepClicked = QtCore.pyqtSignal(int)
    # Emitted whenever the widget's own size changes (a step expanding or
    # collapsing shifts the overall width) - lets an embedding overlay (see
    # UIAnalyze._embed_stepper_overlay) re-center itself on every such
    # change, regardless of what triggered it (click, Back/Next, reset).
    sizeChanged = QtCore.pyqtSignal()

    _CIRCLE = 20  # fixed height of every cell; also the width of a collapsed (non-current) cell
    _LINE_LEN = 12
    _LINE_H = 2
    _PILL_PAD_H = 12  # horizontal padding either side of an expanded cell's caption text
    _ANIM_MS = 190
    _FONT_PX = 9

    def __init__(self, labels: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        # Same reasoning as ControlsWidget/AnalyzeActionBar: this is meant to
        # be embedded on a transparent card (here, a QGraphicsProxyWidget
        # floating over a plot), so the widget's own default opaque
        # background must be suppressed or the corners outside the pill
        # shape would show as a solid rectangle.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._labels = list(labels)
        self._current = 0
        self._max_reached = 0
        self._buttons: List[QtWidgets.QToolButton] = []
        self._lines: List[QtWidgets.QFrame] = []
        self._anims: List[QtCore.QVariantAnimation] = []

        font = QtGui.QFont("Segoe UI")
        font.setPixelSize(self._FONT_PX)
        font.setWeight(QtGui.QFont.Bold)
        self._font = font
        fm = QtGui.QFontMetricsF(font)
        self._expanded_widths = [
            int(fm.horizontalAdvance(label) + self._PILL_PAD_H * 2) for label in self._labels
        ]

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(0)

        for i in range(len(self._labels)):
            if i > 0:
                line = QtWidgets.QFrame()
                line.setFixedSize(self._LINE_LEN, self._LINE_H)
                outer.addWidget(line, 0, QtCore.Qt.AlignVCenter)
                self._lines.append(line)

            btn = QtWidgets.QToolButton()
            btn.setFont(self._font)
            btn.setFixedHeight(self._CIRCLE)
            btn.setFixedWidth(self._CIRCLE)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, idx=i: self._on_clicked(idx))
            outer.addWidget(btn, 0, QtCore.Qt.AlignVCenter)
            self._buttons.append(btn)

            anim = QtCore.QVariantAnimation(self)
            anim.setDuration(self._ANIM_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
            anim.valueChanged.connect(lambda v, b=btn: b.setFixedWidth(int(v)))
            self._anims.append(anim)

        self._apply_current_instant()
        self._restyle()
        ThemeManager.instance().themeChanged.connect(lambda _: self._restyle())

    def _on_clicked(self, idx: int) -> None:
        if idx <= self._max_reached:
            self.stepClicked.emit(idx)

    def set_current(self, index: int) -> None:
        prev = self._current
        self._current = index
        self._max_reached = max(self._max_reached, index)
        if prev != index:
            self._animate_to(prev, self._CIRCLE)
            self._animate_to(index, self._expanded_widths[index])
        self._restyle()

    def reset(self) -> None:
        """Clear "reached" progress entirely and snap (no animation) back to
        step 0 expanded - used when the wizard itself resets, so old steps
        don't keep showing as done/clickable and a stale pill doesn't linger
        expanded from whatever step was current before."""
        self._current = 0
        self._max_reached = 0
        self._apply_current_instant()
        self._restyle()

    def _apply_current_instant(self) -> None:
        for i, btn in enumerate(self._buttons):
            self._anims[i].stop()
            if i == self._current:
                btn.setFixedWidth(self._expanded_widths[i])
                btn.setText(self._labels[i])
            else:
                btn.setFixedWidth(self._CIRCLE)
                btn.setText(str(i + 1))

    def _animate_to(self, index: int, target_width: int) -> None:
        btn = self._buttons[index]
        anim = self._anims[index]
        anim.stop()
        if target_width > self._CIRCLE:
            # Expanding into a pill - swap to the caption immediately so
            # it's legible as soon as there's room, rather than crossfading.
            btn.setText(self._labels[index])
        else:
            # Collapsing back to a circle - swap to the plain number right
            # away so a long caption doesn't clip mid-word as it narrows.
            btn.setText(str(index + 1))
        anim.setStartValue(btn.width())
        anim.setEndValue(target_width)
        anim.start()

    def _restyle(self) -> None:
        for i, btn in enumerate(self._buttons):
            if i == self._current:
                state = "current"
            elif i <= self._max_reached:
                state = "done"
            else:
                state = "future"
            btn.setStyleSheet(self._cell_qss(state))
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
        for i, line in enumerate(self._lines):
            line.setStyleSheet(self._line_qss(done=(i < self._max_reached)))
            line.style().unpolish(line)
            line.style().polish(line)
            line.update()
        self.update()

    def _cell_qss(self, state: str) -> str:
        tok = ThemeManager.instance().tokens()
        if state == "current":
            body = (
                f"background: {tok_css(tok['flat_accent'])}; "
                f"color: {tok_css(tok['flat_on_accent'])}; border: none;"
            )
        elif state == "done":
            body = (
                f"background: {tok_css(tok['flat_accent_weak'])}; "
                f"color: {tok_css(tok['flat_accent'])}; "
                f"border: 1px solid {tok_css(tok['flat_accent'])};"
            )
        else:
            body = (
                f"background: {tok_css(tok['flat_surface2'])}; "
                f"color: {tok_css(tok['flat_text_muted'])}; "
                f"border: 1px solid {tok_css(tok['flat_border'])};"
            )
        radius = self._CIRCLE // 2
        return (
            f"QToolButton {{ {body} border-radius: {radius}px; "
            f"font-weight: 700; font-size: {self._FONT_PX}px; }}"
        )

    @staticmethod
    def _line_qss(done: bool) -> str:
        tok = ThemeManager.instance().tokens()
        color = tok_css(tok["flat_accent"] if done else tok["flat_border"])
        return f"QFrame {{ background: {color}; border: none; }}"

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # flat_surface/flat_border (not the translucent "glass" surface/
        # surface_border ControlsWidget/AnalyzeActionBar use) - those are
        # ~160-170/255 alpha, tuned to sit over the app's own opaque chrome.
        # This pill floats directly over the plot's curves/grid instead, so
        # it needs the fully-opaque flat_* tokens its own circle QSS already
        # uses, or the plot shows through and it reads as washed-out rather
        # than a solid card.
        paint_flat_surface(
            self,
            radius=self.height() / 2.0,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            painter=painter,
        )
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.sizeChanged.emit()
