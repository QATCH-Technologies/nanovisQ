"""Shared horizontal numbered-step indicator.

Promoted from the Export wizard's private `_Stepper` (formerly in
`QATCH.ui.widgets.data_mode_export`) so other panels (e.g. AnalyzeUI) can
reuse the same circles-connected-by-lines step widget.
"""

from PyQt5 import QtCore, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class Stepper(QtWidgets.QWidget):
    """Horizontal numbered-step indicator.

    Circles connected by thin rule lines; the current step is filled solid,
    reached-but-not-current steps are outlined/tinted, future steps are plain
    gray. Clicking a step the user has already reached jumps back to it.
    """

    stepClicked = QtCore.pyqtSignal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self._current = 0
        self._max_reached = 0
        self._circles = []
        self._captions = []
        self._lines = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 8)
        outer.setSpacing(0)

        # Circles and connecting lines live in their OWN grid row (row 0),
        # with captions in a separate row (row 1) below. Keeping captions out
        # of row 0 means row 0's height is just the circle's height, so a
        # vertically-centered line lands on the circle's true center instead
        # of the midpoint of "circle + caption" as a combined column.
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(4)

        col = 0
        for i, label in enumerate(labels):
            if i > 0:
                line = QtWidgets.QFrame()
                line.setFixedHeight(2)
                line.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                grid.addWidget(line, 0, col, QtCore.Qt.AlignVCenter)
                grid.setColumnStretch(col, 1)
                self._lines.append(line)
                col += 1

            circle = QtWidgets.QToolButton()
            circle.setText(str(i + 1))
            circle.setFixedSize(26, 26)
            circle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            circle.clicked.connect(lambda _=False, idx=i: self._on_clicked(idx))
            grid.addWidget(circle, 0, col, QtCore.Qt.AlignCenter)

            cap = QtWidgets.QLabel(label)
            cap.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(cap, 1, col, QtCore.Qt.AlignHCenter)

            self._circles.append(circle)
            self._captions.append(cap)
            grid.setColumnStretch(col, 0)
            col += 1

        outer.addLayout(grid)
        self._restyle()
        ThemeManager.instance().themeChanged.connect(lambda _: self._restyle())

    def _on_clicked(self, idx):
        if idx <= self._max_reached:
            self.stepClicked.emit(idx)

    def set_current(self, index):
        self._current = index
        self._max_reached = max(self._max_reached, index)
        self._restyle()

    def reset(self):
        """Clear "reached" progress entirely - used when the wizard itself
        resets, so old steps don't keep showing as done/clickable."""
        self._current = 0
        self._max_reached = 0
        self._restyle()

    def _restyle(self):
        # setStyleSheet() alone can leave a stale rendered pixmap behind for
        # QSS-styled QToolButtons during rapid restyles (each step click
        # restyles two circles at once) - an explicit unpolish/polish +
        # update() forces an immediate, clean repaint instead of a "ghost"
        # of the previous state lingering under the new one.
        for i, circle in enumerate(self._circles):
            if i == self._current:
                state = "current"
            elif i <= self._max_reached:
                state = "done"
            else:
                state = "future"
            circle.setStyleSheet(self._circle_qss(state))
            circle.style().unpolish(circle)
            circle.style().polish(circle)
            circle.update()

            cap = self._captions[i]
            cap.setStyleSheet(self._caption_qss(active=(i == self._current)))
            cap.style().unpolish(cap)
            cap.style().polish(cap)
            cap.update()
        for i, line in enumerate(self._lines):
            line.setStyleSheet(self._line_qss(done=(i < self._max_reached)))
            line.style().unpolish(line)
            line.style().polish(line)
            line.update()

    @staticmethod
    def _circle_qss(state):
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
        return f"QToolButton {{ {body} border-radius: 13px; font-weight: 700; font-size: 12px; }}"

    @staticmethod
    def _caption_qss(active):
        # Weight is constant (700) regardless of state - varying it between
        # active/inactive changes the text's rendered width slightly, which
        # made the whole bar visibly jitter/resize on every step transition.
        # Only colour differentiates the active step now.
        tok = ThemeManager.instance().tokens()
        color = tok_css(tok["flat_accent"] if active else tok["flat_text_muted"])
        return (
            f"QLabel {{ color: {color}; font-size: 10px; font-weight: 700; "
            "background: transparent; }"
        )

    @staticmethod
    def _line_qss(done):
        tok = ThemeManager.instance().tokens()
        color = tok_css(tok["flat_accent"] if done else tok["flat_border"])
        return f"QFrame {{ background: {color}; border: none; }}"
