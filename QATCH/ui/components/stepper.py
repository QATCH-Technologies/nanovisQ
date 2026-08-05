"""
QATCH.ui.components.stepper.py

Shared horizontal numbered-step indicator.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-05
"""

from PyQt5 import QtCore, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class Stepper(QtWidgets.QWidget):
    """Horizontal numbered-step indicator.

    Circles are connected by thin rule lines; the current step is filled solid,
    reached-but-not-current steps are outlined/tinted, and future steps are plain
    gray. Clicking a step the user has already reached jumps back to it.

    Attributes:
        stepClicked (:class:`~PyQt5.QtCore.pyqtSignal`): Signal emitted with the integer
            index of the step clicked by the user.
    """

    stepClicked = QtCore.pyqtSignal(int)

    def __init__(self, labels, parent: QtWidgets.QWidget | None = None, compact: bool = False):
        """Initializes the Stepper widget.

        Args:
            labels (list of str): A list of string labels for each step in the wizard.
            parent (:class:`~PyQt5.QtWidgets.QWidget`, optional): The parent widget.
            compact (bool, optional): If True, renders smaller circles and margins for contexts
                where the :class:`Stepper` sits in a thin toolbar-like card (e.g. AnalyzeUI's
                workflow bar) rather than as a wizard's own prominent header.
                Defaults to False.
        """
        super().__init__(parent)
        self._current = 0
        self._max_reached = 0
        self._circles = []
        self._captions = []
        self._lines = []
        self._compact = compact
        circle_size = 20 if compact else 26

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(*((2, 2, 2, 4) if compact else (4, 4, 4, 8)))
        outer.setSpacing(0)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(2 if compact else 4)

        col = 0
        for i, label in enumerate(labels):
            if i > 0:
                line = QtWidgets.QFrame()
                line.setFixedHeight(2)
                line.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                grid.addWidget(line, 0, col, QtCore.Qt.AlignmentFlag.AlignVCenter)
                grid.setColumnStretch(col, 1)
                self._lines.append(line)
                col += 1

            circle = QtWidgets.QToolButton()
            circle.setText(str(i + 1))
            circle.setFixedSize(circle_size, circle_size)
            circle.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            circle.clicked.connect(lambda _=False, idx=i: self._on_clicked(idx))
            grid.addWidget(circle, 0, col, QtCore.Qt.AlignmentFlag.AlignCenter)

            cap = QtWidgets.QLabel(label)
            cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(cap, 1, col, QtCore.Qt.AlignmentFlag.AlignHCenter)

            self._circles.append(circle)
            self._captions.append(cap)
            grid.setColumnStretch(col, 0)
            col += 1

        outer.addLayout(grid)
        self._restyle()
        ThemeManager.instance().themeChanged.connect(lambda _: self._restyle())

    def _on_clicked(self, idx) -> None:
        """Internal handler for when a step circle is clicked.

        Emits the stepClicked signal if the selected index has already been reached.

        Args:
            idx (int): The index of the clicked step.
        """
        if idx <= self._max_reached:
            self.stepClicked.emit(idx)

    def set_current(self, index: int) -> None:
        """Updates the current active step.

        Args:
            index (int): The index of the new current step.
        """
        self._current = index
        self._max_reached = max(self._max_reached, index)
        self._restyle()

    def reset(self) -> None:
        """Clears "reached" progress entirely.

        Used when the wizard itself resets, so old steps don't keep showing as
        done/clickable.
        """
        self._current = 0
        self._max_reached = 0
        self._restyle()

    def _restyle(self) -> None:
        """Forces an immediate, clean repaint of the widget states.

        Calling setStyleSheet() alone can leave a stale rendered pixmap behind for
        QSS-styled QToolButtons during rapid restyles (each step click restyles
        two circles at once). An explicit unpolish/polish + update() forces
        an immediate, clean repaint instead of a "ghost" of the previous state
        lingering under the new one.
        """
        for i, circle in enumerate(self._circles):
            if i == self._current:
                state = "current"
            elif i <= self._max_reached:
                state = "done"
            else:
                state = "future"
            circle.setStyleSheet(self._circle_qss(state, self._compact))
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
    def _circle_qss(state, compact: bool = False) -> str:
        """Generates the QSS string for a step circle based on its current state.

        Args:
            state (str): The state of the step, one of "current", "done", or "future".
            compact (bool, optional): Whether compact styling should be applied. Defaults to False.

        Returns:
            str: The generated stylesheet string for the circle.
        """
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
        radius = 10 if compact else 13
        font_size = 10 if compact else 12
        return (
            f"QToolButton {{ {body} border-radius: {radius}px; "
            f"font-weight: 700; font-size: {font_size}px; }}"
        )

    @staticmethod
    def _caption_qss(active: bool) -> str:
        """Generates the QSS string for a step caption.

        Weight is constant (700) regardless of state - varying it between active/inactive
        changes the text's rendered width slightly, which made the whole bar visibly
        jitter/resize on every step transition. Only colour differentiates
        the active step now.

        Args:
            active (bool): True if the caption corresponds to the active step.

        Returns:
            str: The generated stylesheet string for the caption.
        """
        tok = ThemeManager.instance().tokens()
        color = tok_css(tok["flat_accent"] if active else tok["flat_text_muted"])
        return (
            f"QLabel {{ color: {color}; font-size: 10px; font-weight: 700; "
            "background: transparent; }"
        )

    @staticmethod
    def _line_qss(done: bool) -> str:
        """Generates the QSS string for a connecting line.

        Args:
            done (bool): True if the line connects to a completed step.

        Returns:
            str: The generated stylesheet string for the line.
        """
        tok = ThemeManager.instance().tokens()
        color = tok_css(tok["flat_accent"] if done else tok["flat_border"])
        return f"QFrame {{ background: {color}; border: none; }}"
