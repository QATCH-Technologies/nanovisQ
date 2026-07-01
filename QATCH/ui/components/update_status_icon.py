from enum import IntEnum
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager


class UpdateStatusIcon(QtWidgets.QToolButton):
    """A themed, animated icon button that reflects update status.

    Renders an SVG icon tinted to green / yellow / red / gray based on the
    current :class:`State`. Non-static states pulse via a QVariantAnimation.
    Clicking the button emits :pyqtSignal:`update_requested` when the state
    is actionable (not UP_TO_DATE or CHECKING).

    Args:
        icon_path: Filesystem path to the SVG icon to display.
        size: Square pixel dimension of the button (default 20).
        parent: Optional parent widget.
    """

    update_requested = QtCore.pyqtSignal()

    class State(IntEnum):
        UNKNOWN = 0
        CHECKING = 1
        UP_TO_DATE = 2
        OPTIONAL = 3
        MANDATORY = 4

    _COLORS = {
        "light": {
            State.UNKNOWN: QtGui.QColor(150, 165, 180),
            State.CHECKING: QtGui.QColor(150, 165, 180),
            State.UP_TO_DATE: QtGui.QColor(60, 190, 120),
            State.OPTIONAL: QtGui.QColor(240, 170, 50),
            State.MANDATORY: QtGui.QColor(228, 70, 70),
        },
        "dark": {
            State.UNKNOWN: QtGui.QColor(170, 180, 195),
            State.CHECKING: QtGui.QColor(170, 180, 195),
            State.UP_TO_DATE: QtGui.QColor(80, 210, 140),
            State.OPTIONAL: QtGui.QColor(255, 185, 60),
            State.MANDATORY: QtGui.QColor(240, 90, 90),
        },
    }

    _TOOLTIPS = {
        State.UNKNOWN: "Update status unknown — click to check",
        State.CHECKING: "Checking for updates...",
        State.UP_TO_DATE: "Up to date",
        State.OPTIONAL: "Update available — click to update",
        State.MANDATORY: "Required update — click to install",
    }

    _PULSING = (State.CHECKING, State.OPTIONAL, State.MANDATORY)

    def __init__(
        self,
        icon_path: str,
        size: int = 20,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._icon_path = icon_path
        self._size = size
        self._state = self.State.UNKNOWN
        self._detail = ""
        self._glow: float = 1.0

        self.setFixedSize(size, size)
        self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.setIconSize(QtCore.QSize(size, size))
        self.setStyleSheet("QToolButton { background: transparent; border: none; }")
        self.setAutoRaise(True)

        self._base_pixmap = QtGui.QPixmap(icon_path)

        self._pulse = QtCore.QVariantAnimation(self)
        self._pulse.setStartValue(0.25)
        self._pulse.setEndValue(1.0)
        self._pulse.setDuration(900)
        self._pulse.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._on_pulse)

        ThemeManager.instance().themeChanged.connect(lambda _: self._refresh_icon())
        self.clicked.connect(self._on_clicked)
        self._update_tooltip()
        self._refresh_icon()

    def state(self) -> "UpdateStatusIcon.State":
        """Returns the current update state."""
        return self._state

    def setState(self, state: "UpdateStatusIcon.State", detail: str = "") -> None:
        """Set the update state and optional detail text shown in the tooltip.

        Args:
            state: New :class:`State` value.
            detail: Additional tooltip text, e.g. version strings or device names.
        """
        old_pulsing = self._state in self._PULSING
        self._state = state
        self._detail = detail
        self._update_tooltip()

        new_pulsing = state in self._PULSING
        if new_pulsing:
            if not old_pulsing:
                self._glow = 0.25
                self._pulse.stop()
                self._pulse.start()
        else:
            self._pulse.stop()
            self._glow = 1.0

        self._refresh_icon()

    def _on_pulse(self, v: float) -> None:
        self._glow = v
        self._refresh_icon()

    def _update_tooltip(self) -> None:
        tip = self._TOOLTIPS.get(self._state, "")
        if self._detail:
            tip = f"{tip}\n{self._detail}"
        self.setToolTip(tip)

    def _tinted_icon(self, color: QtGui.QColor, alpha_f: float = 1.0) -> QtGui.QIcon:
        if self._base_pixmap.isNull():
            return QtGui.QIcon()
        base = self._base_pixmap.scaled(
            self._size,
            self._size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        dst = QtGui.QPixmap(base.size())
        dst.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(dst)
        p.setOpacity(alpha_f)
        p.drawPixmap(0, 0, base)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def _refresh_icon(self) -> None:
        mode = ThemeManager.instance().mode().value
        colors = self._COLORS.get(mode, self._COLORS["light"])
        color = colors.get(self._state, colors[self.State.UNKNOWN])
        if self._state in self._PULSING:
            alpha = max(0.35, self._glow)
        elif self._state == self.State.UNKNOWN:
            alpha = 0.5
        else:
            alpha = 1.0
        self.setIcon(self._tinted_icon(color, alpha))

    def _on_clicked(self) -> None:
        if self._state not in (self.State.UP_TO_DATE, self.State.CHECKING):
            self.update_requested.emit()
