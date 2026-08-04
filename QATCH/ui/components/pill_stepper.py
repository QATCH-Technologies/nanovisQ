"""
QATCH.ui.components.pill_stepper.py

Small pill-shaped stepper meant to float over the top edge of a plot.

Same numbered-step/click-to-jump contract as
:class:`QATCH.ui.components.stepper.Stepper` (:sig:`stepClicked` signal, :meth:`set_current`,
:meth:`reset`), one row of small circles connected by thin lines, where only the current step expands
into a wider pill revealing its caption text; every other step stays a bare number.
Advancing animates the previously-current pill collapsing back to a circle and the new current
circle expanding into a pill.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from __future__ import annotations

from collections.abc import Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import make_qfont


class PillCellButton(QtWidgets.QToolButton):
    """A single round self-painted button.

    Self-painted rather than relying on Qt's stylesheet engine for `border-radius`.
    Displays either a caption/number (this row's own numbered pills) or a centered icon pixmap
    (a sibling "+"/"-" button placed *beside* this row), but never both.

    Note:
        Confirmed empirically: this widget tree is embedded via
        :class:`QtWidgets.QGraphicsProxyWidget` into a :class:`pyqtgraph.PlotWidget` or
        :class:`pyqtgraph.GraphicsView`. Because pyqtgraph's view does not enable antialiasing
        on its render hints by default, standard QSS `border-radius` circles appear jagged.
        This custom :meth:`paintEvent` explicitly sets its own antialiasing hint to ensure crisp
        rendering regardless of the hosting view.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initializes a new PillCellButton instance.

        Configures background translucency attributes and initializes default color
        and hover state variables.

        Args:
            parent (QtWidgets.QWidget | None, optional): Parent container widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self._fill = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._border = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._text_color = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._icon_pixmap: QtGui.QPixmap | None = None
        self._hovered = False

    def enterEvent(self, event: QtCore.QEvent) -> None:
        """Handles mouse hover enter events to trigger visual update.

        Args:
            event (QtCore.QEvent): Qt event object.
        """
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Handles mouse hover leave events to trigger visual update.

        Args:
            event (QtCore.QEvent): Qt event object.
        """
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def set_colors(
        self, fill: QtGui.QColor, border: QtGui.QColor, text_color: QtGui.QColor
    ) -> None:
        """Sets explicit rendering colors for fill, border, and text.

        Args:
            fill (QtGui.QColor): Color used for filling the inner rounded rectangle.
            border (QtGui.QColor): Color used for the cell outline pen.
            text_color (QtGui.QColor): Color used to paint text labels.
        """
        self._fill = fill
        self._border = border
        self._text_color = text_color
        self.update()

    def set_icon_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        """Sets a centered icon to paint in place of any caption/number text.

        Mutually exclusive with :meth:`setText`. :meth:`paintEvent` always prioritizes
        rendering the icon if one is provided.

        Args:
            pixmap (QtGui.QPixmap | None): Icon pixmap to center and draw, or ``None`` to
                revert to painting text.
        """
        self._icon_pixmap = pixmap
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paints a crisp, antialiased rounded pill/circle and its optional content.

        Args:
            event (QtGui.QPaintEvent): Qt paint event.
        """
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0
        p.setPen(QtGui.QPen(self._border, 1.0))
        p.setBrush(QtGui.QBrush(self._fill))
        p.drawRoundedRect(rect, radius, radius)
        if self._hovered and self.isEnabled():
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QColor(255, 255, 255, 40))
            p.drawRoundedRect(rect, radius, radius)
        if self._icon_pixmap is not None:
            target = QtCore.QRectF(self._icon_pixmap.rect())
            target.moveCenter(rect.center())
            p.drawPixmap(target.topLeft(), self._icon_pixmap)
        elif self.text():
            p.setPen(self._text_color)
            p.setFont(self.font())
            p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.text())
        p.end()


class PillStepper(QtWidgets.QWidget):
    """Horizontal numbered-step indicator where the current step expands to a pill.

    Circles connected by thin rule lines, using the same color language as :class:`Stepper`
    (current = filled solid, reached-but-not-current = outlined/tinted, future = plain gray).
    Clicking a step the user has already reached jumps back to it. The whole row sits on its
    own pill-shaped card background (stadium shape).

    Attributes:
        stepClicked (QtCore.pyqtSignal): Emitted with an integer index when a reachable step is clicked.
        sizeChanged (QtCore.pyqtSignal): Emitted whenever cell expansion/collapse changes widget geometry.
        _CIRCLE (int): Fixed cell height and width of a collapsed (non-current) step.
        _LINE_LEN (int): Length of horizontal connector lines between steps.
        _LINE_H (int): Thickness height of connector lines.
        _PILL_PAD_H (int): Horizontal padding on each side of an expanded cell's caption text.
        _ANIM_MS (int): Duration in milliseconds for expansion/collapse animations.
        _FONT_PX (int): Font size in pixels used for cell captions.
    """

    stepClicked = QtCore.pyqtSignal(int)
    sizeChanged = QtCore.pyqtSignal()

    _CIRCLE = 20
    _LINE_LEN = 12
    _LINE_H = 2
    _PILL_PAD_H = 12
    _ANIM_MS = 190
    _FONT_PX = 9

    def __init__(self, labels: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
        """Initializes a new PillStepper instance.

        Sets background transparency, constructs initial :class:`PillCellButton` cells and
        connector lines for given ``labels``, and sets up theme listener callbacks.

        Args:
            labels (Sequence[str]): Sequence of caption text strings for each step.
            parent (QtWidgets.QWidget | None, optional): Parent container widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

        self._labels = list(labels)
        self._current = 0
        self._max_reached = 0
        self._buttons: list[PillCellButton] = []
        self._lines: list[QtWidgets.QFrame] = []
        self._anims: list[QtCore.QVariantAnimation] = []

        self._font = make_qfont(pixel_size=self._FONT_PX, weight=QtGui.QFont.Bold)
        self._expanded_widths = [self._expanded_width_for(label) for label in self._labels]

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(0)

        for i in range(len(self._labels)):
            if i > 0:
                line = QtWidgets.QFrame()
                line.setFixedSize(self._LINE_LEN, self._LINE_H)
                outer.addWidget(line, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
                self._lines.append(line)

            btn = PillCellButton()
            btn.setFont(self._font)
            btn.setFixedHeight(self._CIRCLE)
            btn.setFixedWidth(self._CIRCLE)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            outer.addWidget(btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            self._buttons.append(btn)

            anim = QtCore.QVariantAnimation(self)
            anim.setDuration(self._ANIM_MS)
            anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
            anim.valueChanged.connect(lambda v, b=btn: b.setFixedWidth(int(v)))
            self._anims.append(anim)

        self._rewire_click_handlers()
        self._apply_current_instant()
        self._restyle()
        ThemeManager.instance().themeChanged.connect(lambda _: self._restyle())

    def _expanded_width_for(self, label: str) -> int:
        """Calculates expanded pill pixel width required to fit a text caption.

        Args:
            label (str): Text string to measure.

        Returns:
            int: Calculated total width in pixels.
        """
        fm = QtGui.QFontMetricsF(self._font)
        return int(fm.horizontalAdvance(label) + self._PILL_PAD_H * 2)

    def _rewire_click_handlers(self) -> None:
        """Reconnects click signals for buttons to carry their updated index positions."""
        for i, btn in enumerate(self._buttons):
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(lambda _checked=False, idx=i: self._on_clicked(idx))

    def _on_clicked(self, idx: int) -> None:
        """Emits :sig:`stepClicked` if the clicked cell index is reachable.

        Args:
            idx (int): Clicked button index.
        """
        if idx <= self._max_reached:
            self.stepClicked.emit(idx)

    def step_count(self) -> int:
        """Return total number of currently shown pills.

        Returns:
            int: Total count of active step pills.
        """
        return len(self._labels)

    def set_current(self, index: int) -> None:
        """Set the active step index and animate transition to expanded pill.

        Args:
            index (int): Target step index to make current.
        """
        prev = self._current
        self._current = index
        self._max_reached = max(self._max_reached, index)
        if prev != index:
            self._animate_to(prev, self._CIRCLE)
            self._animate_to(index, self._expanded_widths[index])
        self._restyle()

    def mark_reached(self, index: int) -> None:
        """Mark step index and all preceding steps as clickable without changing current step.

        Args:
            index (int): Target maximum step index to mark as reached.
        """
        if index > self._max_reached:
            self._max_reached = index
            self._restyle()

    def reset(self) -> None:
        """Clear reached progress and snap state instantly back to initial step 0."""
        self._current = 0
        self._max_reached = 0
        self._apply_current_instant()
        self._restyle()

    def add_step(self, label: str) -> None:
        """Inserts a new pill step immediately before the permanent last cell.

        Animates both the new pill and its connector line in unison.

        Args:
            label (str): Caption text for the newly added step.
        """
        pos = len(self._labels) - 1  # index the new cell will occupy
        insert_at = 2 * pos - 1

        line = QtWidgets.QFrame()
        line.setFixedHeight(self._LINE_H)
        line.setFixedWidth(0)
        self.layout().insertWidget(insert_at, line, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        btn = PillCellButton(self)
        btn.setFont(self._font)
        btn.setFixedHeight(self._CIRCLE)
        btn.setFixedWidth(0)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.layout().insertWidget(insert_at + 1, btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self._ANIM_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(lambda v, b=btn: b.setFixedWidth(int(v)))

        line_anim = QtCore.QVariantAnimation(self)
        line_anim.setDuration(self._ANIM_MS)
        line_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        line_anim.setStartValue(0)
        line_anim.setEndValue(self._LINE_LEN)
        line_anim.valueChanged.connect(lambda v, ln=line: ln.setFixedWidth(int(v)))

        self._labels.insert(pos, label)
        self._buttons.insert(pos, btn)
        self._lines.insert(pos - 1, line)
        self._expanded_widths.insert(pos, self._expanded_width_for(label))
        self._anims.insert(pos, anim)

        self._rewire_click_handlers()
        self._restyle()
        self._animate_to(pos, self._CIRCLE, start_width=0)
        line_anim.start()

    def remove_step(self) -> None:
        """Animates and removes the last non-bookend pill step and its line.

        State tracking updates instantly upon call, while visual removal occurs upon
        animation completion.
        """
        pos = len(self._labels) - 2
        if pos < 1:
            return
        btn = self._buttons[pos]
        line = self._lines[pos - 1]
        anim = self._anims[pos]

        del self._labels[pos]
        del self._buttons[pos]
        del self._lines[pos - 1]
        del self._expanded_widths[pos]
        del self._anims[pos]
        if self._current >= len(self._labels):
            self._current = len(self._labels) - 1
        if self._max_reached >= len(self._labels):
            self._max_reached = len(self._labels) - 1
        self._rewire_click_handlers()
        self._restyle()

        line_anim = QtCore.QVariantAnimation(self)
        line_anim.setDuration(self._ANIM_MS)
        line_anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        line_anim.setStartValue(line.width())
        line_anim.setEndValue(0)
        line_anim.valueChanged.connect(lambda v, ln=line: ln.setFixedWidth(int(v)))
        line_anim.start()

        def _finish(btn=btn, line=line) -> None:
            self.layout().removeWidget(btn)
            self.layout().removeWidget(line)
            btn.deleteLater()
            line.deleteLater()

        try:
            anim.finished.disconnect()
        except TypeError:
            pass
        anim.finished.connect(_finish)
        anim.stop()
        anim.setStartValue(btn.width())
        anim.setEndValue(0)
        anim.start()

    def _apply_current_instant(self) -> None:
        for i, btn in enumerate(self._buttons):
            self._anims[i].stop()
            if i == self._current:
                btn.setFixedWidth(self._expanded_widths[i])
                btn.setText(self._labels[i])
            else:
                btn.setFixedWidth(self._CIRCLE)
                btn.setText(str(i + 1))

    def _animate_to(self, index: int, target_width: int, start_width: int | None = None) -> None:
        """Applies current step width and text layout instantly without animation."""
        btn = self._buttons[index]
        anim = self._anims[index]
        anim.stop()
        if target_width > self._CIRCLE:
            btn.setText(self._labels[index])
        elif target_width > 0:
            btn.setText(str(index + 1))
        anim.setStartValue(start_width if start_width is not None else btn.width())
        anim.setEndValue(target_width)
        anim.start()

    def _restyle(self) -> None:
        """Re-applies themes and colors across cell buttons and connector lines."""
        for i, btn in enumerate(self._buttons):
            if i == self._current:
                state = "current"
            elif i <= self._max_reached:
                state = "done"
            else:
                state = "future"
            fill, border, text_color = self._cell_colors(state)
            btn.set_colors(fill, border, text_color)
        for i, line in enumerate(self._lines):
            line.setStyleSheet(self._line_qss(done=(i < self._max_reached)))
            line.style().unpolish(line)
            line.style().polish(line)
            line.update()
        self.update()

    @staticmethod
    def _cell_colors(state: str) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
        """Resolves color tuples for a given state string from theme tokens.

        Args:
            state (str): State identifier (``"current"``, ``"done"``, or ``"future"``).

        Returns:
            Tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]: Tuple containing ``(fill, border, text_color)``.
        """
        tok = ThemeManager.instance().tokens()
        if state == "current":
            return (
                QtGui.QColor(*tok["flat_accent"]),
                QtGui.QColor(*tok["flat_accent"]),
                QtGui.QColor(*tok["flat_on_accent"]),
            )
        if state == "done":
            return (
                QtGui.QColor(*tok["flat_accent_weak"]),
                QtGui.QColor(*tok["flat_accent"]),
                QtGui.QColor(*tok["flat_accent"]),
            )
        return (
            QtGui.QColor(*tok["flat_surface2"]),
            QtGui.QColor(*tok["flat_text_muted"]),
            QtGui.QColor(*tok["flat_border"]),
        )

    @staticmethod
    def _line_qss(done: bool) -> str:
        """Generates Qt stylesheet rules for step connector lines based on state.

        Args:
            done (bool): Whether the preceding step step boundary has been reached.

        Returns:
            str: QSS string setting line background colors.
        """
        tok = ThemeManager.instance().tokens()
        color = tok_css(tok["flat_accent"] if done else tok["flat_border"])
        return f"QFrame {{ background: {color}; border: none; }}"

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paints flat container surface under the pill stepper widget.

        Args:
            event (QtGui.QPaintEvent): Qt paint event.
        """
        tok = ThemeManager.instance().tokens()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_flat_surface(
            self,
            radius=self.height() / 2.0,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            painter=painter,
        )
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Emits :sig:`sizeChanged` when widget bounds change.

        Args:
            event (QtGui.QResizeEvent): Qt resize event.
        """
        super().resizeEvent(event)
        self.sizeChanged.emit()
