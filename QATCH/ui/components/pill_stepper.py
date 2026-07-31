"""Small pill-shaped stepper meant to float over the top edge of a plot.

Same numbered-step/click-to-jump contract as
`QATCH.ui.components.stepper.Stepper` (`stepClicked` signal, `set_current
(index)`, `reset()`), so it's a drop-in swap at a call site expecting that
interface - but visually different: one row of small circles connected by
thin lines, where only the *current* step expands into a wider pill
revealing its caption text; every other step stays a bare number. Advancing
animates the previously-current pill collapsing back to a circle and the new
current circle expanding into a pill.

Also supports growing/shrinking the row itself - see `add_step`/
`remove_step`/`step_count`/`mark_reached`. Adding/removing only ever happens
immediately before the row's permanent last cell (see `UIAnalyze`'s
Analyze-wizard usage, the only caller today, which owns a pair of external
"+"/"-" buttons *beside* this widget rather than inside it - see
`UIAnalyze._embed_stepper_overlay`), so these are deliberately simple
"grow/shrink the tail" operations, not general insert-anywhere-in-the-row
support.

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

from typing import List, Sequence, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css


class PillCellButton(QtWidgets.QToolButton):
    """A single round button, self-painted rather than relying on Qt's
    style-sheet engine for `border-radius` - shows either a caption/number
    (this row's own numbered pills) or a centered icon pixmap (a sibling
    "+"/"-" button placed *beside* this row - see
    `UIAnalyze._embed_stepper_overlay`), never both.

    Confirmed empirically: this widget tree is embedded via
    `QtWidgets.QGraphicsProxyWidget` into a `pg.PlotWidget`/`GraphicsView`
    (see `UIAnalyze._embed_stepper_overlay`), and pyqtgraph's `GraphicsView`
    doesn't enable antialiasing on its own render hints by default - QSS
    `border-radius` circles rendered through that path came out visibly
    jagged, while this widget's own `paintEvent` (which sets its own
    `Antialiasing` hint, the same technique already used by
    `QATCH.ui.widgets.saved_state_dot.SavedStateDot`) reads crisp regardless
    of the hosting view's own hints.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self._fill = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._border = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._text_color = QtGui.QColor(QtCore.Qt.GlobalColor.transparent)
        self._icon_pixmap: QtGui.QPixmap | None = None
        self._hovered = False

    def enterEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def set_colors(
        self, fill: QtGui.QColor, border: QtGui.QColor, text_color: QtGui.QColor
    ) -> None:
        self._fill = fill
        self._border = border
        self._text_color = text_color
        self.update()

    def set_icon_pixmap(self, pixmap: QtGui.QPixmap | None) -> None:
        """Sets a centered icon to paint in place of any caption/number
        text (mutually exclusive with `setText` - whichever was set most
        recently doesn't matter, `paintEvent` always prefers the icon if
        one's set). Pass `None` to go back to painting text.
        """
        self._icon_pixmap = pixmap
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2.0
        p.setPen(QtGui.QPen(self._border, 1.0))
        p.setBrush(QtGui.QBrush(self._fill))
        p.drawRoundedRect(rect, radius, radius)
        if self._hovered and self.isEnabled():
            # A translucent white wash rather than a lighter()/darker()
            # recompute of self._fill - reads as "raised" regardless of the
            # current theme or whichever fill color was pushed in via
            # set_colors (accent/surface/etc.), with no theme-token
            # dependency inside this otherwise externally-colored widget.
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
        self._buttons: List[PillCellButton] = []
        self._lines: List[QtWidgets.QFrame] = []
        self._anims: List[QtCore.QVariantAnimation] = []

        font = QtGui.QFont("Segoe UI")
        font.setPixelSize(self._FONT_PX)
        font.setWeight(QtGui.QFont.Bold)
        self._font = font
        self._expanded_widths = [self._expanded_width_for(label) for label in self._labels]

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(0)

        for i in range(len(self._labels)):
            if i > 0:
                line = QtWidgets.QFrame()
                line.setFixedSize(self._LINE_LEN, self._LINE_H)
                outer.addWidget(line, 0, QtCore.Qt.AlignVCenter)
                self._lines.append(line)

            btn = PillCellButton()
            btn.setFont(self._font)
            btn.setFixedHeight(self._CIRCLE)
            btn.setFixedWidth(self._CIRCLE)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            outer.addWidget(btn, 0, QtCore.Qt.AlignVCenter)
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
        fm = QtGui.QFontMetricsF(self._font)
        return int(fm.horizontalAdvance(label) + self._PILL_PAD_H * 2)

    def _rewire_click_handlers(self) -> None:
        """Reconnects every numbered cell's `clicked` signal to carry its
        *current* index - needed after `add_step`/`remove_step`, since
        every cell after the splice point shifts position and a stale
        closure would fire the wrong index.
        """
        for i, btn in enumerate(self._buttons):
            try:
                btn.clicked.disconnect()
            except TypeError:
                pass
            btn.clicked.connect(lambda _checked=False, idx=i: self._on_clicked(idx))

    def _on_clicked(self, idx: int) -> None:
        if idx <= self._max_reached:
            self.stepClicked.emit(idx)

    def step_count(self) -> int:
        """Total number of currently-shown pills (including the two
        permanent bookends) - lets an owner translate between its own
        fixed step-numbering scheme and this widget's actual (possibly
        shorter) pill count. See `UIAnalyze._pillstepper_index_for`.
        """
        return len(self._labels)

    def set_current(self, index: int) -> None:
        prev = self._current
        self._current = index
        self._max_reached = max(self._max_reached, index)
        if prev != index:
            self._animate_to(prev, self._CIRCLE)
            self._animate_to(index, self._expanded_widths[index])
        self._restyle()

    def mark_reached(self, index: int) -> None:
        """Marks `index` (and everything before it) as "reached" - i.e.
        clickable, the same side effect `set_current` already has -
        without changing which pill is currently displayed as *current*.

        Used when a step is revealed via `add_step` (a user's "+" click,
        or auto-fit finding more real channels than were previously shown
        - see `UIAnalyze._on_add_step_requested`/`_reveal_steps_for_poi_
        vals`): the user should be able to click straight into a newly-
        revealed step, not be blocked until they've clicked "Next" enough
        times to organically reach it the normal way.
        """
        if index > self._max_reached:
            self._max_reached = index
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

    def add_step(self, label: str) -> None:
        """Inserts a new pill and its own preceding connector line
        immediately before the row's permanent last cell, animating both
        in from width 0 together. See `remove_step` for the mirror-image
        operation.
        """
        pos = len(self._labels) - 1  # index the new cell will occupy
        # The flat layout is [btn0, line0, btn1, line1, ..., btn_{n-1}], so
        # the *existing* line that currently sits directly before the
        # permanent last button - the one that must end up between the
        # new button and that last button, unchanged - is at 2*pos - 1,
        # not 2*pos (that's the last button's own slot). Inserting at
        # 2*pos would land the new line+button *after* that existing
        # line instead of before it, leaving two consecutive connector
        # lines with no button between them.
        insert_at = 2 * pos - 1

        line = QtWidgets.QFrame()
        line.setFixedHeight(self._LINE_H)
        line.setFixedWidth(0)
        self.layout().insertWidget(insert_at, line, 0, QtCore.Qt.AlignVCenter)

        btn = PillCellButton(self)
        btn.setFont(self._font)
        btn.setFixedHeight(self._CIRCLE)
        btn.setFixedWidth(0)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.layout().insertWidget(insert_at + 1, btn, 0, QtCore.Qt.AlignVCenter)

        anim = QtCore.QVariantAnimation(self)
        anim.setDuration(self._ANIM_MS)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.valueChanged.connect(lambda v, b=btn: b.setFixedWidth(int(v)))

        # A second, parallel animation grows the connector line's width in
        # lockstep with the button's own expansion, rather than the line
        # popping in at full length instantly - see remove_step for the
        # mirror-image shrink.
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
        """Animates the last non-bookend pill's width - and its preceding
        connector line's width - down to 0 together, then actually
        detaches both from the layout. A no-op if only the two permanent
        bookend cells remain.

        Splices `self._labels`/`self._buttons`/etc. *immediately* (not
        deferred to the animation's `finished` callback) specifically so
        that two `remove_step()` calls issued back-to-back - e.g. a fast
        double-click on an owner's "-" button, before the first 190ms
        animation has finished - each compute a fresh `pos` and target two
        different cells, rather than both racing to animate (and only ever
        finishing) the same one. Only the *visual* detach/deleteLater is
        deferred; the tracked state (and this widget's own `step_count()`,
        which a caller like `UIAnalyze` reads to stay in sync) is correct
        the instant this method returns.
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
        btn = self._buttons[index]
        anim = self._anims[index]
        anim.stop()
        if target_width > self._CIRCLE:
            # Expanding into a pill - swap to the caption immediately so
            # it's legible as soon as there's room, rather than crossfading.
            btn.setText(self._labels[index])
        elif target_width > 0:
            # Collapsing back to a circle - swap to the plain number right
            # away so a long caption doesn't clip mid-word as it narrows.
            btn.setText(str(index + 1))
        anim.setStartValue(start_width if start_width is not None else btn.width())
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
            fill, border, text_color = self._cell_colors(state)
            btn.set_colors(fill, border, text_color)
        for i, line in enumerate(self._lines):
            line.setStyleSheet(self._line_qss(done=(i < self._max_reached)))
            line.style().unpolish(line)
            line.style().polish(line)
            line.update()
        self.update()

    @staticmethod
    def _cell_colors(state: str) -> Tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
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
