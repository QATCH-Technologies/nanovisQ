"""
glass_option_card.py

A selectable "card" control matching the app's flat control system (see
QATCH.ui.components.flat_paint): a flat panel with a bold title and a
smaller description line beneath it. Cards are grouped via
`GlassOptionCardGroup` for exclusive (radio-style) selection - used wherever
a wireframe shows a labelled option instead of a plain segmented control
(e.g. Import's "When a run already exists" policy, Export's Destination
USB/Folder choice).

Usage
-----
    group = GlassOptionCardGroup()
    c1 = GlassOptionCard("Merge", "Add new, keep both")
    c2 = GlassOptionCard("Replace", "Overwrite existing")
    c3 = GlassOptionCard("Skip", "Leave existing")
    group.addCard(c1, 1)
    group.addCard(c2, 2)
    group.addCard(c3, 3)
    c1.setChecked(True)
    group.toggled.connect(lambda card, checked: ...)
    group.checkedId()       # -> 1
    group.checkedButton()   # -> c1
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.fonts import FONT_SANS, FONT_SANS_SEMIBOLD
from QATCH.ui.styles.theme_manager import ThemeManager


class _RadioDot(QtWidgets.QWidget):
    """A small custom-painted radio indicator: a ring, plus a centered
    solid dot when checked.

    A plain QFrame + QSS can express a ring (border + transparent fill) but
    not a ring with an independently-sized concentric fill, so this is hand
    -painted instead - matching this repo's existing convention of small
    custom-painted indicator glyphs (see QATCH.ui.widgets.saved_state_dot).
    """

    _SIZE = 16
    _DOT_D = 7
    _RING_WIDTH = 1.5

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self._checked = False

    def set_checked(self, checked: bool) -> None:
        if checked != self._checked:
            self._checked = checked
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tok = ThemeManager.instance().tokens()
        ring_color = QtGui.QColor(
            *(tok["flat_accent"] if self._checked else tok["flat_border_strong"])
        )

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setPen(QtGui.QPen(ring_color, self._RING_WIDTH))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        inset = self._RING_WIDTH / 2.0
        p.drawEllipse(QtCore.QRectF(inset, inset, self._SIZE - 2 * inset, self._SIZE - 2 * inset))

        if self._checked:
            dot_color = QtGui.QColor(*tok["flat_accent"])
            p.setPen(QtCore.Qt.NoPen)
            p.setBrush(QtGui.QBrush(dot_color))
            off = (self._SIZE - self._DOT_D) / 2.0
            p.drawEllipse(QtCore.QRectF(off, off, self._DOT_D, self._DOT_D))

        p.end()


class GlassOptionCard(QtWidgets.QFrame):
    """A clickable, checkable card with a title and a one-line description."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, title, description="", parent=None, *, show_radio=False):
        super().__init__(parent)
        self.setObjectName("glassOptionCard")
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._checked = False
        self._title = title
        self._show_radio = show_radio
        self._opacity_effect = None

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(15, 14, 15, 14)
        outer.setSpacing(9)

        if show_radio:
            self._radio_dot = _RadioDot(self)
            outer.addWidget(self._radio_dot, 0, QtCore.Qt.AlignTop)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(8)
        self._title_lbl = QtWidgets.QLabel(title)
        self._title_lbl.setObjectName("optionCardTitle")
        text_col.addWidget(self._title_lbl)
        if description:
            self._desc_lbl = QtWidgets.QLabel(description)
            self._desc_lbl.setObjectName("optionCardDesc")
            self._desc_lbl.setWordWrap(True)
            text_col.addWidget(self._desc_lbl)
        else:
            self._desc_lbl = None
        outer.addLayout(text_col, 1)

        self._apply_qss()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        """Re-style from the active palette when the theme flips."""
        self._apply_qss()
        if self._show_radio:
            self._radio_dot.update()

    # ------------------------------------------------------------------
    def text(self):
        """Return the card's title (parity with QAbstractButton.text())."""
        return self._title

    def setText(self, title):
        """Relabel the card's title in place (e.g. Merge -> Append for CSV)."""
        self._title = title
        self._title_lbl.setText(title)

    def setDescription(self, description):
        if self._desc_lbl is not None:
            self._desc_lbl.setText(description)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked: bool):
        if checked == self._checked:
            return
        self._checked = checked
        if self._show_radio:
            self._radio_dot.set_checked(checked)
        self._apply_qss()

    def setCardEnabled(self, enabled: bool) -> None:
        """Enables/disables the whole card, dimming it to 0.5 opacity when
        disabled.

        A `QGraphicsOpacityEffect` is safe here (unlike on the
        continuously hover-animated glass buttons/toggle): this widget only
        repaints on check-state or theme change, never on a timer or a
        hover-driven cycle, so it doesn't hit the offscreen-pixmap-caching
        ghosting failure mode documented for those other widgets (see
        ui_controls.py's `_PerspectiveAnimator` docstring). Do not "fix"
        this back to a manual per-paint opacity multiply without re-reading
        that reasoning.
        """
        self.setEnabled(enabled)
        self.setCursor(
            QtCore.Qt.CursorShape.ArrowCursor
            if not enabled
            else QtCore.Qt.CursorShape.PointingHandCursor
        )
        if not enabled:
            if self._opacity_effect is None:
                self._opacity_effect = QtWidgets.QGraphicsOpacityEffect(self)
            self._opacity_effect.setOpacity(0.5)
            self.setGraphicsEffect(self._opacity_effect)
        else:
            self.setGraphicsEffect(None)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    @staticmethod
    def _rgba(rgba) -> str:
        """Format a token (r, g, b, a) tuple as a CSS rgba() string."""
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_qss(self):
        tok = ThemeManager.instance().tokens()
        rgba = self._rgba
        if self._checked:
            frame_qss = f"""
                QFrame#glassOptionCard {{
                    background: {rgba(tok["flat_accent_weak"])};
                    border: 1.5px solid {rgba(tok["flat_accent"])};
                    border-radius: 10px;
                }}
            """
        else:
            frame_qss = f"""
                QFrame#glassOptionCard {{
                    background: {rgba(tok["flat_surface"])};
                    border: 1px solid {rgba(tok["flat_border"])};
                    border-radius: 10px;
                }}
                QFrame#glassOptionCard:hover {{
                    background: {rgba(tok["flat_surface2"])};
                    border: 1px solid {rgba(tok["flat_border_strong"])};
                }}
            """
        self.setStyleSheet(frame_qss)
        self._title_lbl.setStyleSheet(
            f"QLabel#optionCardTitle {{ color: {rgba(tok['flat_text'])}; "
            f"font-family: '{FONT_SANS_SEMIBOLD}'; font-size: 13px; background: transparent; }}"
        )
        if self._desc_lbl is not None:
            self._desc_lbl.setStyleSheet(
                f"QLabel#optionCardDesc {{ color: {rgba(tok['flat_text_muted'])}; "
                f"font-family: '{FONT_SANS}'; font-size: 11.5px; background: transparent; }}"
            )


class GlassOptionCardGroup(QtCore.QObject):
    """Exclusive-selection manager for a set of GlassOptionCards.

    Mirrors the handful of QButtonGroup methods already relied on elsewhere
    (checkedId, checkedButton, buttonToggled-style signal) so call sites that
    previously read a QButtonGroup don't need to change shape.
    """

    toggled = QtCore.pyqtSignal(object, bool)  # (card, checked) - fires for the newly checked card

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards = []  # list of (card, id)

    def addCard(self, card: GlassOptionCard, card_id):
        self._cards.append((card, card_id))
        card.clicked.connect(lambda c=card: self._on_card_clicked(c))

    def _on_card_clicked(self, card: GlassOptionCard):
        if card.isChecked():
            return
        for c, _ in self._cards:
            c.setChecked(c is card)
        self.toggled.emit(card, True)

    def checkedButton(self):
        for c, _ in self._cards:
            if c.isChecked():
                return c
        return None

    def checkedId(self):
        for c, cid in self._cards:
            if c.isChecked():
                return cid
        return -1

    def setCheckedId(self, card_id):
        """Programmatically select a card by id, emitting `toggled` for parity
        with the click path (QButtonGroup's exclusivity does this natively;
        our cards don't, so callers that set state in code need this)."""
        target = None
        for c, cid in self._cards:
            checked = cid == card_id
            c.setChecked(checked)
            if checked:
                target = c
        if target is not None:
            self.toggled.emit(target, True)
