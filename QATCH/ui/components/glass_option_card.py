"""
glass_option_card.py

A selectable "card" control: a frosted glass panel with a bold title and a
smaller description line beneath it. Cards are grouped via
``GlassOptionCardGroup`` for exclusive (radio-style) selection - used wherever
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

from PyQt5 import QtCore, QtWidgets


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

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(10)

        if show_radio:
            self._radio_dot = QtWidgets.QFrame()
            self._radio_dot.setObjectName("optionCardRadio")
            self._radio_dot.setFixedSize(16, 16)
            outer.addWidget(self._radio_dot, 0, QtCore.Qt.AlignTop)

        text_col = QtWidgets.QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
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
        self._apply_qss()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    # ------------------------------------------------------------------
    def _apply_qss(self):
        if self._checked:
            frame_qss = """
                QFrame#glassOptionCard {
                    background: rgba(10, 163, 230, 35);
                    border: 1.5px solid rgba(0, 118, 174, 200);
                    border-radius: 10px;
                }
            """
            title_color = "rgba(0, 90, 135, 245)"
            radio_qss = """
                QFrame#optionCardRadio {
                    background: rgba(0, 118, 174, 235);
                    border: 1px solid rgba(0, 118, 174, 235);
                    border-radius: 8px;
                }
            """
        else:
            frame_qss = """
                QFrame#glassOptionCard {
                    background: rgba(255, 255, 255, 130);
                    border: 1px solid rgba(212, 219, 228, 190);
                    border-radius: 10px;
                }
                QFrame#glassOptionCard:hover {
                    background: rgba(255, 255, 255, 180);
                    border: 1px solid rgba(160, 175, 190, 210);
                }
            """
            title_color = "rgba(30, 42, 56, 230)"
            radio_qss = """
                QFrame#optionCardRadio {
                    background: rgba(255, 255, 255, 200);
                    border: 1px solid rgba(150, 160, 175, 180);
                    border-radius: 8px;
                }
            """
        self.setStyleSheet(frame_qss)
        self._title_lbl.setStyleSheet(
            f"QLabel#optionCardTitle {{ color: {title_color}; font-size: 13px; "
            "font-weight: 700; background: transparent; }"
        )
        if self._desc_lbl is not None:
            self._desc_lbl.setStyleSheet(
                "QLabel#optionCardDesc { color: rgba(50, 62, 78, 200); font-size: 11px; "
                "background: transparent; }"
            )
        if self._show_radio:
            self._radio_dot.setStyleSheet(radio_qss)


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
