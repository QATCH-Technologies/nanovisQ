"""
QATCH.components.ui.qatch_option_card.py

Selectable option-card controls for the nanovisQ application.

Provides styled, selectable card widgets that follow the application's
flat control system, as defined by
`QATCH.ui.components.flat_paint`. Option cards are designed for presenting
labelled choices that require more context than a standard button or
segmented control, using a bold title with an optional smaller descriptive
line beneath it.

The module provides two primary components:

    :class:`QATCHOptionCard`:
        A clickable, checkable card containing a title and optional
        description. Cards can optionally display a custom-painted radio
        indicator and expose a button-like `clicked`, `text()`,
        `setText()`, `isChecked()`, and `setChecked()` interface for
        compatibility with existing application call sites.

    :class:`QATCHOptionCardGroup`:
        An exclusive-selection manager for `QATCHOptionCard` instances.
        It provides a subset of the `QButtonGroup` API, including
        `checkedId()`, `checkedButton()`, `setCheckedId()`, and a
        `toggled` signal. Selecting one card automatically clears the
        selection from the other cards in the group.

Cards resolve their colors and typography from the active theme tokens and
refresh their styling when the application theme changes. Checked cards use
the accent surface and border to indicate selection, while unchecked cards
use the standard surface with a stronger hover state.

Option cards are intended for situations where a user must choose between
labelled alternatives, such as import policies or export destinations, rather
than for compact controls that can be represented by a standard button or
segmented selector.

Example:
    Create a group of mutually exclusive options::

        group = QATCHOptionCardGroup()

        c1 = QATCHOptionCard("Merge", "Add new, keep both")
        c2 = QATCHOptionCard("Replace", "Overwrite existing")
        c3 = QATCHOptionCard("Skip", "Leave existing")

        group.addCard(c1, 1)
        group.addCard(c2, 2)
        group.addCard(c3, 3)

        c1.setChecked(True)

    React to selection changes::

        group.toggled.connect(
            lambda card, checked: handle_selection(card)
        )

    Query the current selection::

        selected_id = group.checkedId()
        selected_card = group.checkedButton()

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.styles.typography import FONT_SANS_STACK


class _RadioDot(QtWidgets.QWidget):
    """Render a compact custom radio-button indicator.

    Draws a circular ring with an independently sized, centered solid dot
    when checked. The indicator is custom-painted because a standard
    `QFrame` with Qt stylesheets can express the outer ring but cannot
    independently control the size and appearance of a concentric checked
    indicator.

    The widget follows the application's existing convention of using small
    custom-painted indicator glyphs for precise visual control.

    Attributes:
        _checked: Whether the radio indicator is currently selected.
        _SIZE: Fixed width and height of the indicator in pixels.
        _DOT_D: Diameter of the centered checked-state dot in pixels.
        _RING_WIDTH: Width of the outer ring stroke in pixels.
    """

    _SIZE = 16
    _DOT_D = 7
    _RING_WIDTH = 1.5

    def __init__(self, parent=None) -> None:
        """Initialize the radio indicator.

        Args:
            parent: Optional parent widget.

        Returns:
            None.
        """
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self._checked = False

    def set_checked(self, checked: bool) -> None:
        """Set the checked state of the radio indicator.

        Updates the indicator only when the requested state differs from its
        current state. A repaint is scheduled whenever the state changes.

        Args:
            checked: `True` to display the selected-state dot, or `False`
                to display only the outer ring.

        Returns:
            None.
        """
        if checked != self._checked:
            self._checked = checked
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the radio ring and optional checked-state dot.

        Resolves the ring and accent colors from the active theme tokens.
        Unchecked indicators use `flat_border_strong` for the ring, while
        checked indicators use `flat_accent` for both the ring and the
        centered dot.

        The outer ring is drawn as an antialiased circle with the configured
        stroke width. When checked, a smaller solid circle is drawn centered
        within the ring.

        Args:
            event: Qt paint event generated when the indicator needs to be
                repainted.

        Returns:
            None.
        """
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


class QATCHOptionCard(QtWidgets.QFrame):
    """Provide a clickable, checkable option-selection card.

    Displays a title with an optional one-line or wrapped description inside
    a card-style container. The card can optionally include a custom-painted
    radio indicator for selection-oriented interfaces.

    The widget behaves as a lightweight interactive card rather than a
    standard Qt button. It exposes a `clicked` signal for user interaction
    and maintains its own checked state for visual and application-level
    selection handling.

    Attributes:
        clicked: Signal emitted when the card is activated by the user.
        _checked: Whether the card is currently selected.
        _title: Text displayed as the card title.
        _show_radio: Whether the custom radio indicator is displayed.
        _opacity_effect: Optional opacity effect used by the card's visual
            state handling.
        _radio_dot: Custom radio indicator displayed when `show_radio` is
            enabled.
        _title_lbl: QLabel containing the card's title.
        _desc_lbl: QLabel containing the optional description, or `None`
            when no description was provided.
    """

    clicked = QtCore.pyqtSignal()

    def __init__(self, title, description="", parent=None, *, show_radio=False):
        """Initialize the option card.

        Args:
            title: Text displayed as the card's primary title.
            description: Optional descriptive text displayed beneath the
                title. The description is word-wrapped when provided.
            parent: Optional parent widget.
            show_radio: If `True`, display a custom radio indicator to the
                left of the card content. Defaults to `False`.

        Returns:
            None.
        """
        super().__init__(parent)
        self.setObjectName("qatchOptionCard")
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
        """Refresh the card styling after the application theme changes.

        Reapplies the card stylesheet using the newly active theme tokens. If the
        card displays a radio indicator, the indicator is also scheduled for
        repainting so its colors reflect the new theme.

        Args:
            _mode: Theme mode identifier emitted by the `themeChanged` signal.
                The value is not otherwise used by this handler.

        Returns:
            None.
        """
        self._apply_qss()
        if self._show_radio:
            self._radio_dot.update()

    def text(self):
        """Return the card's title text.

        Provides a `QAbstractButton.text()`-compatible interface so callers can
        treat the option card similarly to a standard button.

        Returns:
            The current title displayed by the card.
        """
        return self._title

    def setText(self, title):
        """Set the card's title text.

        Updates both the internally stored title and the visible title label.
        This provides a `QAbstractButton.setText()`-compatible interface for
        existing call sites.

        Args:
            title: New title text to display.

        Returns:
            None.
        """
        self._title = title
        self._title_lbl.setText(title)

    def setDescription(self, description):
        """Set the card's description text.

        Updates the existing description label when the card was initialized with
        a description. Cards created without a description do not create a
        description label and therefore ignore this call.

        Args:
            description: New descriptive text to display.

        Returns:
            None.
        """
        if self._desc_lbl is not None:
            self._desc_lbl.setText(description)

    def isChecked(self):
        """Return whether the option card is currently selected.

        Returns:
            `True` if the card is checked, otherwise `False`.
        """
        return self._checked

    def setChecked(self, checked: bool):
        """Set the card's checked state.

        Updates the internal selection state and, when enabled, synchronizes the
        optional radio indicator. The card stylesheet is reapplied so the visual
        selection state is updated.

        If the requested state already matches the current state, no update is
        performed.

        Args:
            checked: `True` to select the card, or `False` to clear its
                selection state.

        Returns:
            None.
        """
        if checked == self._checked:
            return
        self._checked = checked
        if self._show_radio:
            self._radio_dot.set_checked(checked)
        self._apply_qss()

    def setCardEnabled(self, enabled: bool) -> None:
        """Enable or disable user interaction for the entire card.

        Updates the card's enabled state and cursor, and applies a 50% opacity
        effect when disabled to visually indicate that the card is unavailable.

        A `QGraphicsOpacityEffect` is intentionally used for this widget because
        the card does not continuously repaint during hover or animation cycles.
        This avoids the offscreen-pixmap caching issues that can occur when the
        same effect is used on continuously animated or frequently repainted
        custom widgets.

        Args:
            enabled: `True` to enable the card for interaction, or `False` to
                disable it and dim the entire card.

        Returns:
            None.
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
            # setGraphicsEffect() deletes the widget's *previous* effect,
            # including when clearing it with None here - so
            # self._opacity_effect would otherwise be left pointing at a
            # destroyed C++ object. Clear the Python-side reference too so
            # a later re-disable lazily creates a fresh effect instead of
            # calling setOpacity() on the deleted one.
            self.setGraphicsEffect(None)
            self._opacity_effect = None

    def mousePressEvent(self, event):
        """Handle mouse presses on the option card.

        Emits `clicked` when the user presses the left mouse button while the
        card is enabled. The event is then passed to the base `QFrame`
        implementation for normal Qt event processing.

        Args:
            event: Mouse event generated by the Qt event system.

        Returns:
            None.
        """
        if event.button() == QtCore.Qt.LeftButton and self.isEnabled():
            self.clicked.emit()
        super().mousePressEvent(event)

    @staticmethod
    def _rgba(rgba) -> str:
        """Format an RGBA token tuple as a CSS color string.

        Converts a four-component `(red, green, blue, alpha)` token into the
        CSS `rgba()` representation used by Qt stylesheets.

        Args:
            rgba: Iterable containing red, green, blue, and alpha channel values.

        Returns:
            A CSS-compatible `rgba(r, g, b, a)` string.
        """
        return f"rgba({rgba[0]}, {rgba[1]}, {rgba[2]}, {rgba[3]})"

    def _apply_qss(self):
        """Apply theme-aware stylesheet styling to the option card.

        Resolves the card's colors from the active theme tokens and updates the
        frame, title label, and optional description label. Checked cards use the
        accent color and accent-weak surface to indicate selection, while
        unchecked cards use the standard surface and border colors with a
        stronger surface and border on hover.

        The title and description labels are styled independently so their
        typography and text colors remain consistent with the application's flat
        control system.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        rgba = self._rgba
        if self._checked:
            frame_qss = f"""
                QFrame#qatchOptionCard {{
                    background: {rgba(tok["flat_accent_weak"])};
                    border: 1.5px solid {rgba(tok["flat_accent"])};
                    border-radius: 10px;
                }}
            """
        else:
            frame_qss = f"""
                QFrame#qatchOptionCard {{
                    background: {rgba(tok["flat_surface"])};
                    border: 1px solid {rgba(tok["flat_border"])};
                    border-radius: 10px;
                }}
                QFrame#qatchOptionCard:hover {{
                    background: {rgba(tok["flat_surface2"])};
                    border: 1px solid {rgba(tok["flat_border_strong"])};
                }}
            """
        self.setStyleSheet(frame_qss)
        self._title_lbl.setStyleSheet(
            f"QLabel#optionCardTitle {{ color: {rgba(tok['flat_text'])}; "
            f"font-family: {FONT_SANS_STACK}; font-size: 13px; font-weight: 600; "
            "background: transparent; }"
        )
        if self._desc_lbl is not None:
            self._desc_lbl.setStyleSheet(
                f"QLabel#optionCardDesc {{ color: {rgba(tok['flat_text_muted'])}; "
                f"font-family: {FONT_SANS_STACK}; font-size: 11.5px; background: transparent; }}"
            )


class QATCHOptionCardGroup(QtCore.QObject):
    """Manage exclusive selection across a group of option cards.

    Provides a lightweight selection manager for :class:`QATCHOptionCard`
    widgets. The group mirrors the subset of `QButtonGroup` behavior used
    by the application, allowing existing call sites to work with option
    cards without requiring the cards themselves to inherit from
    `QAbstractButton`.

    At most one card in the group can be checked at a time. Selecting a card
    automatically clears the checked state of every other card in the group.
    The group also provides lookup methods for the currently selected card
    and its associated identifier.

    Attributes:
        toggled: Signal emitted when a card becomes checked. The signal
            provides the newly checked card and `True` as its arguments.
        _cards: List of `(card, card_id)` pairs representing the cards
            managed by the group.

    Example:
        Create a group and associate cards with application-specific IDs::

            group = QATCHOptionCardGroup(parent)

            group.addCard(card_a, 0)
            group.addCard(card_b, 1)

            group.toggled.connect(
                lambda card, checked: handle_selection(card)
            )

        Retrieve the selected card or its ID::

            selected = group.checkedButton()
            selected_id = group.checkedId()
    """

    toggled = QtCore.pyqtSignal(object, bool)
    """Signal emitted when a card becomes checked.

    The signal arguments are the newly checked :class:`QATCHOptionCard` and
    `True` indicating that the card has been selected.
    """

    def __init__(self, parent=None):
        """Initialize the option card group.

        Args:
            parent: Optional parent `QObject`.

        Returns:
            None.
        """
        super().__init__(parent)
        self._cards = []  # list of (card, id)

    def addCard(self, card: QATCHOptionCard, card_id):
        """Add an option card to the exclusive-selection group.

        The card is associated with the supplied identifier and its
        `clicked` signal is connected to the group's internal selection
        handler.

        Args:
            card: Option card to add to the group.
            card_id: Identifier associated with the card. This value is
                returned by :meth:`checkedId` when the card is selected.

        Returns:
            None.
        """
        self._cards.append((card, card_id))
        card.clicked.connect(lambda c=card: self._on_card_clicked(c))

    def _on_card_clicked(self, card: QATCHOptionCard):
        """Handle a user click on an option card.

        If the clicked card is already selected, no action is taken.
        Otherwise, the clicked card is selected and all other cards in the
        group are unchecked. The `toggled` signal is then emitted for the
        newly selected card.

        Args:
            card: Card that was activated by the user.

        Returns:
            None.
        """
        if card.isChecked():
            return
        for c, _ in self._cards:
            c.setChecked(c is card)
        self.toggled.emit(card, True)

    def checkedButton(self):
        """Return the currently selected option card.

        Returns:
            The checked :class:`QATCHOptionCard`, or `None` if no card in
            the group is currently selected.
        """
        for c, _ in self._cards:
            if c.isChecked():
                return c
        return None

    def checkedId(self):
        """Return the identifier of the currently selected card.

        Returns:
            The ID associated with the checked card, or `-1` when no card
            in the group is selected.
        """
        for c, cid in self._cards:
            if c.isChecked():
                return cid
        return -1

    def setCheckedId(self, card_id):
        """Select a card programmatically by its associated identifier.

        Updates every card in the group so that only the card whose ID matches
        `card_id` remains checked. If a matching card is found, the
        `toggled` signal is emitted for that card to provide behavior
        consistent with user-driven selection.

        Args:
            card_id: Identifier of the card to select.

        Returns:
            None.
        """
        target = None
        for c, cid in self._cards:
            checked = cid == card_id
            c.setChecked(checked)
            if checked:
                target = c
        if target is not None:
            self.toggled.emit(target, True)
