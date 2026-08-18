"""
QATCH.ui.compoennts.poi_chip_field.py

POI chip field component for editing Custom-POI index lists.

This module provides a wrapping row of removable `chip` controls for
editing a list of integer Custom-POI indices. It replaces the visual
presentation of a plain bracket-string :class:`QATCHLineEdit`, such as
`"[8723, 8725, 12180]"`, with individually removable and addable POI
tokens while preserving the existing application's text-based backend
contract.

Backend Contract:
    :class:`POIChipField` is intentionally a thin visual front-end rather
    than the authoritative source of POI data. It operates on an existing,
    real :class:`QATCHLineEdit` supplied as its backing field. Existing
    application code can therefore continue to read and write the POI list
    through the backing field's `text()` and `setText()` methods without
    modification.

    The synchronization behavior is as follows:

    * Whenever the backing field's text changes, including changes made
      programmatically through `setText()`, the chip field reparses the
      serialized values and rebuilds its visible chips. This occurs through
      the backing field's `textChanged` signal and
      :meth:`POIChipField._sync_from_backing`.

    * Whenever the user adds or removes a chip, the complete POI collection
      is normalized and serialized using `str(list_of_ints)` before being
      written back to the backing field. This preserves the existing
      `"[1, 2, 3]"` representation expected by the application.

    * After a user-initiated change is written to the backing field, the
      caller-supplied `on_commit` callback is invoked directly. This is
      necessary because `setText()` does not emit `editingFinished`,
      which would otherwise be responsible for triggering downstream POI
      processing.

    * A re-entrancy guard prevents the programmatic `setText()` performed
      during a chip update from immediately triggering another synchronization
      cycle through `textChanged`. This avoids redundant chip rebuilding
      and ensures the explicitly committed state remains authoritative.

Usage:
    Create the existing `QATCHLineEdit` as usual and pass it to
    :class:`POIChipField` as the backing field. Existing code can continue to
    interact with `custom_poi_text` normally::

        self.custom_poi_text = QATCHLineEdit()
        poi_field = POIChipField(
            self.custom_poi_text,
            self.update_custom_pois,
        )

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-18
"""

from __future__ import annotations

from typing import Callable

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.qatch_line_edit import QATCHLineEdit
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_MONO_STACK, FONT_SANS_STACK


class _FlowLayout(QtWidgets.QLayout):
    """Wrap-aware horizontal layout for arranging fixed-size child widgets.

    Implements a left-to-right flow layout that automatically moves items
    onto subsequent rows when the available horizontal space is exhausted.
    This provides CSS `flex-wrap`-style behavior for Qt, which does not
    provide an equivalent built-in layout.

    The implementation follows the standard Qt Flow Layout example and is
    adapted for fixed-size children, such as the chip controls used by the
    dialog UI. The layout reports a height dependent on its available width
    so that parent layouts can correctly resize the containing widget when
    items wrap across multiple rows.

    Args:
        parent: Optional parent layout or widget accepted by
            :class:`QtWidgets.QLayout`.
        margin: Uniform margin, in pixels, applied to all four sides of the
            layout.
        spacing: Horizontal and vertical spacing, in pixels, between items.

    Attributes:
        _items (List[QtWidgets.QLayoutItem]): Layout items managed by the
            flow layout.
        _spacing (int): Horizontal and vertical spacing between adjacent
            items and wrapped rows.

    Notes:
        The layout does not expand in either direction. Its minimum size is
        calculated from the minimum sizes of its child items and its
        configured contents margins. Wrapping is calculated by
        :meth:`_do_layout`, which can operate in either measurement-only or
        geometry-setting mode.
    """

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6) -> None:
        """Initialize the flow layout.

        Args:
            parent: Optional parent layout or widget.
            margin: Uniform contents margin in pixels.
            spacing: Horizontal and vertical spacing between items in pixels.
        """
        super().__init__(parent)
        self._items: list[QtWidgets.QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        """Add a layout item to the flow layout.

        Args:
            item: Layout item to append to the managed item collection.
        """
        self._items.append(item)

    def count(self) -> int:
        """Return the number of items currently managed by the layout.

        Returns:
            Number of layout items.
        """
        return len(self._items)

    def itemAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        """Return the layout item at the specified index.

        Args:
            index: Zero-based item index.

        Returns:
            The layout item at `index`, or `None` if the index is out of
            range.
        """
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QtWidgets.QLayoutItem | None:
        """Remove and return the layout item at the specified index.

        Args:
            index: Zero-based item index.

        Returns:
            The removed layout item, or `None` if the index is out of range.
        """
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientations:
        """Return the directions in which the layout can expand.

        Returns:
            Empty orientations, indicating that the layout does not request
            expansion in either direction.
        """
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        """Indicate that the layout height depends on its available width.

        Returns:
            `True` because wrapping changes the required layout height.
        """
        return True

    def heightForWidth(self, width: int) -> int:
        """Calculate the layout height required for a given width.

        Args:
            width: Available layout width in pixels.

        Returns:
            Required layout height in pixels, including configured margins.
        """
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        """Position child items within the supplied layout rectangle.

        Args:
            rect: Rectangle defining the geometry available to the layout.
        """
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        """Return the preferred size of the layout.

        Returns:
            The layout's minimum size, which is also used as its size hint.
        """
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        """Calculate the minimum size required by all layout items.

        The minimum width and height are derived from the largest minimum
        dimensions among the child items, with the configured contents
        margins added around the result.

        Returns:
            Minimum size required by the layout and its child items.
        """
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        """Calculate or apply wrapped item geometry.

        Items are placed from left to right until the next item would exceed
        the available width. The item is then moved to the beginning of the
        next row. When `test_only` is true, positions are calculated but
        child geometries are not modified; this mode is used by
        :meth:`heightForWidth`.

        Args:
            rect: Rectangle defining the available layout area.
            test_only: If `True`, calculate the required height without
                modifying child geometry. If `False`, apply calculated
                geometries to the layout items.

        Returns:
            Total vertical extent of the laid-out items, including the bottom
            contents margin.
        """
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y = effective.x(), effective.y()
        line_height = 0

        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())

        return y + line_height - rect.y() + m.bottom()


class _POIChip(QtWidgets.QWidget):
    """Compact removable pill representing a single point-of-interest index.

    Displays the POI value using the application's monospace font alongside
    a small `x` removal glyph. Clicking the removal glyph emits
    :attr:`removeRequested` with this widget as the signal payload.

    The chip's background, border, text, and removal-glyph colors are
    resolved from the active theme's `flat_*` tokens. The chip automatically
    reapplies its styling when the application theme changes.

    Args:
        value: Integer POI index represented by the chip.
        parent: Optional parent widget.

    Attributes:
        value (int): POI index represented by the chip.
        _label (QtWidgets.QLabel): Label displaying the POI index.
        _close (QtWidgets.QLabel): Clickable `x` glyph used to request
            removal.

    Signals:
        removeRequested: Emitted when the chip's removal glyph is clicked.
            The signal payload is this :class:`_POIChip` instance.
    """

    removeRequested = QtCore.pyqtSignal(object)  # emits self

    def __init__(self, value: int, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the POI chip.

        Args:
            value: Integer POI index represented by the chip.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.value = value
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(9, 3, 6, 3)
        lay.setSpacing(6)

        self._label = QtWidgets.QLabel(str(value))
        self._close = QtWidgets.QLabel("x")  # TODO: Swap this to the x svg
        self._close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._close.setToolTip("Remove")
        self._close.mousePressEvent = self._on_close_clicked  # type: ignore[assignment]

        lay.addWidget(self._label)
        lay.addWidget(self._close)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_close_clicked(self, _event: QtGui.QMouseEvent) -> None:
        """Emit a removal request for this chip.

        Args:
            _event: Mouse event generated by clicking the removal glyph.
                The event is intentionally unused.
        """
        self.removeRequested.emit(self)

    def _on_theme_changed(self, _mode: str) -> None:
        """Reapply chip styling after an application theme change.

        Args:
            _mode: Name or identifier of the newly active theme mode.
                The value is not otherwise used because styling is resolved
                directly from the active theme tokens.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply theme-aware styling to the chip and its child labels.

        Background, border, text, and removal-glyph colors are resolved from
        the active `flat_*` theme tokens so the chip remains consistent
        with the application's current light or dark theme.
        """
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(
            f"_POIChip {{ background: {tok_css(tok['flat_surface2'])}; "
            f"border: 1px solid {tok_css(tok['flat_border'])}; border-radius: 6px; }}"
        )
        self._label.setStyleSheet(
            "QLabel { background: transparent; border: none; "
            f"color: {tok_css(tok['flat_text'])}; font-family: {FONT_MONO_STACK}; font-size: 12px; }}"
        )
        self._close.setStyleSheet(
            "QLabel { background: transparent; border: none; "
            f"color: {tok_css(tok['flat_text_muted'])}; font-family: {FONT_SANS_STACK}; "
            "font-size: 13px; }"
        )


class POIChipField(QtWidgets.QWidget):
    """Wrapping editor for a bounded, ordered collection of POI index chips.

    Presents POI indices as removable chips arranged by a wrapping
    :class:`_FlowLayout`, with a small inline `add index…` input used to
    append new indices. The widget is backed by a hidden
    :class:`QATCHLineEdit` so existing code that reads from or writes to the
    backing field continues to work without modification.

    The backing field retains the exact bracket-string representation
    expected by `UIAnalyze.update_custom_pois`. Changes made through the
    chip editor are parsed, normalized, sorted in ascending order, capped at
    `max_chips`, and written back to the backing field.

    The add-index input is hidden automatically when the maximum number of
    chips has been reached and becomes visible again when a chip is removed.
    Input is normalized on every update, so values do not need to be entered
    in ascending order.

    Attributes:
        _backing (QATCHLineEdit): Hidden line edit containing the serialized
            POI index list used by existing application code.
        _on_commit (Callable[[], None] | None): Optional callback invoked
            after a committed POI change.
        _max_chips (int): Maximum number of POI chips allowed.
        _guard (bool): Internal re-entrancy guard used while synchronizing
            the chip representation with the backing field.
        _chips (List[_POIChip]): Currently displayed POI chips.
        _flow (_FlowLayout): Wrapping layout containing the chips and input
            control.
        _new_input (QtWidgets.QLineEdit): Inline input used to add a new POI
            index.

    Notes:
        The backing field remains a real Qt widget rather than being replaced
        by an internal string. This preserves compatibility with existing
        call sites that invoke methods such as `setText()` and `text()`
        directly on the original field.
    """

    def __init__(
        self,
        backing_field: QATCHLineEdit,
        on_commit: Callable[[], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
        *,
        max_chips: int = 5,
    ) -> None:
        """Initialize the POI chip field.

        Args:
            backing_field: Existing line edit whose text stores the serialized
                POI index list.
            on_commit: Optional callback invoked after a POI selection change.
            parent: Optional parent widget.
            max_chips: Maximum number of POI indices allowed.
        """
        super().__init__(parent)
        self._backing = backing_field
        self._on_commit = on_commit
        self._max_chips = max_chips
        self._guard = False
        self._chips: list[_POIChip] = []
        self._backing.setParent(self)
        self._backing.hide()

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(38)
        self._flow = _FlowLayout(self, margin=7, spacing=6)

        self._new_input = QtWidgets.QLineEdit(self)
        self._new_input.setPlaceholderText("add index…")
        self._new_input.setFrame(False)
        self._new_input.setFixedWidth(80)
        self._new_input.returnPressed.connect(self._on_new_index_entered)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

        self._backing.textChanged.connect(self._sync_from_backing)
        self._sync_from_backing(self._backing.text())

    def _on_theme_changed(self, _mode: str) -> None:
        """Reapply the field styling after a theme change.

        Args:
            _mode: Identifier for the newly active theme mode. Styling is
                resolved directly from the active theme tokens.
        """
        self._apply_theme()

    def _apply_theme(self) -> None:
        """Apply the active theme's styling to the chip field and input.

        The field background and border use the shared flat-control tokens,
        while the inline input uses the application's monospace typography
        and current text color.
        """
        tok = ThemeManager.instance().tokens()
        self.setStyleSheet(
            f"POIChipField {{ background: {tok_css(tok['flat_surface'])}; "
            f"border: 1px solid {tok_css(tok['flat_border'])}; border-radius: 7px; }}"
        )
        self._new_input.setStyleSheet(
            "QLineEdit { background: transparent; border: none; "
            f"color: {tok_css(tok['flat_text'])}; font-family: {FONT_MONO_STACK}; "
            "font-size: 12px; padding: 3px 4px; }"
        )

    def values(self) -> list[int]:
        """Return the currently selected POI indices.

        The returned values are normalized to ascending order and limited to
        the configured maximum number of chips. The result represents the
        same values currently displayed by the chip row and serialized in
        the backing field.

        Returns:
            List of selected POI indices in ascending order.
        """
        return self._parse_values()[: self._max_chips]

    def _parse_values(self) -> list[int]:
        """Parse POI indices from the serialized backing-field text.

        The backing field may contain bracketed, comma-separated, or
        whitespace-separated values. Each token is converted through
        `float` before `int` so serialized numeric values such as
        `"1.0"` remain valid. Tokens that cannot be interpreted as numeric
        values are silently ignored.

        Returns:
            List of parsed POI indices in the order they appear in the
            backing field. Ordering and the maximum-chip limit are applied
            separately by the synchronization and public API methods.
        """
        raw = self._backing.text()
        raw = raw.replace("[", "").replace("]", "").replace(",", "")
        values: list[int] = []
        for token in raw.split():
            try:
                values.append(int(float(token)))
            except ValueError:
                continue
        return values

    def _sync_from_backing(self, _text: str = "") -> None:
        """Synchronize the visible chips with the backing field.

        Rebuilds the chip collection whenever the backing field changes,
        unless synchronization is already in progress. This guard prevents
        the widget from recursively rebuilding itself when
        :meth:`_write_back` updates the backing field programmatically.

        Args:
            _text: Current backing-field text emitted by `textChanged`.
                The value is intentionally unused because the current text is
                read directly from the backing field.
        """
        if self._guard:
            return
        self._rebuild_chips(self._parse_values()[: self._max_chips])

    def _rebuild_chips(self, values: list[int]) -> None:
        """Rebuild the visible chip row from a collection of POI indices.

        Existing chip widgets are removed from the flow layout and scheduled
        for deletion before new chips are created. The inline add-index input
        is always retained as the final layout item and is hidden whenever
        `values` has reached `max_chips`.

        Args:
            values: POI indices to display as removable chips.
        """
        while self._flow.count():
            item = self._flow.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._new_input:
                w.setParent(None)
                w.deleteLater()

        self._chips = []
        for v in values:
            chip = _POIChip(v, self)
            chip.removeRequested.connect(self._on_chip_removed)
            self._flow.addWidget(chip)
            self._chips.append(chip)
        self._flow.addWidget(self._new_input)
        self._new_input.setVisible(len(values) < self._max_chips)

    def _write_back(self, values: list[int]) -> None:
        """Normalize, serialize, and commit a new POI index collection.

        Values are sorted in ascending order before being written to the
        backing field. A synchronization guard prevents the resulting
        `textChanged` signal from triggering a redundant rebuild. The
        visible chip row is then rebuilt explicitly, followed by the optional
        commit callback.

        Args:
            values: POI indices to store and display.
        """
        values = sorted(values)
        self._guard = True
        try:
            self._backing.setText(str(values))
        finally:
            self._guard = False
        self._rebuild_chips(values)
        if callable(self._on_commit):
            self._on_commit()

    def _on_chip_removed(self, chip: _POIChip) -> None:
        """Remove a chip and commit the resulting POI index collection.

        The selected values are reconstructed from the currently displayed
        chips, excluding the chip that requested removal, and then passed to
        :meth:`_write_back` for sorting, serialization, and UI
        synchronization.

        Args:
            chip: POI chip requesting removal.
        """
        self._write_back([c.value for c in self._chips if c is not chip])

    def _on_new_index_entered(self) -> None:
        """Parse and add a POI index entered in the inline input.

        The input is cleared immediately after submission. Empty input and
        submissions made while the maximum number of chips is already
        present are ignored. Numeric values are accepted through `float`
        conversion before being converted to integers, allowing entries such
        as `"12.0"`. Invalid non-numeric input is silently ignored.

        Newly accepted values are appended to the current chip values and
        passed to :meth:`_write_back`, which sorts and commits the complete
        collection.
        """
        text = self._new_input.text().strip()
        self._new_input.clear()
        if not text or len(self._chips) >= self._max_chips:
            return
        try:
            value = int(float(text))
        except ValueError:
            return
        self._write_back([c.value for c in self._chips] + [value])
