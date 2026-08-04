"""
poi_chip_field.py

A wrapping row of removable "chip" pills for editing a list of integer
Custom-POI indices, replacing a plain bracket-string QLineEdit
(`"[8723, 8725, 12180]"`) with individually removable/addable tokens.

Backend contract
-----------------
`POIChipField` never becomes the source of truth by itself - it is a thin
visual front-end on top of an existing, real `QATCHLineEdit` (the
"backing field", e.g. `UIAnalyze.custom_poi_text`) that the rest of the app
already reads/writes via `.text()`/`.setText()` (see
`UIAnalyze.update_custom_pois` and its four `setText(f"{poi_vals}")`
call sites):

  * Whenever the backing field's text changes (including programmatic
    `setText()` calls from elsewhere in the app), `POIChipField` re-parses
    it and rebuilds its chips (`textChanged` -> `_sync_from_backing`).
  * Whenever the user adds/removes a chip through this widget, the exact
    same `"[v1, v2, v3]"`-style string (`str(list_of_ints)`) is written
    back into the backing field via `setText`, then `on_commit` (the
    caller-supplied `update_custom_pois` bound method) is invoked
    directly - `setText` alone does not fire `editingFinished`, so nothing
    else would apply the change otherwise. A re-entrancy guard
    (`_guard`) prevents that `setText` from bouncing back into
    `_sync_from_backing` and rebuilding the chips it just built.

Usage
-----
    self.custom_poi_text = QATCHLineEdit()  # unchanged, kept as data store
    poi_field = POIChipField(self.custom_poi_text, self.update_custom_pois)

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.qatch_line_edit import QATCHLineEdit
from QATCH.ui.styles.theme_manager import ThemeManager, tok_css
from QATCH.ui.styles.typography import FONT_MONO_STACK, FONT_SANS_STACK


class _FlowLayout(QtWidgets.QLayout):
    """Left-to-right layout that wraps to a new line when it runs out of
    horizontal room - Qt has no built-in equivalent of CSS `flex-wrap`.

    Standard "Flow Layout" recipe (as documented in Qt's own examples),
    adapted for the chip row's fixed-size children.
    """

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._spacing = spacing
        self.setContentsMargins(margin, margin, margin, margin)

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientations:  # noqa: N802
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
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
    """A single removable pill: monospace index text + a small "x" glyph."""

    removeRequested = QtCore.pyqtSignal(object)  # emits self

    def __init__(self, value: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.value = value
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(9, 3, 6, 3)
        lay.setSpacing(6)

        self._label = QtWidgets.QLabel(str(value))
        self._close = QtWidgets.QLabel("×")
        self._close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._close.setToolTip("Remove")
        self._close.mousePressEvent = self._on_close_clicked  # type: ignore[assignment]

        lay.addWidget(self._label)
        lay.addWidget(self._close)

        self._apply_theme()
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_close_clicked(self, _event: QtGui.QMouseEvent) -> None:
        self.removeRequested.emit(self)

    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
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
    """Wrapping row of removable POI-index chips backed by a hidden
    `QATCHLineEdit` that preserves the exact bracket-string format
    `UIAnalyze.update_custom_pois` parses. See module docstring for the
    full backend contract.

    Capped at `max_chips` entries (the "add index…" input hides itself once
    full, and reappears the moment a chip is removed) and always kept in
    ascending order - every add/remove re-sorts the full value list before
    writing it back, rather than only accepting already-sorted input.
    """

    def __init__(
        self,
        backing_field: QATCHLineEdit,
        on_commit: Optional[Callable[[], None]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        max_chips: int = 5,
    ) -> None:
        super().__init__(parent)
        self._backing = backing_field
        self._on_commit = on_commit
        self._max_chips = max_chips
        self._guard = False
        self._chips: List[_POIChip] = []

        # The backing field is a real widget (kept alive so every existing
        # `self.custom_poi_text.setText(...)` call site keeps working
        # unmodified) but is not part of this widget's visible layout - it
        # is purely a data store now.
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

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------
    def _on_theme_changed(self, _mode: str) -> None:
        self._apply_theme()

    def _apply_theme(self) -> None:
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def values(self) -> List[int]:
        """Returns the current chip values, in ascending order (the same
        list currently reflected in both the chip row and the backing
        field)."""
        return self._parse_values()[: self._max_chips]

    # ------------------------------------------------------------------
    # Parsing / sync (backend contract - see module docstring)
    # ------------------------------------------------------------------
    def _parse_values(self) -> List[int]:
        raw = self._backing.text()
        raw = raw.replace("[", "").replace("]", "").replace(",", "")
        values: List[int] = []
        for token in raw.split():
            try:
                values.append(int(float(token)))
            except ValueError:
                continue
        return values

    def _sync_from_backing(self, _text: str = "") -> None:
        if self._guard:
            return
        self._rebuild_chips(self._parse_values()[: self._max_chips])

    def _rebuild_chips(self, values: List[int]) -> None:
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

    def _write_back(self, values: List[int]) -> None:
        values = sorted(values)
        self._guard = True
        try:
            self._backing.setText(str(values))
        finally:
            self._guard = False
        self._rebuild_chips(values)
        if callable(self._on_commit):
            self._on_commit()

    # ------------------------------------------------------------------
    # User interaction
    # ------------------------------------------------------------------
    def _on_chip_removed(self, chip: _POIChip) -> None:
        self._write_back([c.value for c in self._chips if c is not chip])

    def _on_new_index_entered(self) -> None:
        text = self._new_input.text().strip()
        self._new_input.clear()
        if not text or len(self._chips) >= self._max_chips:
            return
        try:
            value = int(float(text))
        except ValueError:
            return
        self._write_back([c.value for c in self._chips] + [value])
