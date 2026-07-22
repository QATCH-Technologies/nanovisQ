"""
QATCH.ui.components.tabbed_rail_panel

`TabbedRailPanel`: a `ConnectedTabRail` wired to a `QStackedWidget` of pages -
register pages in nav order with `add_page()`, then clicking a row switches
to it. Only the rail's own connected highlight animates (see
`ConnectedTabRail`); the page itself swaps instantly, not a cross-slide of
the whole content area - the highlight sliding to meet the clicked row IS
the transition.

Consolidates the nav-rail + content-stack wiring that used to be duplicated
in both `DataManagementWidget` (mode switcher) and `UserPreferencesWidget`
(section nav), so either - or any future panel that wants this same
"settings-style" vertical nav - can just instantiate one of these.

Usage
-----
    panel = TabbedRailPanel([
        ("import", "Import", icon_path),
        ("export", "Export", icon_path2),
    ])
    panel.add_page("import", import_page_widget)
    panel.add_page("export", export_page_widget)
    panel.currentChanged.connect(handler)  # e.g. on_enter/on_leave hooks
    panel.set_active("import")
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtWidgets

from QATCH.ui.components.connected_tab_rail import ConnectedTabRail


class TabbedRailPanel(QtWidgets.QWidget):
    """Vertical nav rail + content pages, wired together.

    Attributes:
        rail (ConnectedTabRail): The nav rail (exposed for direct access to
            e.g. its icon/color knobs).
        stack (QtWidgets.QStackedWidget): The registered pages.
        currentChanged (pyqtSignal(str)): Emitted with the new page's key
            once the stack has already switched to it - connect this for
            any per-page on_enter/on_leave-style hooks.
    """

    currentChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        modes,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        content_margins: tuple = (14, 10, 14, 14),
        **rail_kwargs,
    ) -> None:
        super().__init__(parent)
        self.rail = ConnectedTabRail(modes, **rail_kwargs)

        self.stack = QtWidgets.QStackedWidget()
        # Transparent: ConnectedTabRail paints the content pane's rounded
        # background (and the connector bridging the active row into it) as
        # one continuous shape behind this stack.
        self.stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
        # Let a page-switch truly hide() the outgoing page (cheapest possible
        # "don't paint this") while it still reserves its layout space -
        # without this, hide()/show() at the start/end of a switch briefly
        # collapses the stack to zero height and forces a full repaint of
        # whatever contains this panel.
        stack_policy = self.stack.sizePolicy()
        stack_policy.setRetainSizeWhenHidden(True)
        self.stack.setSizePolicy(stack_policy)

        content_layout = QtWidgets.QVBoxLayout(self.rail.content_area)
        content_layout.setContentsMargins(*content_margins)
        content_layout.addWidget(self.stack)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.rail, 1)

        self._page_index: dict = {}  # key -> stack index
        self.rail.modeChanged.connect(self._on_rail_changed)

    # ------------------------------------------------------------------
    def add_page(self, key: str, widget: QtWidgets.QWidget) -> None:
        """Registers `widget` as the page for `key`, in the order pages
        should appear in the stack (should match `modes`' order)."""
        self._page_index[key] = self.stack.addWidget(widget)

    def widget(self, key: str) -> Optional[QtWidgets.QWidget]:
        idx = self._page_index.get(key)
        return self.stack.widget(idx) if idx is not None else None

    def set_active(self, key: str) -> None:
        self.rail.set_active(key)

    def active_key(self) -> Optional[str]:
        return self.rail.active_key()

    # ------------------------------------------------------------------
    def _on_rail_changed(self, key: str) -> None:
        idx = self._page_index.get(key)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
        self.currentChanged.emit(key)
