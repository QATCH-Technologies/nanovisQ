"""
QATCH.ui.components.tabbed_rail_panel

A :class:`ConnectedTabRail` wired to a :class:`~PyQt5.QtWidgets.QStackedWidget` of pages.
Register pages in navigation order with :meth:`add_page`, then clicking a row switches
to it. Only the rail's own connected highlight animates (see :class:`ConnectedTabRail`);
the page itself swaps instantly, not a cross-slide of the whole content area - the
highlight sliding to meet the clicked row IS the transition.

Consolidates the nav-rail + content-stack wiring that used to be duplicated
in both `DataManagementWidget` (mode switcher) and `UserPreferencesWidget`
(section nav), so either - or any future panel that wants this same
"settings-style" vertical nav - can just instantiate one of these.

Example:
    .. code-block:: python

        panel = TabbedRailPanel([
            ("import", "Import", icon_path),
            ("export", "Export", icon_path2),
        ])
        panel.add_page("import", import_page_widget)
        panel.add_page("export", export_page_widget)
        panel.currentChanged.connect(handler)  # e.g., on_enter/on_leave hooks
        panel.set_active("import")

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-05
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from QATCH.ui.components.connected_tab_rail import ConnectedTabRail


class TabbedRailPanel(QtWidgets.QWidget):
    """Vertical nav rail and content pages, wired together.

    Attributes:
        rail (:class:`ConnectedTabRail`): The navigation rail. Exposed for direct
            access to modify its settings (e.g., icon or color knobs).
        stack (:class:`~PyQt5.QtWidgets.QStackedWidget`): The container holding the
            registered page widgets.
        currentChanged (:class:`~PyQt5.QtCore.pyqtSignal`): Emitted with the new page's key
            (as a `str`) once the stack has already switched to it. Connect this for
            any per-page on_enter/on_leave-style hooks.
    """

    currentChanged = QtCore.pyqtSignal(str)

    def __init__(
        self,
        modes,
        parent: QtWidgets.QWidget | None = None,
        *,
        content_margins: tuple = (14, 10, 14, 14),
        **rail_kwargs,
    ) -> None:
        """Initializes the TabbedRailPanel.

        Args:
            modes (list): A list of mode configurations (e.g., tuples of `(key, label, icon)`)
                to initialize the :class:`ConnectedTabRail`.
            parent (:class:`~PyQt5.QtWidgets.QWidget`, optional): The parent widget. Defaults to None.
            content_margins (tuple, optional): Margins for the content layout area
                defined as `(left, top, right, bottom)`. Defaults to (14, 10, 14, 14).
            **rail_kwargs: Additional keyword arguments passed directly to the
                :class:`ConnectedTabRail` constructor.
        """
        super().__init__(parent)
        self.rail = ConnectedTabRail(modes, **rail_kwargs)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setStyleSheet("QStackedWidget { background: transparent; border: none; }")
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

    def add_page(self, key: str, widget: QtWidgets.QWidget) -> None:
        """Registers a widget as the page for a given key.

        Pages should be added in the order they are meant to appear in the stack
        (this should typically match the `modes` order passed during initialization).

        Args:
            key (str): The unique identifier for this page.
            widget (:class:`~PyQt5.QtWidgets.QWidget`): The widget to display when this key is active.
        """
        self._page_index[key] = self.stack.addWidget(widget)

    def widget(self, key: str) -> QtWidgets.QWidget | None:
        """Retrieves the registered widget associated with the given key.

        Args:
            key (str): The unique identifier of the page.

        Returns:
            :class:`~PyQt5.QtWidgets.QWidget` | None: The registered widget, or None if the key
            is not found in the stack.
        """
        idx = self._page_index.get(key)
        return self.stack.widget(idx) if idx is not None else None

    def set_active(self, key: str) -> None:
        """Programmatically sets the active page by its key.

        Args:
            key (str): The unique identifier of the page to activate.
        """
        self.rail.set_active(key)

    def active_key(self) -> str | None:
        """Retrieves the key of the currently active page.

        Returns:
            :class:`str` | None: The key of the active page, or None if no page is active.
        """
        return self.rail.active_key()

    def _on_rail_changed(self, key: str) -> None:
        """Handles the rail's internal modeChanged signal.

        Switches the visible widget in the :class:`~PyQt5.QtWidgets.QStackedWidget`
        and emits the public :attr:`currentChanged` signal.

        Args:
            key (str): The unique identifier of the newly selected rail mode.
        """
        idx = self._page_index.get(key)
        if idx is not None:
            self.stack.setCurrentIndex(idx)
        self.currentChanged.emit(key)
