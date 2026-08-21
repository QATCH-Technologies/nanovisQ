"""
QATCH.ui.widgets.floating_menu_widget.py

Floating navigation menu widget.

Provides a frameless, translucent floating menu used to display and
navigate between toolkit sections. The menu presents a titled list of
selectable items with visual states for the active item and mouse-hover
interaction.

The widget manages its own item layout, styling, selection state, and
mouse interaction. Selecting an item delegates navigation to the parent
widget through its `_set_learn_mode` method.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class FloatingMenuWidget(QtWidgets.QWidget):
    """Display a floating, styled navigation menu for toolkit sections.

    Creates a frameless, always-on-top menu with a rounded content surface
    and drop shadow. Menu items are displayed vertically beneath a toolkit
    title and support active-selection and hover styling.

    The widget delegates selected-item navigation to its parent and tracks
    the currently active item internally.
    """

    def __init__(self, parent=None):
        """Initialize the floating toolkit navigation menu.

        Configures the widget as a frameless, translucent tool window, creates
        the styled content container and drop shadow, and initializes the
        layouts and title used to display menu items.

        Args:
            parent (QtWidgets.QWidget, optional): The parent or controller
                associated with the floating menu. If the supplied object has a
                ``parent`` attribute, that object's parent is used as the Qt
                widget parent. Defaults to ``None``.
        """
        super().__init__(parent.parent if hasattr(parent, "parent") else parent)
        self.parent = parent

        # Internal state tracking of the active tab index
        self._active = -1

        # Make the widget frameless, transparent, and always on top
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Reserve space for the shadow effect
        self.setContentsMargins(0, 0, 10, 10)

        # Main layout for the entire window
        container_layout = QtWidgets.QVBoxLayout(self)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # Child widget where content and shadow are applied
        self.content_widget = QtWidgets.QWidget(self)
        self.content_widget.setObjectName("content_widget")
        container_layout.addWidget(self.content_widget)

        # Style the content widget
        self.content_widget.setStyleSheet("""
            #content_widget {
                background-color: #DDDDDD;
                border-radius: 10px;
            }
            """)

        # Style text color of all child widgets
        self.setStyleSheet("color: #333333;")

        # Apply shadow effect
        shadow_effect = QtWidgets.QGraphicsDropShadowEffect(self)
        # Semi-transparent dark gray
        shadow_effect.setColor(QtGui.QColor(69, 69, 69, 180))
        shadow_effect.setOffset(2, 2)
        shadow_effect.setBlurRadius(5)
        self.content_widget.setGraphicsEffect(shadow_effect)

        # Set a fixed size for the floating widget
        # NOTE: Size will auto-fit to label size
        self.setFixedSize(100, 700)

        self.vbox = QtWidgets.QVBoxLayout(self.content_widget)
        self.items = QtWidgets.QVBoxLayout()

        # Remove margins around widgets
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.items.setContentsMargins(0, 0, 0, 0)

        # Add a label to display text
        self.title = QtWidgets.QLabel("VisQ.AI<sup>TM</sup> Toolkit")
        self.title.setStyleSheet(
            "font-weight: bold; font-size: 12px; padding: 10px; padding-left: 15px;"
        )

        self.vbox.addSpacing(25)
        self.vbox.addWidget(self.title)
        self.vbox.addLayout(self.items)
        # self.vbox.addSpacing(15)
        self.vbox.addStretch()

    def addItems(self, items: list):
        """Add navigation items to the floating menu.

        Creates a label for each supplied item, applies the default item
        styling, connects mouse presses to toolkit navigation, and installs
        the event filter used for hover-state handling. The menu is resized
        to accommodate the resulting contents.

        Args:
            items (list): The menu item labels to add in display order.
        """
        for idx, item in enumerate(items):
            label = QtWidgets.QLabel(item)
            # Style the label (padding and background)
            label = self._setStyleSheet(label, False)
            # Connect the mouse press event handler
            label.mousePressEvent = lambda evt, i=idx: self._viewToolkitItem(i)
            # Install the event filter to detect mouseover events
            label.installEventFilter(self)
            self.items.addWidget(label)
        self.setFixedSize(
            self.sizeHint().width() + self.contentsMargins().right(),
            self.sizeHint().height() + self.contentsMargins().bottom(),
        )

    def removeItems(self):
        """Remove all navigation items from the floating menu.

        Detaches each item from the menu layout and schedules its widget for
        deletion.
        """
        while self.items.count():
            item = self.items.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.deleteLater()

    def setActiveItem(self, index: int):
        """Set the currently active navigation item.

        Updates the styling of every menu item so that only the item at the
        supplied index is displayed as selected, then records that index as
        the active item.

        Args:
            index (int): Zero-based index of the item to mark as active.
        """
        for idx in range(self.items.count()):
            label = self.items.itemAt(idx).widget()
            self._setStyleSheet(label, True if idx == index else False)
        self._active = index

    def _setHoverItem(self, index: int):
        """Update menu item styling for a hovered item.

        Preserves the active-item styling while applying the hover styling to
        the specified item. Passing ``-1`` clears the hover state from all
        items.

        Args:
            index (int): Zero-based index of the item currently under the
                mouse, or ``-1`` when no item is hovered.
        """
        for idx in range(self.items.count()):
            label = self.items.itemAt(idx).widget()
            self._setStyleSheet(
                label,
                selected=True if idx == self._active else False,
                hover=True if idx == index else False,
            )

    def _viewToolkitItem(self, index: int):
        """Navigate to the toolkit section represented by an item.

        Validates the requested item index and delegates navigation to the
        parent widget through its ``_set_learn_mode`` method.

        Args:
            index (int): Zero-based index of the toolkit item to activate.

        Raises:
            ValueError: If ``index`` is outside the range of available menu
                items.
        """
        if 0 <= index < self.items.count():
            self.parent._set_learn_mode(tab_index=index)
            # self.setActiveItem(index) # Handled by VisQAIWindow.on_tab_change()
        else:
            raise ValueError(f"Index {index} is out-of-bounds for toolkit items count.")

    def _setStyleSheet(
        self, label: QtWidgets.QLabel, selected: bool, hover: bool = False
    ) -> QtWidgets.QLabel:
        """Apply the appropriate visual state to a menu item.

        Updates the label stylesheet according to whether the item is active,
        hovered, both, or neither.

        Args:
            label (QtWidgets.QLabel): The menu item label to style.
            selected (bool): Whether the item is the currently active item.
            hover (bool, optional): Whether the item is currently being
                hovered. Defaults to ``False``.

        Returns:
            QtWidgets.QLabel: The styled label.
        """
        if hover and selected:
            label.setStyleSheet("padding: 10px; padding-left: 15px; background: #A9E1FA;")
        elif hover:
            label.setStyleSheet("padding: 10px; padding-left: 15px; background: #E5E5E5;")
        elif selected:
            label.setStyleSheet("padding: 10px; padding-left: 15px; background: #B7D3DC;")
        else:
            label.setStyleSheet("padding: 10px; padding-left: 15px;")
        return label

    def eventFilter(self, obj, event):
        """Handle mouse-enter and mouse-leave events for menu items.

        Detects pointer entry and exit on navigation labels and updates their
        hover styling while preserving the active-item state.

        Args:
            obj (QtCore.QObject): The object that received the event.
            event (QtCore.QEvent): The event being processed.

        Returns:
            bool: The result of the base class event-filter implementation.
        """
        if event.type() in [QtCore.QEvent.Enter, QtCore.QEvent.Leave]:
            found = False
            for idx in range(self.items.count()):
                label = self.items.itemAt(idx).widget()
                if obj is label:
                    found = True
                    break
        if event.type() == QtCore.QEvent.Enter:
            # print(f"Enter {obj.__class__.__name__} {obj.text() if hasattr(obj, 'text') else ''}")
            if found:
                self._setHoverItem(idx)
        if event.type() == QtCore.QEvent.Leave:
            # print(f"Leave {obj.__class__.__name__} {obj.text() if hasattr(obj, 'text') else ''}")
            if found:
                self._setHoverItem(-1)
        return super().eventFilter(obj, event)
