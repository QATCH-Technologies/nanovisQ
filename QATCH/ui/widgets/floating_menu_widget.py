from PyQt5 import QtCore, QtGui, QtWidgets


class FloatingMenuWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent.parent if hasattr(parent, "parent") else parent)
        self.parent = parent

        # Internal state tracking of the active tab index
        self._active = -1

        # Make the widget frameless, transparent, and always on top
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

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
        while self.items.count():
            item = self.items.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None:
                widget.deleteLater()

    def setActiveItem(self, index: int):
        for idx in range(self.items.count()):
            label = self.items.itemAt(idx).widget()
            self._setStyleSheet(label, True if idx == index else False)
        self._active = index

    def _setHoverItem(self, index: int):
        for idx in range(self.items.count()):
            label = self.items.itemAt(idx).widget()
            self._setStyleSheet(
                label,
                selected=True if idx == self._active else False,
                hover=True if idx == index else False,
            )

    def _viewToolkitItem(self, index: int):
        if 0 <= index < self.items.count():
            self.parent._set_learn_mode(tab_index=index)
            # self.setActiveItem(index) # Handled by VisQAIWindow.on_tab_change()
        else:
            raise ValueError(f"Index {index} is out-of-bounds for toolkit items count.")

    def _setStyleSheet(
        self, label: QtWidgets.QLabel, selected: bool, hover: bool = False
    ) -> QtWidgets.QLabel:
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
