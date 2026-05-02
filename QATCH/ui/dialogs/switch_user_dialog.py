import os
from typing import List, Optional, Tuple
from PyQt5 import QtCore, QtGui, QtWidgets
from QATCH.common.architecture import Architecture
from QATCH.ui.components.user_tile import DraggableUserTile, DropPlaceholder

class SwitchUserDialog(QtWidgets.QDialog):
    """Frameless, popup profile-picker anchored near the avatar button."""

    user_selected = QtCore.pyqtSignal(str, str)
    add_user_requested = QtCore.pyqtSignal()
    users_reordered = QtCore.pyqtSignal(list)

    _AVATAR_D: int = 52
    _COLS: int = 3

    _TILE_W = 72
    _TILE_H = 88
    _H_SPACE = 16
    _V_SPACE = 12

    def __init__(
        self,
        users: List[Tuple[str, str]],
        parent: Optional[QtWidgets.QWidget] = None,
        current_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            parent,
            QtCore.Qt.FramelessWindowHint | QtCore.Qt.Popup | QtCore.Qt.NoDropShadowWindowHint,
        )

        self._current_name = current_name
        self._users = list(users)

        self._tile_widgets = []
        self._live_tiles = []
        self.placeholder = None
        self._drag_offset = QtCore.QPoint()

        self._animations = []

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._apply_styles()
        self._build_ui()

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QFrame#switchCard {{
                background: rgba(244, 247, 249, 230);
                border: 1px solid rgba(255, 255, 255, 220);
                border-radius: 12px;
            }}
            QLabel#switchTitle {{
                color: rgba(60, 60, 60, 220);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11pt;
                font-weight: 600;
            }}
            QLabel#switchSubtitle {{
                color: rgba(60, 60, 60, 160);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 8.5pt;
            }}
            QFrame#switchSep {{
                border-top: 1px solid rgba(200, 210, 220, 150);
                max-height: 1px;
            }}
            QLabel#tileName {{
                color: rgba(60, 60, 60, 200);
                font-size: 8.5pt;
                font-weight: 500;
            }}
            
            /* --- Scroll Area & Scrollbar Styling --- */
            QScrollArea#gridScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea#gridScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QScrollBar:vertical {{
                border: none;
                background: transparent;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(160, 175, 190, 150);
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(130, 150, 170, 200);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
                border: none;
            }}
            
            QPushButton#userTileBtn,
            QPushButton#addTileBtn {{
                background: transparent;
                border-radius: {self._AVATAR_D // 2}px;
                border: 2px solid transparent;
            }}
            QPushButton#addTileBtn {{
                background-color: rgba(229, 229, 229, 150);
            }}
            QPushButton#addTileBtn:hover {{
                background-color: rgba(210, 215, 220, 180);
            }}
            QPushButton#userTileBtn:hover {{
                background-color: rgba(229, 229, 229, 120);
            }}
            QPushButton#userTileBtn[current="true"] {{
                border-color: rgba(10, 163, 230, 120);
                background: rgba(10, 163, 230, 25);
            }}
        """)

    def _build_ui(self) -> None:
        self.root_layout = QtWidgets.QVBoxLayout(self)
        self.root_layout.setContentsMargins(10, 10, 10, 10)

        self.card = QtWidgets.QFrame()
        self.card.setObjectName("switchCard")

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        shadow.setOffset(0, 4)
        self.card.setGraphicsEffect(shadow)

        self.root_layout.addWidget(self.card)

        cv = QtWidgets.QVBoxLayout(self.card)
        cv.setContentsMargins(16, 16, 16, 20)
        cv.setSpacing(8)

        title = QtWidgets.QLabel("Switch Users")
        title.setObjectName("switchTitle")
        title.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(title)

        sep = QtWidgets.QFrame()
        sep.setObjectName("switchSep")
        cv.addWidget(sep)

        # 1. Setup Scroll Area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setObjectName("gridScrollArea")
        self.scroll_area.setWidgetResizable(False)  # We will manually size the inner widget
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        # 2. Setup the Absolute Positioning Container
        self.grid_container = QtWidgets.QWidget()
        self.grid_container.setObjectName("gridContainer")
        self.scroll_area.setWidget(self.grid_container)

        cv.addWidget(self.scroll_area, 0, QtCore.Qt.AlignHCenter)

        self._build_add_button()
        self._init_tiles()

    def _build_add_button(self):
        self.add_container = QtWidgets.QWidget(self.grid_container)
        self.add_container.setFixedSize(self._TILE_W, self._TILE_H)

        alayout = QtWidgets.QVBoxLayout(self.add_container)
        alayout.setContentsMargins(0, 0, 0, 0)
        alayout.setSpacing(6)

        btn = QtWidgets.QPushButton()
        btn.setObjectName("addTileBtn")
        btn.setFixedSize(self._AVATAR_D, self._AVATAR_D)
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setIcon(
            QtGui.QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "add-user.svg"))
        )
        btn.setIconSize(QtCore.QSize(self._AVATAR_D, self._AVATAR_D))
        btn.clicked.connect(self._on_add)

        lbl = QtWidgets.QLabel("Add\nUser")
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setObjectName("tileName")

        alayout.addWidget(btn, 0, QtCore.Qt.AlignHCenter)
        alayout.addWidget(lbl, 0, QtCore.Qt.AlignHCenter)

    def _init_tiles(self):
        for widget in self._tile_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._tile_widgets.clear()

        for idx, (name, initials) in enumerate(self._users):
            tile = DraggableUserTile(
                name=name,
                initials=initials,
                index=idx,
                is_current=(name == self._current_name),
                avatar_size=self._AVATAR_D,
                parent=self.grid_container,
            )
            tile.clicked.connect(self._on_select)
            tile.show()
            self._tile_widgets.append(tile)

        self._live_tiles = list(self._tile_widgets)
        self._layout_tiles(animate=False)

    def _layout_tiles(self, animate=True, exclude=None):

        total_items = len(self._live_tiles) + 1
        cols = max(1, min(self._COLS, total_items))
        rows = (total_items + cols - 1) // cols

        # Calculate strict geometries
        container_w = cols * self._TILE_W + (cols - 1) * self._H_SPACE
        container_h = rows * self._TILE_H + (rows - 1) * self._V_SPACE
        self.grid_container.setFixedSize(container_w, container_h)

        # Cap the scroll area viewport to exactly 2 rows tall (188px)
        max_scroll_h = (2 * self._TILE_H) + self._V_SPACE
        scrollbar_allowance = 12 if container_h > max_scroll_h else 0

        self.scroll_area.setFixedWidth(container_w + scrollbar_allowance)
        self.scroll_area.setFixedHeight(min(container_h, max_scroll_h))

        self._animations.clear()

        # Flow live tiles
        for idx, widget in enumerate(self._live_tiles):
            if widget == exclude:
                continue

            row, col = divmod(idx, cols)
            target_x = col * (self._TILE_W + self._H_SPACE)
            target_y = row * (self._TILE_H + self._V_SPACE)

            self._slide_widget(widget, target_x, target_y, animate)

        # Flow the Add button to the very end
        add_idx = len(self._live_tiles)
        row, col = divmod(add_idx, cols)
        target_x = col * (self._TILE_W + self._H_SPACE)
        target_y = row * (self._TILE_H + self._V_SPACE)

        self._slide_widget(self.add_container, target_x, target_y, animate)
        self.adjustSize()

    def _slide_widget(self, widget, target_x, target_y, animate):
        target_pos = QtCore.QPoint(target_x, target_y)

        if not animate or widget.pos() == target_pos:
            widget.move(target_pos)
            return

        anim = QtCore.QPropertyAnimation(widget, b"pos", self)
        anim.setDuration(200)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        anim.setEndValue(target_pos)
        anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)
        self._animations.append(anim)

    def _get_slot_center(self, idx: int) -> QtCore.QPoint:
        total_items = len(self._tile_widgets) + 1
        cols = max(1, min(self._COLS, total_items))
        row, col = divmod(idx, cols)

        x = col * (self._TILE_W + self._H_SPACE)
        y = row * (self._TILE_H + self._V_SPACE)

        return QtCore.QPoint(x + self._TILE_W // 2, y + self._TILE_H // 2)

    # --- CUSTOM GEOMETRIC DRAG SYSTEM ---

    def start_custom_drag(self, tile) -> None:
        self._drag_offset = tile.mapFromGlobal(QtGui.QCursor.pos())

        self.placeholder = DropPlaceholder(self._TILE_W, self._TILE_H, self.grid_container)
        self.placeholder.show()

        self.placeholder.lower()
        tile.raise_()

        idx = self._live_tiles.index(tile)
        self._live_tiles[idx] = self.placeholder

        self._layout_tiles(animate=True, exclude=tile)

    def process_custom_drag(self, tile, global_mouse_pos: QtCore.QPoint) -> None:
        # 1. Edge-Detection Auto-Scrolling
        vp_pos = self.scroll_area.viewport().mapFromGlobal(global_mouse_pos)
        vbar = self.scroll_area.verticalScrollBar()

        # If cursor is within 20px of the top/bottom viewport edges, scroll
        if vp_pos.y() < 20:
            vbar.setValue(vbar.value() - 8)
        elif vp_pos.y() > self.scroll_area.viewport().height() - 20:
            vbar.setValue(vbar.value() + 8)

        # 2. Update Tile Position
        local_pos = self.grid_container.mapFromGlobal(global_mouse_pos)

        new_x = local_pos.x() - self._drag_offset.x()
        new_y = local_pos.y() - self._drag_offset.y()

        # Clamp bounds strictly to the inner grid container
        max_x = self.grid_container.width() - tile.width()
        max_y = self.grid_container.height() - tile.height()

        new_x = max(0, min(new_x, max_x))
        new_y = max(0, min(new_y, max_y))

        tile.move(new_x, new_y)

        # 3. Collision Logic
        tile_center = tile.geometry().center()
        closest_idx = self._live_tiles.index(self.placeholder)
        min_dist = float("inf")

        for idx in range(len(self._live_tiles)):
            slot_center = self._get_slot_center(idx)
            dist = (tile_center - slot_center).manhattanLength()
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        add_slot_center = self._get_slot_center(len(self._live_tiles))
        dist_to_add = (tile_center - add_slot_center).manhattanLength()
        if dist_to_add < min_dist:
            closest_idx = len(self._live_tiles) - 1

        current_idx = self._live_tiles.index(self.placeholder)
        if closest_idx != current_idx:
            self._live_tiles.remove(self.placeholder)
            self._live_tiles.insert(closest_idx, self.placeholder)
            self._layout_tiles(animate=True, exclude=tile)

    def end_custom_drag(self, tile) -> None:
        final_idx = self._live_tiles.index(self.placeholder)
        source_idx = self._tile_widgets.index(tile)

        self._live_tiles[final_idx] = tile

        if self.placeholder:
            self.placeholder.deleteLater()
            self.placeholder = None

        if final_idx != source_idx:
            user = self._users.pop(source_idx)
            self._users.insert(final_idx, user)

            widget = self._tile_widgets.pop(source_idx)
            self._tile_widgets.insert(final_idx, widget)

            for i, w in enumerate(self._tile_widgets):
                w.index = i

            self.users_reordered.emit(self._users)

        self._layout_tiles(animate=True)

    def _on_select(self, name: str, initials: str) -> None:
        self.user_selected.emit(name, initials)
        self.accept()

    def _on_add(self) -> None:
        self.add_user_requested.emit()
        self.accept()
