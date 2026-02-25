from PyQt5 import QtCore, QtWidgets


class ReorderableCardContainer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

        # 1. Force items to stack at the top
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        self.dragged_card = None
        self.placeholder = None
        self.drag_offset = QtCore.QPoint(0, 0)

    def start_drag(self, card, global_mouse_pos, offset):
        self.dragged_card = card
        self.drag_offset = offset

        # Create placeholder
        self.placeholder = QtWidgets.QWidget()
        self.placeholder.setFixedSize(card.size())
        self.placeholder.setObjectName("dragPlaceholder")

        idx = self.main_layout.indexOf(card)
        self.main_layout.takeAt(idx)
        self.main_layout.insertWidget(idx, self.placeholder)

        card.setParent(self)
        card.raise_()
        self.update_drag(global_mouse_pos)
        card.show()

    def update_drag(self, global_mouse_pos):
        if not self.dragged_card:
            return

        # Move card
        local_pos = self.mapFromGlobal(global_mouse_pos)
        target_y = local_pos.y() - self.drag_offset.y()
        target_x = self.main_layout.contentsMargins().left()
        self.dragged_card.move(target_x, target_y)

        # 2. Logic: Use the TOP of the dragged card (the handle position)
        drag_focus_y = target_y

        placeholder_idx = self.main_layout.indexOf(self.placeholder)
        new_idx = placeholder_idx

        count = self.main_layout.count()

        for i in range(count):
            item = self.main_layout.itemAt(i)
            widget = item.widget()

            if widget is None or widget == self.dragged_card:
                continue

            w_geo = widget.geometry()
            w_center_y = w_geo.y() + w_geo.height() / 2

            # If our handle is above the center of the target card, insert before it
            if drag_focus_y < w_center_y:
                # If we are currently "after" this slot, we need to move "up"
                if i < placeholder_idx:
                    new_idx = i
                else:
                    new_idx = i
                break
            else:
                # If we are below the center, we belong after.
                # If loop finishes without breaking, new_idx will stay at 'count' (handled below)
                if i >= new_idx:
                    new_idx = i + 1

        # 3. Apply Move
        if new_idx != placeholder_idx:
            new_idx = max(0, min(new_idx, count))
            self.main_layout.takeAt(placeholder_idx)
            self.main_layout.insertWidget(new_idx, self.placeholder)

    def finish_drag(self):
        if not self.dragged_card:
            return

        final_idx = self.main_layout.indexOf(self.placeholder)
        self.main_layout.takeAt(final_idx)
        self.placeholder.deleteLater()
        self.placeholder = None

        self.main_layout.insertWidget(final_idx, self.dragged_card)
        self.dragged_card = None
