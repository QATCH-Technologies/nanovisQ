from PyQt5.QtWidgets import (
    QListWidget,
)

from PyQt5.QtCore import (
    Qt,
)
from PyQt5.QtGui import (
    QMouseEvent,
)


class ToggleListWidget(QListWidget):
    """A QListWidget that allows toggling item selection.

    Extends the standard QListWidget to provide a "toggle" behavior where
    clicking an already-selected item deselects it, provided it is the
    only item currently selected.
    """

    def mousePressEvent(self, event: QMouseEvent):  # noqa: N802
        """Overrides the mouse press event to handle selection toggling.

        If a user left-clicks a single item that is already selected (without
        keyboard modifiers), the selection is cleared. Otherwise, the standard
        QListWidget selection behavior is executed.

        Args:
            event (QMouseEvent): The mouse event containing position,
                button, and modifier information.
        """
        if (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            item = self.itemAt(event.pos())
            if item is not None and item.isSelected() and len(self.selectedItems()) == 1:
                self.clearSelection()
                self.setCurrentItem(None)
                event.accept()
                return
        super().mousePressEvent(event)
