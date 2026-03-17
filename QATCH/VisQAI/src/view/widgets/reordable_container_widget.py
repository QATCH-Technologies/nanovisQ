"""
reordable_container_widget.py

A drag-and-drop reorderable card container for PyQt5 layouts.

Provides ``ReorderableCardContainer``, a ``QWidget`` that manages free-form
vertical drag-and-drop reordering of child card widgets.  During a drag the
card is reparented to the container and floats freely over the layout while a
fixed-size placeholder widget holds the card's intended insertion slot.  The
placeholder slides to the correct position in real time as the user drags,
giving immediate visual feedback before the drop is committed.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

from PyQt5 import QtCore, QtWidgets


class ReorderableCardContainer(QtWidgets.QWidget):
    """A vertical card container that supports live drag-and-drop reordering.

    Manages all drag state and layout mutations required for smooth card
    reordering.  Child cards are responsible for detecting mouse events and
    calling ``start_drag``, ``update_drag``, and ``finish_drag`` at the
    appropriate points in the drag lifecycle; this class owns all layout
    manipulation and placeholder management.

    During a drag, the card being moved is:

    * Detached from the ``QVBoxLayout`` and reparented directly to this widget
      so it can be positioned freely with ``move()``.
    * Replaced in the layout by a fixed-size ``QWidget`` placeholder
      (``objectName`` ``"dragPlaceholder"``) that reserves the insertion slot
      and shifts sibling cards in real time.

    On ``finish_drag`` the placeholder is removed, and the card is reinserted
    at the placeholder's final index, committing the new order.

    Attributes:
        main_layout (QtWidgets.QVBoxLayout): The vertical layout that holds all
            child card widgets.  Configured with 15 px margins, 10 px spacing,
            and top alignment so cards stack from the top rather than being
            spread across the full widget height.
        dragged_card (QtWidgets.QWidget | None): The card widget currently
            being dragged, or ``None`` when no drag is in progress.
        placeholder (QtWidgets.QWidget | None): The transparent placeholder
            widget occupying the dragged card's reserved slot in the layout,
            or ``None`` when no drag is in progress.
        drag_offset (QtCore.QPoint): The offset from the card's top-left corner
            to the point where the user grabbed it, used to keep the floating
            card anchored under the cursor rather than snapping its corner to
            the mouse position.
    """

    def __init__(self, parent=None):
        """Initialise the container layout and drag-state attributes.

        Creates a top-aligned ``QVBoxLayout`` with 15 px margins and 10 px
        inter-card spacing.  All drag-state attributes start as ``None`` /
        zero and are populated only while a drag is active.

        Args:
            parent (QtWidgets.QWidget | None): Optional Qt parent widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(10)

        # Force items to stack at the top
        self.main_layout.setAlignment(QtCore.Qt.AlignTop)

        self.dragged_card = None
        self.placeholder = None
        self.drag_offset = QtCore.QPoint(0, 0)

    def start_drag(self, card, global_mouse_pos, offset):
        """Begin a drag operation for *card*.

        Performs the following steps in order:

        1. Records *card* as ``dragged_card`` and stores *offset*.
        2. Creates a ``QWidget`` placeholder sized to match *card* and inserts
           it into ``main_layout`` at the index *card* previously occupied.
        3. Removes *card* from the layout, reparents it to this container, and
           raises it above all sibling widgets so it floats visually.
        4. Calls ``update_drag`` with *global_mouse_pos* to snap the floating
           card to the cursor immediately on drag start.

        Args:
            card (QtWidgets.QWidget): The card widget to drag.  Must currently
                be a direct child of ``main_layout``.
            global_mouse_pos (QtCore.QPoint): The cursor position in global
                screen coordinates at the moment the drag begins.
            offset (QtCore.QPoint): The offset from *card*'s top-left corner
                to the grab point, so the card does not snap its corner to
                the cursor position.
        """
        self.dragged_card = card
        self.drag_offset = offset

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
        """Reposition the floating card and update the placeholder slot.

        Called on every ``mouseMoveEvent`` while a drag is active.  Converts
        *global_mouse_pos* to container-local coordinates, moves the floating
        card so the grab point stays under the cursor, then walks the layout
        items to determine the correct insertion index for the placeholder:

        * If the top edge of the dragged card is *above* the vertical centre
          of a sibling card, the placeholder moves to that sibling's index
          (insert before).
        * If the top edge is *below* the centre, the placeholder moves to one
          past that sibling's index (insert after).
        * The first matching condition breaks the loop; if no condition fires,
          ``new_idx`` accumulates to the last valid position.

        Does nothing when no drag is active (``dragged_card`` is ``None``).

        Args:
            global_mouse_pos (QtCore.QPoint): Current cursor position in global
                screen coordinates.
        """
        if not self.dragged_card:
            return

        # Move card
        local_pos = self.mapFromGlobal(global_mouse_pos)
        target_y = local_pos.y() - self.drag_offset.y()
        target_x = self.main_layout.contentsMargins().left()
        self.dragged_card.move(target_x, target_y)
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
            if drag_focus_y < w_center_y:
                if i < placeholder_idx:
                    new_idx = i
                else:
                    new_idx = i
                break
            else:
                if i >= new_idx:
                    new_idx = i + 1

        # Apply Move
        if new_idx != placeholder_idx:
            new_idx = max(0, min(new_idx, count))
            self.main_layout.takeAt(placeholder_idx)
            self.main_layout.insertWidget(new_idx, self.placeholder)

    def finish_drag(self):
        """Commit the drag, remove the placeholder, and restore the card to the layout.

        Reads the placeholder's current layout index as the final insertion
        position, removes and destroys the placeholder, then reinserts
        ``dragged_card`` at that index.  Clears ``dragged_card`` and
        ``placeholder`` to signal that no drag is in progress.

        Does nothing when no drag is active (``dragged_card`` is ``None``).
        """
        if not self.dragged_card:
            return

        final_idx = self.main_layout.indexOf(self.placeholder)
        self.main_layout.takeAt(final_idx)
        self.placeholder.deleteLater()
        self.placeholder = None

        self.main_layout.insertWidget(final_idx, self.dragged_card)
        self.dragged_card = None
