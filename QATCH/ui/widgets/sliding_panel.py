from typing import List, Optional

from PyQt5 import QtCore, QtWidgets


class SlidingPanel(QtWidgets.QWidget):
    """A horizontal sliding container that hosts multiple fixed-width pages.

    This widget implements a simple page-based carousel by arranging child
    widgets side-by-side inside an internal container and animating its
    horizontal position.

    Each page has a fixed width, and navigation is achieved by shifting the
    inner container left/right using a QPropertyAnimation on its position.

    Attributes:
        _pw: Fixed width of each page.
        _anim: Active page transition animation, if any.
        _pages: List of registered page widgets.
        _inner: Internal container widget that holds all pages.
    """

    def __init__(self, page_width: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """Initializes the sliding panel with a fixed page width.

        Args:
            page_width: Width in pixels assigned to each page.
            parent: Optional parent widget.
        """
        super().__init__(parent)

        self._pw = page_width
        self._anim: Optional[QtCore.QPropertyAnimation] = None
        self._pages: List[QtWidgets.QWidget] = []

        self.setFixedWidth(page_width)
        self.setContentsMargins(0, 0, 0, 0)
        self.setAutoFillBackground(False)

        self._inner = QtWidgets.QWidget(self)
        self._inner.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground,
            True,
        )
        self._inner.setAutoFillBackground(False)
        self._inner.move(0, 0)

    def add_page(self, widget: QtWidgets.QWidget) -> int:
        """Adds a page to the sliding panel.

        Pages are positioned horizontally in sequence inside the internal
        container.

        Args:
            widget: The page widget to add.

        Returns:
            The index of the newly added page.
        """
        idx = len(self._pages)

        widget.setParent(self._inner)
        widget.setFixedWidth(self._pw)
        widget.move(idx * self._pw, 0)

        self._pages.append(widget)
        return idx

    def finalize(self, fallback_height: int = 300) -> None:
        """Finalizes layout sizing after all pages have been added.

        This ensures each page fills the vertical space and the internal
        container is sized to accommodate all pages horizontally.

        Args:
            fallback_height: Height to use if the widget has not been laid out
                yet and reports a zero height.
        """
        if not self._pages:
            return

        h = self.height() if self.height() > 0 else fallback_height

        # Ensure every page fills the vertical space and is correctly aligned.
        for i, p in enumerate(self._pages):
            p.setFixedSize(self._pw, h)
            p.move(i * self._pw, 0)

        # Expand inner container to fit all pages horizontally.
        self._inner.setFixedSize(len(self._pages) * self._pw, h)

    def slide_to(self, page_idx: int, duration: int = 360) -> None:
        """Animates the panel to the specified page index.

        Args:
            page_idx: Target page index to display.
            duration: Duration (ms) of the slide animation.

        Notes:
            - Cancels any in-progress animation before starting a new one.
            - Uses easing for smooth in/out motion (InOutQuart).
            - Animates the internal container's position.
        """
        if page_idx >= len(self._pages) or page_idx < 0:
            return

        target_x = -page_idx * self._pw

        if self._anim is not None:
            try:
                if self._anim.state() == QtCore.QPropertyAnimation.State.Running:
                    self._anim.stop()
            except RuntimeError:
                pass
            self._anim = None

        self._anim = QtCore.QPropertyAnimation(self._inner, b"pos")
        self._anim.setStartValue(self._inner.pos())
        self._anim.setEndValue(QtCore.QPoint(target_x, 0))
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)

        # Allow Qt to clean up the animation object once finished.
        self._anim.start(QtCore.QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
