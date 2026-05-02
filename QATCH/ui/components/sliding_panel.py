from typing import List, Optional
from PyQt5 import QtCore, QtWidgets


class SlidingPanel(QtWidgets.QWidget):
    """Two-page sliding container for the sign-in ↔ recover-password transition.

    The widget's own width is fixed to *page_width* so Qt automatically clips
    any child content that extends beyond that boundary.  Both pages are placed
    side-by-side inside an inner container; animating that container's ``pos``
    property slides between them without any off-screen content leaking through.

    Typical usage::

        slider = _SlidingPanel(page_width=290)
        slider.add_page(sign_in_widget)
        slider.add_page(recover_widget)
        # After all pages have been added and their layouts are populated:
        QtCore.QTimer.singleShot(0, slider.finalize)
        # Later, to animate:
        slider.slide_to(1)   # → recover
        slider.slide_to(0)   # → sign-in
    """

    def __init__(self, page_width: int, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pw = page_width
        self._anim: Optional[QtCore.QPropertyAnimation] = None

        # Fixed width causes Qt to clip children that extend past the right edge.
        self.setFixedWidth(page_width)
        self.setContentsMargins(0, 0, 0, 0)
        # Prevent this widget from painting its own opaque background, which
        # would cover the GlassCard's custom-painted glass effect below it.
        self.setAutoFillBackground(False)

        # Inner host — spans (n_pages × page_width) horizontally.
        self._inner = QtWidgets.QWidget(self)
        self._inner.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._inner.move(0, 0)
        self._inner.setContentsMargins(0, 0, 0, 0)
        # Same transparency requirement: _inner must not paint a background or
        # the second page will appear as an opaque white slab to the right.
        self._inner.setAutoFillBackground(False)
        self._pages: List[QtWidgets.QWidget] = []

    # ── public ────────────────────────────────────────────────────────────────
    def add_page(self, widget: QtWidgets.QWidget) -> int:
        """Reparent *widget* into the inner container and return its page index."""
        idx = len(self._pages)
        widget.setParent(self._inner)
        widget.setFixedWidth(self._pw)
        widget.move(idx * self._pw, 0)
        self._pages.append(widget)
        return idx

    def finalize(self, fallback_height: int = 300) -> None:
        """Size pages and inner container to match the slider's fixed height.

        setFixedHeight must be called on the slider *before* this runs so the
        loginCard renders at the correct size on first paint.  finalize only
        sizes the inner host and each page to match that height so the slide
        animation works correctly.
        """
        if not self._pages:
            return
        h = self.height() if self.height() > 0 else fallback_height
        for i, p in enumerate(self._pages):
            p.setFixedSize(self._pw, h)
            p.move(i * self._pw, 0)
        self._inner.setFixedSize(len(self._pages) * self._pw, h)

    def slide_to(self, page_idx: int, duration: int = 360) -> None:
        """Animate the inner container to reveal *page_idx*."""
        target_x = -page_idx * self._pw
        anim = QtCore.QPropertyAnimation(self._inner, b"pos")
        anim.setStartValue(self._inner.pos())
        anim.setEndValue(QtCore.QPoint(target_x, 0))
        anim.setDuration(duration)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        anim.start()
        self._anim = anim  # keep reference to prevent GC mid-animation
