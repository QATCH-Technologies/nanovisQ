from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

# Persistent rounded glass styling shared by every AnimatedComboBox in the app.
# Critically, this hides the NATIVE drop-down arrow/box so the only chevron
# shown is the custom spinning one (arrow_lbl). Without this, the combo renders
# two arrows.
_COMBO_GLASS_QSS = """
    QComboBox {
        background: rgba(255, 255, 255, 150);
        border: 1px solid rgba(120, 130, 145, 150);
        border-radius: 14px;
        padding-left: 14px;
        padding-right: 30px;          /* room for the custom chevron */
        color: rgb(40, 50, 62);
        font-weight: bold;
        min-height: 26px;
    }
    QComboBox:hover {
        background: rgba(255, 255, 255, 200);
        border: 1px solid rgba(90, 100, 115, 190);
    }
    QComboBox:on,
    QComboBox:focus {
        background: rgba(255, 255, 255, 225);
        border: 1px solid rgba(10, 163, 230, 200);
    }
    /* Kill the native drop-down region + arrow (custom chevron replaces it). */
    QComboBox::drop-down {
        border: none;
        background: transparent;
        width: 0px;
    }
    QComboBox::down-arrow {
        image: none;
        width: 0px;
        height: 0px;
    }
    /* Pop-up list — transparent so the rounded container shape shows cleanly.
       The container (QComboBoxPrivateContainer) paints the solid rounded
       background; the view only styles its rows. */
    QComboBox QAbstractItemView {
        background: transparent;
        border: none;
        color: rgb(40, 50, 62);
        padding: 4px;
        selection-background-color: rgba(10, 163, 230, 40);
        selection-color: #0AA3E6;
        outline: none;
    }
    QComboBox QAbstractItemView::item {
        min-height: 24px;
        border-radius: 6px;
        padding-left: 8px;
    }
"""


class AnimatedComboBox(QtWidgets.QComboBox):
    """A QComboBox that smoothly spins its drop-down arrow on open/close.

    The native arrow is suppressed via QSS; the only visible chevron is the
    custom ``arrow_lbl`` pixmap, which rotates 180 degrees when the popup opens.
    Rounded glass styling is applied automatically and persistently so every
    instance matches the rest of the app.
    """

    def __init__(self, icon_path: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setView(QtWidgets.QListView())
        self.setStyleSheet(_COMBO_GLASS_QSS)
        self.setCursor(QtCore.Qt.PointingHandCursor)

        # Pre-configure the popup container ONCE, now, before it is ever shown.
        # Doing this lazily inside showPopup() (after super().showPopup() has
        # already shown + sized the container) causes first-open artifacts: a
        # transient scrollbar and an un-rounded corner, because the flags/QSS
        # land after the first paint. Styling it up front avoids that.
        self._container = self.view().parentWidget()
        self._style_popup_container()

        # --- Arrow Icon Setup ---
        self.arrow_lbl = QtWidgets.QLabel(self)
        self.arrow_lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.arrow_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.arrow_lbl.setStyleSheet("background: transparent; border: none;")

        self._base_pixmap = QtGui.QPixmap(icon_path).scaled(
            11, 11, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.arrow_lbl.setPixmap(self._base_pixmap)

        # --- Animation Setup ---
        self._anim = QtCore.QVariantAnimation(self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._on_spin_frame)
        self._current_angle = 0.0

    def _style_popup_container(self) -> None:
        """Apply frameless flags + rounded opaque QSS to the popup container.

        Idempotent. The list view itself is also forced to never show a
        scrollbar frame artifact on first open.
        """
        container = getattr(self, "_container", None) or self.view().parentWidget()
        if container is None:
            return
        self._container = container
        container.setWindowFlags(
            QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint | QtCore.Qt.NoDropShadowWindowHint
        )
        container.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
        container.setStyleSheet("""
            QComboBoxPrivateContainer {
                background-color: rgb(245, 247, 250);
                border: 1px solid rgba(180, 192, 205, 220);
                border-radius: 10px;
            }
            """)
        container.setContentsMargins(1, 1, 1, 1)

        # Avoid the first-open scrollbar flash: let the view auto-size and only
        # scroll when genuinely needed, with no per-pixel partial row.
        view = self.view()
        view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        view.setResizeMode(QtWidgets.QListView.Adjust)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.arrow_lbl.setGeometry(self.width() - 28, 0, 24, self.height())

    def showPopup(self) -> None:
        # Re-assert container styling before showing so the very first open is
        # already correct, then show, then force a clean re-layout so the
        # rounded corners and content size settle without a second open.
        self._style_popup_container()
        super().showPopup()

        container = getattr(self, "_container", None)
        if container is not None:
            container.updateGeometry()
            container.update()
            # Nudge the layout once on the next event-loop tick to clear any
            # stale first-paint geometry, then clip to a rounded mask so the
            # scrollbar corner (a square child) can't square off the corners.
            QtCore.QTimer.singleShot(0, self._apply_popup_mask)

        self._spin_to(180.0)

    def _apply_popup_mask(self) -> None:
        """Clip the popup container to a rounded rect.

        QSS ``border-radius`` does not clip child widgets (notably the
        scrollbar corner), which leaves a squared-off corner. A region mask
        forces the actual window outline to follow the rounded shape.
        """
        container = getattr(self, "_container", None)
        if container is None or not container.isVisible():
            return
        radius = 10
        rect = container.rect()
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(rect), radius, radius)
        region = QtGui.QRegion(path.toFillPolygon().toPolygon())
        container.setMask(region)
        container.update()

    def hidePopup(self) -> None:
        super().hidePopup()
        self._spin_to(0.0)

    def _spin_to(self, target_angle: float) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._current_angle)
        self._anim.setEndValue(target_angle)
        self._anim.start()

    def _on_spin_frame(self, angle: float) -> None:
        self._current_angle = angle
        transform = QtGui.QTransform().rotate(angle)
        rotated = self._base_pixmap.transformed(transform, QtCore.Qt.SmoothTransformation)
        self.arrow_lbl.setPixmap(rotated)
