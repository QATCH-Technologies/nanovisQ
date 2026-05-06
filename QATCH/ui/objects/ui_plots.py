"""
QATCH.ui.mainWindow_ui_plots

Three-tile drag-to-snap plot layout with Windows 11-style Snap Assist.

──────────────────────────────────────────────────────────────────────
Default layout
──────────────────────────────────────────────────────────────────────
┌──────────────────────┬─────────────┐
│                      │  Amplitude  │
│  Resonance /         ├─────────────┤
│  Dissipation         │ Temperature │
└──────────────────────┴─────────────┘

──────────────────────────────────────────────────────────────────────
Interaction model  (Win11-authentic two-phase snap)
──────────────────────────────────────────────────────────────────────
Phase 1 — Drag
  • Grab any tile by its header bar.
  • Edge zones (left / right / top / bottom) and an equal-column center
    zone light up as the cursor enters them.
  • A ghost silhouette shows where the tile WILL land.

Phase 2 — Snap & Assist
  • Release in an edge zone → the tile snaps there immediately.
  • A "Snap Assist" panel materialises in the *remaining* space,
    showing the other tiles as large clickable cards (exactly as Win11
    shows open-window thumbnails after snapping).
  • Clicking a card fixes that tile in the top/left sub-slot;
    the last tile fills the remaining sub-slot automatically.
  • The assist panel auto-dismisses after 4 s or on outside click.
  • Releasing in the CENTER zone skips assist and gives equal columns.

Other controls
  • ✕ on any header hides that tile (data is never interrupted).
  • "⊕ Restore" button (top-right) brings hidden tiles back.

──────────────────────────────────────────────────────────────────────
Backward-compat widget names
──────────────────────────────────────────────────────────────────────
  ui2.plt      — Amplitude            GraphicsLayoutWidget (unchanged)
  ui2.plt_temp — Temperature          GraphicsLayoutWidget (new)
  ui2.pltB     — Resonance/Dissipation GraphicsLayoutWidget (unchanged)

Author(s)
    Alexander Ross  (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)

Date:
    2026-05-06
"""

from __future__ import annotations

import copy, os
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget
from pyqtgraph import GraphicsLayoutWidget

from QATCH.common.architecture import Architecture
from QATCH.core.constants import Constants
from QATCH.ui.popUp import PopUp

# ============================================================
#  Tile identifiers and metadata
# ============================================================

AMP_TILE = "amp"
TEMP_TILE = "temp"
RFD_TILE = "rfd"

_TILE_TITLES: Dict[str, str] = {
    AMP_TILE: "Amplitude",
    TEMP_TILE: "Temperature",
    RFD_TILE: "Resonance / Dissipation",
}

_TILE_ACCENT: Dict[str, QtGui.QColor] = {
    AMP_TILE: QtGui.QColor(46, 155, 218),  # blue
    TEMP_TILE: QtGui.QColor(240, 156, 53),  # amber
    RFD_TILE: QtGui.QColor(72, 190, 120),  # green
}

# Zone opposite mapping used when auto-filling remaining tiles
_OPPOSITE: Dict[str, str] = {
    "left": "right",
    "right": "left",
    "top": "bottom",
    "bottom": "top",
    "center": "center",
}

# ============================================================
#  Layout constants
# ============================================================

_GAP = 6  # px gap between tiles
_HEADER_H = 28  # px title-bar height
_EDGE_FRAC = 0.28  # fraction of container that is an "edge zone"
_LARGE_FRAC = 0.58  # fraction given to the snapped (focus) tile
_DRAG_THRESH = 8  # px movement before drag activates


# ============================================================
#  _Slot  — lightweight layout descriptor per tile
# ============================================================


class _Slot:
    """Records which zone a tile occupies and its rank within that zone.

    zone  : "left" | "right" | "top" | "bottom" | "center"
    rank  : 0 = occupies the zone solo; 1 = first of a stacked pair;
            2 = second of a stacked pair; 3 = third (equal columns only)
    """

    __slots__ = ("zone", "rank")

    def __init__(self, zone: str = "center", rank: int = 1) -> None:
        self.zone = zone
        self.rank = rank

    def copy(self) -> "_Slot":
        return _Slot(self.zone, self.rank)

    def __repr__(self) -> str:
        return f"_Slot({self.zone!r}, {self.rank})"


# Default layout: Resonance/Diss on the left, Amp+Temp stacked right
_DEFAULT_SLOTS: Dict[str, _Slot] = {
    RFD_TILE: _Slot("left", 0),
    AMP_TILE: _Slot("right", 1),
    TEMP_TILE: _Slot("right", 2),
}


# ============================================================
#  Pure geometry helpers
# ============================================================


def _stack_v(tiles: List[str], x: int, y: int, w: int, h: int, g: int) -> Dict[str, QtCore.QRect]:
    """Stack tiles vertically (equal height) inside (x,y,w,h)."""
    n = len(tiles)
    if n == 0:
        return {}
    if n == 1:
        return {tiles[0]: QtCore.QRect(x, y, w, h)}
    unit = (h - (n - 1) * g) // n
    out = {}
    for i, tid in enumerate(tiles):
        ty = y + i * (unit + g)
        th = h - (n - 1) * (unit + g) if i == n - 1 else unit
        out[tid] = QtCore.QRect(x, ty, w, max(th, 1))
    return out


def _stack_h(tiles: List[str], x: int, y: int, w: int, h: int, g: int) -> Dict[str, QtCore.QRect]:
    """Stack tiles horizontally (equal width) inside (x,y,w,h)."""
    n = len(tiles)
    if n == 0:
        return {}
    if n == 1:
        return {tiles[0]: QtCore.QRect(x, y, w, h)}
    unit = (w - (n - 1) * g) // n
    out = {}
    for i, tid in enumerate(tiles):
        tx = x + i * (unit + g)
        tw = w - (n - 1) * (unit + g) if i == n - 1 else unit
        out[tid] = QtCore.QRect(tx, y, max(tw, 1), h)
    return out


def _compute_rects(
    visible: List[str],
    slots: Dict[str, _Slot],
    w: int,
    h: int,
    g: int,
) -> Dict[str, QtCore.QRect]:
    """Compute pixel QRect for every tile in *visible* given slot assignments.

    Groups tiles by zone, then positions each group using the appropriate
    stacking helper.  The "focus" tile (rank 0 in left/right/top/bottom)
    receives ``_LARGE_FRAC`` of the container's relevant dimension.
    """
    n = len(visible)
    if n == 0:
        return {}
    if n == 1:
        return {visible[0]: QtCore.QRect(0, 0, w, h)}

    def by_zone(z: str) -> List[str]:
        return sorted(
            [t for t in visible if slots.get(t, _Slot()).zone == z],
            key=lambda t: slots.get(t, _Slot()).rank,
        )

    left = by_zone("left")
    right = by_zone("right")
    top = by_zone("top")
    bottom = by_zone("bottom")
    center = by_zone("center")

    # ---- Left / Right arrangement ----------------------------------------
    if left or right:
        if left and right:
            lw = max(int(w * _LARGE_FRAC), 1)
            rw = max(w - lw - g, 1)
            return {
                **_stack_v(left, 0, 0, lw, h, g),
                **_stack_v(right, lw + g, 0, rw, h, g),
            }
        # Solo zone — fill everything
        all_lr = left or right
        return _stack_v(all_lr, 0, 0, w, h, g)

    # ---- Top / Bottom arrangement ----------------------------------------
    if top or bottom:
        if top and bottom:
            th = max(int(h * _LARGE_FRAC), 1)
            bh = max(h - th - g, 1)
            return {
                **_stack_h(top, 0, 0, w, th, g),
                **_stack_h(bottom, 0, th + g, w, bh, g),
            }
        all_tb = top or bottom
        return _stack_h(all_tb, 0, 0, w, h, g)

    # ---- Center: equal columns -------------------------------------------
    return _stack_h(center or visible, 0, 0, w, h, g)


# ============================================================
#  PlotsWindow  (outer QMainWindow shell — API unchanged)
# ============================================================


class PlotsWindow(QtWidgets.QMainWindow):
    def __init__(self, samples=Constants.argument_default_samples):
        super().__init__()
        self.ui2 = UIPlots()
        self.ui2.setupUi(self)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        res = PopUp.question(
            self,
            Constants.app_title,
            "Are you sure you want to quit QATCH Q-1 application now?",
            True,
        )
        if res:
            QtWidgets.QApplication.quit()
        else:
            event.ignore()


# ============================================================
#  GlassPlotPanel
# ============================================================


class GlassPlotPanel(QtWidgets.QWidget):
    """Frosted-glass card wrapping a pyqtgraph GraphicsLayoutWidget."""

    _R = 10.0
    _M = 3

    def __init__(
        self, plot_widget: GraphicsLayoutWidget, parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(self._M, self._M, self._M, self._M)
        lay.setSpacing(0)
        lay.addWidget(plot_widget)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        rf = QtCore.QRectF(self.rect())
        path = QtGui.QPainterPath()
        path.addRoundedRect(rf, self._R, self._R)
        p.setClipPath(path)
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))
        sh = QtGui.QLinearGradient(0, 0, 0, 32)
        sh.setColorAt(0, QtGui.QColor(255, 255, 255, 50))
        sh.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(sh))
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rf.adjusted(0.5, 0.5, -0.5, -0.5), self._R, self._R)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 80), 1.0))
        p.drawRoundedRect(rf.adjusted(1.5, 1.5, -1.5, -1.5), self._R - 1.5, self._R - 1.5)
        p.end()


# ============================================================
#  _TileHeader  — draggable glass title bar
# ============================================================


class _TileHeader(QtWidgets.QWidget):
    closeClicked = QtCore.pyqtSignal()

    def __init__(self, tile_id: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._accent = _TILE_ACCENT.get(tile_id, QtGui.QColor(46, 155, 218))
        self.setFixedHeight(_HEADER_H)
        self.setCursor(QtGui.QCursor(QtCore.Qt.SizeAllCursor))

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(8, 0, 4, 0)
        row.setSpacing(5)

        grip = QtWidgets.QLabel("⠿")
        grip.setStyleSheet("color:rgba(30,40,55,100);font-size:13px;")
        grip.setFixedWidth(14)
        row.addWidget(grip)

        a = self._accent
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet(f"color:rgba({a.red()},{a.green()},{a.blue()},220);font-size:9px;")
        dot.setFixedWidth(10)
        row.addWidget(dot)

        lbl = QtWidgets.QLabel(_TILE_TITLES.get(tile_id, tile_id))
        lbl.setStyleSheet("color:rgba(30,40,55,200);font-size:10px;font-weight:600;")
        row.addWidget(lbl, 1)

        btn = QtWidgets.QToolButton()
        btn.setText("✕")
        btn.setFixedSize(18, 18)
        btn.setStyleSheet("""
            QToolButton{background:transparent;border:none;
                        color:rgba(30,40,55,110);font-size:9px;border-radius:9px;}
            QToolButton:hover{background:rgba(210,50,50,210);color:white;}
            QToolButton:pressed{background:rgba(180,30,30,230);color:white;}
        """)
        btn.setToolTip("Hide this tile (data acquisition continues)")
        btn.clicked.connect(self.closeClicked)
        row.addWidget(btn)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 80))
        a = self._accent
        p.setPen(QtGui.QPen(QtGui.QColor(a.red(), a.green(), a.blue(), 55), 1.0))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        p.end()


# ============================================================
#  DraggableTilePanel
# ============================================================


class DraggableTilePanel(QtWidgets.QWidget):
    """Plot tile with a draggable glass header and close (✕) button."""

    dragStarted = QtCore.pyqtSignal(object, QtCore.QPoint)
    dragMoved = QtCore.pyqtSignal(object, QtCore.QPoint)
    dragEnded = QtCore.pyqtSignal(object, QtCore.QPoint)
    closeRequested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        tile_id: str,
        plot_widget: GraphicsLayoutWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._tile_id = tile_id
        self._drag_start: Optional[QtCore.QPoint] = None
        self._in_drag = False

        vb = QtWidgets.QVBoxLayout(self)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(0)

        self._header = _TileHeader(tile_id, self)
        self._header.closeClicked.connect(lambda: self.closeRequested.emit(tile_id))
        vb.addWidget(self._header)

        self._glass = GlassPlotPanel(plot_widget, self)
        vb.addWidget(self._glass, 1)

        self._header.setMouseTracking(True)
        self._header.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if obj is not self._header:
            return super().eventFilter(obj, ev)
        t = ev.type()
        if t == QtCore.QEvent.MouseButtonPress and ev.button() == QtCore.Qt.LeftButton:
            self._drag_start = ev.globalPos()
            self._in_drag = False
        elif t == QtCore.QEvent.MouseMove and self._drag_start is not None:
            d = ev.globalPos() - self._drag_start
            if not self._in_drag and (d.x() ** 2 + d.y() ** 2) ** 0.5 > _DRAG_THRESH:
                self._in_drag = True
                self.dragStarted.emit(self, ev.globalPos())
            elif self._in_drag:
                self.dragMoved.emit(self, ev.globalPos())
        elif t == QtCore.QEvent.MouseButtonRelease and ev.button() == QtCore.Qt.LeftButton:
            if self._in_drag:
                self.dragEnded.emit(self, ev.globalPos())
            self._drag_start = None
            self._in_drag = False
        return super().eventFilter(obj, ev)

    @property
    def tile_id(self) -> str:
        return self._tile_id

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 55), 1.0))
        p.drawRoundedRect(QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10.0, 10.0)
        p.end()


# ============================================================
#  _SnapZoneOverlay  — shown DURING drag
# ============================================================


class _SnapZoneOverlay(QtWidgets.QWidget):
    """Highlights the five snap zones while a tile is being dragged.

    Zone map
    ────────
       ┌──────────────────────┐
       │         TOP          │
       ├──────┬───────┬───────┤
       │ LEFT │  CTR  │ RIGHT │
       ├──────┴───────┴───────┤
       │        BOTTOM        │
       └──────────────────────┘
    """

    _FILL = QtGui.QColor(46, 155, 218, 24)
    _ACTIVE = QtGui.QColor(46, 155, 218, 75)
    _GHOST = QtGui.QColor(46, 155, 218, 18)
    _GSTROKE = QtGui.QColor(46, 155, 218, 150)

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self._active: Optional[str] = None
        self._preview: Dict[str, QtCore.QRect] = {}
        self.hide()

    # ──────────────────────────────────────────

    @staticmethod
    def zone_rects(w: int, h: int) -> Dict[str, QtCore.QRect]:
        ew, eh = int(w * _EDGE_FRAC), int(h * _EDGE_FRAC)
        return {
            "top": QtCore.QRect(0, 0, w, eh),
            "bottom": QtCore.QRect(0, h - eh, w, eh),
            "left": QtCore.QRect(0, 0, ew, h),
            "right": QtCore.QRect(w - ew, 0, ew, h),
            "center": QtCore.QRect(ew, eh, w - 2 * ew, h - 2 * eh),
        }

    def zone_at(self, local: QtCore.QPoint) -> str:
        zr = self.zone_rects(self.width(), self.height())
        for name in ("top", "bottom", "left", "right", "center"):
            if zr[name].contains(local):
                return name
        return "center"

    def set_state(self, active: Optional[str], preview: Dict[str, QtCore.QRect]) -> None:
        self._active = active
        self._preview = preview
        self.update()

    # ──────────────────────────────────────────

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        zr = self.zone_rects(self.width(), self.height())
        for name, rect in zr.items():
            act = name == self._active
            p.fillRect(rect, self._ACTIVE if act else self._FILL)
            if act:
                p.setPen(QtGui.QPen(QtGui.QColor(46, 155, 218, 140), 1.5))
                p.setBrush(QtCore.Qt.NoBrush)
                p.drawRect(rect.adjusted(1, 1, -1, -1))

            # Zone label
            p.setPen(QtGui.QPen(QtGui.QColor(46, 155, 218, 75 if act else 40)))
            f = p.font()
            f.setPointSize(8)
            f.setBold(act)
            p.setFont(f)
            p.drawText(rect, QtCore.Qt.AlignCenter, name.upper())

        # Ghost preview
        for tid, rect in self._preview.items():
            rf = QtCore.QRectF(rect).adjusted(4, 4, -4, -4)
            p.fillRect(rect, self._GHOST)
            p.setPen(QtGui.QPen(self._GSTROKE, 1.5, QtCore.Qt.DashLine))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRoundedRect(rf, 8.0, 8.0)
            accent = _TILE_ACCENT.get(tid, QtGui.QColor(46, 155, 218))
            p.setPen(QtGui.QPen(QtGui.QColor(accent.red(), accent.green(), accent.blue(), 180)))
            f2 = p.font()
            f2.setPointSize(9)
            f2.setBold(True)
            p.setFont(f2)
            p.drawText(rf, QtCore.Qt.AlignCenter, _TILE_TITLES.get(tid, tid))

        p.end()


# ============================================================
#  _SnapAssistCard  — one clickable tile card inside the assist panel
# ============================================================


class _SnapAssistCard(QtWidgets.QAbstractButton):
    """Clickable glass card representing a tile in the Snap Assist panel."""

    def __init__(self, tile_id: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._tile_id = tile_id
        self._accent = _TILE_ACCENT.get(tile_id, QtGui.QColor(46, 155, 218))
        self._hovered = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setToolTip(f"Place {_TILE_TITLES.get(tile_id, tile_id)} here first")

    @property
    def tile_id(self) -> str:
        return self._tile_id

    def enterEvent(self, ev) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(ev)

    def leaveEvent(self, ev) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(ev)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rf = QtCore.QRectF(self.rect()).adjusted(3, 3, -3, -3)
        a = self._accent
        alpha = 90 if self._hovered else 45
        p.fillRect(self.rect(), QtGui.QColor(a.red(), a.green(), a.blue(), alpha))
        pen_a = 220 if self._hovered else 130
        p.setPen(QtGui.QPen(QtGui.QColor(a.red(), a.green(), a.blue(), pen_a), 2.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(rf, 10.0, 10.0)
        p.setPen(QtGui.QPen(QtGui.QColor(a.red(), a.green(), a.blue(), 230)))
        f = p.font()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.drawText(rf, QtCore.Qt.AlignCenter, _TILE_TITLES.get(self._tile_id, self._tile_id))
        p.end()


# ============================================================
#  _SnapAssistOverlay  — shown AFTER snap (Win11 Snap Assist)
# ============================================================


class _SnapAssistOverlay(QtWidgets.QWidget):
    """Win11-style Snap Assist panel shown in the remaining space after snap.

    Displays the remaining (un-snapped) tiles as large clickable cards.
    Clicking a card promotes it to the first sub-slot; the other tile
    automatically fills the second sub-slot.  Auto-dismisses after 4 s.
    """

    tileChosen = QtCore.pyqtSignal(str)  # tile_id promoted to rank 1
    dismissed = QtCore.pyqtSignal()

    _AUTO_DISMISS_MS = 4000

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self._cards: List[_SnapAssistCard] = []

        # Auto-dismiss timer
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.dismiss)

        # Header label
        self._header_lbl = QtWidgets.QLabel("Choose where to place:", self)
        self._header_lbl.setStyleSheet(
            "color:rgba(30,40,55,180);font-size:10px;font-weight:700;" "background:transparent;"
        )
        self._header_lbl.setAlignment(QtCore.Qt.AlignCenter)

        # Dismiss button
        self._close = QtWidgets.QToolButton(self)
        self._close.setText("✕")
        self._close.setFixedSize(22, 22)
        self._close.setStyleSheet("""
            QToolButton{background:rgba(255,255,255,120);border:none;
                        color:rgba(30,40,55,150);font-size:10px;border-radius:11px;}
            QToolButton:hover{background:rgba(210,50,50,200);color:white;}
        """)
        self._close.setToolTip("Keep current order and dismiss")
        self._close.clicked.connect(self.dismiss)

        self.hide()

    # ──────────────────────────────────────────

    def show_for(self, available_rect: QtCore.QRect, tile_ids: List[str]) -> None:
        """Populate cards and show the assist panel inside *available_rect*."""
        # Remove old cards
        for c in self._cards:
            c.deleteLater()
        self._cards.clear()

        for tid in tile_ids:
            card = _SnapAssistCard(tid, self)
            card.clicked.connect(lambda _, t=tid: self._on_card(t))
            self._cards.append(card)

        self.setGeometry(available_rect)
        self._relayout()
        self.show()
        self.raise_()
        self._timer.start(self._AUTO_DISMISS_MS)

    def dismiss(self) -> None:
        self._timer.stop()
        self.hide()
        self.dismissed.emit()

    def _on_card(self, tile_id: str) -> None:
        self._timer.stop()
        self.hide()
        self.tileChosen.emit(tile_id)

    # ──────────────────────────────────────────

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self) -> None:
        """Position the dismiss button, header, and cards."""
        w, h = self.width(), self.height()
        pad = 12
        hdr_h = 22
        btn_s = 22

        self._close.move(w - btn_s - 6, 6)
        self._header_lbl.setGeometry(pad, pad, w - 2 * pad, hdr_h)

        n = len(self._cards)
        if n == 0:
            return
        card_h = max((h - pad - hdr_h - pad - (n - 1) * _GAP) // n, 40)
        for i, card in enumerate(self._cards):
            cy = pad + hdr_h + _GAP + i * (card_h + _GAP)
            card.setGeometry(pad, cy, w - 2 * pad, card_h)

    # ──────────────────────────────────────────

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rf = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)

        # Frosted background
        p.fillRect(self.rect(), QtGui.QColor(240, 246, 252, 210))
        # Border
        p.setPen(QtGui.QPen(QtGui.QColor(46, 155, 218, 150), 1.5))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(rf, 12.0, 12.0)

        # "Snap Assist" watermark label at top
        p.setPen(QtGui.QPen(QtGui.QColor(46, 155, 218, 120)))
        f = p.font()
        f.setPointSize(7)
        f.setBold(False)
        p.setFont(f)
        p.drawText(
            QtCore.QRectF(0, self.height() - 18, self.width(), 16),
            QtCore.Qt.AlignCenter,
            "Snap Assist  •  auto-dismisses in 4 s",
        )
        p.end()


# ============================================================
#  _DragGhost  — floating silhouette during drag
# ============================================================


class _DragGhost(QtWidgets.QWidget):
    def __init__(self, tile_id: str, size: QtCore.QSize, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self._accent = _TILE_ACCENT.get(tile_id, QtGui.QColor(46, 155, 218))
        self._title = _TILE_TITLES.get(tile_id, tile_id)
        self.resize(size)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        rf = QtCore.QRectF(self.rect()).adjusted(3, 3, -3, -3)
        a = self._accent
        p.fillRect(self.rect(), QtGui.QColor(a.red(), a.green(), a.blue(), 28))
        p.setPen(QtGui.QPen(QtGui.QColor(a.red(), a.green(), a.blue(), 160), 2.0))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(rf, 10.0, 10.0)
        p.setPen(QtGui.QPen(QtGui.QColor(a.red(), a.green(), a.blue(), 200)))
        f = p.font()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.drawText(rf, QtCore.Qt.AlignCenter, self._title)
        p.end()


# ============================================================
#  TileGrid  — the main container
# ============================================================


class TileGrid(QtWidgets.QWidget):
    """Manages three draggable plot tiles with Win11-style two-phase snap.

    Phase 1 — Drag: ``_SnapZoneOverlay`` shows drop targets.
    Phase 2 — Snap Assist: ``_SnapAssistOverlay`` lets the user set
    the stacking order of remaining tiles in the available zone.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._tiles: Dict[str, DraggableTilePanel] = {}
        self._order: List[str] = []  # registration order
        self._hidden: List[str] = []
        self._slots: Dict[str, _Slot] = {}  # drives geometry
        self._dragging: Optional[DraggableTilePanel] = None
        self._ghost: Optional[_DragGhost] = None

        self._zone_overlay = _SnapZoneOverlay(self)
        self._assist_overlay = _SnapAssistOverlay(self)
        self._assist_overlay.tileChosen.connect(self._on_assist_chosen)
        self._assist_overlay.dismissed.connect(self._layout_tiles)

        # Restore button
        self._restore_btn = QtWidgets.QToolButton(self)
        self._restore_btn.setText("⊕ Restore Tile")
        self._restore_btn.setStyleSheet("""
            QToolButton{
                background:rgba(255,255,255,160);color:rgba(30,40,55,200);
                border:1px solid rgba(200,210,220,160);border-radius:6px;
                padding:5px 10px;font-size:10px;font-weight:600;}
            QToolButton:hover{
                background:rgba(46,155,218,180);color:white;
                border-color:rgba(46,155,218,220);}
        """)
        self._restore_btn.setToolTip("Restore the last hidden tile")
        self._restore_btn.hide()
        self._restore_btn.clicked.connect(self._restore_next)

    # ── Public API ────────────────────────────────────────────────────

    def add_tile(
        self, tile_id: str, panel: DraggableTilePanel, slot: Optional[_Slot] = None
    ) -> None:
        self._tiles[tile_id] = panel
        self._order.append(tile_id)
        self._slots[tile_id] = slot or _Slot("center", len(self._order))
        panel.setParent(self)
        panel.dragStarted.connect(self._on_drag_started)
        panel.dragMoved.connect(self._on_drag_moved)
        panel.dragEnded.connect(self._on_drag_ended)
        panel.closeRequested.connect(self._on_close)

    # ── Qt events ─────────────────────────────────────────────────────

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        self._zone_overlay.setGeometry(self.rect())
        self._layout_tiles()
        self._pin_restore()

    # ── Layout ────────────────────────────────────────────────────────

    def _visible(self) -> List[str]:
        return [t for t in self._order if t not in self._hidden]

    def _layout_tiles(self) -> None:
        vis = self._visible()
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return

        rects = _compute_rects(vis, self._slots, w, h, _GAP)
        for tid, rect in rects.items():
            tile = self._tiles[tid]
            tile.setGeometry(rect)
            tile.setVisible(True)
            tile.raise_()
        for tid in self._hidden:
            self._tiles[tid].setVisible(False)

        self._zone_overlay.setGeometry(self.rect())
        self._zone_overlay.raise_()
        if self._ghost:
            self._ghost.raise_()
        self._assist_overlay.raise_()
        self._restore_btn.raise_()

    # ── Drag phase ────────────────────────────────────────────────────

    def _on_drag_started(self, tile: DraggableTilePanel, gp: QtCore.QPoint) -> None:
        # Dismiss any open assist panel
        if self._assist_overlay.isVisible():
            self._assist_overlay.dismiss()

        self._dragging = tile
        local = self.mapFromGlobal(gp)
        ghost = _DragGhost(tile.tile_id, tile.size(), self)
        ghost.move(local - QtCore.QPoint(tile.width() // 2, _HEADER_H // 2))
        ghost.show()
        ghost.raise_()
        self._ghost = ghost
        self._zone_overlay.show()
        self._zone_overlay.raise_()

    def _on_drag_moved(self, tile: DraggableTilePanel, gp: QtCore.QPoint) -> None:
        local = self.mapFromGlobal(gp)
        if self._ghost:
            self._ghost.move(local - QtCore.QPoint(self._ghost.width() // 2, _HEADER_H // 2))
            self._ghost.raise_()

        zone = self._zone_overlay.zone_at(local)
        vis = self._visible()
        tid = tile.tile_id

        # Build a preview slot-set with the dragged tile in the hovered zone
        preview_slots = {t: self._slots[t].copy() for t in vis}
        _apply_snap(tid, zone, vis, preview_slots)

        preview_rects = _compute_rects(vis, preview_slots, self.width(), self.height(), _GAP)
        self._zone_overlay.set_state(zone, preview_rects)

    def _on_drag_ended(self, tile: DraggableTilePanel, gp: QtCore.QPoint) -> None:
        local = self.mapFromGlobal(gp)
        zone = self._zone_overlay.zone_at(local)
        tid = tile.tile_id
        vis = self._visible()

        # Clean up drag visuals
        if self._ghost:
            self._ghost.deleteLater()
            self._ghost = None
        self._zone_overlay.set_state(None, {})
        self._zone_overlay.hide()
        self._dragging = None

        # Commit the snap
        _apply_snap(tid, zone, vis, self._slots)
        self._layout_tiles()

        # Show Snap Assist if we snapped to an edge zone
        if zone in ("left", "right", "top", "bottom"):
            self._show_snap_assist(tid, zone)

    # ── Snap Assist phase ─────────────────────────────────────────────

    def _show_snap_assist(self, snapped_tid: str, zone: str) -> None:
        """Show the Snap Assist panel in the available (remaining) zone."""
        vis = self._visible()
        remaining = [t for t in vis if t != snapped_tid]
        if not remaining:
            return

        # Compute the pixel rect of the "available zone"
        rects = _compute_rects(vis, self._slots, self.width(), self.height(), _GAP)
        # Determine the bounding rect of the remaining tiles
        opp_zone = _OPPOSITE.get(zone, "center")
        opp_tiles = [t for t in remaining if self._slots.get(t, _Slot()).zone == opp_zone]
        if not opp_tiles:
            opp_tiles = remaining

        # Merge their rects into one bounding rect
        merged: Optional[QtCore.QRect] = None
        for t in opp_tiles:
            r = rects.get(t)
            if r:
                merged = r if merged is None else merged.united(r)

        if merged is None:
            return

        # Inflate slightly for padding
        assist_rect = merged.adjusted(-4, -4, 4, 4)
        self._assist_overlay.show_for(assist_rect, remaining)

    def _on_assist_chosen(self, chosen_tid: str) -> None:
        """User clicked a card — promote *chosen_tid* to rank 1."""
        # Find the zone of the chosen tile
        zone = self._slots.get(chosen_tid, _Slot()).zone
        vis = self._visible()

        # Re-rank tiles in that zone: chosen → rank 1, others follow
        zone_tiles = sorted(
            [t for t in vis if self._slots.get(t, _Slot()).zone == zone],
            key=lambda t: self._slots[t].rank,
        )
        # Move chosen_tid to front
        zone_tiles.remove(chosen_tid)
        zone_tiles.insert(0, chosen_tid)
        for i, t in enumerate(zone_tiles):
            self._slots[t].rank = i + 1

        self._layout_tiles()

    # ── Close / Restore ───────────────────────────────────────────────

    def _on_close(self, tile_id: str) -> None:
        if tile_id not in self._hidden:
            self._hidden.append(tile_id)
        self._layout_tiles()
        self._restore_btn.setVisible(bool(self._hidden))
        self._pin_restore()

    def _restore_next(self) -> None:
        if self._hidden:
            self._hidden.pop(0)
        self._restore_btn.setVisible(bool(self._hidden))
        self._layout_tiles()
        self._pin_restore()

    def _pin_restore(self) -> None:
        bw = self._restore_btn.sizeHint().width() + 4
        self._restore_btn.move(self.width() - bw - 6, 6)
        self._restore_btn.raise_()


# ============================================================
#  _apply_snap  — pure slot-mutation helper
# ============================================================


def _apply_snap(
    tid: str,
    zone: str,
    visible: List[str],
    slots: Dict[str, _Slot],
) -> None:
    """Commit tile *tid* to *zone* and auto-assign the remaining tiles.

    For edge zones (left/right/top/bottom):
      • *tid* gets rank 0 in *zone* (solo in that column/row).
      • Remaining tiles fill the opposite zone, stacked in their
        current relative order (rank 1, 2, …).

    For "center":
      • All visible tiles get equal-column center slots.
    """
    remaining = [t for t in visible if t != tid]

    if zone == "center":
        for i, t in enumerate(visible):
            slots[t] = _Slot("center", i + 1)
        return

    slots[tid] = _Slot(zone, 0)
    opp = _OPPOSITE[zone]

    # Preserve the relative order of remaining tiles within the opp zone
    opp_sorted = sorted(
        remaining,
        key=lambda t: slots.get(t, _Slot(rank=99)).rank,
    )
    for i, t in enumerate(opp_sorted):
        slots[t] = _Slot(opp, i + 1)


# ============================================================
#  Shared helpers
# ============================================================


def _make_plot_widget(parent: QtWidgets.QWidget) -> GraphicsLayoutWidget:
    w = GraphicsLayoutWidget(parent)
    w.setAutoFillBackground(False)
    w.setStyleSheet("border:0px;")
    w.setFrameShape(QtWidgets.QFrame.StyledPanel)
    w.setFrameShadow(QtWidgets.QFrame.Plain)
    w.setLineWidth(0)
    w.setMinimumSize(80, 60)
    return w


# ============================================================
#  UIPlots  — consumed by PlotsWindow
# ============================================================


class UIPlots:
    """Three-tile Win11-snap plot layout.

    Default (matches the screenshots)
    ──────────────────────────────────
    ┌──────────────────────┬─────────────┐
    │                      │  Amplitude  │
    │  Resonance /         ├─────────────┤
    │  Dissipation         │ Temperature │
    └──────────────────────┴─────────────┘

    Backward-compat widget attributes
    ──────────────────────────────────
    ui2.plt      — Amplitude             GraphicsLayoutWidget
    ui2.plt_temp — Temperature           GraphicsLayoutWidget  (new)
    ui2.pltB     — Resonance+Dissipation GraphicsLayoutWidget
    """

    def setupUi(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880

        MainWindow2.setObjectName("MainWindow2")
        MainWindow2.setMinimumSize(QtCore.QSize(1000, 250))
        if USE_FULLSCREEN:
            MainWindow2.resize(1701, 1435)
            MainWindow2.move(0, 0)
        else:
            MainWindow2.move(692, 0)
        MainWindow2.setStyleSheet("")
        MainWindow2.setTabShape(QtWidgets.QTabWidget.Rounded)

        # Central widget ─────────────────────────────────────────────
        self.centralwidget = QtWidgets.QWidget(MainWindow2)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        root = QtWidgets.QVBoxLayout(self.centralwidget)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(0)

        # Three GraphicsLayoutWidgets ────────────────────────────────
        self.plt = _make_plot_widget(self.centralwidget)
        self.plt.setObjectName("plt")

        self.plt_temp = _make_plot_widget(self.centralwidget)
        self.plt_temp.setObjectName("plt_temp")

        self.pltB = _make_plot_widget(self.centralwidget)
        self.pltB.setObjectName("pltB")

        # Tile panels ────────────────────────────────────────────────
        self._amp_panel = DraggableTilePanel(AMP_TILE, self.plt, self.centralwidget)
        self._temp_panel = DraggableTilePanel(TEMP_TILE, self.plt_temp, self.centralwidget)
        self._rfd_panel = DraggableTilePanel(RFD_TILE, self.pltB, self.centralwidget)

        # TileGrid — initial slots match the target default layout ────
        self.tile_grid = TileGrid(self.centralwidget)
        self.tile_grid.add_tile(RFD_TILE, self._rfd_panel, _Slot("left", 0))  # large left column
        self.tile_grid.add_tile(AMP_TILE, self._amp_panel, _Slot("right", 1))  # top of right column
        self.tile_grid.add_tile(
            TEMP_TILE, self._temp_panel, _Slot("right", 2)
        )  # bottom of right column

        root.addWidget(self.tile_grid)
        MainWindow2.setCentralWidget(self.centralwidget)

        # Legacy splitter stubs (kept so external call-sites don't crash)
        self.Layout_graphs = QtWidgets.QWidget()
        self.btnCollapse = QtWidgets.QToolButton()
        self.btnExpand = QtWidgets.QToolButton()

        self.retranslateUi(MainWindow2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow2)

    # ────────────────────────────────────────────────────────────────

    def retranslateUi(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2", "{} {} - Plots".format(Constants.app_title, Constants.app_version)
            )
        )

    def Ui(self, MainWindow2: QtWidgets.QMainWindow) -> None:
        self.retranslateUi(MainWindow2)

    # Legacy no-op shims
    def handleSplitterMoved(self, pos: int = 0, index: int = 0) -> None:
        pass

    def handleSplitterButton(self, collapse: bool = True) -> None:
        pass
