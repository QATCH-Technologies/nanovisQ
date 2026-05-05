"""
QATCH.ui.mainWindow_ui_plots

Plot layout for the nanovisQ application with glass-morphism styling that
matches the modeMenuScrollArea palette from ui_main_theme.qss.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-05-05
"""

import os
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget
from pyqtgraph import GraphicsLayoutWidget

from QATCH.common.architecture import Architecture
from QATCH.core.constants import Constants

# ---------------------------------------------------------------------------
# Glass-morphism primitive
# ---------------------------------------------------------------------------


class GlassPlotPanel(QtWidgets.QWidget):
    """Frosted-glass card that wraps a pyqtgraph GraphicsLayoutWidget.

    Uses the same rgba(255,255,255,160) palette as #modeMenuScrollArea and
    the GlassControlsWidget so every panel in the run view reads as a
    cohesive glass surface.

    The inner plot widget is inset by ``_INNER_MARGIN`` px on every side so
    its square corners sit well inside the panel's rounded frame — no masking
    required.

    Visual pipeline (back → front):
        1. rgba(255,255,255,160) frosted white fill
        2. Faint #E4EBF1 cool tint (matches main-window gradient undertone)
        3. Top-edge shimmer gradient
        4. Outer border  rgba(255,255,255,220)
        5. Inner border  rgba(200,210,220,80)
    """

    _RADIUS: float = 12.0
    _INNER_MARGIN: int = 4  # keeps plot corners inside the rounded frame

    def __init__(
        self,
        plot_widget: GraphicsLayoutWidget,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

        m = self._INNER_MARGIN
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(m, m, m, m)
        layout.setSpacing(0)
        layout.addWidget(plot_widget)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())

        # Clip background fills to the rounded rectangle
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Frosted white fill — matches #modeMenuScrollArea exactly
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))
        # Faint cool tint from the app gradient undertone
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 36)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 55))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        # Outer — bright white rim (matches the sidebar card border)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        # Inner — subtle cool-grey definition stroke
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 80), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()


# ---------------------------------------------------------------------------
# Shared QSS for splitter handle buttons
# ---------------------------------------------------------------------------

_SPLITTER_BTN_QSS = """
    QToolButton {
        background: rgba(255, 255, 255, 100);
        color: rgba(30, 40, 55, 180);
        border: 1px solid rgba(200, 210, 220, 120);
        border-radius: 4px;
        padding: 3px;
    }
    QToolButton:hover {
        background: rgba(229, 229, 229, 200);
        border: 1px solid rgba(180, 190, 200, 160);
    }
    QToolButton:pressed {
        background: rgba(200, 210, 218, 240);
        border: 1px solid rgba(160, 175, 190, 180);
    }
"""

_SPLITTER_QSS = """
    QSplitter {
        background: transparent;
    }
    QSplitter::handle {
        background: transparent;
        border: none;
    }
"""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UIPlots:

    def setupUi(self, MainWindow2):
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

        self.centralwidget = QtWidgets.QWidget(MainWindow2)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)

        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        # Retain original vertical rhythm: no top gap, standard padding elsewhere
        self.gridLayout.setContentsMargins(8, 0, 8, 8)

        # Splitter ---------------------------------------------------------
        self.Layout_graphs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.Layout_graphs.setObjectName("Layout_graphs")
        self.Layout_graphs.setStyleSheet(_SPLITTER_QSS)

        # Plot widgets -----------------------------------------------------
        self.plt = GraphicsLayoutWidget(self.centralwidget)
        self.pltB = GraphicsLayoutWidget(self.centralwidget)

        self.plt.setAutoFillBackground(False)
        self.plt.setStyleSheet("border: 0px;")
        self.plt.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plt.setFrameShadow(QtWidgets.QFrame.Plain)
        self.plt.setLineWidth(0)
        self.plt.setObjectName("plt")
        self.plt.setMinimumWidth(333)

        self.pltB.setAutoFillBackground(False)
        self.pltB.setStyleSheet("border: 0px;")
        self.pltB.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.pltB.setFrameShadow(QtWidgets.QFrame.Plain)
        self.pltB.setLineWidth(0)
        self.pltB.setObjectName("pltB")
        self.pltB.setMinimumWidth(666)

        # Wrap each plot in a glass card and add to the splitter -----------
        self.plt_panel = GlassPlotPanel(self.plt)
        self.pltB_panel = GlassPlotPanel(self.pltB)

        self.Layout_graphs.addWidget(self.pltB_panel)
        self.Layout_graphs.addWidget(self.plt_panel)

        # Splitter collapse/expand handle ----------------------------------
        self.Layout_graphs.setHandleWidth(12)
        handle = self.Layout_graphs.handle(1)

        layout_s = QtWidgets.QVBoxLayout()
        layout_s.setContentsMargins(0, 0, 0, 0)
        layout_s.setSpacing(4)
        layout_s.addStretch()

        self.btnCollapse = QtWidgets.QToolButton(handle)
        self.btnCollapse.setArrowType(QtCore.Qt.RightArrow)
        self.btnCollapse.setStyleSheet(_SPLITTER_BTN_QSS)
        self.btnCollapse.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnCollapse.clicked.connect(lambda: self.handleSplitterButton(True))
        layout_s.addWidget(self.btnCollapse)

        self.btnExpand = QtWidgets.QToolButton(handle)
        self.btnExpand.setArrowType(QtCore.Qt.LeftArrow)
        self.btnExpand.setStyleSheet(_SPLITTER_BTN_QSS)
        self.btnExpand.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btnExpand.clicked.connect(lambda: self.handleSplitterButton(False))
        layout_s.addWidget(self.btnExpand)

        layout_s.addStretch()
        handle.setLayout(layout_s)
        self.btnExpand.setVisible(False)

        self.Layout_graphs.splitterMoved.connect(self.handleSplitterMoved)

        self.gridLayout.addWidget(self.Layout_graphs, 2, 1, 1, 1)
        MainWindow2.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow2)

    def handleSplitterMoved(self, pos, index):
        collapsed = self.Layout_graphs.sizes()[-1] == 0
        self.btnCollapse.setVisible(not collapsed)
        self.btnExpand.setVisible(collapsed)

    def handleSplitterButton(self, collapse=True):
        if collapse:
            self.btnCollapse.setVisible(False)
            self.btnExpand.setVisible(True)
            self.Layout_graphs.setSizes([1, 0])
        else:
            self.btnCollapse.setVisible(True)
            self.btnExpand.setVisible(False)
            width = self.Layout_graphs.width()
            self.Layout_graphs.setSizes([int(width * 0.65), int(width * 0.35)])

    def retranslateUi(self, MainWindow2):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2",
                "{} {} - Plots".format(Constants.app_title, Constants.app_version),
            )
        )

    def Ui(self, MainWindow2):
        _translate = QtCore.QCoreApplication.translate
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-icon.png")
        MainWindow2.setWindowIcon(QtGui.QIcon(icon_path))
        MainWindow2.setWindowTitle(
            _translate(
                "MainWindow2",
                "{} {} - Plots".format(Constants.app_title, Constants.app_version),
            )
        )
