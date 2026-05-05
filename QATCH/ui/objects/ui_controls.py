"""
QATCH.ui.mainWindow_ui

The mainWindow_ui module handles the drawing, UI control elements,
and UI actions for the main window of the main application.

Author(s)
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)
    Others...

Date:
    2026-01-26
"""

import os
from typing import Optional
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QDesktopWidget,
)

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants, OperationType
from QATCH.ui.widgets.well_plate_widget import WellPlate
from QATCH.ui.popUp import PopUp
from QATCH.ui.components.number_icon_button import NumberIconButton
from QATCH.ui.components.run_controls_button import RunControls

# ---------------------------------------------------------------------------
# Glass-morphism primitives
# ---------------------------------------------------------------------------


class GlassControlsWidget(QtWidgets.QWidget):
    """Frosted-glass container that provides the toolbar's gradient backdrop.

    Renders the same cool-blue gradient palette used by GlassCard in
    ui_login when no live backdrop is available, overlaid with the standard
    white-tint, shimmer, and dual-border glass language.
    """

    _RADIUS: float = 10.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())

        # Clip to rounded rectangle
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Match modeMenuScrollArea: rgba(255,255,255,160) on the app gradient
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 160))

        # Faint cool tint — same underlying gradient tone as #E4EBF1
        p.fillRect(self.rect(), QtGui.QColor(228, 235, 241, 18))

        # Soft top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 32)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 50))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Border — matches the sidebar's rgba(255,255,255,220) border
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 220), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 210, 220, 80), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()


class GlassHeaderLabel(QtWidgets.QLabel):
    """Section-header label rendered as a brand-blue glass panel.

    Replaces the legacy solid ``background: #008EC0`` headers.  The
    hand-painted background carries the same shimmer/border pipeline as
    GlassCard while maintaining the QATCH cool-blue identity.
    """

    _RADIUS: float = 4.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        # Text colour and padding only — background handled in paintEvent
        self.setStyleSheet(
            "QLabel { color: rgba(255, 255, 255, 230); "
            "padding: 2px 6px; font-weight: bold; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Brand-blue gradient base
        grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(0, 118, 174))
        grad.setColorAt(1.0, QtGui.QColor(0, 158, 210))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        # Glass tints
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 45))
        p.fillRect(self.rect(), QtGui.QColor(180, 220, 245, 30))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.65)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 55))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(80, 160, 215, 130), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 130), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        # Render text via base class (respects alignment, QSS color)
        super().paintEvent(event)


class GlassStatusLabel(QtWidgets.QLabel):
    """Frosted-white glass panel for status and info displays.

    Replaces the legacy ``background: white; border: 1px solid #cccccc``
    status labels with a translucent glass treatment that integrates
    seamlessly with GlassControlsWidget.
    """

    _RADIUS: float = 5.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: rgba(28, 40, 52, 210); " "padding: 2px 6px; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Frosted white glass base
        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 155))
        p.fillRect(self.rect(), QtGui.QColor(210, 225, 240, 40))

        # Top shimmer
        shimmer = QtGui.QLinearGradient(0, 0, 0, 36)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 80))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        # Borders
        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(120, 160, 200, 110), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 160), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        super().paintEvent(event)


class GlassWarningLabel(QtWidgets.QLabel):
    """Orange glass warning banner for the Advanced Settings dialog."""

    _RADIUS: float = 4.0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setStyleSheet(
            "QLabel { color: white; font-weight: bold; "
            "padding: 2px 6px; background: transparent; }"
        )

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing)

        rect_f = QtCore.QRectF(self.rect())
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect_f, self._RADIUS, self._RADIUS)
        p.setClipPath(clip)

        # Warm orange glass gradient
        grad = QtGui.QLinearGradient(0, 0, self.width(), self.height())
        grad.setColorAt(0.0, QtGui.QColor(210, 80, 0))
        grad.setColorAt(1.0, QtGui.QColor(255, 125, 20))
        p.fillRect(self.rect(), QtGui.QBrush(grad))

        p.fillRect(self.rect(), QtGui.QColor(255, 255, 255, 40))

        shimmer = QtGui.QLinearGradient(0, 0, 0, self.height() * 0.65)
        shimmer.setColorAt(0.0, QtGui.QColor(255, 255, 255, 50))
        shimmer.setColorAt(1.0, QtGui.QColor(255, 255, 255, 0))
        p.fillRect(self.rect(), QtGui.QBrush(shimmer))

        p.setClipping(False)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(190, 80, 0, 140), 1.0))
        p.drawRoundedRect(rect_f.adjusted(0.5, 0.5, -0.5, -0.5), self._RADIUS, self._RADIUS)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 1.0))
        p.drawRoundedRect(
            rect_f.adjusted(1.5, 1.5, -1.5, -1.5),
            self._RADIUS - 1.5,
            self._RADIUS - 1.5,
        )

        p.end()
        super().paintEvent(event)


# ---------------------------------------------------------------------------
# Shared QSS fragments
# ---------------------------------------------------------------------------

_GLASS_BUTTON_QSS = """
    QPushButton {{
        background: transparent;
        color: rgba(30, 40, 55, 200);
        border: 1px solid transparent;
        border-radius: 4px;
        padding: {padding};
        font-size: 12px;
    }}
    QPushButton:hover {{
        background: rgba(229, 229, 229, 150);
        border: 1px solid transparent;
    }}
    QPushButton:pressed {{
        background: rgba(229, 229, 229, 200);
        border: 1px solid transparent;
    }}
    QPushButton:disabled {{
        color: rgba(30, 40, 55, 90);
        background: transparent;
        border: 1px solid transparent;
    }}
"""

_GLASS_TOOLBAR_QSS = """
    QToolBar {
        background: transparent;
        border: none;
        spacing: 2px;
    }
    QToolButton {
        background: transparent;
        color: rgba(30, 40, 55, 200);
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 4px 8px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 12px;
    }
    QToolButton:hover {
        background: rgba(229, 229, 229, 150);
        border: 1px solid transparent;
    }
    QToolButton:pressed {
        background: rgba(229, 229, 229, 200);
        border: 1px solid transparent;
    }
    QToolButton:checked {
        background: transparent;
        border: 1px solid transparent;
    }
    QToolButton:disabled {
        color: rgba(30, 40, 55, 90);
        background: transparent;
        border: 1px solid transparent;
    }
    QToolBar::separator {
        background: rgba(0, 0, 0, 22);
        width: 1px;
        margin: 5px 4px;
    }
"""

_GLASS_PROGRESSBAR_QSS = """
    QProgressBar {
        border: 1px solid rgba(0, 0, 0, 25);
        border-radius: 4px;
        text-align: center;
        color: rgba(30, 40, 55, 200);
        background: rgba(255, 255, 255, 120);
        font-weight: bold;
    }
    QProgressBar::chunk {
        background: qlineargradient(
            spread:pad, x1:0, y1:0, x2:1, y2:0,
            stop:0 rgba(10, 163, 230, 130),
            stop:1 rgba(10, 163, 230, 90)
        );
        border-radius: 3px;
    }
"""

_GLASS_TEMP_CONTROLLER_QSS = """
    QWidget#tempController {
        background: rgba(229, 229, 229, 80);
        border: none;
        border-radius: 6px;
    }
    QLabel {
        background: transparent;
        border: none;
        color: rgba(30, 40, 55, 200);
    }
    QSlider::groove:horizontal {
        height: 4px;
        background: rgba(0, 0, 0, 30);
        border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #0AA3E6;
        border: 1px solid rgba(0, 130, 200, 200);
        width: 12px;
        height: 12px;
        margin: -4px 0;
        border-radius: 6px;
    }
    QSlider::sub-page:horizontal {
        background: rgba(10, 163, 230, 120);
        border-radius: 2px;
    }
    QSlider::handle:horizontal:disabled {
        background: rgba(150, 170, 190, 140);
        border: 1px solid rgba(0, 0, 0, 30);
    }
"""


# ---------------------------------------------------------------------------
# Temperature label — emits textUpdated so the display panel can react
# ---------------------------------------------------------------------------


class TemperatureLabel(QtWidgets.QLabel):
    """QLabel that fires textUpdated whenever setText() is called.

    Keeps full backward compatibility (callers use setText as normal) while
    letting the new split-display panel observe changes without polling.
    """

    textUpdated = QtCore.pyqtSignal(str)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.textUpdated.emit(text)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class UIControls:  # QtWidgets.QMainWindow

    def setupUi(self, MainWindow1):
        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880
        SHOW_SIMPLE_CONTROLS = True
        self.cal_initialized = False
        self.parent = MainWindow1

        MainWindow1.setObjectName("MainWindow1")
        MainWindow1.setMinimumSize(QtCore.QSize(1000, 50))
        if Architecture.get_os() is OSType.macosx:
            MainWindow1.resize(1080, 188)
        elif USE_FULLSCREEN:
            MainWindow1.resize(2880, 390)
            MainWindow1.move(0, 1485)
        else:
            MainWindow1.resize(1503, 175)
            MainWindow1.move(7, 567)
        MainWindow1.setStyleSheet("")
        MainWindow1.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow1)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.Layout_controls = QtWidgets.QGridLayout()
        self.Layout_controls.setObjectName("Layout_controls")

        # frequency/quartz combobox -------------------------------------------
        self.cBox_Speed = QtWidgets.QComboBox()
        self.cBox_Speed.setEditable(False)
        self.cBox_Speed.setObjectName("cBox_Speed")
        if USE_FULLSCREEN:
            self.cBox_Speed.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Speed, 4, 1, 1, 1)

        # stop button ---------------------------------------------------------
        self.pButton_Stop = QtWidgets.QPushButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/stop_icon.ico")
        self.pButton_Stop.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Stop.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Stop.setObjectName("pButton_Stop")
        if USE_FULLSCREEN:
            self.pButton_Stop.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Stop, 3, 6, 1, 1)

        # COM port combobox ---------------------------------------------------
        self.cBox_Port = QtWidgets.QComboBox()
        self.cBox_Port.setEditable(False)
        self.cBox_Port.setObjectName("cBox_Port")
        if USE_FULLSCREEN:
            self.cBox_Port.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Port, 2, 1, 1, 1)

        # Identify button ---------------------------------------------------------
        self.pButton_ID = QtWidgets.QPushButton()
        self.pButton_ID.setToolTip("Identify selected Serial COM Port")
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/identify-icon.png")
        self.pButton_ID.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_ID.setStyleSheet(_GLASS_BUTTON_QSS.format(padding="3px"))
        self.pButton_ID.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        if USE_FULLSCREEN:
            self.pButton_ID.setMinimumSize(QtCore.QSize(60, 50))
        else:
            self.pButton_ID.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ID.setObjectName("pButton_ID")
        self.Layout_controls.addWidget(self.pButton_ID, 2, 2, 1, 1)

        # Refresh button ---------------------------------------------------------
        self.pButton_Refresh = QtWidgets.QPushButton()
        self.pButton_Refresh.setToolTip("Refresh Serial COM Port list")
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/refresh-icon.png")
        self.pButton_Refresh.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Refresh.setStyleSheet(
            _GLASS_BUTTON_QSS.format(padding="3px") + "QPushButton { margin-right: 9px; }"
        )
        self.pButton_Refresh.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        if USE_FULLSCREEN:
            self.pButton_Refresh.setMinimumSize(QtCore.QSize(70, 50))
        else:
            self.pButton_Refresh.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Refresh.setObjectName("pButton_Refresh")
        self.Layout_controls.addWidget(self.pButton_Refresh, 2, 3, 1, 1)

        # Operation mode - source ---------------------------------------------
        self.cBox_Source = QtWidgets.QComboBox()
        self.cBox_Source.setObjectName("cBox_Source")
        if USE_FULLSCREEN:
            self.cBox_Source.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Source, 2, 0, 1, 1)

        # Frequency hopping checkbox ------------------------------------------
        self.chBox_freqHop = QtWidgets.QCheckBox()
        self.chBox_freqHop.setEnabled(True)
        self.chBox_freqHop.setChecked(False)
        self.chBox_freqHop.setObjectName("chBox_freqHop")
        self.Layout_controls.addWidget(self.chBox_freqHop, 4, 2, 1, 2)

        # Noise correction checkbox -------------------------------------------
        self.chBox_correctNoise = QtWidgets.QCheckBox()
        self.chBox_correctNoise.setEnabled(True)
        self.chBox_correctNoise.setChecked(True)
        self.chBox_correctNoise.setObjectName("chBox_correctNoise")
        self.Layout_controls.addWidget(self.chBox_correctNoise, 5, 1, 1, 3)

        # Cartridge Auto-Lock -------------------------------------------------
        self.l9 = GlassHeaderLabel("Cartridge Auto-Lock")
        if USE_FULLSCREEN:
            self.l9.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l9, 1, 4, 1, 1)

        # Cartridge Controls --------------------------------------------------
        self.rButton_Automatic = QtWidgets.QRadioButton("Automatic")
        self.rButton_Automatic.setToolTip("""
            <b><u>Automatic:</u></b><br/>
            - Locks before init/run<br/>
            - Useful if/when user forgets
            """)
        self.rButton_Automatic.setChecked(True)  # default
        self.rButton_Manual = QtWidgets.QRadioButton("Manual")
        self.rButton_Manual.setToolTip("""
            <b><u>Manual:</u></b><br/>
            - You control lock position<br/>
            - Must lock before init/run
            """)
        self.rCartridgeMode = QtWidgets.QButtonGroup()
        self.rCartridgeMode.addButton(self.rButton_Automatic, 1)
        self.rCartridgeMode.addButton(self.rButton_Manual, 0)
        self.layMode = QtWidgets.QVBoxLayout()
        self.layMode.addWidget(self.rButton_Automatic)
        self.layMode.addWidget(self.rButton_Manual)
        self.grpMode = QtWidgets.QGroupBox("Auto-Lock Mode:")
        self.grpMode.setLayout(self.layMode)
        self.Layout_controls.addWidget(self.grpMode, 2, 4, 3, 1)

        # start button --------------------------------------------------------
        self.pButton_Start = QtWidgets.QPushButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/start_icon.ico")
        self.pButton_Start.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Start.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Start.setObjectName("pButton_Start")
        if USE_FULLSCREEN:
            self.pButton_Start.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Start, 2, 6, 1, 1)

        # Add signal for Run Controls UI to handle START from Advanced menu ---
        self.pButton_Start.clicked.connect(
            lambda: (
                self.run_controls.set_running(True)
                if (
                    (OperationType(self.cBox_Source.currentIndex()) == OperationType.measurement)
                    and hasattr(self, "run_controls")
                )
                else None
            )
        )

        # clear plots button --------------------------------------------------
        self.pButton_Clear = QtWidgets.QPushButton()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/clear_icon.ico")
        self.pButton_Clear.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Clear.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Clear.setObjectName("pButton_Clear")
        if USE_FULLSCREEN:
            self.pButton_Clear.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Clear, 2, 5, 1, 1)

        # reference button ----------------------------------------------------
        self.pButton_Reference = QtWidgets.QPushButton()
        self.pButton_Reference.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Reference.setObjectName("pButton_Reference")
        self.pButton_Reference.setCheckable(True)
        if USE_FULLSCREEN:
            self.pButton_Reference.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_Reference, 3, 5, 1, 1)

        # restore factory defaults --------------------------------------------
        self.pButton_ResetApp = QtWidgets.QPushButton()
        self.pButton_ResetApp.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ResetApp.setObjectName("pButton_ResetApp")
        if USE_FULLSCREEN:
            self.pButton_ResetApp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pButton_ResetApp, 4, 5, 1, 1)

        # samples SpinBox -----------------------------------------------------
        self.sBox_Samples = QtWidgets.QSpinBox()
        self.sBox_Samples.setMinimum(1)
        self.sBox_Samples.setMaximum(100000)
        self.sBox_Samples.setProperty("value", 500)
        self.sBox_Samples.setObjectName("sBox_Samples")
        self.sBox_Samples.setVisible(False)
        self.Layout_controls.addWidget(self.sBox_Samples, 2, 4, 1, 1)

        # export file CheckBox ------------------------------------------------
        self.chBox_export = QtWidgets.QCheckBox()
        self.chBox_export.setEnabled(True)
        self.chBox_export.setObjectName("chBox_export")
        self.chBox_export.setVisible(False)
        self.Layout_controls.addWidget(self.chBox_export, 4, 4, 1, 1)

        # temperature Control slider ------------------------------------------
        self.slTemp = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slTemp.setMinimum(8)
        self.slTemp.setMaximum(40)
        self.slTemp.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slTemp.setTickInterval(1)
        self.slTemp.setSingleStep(1)
        self.slTemp.setPageStep(5)
        self.Layout_controls.addWidget(self.slTemp, 3, 4, 1, 1)

        # temperature Control label (hidden data conduit — kept for compat) ----
        self.lTemp = TemperatureLabel()
        self.lTemp.setText("PV:--.--C SP:--.--C OP:----")
        self.lTemp.setAlignment(QtCore.Qt.AlignCenter)
        self.lTemp.setFont(QtGui.QFont("Consolas", -1))
        self.lTemp.hide()
        self.Layout_controls.addWidget(self.lTemp, 2, 4, 1, 1)

        # temperature Control button ------------------------------------------
        self.pTemp = QtWidgets.QPushButton()
        self.pTemp.setText("Start Temp Control")
        if USE_FULLSCREEN:
            self.pTemp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pTemp, 4, 4, 1, 1)

        # Control Buttons ------------------------------------------------------
        self.l = GlassHeaderLabel("Control Buttons")
        if USE_FULLSCREEN:
            self.l.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l, 1, 5, 1, 2)

        # Operation Mode -------------------------------------------------------
        self.l0 = GlassHeaderLabel("Operation Mode")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l0, 1, 0, 1, 1)

        # Resonance Frequency / Quartz Sensor ---------------------------------
        self.l2 = GlassHeaderLabel("Resonance Frequency / Quartz Sensor")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l2, 3, 1, 1, 3)

        # Serial COM Port -----------------------------------------------------
        self.l1 = GlassHeaderLabel("Serial COM Port")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l1, 1, 1, 1, 3)

        # logo ----------------------------------------------------------------
        self.l3 = QtWidgets.QLabel()
        self.l3.setAlignment(QtCore.Qt.AlignRight)
        self.Layout_controls.addWidget(self.l3, 4, 7, 1, 1)
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/qatch-logo_full.jpg")
        if USE_FULLSCREEN:
            pixmap = QtGui.QPixmap(icon_path)
            pixmap = pixmap.scaled(250, 50, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            self.l3.setPixmap(pixmap)
        else:
            self.l3.setPixmap(QtGui.QPixmap(icon_path))

        # qatch link ----------------------------------------------------------
        self.l4 = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.l4, 3, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.l4.linkActivated.connect(link)
        self.l4.setAlignment(QtCore.Qt.AlignRight)
        self.l4.setText(
            '<a href="https://qatchtech.com/"> <font size=4 color=#008EC0 >qatchtech.com</font>'
        )

        # info@qatchtech.com Mail -----------------------------------------------
        self.lmail = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lmail, 2, 7, 1, 1)

        def linkmail(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lmail.linkActivated.connect(linkmail)
        self.lmail.setAlignment(QtCore.Qt.AlignRight)
        self.lmail.setText(
            '<a href="mailto:info@qatchtech.com"> <font color=#008EC0 >info@qatchtech.com</font>'
        )

        # software user guide --------------------------------------------------------
        self.lg = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lg, 1, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lg.linkActivated.connect(link)
        self.lg.setAlignment(QtCore.Qt.AlignRight)
        self.lg.setText(
            '<a href="file://{}/docs/userguide.pdf"> <font color=#008EC0 >User Guide</font>'.format(
                Architecture.get_path()
            )
        )

        # Save file / TEC Temperature Control header --------------------------
        self.infosave = GlassHeaderLabel("TEC Temperature Control")
        if USE_FULLSCREEN:
            self.infosave.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infosave, 1, 4, 1, 1)

        # Program Status standby ----------------------------------------------
        self.infostatus = GlassStatusLabel()
        self.infostatus.setAlignment(QtCore.Qt.AlignCenter)
        self.infostatus.setText("Program Status Standby")
        if USE_FULLSCREEN:
            self.infostatus.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infostatus, 5, 5, 1, 2)

        # Infobar -------------------------------------------------------------
        self.infobar = QtWidgets.QLineEdit()
        self.infobar.setReadOnly(True)
        self.infobar_label = GlassStatusLabel()
        self.infobar.textChanged.connect(self.infobar_label.setText)
        if SHOW_SIMPLE_CONTROLS:
            self.infobar.textChanged.connect(self._update_progress_text)
        if USE_FULLSCREEN:
            self.infobar_label.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infobar_label, 0, 0, 1, 7)

        # Multiplex -----------------------------------------------------------
        self.lmp = GlassHeaderLabel("Multiplex Mode")
        if USE_FULLSCREEN:
            self.lmp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.lmp, 3, 0, 1, 1)

        self.cBox_MultiMode = QtWidgets.QComboBox()
        self.cBox_MultiMode.setObjectName("cBox_MultiMode")
        self.cBox_MultiMode.addItems(["1 Channel", "2 Channels", "3 Channels", "4 Channels"])
        self.cBox_MultiMode.setCurrentIndex(0)
        if USE_FULLSCREEN:
            self.cBox_MultiMode.setFixedHeight(50)

        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/")
        self.pButton_PlateConfig = QtWidgets.QPushButton(
            QtGui.QIcon(os.path.join(icon_path, "advanced.png")), ""
        )
        self.pButton_PlateConfig.setToolTip("Plate Configuration...")
        self.pButton_PlateConfig.clicked.connect(self.doPlateConfig)
        self.hBox_MultiConfig = QtWidgets.QHBoxLayout()
        self.hBox_MultiConfig.addWidget(self.cBox_MultiMode, 3)
        self.hBox_MultiConfig.addWidget(self.pButton_PlateConfig, 1)
        self.Layout_controls.addLayout(self.hBox_MultiConfig, 4, 0, 1, 1)

        self.chBox_MultiAuto = QtWidgets.QCheckBox()
        self.chBox_MultiAuto.setEnabled(True)
        self.chBox_MultiAuto.setChecked(True)
        self.chBox_MultiAuto.setObjectName("chBox_MultiAuto")
        self.Layout_controls.addWidget(self.chBox_MultiAuto, 5, 0, 1, 1)

        # Progressbar ---------------------------------------------------------
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.run_progress_bar.setObjectName("progressBar")
        self.run_progress_bar.setStyleSheet(_GLASS_PROGRESSBAR_QSS)

        if USE_FULLSCREEN:
            self.run_progress_bar.setFixedHeight(50)
        if SHOW_SIMPLE_CONTROLS:
            self.run_progress_bar.valueChanged.connect(self._update_progress_value)

        self.run_progress_bar.setValue(0)
        self.run_progress_bar.setHidden(True)

        self.Layout_controls.setColumnStretch(0, 0)
        self.Layout_controls.setColumnStretch(1, 1)
        self.Layout_controls.setColumnStretch(2, 0)
        self.Layout_controls.setColumnStretch(3, 0)
        self.Layout_controls.setColumnStretch(4, 2)
        self.Layout_controls.setColumnStretch(5, 2)
        self.Layout_controls.setColumnStretch(6, 2)
        self.Layout_controls.addWidget(self.run_progress_bar, 0, 7, 1, 1)
        self.gridLayout.addLayout(self.Layout_controls, 7, 1, 1, 1)

        # ---- Simple / toolbar layout ----------------------------------------

        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolBar = QtWidgets.QHBoxLayout()

        self.tool_bar = QtWidgets.QToolBar()
        self.tool_bar.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar.setStyleSheet(_GLASS_TOOLBAR_QSS)

        self.tool_NextPortRow = NumberIconButton()
        self.tool_NextPortRow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_NextPortRow.setText("Next Port")
        self.tool_NextPortRow.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_NextPortRow.clicked.connect(self.action_next_port)
        self.action_NextPortRow = self.tool_bar.addWidget(self.tool_NextPortRow)

        self.action_NextPortSep = self.tool_bar.addSeparator()

        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/")

        icon_init = QtGui.QIcon()
        icon_init.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "initialize.png")), QtGui.QIcon.Normal
        )
        self.tool_Initialize = QtWidgets.QToolButton()
        self.tool_Initialize.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Initialize.setIcon(icon_init)
        self.tool_Initialize.setText("Initialize")
        self.tool_Initialize.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_Initialize.clicked.connect(self.action_initialize)
        self.tool_bar.addWidget(self.tool_Initialize)

        self.tool_bar.addSeparator()

        # RunControls composite widget ----------------------------------------
        self.run_controls = RunControls()
        self.run_controls.startRequested.connect(self.action_start)
        self.run_controls.stopRequested.connect(self.action_stop)
        self.run_controls.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.run_controls.setEnabled(False)
        self.tool_Start = self.run_controls  # backward-compat alias
        self.tool_Stop = self.run_controls
        self.tool_bar.addWidget(self.run_controls)
        self.tool_bar.addSeparator()

        icon_reset = QtGui.QIcon()
        icon_reset.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "reset.png")), QtGui.QIcon.Normal
        )
        self.tool_Reset = QtWidgets.QToolButton()
        self.tool_Reset.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Reset.setIcon(icon_reset)
        self.tool_Reset.setText("Reset")
        self.tool_Reset.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_Reset.clicked.connect(self.action_reset)
        self.tool_bar.addWidget(self.tool_Reset)

        self.tool_bar.addSeparator()

        self._warningTimer = QtCore.QTimer()
        self._warningTimer.setSingleShot(True)
        self._warningTimer.timeout.connect(self.action_tempcontrol_warning)
        self._warningTimer.setInterval(2000)

        icon_temp = QtGui.QIcon()
        icon_temp.addPixmap(QtGui.QPixmap(os.path.join(icon_path, "temp.png")), QtGui.QIcon.Normal)
        self.tool_TempControl = QtWidgets.QToolButton()
        self.tool_TempControl.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_TempControl.setIcon(icon_temp)
        self.tool_TempControl.setText("Temp Control")
        self.tool_TempControl.setCheckable(True)
        self.tool_TempControl.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_TempControl.clicked.connect(self.action_tempcontrol)
        self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_start
        self.tool_TempControl.leaveEvent = self.action_tempcontrol_warn_stop
        self.tool_bar.addWidget(self.tool_TempControl)

        self.toolBar.addWidget(self.tool_bar)

        # Temperature controller widget — starts collapsed, expands on toggle ----
        self.tempController = QtWidgets.QWidget()
        self.tempController.setObjectName("tempController")
        self.tempController.enterEvent = self.action_tempcontrol_warn_start
        self.tempController.leaveEvent = self.action_tempcontrol_warn_stop
        self.tempController.setMinimumWidth(0)
        self.tempController.setMaximumWidth(0)  # collapsed until activated
        self.tempController.setStyleSheet(_GLASS_TEMP_CONTROLLER_QSS)

        # Status label — coloured background + explanatory text ----------------
        self.tempStatusBar = QtWidgets.QLabel("Offline")
        self.tempStatusBar.setFixedHeight(16)
        self.tempStatusBar.setAlignment(QtCore.Qt.AlignCenter)
        _status_font = QtGui.QFont()
        _status_font.setPointSize(7)
        _status_font.setBold(True)
        self.tempStatusBar.setFont(_status_font)
        self.tempStatusBar.setStyleSheet(
            "QLabel { background: rgba(150, 155, 160, 120); color: rgba(30,40,55,160); "
            "border-radius: 2px; padding: 0 4px; }"
        )

        # Split value labels — right-side stack
        value_font = QtGui.QFont("Consolas", 7)
        self.lPV = QtWidgets.QLabel("PV  --.--°C")
        self.lSP = QtWidgets.QLabel("SP  --.--°C")
        self.lOP = QtWidgets.QLabel("OP  ----")
        for lbl in (self.lPV, self.lSP, self.lOP):
            lbl.setFont(value_font)
            lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            lbl.setStyleSheet("background: transparent; border: none; color: rgba(30,40,55,200);")
        value_stack = QtWidgets.QVBoxLayout()
        value_stack.setContentsMargins(4, 0, 2, 0)
        value_stack.setSpacing(1)
        value_stack.addWidget(self.lPV)
        value_stack.addWidget(self.lSP)
        value_stack.addWidget(self.lOP)

        # Slider + values side by side
        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(4, 0, 4, 2)
        content_row.setSpacing(4)
        content_row.addWidget(self.slTemp, 1)
        content_row.addLayout(value_stack)

        # Content area (left portion of panel)
        _content_area = QtWidgets.QVBoxLayout()
        _content_area.setContentsMargins(4, 4, 4, 4)
        _content_area.setSpacing(3)
        _content_area.addWidget(self.tempStatusBar)
        _content_area.addLayout(content_row)

        # Arrow strip — fixed-width clickable button pinned to the right edge.
        # Clicking it toggles the panel just like the toolbar button does.
        # Drop arrow_left.svg / arrow_right.svg into QATCH/icons/ and they
        # will be loaded automatically; Unicode arrows are the fallback.
        self.tempArrowStrip = QtWidgets.QPushButton()
        self.tempArrowStrip.setFixedWidth(18)
        self.tempArrowStrip.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tempArrowStrip.setStyleSheet("""
            QPushButton {
                background: rgba(0, 0, 0, 12);
                border: none;
                border-left: 1px solid rgba(0, 0, 0, 18);
                color: rgba(30, 40, 55, 200);
                font-size: 11px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background: rgba(229, 229, 229, 150);
                color: rgba(30, 40, 55, 255);
            }
            QPushButton:pressed {
                background: rgba(200, 210, 218, 180);
            }
        """)
        self.tempArrowStrip.clicked.connect(self._toggle_temp_controller)
        self._set_temp_arrow(expand=False)  # start collapsed → right arrow

        # Assemble controller panel — content left, arrow right
        self.tempLayout = QtWidgets.QHBoxLayout()
        self.tempLayout.setContentsMargins(0, 0, 0, 0)
        self.tempLayout.setSpacing(0)
        self.tempLayout.addLayout(_content_area, 1)
        self.tempLayout.addWidget(self.tempArrowStrip)
        self.tempController.setLayout(self.tempLayout)
        self.toolBar.addWidget(self.tempController)

        # Wire live temperature updates to the display panel
        self.lTemp.textUpdated.connect(self._update_temp_display)

        self.toolBar.addStretch()

        self.tool_bar_2 = QtWidgets.QToolBar()
        self.tool_bar_2.setIconSize(QtCore.QSize(50, 30))
        self.tool_bar_2.setStyleSheet(_GLASS_TOOLBAR_QSS)

        icon_advanced = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/advanced.png")
        icon_advanced.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        self.tool_Advanced = QtWidgets.QToolButton()
        self.tool_Advanced.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_Advanced.setIcon(icon_advanced)
        self.tool_Advanced.setText("Advanced")
        self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_bar_2.addWidget(self.tool_Advanced)

        self.tool_bar_2.addSeparator()

        icon_user = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/user.png")
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal)
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Disabled)
        self.tool_User = QtWidgets.QToolButton()
        self.tool_User.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.tool_User.setIcon(icon_user)
        self.tool_User.setText("Anonymous")
        self.tool_User.setEnabled(False)
        self.tool_bar_2.addWidget(self.tool_User)

        self.toolBar.addWidget(self.tool_bar_2)

        self.toolBar.setContentsMargins(8, 4, 8, 4)

        # Glass container for the entire toolbar row --------------------------
        self.toolBarWidget = GlassControlsWidget()
        self.toolBarWidget.setLayout(self.toolBar)

        self.toolLayout.addWidget(self.toolBarWidget)
        self.toolLayout.addWidget(self.run_progress_bar)

        if SHOW_SIMPLE_CONTROLS:
            self.toolLayout.setContentsMargins(6, 6, 6, 0)
            self.centralwidget.setLayout(self.toolLayout)

            self.Layout_controls.removeWidget(self.infosave)
            self.Layout_controls.removeWidget(self.lTemp)
            self.Layout_controls.removeWidget(self.slTemp)
            self.Layout_controls.removeWidget(self.pTemp)
            self.Layout_controls.removeWidget(self.run_progress_bar)
            self.Layout_controls.removeWidget(self.lg)
            self.Layout_controls.removeWidget(self.lmail)
            self.Layout_controls.removeWidget(self.l4)
            self.Layout_controls.removeWidget(self.l3)
            self.Layout_controls.removeWidget(self.infostatus)

            self.advancedwidget = QtWidgets.QWidget()
            self.advancedwidget.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.WindowStaysOnTopHint)
            self.advancedwidget.setWhatsThis("These settings are for Advanced Users ONLY!")
            warningWidget = GlassWarningLabel(f"WARNING: {self.advancedwidget.whatsThis()}")
            warningLayout = QtWidgets.QVBoxLayout()
            warningLayout.addWidget(warningWidget)
            warningLayout.addLayout(self.gridLayout)
            self.advancedwidget.setLayout(warningLayout)
        else:
            self.centralwidget.setLayout(self.gridLayout)

        MainWindow1.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow1)

    def _update_progress_text(self):
        # get innerText from HTML in infobar
        plain_text = self.infobar.text()
        color = plain_text[plain_text.rindex("color=") + 6 : plain_text.rindex("color=") + 6 + 7]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[plain_text.index(">") + 1 :]
        plain_text = plain_text[0 : plain_text.rindex("<")]
        # remove any formatting tags: <b>, <i>, <u>
        while plain_text.rfind("<") != plain_text.find("<"):
            plain_text = plain_text[0 : plain_text.rindex("<")]
            plain_text = plain_text[plain_text.index(">") + 1 :]
        if len(plain_text) == 0:
            plain_text = "Progress: Not Started"
        else:
            plain_text = "Status: {}".format(plain_text)
        self.run_progress_bar.setFormat(plain_text)
        styleBar = _GLASS_PROGRESSBAR_QSS.replace(
            "color: #1a3050;", "color: {COLOR}; font-weight: bold;"
        ).replace("{COLOR}", color)
        self.run_progress_bar.setStyleSheet(styleBar)

    def _update_progress_value(self):
        if self.cBox_Source.currentIndex() == OperationType.measurement.value:
            pass
        else:
            self.run_progress_bar.setFormat("Progress: %p%")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(
            _translate(
                "MainWindow",
                "{} {} - Setup/Control".format(Constants.app_title, Constants.app_version),
            )
        )
        icon_path = os.path.join(Architecture.get_path(), "QATCH/icons/")
        MainWindow.setWindowIcon(QtGui.QIcon(os.path.join(icon_path, "qatch-icon.png")))
        self.advancedwidget.setWindowIcon(QtGui.QIcon(os.path.join(icon_path, "advanced.png")))
        self.advancedwidget.setWindowTitle(_translate("MainWindow2", "Advanced Settings"))
        self.pButton_Stop.setText(_translate("MainWindow", " STOP"))
        self.pButton_Start.setText(_translate("MainWindow", "START"))
        self.pButton_Clear.setText(_translate("MainWindow", "Clear Plots"))
        self.pButton_Reference.setText(_translate("MainWindow", "Set/Reset Reference"))
        self.pButton_ResetApp.setText(_translate("MainWindow", "Factory Defaults"))
        self.sBox_Samples.setSuffix(_translate("MainWindow", " / 5 min"))
        self.sBox_Samples.setPrefix(_translate("MainWindow", ""))
        self.chBox_export.setText(_translate("MainWindow", "Txt Export Sweep File"))
        self.chBox_freqHop.setText(_translate("MainWindow", "Mode Hop"))
        self.chBox_correctNoise.setText(_translate("MainWindow", "Show amplitude curve"))
        self.chBox_MultiAuto.setText(_translate("MainWindow", "Auto-detect channel count"))

    def action_next_port(self):
        """Method to handle advancing to the next port."""
        try:
            self.action_NextPortRow.setEnabled(False)

            controller_port = None
            for i in range(self.cBox_Port.count()):
                if self.cBox_Port.itemText(i).startswith("80:"):
                    controller_port = self.cBox_Port.itemData(i)
                    break

            if controller_port is None:
                Log.e("FLUX controller not found. Is it connected and powered on?")
                self.tool_NextPortRow.setIconError()
                self.action_NextPortRow.setEnabled(True)
                return

            next_port_num = self.tool_NextPortRow.value()

            if hasattr(self, "fluxThread"):
                if self.fluxThread.isRunning():
                    Log.d("Waiting for FLUX controller to stop.")
                    if not self.fluxThread.wait(msecs=3000):
                        Log.w(
                            "Prior Flux controller thread still busy; skipping new Next Port request."
                        )
                        self.tool_NextPortRow.setEnabled(True)
                        return
            Log.d("Starting FLUX controller thread.")
            self.fluxThread = QtCore.QThread()
            self.fluxWorker = FLUXControl()
            self.fluxWorker.set_ports(controller=controller_port, next_port=next_port_num)
            self.fluxWorker.moveToThread(self.fluxThread)
            self.fluxThread.worker = self.fluxWorker
            self.fluxThread.started.connect(self.fluxWorker.run)
            self.fluxWorker.finished.connect(self.fluxThread.quit)
            self.fluxWorker.result.connect(self.next_port_result)
            self.fluxThread.start()

        except Exception as e:
            Log.e(f"action_next_port ERROR: {e}")
            self.tool_NextPortRow.setIconError()
            self.action_NextPortRow.setEnabled(True)

    def next_port_result(self, success):
        try:
            self.action_NextPortRow.setEnabled(True)

            if success:
                self.parent.parent.active_multi_ch = self.tool_NextPortRow.value()
                self.parent.parent.set_multi_mode()
            else:
                self.tool_NextPortRow.setIconError()

                if PopUp.critical(
                    self,
                    "Next Port Failed",
                    "ERROR: Flux controller failed to move to the next port.",
                    btn1_text="Reset",
                ):
                    self.tool_NextPortRow.click()

        except Exception as e:
            Log.e(f"next_port_result ERROR: {e}")

    def action_initialize(self):
        """Method to handle initialization UI actions."""
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.calibration.value)
            if hasattr(self, "run_controls"):
                self.run_controls.set_running(False)
                self.run_controls.update_progress(0, 5, "Ready")
                self.run_controls.setEnabled(False)
            self.pButton_Start.clicked.emit()
            self.cal_initialized = True

    def action_start(self):
        """Method to handle start UI actions."""
        if self.pButton_Start.isEnabled():
            self.cBox_Source.setCurrentIndex(OperationType.measurement.value)
            self.pButton_Start.clicked.emit()

    def action_stop(self):
        """Method to handle stop UI actions."""
        if self.pButton_Stop.isEnabled():
            self.cal_initialized = False
            self.pButton_Stop.clicked.emit()

    def action_reset(self):
        """Method to handle reset UI actions."""
        if self.tool_TempControl.isChecked():
            self.tool_TempControl.setChecked(False)
            self.tool_TempControl.clicked.emit()  # triggers action_tempcontrol → collapse
        self.slTemp.setValue(25)
        if self.pButton_Start.isEnabled():
            self.pButton_Clear.clicked.emit()
            self.pButton_Refresh.clicked.emit()
        self.infostatus.setText("Program Status Standby")

        self.cal_initialized = False
        if hasattr(self, "run_controls"):
            self.run_controls.set_running(False)
            self.run_controls.update_progress(0, 5, "Idle")
            self.run_controls.setEnabled(False)

        self.tool_TempControl.setEnabled(self.cBox_Port.count() > 1)

    def _animate_temp_controller(self, expand: bool) -> None:
        """Smoothly expand or collapse the temperature controller panel."""
        _EXPANDED_W = 240
        start = self.tempController.maximumWidth()
        if start > _EXPANDED_W:
            start = _EXPANDED_W
        end = _EXPANDED_W if expand else 0

        if start == end:
            return

        self._set_temp_arrow(expand)

        self._temp_anim = QtCore.QPropertyAnimation(self.tempController, b"maximumWidth")
        self._temp_anim.setDuration(220)
        self._temp_anim.setStartValue(start)
        self._temp_anim.setEndValue(end)
        self._temp_anim.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self._temp_anim.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def _toggle_temp_controller(self) -> None:
        """Called by the in-panel arrow button to collapse or expand the panel."""
        self.tool_TempControl.setChecked(not self.tool_TempControl.isChecked())
        self.action_tempcontrol()

    def _set_temp_arrow(self, expand: bool) -> None:
        """Point the in-panel arrow button left (collapse) or right (expand).

        Loads ``QATCH/icons/arrow_left.svg`` / ``arrow_right.svg`` when present
        and sizes the icon to fill the button.  Falls back to Unicode arrows.
        """
        direction = "left" if expand else "right"
        svg_path = os.path.join(Architecture.get_path(), "QATCH", "icons", f"{direction}_arrow.svg")
        icon = QtGui.QIcon(svg_path)
        if not icon.isNull():
            self.tempArrowStrip.setIcon(icon)
            self.tempArrowStrip.setIconSize(QtCore.QSize(self.tempArrowStrip.width() or 14, 14))
            self.tempArrowStrip.setText("")
        else:
            self.tempArrowStrip.setIcon(QtGui.QIcon())
            self.tempArrowStrip.setText("‹" if expand else "›")

    def _update_temp_display(self, text: str) -> None:
        """Parse the combined temp string and update the split display + status bar.

        Expected format: ``"PV:25.03C SP:25.00C OP:[50%]"``
        Falls back gracefully when values are placeholder dashes.
        """
        parts: dict = {}
        for segment in text.split():
            if ":" in segment:
                key, val = segment.split(":", 1)
                parts[key] = val

        pv_str = parts.get("PV", "--.--C")
        sp_str = parts.get("SP", "--.--C")
        op_str = parts.get("OP", "----")

        self.lPV.setText(f"PV  {pv_str}")
        self.lSP.setText(f"SP  {sp_str}")
        self.lOP.setText(f"OP  {op_str}")

        # Determine status colour and descriptive label
        try:
            pv = float(pv_str.rstrip("C"))
            sp = float(sp_str.rstrip("C"))
            if abs(pv - sp) <= 0.5:
                status_text = "Ready"
                bg_colour = "rgba(60, 200, 90, 220)"
                text_colour = "rgba(255, 255, 255, 230)"
            elif pv < sp:
                status_text = "Heating to setpoint..."
                bg_colour = "rgba(240, 190, 0, 220)"
                text_colour = "rgba(30, 20, 0, 200)"
            else:
                status_text = "Cooling to setpoint..."
                bg_colour = "rgba(240, 140, 0, 220)"
                text_colour = "rgba(30, 20, 0, 200)"
        except ValueError:
            status_text = "Offline"
            bg_colour = "rgba(150, 155, 160, 120)"
            text_colour = "rgba(30, 40, 55, 160)"

        self.tempStatusBar.setText(status_text)
        self.tempStatusBar.setStyleSheet(
            f"QLabel {{ background: {bg_colour}; color: {text_colour}; "
            "border-radius: 2px; padding: 0 4px; font-weight: bold; }}"
        )

    def action_tempcontrol(self):
        is_checked = self.tool_TempControl.isChecked()
        self._animate_temp_controller(is_checked)
        if is_checked:
            if self.pTemp.text().find("Stop") < 0:
                self.pTemp.clicked.emit()
            # Focus the slider after the panel finishes expanding
            QtCore.QTimer.singleShot(230, self.slTemp.setFocus)
        else:
            if self.pTemp.text().find("Stop") >= 0:
                self.pTemp.clicked.emit()

    def action_tempcontrol_warn_start(self, event):
        self.event_windowPos = event.windowPos()
        self._warningTimer.start()

    def action_tempcontrol_warn_stop(self, event):
        self._warningTimer.stop()

    def action_tempcontrol_warn_now(self, event):
        self.event_windowPos = event.windowPos()
        self.action_tempcontrol_warning()

    def action_tempcontrol_warning(self):
        if self.tool_TempControl.isChecked() and not self.tool_TempControl.isEnabled():
            Log.w("WARNING: Temp Control mode cannot be changed during an active run.")
            if self.event_windowPos.x() >= self.tempController.mapToGlobal(QtCore.QPoint(0, 0)).x():
                Log.w(
                    'To adjust Temp Control: Press "Stop" first, then adjust setpoint accordingly.'
                )
            else:
                Log.w('To stop Temp Control: Press "Stop" first, then click "Temp Control" button.')

    def action_advanced(self, obj):
        if self.advancedwidget.isVisible():
            self.advancedwidget.hide()
        self.advancedwidget.move(0, 0)
        self.advancedwidget.show()
        self.pButton_PlateConfig.setFixedWidth(self.pButton_PlateConfig.height())

    def doPlateConfig(self):
        if hasattr(self, "wellPlateUI"):
            if self.wellPlateUI.isVisible():
                self.wellPlateUI.close()

        num_ports = self.cBox_Port.count() - 1
        if num_ports == 5:
            num_ports = 4
        i = self.cBox_Port.currentText()
        i = 0 if i.find(":") == -1 else int(i.split(":")[0], base=16)
        if i % 9 == i:
            well_width = 4
            well_height = 1
        else:
            well_width = 6
            well_height = 4
        num_channels = self.cBox_MultiMode.currentIndex() + 1
        if num_ports not in [well_width, well_height] or num_ports == 1:
            PopUp.warning(
                self.parent,
                "Plate Configuration",
                f"<b>Multiplex device(s) are required for plate configuration.</b><br/>"
                + f"You must have exactly 4 device ports connected for this mode.<br/>"
                + f"Currently connected device port count is: {num_ports} (not 4)",
            )
        else:
            self.wellPlateUI = WellPlate(well_width, well_height, num_channels)
