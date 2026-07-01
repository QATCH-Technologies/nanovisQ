"""
QATCH.ui.interfaces.ui_controls
================================
UI definition and behaviour for the Controls Window.

This module owns the `UIControls` class, which builds and manages every
widget that appears in the floating controls bar shown below the main plot
area.  It is the logical successor to the auto-generated `mainWindow_ui`
file, hand-crafted to support two parallel layout modes:

* **Toolbar (simple) mode** - the default production layout: a single
  horizontal toolbar row with Initialize / Run / Reset / Temp-Control /
  Advanced / Account buttons and a collapsible TEC side-panel.
* **Classic grid mode** - the legacy, full-featured `QGridLayout` kept for
  reference and used when `SHOW_SIMPLE_CONTROLS` is `False`.

The module also contains the device-configuration editor (name, position ID,
temperature calibration, and lid POGO timing), the advanced-options panel,
the well-plate configuration dialog launcher, and all associated action
handlers, signal wiring, and helper widgets.

Author(s):
    Alexander Ross  <alexander.ross@qatchtech.com>
    Paul MacNichol  <paul.macnichol@qatchtech.com>

Date:
    2026-06-30
"""

import os
import re
from time import monotonic
from typing import TYPE_CHECKING, Any

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDesktopWidget

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.fileStorage import FileStorage
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants, OperationType, UserRoles
from QATCH.processors.Device import serial as DeviceSerial
from QATCH.ui.components import (
    AnimatedComboBox,
    AnimatedDoubleSpinBox,
    BorderlessActionButton,
    FLUXControl,
    GlassPushButton,
    GlassToggle,
    NumberIconButton,
    RunControls,
)
from QATCH.ui.labels import (
    DeviceConfigLabel,
    HeaderLabel,
    SectionHeader,
    StatusLabel,
    TemperatureLabel,
)
from QATCH.ui.popUp import PopUp
from QATCH.ui.styles.theme_manager import ThemeManager
from QATCH.ui.widgets import (
    AdvancedMainWidget,
    ControlsWidget,
    SavedStateDot,
    UserProfilesManagerWidget,
    WellPlate,
)
from QATCH.ui.widgets.account_popup import AccountPopup

if TYPE_CHECKING:
    from QATCH.ui.mainWindow import MainWindow
    from QATCH.ui.windows import ControlsWindow


def _tok_css(rgba: tuple) -> str:
    r, g, b, a = rgba
    if a == 255:
        return f"#{r:02X}{g:02X}{b:02X}"
    return f"rgba({r}, {g}, {b}, {a})"


def _hairline() -> QtWidgets.QFrame:
    """Creates a 1px divider matching the account dropdown's hairline separators.

    This factory function generates a horizontal `QFrame` styled as a subtle
    divider. It is designed to be used for visually grouping elements within
    the UI while maintaining a clean, minimalist aesthetic.

    Returns:
        QtWidgets.QFrame: A configured frame object representing the hairline.
    """
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setObjectName("CtrlHairline")
    return line


class _FieldStateAction:
    """A minimal stand-in for a QLineEdit trailing QAction.

    The device-config save/reset logic tracks each field's state purely through
    a QAction's `iconText()` ("blank" / "unsaved" / "saved") and occasionally
    `setIcon()`. A bare QWidget like :class:`_RangeSliderField` has no such
    line-edit action, so this lightweight object provides the same tiny surface
    (`setIcon` / `setIconText` / `iconText`) and nothing else. It is used
    as the authoritative state holder for the slider fields, keeping every
    existing call site working without special-casing.
    """

    def __init__(self) -> None:
        """Initializes the _FieldStateAction with default state."""
        self._icon_text = "blank"
        self._icon = None

    def setIcon(self, icon) -> None:  # noqa: N802
        """Sets the icon for the action.

        Args:
            icon: The icon object to be stored.
        """
        self._icon = icon

    def icon(self):
        """Returns the current icon.

        Returns:
            The stored icon object.
        """
        return self._icon

    def setIconText(self, text: str) -> None:  # noqa: N802
        """Sets the descriptive text for the icon.

        Args:
            text: The state string (e.g., "blank", "unsaved", "saved").
        """
        self._icon_text = text or ""

    def iconText(self) -> str:  # noqa: N802
        """Returns the current icon text.

        Returns:
            The stored state string.
        """
        return self._icon_text


class _RangeSliderField(QtWidgets.QWidget):
    """A horizontal slider paired with a live, editable numeric box.

    Replaces the preset dropdown + read-only box for the Lid POGO calibration
    fields. Dragging the slider updates the number box; typing in the number box
    (or stepping it) moves the slider. A single `valueChanged(int)` signal is
    emitted on any change so callers can mark the field unsaved.

    The numeric box exposes `text()` / `setText()` and the slider range is
    configurable, so the surrounding save/reset/default logic only needs to read
    and write a string value (mirroring the old read-only QLineEdit contract).

    Attributes:
        valueChanged (pyqtSignal): Emitted when the slider or spin box value changes.
            Emits an integer representing the new value.
    """

    valueChanged = QtCore.pyqtSignal(int)

    def __init__(
        self,
        minimum: int,
        maximum: int,
        suffix: str = "",
        parent=None,
        up_icon_path: str = "",
        down_icon_path: str = "",
    ) -> None:
        """Initializes the _RangeSliderField.

        Args:
            minimum: The minimum allowed value for the slider and spin box.
            maximum: The maximum allowed value for the slider and spin box.
            suffix: A string suffix to append to the spin box value.
            parent: The parent widget, if any.
            up_icon_path: File path for the spin box's increment icon.
            down_icon_path: File path for the spin box's decrement icon.
        """
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setStyleSheet("background: transparent;")
        self._suffix = suffix
        self._guard = False  # re-entrancy guard for cross-updates

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setObjectName("RangeSlider")
        self.slider.setMinimum(minimum)
        self.slider.setMaximum(maximum)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(5)
        self.slider.setMinimumWidth(120)
        self.slider.valueChanged.connect(self._on_slider)

        # Animated spin box (integer-configured: 0 decimals, step 1) with custom
        # up/down chevrons, matching the temperature-calibration inputs.
        self.spin = AnimatedDoubleSpinBox(
            up_icon_path=up_icon_path,
            down_icon_path=down_icon_path,
        )
        self.spin.setDecimals(0)
        self.spin.setMinimum(float(minimum))
        self.spin.setMaximum(float(maximum))
        self.spin.setSingleStep(1)
        self.spin.setSuffix(suffix)
        self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.spin.setFixedWidth(78)
        self.spin.valueChanged.connect(self._on_spin)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin, 0)

    def addAction(self, *args, **kwargs):  # noqa: N802
        """Mimics `QLineEdit.addAction(icon, position)`.

        The device-config code adds a trailing action to each input solely to
        carry save-state via `iconText()`. This widget has no line-edit, so we
        return a lightweight :class:`_FieldStateAction` instead of touching the
        QWidget action list (which only accepts real QActions and would raise on
        a QIcon). If called with a real QAction (the QWidget signature), defer
        to the base implementation.

        Returns:
            A _FieldStateAction if an icon is passed, otherwise the result of
            the base QWidget.addAction implementation.
        """
        if args and isinstance(args[0], QtGui.QIcon):
            return _FieldStateAction()
        return super().addAction(*args, **kwargs)

    def _on_slider(self, v: int) -> None:
        """Handles slider changes by updating the spin box and emitting signals.

        Args:
            v (int): value to change spinbox to.
        """
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(v)
        self._guard = False
        self.valueChanged.emit(v)

    def _on_spin(self, v: Any) -> None:
        """Handles spin box changes by updating the slider and emitting signals.

        Args:
            v (Any): Value to set slider to.
        """
        if self._guard:
            return
        iv = int(round(float(v)))
        self._guard = True
        self.slider.setValue(iv)
        self._guard = False
        self.valueChanged.emit(iv)

    def value(self) -> int:
        """Returns the current numeric value.

        Returns:
            int: the current numeric value set in the spinbox.
        """
        return int(round(float(self.spin.value())))

    def setValue(self, v: int) -> None:  # noqa: N802
        """Sets the current value for both the slider and spin box.

        Args:
            v: The integer value to set.
        """
        self._guard = True
        try:
            iv = int(round(float(v)))
        except (TypeError, ValueError):
            iv = self.slider.minimum()
        self.slider.setValue(iv)
        self.spin.setValue(float(iv))
        self._guard = False

    def text(self) -> str:  # noqa: N802
        """Returns the current value as a string.

        Returns:
            str: Value of the slider as a string.
        """
        return str(self.value())

    def setText(self, text: str) -> None:  # noqa: N802
        """Sets the value from a string input.

        Args:
            text: A string representation of the integer value.
        """
        try:
            self.setValue(int(round(float(str(text).strip()))))
        except (TypeError, ValueError):
            pass

    def hasAcceptableInput(self) -> bool:  # noqa: N802
        """Checks if the current value is within the defined bounds.

        Returns:
            True if the value is within range, False otherwise.
        """
        return self.slider.minimum() <= self.value() <= self.slider.maximum()

    def setPlaceholderText(self, text: str) -> None:  # noqa: N802
        """Placeholder text is not supported; this method is a no-op."""
        return

    def clear(self) -> None:
        """Resets the value to the slider's minimum."""
        self.setValue(self.slider.minimum())


class LabeledToggle(QtWidgets.QWidget):
    """A GlassToggle paired with a text label in a horizontal row.

    Exposes the subset of the QCheckBox API used by the rest of the app
    (`isChecked`, `setChecked`, `setEnabled`, `setText`, `toggled`)
    so it can stand in for a checkbox without touching call sites.

    Attributes:
        toggled (pyqtSignal): A signal forwarded from the internal GlassToggle
            that is emitted when the toggle state changes.
    """

    def __init__(self, text: str = "", parent=None, *, label_left: bool = False) -> None:
        """Initializes the LabeledToggle.

        Args:
            text: The label text to display next to the toggle.
            parent: The parent widget, if any.
            label_left: If True, positions the label to the left of the toggle;
                otherwise, positions it to the right.
        """
        super().__init__(parent)
        self.toggle = GlassToggle(self)
        self.label = QtWidgets.QLabel(text, self)
        self.label.setObjectName("CtrlToggleLabel")
        self.label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        if label_left:
            lay.addWidget(self.label)
            lay.addWidget(self.toggle)
            lay.addStretch()
        else:
            lay.addWidget(self.toggle)
            lay.addWidget(self.label)
            lay.addStretch()
        self.toggled = self.toggle.toggled

    def isChecked(self) -> bool:
        """Returns the current checked state of the toggle."""
        return self.toggle.isChecked()

    def setChecked(self, checked: bool) -> None:
        """Sets the checked state of the toggle.

        Args:
            checked: The boolean state to apply.
        """
        self.toggle.setChecked(checked)

    def setText(self, text: str) -> None:
        """Sets the text for the label.

        Args:
            text: The new label string.
        """
        self.label.setText(text)

    def text(self) -> str:
        """Returns the current label text."""
        return self.label.text()

    def setEnabled(self, enabled: bool) -> None:
        """Sets the enabled state for the widget and its children.

        Args:
            enabled: The boolean state to apply to the entire widget and children.
        """
        super().setEnabled(enabled)
        self.toggle.setEnabled(enabled)
        self.label.setEnabled(enabled)


class _PerspectiveAnimator(QtCore.QObject):
    """Plays a gentle entrance on a perspective container each time it is shown.

    IMPORTANT: this deliberately does NOT use QGraphicsOpacityEffect. Wrapping a
    container that holds custom-painted children (combos, toggles,
    buttons) in a graphics effect caches them into an offscreen pixmap, which
    causes ghosting, duplicated section labels, and widgets vanishing on hover.
    Instead the fade is applied to the top-level popup window via
    setWindowOpacity (no pixmap caching), paired with a brief top-margin slide.

    Attributes:
        _container (QtWidgets.QWidget): The target widget container to animate.
        _slide (QtCore.QVariantAnimation): Animation for the vertical sliding movement.
        _fade (QtCore.QVariantAnimation): Animation for the window opacity transition.
    """

    def __init__(self, container: QtWidgets.QWidget) -> None:
        """Initializes the animator and attaches an event filter to the container.

        Args:
            container: The widget container whose window will be animated.
        """
        super().__init__(container)
        self._container = container

        # Guard flag: ensures only one _begin_slide is ever scheduled per show
        # event cycle.  Multiple ShowEvents can fire on the container during a
        # single open (e.g. from set_page's show() and set_advanced_perspective's
        # show()), particularly on the first open when the container transitions
        # from a hidden top-level widget.  Without this guard the animation
        # starts twice, producing the "doubly renders and animates" glitch.
        self._start_pending: bool = False

        # Slide the whole popup window DOWN into place (start 12px above the
        # final position), matching the account menu. Animating the inner
        # top-margin instead made content rise up from the bottom, which read
        # as a bottom-up animation.
        self._slide = QtCore.QVariantAnimation(self)
        self._slide.setDuration(220)
        self._slide.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._slide.setStartValue(0.0)
        self._slide.setEndValue(1.0)
        self._slide.valueChanged.connect(self._apply_slide)
        self._slide_from = None
        self._slide_to = None
        self._slide_offset = 12

        self._fade = QtCore.QVariantAnimation(self)
        self._fade.setDuration(200)
        self._fade.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.valueChanged.connect(self._apply_fade)
        self._fade.finished.connect(self._finish_fade)

        container.installEventFilter(self)

    def _apply_slide(self, t: float) -> None:
        """Calculates and applies the window position during the slide animation.

        Args:
            t (float): Normalized time value (0.0 to 1.0).
        """
        win = self._container.window()
        if win is None or self._slide_to is None:
            return
        slide_from = self._slide_from
        if slide_from is not None:
            x = self._slide_to.x()
            y = int(slide_from.y() + (self._slide_to.y() - slide_from.y()) * float(t))
        else:
            y = self._slide_to.y()
        win.move(x, y)

    def _apply_fade(self, v: float) -> None:
        """Applies the window opacity during the fade animation.

        Args:
            v (float): Opacity value (0.0 to 1.0).
        """
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(float(v))

    def _finish_fade(self) -> None:
        """Ensures the window is fully opaque and at the final position upon finish."""
        win = self._container.window()
        if win is not None:
            win.setWindowOpacity(1.0)
            if self._slide_to is not None:
                win.move(self._slide_to)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """Filters show events to trigger animations after the widget is visible.

        Args:
            obj: The object the event is sent to.
            event: The QEvent object.

        Returns:
            True if the event was handled, otherwise the base class result.
        """
        if obj is self._container and event.type() == QtCore.QEvent.Type.Show:
            # Deduplicate: only schedule _begin_slide once per open cycle.
            # The fade is NOT started here — _begin_slide starts both animations
            # together after the popup window is confirmed visible, preventing
            # the fade from running ahead of the slide (and ahead of show()).
            if not self._start_pending:
                self._start_pending = True
                QtCore.QTimer.singleShot(0, self._begin_slide)
        return super().eventFilter(obj, event)

    def _begin_slide(self) -> None:
        """Calculates start/end positions and starts both fade and slide animations.

        Called once per open cycle via a zero-delay singleShot, after all
        queued ShowEvents have been processed.  Resets the _start_pending
        guard so the next open cycle can arm again.
        """
        self._start_pending = False
        win = self._container.window()
        if win is None or not win.isVisible():
            return
        final_pos = win.pos()
        self._slide_to = QtCore.QPoint(final_pos)
        self._slide_from = QtCore.QPoint(final_pos.x(), final_pos.y() - self._slide_offset)
        win.move(self._slide_from)
        self._fade.stop()
        self._fade.start()
        self._slide.stop()
        self._slide.start()


class UIControls:
    """UI definition and controller for the QATCH Controls Window.

    `UIControls` is mixed into `ControlsWindow` (a `QMainWindow`
    subclass) via the standard Qt Designer pattern: `setupUi` is called
    once during construction to build all widgets, and the remaining methods
    implement the associated actions, signals, and helper logic.

    Layout architecture
    -------------------
    Two complete sets of widgets are built during `setupUi`:

    * **`Layout_controls`** - a `QGridLayout` that places every control in a fixed grid.
      Always constructed so external code can reference every widget by name;
      widgets unused in the active mode are detached from the layout before the
       window is shown.
    * **Toolbar layout** (`toolLayout` / `toolBar`) - the
      production-default single-row toolbar plus a collapsible TEC
      side-panel and a secondary toolbar for Advanced / Account actions.

    Sub-panels
    --------------
    * **Advanced panel** - a floating `AdvancedMainWidget` popup anchored
      to the toolbar.  Built lazily on first open; its layout is assembled
      eagerly by `_build_advanced_layout` so widget references are always
      valid.
    * **Device-info editor** - a sectioned two-column form (name, position
      ID, temperature calibration, lid POGO timing) that slides in as a
      "perspective" inside the advanced popup.
    * **Well-plate configurator** - launched by `doPlateConfig`; appears
      as a standalone `WellPlate` dialog sized to the detected multiplex
      geometry.
    """

    def setupUi(self, MainWindow1):
        """Build and assemble all widgets for the controls window.

        Execution is grouped into seven phases:

        1. **Window & layout scaffolding** - geometry, shared constants, and the
           two top-level layout containers (`gridLayout` / `Layout_controls`
           for the classic grid view; `toolLayout` / `toolBar` for the
           simple toolbar view).

        2. **Classic grid-layout widgets** (`Layout_controls`) - every control
           widget used in the full-featured, legacy grid view.  These are always
           created so external code can reference them even when
           `SHOW_SIMPLE_CONTROLS` is active.  The mode that is NOT active has
           its widgets removed from the grid before the window is shown.

        3. **Toolbar (simple) layout** - the primary visible row of controls
           (Initialize / Run / Reset / TempControl / Advanced / Account).
           Activated when `SHOW_SIMPLE_CONTROLS` is `True`.

        4. **TEC temperature side-panel** - a collapsible widget that slides in
           next to the toolbar when temperature control is toggled on.

        5. **Icon-state singletons** - the three `QIcon` objects
           (`blankIcon`, `savedIcon`, `unsavedIcon`) used by device-config
           field dots to reflect save state.

        6. **Device-info container** - the configuration editor for the
           connected device (name, position ID, temperature calibration, lid
           POGO timing).  Laid out as a sectioned, two-column panel that mirrors
           the Advanced perspective.

        7. **Finalization** - sets the central widget, calls `retranslateUi`,
           and wires remaining Qt slots via `connectSlotsByName`.

        Args:
            MainWindow1: The `ControlsWindow` parent that receives the
                assembled UI.  Stored as `self.parent`.
        """
        # window geometry & shared layout scaffolding

        USE_FULLSCREEN = QDesktopWidget().availableGeometry().width() == 2880
        SHOW_SIMPLE_CONTROLS = True
        self.cal_initialized = False
        self.parent: "ControlsWindow" = MainWindow1

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

        # Shared chevron icon for all animated combo boxes.
        self._combo_chevron = os.path.join(
            Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
        )

        # grid-layout widgets

        # frequency/quartz combobox
        self.cBox_Speed = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_Speed.setEditable(False)
        self.cBox_Speed.setObjectName("cBox_Speed")
        if USE_FULLSCREEN:
            self.cBox_Speed.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Speed, 4, 1, 1, 1)

        # Shared control-button sizing (thick enough for icon + label).
        _CTRL_BTN_H = 40
        _CTRL_ICON = QtCore.QSize(20, 20)
        self.pButton_Stop = GlassPushButton(variant="neutral")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "stop-filled.svg")
        self.pButton_Stop.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Stop.setIconSize(_CTRL_ICON)
        self.pButton_Stop.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Stop.setFixedHeight(_CTRL_BTN_H)
        self.pButton_Stop.setObjectName("pButton_Stop")
        self.pButton_Stop.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_Stop, 3, 6, 1, 1)

        # COM port combobox
        self.cBox_Port = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_Port.setEditable(False)
        self.cBox_Port.setObjectName("cBox_Port")
        if USE_FULLSCREEN:
            self.cBox_Port.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Port, 2, 1, 1, 1)

        # Identify button
        _CIRCLE_D = 34
        self.pButton_ID = GlassPushButton(variant="default")
        self.pButton_ID.setToolTip("Identify selected Serial COM Port")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "search.svg")
        self.pButton_ID.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_ID.setIconSize(QtCore.QSize(18, 18))
        self.pButton_ID.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.pButton_ID.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_ID.setObjectName("pButton_ID")
        self.Layout_controls.addWidget(self.pButton_ID, 2, 2, 1, 1)

        # Refresh button
        self.pButton_Refresh = GlassPushButton(variant="default")
        self.pButton_Refresh.setToolTip("Refresh Serial COM Port list")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "refresh-cw.svg")
        self.pButton_Refresh.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Refresh.setIconSize(QtCore.QSize(18, 18))
        self.pButton_Refresh.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.pButton_Refresh.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_Refresh.setObjectName("pButton_Refresh")
        self.Layout_controls.addWidget(self.pButton_Refresh, 2, 3, 1, 1)

        # Configure button - replaces the in-dropdown "Configure..." item.
        # Wired to the main window's device-info handler in mainWindow.py.
        self.pButton_Configure = GlassPushButton(variant="default")
        self.pButton_Configure.setToolTip("Configure device / position info")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "gear.svg")
        self.pButton_Configure.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Configure.setIconSize(QtCore.QSize(18, 18))
        self.pButton_Configure.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.pButton_Configure.setFixedSize(_CIRCLE_D, _CIRCLE_D)
        self.pButton_Configure.setObjectName("pButton_Configure")
        self.Layout_controls.addWidget(self.pButton_Configure, 2, 4, 1, 1)
        # Reflect device-connection state ('No device connected' when empty).
        self.cBox_Port.currentIndexChanged.connect(self._update_configure_enabled)
        self._update_configure_enabled()

        # Operation mode - source
        self.cBox_Source = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_Source.setObjectName("cBox_Source")
        if USE_FULLSCREEN:
            self.cBox_Source.setFixedHeight(50)
        self.Layout_controls.addWidget(self.cBox_Source, 2, 0, 1, 1)

        # Frequency hopping toggle
        self.chBox_freqHop = LabeledToggle("Mode Hop")
        self.chBox_freqHop.setEnabled(True)
        self.chBox_freqHop.setChecked(False)
        self.chBox_freqHop.setObjectName("chBox_freqHop")
        self.Layout_controls.addWidget(self.chBox_freqHop, 4, 2, 1, 2)

        # Noise correction toggle
        self.chBox_correctNoise = LabeledToggle("Show amplitude curve")
        self.chBox_correctNoise.setEnabled(True)
        self.chBox_correctNoise.setChecked(True)
        self.chBox_correctNoise.setObjectName("chBox_correctNoise")
        self.Layout_controls.addWidget(self.chBox_correctNoise, 5, 1, 1, 3)

        # Cartridge Auto-Lock
        self.l9 = HeaderLabel("Cartridge Auto-Lock")
        if USE_FULLSCREEN:
            self.l9.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l9, 1, 4, 1, 1)

        # Cartridge Controls
        self.toggle_Cartridge = GlassToggle()
        self.toggle_Cartridge.setToolTip("""
            <b><u>Auto-Lock Mode:</u></b><br/>
            <b>Automatic</b> (on): locks before init/run; useful if the user forgets.<br/>
            <b>Manual</b> (off): you control lock position; must lock before init/run.
            """)
        self.toggle_Cartridge.setChecked(True)  # default: Automatic

        self.lbl_lock_manual = QtWidgets.QLabel("Manual")
        self.lbl_lock_auto = QtWidgets.QLabel("Automatic")
        for _lbl in (self.lbl_lock_manual, self.lbl_lock_auto):
            _lbl.setObjectName("CtrlToggleLabel")
            _lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self.layMode = QtWidgets.QHBoxLayout()
        self.layMode.setContentsMargins(0, 0, 0, 0)
        self.layMode.setSpacing(8)
        self.layMode.addWidget(self.lbl_lock_manual)
        self.layMode.addWidget(self.toggle_Cartridge)
        self.layMode.addWidget(self.lbl_lock_auto)
        self.layMode.addStretch()
        self.grpMode = QtWidgets.QGroupBox("Auto-Lock Mode:")
        self.grpMode.setLayout(self.layMode)
        self.Layout_controls.addWidget(self.grpMode, 2, 4, 3, 1)

        # start button
        self.pButton_Start = GlassPushButton(variant="neutral")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "play-filled.svg")
        self.pButton_Start.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Start.setIconSize(_CTRL_ICON)
        self.pButton_Start.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Start.setFixedHeight(_CTRL_BTN_H)
        self.pButton_Start.setObjectName("pButton_Start")
        self.pButton_Start.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_Start, 2, 6, 1, 1)

        # Add signal for Run Controls UI to handle START from Advanced menu
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

        # clear plots button
        self.pButton_Clear = GlassPushButton(variant="neutral")
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "clear-plot.svg")
        self.pButton_Clear.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_Clear.setIconSize(_CTRL_ICON)
        self.pButton_Clear.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_Clear.setFixedHeight(_CTRL_BTN_H)
        self.pButton_Clear.setObjectName("pButton_Clear")
        self.pButton_Clear.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_Clear, 2, 5, 1, 1)

        # Plot mode toggle
        self.toggle_PlotMode = GlassToggle()
        self.toggle_PlotMode.setToolTip(
            "<b>Plot Mode</b><br/>Off: Absolute &nbsp;|&nbsp; On: Reference"
        )
        self.toggle_PlotMode.setChecked(False)  # default: Absolute

        self.lbl_plot_absolute = QtWidgets.QLabel("Absolute")
        self.lbl_plot_reference = QtWidgets.QLabel("Reference")
        for _lbl in (self.lbl_plot_absolute, self.lbl_plot_reference):
            _lbl.setObjectName("CtrlToggleLabel")
            _lbl.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        # Alias: existing code refers to pButton_Reference for enable/check/click.
        self.pButton_Reference = self.toggle_PlotMode

        # restore factory defaults
        self.pButton_ResetApp = GlassPushButton(variant="neutral")
        self.pButton_ResetApp.setIconSize(_CTRL_ICON)
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "factory-reset.svg")
        self.pButton_ResetApp.setIcon(QtGui.QIcon(QtGui.QPixmap(icon_path)))
        self.pButton_ResetApp.setMinimumSize(QtCore.QSize(0, 0))
        self.pButton_ResetApp.setFixedHeight(_CTRL_BTN_H)
        self.pButton_ResetApp.setObjectName("pButton_ResetApp")
        self.pButton_ResetApp.set_icon_left(True)
        self.Layout_controls.addWidget(self.pButton_ResetApp, 4, 5, 1, 1)

        # samples SpinBox
        self.sBox_Samples = QtWidgets.QSpinBox()
        self.sBox_Samples.setMinimum(1)
        self.sBox_Samples.setMaximum(100000)
        self.sBox_Samples.setProperty("value", 500)
        self.sBox_Samples.setObjectName("sBox_Samples")
        self.sBox_Samples.setVisible(False)
        self.Layout_controls.addWidget(self.sBox_Samples, 2, 4, 1, 1)

        # export file CheckBox
        self.chBox_export = QtWidgets.QCheckBox()
        self.chBox_export.setEnabled(True)
        self.chBox_export.setObjectName("chBox_export")
        self.chBox_export.setVisible(False)
        self.Layout_controls.addWidget(self.chBox_export, 4, 4, 1, 1)

        # temperature Control slider
        self.slTemp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slTemp.setMinimum(8)
        self.slTemp.setMaximum(40)
        self.slTemp.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slTemp.setTickInterval(1)
        self.slTemp.setSingleStep(1)
        self.slTemp.setPageStep(5)
        self.Layout_controls.addWidget(self.slTemp, 3, 4, 1, 1)

        # temperature Control label
        self.lTemp = TemperatureLabel()
        self.lTemp.setText("PV:--.--C SP:--.--C OP:----")
        self.lTemp.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lTemp.setFont(QtGui.QFont("Consolas", -1))
        self.lTemp.hide()
        self.Layout_controls.addWidget(self.lTemp, 2, 4, 1, 1)

        # temperature Control button
        self.pTemp = QtWidgets.QPushButton()
        self.pTemp.setText("Start Temp Control")
        if USE_FULLSCREEN:
            self.pTemp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.pTemp, 4, 4, 1, 1)

        # Control Buttons
        self.l = HeaderLabel("Control Buttons")
        if USE_FULLSCREEN:
            self.l.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l, 1, 5, 1, 2)

        # Operation Mode
        self.l0 = HeaderLabel("Operation Mode")
        if USE_FULLSCREEN:
            self.l0.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l0, 1, 0, 1, 1)

        # Resonance Frequency / Quartz Sensor
        self.l2 = HeaderLabel("Resonance Frequency / Quartz Sensor")
        if USE_FULLSCREEN:
            self.l2.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l2, 3, 1, 1, 3)

        # Serial COM Port
        self.l1 = HeaderLabel("Serial COM Port")
        if USE_FULLSCREEN:
            self.l1.setFixedHeight(50)
        self.Layout_controls.addWidget(self.l1, 1, 1, 1, 3)

        # logo
        self.l3 = QtWidgets.QLabel()
        self.l3.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.Layout_controls.addWidget(self.l3, 4, 7, 1, 1)
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-logo_full.jpg")
        if USE_FULLSCREEN:
            pixmap = QtGui.QPixmap(icon_path)
            pixmap = pixmap.scaled(
                250,
                50,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
            self.l3.setPixmap(pixmap)
        else:
            self.l3.setPixmap(QtGui.QPixmap(icon_path))

        # qatch link
        self.l4 = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.l4, 3, 7, 1, 1)

        def link(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.l4.linkActivated.connect(link)
        self.l4.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.l4.setText(
            '<a href="https://qatchtech.com/"> <font size=4 color=#008EC0 >qatchtech.com</font>'
        )

        # info@qatchtech.com Mail
        self.lmail = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lmail, 2, 7, 1, 1)

        def linkmail(linkStr):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(linkStr))

        self.lmail.linkActivated.connect(linkmail)
        self.lmail.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.lmail.setText(
            '<a href="mailto:info@qatchtech.com"> <font color=#008EC0 >info@qatchtech.com</font>'
        )

        # software user guide
        self.lg = QtWidgets.QLabel()
        self.Layout_controls.addWidget(self.lg, 1, 7, 1, 1)
        self.lg.linkActivated.connect(link)  # reuses link() defined above
        self.lg.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.lg.setText(
            '<a href="file://{}/docs/userguide.pdf"> <font color=#008EC0 >User Guide</font>'.format(
                Architecture.get_path()
            )
        )

        # Save file / TEC Temperature Control header
        self.infosave = HeaderLabel("TEC Temperature Control")
        if USE_FULLSCREEN:
            self.infosave.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infosave, 1, 4, 1, 1)

        # Program Status standby
        self.infostatus = StatusLabel()
        self.infostatus.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.infostatus.setText("Program Status Standby")
        if USE_FULLSCREEN:
            self.infostatus.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infostatus, 5, 5, 1, 2)

        # Infobar
        self.infobar = QtWidgets.QLineEdit()
        self.infobar.setReadOnly(True)
        self.infobar_label = StatusLabel()
        self.infobar.textChanged.connect(self.infobar_label.setText)
        if SHOW_SIMPLE_CONTROLS:
            self.infobar.textChanged.connect(self._update_progress_text)
        if USE_FULLSCREEN:
            self.infobar_label.setFixedHeight(50)
        self.Layout_controls.addWidget(self.infobar_label, 0, 0, 1, 7)

        # Multiplex
        self.lmp = HeaderLabel("Multiplex Mode")
        if USE_FULLSCREEN:
            self.lmp.setFixedHeight(50)
        self.Layout_controls.addWidget(self.lmp, 3, 0, 1, 1)

        self.cBox_MultiMode = AnimatedComboBox(icon_path=self._combo_chevron)
        self.cBox_MultiMode.setObjectName("cBox_MultiMode")
        self.cBox_MultiMode.addItems(["1 Channel", "2 Channels", "3 Channels", "4 Channels"])
        self.cBox_MultiMode.setCurrentIndex(0)
        if USE_FULLSCREEN:
            self.cBox_MultiMode.setFixedHeight(50)

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self.pButton_PlateConfig = GlassPushButton(variant="default")
        self.pButton_PlateConfig.setIcon(QtGui.QIcon(os.path.join(icon_path, "gear.svg")))
        self.pButton_PlateConfig.setIconSize(QtCore.QSize(18, 18))
        self.pButton_PlateConfig.setToolTip("Plate Configuration...")
        self.pButton_PlateConfig.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.pButton_PlateConfig.setFixedSize(_CIRCLE_D, _CIRCLE_D)  # square -> circle
        self.pButton_PlateConfig.clicked.connect(self.doPlateConfig)
        self.hBox_MultiConfig = QtWidgets.QHBoxLayout()
        self.hBox_MultiConfig.addWidget(self.cBox_MultiMode, 3)
        self.hBox_MultiConfig.addWidget(self.pButton_PlateConfig, 1)
        self.Layout_controls.addLayout(self.hBox_MultiConfig, 4, 0, 1, 1)

        # Disable Plate Configuration when only a single channel is selected or
        # available - a 1-channel setup has no plate layout to configure.
        self.cBox_MultiMode.currentIndexChanged.connect(self._update_plate_config_enabled)
        self._update_plate_config_enabled()

        self.chBox_MultiAuto = LabeledToggle("Auto-detect channel count")
        self.chBox_MultiAuto.setEnabled(True)
        self.chBox_MultiAuto.setChecked(True)
        self.chBox_MultiAuto.setObjectName("chBox_MultiAuto")
        self.Layout_controls.addWidget(self.chBox_MultiAuto, 5, 0, 1, 1)

        # Progressbar
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setGeometry(QtCore.QRect(0, 0, 50, 10))
        self.run_progress_bar.setObjectName("progressBar")

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

        # toolbar layout

        self.toolLayout = QtWidgets.QVBoxLayout()
        self.toolBar = QtWidgets.QHBoxLayout()

        self.tool_bar = QtWidgets.QToolBar()
        self.tool_bar.setObjectName("CtrlToolBar")
        self.tool_bar.setIconSize(QtCore.QSize(50, 30))

        self.tool_NextPortRow = NumberIconButton()
        self.tool_NextPortRow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # type: ignore
        self.tool_NextPortRow.setText("Next Port")
        self.tool_NextPortRow.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_NextPortRow.clicked.connect(self.action_next_port)
        self.action_NextPortRow = self.tool_bar.addWidget(self.tool_NextPortRow)

        self.action_NextPortSep = self.tool_bar.addSeparator()

        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons")

        icon_init = QtGui.QIcon()
        icon_init.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "speedometer.svg")), QtGui.QIcon.Mode.Normal
        )
        self.tool_Initialize = QtWidgets.QToolButton()
        self.tool_Initialize.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # type: ignore
        self.tool_Initialize.setIcon(icon_init)
        self.tool_Initialize.setText("Initialize")
        self.tool_Initialize.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_Initialize.clicked.connect(self.action_initialize)
        self.tool_bar.addWidget(self.tool_Initialize)

        self.tool_bar.addSeparator()

        # RunControls composite widget
        self.run_controls = RunControls()
        self.run_controls.startRequested.connect(self.action_start)
        self.run_controls.stopRequested.connect(self.action_stop)
        self.run_controls.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.run_controls.setEnabled(False)
        self.tool_Start = self.run_controls  # backward-compat alias
        self.tool_Stop = self.run_controls
        self.tool_bar.addWidget(self.run_controls)
        self.tool_bar.addSeparator()

        icon_reset = QtGui.QIcon()
        icon_reset.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "reset.svg")), QtGui.QIcon.Mode.Normal
        )
        self.tool_Reset = QtWidgets.QToolButton()
        self.tool_Reset.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # type: ignore
        self.tool_Reset.setIcon(icon_reset)
        self.tool_Reset.setText("Reset")
        self.tool_Reset.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_Reset.clicked.connect(self.action_reset)
        self.tool_bar.addWidget(self.tool_Reset)

        self.tool_bar.addSeparator()

        self._warningTimer = QtCore.QTimer()
        self._warningTimer.setSingleShot(True)
        self._warningTimer.timeout.connect(self.action_tempcontrol_warning)
        self._warningTimer.setInterval(2000)

        icon_temp = QtGui.QIcon()
        icon_temp.addPixmap(
            QtGui.QPixmap(os.path.join(icon_path, "temperature-control.svg")),
            QtGui.QIcon.Mode.Normal,
        )
        self.tool_TempControl = QtWidgets.QToolButton()
        self.tool_TempControl.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # type: ignore
        self.tool_TempControl.setIcon(icon_temp)
        self.tool_TempControl.setText("Temp Control")
        self.tool_TempControl.setCheckable(True)
        self.tool_TempControl.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_TempControl.clicked.connect(self.action_tempcontrol)
        self.tool_TempControl.enterEvent = self.action_tempcontrol_warn_start  # type: ignore[assignment]
        self.tool_TempControl.leaveEvent = self.action_tempcontrol_warn_stop  # type: ignore[assignment]
        self.tool_bar.addWidget(self.tool_TempControl)

        self.toolBar.addWidget(self.tool_bar)

        # TEC temperature side-panel
        self.tempController = QtWidgets.QWidget()
        self.tempController.setObjectName("tempController")
        self.tempController.enterEvent = self.action_tempcontrol_warn_start  # type: ignore[assignment]
        self.tempController.leaveEvent = self.action_tempcontrol_warn_stop  # type: ignore[assignment]
        self.tempController.setMinimumWidth(0)
        self.tempController.setMaximumWidth(0)  # collapsed until activated

        # Status banner
        self.tempStatusBar = QtWidgets.QLabel("Offline")
        self.tempStatusBar.setObjectName("tempStatusBanner")
        self.tempStatusBar.setFixedHeight(18)
        self.tempStatusBar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        _status_font = QtGui.QFont()
        _status_font.setPointSize(7)
        _status_font.setBold(True)
        self.tempStatusBar.setFont(_status_font)

        # Status (top) above slider (bottom)
        left_col = QtWidgets.QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)
        left_col.addWidget(self.tempStatusBar)
        left_col.addWidget(self.slTemp)

        # PID Info panel
        value_font = QtGui.QFont("Consolas", 7)
        self.lPV = QtWidgets.QLabel("PV  --.--°C")
        self.lSP = QtWidgets.QLabel("SP  --.--°C")
        self.lOP = QtWidgets.QLabel("OP  ----")
        for lbl in (self.lPV, self.lSP, self.lOP):
            lbl.setObjectName("TempPidValue")
            lbl.setFont(value_font)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)  # type: ignore

        self.tempPidInfo = QtWidgets.QFrame()
        self.tempPidInfo.setObjectName("tempPidInfo")
        pid_layout = QtWidgets.QVBoxLayout(self.tempPidInfo)
        pid_layout.setContentsMargins(8, 4, 8, 4)
        pid_layout.setSpacing(1)

        pid_header = QtWidgets.QLabel("PID INFO")
        pid_header.setObjectName("tempPidHeader")
        pid_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pid_layout.addWidget(pid_header)
        pid_layout.addWidget(self.lPV)
        pid_layout.addWidget(self.lSP)
        pid_layout.addWidget(self.lOP)

        # Assemble panel
        self.tempLayout = QtWidgets.QHBoxLayout()
        self.tempLayout.setContentsMargins(8, 6, 8, 6)
        self.tempLayout.setSpacing(8)
        self.tempLayout.addLayout(left_col, 1)
        self.tempLayout.addWidget(self.tempPidInfo, 0)
        self.tempController.setLayout(self.tempLayout)
        self.toolBar.addWidget(self.tempController)

        # Set initial chevron on the toolbar button
        self._set_temp_arrow(expand=False)

        # Wire live temperature updates to the display panel
        self.lTemp.text_updated.connect(self._update_temp_display)

        self.toolBar.addStretch()

        self.tool_bar_2 = QtWidgets.QToolBar()
        self.tool_bar_2.setObjectName("CtrlToolBar")
        self.tool_bar_2.setIconSize(QtCore.QSize(50, 30))

        icon_advanced = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "gear.svg")
        icon_advanced.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Mode.Normal)
        self.tool_Advanced = QtWidgets.QToolButton()
        self.tool_Advanced.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # type: ignore
        self.tool_Advanced.setIcon(icon_advanced)
        self.tool_Advanced.setText("Advanced")
        self.tool_Advanced.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_Advanced.clicked.connect(self.action_advanced)
        self.tool_bar_2.addWidget(self.tool_Advanced)

        self.tool_bar_2.addSeparator()

        icon_user = QtGui.QIcon()
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "user-circle.svg")
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Mode.Normal)
        icon_user.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Mode.Disabled)
        self.tool_User = QtWidgets.QToolButton()
        self.tool_User.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)  # type: ignore
        self.tool_User.setIcon(icon_user)
        self.tool_User.setText("Account")
        self.tool_User.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.tool_User.setEnabled(self._is_user_signed_in())
        self.tool_User.clicked.connect(self._toggle_account_popup)
        self.tool_bar_2.addWidget(self.tool_User)

        self.toolBar.addWidget(self.tool_bar_2)

        self.toolBar.setContentsMargins(8, 4, 8, 4)

        # Container for the entire toolbar row
        self.toolBarWidget = ControlsWidget()
        self.toolBarWidget.setLayout(self.toolBar)

        self.toolLayout.addWidget(self.toolBarWidget)
        self.toolLayout.addWidget(self.run_progress_bar)

        # Activate the appropriate layout mode.  Grid widgets are always
        # built first so external code can reference them; unused widgets
        # for the active mode are then detached from the layout.
        if SHOW_SIMPLE_CONTROLS:
            self.toolLayout.setContentsMargins(0, 0, 0, 0)
            self.centralwidget.setLayout(self.toolLayout)

            # Detach classic-only widgets that have no role in the toolbar view.
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

            # Build the Advanced perspective
            self._advanced_controls_layout = self._build_advanced_layout()
            self.advanced_container = AdvancedMainWidget.build_container(
                self._advanced_controls_layout
            )
            self._advanced_content_container = self.advanced_container
            self._install_perspective_animation(self.advanced_container)
        else:
            # Use the full grid as the central layout.
            self.centralwidget.setLayout(self.gridLayout)

        # icon-state singletons
        self.blankIcon = QtGui.QIcon()
        self.savedIcon = QtGui.QIcon()
        self.unsavedIcon = QtGui.QIcon()

        # Device-info configuration editor
        self.device_info_container = QtWidgets.QWidget()
        self.device_info_container.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True
        )
        self.device_info_container.setStyleSheet("background: transparent;")
        # The device perspective uses the same sectioned, two-column layout as
        # the advanced view, so it sizes to its content and shares the advanced
        # view's footprint rather than forcing an oversized floor.
        self.device_info_container.setMinimumWidth(500)
        self.device_info_container.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum
        )

        # Back Button + Title
        self.back_btn = QtWidgets.QPushButton()
        self.back_btn.setIcon(
            QtGui.QIcon(os.path.join(Architecture.get_path(), "QATCH", "icons", "left-arrow.svg"))
        )
        self.back_btn.setIconSize(QtCore.QSize(18, 18))
        self.back_btn.setFixedSize(32, 32)
        self.back_btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.back_btn.setObjectName("DeviceBackBtn")
        self.back_btn.setToolTip("Back to Advanced Options")
        self.back_btn.clicked.connect(self.on_device_config_editor_close)

        # Title icon
        _dev_icons_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")
        self.device_config_icon = QtWidgets.QLabel()
        self.device_config_icon.setFixedSize(18, 18)
        self.device_config_icon.setScaledContents(True)
        self.device_config_icon.setStyleSheet("background: transparent; border: none;")
        _dev_gear = QtGui.QPixmap(os.path.join(_dev_icons_dir, "gear.svg"))
        if not _dev_gear.isNull():
            self.device_config_icon.setPixmap(_dev_gear)
        self.device_config_title = DeviceConfigLabel("Configuration Editor for Device")
        self.device_config_title.setObjectName("DeviceConfigTitle")
        self.ConfigBannerWidget = self.device_config_title

        # Hover-info icon mirroring the advanced view's _InfoIcon usage.
        self._device_info_text = (
            "Configuration editor for the connected device - set its name, "
            "position ID, temperature calibration, and lid POGO timing."
        )
        try:
            from QATCH.ui.widgets.advanced_main_widget import _InfoIcon  # noqa: PLC0415

            self.device_config_info = _InfoIcon(
                os.path.join(_dev_icons_dir, "warning-circle.svg"),
                tooltip=self._device_info_text,
            )
        except Exception:
            self.device_config_info = QtWidgets.QLabel()
            self.device_config_info.setFixedSize(16, 16)
            self.device_config_info.setScaledContents(True)
            self.device_config_info.setStyleSheet("background: transparent; border: none;")
            _info_pix = QtGui.QPixmap(os.path.join(_dev_icons_dir, "warning-circle.svg"))
            if not _info_pix.isNull():
                self.device_config_info.setPixmap(_info_pix)
            self.device_config_info.setToolTip(self._device_info_text)

        # Header row
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(self.back_btn, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.device_config_icon, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.device_config_title, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addWidget(self.device_config_info, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        header_layout.addStretch()

        # Input validators
        self.validDeviceName = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r'[^\\/:*?"\'<>|]{1,12}')
        )  # 1-12 character string without forbidden characters
        self.validDevicePid = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-9A-Fa-f]{1,2}")
        )  # HEX values '00' thru 'FF'
        self.validTempOffset = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(
                r"-?(?:[0-5](?:\.\d{0,2})?|6(?:\.(?:[0-2]\d?|3[0-5]?))?|6\.?|\.\d{1,2})"
            )
        )  # Decimals -6.35 thru 6.35 (inclusive) with precision 2
        self.validPogoPosition = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-9]?[0-9]|1[0-7][0-9]|180")
        )  # 0 thru 180 degrees (rotation)
        self.validPogoDelayMs = QtGui.QRegularExpressionValidator(
            QtCore.QRegularExpression(r"[0-1]?[0-9]?[0-9]|2[0-4][0-9]|25[0-4]")
        )  # 0 thru 254 ms per step

        self._field_dots = {}
        self._field_widgets = {}

        # A lightweight poller keeps every status dot in sync with its action's
        self._dot_sync_timer = QtCore.QTimer()
        self._dot_sync_timer.setInterval(150)
        self._dot_sync_timer.timeout.connect(self._sync_all_dots)
        self._dot_sync_timer.start()

        def _make_dot(action, widget):
            dot = SavedStateDot()
            self._field_dots[action] = dot
            self._field_widgets[action] = widget
            return dot

        # Device Name
        self.device_name_input = QtWidgets.QLineEdit()
        self.device_name_input.setObjectName("CtrlInputPill")
        self.device_name_input.setValidator(self.validDeviceName)
        self.device_name_input.setMinimumWidth(160)
        self.device_name_action = self.device_name_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.device_name_input.textEdited.connect(
            lambda text, action=self.device_name_action: self.on_text_edit(text, action)  # type: ignore
        )
        self.device_name_dot = _make_dot(self.device_name_action, self.device_name_input)

        # Device Position ID
        self.device_pid_input = QtWidgets.QLineEdit()
        self.device_pid_input.setObjectName("CtrlInputPill")
        self.device_pid_input.setValidator(self.validDevicePid)
        self.device_pid_input.setMinimumWidth(160)
        self.device_pid_action = self.device_pid_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.device_pid_input.textEdited.connect(
            lambda text, action=self.device_pid_action: self.on_text_edit(text, action)  # type: ignore
        )
        self.device_pid_dot = _make_dot(self.device_pid_action, self.device_pid_input)

        self.device_config_default = BorderlessActionButton("Default")
        self.device_config_default.clicked.connect(self.on_device_config_default)
        self.device_config_save = BorderlessActionButton("Save", tone="primary")
        self.device_config_save.clicked.connect(self.on_device_config_save)
        self.device_config_reset = BorderlessActionButton("Reset")
        self.device_config_reset.clicked.connect(self.on_device_config_reset)

        # Constant Temperature Calibration
        self.temp_cal_always_input = AnimatedDoubleSpinBox(
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.temp_cal_always_input.setDecimals(2)
        self.temp_cal_always_input.setRange(-6.35, 6.35)
        self.temp_cal_always_input.setSingleStep(0.25)
        self.temp_cal_always_input.setSuffix(" °C")
        self.temp_cal_always_input.setMinimumWidth(142)
        self.temp_cal_always_icon = QtWidgets.QLineEdit()
        self.temp_cal_always_icon.setObjectName("CtrlIconBox")
        self.temp_cal_always_icon.setFixedWidth(self.temp_cal_always_icon.sizeHint().height())
        self.temp_cal_always_icon.setReadOnly(True)
        self.temp_cal_always_action = self.temp_cal_always_icon.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.temp_cal_always_input.textChanged.connect(
            lambda text, action=self.temp_cal_always_action: self.on_text_edit(text, action)  # type: ignore
        )
        self.temp_cal_always_input.editingFinished.connect(
            lambda widget=self.temp_cal_always_input: self.on_edit_finish(widget)
        )
        self.temp_cal_always_dot = _make_dot(
            self.temp_cal_always_action, self.temp_cal_always_input
        )

        # Row 1R: Running Temperature Calibration
        self.temp_cal_measure_input = AnimatedDoubleSpinBox(
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.temp_cal_measure_input.setDecimals(2)
        self.temp_cal_measure_input.setRange(-6.35, 6.35)
        self.temp_cal_measure_input.setSingleStep(0.25)
        self.temp_cal_measure_input.setSuffix(" °C")
        self.temp_cal_measure_input.setMinimumWidth(142)
        self.temp_cal_measure_icon = QtWidgets.QLineEdit()
        self.temp_cal_measure_icon.setObjectName("CtrlIconBox")
        self.temp_cal_measure_icon.setFixedWidth(self.temp_cal_measure_icon.sizeHint().height())
        self.temp_cal_measure_icon.setReadOnly(True)
        self.temp_cal_measure_action = self.temp_cal_measure_icon.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.temp_cal_measure_input.textChanged.connect(
            lambda text, action=self.temp_cal_measure_action: self.on_text_edit(text, action)  # type: ignore
        )
        self.temp_cal_measure_input.editingFinished.connect(
            lambda widget=self.temp_cal_measure_input: self.on_edit_finish(widget)
        )
        self.temp_cal_measure_dot = _make_dot(
            self.temp_cal_measure_action, self.temp_cal_measure_input
        )

        self.temp_cal_default = BorderlessActionButton("Default")
        self.temp_cal_default.clicked.connect(self.on_temp_cal_default)
        self.temp_cal_save = BorderlessActionButton("Save", tone="primary")
        self.temp_cal_save.clicked.connect(self.on_temp_cal_save)
        self.temp_cal_reset = BorderlessActionButton("Reset")
        self.temp_cal_reset.clicked.connect(self.on_temp_cal_reset)

        # Lid Pogo Distance
        self.lid_pogo_distance_field = _RangeSliderField(
            10,
            50,
            suffix="",
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.lid_pogo_distance_input = self.lid_pogo_distance_field  # string-compatible alias
        self.lid_pogo_distance_combo = AnimatedComboBox(icon_path=self._combo_chevron)
        self.lid_pogo_distance_combo.addItems(["Most", "More", "Normal", "Less", "Least", "Custom"])
        self.lid_pogo_distance_combo.setCurrentIndex(2)
        self.lid_pogo_distance_combo.hide()
        self.lid_pogo_distance_values = {
            "Most": 50,
            "More": 40,
            "Normal": 30,
            "Less": 20,
            "Least": 10,
        }
        self.lid_pogo_distance_field.setValue(30)
        self.lid_pogo_distance_action = self.lid_pogo_distance_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.lid_pogo_distance_field.valueChanged.connect(
            lambda v, action=self.lid_pogo_distance_action: self.on_text_edit(str(v), action)  # type: ignore
        )
        self.lid_pogo_distance_dot = _make_dot(
            self.lid_pogo_distance_action, self.lid_pogo_distance_field
        )

        # Lid Pogo Delay
        self.lid_pogo_delay_field = _RangeSliderField(
            0,
            254,
            suffix="",
            up_icon_path=os.path.join(Architecture.get_path(), "QATCH", "icons", "up-chevron.svg"),
            down_icon_path=os.path.join(
                Architecture.get_path(), "QATCH", "icons", "down-chevron.svg"
            ),
        )
        self.lid_pogo_delay_input = self.lid_pogo_delay_field  # string-compatible alias
        self.lid_pogo_delay_combo = AnimatedComboBox(icon_path=self._combo_chevron)
        self.lid_pogo_delay_combo.addItems(
            ["Fastest", "Fast", "Normal", "Slow", "Slowest", "Custom"]
        )
        self.lid_pogo_delay_combo.setCurrentIndex(2)
        self.lid_pogo_delay_combo.hide()
        self.lid_pogo_delay_values = {
            "Fastest": 10,
            "Fast": 20,
            "Normal": 30,
            "Slow": 40,
            "Slowest": 50,
        }
        self.lid_pogo_delay_field.setValue(30)
        self.lid_pogo_delay_action = self.lid_pogo_delay_input.addAction(
            self.blankIcon, QtWidgets.QLineEdit.TrailingPosition
        )
        self.lid_pogo_delay_field.valueChanged.connect(
            lambda v, action=self.lid_pogo_delay_action: self.on_text_edit(str(v), action)  # type: ignore
        )
        self.lid_pogo_delay_dot = _make_dot(self.lid_pogo_delay_action, self.lid_pogo_delay_field)

        self.lid_pogo_default = BorderlessActionButton("Default")
        self.lid_pogo_default.clicked.connect(self.on_lid_pogo_default)
        self.lid_pogo_save = BorderlessActionButton("Save", tone="primary")
        self.lid_pogo_save.clicked.connect(self.on_lid_pogo_save)
        self.lid_pogo_reset = BorderlessActionButton("Reset")
        self.lid_pogo_reset.clicked.connect(self.on_lid_pogo_reset)

        # Sectioned layout
        def dev_section(title, *rows):
            col = QtWidgets.QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(6)
            col.addWidget(SectionHeader(title))
            col.addWidget(_hairline())
            for row in rows:
                if isinstance(row, QtWidgets.QLayout):
                    col.addLayout(row)
                else:
                    col.addWidget(row)
            col.addStretch()
            return col

        def dev_field_row(label_html, field_widget, dot, *extra_widgets):
            """One field row: [status dot] label : [widget(s)].

            The leading dot makes each field's save state legible at a glance.
            """
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(dot, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            lbl = QtWidgets.QLabel(label_html)
            lbl.setMinimumWidth(86)
            row.addWidget(lbl, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(field_widget, 1)
            for w in extra_widgets:
                row.addWidget(w, 0)
            return row

        def dev_btn_row(*buttons):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 4, 0, 0)
            row.setSpacing(4)
            row.addStretch()
            for b in buttons:
                row.addWidget(b)
            return row

        # Device Configuration
        device_section = dev_section(
            "Device Configuration",
            dev_field_row("Device Name:", self.device_name_input, self.device_name_dot),
            dev_field_row("Position ID:", self.device_pid_input, self.device_pid_dot),
            dev_btn_row(
                self.device_config_default,
                self.device_config_reset,
                self.device_config_save,
            ),
        )

        # Temperature Calibration
        temp_section = dev_section(
            "Temperature Calibration",
            dev_field_row(
                "\u0394T<sub>always</sub>:",
                self.temp_cal_always_input,
                self.temp_cal_always_dot,
                self.temp_cal_always_icon,
            ),
            dev_field_row(
                "\u0394T<sub>measure</sub>:",
                self.temp_cal_measure_input,
                self.temp_cal_measure_dot,
                self.temp_cal_measure_icon,
            ),
            dev_btn_row(
                self.temp_cal_default,
                self.temp_cal_reset,
                self.temp_cal_save,
            ),
        )

        # Lid POGO Calibration
        pogo_section = dev_section(
            "Lid POGO Calibration",
            dev_field_row("Servo Steps:", self.lid_pogo_distance_field, self.lid_pogo_distance_dot),
            dev_field_row("Servo Delay:", self.lid_pogo_delay_field, self.lid_pogo_delay_dot),
            dev_btn_row(
                self.lid_pogo_default,
                self.lid_pogo_reset,
                self.lid_pogo_save,
            ),
        )

        # Device Config + Temperature Calibration side-by-side
        dev_top_row = QtWidgets.QHBoxLayout()
        dev_top_row.setSpacing(22)
        dev_top_row.addLayout(device_section, 1)
        dev_top_row.addLayout(temp_section, 1)

        # Global action buttons (Default All / Reset All / Save All)
        self.device_default_all = BorderlessActionButton("Default All")
        self.device_default_all.clicked.connect(self.on_device_default_all)
        self.device_reset_all = BorderlessActionButton("Reset All")
        self.device_reset_all.clicked.connect(self.on_device_reset_all)
        self.device_save_all = BorderlessActionButton("Save All", tone="primary")
        self.device_save_all.clicked.connect(self.on_device_save_all)

        all_btn_row = QtWidgets.QHBoxLayout()
        all_btn_row.setContentsMargins(0, 0, 0, 0)
        all_btn_row.setSpacing(10)
        all_btn_row.addStretch()
        all_btn_row.addWidget(self.device_default_all)
        all_btn_row.addWidget(self.device_reset_all)
        all_btn_row.addWidget(self.device_save_all)

        # header row, the two top sections, the POGO section
        bannerLayout = QtWidgets.QVBoxLayout(self.device_info_container)
        bannerLayout.setContentsMargins(2, 2, 2, 2)
        bannerLayout.setSpacing(12)
        bannerLayout.addLayout(header_layout)
        bannerLayout.addLayout(dev_top_row)
        bannerLayout.addLayout(pogo_section)
        bannerLayout.addStretch(1)
        bannerLayout.addWidget(_hairline())
        bannerLayout.addLayout(all_btn_row)

        # Hide it initially; the popup hosts and reveals it via a slide.
        self.device_info_container.hide()

        #  Finalization
        MainWindow1.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow1)

        ThemeManager.instance().themeChanged.connect(lambda _: self._refresh_toolbar_icons())
        self._refresh_toolbar_icons()

    def on_text_edit(self, text: str, action: QtWidgets.QAction) -> None:
        """Updates the action's visual state based on whether the input text is empty.

        This method synchronizes the action's icon and icon text to reflect a
        "saved/blank" or "unsaved" state, then triggers a sync of the visual
        dot indicator.

        Args:
            text (str): The current string content from the input field.
            action (QAction): The QAction associated with the field that tracks the
                save/edit state.
        """
        if len(text):
            action.setIcon(self.unsavedIcon)
            action.setIconText("unsaved")
        else:
            action.setIcon(self.blankIcon)
            action.setIconText("blank")
        self._sync_dot(action)

    def _sync_dot(self, action):
        """Update the glowing status dot paired with `action` to its state."""
        dot = getattr(self, "_field_dots", {}).get(action)
        if dot is not None:
            dot.set_state(action.iconText() or "blank")

    def _sync_all_dots(self) -> None:
        """Refresh every field dot from its action's current iconText."""
        for action, dot in getattr(self, "_field_dots", {}).items():
            dot.set_state(action.iconText() or "blank")

    def _has_unsaved_device_changes(self) -> bool:
        """True if any device field currently holds unsaved input.

        Returns:
            bool: True if unsaved changes persist, false otherwise.
        """
        for action in getattr(self, "_field_dots", {}):
            if action.iconText() == "unsaved":
                return True
        return False

    def _pulse_unsaved_device_fields(self) -> None:
        """Flashes the dot and the input border of every field with unsaved changes.

        Iterates through tracked fields; for any action marked as "unsaved",
        triggers a flash animation on the associated dot indicator and
        pulses the border of the corresponding input widget. This is typically
        called to draw attention to pending edits when a user attempts to
        navigate away.
        """
        for action, dot in getattr(self, "_field_dots", {}).items():
            if action.iconText() == "unsaved":
                dot.flash(times=3)
                widget = getattr(self, "_field_widgets", {}).get(action)
                if widget is not None:
                    self._pulse_widget_border(widget)

    def _pulse_widget_border(self, widget) -> None:
        """Briefly pulses an amber border on the widget to flag unsaved input.

        This method applies a transient amber border via Qt Style Sheets (QSS)
        using a series of timers to toggle the style. If the widget is a
        _RangeSliderField, the pulse is applied to its internal spin box;
        otherwise, it is applied to the widget itself.

        Args:
            widget: The input widget (or container) to be pulsed.
        """
        target = getattr(widget, "spin", widget)
        base_qss = target.styleSheet()
        pulse_rgba = ThemeManager.instance().tokens()["ctrl_pulse_border"]
        amber = (
            target.__class__.__name__ + f" {{ border: 1px solid {_tok_css(pulse_rgba)}; "
            "border-radius: 12px; }"
        )

        def _on():
            target.setStyleSheet(base_qss + amber)

        def _off():
            target.setStyleSheet(base_qss)

        for i in range(3):
            QtCore.QTimer.singleShot(i * 320, _on)
            QtCore.QTimer.singleShot(i * 320 + 160, _off)

    def on_distance_edit(self, text: str, action: QtWidgets.QAction) -> None:
        """Updates the distance input state and read-only status based on the selected value.

        Delegates the action/icon sync to the base text-edit handler. If the
        selected text is "Custom", the input field becomes editable. Otherwise,
        the input is set to read-only and updated with the value corresponding
        to the selected distance key.

        Args:
            text: The selected distance category or value string.
            action: The QAction associated with the field tracking the edit state.
        """
        self.on_text_edit(text, action)
        if text == "Custom":
            self.lid_pogo_distance_input.setReadOnly(False)
        else:
            # Determine the value from the mapping, defaulting to 30 if not found
            if text in self.lid_pogo_distance_values.keys():
                val = self.lid_pogo_distance_values[text]
            else:
                val = 30
            self.lid_pogo_distance_input.setReadOnly(True)
            self.lid_pogo_distance_input.setText(str(val))

    def on_delay_edit(self, text: str, action: QtWidgets.QAction):
        """Updates the delay input state and read-only status based on the selected value.

        Delegates the action/icon sync to the base text-edit handler. If the
        selected text is "Custom", the input field becomes editable. Otherwise,
        the input is set to read-only and updated with the value corresponding
        to the selected delay key from the configuration dictionary.

        Args:
            text: The selected delay category or value string.
            action: The QAction associated with the field tracking the edit state.
        """
        self.on_text_edit(text, action)
        if text == "Custom":
            self.lid_pogo_delay_input.setReadOnly(False)
        else:
            if text in self.lid_pogo_delay_values.keys():
                val = self.lid_pogo_delay_values[text]
            else:
                val = 30
            self.lid_pogo_delay_input.setReadOnly(True)
            self.lid_pogo_delay_input.setText(str(val))

    def on_edit_finish(self, widget: QtWidgets.QDoubleSpinBox) -> None:
        """Handles the completion of an edit operation on a double spin box.

        This method is currently a placeholder (no-op) designed to be overridden or
        expanded if specific logic is required upon the completion of a user
        interaction with a spin box.

        Args:
            widget: The QDoubleSpinBox instance that has finished its edit.
        """
        return

    def on_device_config_default(self) -> None:
        """Resets the device configuration inputs to their default values.

        Extracts the default device name from the ConfigBannerWidget and sets a
        default PID of "FF". If the current input values differ from these
        defaults, the inputs are updated, and the associated actions are marked
        as "unsaved". Finally, all visual dot indicators are synchronized.
        """
        default_name = self.ConfigBannerWidget.text().split()[-1]
        default_pid = "FF"

        if self.device_name_input.text() != default_name:
            self.device_name_input.setText(default_name)
            device_name_action = self.device_name_action
            if device_name_action is not None:
                device_name_action.setIcon(self.unsavedIcon)
                device_name_action.setIconText("unsaved")

        if self.device_pid_input.text() != default_pid:
            self.device_pid_input.setText(default_pid)
            device_pid_action = self.device_pid_action
            if device_pid_action is not None:
                device_pid_action.setIcon(self.unsavedIcon)
                device_pid_action.setIconText("unsaved")

        self._sync_all_dots()

    def _delayed_identify(self, mainWindow: "MainWindow") -> None:
        """Performs port identification if the window is not currently identifying.

        This helper method checks the current identification state of the provided
        MainWindow. If identification is not in progress, it triggers the
        port identification process. This is typically used to safely chain
        identification calls after configuration changes.

        Args:
            mainWindow: The primary application window instance responsible
                for port management.
        """
        if not mainWindow._identifying:
            mainWindow._port_identify()

    def on_device_config_save(self) -> None:
        """Handles the save process for device configuration inputs.

        Validates and saves the current device name and PID if they are marked
        as "unsaved". If the save is successful, the associated action state is
        updated to "saved". Following successful saves, it triggers post-save
        operations such as file cleanup and deferred device identification or
        port refreshing.

        The method performs the following steps:
        1. Checks for pending changes via iconText().
        2. Validates inputs using hasAcceptableInput().
        3. Invokes specific save handlers (save_device_name_input, save_device_pid_input).
        4. Triggers necessary UI or firmware updates based on the save outcomes.
        5. Synchronizes all visual status indicators.
        """
        ok_name = False
        ok_pid = False
        dif = None

        device_name_action = self.device_name_action
        assert device_name_action is not None
        if device_name_action.iconText() == "unsaved":
            if self.device_name_input.hasAcceptableInput():
                text = self.device_name_input.text()
                Log.d("Save device name =", text)
                ok_name = self.save_device_name_input(text)
                if ok_name:
                    device_name_action.setIcon(self.savedIcon)
                    device_name_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid 'Device Name' input: {self.device_name_input.text()} (out of valid range)"
                )

        device_pid_action = self.device_pid_action
        assert device_pid_action is not None
        if device_pid_action.iconText() == "unsaved":
            if self.device_pid_input.hasAcceptableInput():
                text = self.device_pid_input.text()
                Log.d("Save device pid =", text)
                ok_pid, dif = self.save_device_pid_input(text)
                if ok_pid:
                    device_pid_action.setIcon(self.savedIcon)
                    device_pid_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid 'Position ID' input: {self.device_pid_input.text()} (out of valid range)"
                )

        mainWindow: "MainWindow" = self.parent.parent
        if ok_pid:
            if dif is not None:
                try:
                    os.remove(dif)
                except Exception as e:
                    Log.e(f"Failed to delete file: {dif} with error: {e}")

            # Then in your timer calls:
            QtCore.QTimer.singleShot(1000, lambda: self._delayed_identify(mainWindow))
            QtCore.QTimer.singleShot(
                4000, lambda: mainWindow._port_identify() if mainWindow._identifying else None
            )
        elif ok_name:
            # Refresh the port list to reflect the updated device name
            mainWindow._refresh_ports()

        self._sync_all_dots()

    def on_device_config_reset(self) -> None:
        """Resets the device name and PID configuration fields to a querying state.

        Checks the current save state of the device name and PID actions. If a
        field is not already "saved," it transitions the UI to a "querying"
        placeholder state, clears the existing input, and schedules a reset
        operation to re-fetch the configuration values. Finally, updates all
        field status indicators.
        """
        assert self.device_name_action is not None
        assert self.device_name_input is not None
        assert self.device_pid_action is not None
        assert self.device_pid_input is not None

        if self.device_name_action.iconText() != "saved":
            Log.d("Reset device name")
            self.device_name_input.clear()
            self.device_name_input.setPlaceholderText("Querying...")
            self.device_name_action.setIcon(self.blankIcon)
            self.device_name_action.setIconText("querying")
            QtCore.QTimer.singleShot(500, self.reset_device_name_input)

        if self.device_pid_action.iconText() != "saved":
            Log.d("Reset device pid")
            self.device_pid_input.clear()
            self.device_pid_input.setPlaceholderText("Querying...")
            self.device_pid_action.setIcon(self.blankIcon)
            self.device_pid_action.setIconText("querying")
            QtCore.QTimer.singleShot(1000, self.reset_device_pid_input)

        self._sync_all_dots()

    def on_temp_cal_default(self) -> None:
        """Resets the temperature calibration inputs to their default values ("0.00").

        If the current input values for the "always" and "measure" calibration
        fields differ from the default "0.00", they are updated, marked as
        "unsaved," and all visual dot indicators are synchronized.
        """
        assert self.temp_cal_always_action is not None
        assert self.temp_cal_measure_action is not None

        default_always = "0.00"
        default_measure = "0.00"

        # Reset the "always" calibration field if it differs from default
        if self.temp_cal_always_input.text() != default_always:
            self.temp_cal_always_input.setValue(
                self.temp_cal_always_input.valueFromText(default_always)
            )
            self.temp_cal_always_action.setIcon(self.unsavedIcon)
            self.temp_cal_always_action.setIconText("unsaved")

        # Reset the "measure" calibration field if it differs from default
        if self.temp_cal_measure_input.text() != default_measure:
            self.temp_cal_measure_input.setValue(
                self.temp_cal_measure_input.valueFromText(default_measure)
            )
            self.temp_cal_measure_action.setIcon(self.unsavedIcon)
            self.temp_cal_measure_action.setIconText("unsaved")

        self._sync_all_dots()

    def on_temp_cal_save(self) -> None:
        """Handles the saving of temperature calibration fields.

        Validates the "always" and "measure" calibration inputs. If an input is
        marked as "unsaved" and contains valid data, it is formatted to two decimal
        places, persisted via the save handlers, and the corresponding action
        status is updated to "saved". Finally, updates all field status indicators.
        """
        assert self.temp_cal_always_action is not None
        assert self.temp_cal_measure_action is not None

        # Process "always" temperature calibration
        if self.temp_cal_always_action.iconText() == "unsaved":
            if self.temp_cal_always_input.hasAcceptableInput():
                text = f"{self.temp_cal_always_input.value():.2f}"
                Log.d("Save T_always =", text)
                if self.save_temp_cal_always_input(text):
                    self.temp_cal_always_action.setIcon(self.savedIcon)
                    self.temp_cal_always_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid T_always input: {self.temp_cal_always_input.text()} "
                    "(out of valid range)"
                )

        # Process "measure" temperature calibration
        if self.temp_cal_measure_action.iconText() == "unsaved":
            if self.temp_cal_measure_input.hasAcceptableInput():
                text = f"{self.temp_cal_measure_input.value():.2f}"
                Log.d("Save T_measure =", text)
                if self.save_temp_cal_measure_input(text):
                    self.temp_cal_measure_action.setIcon(self.savedIcon)
                    self.temp_cal_measure_action.setIconText("saved")
            else:
                Log.e(
                    f"Invalid T_measure input: {self.temp_cal_measure_input.text()} "
                    "(out of valid range)"
                )

        self._sync_all_dots()

    def on_temp_cal_reset(self) -> None:
        """Resets the temperature calibration fields to a querying state.

        Checks the current save state of the temperature calibration actions. If a
        field is not "saved," it clears the input, transitions the action state
        to "querying," and schedules a reset operation to re-fetch the
        calibration values. Updates all status indicators upon completion.
        """
        assert self.temp_cal_always_action is not None
        assert self.temp_cal_measure_action is not None

        # Reset "always" calibration if not saved
        if self.temp_cal_always_action.iconText() != "saved":
            Log.d("Reset T_always")
            self.temp_cal_always_input.clear()
            self.temp_cal_always_action.setIcon(self.blankIcon)
            self.temp_cal_always_action.setIconText("querying")
            QtCore.QTimer.singleShot(1, self.reset_temp_cal_measure_input)

        # Reset "measure" calibration if not saved
        if self.temp_cal_measure_action.iconText() != "saved":
            Log.d("Reset T_measure")
            self.temp_cal_measure_input.clear()
            self.temp_cal_measure_action.setIcon(self.blankIcon)
            self.temp_cal_measure_action.setIconText("querying")
            QtCore.QTimer.singleShot(3000, self.reset_temp_cal_measure_input)

        self._sync_all_dots()

    def on_lid_pogo_default(self) -> None:
        """Resets the Lid POGO distance and delay configuration fields to their
        "Normal" default values.

        Compares current input values against the defaults retrieved from the
        value dictionaries. If values differ, the inputs are updated to the
        defaults, marked as "unsaved," and all visual indicators are synchronized.
        """
        assert self.lid_pogo_distance_action is not None
        assert self.lid_pogo_delay_action is not None

        default_distance = str(self.lid_pogo_distance_values["Normal"])
        default_delay = str(self.lid_pogo_delay_values["Normal"])

        # Reset distance if it deviates from the default
        if self.lid_pogo_distance_input.text() != default_distance:
            self.lid_pogo_distance_input.setText(default_distance)
            self.lid_pogo_distance_action.setIcon(self.unsavedIcon)
            self.lid_pogo_distance_action.setIconText("unsaved")

        # Reset delay if it deviates from the default
        if self.lid_pogo_delay_input.text() != default_delay:
            self.lid_pogo_delay_input.setText(default_delay)
            self.lid_pogo_delay_action.setIcon(self.unsavedIcon)
            self.lid_pogo_delay_action.setIconText("unsaved")

        self._sync_all_dots()

    def on_lid_pogo_save(self) -> None:
        """Handles the saving of Lid POGO calibration settings.

        Validates the distance and delay inputs. If inputs are marked as "unsaved"
        and are within acceptable ranges, they are marked as "saved." If all
        pending changes are validated successfully, the calibration command is
        sent to the device.

        The method performs the following:
        1. Validates each field marked as "unsaved".
        2. Marks fields as "saved" upon successful validation.
        3. Blocks the final save command if any field fails validation (form_error).
        4. Synchronizes visual indicators for all fields.
        """
        assert self.lid_pogo_distance_action is not None
        assert self.lid_pogo_delay_action is not None

        send_lid_cal_cmd = False
        form_error = False

        # Process Lid POGO Distance
        if self.lid_pogo_distance_action.iconText() == "unsaved":
            if self.lid_pogo_distance_input.hasAcceptableInput():
                text = self.lid_pogo_distance_input.text()
                Log.d("Save lid pogo distance =", text)
                self.lid_pogo_distance_action.setIcon(self.savedIcon)
                self.lid_pogo_distance_action.setIconText("saved")
                send_lid_cal_cmd = True
            else:
                Log.e(
                    f"Invalid 'Servo Steps' input: {self.lid_pogo_distance_input.text()} "
                    "(out of valid range)"
                )
                form_error = True

        # Process Lid POGO Delay
        if self.lid_pogo_delay_action.iconText() == "unsaved":
            if self.lid_pogo_delay_input.hasAcceptableInput():
                text = self.lid_pogo_delay_input.text()
                Log.d("Save lid pogo delay =", text)
                self.lid_pogo_delay_action.setIcon(self.savedIcon)
                self.lid_pogo_delay_action.setIconText("saved")
                send_lid_cal_cmd = True
            else:
                Log.e(
                    f"Invalid 'Servo Delay' input: {self.lid_pogo_delay_input.text()} "
                    "(out of valid range)"
                )
                form_error = True

        # Finalize calibration command if changes were valid
        if send_lid_cal_cmd and not form_error:
            self.save_lid_pogo_calibration()

        self._sync_all_dots()

    def on_lid_pogo_reset(self) -> None:
        """Resets the Lid POGO distance and delay fields to a querying state.

        Checks the current save state of the distance and delay actions. If either
        is not "saved," it clears the input, sets a placeholder, transitions the
        action to "querying," and schedules a deferred call to re-fetch the
        Lid POGO calibration values. Finally, synchronizes all status indicators.
        """
        assert self.lid_pogo_distance_action is not None
        assert self.lid_pogo_delay_action is not None

        get_lid_cal = False

        # Reset distance field if not saved
        if self.lid_pogo_distance_action.iconText() != "saved":
            Log.d("Reset lid pogo distance")
            self.lid_pogo_distance_input.clear()
            self.lid_pogo_distance_input.setPlaceholderText("...")
            self.lid_pogo_distance_action.setIcon(self.blankIcon)
            self.lid_pogo_distance_action.setIconText("querying")
            get_lid_cal = True

        # Reset delay field if not saved
        if self.lid_pogo_delay_action.iconText() != "saved":
            Log.d("Reset lid pogo delay")
            self.lid_pogo_delay_input.clear()
            self.lid_pogo_delay_input.setPlaceholderText("...")
            self.lid_pogo_delay_action.setIcon(self.blankIcon)
            self.lid_pogo_delay_action.setIconText("querying")
            get_lid_cal = True

        # Trigger async re-fetch if any field was reset
        if get_lid_cal:
            QtCore.QTimer.singleShot(1500, self.get_lid_pogo_calibration)

        self._sync_all_dots()

    def on_device_default_all(self) -> None:
        """Applies the 'Default' operation to every configuration section at once."""
        self.on_device_config_default()
        self.on_temp_cal_default()
        self.on_lid_pogo_default()
        self._sync_all_dots()

    def on_device_reset_all(self) -> None:
        """Applies the 'Reset' operation to every configuration section at once."""
        self.on_device_config_reset()
        self.on_temp_cal_reset()
        self.on_lid_pogo_reset()
        self._sync_all_dots()

    def on_device_save_all(self) -> None:
        """Applies the 'Save' operation to every configuration section at once."""
        self.on_device_config_save()
        self.on_temp_cal_save()
        self.on_lid_pogo_save()
        self._sync_all_dots()

    def save_device_name_input(self, text: str) -> bool:
        """
        Sanitizes user input and saves the new device name to the device's info file.

        Args:
            text (str): The raw device name input provided by the user.

        Returns:
            bool: True if the device name was successfully updated and saved, False otherwise.
        """
        main_window: "MainWindow" = self.parent.parent
        dev_handle = None
        target_dev_info = None
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if dev_info.get("NAME") and dev_info.get("PORT") == main_window._selected_port:
                dev_handle = dev_name
                target_dev_info = dev_info
                break

        # Ensure we actually found a matching device
        if not dev_handle or not target_dev_info:
            Log.e("Failed to update name: No matching device found for the selected port.")
            return False

        # Sanitization
        for invalid_char in Constants.invalidChars:
            text = text.replace(invalid_char, "")

        text = text.strip().replace(" ", "_")
        text = text[:12].upper()

        # Fallback to the USB identifier if the sanitized string is empty
        if not text:
            text = target_dev_info.get("USB", "UNKNOWN_DEV")

        # Apply changes
        try:
            Log.i(f"Set on device '{dev_handle}': NAME = {text}")

            # Update UI if the text differs from the current input box
            if text != self.device_name_input.text():
                self.device_name_input.setText(text)

            # Construct the file path
            dev_file = os.path.join(
                Constants.csv_calibration_export_path,
                dev_handle,
                f"{Constants.txt_device_info_filename}.{Constants.txt_extension}",
            )

            # Read existing lines, update the first line, and write back
            with open(dev_file, "r") as file:
                dev_lines = file.readlines()

            if dev_lines:
                dev_lines[0] = f"NAME: {text}\n"
            else:
                dev_lines = [f"NAME: {text}\n"]  # Failsafe if file was completely empty

            with open(dev_file, "w") as file:
                file.writelines(dev_lines)

            Log.i("Program 'Name' operation was successful!")
            return True

        except Exception as e:
            Log.e(f"Failed to update name entered by user. Error: {e}")
            return False

    def reset_device_name_input(self) -> None:
        """
        Resets the device name input field to its currently saved name.
        If no custom name is found, it defaults to the selected port name.
        Also updates the UI to reflect a 'saved' state.
        """
        assert self.device_name_action is not None

        main_window = self.parent.parent
        selected_port = main_window._selected_port

        # Default fallback is the port name
        friendly_name = selected_port

        # Locate the device matching the selected port
        device_list = FileStorage.DEV_get_device_list()
        for i, dev_name in device_list:
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if dev_info.get("PORT") == selected_port and dev_info.get("NAME"):
                friendly_name = dev_info.get("NAME")
                break

        self.device_name_input.setText(friendly_name)
        self.device_name_input.setPlaceholderText(None)
        self.device_name_action.setIcon(self.savedIcon)
        self.device_name_action.setIconText("saved")

    def save_device_pid_input(self, text: str) -> tuple[bool, str | None]:
        """
        Parses and validates user input for a device Position ID (PID), updates the UI,
        writes the new PID to the device's EEPROM, and refreshes the device LCD.

        Args:
            text (str): The hex string representing the new PID entered by the user.

        Returns:
            tuple[bool, Optional[str]]: A tuple containing a boolean indicating if the
                                        PID changed, and the path to a stale device info
                                        file to be removed (if applicable, else None).
        """
        main_window = self.parent.parent
        selected_port = main_window._selected_port

        dev_handle = None
        pid_old = 0xFF  # Default/unassigned

        # Locate the target device and extract its current PID
        for i, dev_name in FileStorage.DEV_get_device_list():
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if dev_info.get("PORT") == selected_port:
                dev_handle = dev_name
                if "PID" in dev_info:
                    try:
                        pid_old = int(dev_info["PID"], base=16)
                    except ValueError:
                        pass  # Keep default 0xFF if parsing fails
                break

        # Exit safely if the device wasn't found
        if not dev_handle:
            Log.e("Failed to update PID: No matching device found for the selected port.")
            return False, None

        # Parse and validate the new PID input
        valid_pids = {0x1, 0x2, 0x3, 0x4, 0xA, 0xB, 0xC, 0xD, 0x00, 0x80, 0xFF}

        try:
            pid_new = int(text, base=16)
            if pid_new not in valid_pids:
                Log.w("Out-of-range PID entered by user. Using default: 0xFF")
                pid_new = 0xFF
        except ValueError:
            # Catching ValueError specifically for invalid hex/integer conversions
            Log.w("Non-numeric PID entered by user. Using default: 0xFF")
            pid_new = 0xFF

        Log.i(f"Set on device '{dev_handle}': PID = {pid_new}")

        # Update the UI Text Box
        pid_str = f"{pid_new:X}"
        if pid_str != self.device_pid_input.text():
            self.device_pid_input.setText(pid_str)

        # Write to EEPROM if the PID has changed
        pid_changed = pid_new != pid_old

        if pid_changed:
            if main_window.setEEPROM(selected_port, 0, pid_new):
                Log.i("Device EEPROM write PID success!")
            Log.i("Program 'Position ID' operation was successful!")
        else:
            Log.w("Program 'Position ID' operation resulted in no change!")
            return False, None

        # Refresh LCD and cleanup stale files
        try:
            with DeviceSerial(
                port=selected_port,
                baudrate=Constants.serial_default_speed,
                stopbits=DeviceSerial.STOPBITS_ONE,
                bytesize=DeviceSerial.EIGHTBITS,
                timeout=Constants.serial_timeout_ms,
                write_timeout=Constants.serial_writetimeout_ms,
            ) as _serial:

                _serial.write(b"MULTI INIT 0\n")

        except Exception as e:
            Log.e(f"Unable to refresh LCD. PID error may be stale. Error: {e}")

        try:
            i_old = 0 if pid_old == 0xFF else pid_old
            dev_folder_old = f"{i_old}_{dev_handle}" if i_old > 0 else dev_handle
            dev_info_file_old = os.path.join(
                Constants.csv_calibration_export_path,
                dev_folder_old,
                f"{Constants.txt_device_info_filename}.txt",
            )

            if os.path.exists(dev_info_file_old):
                Log.d(
                    f"Queueing removal of stale DEV_INFO file for {dev_handle} with PID {pid_new}..."
                )
                return True, dev_info_file_old

        except Exception as e:
            Log.e(f"Unable to check for stale DEV_INFO file removal. Error: {e}")

        return True, None

    def reset_device_pid_input(self) -> None:
        """
        Resets the device PID input field to its currently saved value.
        Cross-references the stored PID with the active UI port list to handle
        potential mismatches, and updates the UI to reflect a 'saved' state.
        """
        assert self.device_pid_action is not None

        main_window: "MainWindow" = self.parent.parent
        selected_port = main_window._selected_port

        pid_old = 0xFF  # Default/unassigned

        # Look up the stored PID from the device info files
        for i, dev_name in FileStorage.DEV_get_device_list():
            dev_info = FileStorage.DEV_info_get(i, dev_name)

            if dev_info.get("PORT") == selected_port:
                pid_value = dev_info.get("PID")
                if pid_value:
                    try:
                        pid_old = int(pid_value, base=16)
                    except ValueError:
                        Log.w(
                            f"Could not parse stored PID '{pid_value}' as hex. Using default 0xFF."
                        )
                break

        # Confirm the stored PID matches the one actively listed in the COM Port combobox
        try:
            port_combobox = main_window.ControlsWin.ui1.cBox_Port
            idx = port_combobox.findData(selected_port)

            if idx >= 0:
                device_text = port_combobox.itemText(idx)
                if ":" in device_text:
                    parsed_pid_str = device_text.split(":")[0]
                    try:
                        active_ui_pid = int(parsed_pid_str, base=16)

                        if active_ui_pid != pid_old:
                            Log.e(
                                f"Conflicting device info: using PID {active_ui_pid} instead of reported {pid_old}!"
                            )
                            pid_old = active_ui_pid
                    except ValueError:
                        Log.w(f"Failed to parse PID '{parsed_pid_str}' from COM port list.")

        except Exception as e:
            # Catching the exact exception prevents silent failures on UI changes
            Log.e(f"ERROR: Unable to check if PID in COM Port list matches DEV_INFO. Error: {e}")

        # Update the UI
        pid_str = f"{pid_old:X}"

        self.device_pid_input.setText(pid_str)
        self.device_pid_input.setPlaceholderText(None)
        self.device_pid_action.setIcon(self.savedIcon)
        self.device_pid_action.setIconText("saved")

    def save_temp_cal_always_input(self, text: str) -> bool:
        """
        Parses a user-provided temperature calibration value, scales it,
        converts it to an 8-bit integer, and writes it to the device's EEPROM at address 1.

        Args:
            text (str): The raw string input from the user representing the temperature in Celsius.

        Returns:
            bool: True if the EEPROM write was successful, False otherwise.
        """
        main_window = self.parent.parent
        selected_port = main_window._selected_port
        dev_handle = None

        # Locate the target device matching the selected port
        for i, dev_name in FileStorage.DEV_get_device_list():
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if dev_info.get("PORT") == selected_port:
                dev_handle = dev_name
                break

        # Exit if the device wasn't found
        if not dev_handle:
            Log.e("Failed to save TEMP CAL1: No matching device found for the selected port.")
            return False

        # Parse, scale, and bound the calibration value
        cal_new = 0xFF
        try:
            raw_cal = int(float(text) * 20.0)
            masked_cal = raw_cal & 0xFF
            if 0 <= masked_cal <= 255:
                cal_new = masked_cal
            else:
                Log.w(f"Out-of-range CAL1 ({masked_cal}) entered. Using default: 0xFF")

        except ValueError:
            Log.w("Non-numeric CAL1 entered by user. Using default: 0xFF")

        Log.i(f"Set on device '{dev_handle}': CAL1 = {cal_new} ({text}C)")

        # Write to EEPROM and handle logging
        success = main_window.setEEPROM(selected_port, 1, cal_new)

        if success:
            Log.i("Device EEPROM write CAL1 success!")
            Log.i("Program 'TEMP CAL1' operation was successful!")
        else:
            Log.e("Failed to write EEPROM address for CAL1.")
            Log.e("Program 'TEMP CAL1' operation was NOT successful!")

        return success

    def save_temp_cal_measure_input(self, text: str) -> bool:
        """
        Parses a user-provided temperature measurement calibration value (CAL2), scales it,
        converts it to an 8-bit integer, and writes it to the device's EEPROM at address 3.

        Args:
            text (str): The raw string input from the user representing the temperature in Celsius.

        Returns:
            bool: True if the EEPROM write was successful, False otherwise.
        """
        main_window: "MainWindow" = self.parent.parent
        selected_port = main_window._selected_port
        dev_handle = None

        # Locate the target device matching the selected port
        for i, dev_name in FileStorage.DEV_get_device_list():
            dev_info = FileStorage.DEV_info_get(i, dev_name)
            if dev_info.get("PORT") == selected_port:
                dev_handle = dev_name
                break

        # Exit if the device wasn't found
        if not dev_handle:
            Log.e("Failed to save TEMP CAL2: No matching device found for the selected port.")
            return False

        cal_new = 0xFF
        try:
            raw_cal = int(float(text) * 20.0)
            masked_cal = raw_cal & 0xFF
            if 0 <= masked_cal <= 255:
                cal_new = masked_cal
            else:
                Log.w(f"Out-of-range CAL2 ({masked_cal}) entered. Using default: 0xFF")

        except ValueError:
            Log.w("Non-numeric CAL2 entered by user. Using default: 0xFF")

        Log.i(f"Set on device '{dev_handle}': CAL2 = {cal_new} ({text}C)")

        # Write to EEPROM and handle logging
        success = main_window.setEEPROM(selected_port, 3, cal_new)

        if success:
            Log.i("Device EEPROM write CAL2 success!")
            Log.i("Program 'TEMP CAL2' operation was successful!")
        else:
            Log.e("Failed to write EEPROM address for CAL2.")
            Log.e("Program 'TEMP CAL2' operation was NOT successful!")

        return success

    def reset_temp_cal_measure_input(self) -> None:
        """
        Resets the measurement temperature calibration input field (CAL2)
        to the current TEC offset. Forces a hardware update if cached
        data is older than 10 seconds.
        """
        assert self.temp_cal_measure_action is not None

        main_window = self.parent.parent
        tec_worker = main_window.tecWorker
        start_time = monotonic()

        # Determine if we need fresh data from the TEC hardware
        last_reply = tec_worker.last_reply()
        tec_update_required = True

        if last_reply and (start_time - last_reply < 10.0):
            tec_update_required = False

        # Trigger the hardware update if our cached data is stale
        if tec_update_required:
            Log.i("Updating TEC parameters for CAL2...")
            tec_worker.set_port(main_window._selected_port)
            tec_worker._tec_update()  # Force read to update software cached offsets

        # Update the UI with the cached offset
        # NOTE: Since there is no delay here, the UI will update with
        # the current cache immediately.
        set_cal2 = tec_worker._tec_offset2

        try:
            parsed_value = self.temp_cal_measure_input.valueFromText(str(set_cal2))
            self.temp_cal_measure_input.setValue(parsed_value)
        except Exception as e:
            Log.e(f"Failed to parse or set TEC offset '{set_cal2}' to UI. Error: {e}")

        # Update the action button UI to reflect a 'saved' state
        self.temp_cal_measure_action.setIcon(self.savedIcon)
        self.temp_cal_measure_action.setIconText("saved")

    def save_lid_pogo_calibration(self) -> None:
        """
        Sends the LID calibration command to the hardware using the distance and delay
        values provided in the UI. Automatically handles serial port configuration.
        """
        main_window = self.parent.parent

        # Validate and extract inputs
        try:
            distance = int(self.lid_pogo_distance_input.text())
            delay = int(self.lid_pogo_delay_input.text())
        except ValueError:
            Log.e(
                "Invalid LID calibration input. Please enter numeric values for distance and delay."
            )
            return

        cal_start = 100
        cal_stop = cal_start + distance
        try:
            with DeviceSerial(
                port=main_window._selected_port,
                baudrate=Constants.serial_default_speed,
                stopbits=DeviceSerial.STOPBITS_ONE,
                bytesize=DeviceSerial.EIGHTBITS,
                timeout=Constants.serial_timeout_ms,
                write_timeout=Constants.serial_writetimeout_ms,
            ) as lid_serial:

                cmd = f"LID CAL {cal_start},{cal_stop},{delay}\n"
                lid_serial.write(cmd.encode())
                Log.i(f"LID calibration command sent: {cmd.strip()}")

        except Exception as e:
            Log.e(f"Unable to send LID CAL command. Error: {e}")

    def get_lid_pogo_calibration(self) -> None:
        """
        Queries the hardware for current LID calibration parameters (distance and delay),
        parses the response, and updates the corresponding UI components.
        """
        assert self.lid_pogo_distance_action is not None

        main_window = self.parent.parent
        pogo_distance, pogo_delay = 30, 30  # Default values

        # Fetch data from hardware
        try:
            with DeviceSerial(
                port=main_window._selected_port,
                baudrate=Constants.serial_default_speed,
                stopbits=DeviceSerial.STOPBITS_ONE,
                bytesize=DeviceSerial.EIGHTBITS,
                timeout=Constants.serial_timeout_ms,
                write_timeout=Constants.serial_writetimeout_ms,
            ) as lid_serial:
                lid_serial.write(b"LID CAL\n")
                response = lid_serial.read_until()

            # Parse response (Expected format example: "LID CAL 100,130,30")
            if response:
                decoded = response.decode().strip().split()[-1]
                params = decoded.split(",")
                if len(params) >= 3:
                    start, stop, delay = map(int, params)
                    pogo_distance = abs(stop - start)
                    pogo_delay = delay

        except Exception as e:
            Log.e(f"Failed to retrieve or parse LID CAL from device. Error: {e}")

        # Update UI components
        self._update_input_field(
            self.lid_pogo_distance_input,
            self.lid_pogo_distance_combo,
            self.lid_pogo_distance_action,  # type: ignore
            self.lid_pogo_distance_values,
            pogo_distance,
        )

        self._update_input_field(
            self.lid_pogo_delay_input,
            self.lid_pogo_delay_combo,
            self.lid_pogo_delay_action,  # type: ignore
            self.lid_pogo_delay_values,
            pogo_delay,
        )

        self._sync_all_dots()

    def _update_input_field(
        self,
        input_widget: QtWidgets.QLineEdit,
        combo_widget: QtWidgets.QComboBox,
        action_widget: QtWidgets.QAction,
        value_map: dict[str, int],
        found_val: int,
    ) -> None:
        """
        Synchronizes UI components with a retrieved hardware value.

        Updates the input field text and selects the corresponding index in the
        associated combo box. This update only occurs if the action widget's current
        state indicates it is uninitialized ('blank') or currently 'querying'.

        Args:
            input_widget: The QLineEdit displaying the current numeric value.
            combo_widget: The QComboBox providing preset options for the value.
            action_widget: The QAction representing the state of the setting.
            value_map: A dictionary of display labels to numeric values.
            found_val: The numeric value retrieved from the device to be displayed.
        """
        # Only update if the current state requires synchronization
        if action_widget.iconText() in ("blank", "querying"):
            idx = next(
                (i for i, val in enumerate(value_map.values()) if val == found_val),
                combo_widget.count() - 1,
            )

            # Update the UI state
            combo_widget.setCurrentIndex(idx)
            input_widget.setText(str(found_val))
            input_widget.setPlaceholderText(None)
            action_widget.setIcon(self.savedIcon)
            action_widget.setIconText("saved")

    def blank_device_config_icon_text(self) -> None:
        """
        Resets the status icons for all device configuration fields
        to the 'blank' state, signaling that data is currently unknown
        or awaiting a new query.
        """
        # Grouping actions in a list allows for cleaner, loop-based updates
        action_widgets = [
            self.device_name_action,
            self.device_pid_action,
            self.temp_cal_always_action,
            self.temp_cal_measure_action,
            self.lid_pogo_distance_action,
            self.lid_pogo_delay_action,
        ]

        for action in action_widgets:
            action.setIconText("blank")

        self._sync_all_dots()

    def on_device_config_editor_close(self) -> None:
        """
        Handles the close action for the device config editor. Validates that
        all fields are saved before allowing navigation; otherwise, pulses
        unsaved fields.
        """
        actions = [
            self.device_name_action,
            self.device_pid_action,
            self.temp_cal_always_action,
            self.temp_cal_measure_action,
            self.lid_pogo_distance_action,
            self.lid_pogo_delay_action,
        ]

        # Check if any action is marked as 'unsaved'
        if any(action.iconText() == "unsaved" for action in actions):
            Log.w(
                "You have unsaved device configuration input. "
                "Please Save or Reset before closing."
            )
            self._pulse_unsaved_device_fields()
            return

        # If inside a popup, slide back to advanced view; otherwise hide.
        popup = getattr(self, "_advanced_popup", None)
        if popup and popup.isVisible():
            popup.show_advanced_perspective(animated=True)
        else:
            # Fallback to hiding the container or its host window
            container = self.device_info_container
            if container:
                host = container.window()
                (host if host else container).hide()

    def _update_progress_text(self) -> None:
        """
        Extracts plain text from the infobar HTML and updates the
        progress bar format.
        """
        html = self.infobar.text()

        plain_text = re.sub(r"<[^>]+>", "", html).strip()

        if not plain_text:
            display_text = "Progress: Not Started"
        else:
            display_text = f"Status: {plain_text}"

        self.run_progress_bar.setFormat(display_text)

    def _update_progress_value(self) -> None:
        """Sets the progress bar format based on the current operation type."""
        # Using a direct boolean comparison is cleaner than a pass/else block
        if self.cBox_Source.currentIndex() != OperationType.measurement.value:
            self.run_progress_bar.setFormat("Progress: %p%")

    def retranslateUi(self, MainWindow: QtWidgets.QMainWindow) -> None:
        """
        Updates the interface text and window iconography to reflect the current
        application settings and localization state.

        Args:
            MainWindow (QMainWindow): the parent window to rescale to.
        """
        _translate = QtCore.QCoreApplication.translate

        # Update Window Title
        title = f"{Constants.app_title} {Constants.app_version} - Setup/Control"
        MainWindow.setWindowTitle(_translate("MainWindow", title))

        # Update Window Icon
        icon_path = os.path.join(Architecture.get_path(), "QATCH", "icons", "qatch-icon.png")
        MainWindow.setWindowIcon(QtGui.QIcon(icon_path))

        # Update UI Elements
        self.pButton_Stop.setText(_translate("MainWindow", " STOP"))
        self.pButton_Start.setText(_translate("MainWindow", "START"))
        self.pButton_Clear.setText(_translate("MainWindow", "Clear Plots"))
        self.pButton_ResetApp.setText(_translate("MainWindow", "Factory Defaults"))

        self.sBox_Samples.setSuffix(_translate("MainWindow", " / 5 min"))
        self.sBox_Samples.setPrefix(_translate("MainWindow", ""))

        self.chBox_export.setText(_translate("MainWindow", "Txt Export Sweep File"))
        self.chBox_freqHop.setText(_translate("MainWindow", "Mode Hop"))
        self.chBox_correctNoise.setText(_translate("MainWindow", "Show amplitude curve"))
        self.chBox_MultiAuto.setText(_translate("MainWindow", "Auto-detect channel count"))

    def _tinted_icon(self, svg_path: str, color: QtGui.QColor) -> QtGui.QIcon:
        """Renders an SVG file into a QIcon with all opaque pixels tinted to `color`.

        Uses SourceAtop composition so the SVG silhouette is preserved but every
        non-transparent pixel adopts the supplied color, enabling icons to track
        the active light/dark theme.

        Args:
            svg_path: Filesystem path to the SVG file.
            color: The tint color to apply over the rendered SVG.

        Returns:
            A QIcon containing the tinted pixmap, or an empty QIcon if the SVG
            cannot be loaded.
        """
        src = QtGui.QPixmap(svg_path)
        if src.isNull():
            return QtGui.QIcon()
        dst = QtGui.QPixmap(src.size())
        dst.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(dst)
        p.drawPixmap(0, 0, src)
        p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
        p.fillRect(dst.rect(), color)
        p.end()
        return QtGui.QIcon(dst)

    def _refresh_toolbar_icons(self) -> None:
        """Recolors the SVG toolbar icons to match the active theme's text color.

        Called once during setup and again whenever the theme changes so that
        static SVG icons (which carry their own embedded fill color) are
        re-rendered with the correct light-or-dark tint rather than staying
        permanently dark or light regardless of the active mode.
        """
        tok = ThemeManager.instance().tokens()
        color = QtGui.QColor(*tok["plot_text_normal"])
        icon_dir = os.path.join(Architecture.get_path(), "QATCH", "icons")

        buttons_icons = [
            (getattr(self, "tool_Initialize", None), "speedometer.svg"),
            (getattr(self, "tool_Reset", None), "reset.svg"),
            (getattr(self, "tool_TempControl", None), "temperature-control.svg"),
            (getattr(self, "tool_Advanced", None), "gear.svg"),
            (getattr(self, "tool_User", None), "user-circle.svg"),
        ]

        for btn, icon_name in buttons_icons:
            if btn is not None:
                btn.setIcon(self._tinted_icon(os.path.join(icon_dir, icon_name), color))

    def action_next_port(self) -> None:
        """Advance the FLUX controller to the next port.

        This method validates that a FLUX controller is available, ensures that
        any previously launched FLUX worker thread has completed, and then starts
        a new worker thread to advance the hardware to the selected port.

        The trigger action is temporarily disabled while setup is performed to
        prevent duplicate requests.
        """
        assert self.action_NextPortRow is not None
        try:
            self.action_NextPortRow.setEnabled(False)

            controller_port = next(
                (
                    self.cBox_Port.itemData(i)
                    for i in range(self.cBox_Port.count())
                    if self.cBox_Port.itemText(i).startswith("80:")
                ),
                None,
            )

            if controller_port is None:
                Log.e("FLUX controller not found. Is it connected and powered on?")
                self.tool_NextPortRow.setIconError()
                return

            flux_thread = getattr(self, "fluxThread", None)
            if flux_thread and flux_thread.isRunning():
                Log.d("Waiting for FLUX controller thread to stop.")

                if not flux_thread.wait(msecs=3000):
                    Log.w("Previous FLUX controller thread is still running; skipping request.")
                    return

            self._start_flux_thread(
                controller_port,
                self.tool_NextPortRow.value(),
            )

        except Exception as exc:
            Log.e(f"action_next_port ERROR: {exc}")
            self.tool_NextPortRow.setIconError()

        finally:
            self.action_NextPortRow.setEnabled(True)

    def _start_flux_thread(self, controller_port: str, next_port_num: int) -> None:
        """Create and start a FLUX controller worker thread.

        Initializes a new `QThread` and `FLUXControl` worker instance,
        configures the worker with the specified controller and target port,
        moves the worker to the background thread, connects the required
        signals and slots, and starts execution.

        The worker emits a `result` signal when the port-switch operation
        completes and a `finished` signal to terminate the thread's event
        loop.

        Args:
            controller_port: Serial port identifier for the FLUX controller.
            next_port_num: Port number that the controller should switch to.

        """
        Log.d("Starting FLUX controller thread.")

        # Create thread and worker.
        self.fluxThread = QtCore.QThread()
        self.fluxWorker = FLUXControl()

        # Configure worker.
        self.fluxWorker.set_ports(
            controller=controller_port,
            next_port=next_port_num,
        )
        self.fluxWorker.moveToThread(self.fluxThread)

        # Connect thread lifecycle signals.
        self.fluxThread.started.connect(self.fluxWorker.run)
        self.fluxWorker.finished.connect(self.fluxThread.quit)
        self.fluxWorker.finished.connect(self.fluxWorker.deleteLater)

        # Connect worker result signals.
        self.fluxWorker.result.connect(self.next_port_result)

        # Start execution.
        self.fluxThread.start()

    def next_port_result(self, success: bool) -> None:
        """Handle completion of a FLUX controller port-switch operation.

        This slot is invoked when the FLUX worker emits its `result` signal.
        On success, the active multi-channel port is updated and the parent
        widget is instructed to refresh its multi-mode configuration. On
        failure, an error indicator is displayed and the user is given the
        option to retry the operation.

        Args:
            success: `True` if the FLUX controller successfully switched to
                the requested port; otherwise `False`.
        """
        assert self.action_NextPortRow is not None
        try:
            self.action_NextPortRow.setEnabled(True)

            if success:
                self.parent.parent.active_multi_ch = self.tool_NextPortRow.value()
                self.parent.parent.set_multi_mode()
                return

            self.tool_NextPortRow.setIconError()

            retry = PopUp.critical(
                self,
                "Next Port Failed",
                "ERROR: Flux controller failed to move to the next port.",
                btn1_text="Reset",
            )

            if retry:
                self.tool_NextPortRow.click()

        except Exception as exc:
            Log.e(f"next_port_result: {exc}")

    def action_initialize(self) -> None:
        """Start the calibration initialization sequence.

        If the Start button is currently enabled, this method configures the
        operation source for calibration mode, resets the run controls UI,
        triggers the start action, and marks the calibration process as
        initialized.

        Returns:
            None

        Side Effects:
            - Sets the source selection to calibration mode.
            - Resets and disables the run controls widget, if present.
            - Emits the Start button's `clicked` signal.
            - Sets `self.cal_initialized` to `True`.
        """
        if not self.pButton_Start.isEnabled():
            return

        self.cBox_Source.setCurrentIndex(OperationType.calibration.value)

        run_controls = getattr(self, "run_controls", None)
        if run_controls is not None:
            run_controls.set_running(False)
            run_controls.update_progress(0, 5, "Ready")
            run_controls.setEnabled(False)

        self.pButton_Start.clicked.emit()
        self.cal_initialized = True

    def action_start(self) -> None:
        """Start a measurement operation.

        If the Start button is enabled, this method configures the source
        selection for measurement mode and triggers the Start button's
        associated action by emitting its `clicked` signal.
        """
        if not self.pButton_Start.isEnabled():
            return

        self.cBox_Source.setCurrentIndex(OperationType.measurement.value)
        self.pButton_Start.clicked.emit()

    def action_stop(self) -> None:
        """Stop the current operation and reset UI state.

        This method handles the Stop UI action by clearing calibration state,
        emitting the Stop button signal, and resetting all device plots to an
        idle state.

        Returns:
            None
        """
        if not self.pButton_Stop.isEnabled():
            return

        self.cal_initialized = False
        self.pButton_Stop.clicked.emit()

        num_devices = getattr(self, "multiplex_plots", 1)

        plots_win = getattr(self.parent.parent, "PlotsWin", None)
        if plots_win and hasattr(plots_win, "ui2"):
            left_pane = plots_win.ui2.left_pane
            for i in range(num_devices):
                left_pane.set_device_state(i, "idle")

    def action_reset(self) -> None:
        """Reset the application to its standby state.

        This method restores the UI and runtime state to a known idle
        condition. Any active temperature-control panel is collapsed,
        temperature settings are reset to their default value, run controls
        are cleared, device indicators are returned to the idle state, and
        calibration status is cleared.
        """
        # Collapse temperature control if currently active.
        if self.tool_TempControl.isChecked():
            self.tool_TempControl.setChecked(False)
            self.tool_TempControl.clicked.emit()

        # Restore default temperature.
        self.slTemp.setValue(25)

        # Clear and refresh UI state.
        if self.pButton_Start.isEnabled():
            self.pButton_Clear.clicked.emit()
            self.pButton_Refresh.clicked.emit()

        self.infostatus.setText("Program Status Standby")
        self.cal_initialized = False

        # Reset run controls.
        run_controls = getattr(self, "run_controls", None)
        if run_controls is not None:
            run_controls.set_running(False)
            run_controls.update_progress(0, 5, "Idle")
            run_controls.setEnabled(False)

        # Enable temperature control only when ports are available.
        self.tool_TempControl.setEnabled(self.cBox_Port.count() > 0)

        # Reset device states.
        num_devices = getattr(self, "multiplex_plots", 1)

        plots_win = getattr(self.parent.parent, "PlotsWin", None)
        if plots_win and hasattr(plots_win, "ui2"):
            left_pane = plots_win.ui2.left_pane
            for i in range(num_devices):
                left_pane.set_device_state(i, "idle")

    def _animate_temp_controller(self, expand: bool) -> None:
        """Animate the temperature controller panel.

        Expands or collapses the temperature controller panel by animating its
        `maximumWidth` property. The corresponding toggle-arrow icon is
        updated to reflect the target state before the animation begins.

        Args:
            expand: `True` to expand the panel to its configured width;
                `False` to collapse it.
        """
        expanded_width = 280

        current_width = min(
            self.tempController.maximumWidth(),
            expanded_width,
        )
        target_width = expanded_width if expand else 0

        if current_width == target_width:
            return

        self._set_temp_arrow(expand)

        animation = QtCore.QPropertyAnimation(
            self.tempController,
            b"maximumWidth",
        )
        animation.setDuration(220)
        animation.setStartValue(current_width)
        animation.setEndValue(target_width)
        animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)

        self._temp_anim = animation
        animation.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

    def _toggle_temp_controller(self) -> None:
        """Programmatically toggle the panel via the toolbar button.

        Kept for backward compatibility - external callers (e.g. shortcuts) can
        still use this to flip the temperature controller open/closed.
        """
        self.tool_TempControl.setChecked(not self.tool_TempControl.isChecked())
        self.action_tempcontrol()

    def _set_temp_arrow(self, expand: bool) -> None:
        """Update the toolbar button's chevron to indicate panel state.

        Per the wireframe, the chevron lives on the toolbar button itself -
        `Temp Control ›` when collapsed (clicking expands), `Temp Control ‹`
        when expanded (clicking collapses).  No in-panel arrow strip is used.
        """
        if hasattr(self, "tool_TempControl"):
            self.tool_TempControl.setText("Temp Control ‹" if expand else "Temp Control ›")

    def _update_temp_display(self, text: str) -> None:
        """Parse temperature telemetry and update UI display + status bar.

        Expected format (order-insensitive):
            "PV:25.03C SP:25.00C OP:50%"

        Missing or malformed values fall back to placeholders and mark the
        system as Offline.
        """
        tok = ThemeManager.instance().tokens()

        def parse_kv(raw: str) -> dict:
            out = {}
            for segment in raw.split():
                if ":" in segment:
                    k, v = segment.split(":", 1)
                    out[k] = v.strip("[]")
            return out

        parts = parse_kv(text)

        pv_str = parts.get("PV", "--.--C")
        sp_str = parts.get("SP", "--.--C")
        op_str = parts.get("OP", "----")

        self.lPV.setText(f"PV  {pv_str}")
        self.lSP.setText(f"SP  {sp_str}")
        self.lOP.setText(f"OP  {op_str}")

        border_rgba = tok["ctrl_temp_status_border"]

        try:
            pv = float(pv_str.rstrip("C"))
            sp = float(sp_str.rstrip("C"))

            if abs(pv - sp) <= 0.5:
                status_text = "Ready"
                bg_rgba = tok["ctrl_temp_ready_bg"]
                text_rgba = tok["ctrl_temp_ready_text"]

            elif pv < sp:
                status_text = "Heating to setpoint..."
                bg_rgba = tok["ctrl_temp_heating_bg"]
                text_rgba = tok["ctrl_temp_heating_text"]

            else:
                status_text = "Cooling to setpoint..."
                bg_rgba = tok["ctrl_temp_cooling_bg"]
                text_rgba = tok["ctrl_temp_cooling_text"]

        except (ValueError, TypeError):
            status_text = "Offline"
            bg_rgba = tok["ctrl_temp_status_offline_bg"]
            text_rgba = tok["ctrl_temp_status_offline_text"]

        self.tempStatusBar.setText(status_text)
        self.tempStatusBar.setStyleSheet(
            "QLabel { "
            f"background: {_tok_css(bg_rgba)}; "
            f"color: {_tok_css(text_rgba)}; "
            f"border: 1px solid {_tok_css(border_rgba)}; "
            "border-radius: 3px; padding: 0 6px; font-weight: bold; }"
        )

    def action_tempcontrol(self) -> None:
        """Toggle the temperature control panel and synchronize temperature state.

        Expands or collapses the temperature controller panel and ensures the
        underlying temperature control state is kept consistent with the UI.

        When opening the panel, focus is moved to the temperature slider after
        the expansion animation completes.
        """
        is_checked = self.tool_TempControl.isChecked()

        self._animate_temp_controller(is_checked)

        temp_is_running = "Stop" in self.pTemp.text()
        if is_checked and not temp_is_running:
            self.pTemp.clicked.emit()

        elif not is_checked and temp_is_running:
            self.pTemp.clicked.emit()
        if is_checked:
            QtCore.QTimer.singleShot(230, self.slTemp.setFocus)

    def action_tempcontrol_warn_start(self, event: QtCore.QEvent) -> None:
        """Store event position and start the warning timer.

        Captures the window position from the triggering event (if available)
        and starts the warning timer used for temperature-control feedback.
        """
        pos = getattr(event, "windowPos", None)

        if callable(pos):
            self.event_windowPos = pos()

        self._warningTimer.start()

    def action_tempcontrol_warn_stop(self, event: QtCore.QEvent) -> None:
        """Stop the temperature-control warning timer.

        This handler is triggered when the warning condition ends, ensuring
        any active warning timer is cancelled.
        """
        self._warningTimer.stop()

    def action_tempcontrol_warn_now(self, event: QtCore.QEvent) -> None:
        """Trigger an immediate temperature-control warning.

        Captures the event window position (if available) and then executes
        the warning handler immediately.
        """
        pos = getattr(event, "windowPos", None)

        if callable(pos):
            self.event_windowPos = pos()

        self.action_tempcontrol_warning()

    def action_tempcontrol_warning(self) -> None:
        """Log guidance when the user attempts to change Temp Control during a run.

        The message is contextualized based on whether the interaction occurred
        to the left or right of the temperature controller panel.
        """
        if not (self.tool_TempControl.isChecked() and not self.tool_TempControl.isEnabled()):
            return

        Log.w("WARNING: Temp Control mode cannot be changed during an active run.")
        event_pos = getattr(self, "event_windowPos", None)
        if event_pos is None:
            return

        panel_x = self.tempController.mapToGlobal(QtCore.QPoint(0, 0)).x()

        if event_pos.x() >= panel_x:
            Log.w('To adjust Temp Control: Press "Stop" first, then adjust setpoint accordingly.')
        else:
            Log.w('To stop Temp Control: Press "Stop" first, then click "Temp Control" button.')

    def _install_perspective_animation(self, container: QtWidgets.QWidget) -> None:
        """Install a fade + slide-up entrance animation for a UI container.

        The animation is triggered whenever the container is shown, providing
        a smooth transition between perspectives (e.g., advanced view and
        device-info view) instead of an abrupt appearance.

        A dedicated `_PerspectiveAnimator` event filter is attached to the
        container to intercept show events and run the animation.

        Args:
            container: The QWidget that will receive the entrance animation.
        """
        animator = _PerspectiveAnimator(container)
        if not hasattr(self, "_perspective_animators"):
            self._perspective_animators = []

        self._perspective_animators.append(animator)

        container.installEventFilter(animator)

    def _build_advanced_layout(self) -> QtWidgets.QLayout:
        """Assemble the advanced-panel widgets into a clean, sectioned layout.

        Called once during `setupUi` to produce the `QVBoxLayout` that is
        handed to `AdvancedMainWidget.build_container`.  The layout is stored
        as `self._advanced_controls_layout` and reused every time the popup
        is opened or recreated.

        Layout structure
        ----------------
        Two side-by-side columns share the available width (3 : 2 ratio):

        **Left column** - connection and signal settings:

        * Operation Mode (`cBox_Source`)
        * Serial COM Port (`cBox_Port` + ID / Refresh / Configure buttons)
        * Resonance Frequency / Quartz Sensor (`cBox_Speed` + freq-hop toggle)
        * Multiplex Mode (`cBox_MultiMode` + plate-config button + auto-detect)
        * Amplitude curve toggle (`chBox_correctNoise`)

        **Right column** - cartridge and plot controls:

        * Cartridge Auto-Lock (`toggle_Cartridge` with Manual / Automatic labels)
        * Plot Mode (`toggle_PlotMode` with Absolute / Reference labels)
        * Control Buttons (Start / Stop / Clear / Reset stacked vertically)

        **Status readout** - a borderless `QLabel` (`infobar_readout`) at
        the foot of the panel, separated by a hairline, mirrors live text from
        `self.infobar` with the legacy prefix stripped.

        Returns:
            QtWidgets.QLayout
                The fully assembled outer `QVBoxLayout` ready to be passed to
                `AdvancedMainWidget.build_container`.
        """

        def section(title, *rows, stretch_last=False):
            """A titled vertical group: header, hairline, then content rows."""
            col = QtWidgets.QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(6)
            col.addWidget(SectionHeader(title))
            col.addWidget(_hairline())
            for row in rows:
                if isinstance(row, QtWidgets.QLayout):
                    col.addLayout(row)
                else:
                    col.addWidget(row)
            if not stretch_last:
                col.addStretch()
            return col

        def hrow(*widgets, spacing=6):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(spacing)
            for w in widgets:
                if isinstance(w, QtWidgets.QLayout):
                    row.addLayout(w)
                else:
                    row.addWidget(w)
            return row

        # Detach the cartridge toggle from its old QGroupBox so we can restyle.
        self.grpMode.setParent(None)

        # Left column
        op_section = section("Operation Mode", self.cBox_Source)

        # Port row
        self.cBox_Port.setMinimumWidth(0)
        self.cBox_Port.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        port_row = hrow(
            self.cBox_Port,
            self.pButton_ID,
            self.pButton_Refresh,
            self.pButton_Configure,
        )
        port_section = section("Serial COM Port", port_row)

        res_row = hrow(self.cBox_Speed, self.chBox_freqHop)
        res_section = section("Resonance Frequency / Quartz Sensor", res_row)

        # Rebuild the multiplex row
        multi_row = hrow(self.cBox_MultiMode, self.pButton_PlateConfig)
        multi_section = section("Multiplex Mode", multi_row, self.chBox_MultiAuto)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(14)
        left_col.addLayout(op_section)
        left_col.addLayout(port_section)
        left_col.addLayout(res_section)
        left_col.addLayout(multi_section)
        left_col.addWidget(self.chBox_correctNoise)
        left_col.addStretch()

        # Right column-
        lock_row = hrow(self.lbl_lock_manual, self.toggle_Cartridge, self.lbl_lock_auto, spacing=8)

        # Plotting mode toggle
        plot_mode_row = hrow(
            self.lbl_plot_absolute,
            self.toggle_PlotMode,
            self.lbl_plot_reference,
            spacing=8,
        )

        btns = QtWidgets.QVBoxLayout()
        btns.setSpacing(8)
        btns.addWidget(self.pButton_Start)
        btns.addWidget(self.pButton_Stop)
        btns.addWidget(self.pButton_Clear)
        btns.addWidget(self.pButton_ResetApp)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(14)
        right_col.addLayout(section("Cartridge Auto-Lock", lock_row, stretch_last=False))
        right_col.addLayout(section("Plot Mode", plot_mode_row, stretch_last=False))
        right_col.addLayout(section("Control Buttons", btns, stretch_last=False))
        right_col.addStretch()

        # Assemble columns
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(22)
        columns.addLayout(left_col, 3)
        columns.addLayout(right_col, 2)

        # Status readout
        self.infobar_readout = QtWidgets.QLabel()
        self.infobar_readout.setObjectName("infobarReadout")
        self.infobar_readout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft  # type: ignore
        )

        def _mirror_status(t):
            clean = self._strip_infobar_prefix(t)
            self.infobar_readout.setText(f"Status:  {clean}" if clean else "Status:  Ready")

        # Mirror live status text from the existing infobar line edit.
        self.infobar.textChanged.connect(_mirror_status)
        _mirror_status(self.infobar.text())

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(10)
        outer.addLayout(columns)
        outer.addWidget(_hairline())
        outer.addWidget(self.infobar_readout)

        return outer

    def action_advanced(self, obj=None) -> None:
        """Toggle the advanced control panel popup.

        Opens or closes the advanced controls widget anchored to the toolbar
        button. If the popup is created, its content container is stored for
        later layout and size queries.
        """
        main_window = getattr(self, "parent", None)

        popup = AdvancedMainWidget.toggle(
            owner=self,
            anchor=self.tool_Advanced,
            controls_layout=self._advanced_controls_layout,
            main_window=main_window,
        )

        if popup is None:
            return

        self.advanced_container = popup.content_container

    def _ensure_advanced_popup(self) -> AdvancedMainWidget | None:
        """Return the active advanced popup, creating it if necessary.

        If the popup is already visible, the existing instance is returned.
        Otherwise, a new popup is created via `AdvancedMainWidget.toggle`
        and its content container is stored for later use.
        """
        popup = getattr(self, "_advanced_popup", None)

        # Reuse existing visible popup
        if popup is not None and popup.isVisible():
            return popup

        main_window = getattr(self, "parent", None)

        popup = AdvancedMainWidget.toggle(
            owner=self,
            anchor=self.tool_Advanced,
            controls_layout=self._advanced_controls_layout,
            main_window=main_window,
        )

        if popup is None:
            return None

        # Persist references consistently
        self._advanced_popup = popup
        self.advanced_container = popup.content_container

        return popup

    def show_device_config_editor(self):
        """Open the advanced popup and switch to the device configuration view.

        Ensures the advanced popup is visible, registers the device perspective
        if available, and animates the transition to the device configuration
        page inside the popup.
        """
        popup = self._ensure_advanced_popup()

        if popup is None:
            return None

        # Register device perspective if available
        if self.device_info_container is not None:
            popup.set_device_perspective(self.device_info_container)

        popup.show_device_perspective(animated=True)

        return popup

    @staticmethod
    def _is_user_signed_in() -> bool:
        """Check whether a valid user session is active.

        Returns:
            True if a valid session exists, otherwise False.
        """
        try:
            is_valid, _ = UserProfiles.session_info()
            return bool(is_valid)

        except (ImportError, AttributeError, RuntimeError):
            return False

    def refresh_user_button_state(self) -> None:
        """Enable/disable Account UI based on authentication state.

        Disables the Account toolbar button and closes any open account popup
        when the user is signed out. Re-enables the button when signed in.
        """
        signed_in = self._is_user_signed_in()

        tool_user = getattr(self, "tool_User", None)
        if tool_user is not None:
            tool_user.setEnabled(signed_in)

        # If user signed out, close any open account popup
        if signed_in:
            return

        popup = getattr(self, "_account_popup", None)
        if popup is None:
            return

        try:
            popup.close()
        except Exception:
            pass

    def _toggle_account_popup(self) -> None:
        """Show or recreate the account popup anchored under the Account button.

        Ensures only one popup instance exists at a time. Any previous popup is
        safely closed and scheduled for deletion before a new one is created.
        The popup is anchored to the Account button and auto-closes on outside
        interaction or main window resize.
        """
        # Close and cleanup any existing popup.  Stop animations first so no
        # pending timer callbacks can fire on the widget after it is scheduled
        # for deletion.
        prev = getattr(self, "_account_popup", None)
        if prev is not None:
            try:
                prev._enter_slide.stop()
                prev._enter_fade.stop()
            except Exception:
                pass
            prev.close()
            prev.deleteLater()

        self._account_popup = AccountPopup(
            open_manager_cb=self._open_user_manager,
            sign_out_cb=self._sign_out_current_user,
        )

        main_window = getattr(self, "parent", None)

        anchor = getattr(self, "tool_User", None)
        if anchor is None:
            return

        self._account_popup.show_anchored_to(anchor, main_window=main_window)

    def _sign_out_current_user(self) -> None:
        """Sign out the current user and reset UI state to anonymous mode."""
        try:
            main_win = getattr(self.parent.parent, "MainWin", None)
            if main_win is None:
                return

            can_sign_out = main_win.ui0._set_no_user_mode(None)
            if not can_sign_out:
                Log.d("User has unsaved changes in Analyze mode. Sign out aborted.")
                return

            UserProfiles().session_end()

            # Extract username
            username_label = getattr(self.parent, "username", None)
            name = username_label.text()[6:] if username_label else "Unknown"

            Log.i(f"Goodbye, {name}! You have been signed out.")

            # UI reset
            self.parent.username.setText("User: [NONE]")
            self.parent.userrole = UserRoles.NONE
            self.parent.signinout.setText("&Sign In")
            self.parent.manage.setText("&Manage Users...")
            self.parent.ui1.tool_User.setText("Anonymous")

            analyze_proc = getattr(self.parent.parent, "AnalyzeProc", None)
            if analyze_proc:
                analyze_proc.tool_User.setText("Anonymous")

            self.refresh_user_button_state()

        except Exception as exc:
            Log.e(f"Error signing out user: {exc}")

    def _open_user_manager(self) -> None:
        """Open the User Profiles Manager overlay (admin-only)."""

        try:
            is_valid, user_info = UserProfiles.session_info()

            # Enforce admin-only access
            if not (is_valid and user_info and user_info[2] == UserRoles.ADMIN.name):
                Log.w("User Profiles Manager: an admin session is required to manage users.")
                return

            admin_name = user_info[0] or ""

            # Resolve main application window safely
            main_app = getattr(self.parent, "parent", None)
            main_win = getattr(main_app, "MainWin", None)

            if main_win is not None:
                parent_win = main_win.centralWidget() or main_win
            else:
                fallback = getattr(self, "parent", None)
                assert fallback is not None
                parent_win = (
                    fallback.centralWidget() if hasattr(fallback, "centralWidget") else fallback
                )

            Log.d(f"[UIControls] resolved parent_win={parent_win}")

            # Reuse existing manager if possible
            existing = getattr(self, "_user_profiles_manager", None)

            if existing is not None:
                if existing.isVisible():
                    existing.raise_()
                    return

                existing.deleteLater()
                self._user_profiles_manager = None

            # Create new overlay
            manager = UserProfilesManagerWidget(
                parent=parent_win,
                admin_name=admin_name,
            )

            self._user_profiles_manager = manager

            # Ensure full overlay coverage before showing
            try:
                manager.setGeometry(parent_win.rect())
            except Exception:
                pass

            manager.show()

        except Exception as exc:
            Log.e(f"UIControls._open_user_manager error: {exc}")

    @staticmethod
    def _strip_infobar_prefix(text: str) -> str:
        """Strip legacy Infobar prefix and HTML markup from status text.

        Removes HTML tags, normalizes whitespace, and removes any leading
        "Infobar" label variants.
        """
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove leading "Infobar" label if present
        text = re.sub(r"^infobar[:\s-]*", "", text, flags=re.IGNORECASE)

        return text

    def _update_plate_config_enabled(self, *args) -> None:
        """Enable Plate Config only when multi-channel mode is available.

        The Plate Config button is disabled when:
        - Only one multiplex mode option exists, or
        - The current selection is single-channel mode (index 0).
        """
        plate_btn = getattr(self, "pButton_PlateConfig", None)
        if plate_btn is None:
            return

        mode_box = self.cBox_MultiMode

        has_multiple_modes = mode_box.count() > 1
        is_multi_channel = mode_box.currentIndex() > 0

        plate_btn.setEnabled(has_multiple_modes and is_multi_channel)

    def _update_configure_enabled(self, *args) -> None:
        """Enable or disable the Configure button based on device connection state.

        The button is enabled only when at least one real device port is present.
        Legacy sentinel entries (CMD_DEV_INFO) are ignored when determining
        connection state.
        """
        btn = getattr(self, "pButton_Configure", None)
        if btn is None:
            return

        port_box = self.cBox_Port

        real_ports = sum(
            1 for i in range(port_box.count()) if port_box.itemData(i) != "CMD_DEV_INFO"
        )

        connected = real_ports > 0

        btn.setEnabled(connected)
        btn.setToolTip("Configure device / position info" if connected else "No device connected")

    def doPlateConfig(self):
        """Open (or re-open) the well-plate configuration dialog.

        Determines the plate geometry from the currently selected device's
        position ID (PID) and validates that the number of connected device
        ports matches that geometry before launching the `WellPlate` dialog.

        A warning dialog is shown when the port count does not satisfy the
        expected geometry or when only a single device is connected.

        If a `WellPlate` window is already open it is closed before the new
        one is created, ensuring only one instance exists at a time.
        """
        # Close any existing well-plate dialog before opening a new one.
        if hasattr(self, "wellPlateUI") and self.wellPlateUI.isVisible():
            self.wellPlateUI.close()

        # Count only real device ports
        num_ports = sum(
            1
            for idx in range(self.cBox_Port.count())
            if self.cBox_Port.itemData(idx) != "CMD_DEV_INFO"
        )

        # Parse the hex PID from the selected port's display text.
        port_text = self.cBox_Port.currentText()
        pid = 0 if ":" not in port_text else int(port_text.split(":")[0], base=16)

        # PIDs 0-8 indicate a single-device or unassigned slot.
        # PIDs 9+ indicate a multiplex device full plate layout.
        if pid < 9:
            well_width, well_height = 4, 1
        else:
            well_width, well_height = 6, 4

        num_channels = self.cBox_MultiMode.currentIndex() + 1

        if num_ports not in (well_width, well_height) or num_ports == 1:
            PopUp.warning(
                self.parent,
                "Plate Configuration",
                "<b>Multiplex device(s) are required for plate configuration.</b><br/>"
                + "You must have exactly 4 device ports connected for this mode.<br/>"
                + f"Currently connected device port count is: {num_ports} (not 4)",
            )
        else:
            self.wellPlateUI = WellPlate(well_width, well_height, num_channels)
