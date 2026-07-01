from typing import TYPE_CHECKING

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.interfaces import UIMode
from QATCH.ui.windows.base_window import BaseWindow

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow


class ModeWindow(BaseWindow):
    """
    Window that handles specific application modes.

    This window integrates into the main application UI, sets up its own
    interface, and monitors global application focus and input events.
    """

    def __init__(self, parent: "MainWindow"):
        """
        Initializes the ModeWindow, registers it with the parent menu,
        and sets up global event filtering.

        Args:
            parent: The main application window instance.
        """
        super().__init__()

        # Register this window within the parent's menu system
        parent.ControlsWin._create_menu(self)

        # Initialize the specific UI components
        self.ui = UIMode()
        self.ui.setup_ui(self, parent)

        # Setup global event tracking
        self._setup_global_signals()

    def _setup_global_signals(self) -> None:
        """
        Connects global application signals for focus changes and event filtering.
        """
        app = QtWidgets.QApplication.instance()
        if isinstance(app, (QtWidgets.QApplication, QtGui.QGuiApplication)):
            app.focusWindowChanged.connect(self.focusWindowChanged)
            app.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """
        Intercepts global application events to manage floating widget visibility.

        This method monitors for mouse button presses anywhere in the application.
        If a click occurs outside of specific 'ignored' widgets, the floating
        widget is hidden.

        Args:
            obj (QObject): The source QObject of the event.
            event (QEvent): The QEvent instance being processed.

        Returns:
            bool: True if the event was consumed, otherwise False.
        """
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if isinstance(event, QtGui.QMouseEvent):
                widget_clicked = QtWidgets.QApplication.widgetAt(event.globalPos())

                # Define widgets that should NOT trigger a hide action
                ignored_widgets = [self.ui.mode_learn]

                if self.ui.floating_widget.isVisible() and widget_clicked not in ignored_widgets:
                    self.ui.floating_widget.hide()

        return super().eventFilter(obj, event)

    def focusWindowChanged(self, focus_window: QtGui.QWindow) -> None:  # noqa: N802
        """
        Handles application-wide focus changes to manage the floating widget state.

        When the application focus shifts to another window or away from the
        application entirely (None), the floating widget is hidden to ensure
        it does not remain orphaned on the screen.

        Args:
            focus_window (QtGui.QWindow): The QWindow that currently has focus, or None if
                        no window in the application is focused.
        """
        if focus_window is None or focus_window != self.windowHandle():
            if self.ui.floating_widget.isVisible():
                self.ui.floating_widget.hide()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:  # noqa: N802
        """
        Handles the window move event to prevent the floating widget
        from becoming detached from the main window.

        When the main window moves, the floating widget is hidden. It is
        expected to be repositioned and shown again when triggered by
        the user or the application state.

        Args:
            event: The QMoveEvent object containing old and new position data.
        """
        if self.ui.floating_widget.isVisible():
            self.ui.floating_widget.hide()

        super().moveEvent(event)

    def _before_quit(self) -> None:
        """Removes the global application event filter before the process exits."""
        app = QtWidgets.QApplication.instance()
        if app:
            app.removeEventFilter(self)
