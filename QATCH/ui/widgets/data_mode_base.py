"""
QATCH.ui.widgets.data_mode_base.py

Base widget contract for data-management mode pages.

Provides the :class:`DataModeWidget` base class used by the Import, Export,
Recover, Advanced, and History modes of the data-management interface.

Each mode subclass defines a unique `MODE_KEY` and user-facing
`MODE_LABEL`, then implements its page construction and lifecycle hooks.
The base class provides a shared :class:`DataServices` instance, a
transparent root layout, and common signal routing for GUI freeze state and
mode-specific progress updates.

Subclass contract:
    MODE_KEY (str):
        Unique channel key identifying the mode (for example, `"import"` or
        `"export"`).
    MODE_LABEL (str):
        User-facing label displayed by the parent mode selector.
    build():
        Construct and arrange the mode's widgets and layouts.
    on_enter():
        Called when the mode becomes the active page.
    on_leave():
        Called when the mode is no longer active.
    on_freeze(frozen):
        Enable or disable interactive controls while a shared data operation
        is running.
    on_progress(label, pct, color):
        Handle progress updates routed to this mode's `MODE_KEY`.

The parent container can therefore manage all data-management modes through a
uniform interface without needing to know the implementation details of any
individual mode.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-21
"""

from PyQt5 import QtCore, QtWidgets

from QATCH.common.data_service import DataServices

TAG = "[DataMode]"


class DataModeWidget(QtWidgets.QWidget):
    """Base widget for individual data-management mode pages.

    Provides the common interface and service integration shared by all
    data-management modes. Subclasses identify themselves with `MODE_KEY`
    and `MODE_LABEL` and implement the page construction and lifecycle
    hooks used by the parent container.

    Attributes:
        MODE_KEY (str): Unique channel key identifying the mode.
        MODE_LABEL (str): User-facing label for the mode selector.
        services (DataServices): Shared data-management service instance.
        root (QtWidgets.QVBoxLayout): Transparent root layout used to build
            the mode's page content.

    Raises:
        NotImplementedError: If a subclass does not define both `MODE_KEY`
            and `MODE_LABEL`.
    """

    MODE_KEY: str = ""
    MODE_LABEL: str = ""

    def __init__(self, services: DataServices, parent=None) -> None:
        """Initialize the data-management mode widget.

        Args:
            services (DataServices): Shared service object providing GUI
                freeze-state and progress signals.
            parent (QtWidgets.QWidget, optional): Parent widget. Defaults to
                None.

        Raises:
            NotImplementedError: If `MODE_KEY` or `MODE_LABEL` is not
                defined by the subclass.
        """
        super().__init__(parent)
        if not self.MODE_KEY or not self.MODE_LABEL:
            raise NotImplementedError(f"{type(self).__name__} must set MODE_KEY and MODE_LABEL")
        self.services = services

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.root = QtWidgets.QVBoxLayout(self)
        self.root.setContentsMargins(0, 0, 0, 0)
        self.root.setSpacing(12)

        # Shared-service subscriptions every mode gets for free. Subclasses can
        # ignore the ones they don't use by leaving the default no-op overrides.
        self.services.freeze_gui.connect(self.on_freeze)
        self.services.progress.connect(self._route_progress)

        self.build()

    def build(self) -> None:
        """Construct the mode's widgets and layout.

        Subclasses must override this method to populate `root` with the
        controls and content specific to the mode.

        Raises:
            NotImplementedError: Always raised by the base implementation.
        """
        raise NotImplementedError

    def on_enter(self) -> None:
        """Handle activation of the mode.

        Called by the parent container when this mode becomes the active page.
        Subclasses may override this method to perform initialization or
        refresh operations when entering the mode.
        """

    def on_leave(self) -> None:
        """Handle deactivation of the mode.

        Called by the parent container when switching away from this mode.
        Subclasses may override this method to perform cleanup or suspend
        mode-specific activity.
        """

    def on_freeze(self, frozen: bool) -> None:
        """Enable or disable controls while a shared task is running.

        This is a no-op hook in the base class. Subclasses can override it to
        disable or restore interactive controls in response to the shared
        `DataServices.freeze_gui` signal.

        Args:
            frozen (bool): Whether interactive controls should be frozen.
        """

    def on_progress(self, label: str, pct: int, color: str) -> None:
        """Handle a progress update routed to this mode.

        This is a no-op hook in the base class. Subclasses can override it to
        update their progress indicators when a progress event is published
        on their `MODE_KEY` channel.

        Args:
            label (str): Human-readable description of the current operation.
            pct (int): Progress percentage.
            color (str): Color associated with the progress state.
        """

    def _route_progress(
        self,
        channel: str,
        label: str,
        pct: int,
        color: str,
    ) -> None:
        """Route a service progress update to the matching mode.

        Progress events are published using named channels rather than
        positional mode indices. Updates are forwarded to `on_progress()`
        only when `channel` matches this widget's `MODE_KEY`.

        Args:
            channel (str): Named progress channel associated with the update.
            label (str): Human-readable description of the current operation.
            pct (int): Progress percentage.
            color (str): Color associated with the progress state.
        """
        if channel == self.MODE_KEY:
            self.on_progress(label, pct, color)
