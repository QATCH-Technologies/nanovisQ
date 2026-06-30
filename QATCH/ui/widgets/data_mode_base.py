"""DataModeWidget - base class for every data-management mode page.

Each mode (import, export, recover, advanced, history) subclasses this. The
base wires up the shared ``DataServices`` connections and defines the small
contract the parent container relies on, so the container can treat every mode
uniformly and never needs to know mode internals.

Contract a subclass implements:
    MODE_KEY    : str               - unique key ("import", "export", ...)
    MODE_LABEL  : str               - segmented-control label ("Import", ...)
    build()                         - construct the page's widgets/layout
    on_enter()                      - called when this mode becomes active
    on_leave()                      - called when switching away (optional)
    on_freeze(frozen: bool)         - enable/disable controls (optional)

The base provides ``self.services`` and a transparent root layout to build on.
"""

from PyQt5 import QtCore, QtWidgets

from QATCH.common.data_service import DataServices

TAG = "[DataMode]"


class DataModeWidget(QtWidgets.QWidget):
    MODE_KEY: str = ""
    MODE_LABEL: str = ""

    def __init__(self, services: DataServices, parent=None):
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

    # ---- Subclass hooks -------------------------------------------------
    def build(self):
        """Construct the page. Override in subclasses."""
        raise NotImplementedError

    def on_enter(self):
        """Called when this mode becomes the active page."""

    def on_leave(self):
        """Called when switching away from this mode."""

    def on_freeze(self, frozen: bool):
        """Enable/disable interactive controls during a running task."""

    def on_progress(self, label: str, pct: int, color: str):
        """Receive progress updates addressed to THIS mode's channel."""

    # ---- Internal -------------------------------------------------------
    def _route_progress(self, channel: str, label: str, pct: int, color: str):
        """Only forward progress whose channel matches this mode's key.

        This replaces the old setProgress(tab=0/1) index routing: each mode
        listens on its own named channel and updates its own progress display.
        """
        if channel == self.MODE_KEY:
            self.on_progress(label, pct, color)
