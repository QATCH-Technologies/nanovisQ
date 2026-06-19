"""Recover mode — wraps the existing self-contained RunRecoveryDialog.

This mode already exists as a standalone widget; the submodule simply hosts it
inside the mode-page contract so the container can treat it like any other mode.
No logic to port.
"""

from QATCH.ui.widgets.data_mode_base import DataModeWidget
from QATCH.ui.dialogs.run_recovery_dialog import RunRecoveryDialog


class RecoverMode(DataModeWidget):
    MODE_KEY = "recover"
    MODE_LABEL = "Recover"

    def build(self):
        self.recovery = RunRecoveryDialog()
        self.root.addWidget(self.recovery)

    def on_enter(self):
        # If RunRecoveryDialog has a refresh/reload entry point, call it here.
        pass
