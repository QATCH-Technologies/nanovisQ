from typing import TYPE_CHECKING

from QATCH.ui.interfaces import UIAnalyze
from QATCH.ui.windows.base_window import BaseWindow

if TYPE_CHECKING:
    from QATCH.ui.main_window import MainWindow


class AnalyzeWindow(BaseWindow):
    def __init__(self, parent: "MainWindow") -> None:
        super().__init__()
        self.ui = UIAnalyze()
        self.ui.setup_ui(self, parent)
