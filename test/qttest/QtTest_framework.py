"""Headless, non-interactive test utilities for Qt-based software.

Test modules can subclass :class:`QtTestCase` and use ``assert_performance``
for repeatable performance checks. Run this module directly to discover and
execute tests with ``unittest``.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import time
import unittest
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer

from app import QATCH

# Configure Qt before importing a binding so tests do not require a display.
# os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _load_qt() -> tuple[Any, Any, Any, Optional[str]]:
    """Load the first installed supported Qt binding, if any."""
    for binding in ("PySide6", "PyQt6", "PySide2", "PyQt5"):
        try:
            package = __import__(binding, fromlist=["QtWidgets", "QtCore", "QtTest"])
            return package.QtWidgets, package.QtCore, package.QtTest, binding
        except (ImportError, AttributeError):
            continue
    return None, None, None, None


QtWidgets, QtCore, QtTest, QT_BINDING = _load_qt()


@dataclass(frozen=True)
class PerformanceGuideline:
    """Allowed timing limits for a repeatedly measured operation (seconds)."""

    max_median_seconds: Optional[float] = None
    max_p95_seconds: Optional[float] = None
    max_coefficient_of_variation: Optional[float] = None
    runs: int = 7

    def __post_init__(self) -> None:
        if self.runs < 2:
            raise ValueError("runs must be at least 2")
        for name in (
            "max_median_seconds",
            "max_p95_seconds",
            "max_coefficient_of_variation",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PerformanceResult:
    """Timing statistics from :func:`measure_performance`."""

    samples_seconds: tuple[float, ...]
    median_seconds: float
    p95_seconds: float
    coefficient_of_variation: float


def measure_performance(
    operation: Callable[[], Any], *, runs: int = 7, warmup_runs: int = 1
) -> PerformanceResult:
    """Measure an operation with a monotonic clock and optional warm-up."""
    if runs < 2:
        raise ValueError("runs must be at least 2")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    for _ in range(warmup_runs):
        operation()

    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)

    ordered = sorted(samples)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    mean = statistics.mean(samples)
    return PerformanceResult(
        samples_seconds=tuple(samples),
        median_seconds=statistics.median(samples),
        p95_seconds=p95,
        coefficient_of_variation=statistics.stdev(samples) / mean if mean else 0.0,
    )


class QtTestFramework:

    def launch_application(self) -> QATCH:
        # Keep unittest's command-line arguments from being parsed by Qt.
        app = QATCH([sys.argv[0]])
        app.run(interactive=True)  # non-blocking
        return app

    def wait_for_app_load(self, app: QATCH, timeout_sec: int = 30) -> bool:
        print("Waiting for app load...")
        # time.sleep(3)
        wait_time = 0
        while not app.win.ReadyToShow:
            QTest.qSleep(1000)  # time.sleep(1)
            wait_time += 1
            if wait_time > timeout_sec:
                print("Timeout waiting for app load")
                return False
        print("App loaded!")
        return True

    def click_modal_if_blocking(self, button: QMessageBox.StandardButton) -> bool:
         # Find the currently active modal window
        box = QApplication.activeWindow()
        if isinstance(box, QMessageBox):
            QTest.qSleep(100)  # time.sleep(0.1)
            # Extract the actual QPushButton corresponding to 'Yes', 'No', or 'Cancel'
            button_widget = box.button(button)
            QTest.mouseClick(button_widget, QtCore.Qt.MouseButton.LeftButton)
            return True
        return False

    def get_current_view_mode(self, app: QATCH) -> str:
        current_mode_widget = app.win.MainWin.ui0.splitter.widget(0)
        if current_mode_widget is app.win.MainWin.ui0.userview:
            return "mode-signin"
        if current_mode_widget is app.win.MainWin.ui0.runview:
            return "mode-capture"
        if current_mode_widget is app.win.MainWin.ui0.analyze:
            return "mode-analyze"
        if current_mode_widget is app.win.MainWin.ui0.learn_ui:
            return "tool-visqai"
        if current_mode_widget is app.win.MainWin.ui0.donnan_ui:
            return "tool-donnan-calc"
        if current_mode_widget is app.win.MainWin.ui0.injection_ui:
            return "tool-inject-calc"
        return "unknown"

    def handle_dialog(self, button: QMessageBox.StandardButton, max_tries: int, attempt: int = 1):
        if attempt < max_tries and not self.click_modal_if_blocking(button):
            QTimer.singleShot(100, lambda: self.handle_dialog(button, max_tries, attempt+1))
        else:
            self.DIALOG_TIMER_ACTIVE = False

    def enter_credentials(self, app: QATCH, initials: str, password: str) -> bool:
        # The native hacky way:
        # app.win.LoginWin.ui5.user_initials.setText(initials)
        # app.win.LoginWin.ui5.user_password.setText(password)
        # app.win.LoginWin.ui5.sign_in.click()

        # The QTest way:
        QTest.mouseClick(app.win.LoginWin.ui5.user_initials, QtCore.Qt.MouseButton.LeftButton)
        QTest.keyClicks(app.win.LoginWin.ui5.user_initials, initials)
        assert app.win.LoginWin.ui5.user_initials.text() == initials

        QTest.mouseClick(app.win.LoginWin.ui5.user_password, QtCore.Qt.MouseButton.LeftButton)
        QTest.keyClicks(app.win.LoginWin.ui5.user_password, password)
        assert app.win.LoginWin.ui5.user_password.text() == password

        self.DIALOG_TIMER_ACTIVE = True
        QTimer.singleShot(100, lambda: self.handle_dialog(QMessageBox.No, max_tries=100))
        QTest.mouseClick(app.win.LoginWin.ui5.sign_in, QtCore.Qt.MouseButton.LeftButton)
        while self.DIALOG_TIMER_ACTIVE:
            QtWidgets.QApplication.processEvents()

    def close_application(self, app: QATCH) -> None:

        self.DIALOG_TIMER_ACTIVE = True
        QTimer.singleShot(100, lambda: self.handle_dialog(QMessageBox.Yes, max_tries=10))
        app.win.MainWin.close()
        while self.DIALOG_TIMER_ACTIVE:
            QtWidgets.QApplication.processEvents()

    def hide_application(self, app: QATCH) -> None:
        app.win.MainWin.hide()

    def minimize_application(self, app: QATCH) -> None:
        app.win.MainWin.setWindowState(QtCore.Qt.WindowMinimized)

class QtTestCase(unittest.TestCase):
    """Unittest base class that provides a shared headless Qt application."""

    _qt_application: Any = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if QtWidgets is None:
            raise unittest.SkipTest("No supported Qt binding is installed")
        app = QtWidgets.QApplication.instance()
        if app is None:
            # Keep unittest's command-line arguments from being parsed by Qt.
            cls._qt_application = QtWidgets.QApplication([sys.argv[0]])
        else:
            cls._qt_application = app

    def process_events(self) -> None:
        """Process pending Qt events without starting an interactive loop."""
        if QtWidgets is not None and QtWidgets.QApplication.instance() is not None:
            QtWidgets.QApplication.processEvents()

    def assert_performance(
        self,
        operation: Callable[[], Any],
        guideline: PerformanceGuideline,
        *,
        warmup_runs: int = 1,
        message: Optional[str] = None,
    ) -> PerformanceResult:
        """Measure an operation and assert its timing meets the guideline."""
        result = measure_performance(
            operation, runs=guideline.runs, warmup_runs=warmup_runs
        )
        prefix = f"{message}: " if message else ""
        checks = (
            (guideline.max_median_seconds, result.median_seconds, "median", ".6f"),
            (guideline.max_p95_seconds, result.p95_seconds, "95th percentile", ".6f"),
            (
                guideline.max_coefficient_of_variation,
                result.coefficient_of_variation,
                "coefficient of variation",
                ".3f",
            ),
        )
        for maximum, actual, label, number_format in checks:
            if maximum is not None:
                self.assertLessEqual(
                    actual,
                    maximum,
                    f"{prefix}{label} {format(actual, number_format)} exceeds "
                    f"{format(maximum, number_format)}",
                )
        return result


def run_tests(
    start_directory: str = ".", pattern: str = "test*.py", verbosity: int = 2
) -> unittest.result.TestResult:
    """Discover and run tests without prompts or a GUI event loop."""
    suite = unittest.defaultTestLoader.discover(
        start_dir=start_directory, pattern=pattern
    )
    return unittest.TextTestRunner(verbosity=verbosity).run(suite)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-directory", default=".", help="directory to discover tests in"
    )
    parser.add_argument("--pattern", default="test*.py", help="test filename pattern")
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args(argv)
    result = run_tests(args.start_directory, args.pattern, args.verbosity)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
