"""Smoke test for the default view opened after signing in."""

import os
from pathlib import Path
import unittest
from time import sleep

from PyQt5.QtTest import QTest

from .QtTest_framework import QtTestFramework, QtTestCase


class TestDefaultModeView(QtTestCase):
    """Verify that a valid user reaches the application's default view."""

    def setUp(self):
        self.qt_test = QtTestFramework()
        self.application = self.qt_test.launch_application()
        # Ensure the window is fully rendered and active before testing
        QTest.qWaitForWindowActive(self.application.win.MainWin)

    def tearDown(self):
        self.qt_test.hide_application(self.application)
        # self.process_events()
        # sleep(3)
        # self.process_events()
        # sleep(3)

    def test_admin_user_sign_in_opens_capture_mode_view(self):
        username = os.environ.get("QATCH_TEST_USERNAME", "TAU2")
        password = os.environ.get("QATCH_TEST_PASSWORD", "12345678")
        if not username or not password:
            self.skipTest(
                "Set QATCH_TEST_USERNAME and QATCH_TEST_PASSWORD to run this test"
            )

        sign_in = self.qt_test.wait_for_app_load(self.application, 30)
        if sign_in:
            mode_view = self.qt_test.get_current_view_mode(self.application)

            self.assertEqual(mode_view, "mode-signin")

            self.qt_test.enter_credentials(self.application, username, password)

        mode_view = self.qt_test.get_current_view_mode(self.application)

        self.assertEqual(mode_view, "mode-capture")

    def test_capture_user_sign_in_opens_capture_mode_view(self):
        username = os.environ.get("QATCH_TEST_USERNAME", "TCU")
        password = os.environ.get("QATCH_TEST_PASSWORD", "12345678")
        if not username or not password:
            self.skipTest(
                "Set QATCH_TEST_USERNAME and QATCH_TEST_PASSWORD to run this test"
            )

        sign_in = self.qt_test.wait_for_app_load(self.application, 30)
        if sign_in:
            mode_view = self.qt_test.get_current_view_mode(self.application)

            self.assertEqual(mode_view, "mode-signin")

            self.qt_test.enter_credentials(self.application, username, password)

        mode_view = self.qt_test.get_current_view_mode(self.application)

        self.assertEqual(mode_view, "mode-capture")

    def test_analyze_user_sign_in_opens_analyze_mode_view(self):
        username = os.environ.get("QATCH_TEST_USERNAME", "TAU")
        password = os.environ.get("QATCH_TEST_PASSWORD", "12345678")
        if not username or not password:
            self.skipTest(
                "Set QATCH_TEST_USERNAME and QATCH_TEST_PASSWORD to run this test"
            )

        sign_in = self.qt_test.wait_for_app_load(self.application, 30)
        if sign_in:
            mode_view = self.qt_test.get_current_view_mode(self.application)

            self.assertEqual(mode_view, "mode-signin")

            self.qt_test.enter_credentials(self.application, username, password)

        mode_view = self.qt_test.get_current_view_mode(self.application)

        self.assertEqual(mode_view, "mode-analyze")

    def test_operate_user_sign_in_opens_capture_mode_view(self):
        username = os.environ.get("QATCH_TEST_USERNAME", "TOU")
        password = os.environ.get("QATCH_TEST_PASSWORD", "12345678")
        if not username or not password:
            self.skipTest(
                "Set QATCH_TEST_USERNAME and QATCH_TEST_PASSWORD to run this test"
            )

        sign_in = self.qt_test.wait_for_app_load(self.application, 30)
        if sign_in:
            mode_view = self.qt_test.get_current_view_mode(self.application)

            self.assertEqual(mode_view, "mode-signin")

            self.qt_test.enter_credentials(self.application, username, password)

        mode_view = self.qt_test.get_current_view_mode(self.application)

        self.assertEqual(mode_view, "mode-capture")