import os
import sys


class Architecture:
    @staticmethod
    def get_path():
        """
        Returns the absolute path to the project root.
        Works for development and PyInstaller bundles.
        """
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
