"""
architecture.py

Minimal Architecture module for VisQ.AI application to run headless.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os
import sys


class Architecture:
    @staticmethod
    def get_path():
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))
