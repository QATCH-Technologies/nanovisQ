"""
architecture.py

Module for system architecture detection and environment metadata.

This module provides utilities to identify the underlying operating system,
verify Python runtime versions, and manage directory paths—handling both
standard Python environments and executable bundles.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)
    Other QATCH Technologies contributors

Date:
    2026-03-20
"""

from enum import Enum
import os
import platform
import sys


class OSType(Enum):
    """
    Enum representing supported operating system types.
    """

    unknown = 0
    linux = 1
    macosx = 2
    windows = 3


class Architecture:
    """
    Provides architecture-specific methods for OS detection, Python versioning,
    and working directory management. All methods are static and utility-focused.
    """

    @staticmethod
    def get_os() -> OSType:
        """
        Detects the current operating system and returns its type as an OSType enum.
        Returns:
            OSType: Enum value representing the OS type (linux, windows, macosx, unknown).
        """
        tmp = str(Architecture.get_os_type())
        if "Linux" in tmp:
            return OSType.linux
        elif "Windows" in tmp:
            return OSType.windows
        elif "Darwin" in tmp or "macOS" in tmp:
            return OSType.macosx
        else:
            return OSType.unknown

    @staticmethod
    def get_os_name() -> str:
        """
        Retrieves the name of the current operating system node (hostname).

        Returns:
            str: The OS node name as reported by platform.node().
        """
        return platform.node()

    @staticmethod
    def get_os_type() -> str:
        """
        Retrieves the platform type string for the current operating system.

        Returns:
            str: The OS type string as reported by platform.platform().
        """
        return platform.platform()

    @staticmethod
    def get_path() -> str:
        """
        Retrieves the current working directory or the temporary path if running from an EXE bundle.

        Returns:
            str: Path of the current working directory or bundle extraction directory.
        """
        if getattr(sys, "frozen", False):
            # we are running in a bundle from an EXE
            bundle_dir = sys._MEIPASS  # type: ignore[attr-defined]
        else:
            # we are running in a normal Python environment
            bundle_dir = os.getcwd()
        return bundle_dir

    @staticmethod
    def get_python_version() -> str:
        """
        Returns the running Python version as a string in 'major.minor.release' format.

        Returns:
            str: Python version string.
        """
        version = sys.version_info
        return str("{}.{}.{}".format(version[0], version[1], version[2]))

    @staticmethod
    def is_python_version(major: int, minor: int = 0) -> bool:
        """
        Checks if the running Python version is greater than or equal to the specified version.

        Args:
            major (int): Major version to check against.
            minor (int, optional): Minor version to check against. Defaults to 0.

        Returns:
            bool: True if the running Python version is >= specified version, False otherwise.
        """
        version = sys.version_info
        if version[0] >= major and version[1] >= minor:
            return True
        return False
