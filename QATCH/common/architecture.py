import platform
import sys
import os

from enum import Enum

###############################################################################
# Architecture specific methods: OS types, Python version
###############################################################################


class Architecture:
    """
    Provides architecture-specific methods for OS detection, Python versioning,
    and working directory management. All methods are static and utility-focused.
    """

    ###########################################################################
    # Gets the current OS
    ###########################################################################
    @staticmethod
    def get_os():
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
        elif "Darwin" in tmp:
            return OSType.macosx
        elif "macOS" in tmp:
            return OSType.macosx
        else:
            return OSType.unknown

    ###########################################################################
    # Gets the current OS name string (as reported by platform)
    ###########################################################################
    @staticmethod
    def get_os_name():
        """
        Returns the name of the current operating system node (hostname).
        Returns:
            str: The OS node name as reported by platform.node().
        """
        return platform.node()

    ###########################################################################
    # Gets the current OS type string (as reported by platform)
    ###########################################################################
    @staticmethod
    def get_os_type():
        """
        Returns the platform type string for the current operating system.
        Returns:
            str: The OS type string as reported by platform.platform().
        """
        return platform.platform()

    ###########################################################################
    # Gets the PWD or CWD of the currently running application
    # (Print Working Directory, Change Working Directory)
    ###########################################################################
    @staticmethod
    def get_path():
        """
        Returns the current working directory or the temporary path if running from an EXE bundle.
        Returns:
            str: Path of the current working directory or bundle extraction directory.
        """
        if getattr(sys, 'frozen', False):
            # we are running in a bundle from an EXE
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.getcwd()
        return bundle_dir

    ###########################################################################
    # Gets the running Python version
    ###########################################################################
    @staticmethod
    def get_python_version():
        """
        Returns the running Python version as a string in 'major.minor.release' format.
        Returns:
            str: Python version string.
        """
        version = sys.version_info
        return str("{}.{}.{}".format(version[0], version[1], version[2]))

    ###########################################################################
    # Checks if the running Python version is >= than the specified version
    ###########################################################################

    @staticmethod
    def is_python_version(major, minor=0):
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

###############################################################################
# Enum for OS types
###############################################################################


class OSType(Enum):
    """
    Enum representing supported operating system types.
    """
    unknown = 0
    linux = 1
    macosx = 2
    windows = 3
