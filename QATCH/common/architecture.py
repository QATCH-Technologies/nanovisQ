import platform
import sys
import os

from enum import Enum

###############################################################################
# Architecture specific methods: OS types, Python version
###############################################################################


class Architecture:

    ###########################################################################
    # Gets the current OS
    ###########################################################################
    @staticmethod
    def get_os():
        # :return: OS type by OSType enum.
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
        # :return: OS name :rtype: str.
        return platform.node()

    ###########################################################################
    # Gets the current OS type string (as reported by platform)
    ###########################################################################
    @staticmethod
    def get_os_type():
        # :return: OS type :rtype: str.
        return platform.platform()

    ###########################################################################
    # Gets the PWD or CWD of the currently running application
    # (Print Working Directory, Change Working Directory)
    ###########################################################################
    @staticmethod
    def get_path():
        # :return: Path of the PWD or CWD :rtype: str.
        # :if EXE: Temporary path of extracted bundle.
        # :if _PY: Working directory of parent module.
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
        # :return: Python version formatted as major.minor.release :rtype: str.
        version = sys.version_info
        return str("{}.{}.{}".format(version[0], version[1], version[2]))

    ###########################################################################
    # Checks if the running Python version is >= than the specified version
    ###########################################################################

    @staticmethod
    def is_python_version(major, minor=0):
        """
        :param major: Major value of the version :type major: int.
        :param minor: Minor value of the version :type minor: int.
        :return: True if the version specified is >= than the current version.
        :rtype: bool.
        """
        version = sys.version_info
        if version[0] >= major and version[1] >= minor:
            return True
        return False

###############################################################################
# Enum for OS types
###############################################################################


class OSType(Enum):
    unknown = 0
    linux = 1
    macosx = 2
    windows = 3
