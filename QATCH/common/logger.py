"""
Logging module for the QATCH Nanovis system.

This module provides the Logger class for configuring and managing
logging handlers, file rotation, and console output, as well as the 
LoggerLevel enum defining available log levels.

Author:
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-16

Version:
    ?
"""
import logging
import logging.handlers
import sys
import os
import warnings
from enum import Enum

from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.core.constants import Constants

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


class Logger:
    """
    Central logger for the QATCH Nanovis application.
    """

    @staticmethod
    def create() -> None:
        """
        Create and configure the QATCH root logger.

        Returns:
            None
        """
        try:
            from absl import logging as absl_logging
            absl_logging.set_verbosity(absl_logging.ERROR)
            absl_logging.set_stderrthreshold('error')
            logging.getLogger('absl').setLevel(logging.ERROR)
        except ImportError:
            pass

        log_format_file = logging.Formatter(
            fmt='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s'
        )
        log_format_console = logging.Formatter(
            fmt='%(asctime)s\t%(levelname)s\t%(message)s',
            datefmt='%Y-%m-%d %I:%M:%S %p'
        )

        top_level_logger = logging.getLogger("QATCH")
        top_level_logger.setLevel(logging.DEBUG)

        if top_level_logger.hasHandlers():
            return

        FileManager.create_dir(Constants.log_export_path)
        file_handler = logging.handlers.RotatingFileHandler(
            f"{Constants.log_export_path}/{Constants.log_filename}",
            maxBytes=Constants.log_max_bytes,
            backupCount=0
        )
        file_handler.setFormatter(log_format_file)
        file_handler.setLevel(logging.DEBUG)
        top_level_logger.addHandler(file_handler)

        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format_console)
        console_handler.setLevel(logging.INFO)
        top_level_logger.addHandler(console_handler)

        top_level_logger.info("Added logging handlers")

    def write(self, message) -> None:
        """
        Write a message to the logger, filtering out internal logger calls.

        Args:
            message(str): The message to write.

        Returns:
            None
        """
        tag = "ERROR"
        msg = message.strip()
        if msg.find("QATCH.logger") >= 0:
            return
        if len(msg) > 0:
            Logger.e(tag, msg)

    def flush(self) -> None:
        """
        Flush method for compatibility with file-like interfaces.

        Currently a no-op for Python 3 compatibility.

        Returns:
            None
        """
        pass

    def __repr__(self) -> str:
        """
        Return the representation of the Logger instance.

        Returns:
            str: Representation string.
        """
        return '<%s (%s)>' % (self.__class__.__name__, "QATCH.logger")

    @staticmethod
    def close() -> None:
        """
        Shutdown the logging system, closing all handlers.

        Returns:
            None
        """
        # The Python interpreter calls logging.shutdown() automatically at exit.
        # Manually calling it inside a handler's own close method can cause recursion.
        # logging.shutdown()
        top_level_logger = logging.getLogger("QATCH")
        top_level_logger.info("Closed logging handlers")
        top_level_logger.handlers.clear()

    @staticmethod
    def d(tag, msg="") -> None:
        """
        Log a debug-level message with a tag.

        Args:
            tag(str): Identifier tag for the log.
            msg(str): Message content.

        Returns:
            None
        """
        logger = logging.getLogger("QATCH.logger")
        logger.debug(f"{tag} {msg}")

    @staticmethod
    def i(tag, msg="") -> None:
        """
        Log an info-level message with a tag.

        Args:
            tag(str): Identifier tag for the log.
            msg(str): Message content.

        Returns:
            None
        """
        logger = logging.getLogger("QATCH.logger")
        logger.info(f"{tag} {msg}")

    @staticmethod
    def w(tag, msg="") -> None:
        """
        Log a warning-level message with a tag.

        Args:
            tag(str): Identifier tag for the log.
            msg(str): Message content.

        Returns:
            None
        """
        logger = logging.getLogger("QATCH.logger")
        logger.warning(f"{tag} {msg}")

    @staticmethod
    def e(tag, msg="") -> None:
        """
        Log an error-level message with a tag.

        Args:
            tag(str): Identifier tag for the log.
            msg(str): Message content.

        Returns:
            None
        """
        logger = logging.getLogger("QATCH.logger")
        logger.error(f"{tag} {msg}")

    @staticmethod
    def _show_user_info() -> None:
        """
        Log user and system information at info and debug levels.

        Displays application name, version, build date,
        OS details, Python version, and working directory.

        Returns:
            None
        """
        tag = ""
        header = f" {Constants.app_title} {Constants.app_version} "
        Logger.i("-" * len(header))
        Logger.i(header)
        Logger.i("-" * len(header))
        Logger.i(tag, f"Build Date: {Constants.app_date}")
        Logger.i(f"{tag} SYSTEM INFORMATIONS:")
        Logger.i(tag, f"PC Name: {Architecture.get_os_name()}")
        Logger.i(tag, f"Platform: {Architecture.get_os_type()}")
        Logger.i(tag, f"Python version: {Architecture.get_python_version()}")
        Logger.i(tag, f"Path: {os.getcwd()}")
        if getattr(sys, 'frozen', False):
            Logger.d(tag, f"_MEIPASS: {sys._MEIPASS}")
        Logger.i("-" * len(header))

        frozen = 'not'
        if getattr(sys, 'frozen', False):
            frozen = 'ever so'
            bundle_dir = sys._MEIPASS
        else:
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        Logger.d("=== DEBUG INFORMATIONS ===")
        Logger.d(f"we are {frozen} frozen")
        Logger.d('bundle dir is', bundle_dir)
        Logger.d('sys.argv[0] is', sys.argv[0])
        Logger.d('sys.executable is', sys.executable)
        Logger.d('os.getcwd is', os.getcwd())


class LoggerLevel(Enum):
    """
    Enumeration for available logger levels.

    Attributes:
        CRITICAL(int): Critical level.
        ERROR(int): Error level.
        WARNING(int): Warning level.
        INFO(int): Info level.
        DEBUG(int): Debug level.
        ANY(int): All messages.
    """
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    ANY = logging.NOTSET
