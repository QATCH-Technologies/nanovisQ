"""
logger.py

This module provides the Logger class for configuring and managing
logging handlers, file rotation, and console output, as well as the
LoggerLevel enum defining available log levels.

This implementation is backed by `loguru` but preserves the original
public API (Logger.create / d / i / w / e / close, write / flush for
stdout redirection, and the LoggerLevel enum) so that no call sites
need to change.

Author:
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-17

Version:
    2.0.0
"""

import logging
import sys
import os
from enum import Enum

from loguru import logger as _loguru

from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.core.constants import Constants

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Module-level flag so create() is idempotent (mirrors the old
# `if top_level_logger.hasHandlers(): return` guard).
_CONFIGURED = False


class _LogHandler(logging.Handler):
    """
    Route records from the stdlib `logging` module into loguru.

    This captures output from third-party libraries (TensorFlow, absl,
    urllib3, etc.) that log via the standard library, so everything ends
    up in the same sinks with consistent formatting.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk back to the frame that originated the logging call so the
        # reported module/function is accurate rather than this shim.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


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
        global _CONFIGURED
        if _CONFIGURED:
            return
        _loguru.remove()
        FileManager.create_dir(Constants.log_export_path)
        log_path = f"{Constants.log_export_path}/{Constants.log_filename}"
        _loguru.add(
            log_path,
            level="DEBUG",
            rotation=Constants.log_max_bytes,
            retention=0,  # backupCount=0 in the original
            encoding="utf-8",
            enqueue=True,  # keep ONLY for cross-process-safe file writes
            backtrace=False,  # was True; avoid extra-frame capture per record
            diagnose=False,  # was True; variable-level introspection is expensive
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {process} | {thread.name: <10} | "
                "{level: <8} | {name}:{function}:{line} - {message}"
            ),
        )
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

        _loguru.level("DEBUG", color="<dim>")
        _loguru.level("INFO", color="<green>")
        _loguru.level("WARNING", color="<yellow><bold>")
        _loguru.level("ERROR", color="<red><bold>")
        _loguru.level("CRITICAL", color="<RED><white><bold>")

        _loguru.add(
            sys.stdout,
            level="INFO",
            colorize=True,
            enqueue=False,  # main-process console; no async queue needed
            format=(
                "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> | "
                "<level>{level: <8}</level> | "
                "<magenta>{name}:{line}</magenta> | "
                "<level>{message}</level>"
            ),
        )

        try:
            from absl import logging as absl_logging

            absl_logging.set_verbosity(absl_logging.ERROR)
            absl_logging.set_stderrthreshold("error")
        except ImportError:
            pass

        # NOTE: Bridge stdlib logging into loguru, but only WARNING and above.
        # Anything below that from third-party libs (matplotlib font manager,
        # urllib3, PIL, etc.) is noise we never want only our own Log.*
        # calls, which go through loguru directly, should appear at DEBUG.
        logging.basicConfig(handlers=[_LogHandler()], level=logging.WARNING, force=True)
        for noisy in (
            "absl",
            "tensorflow",
            "matplotlib",
            "matplotlib.font_manager",
            "PIL",
            "urllib3",
            "asyncio",
        ):
            lg = logging.getLogger(noisy)
            lg.handlers.clear()
            lg.setLevel(logging.ERROR)
            lg.propagate = True

        _CONFIGURED = True
        Logger.i("SYSTEM", "Added logging handlers")

    def write(self, message) -> None:
        """
        Write a message to the logger, filtering out internal logger calls.
        Redirects standard output streams cleanly.

        Args:
            message(str): The message to write.

        Returns:
            None
        """
        tag = "STDERR"
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
        return "<%s (%s)>" % (self.__class__.__name__, "QATCH.logger")

    @staticmethod
    def close() -> None:
        """
        Shutdown the logging system, closing all handlers.

        Returns:
            None
        """
        global _CONFIGURED
        top_level_logger = logging.getLogger("QATCH")
        Logger.i("SYSTEM", "Closed logging handlers")
        _loguru.remove()
        top_level_logger.handlers.clear()
        _CONFIGURED = False

    @staticmethod
    def _log(level, tag, msg) -> None:
        """
        Internal dispatch: formats tag and message elegantly, escapes
        loguru markup so literals aren't parsed as color tags, and ensures
        the correct calling frame depth is captured.
        """
        text = f"{tag} {msg}".strip() if msg is not None else str(tag)
        _loguru.opt(depth=2).log(level, "{}", text)

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
        Logger._log("DEBUG", tag, msg)

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
        Logger._log("INFO", tag, msg)

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
        Logger._log("WARNING", tag, msg)

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
        Logger._log("ERROR", tag, msg)

    @staticmethod
    def _show_user_info() -> None:
        """
        Log user and system information at info and debug levels.

        Displays application name, version, build date,
        OS details, Python version, and working directory.

        Returns:
            None
        """
        tag = "SYSINFO"
        header = f" {Constants.app_title} {Constants.app_version} "
        Logger.i(tag, "-" * len(header))
        Logger.i(tag, header)
        Logger.i(tag, "-" * len(header))
        Logger.i(tag, f"Build Date: {Constants.app_date}")
        Logger.i(tag, "SYSTEM INFORMATIONS:")
        Logger.i(tag, f"PC Name: {Architecture.get_os_name()}")
        Logger.i(tag, f"Platform: {Architecture.get_os_type()}")
        Logger.i(tag, f"Python version: {Architecture.get_python_version()}")
        Logger.i(tag, f"Path: {os.getcwd()}")
        if getattr(sys, "frozen", False):
            Logger.d(tag, f"_MEIPASS: {sys._MEIPASS}")
        Logger.i(tag, "-" * len(header))

        frozen = "ever so" if getattr(sys, "frozen", False) else "not"
        bundle_dir = (
            sys._MEIPASS
            if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__))
        )

        Logger.d("DEBUG", "=== DEBUG INFORMATIONS ===")
        Logger.d("DEBUG", f"we are {frozen} frozen")
        Logger.d("DEBUG", f"bundle dir is {bundle_dir}")
        Logger.d("DEBUG", f"sys.argv[0] is {sys.argv[0]}")
        Logger.d("DEBUG", f"sys.executable is {sys.executable}")
        Logger.d("DEBUG", f"os.getcwd is {os.getcwd()}")


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
