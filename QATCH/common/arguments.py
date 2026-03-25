"""
arguments.py

Command-line argument parsing module for the QATCH application.

This module provides the Arguments class, which encapsulates the logic for
defining, parsing, and retrieving command-line configurations such as
logging verbosity and plot sample counts.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-20
"""

import argparse

from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants

TAG = "[Arguments]"


class Arguments:
    """Handles creation and parsing of command-line arguments.

    This class manages the lifecycle of the ArgumentParser and provides
    getter methods to access user-defined configurations for logging
    and data visualization.

    Attributes:
        _parser (argparse.Namespace, optional): The parsed command-line
            arguments. Defaults to None until create() is called.
    """

    def __init__(self):
        """Initializes the Arguments class with a null parser."""
        self._parser = None

    def create(self):
        """Creates and parses command-line arguments for the application.

        Defines flags for info/debug logging, console verbosity, and sample
        counts. Results are stored in the `_parser` attribute.

        Supported arguments:
            -i / --info: Enable info level logs.
            -d / --debug: Enable debug level logs.
            -v / --verbose: Force logging to the console.
            -s / --samples: Integer count of samples for plotting.
        """
        parser = argparse.ArgumentParser(
            description="SOFTWARE\nA real time plotting and logging application"
        )
        parser.add_argument(
            "-i",
            "--info",
            dest="log_level_info",
            action="store_true",
            help="Enable info messages",
        )
        parser.add_argument(
            "-d",
            "--debug",
            dest="log_level_debug",
            action="store_true",
            help="Enable debug messages",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            dest="log_to_console",
            action="store_true",
            help="Show log messages in console",
            default=Constants.log_default_console_log,
        )
        parser.add_argument(
            "-s",
            "--samples",
            dest="user_samples",
            default=Constants.argument_default_samples,
            help="Specify number of sample to show on plot",
        )
        self._parser = parser.parse_args()

    def set_user_log_level(self):
        """Sets the global log level based on parsed command-line arguments.

        If the parser has not been initialized via create(), a warning is logged.
        """
        if self._parser is not None:
            self._parse_log_level()
        else:
            Log.w(TAG, "Parser was not created!")

    def get_user_samples(self) -> int:
        """Retrieves the number of samples to display.

        Returns:
            int: The user-specified sample count or the default constant value.
        """
        return int(self._parser.user_samples)

    def get_user_console_log(self) -> bool:
        """Checks if console logging is enabled.

        Returns:
            bool: True if logs should be output to the console, False otherwise.
        """
        return self._parser.log_to_console

    def _parse_log_level(self):
        """Configures the Logger instance based on internal parser values.

        This is an internal helper called by set_user_log_level to trigger
        the actual Logger initialization.
        """
        Log.create()
