import argparse

from QATCH.common.logger import Logger as Log
from QATCH.common.logger import LoggerLevel
from QATCH.core.constants import Constants


TAG = ""#"Arguments"


###############################################################################
# Creates and parses the arguments to be used by the application
###############################################################################
class Arguments:


    ###########################################################################
    # Initializes
    ###########################################################################
    def __init__(self):
        self._parser = None


    ###########################################################################
    # Creates and parses the arguments to be used by the application.
    ###########################################################################
    def create(self):
        parser = argparse.ArgumentParser(
                            description='SOFTWARE\nA real time plotting and logging application')
        parser.add_argument("-i", "--info",
                            dest="log_level_info",
                            action='store_true',
                            help="Enable info messages"
                            )

        parser.add_argument("-d", "--debug",
                            dest="log_level_debug",
                            action='store_true',
                            help="Enable debug messages"
                            )

        parser.add_argument("-v", "--verbose",
                            dest="log_to_console",
                            action='store_true',
                            help="Show log messages in console",
                            default=Constants.log_default_console_log
                            )

        parser.add_argument("-s", "--samples",
                            dest="user_samples",
                            default=Constants.argument_default_samples,
                            help="Specify number of sample to show on plot"
                            )
        self._parser = parser.parse_args()


    ###########################################################################
    # Sets the user specified log level
    ###########################################################################
    def set_user_log_level(self):
        if self._parser is not None:
            self._parse_log_level()
        else:
            Log.w(TAG, "Parser was not created!")
            return None

    ###########################################################################
    # Gets the user specified samples to show in the plot
    ###########################################################################
    def get_user_samples(self):
        """
        :return: Samples specified by user, or default value if not specified.
        :rtype: int.
        """
        return int(self._parser.user_samples)

    ###########################################################################
    # Gets the user specified log to console flag
    ###########################################################################
    def get_user_console_log(self):
        """
        :return: True if log to console is enabled.
        :rtype: bool.
        """
        return self._parser.log_to_console

    ###########################################################################
    # Sets the log level depending on user specification:
    # enable or disable log to console
    ###########################################################################
    def _parse_log_level(self):
        log_to_console = self.get_user_console_log()
        level = LoggerLevel.INFO
        if self._parser.log_level_info:
            level = LoggerLevel.INFO
        elif self._parser.log_level_debug:
            level = LoggerLevel.DEBUG
        Log.create()
