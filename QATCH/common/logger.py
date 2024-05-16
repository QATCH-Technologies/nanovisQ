import logging
import logging.handlers
import sys
import os
from enum import Enum

from QATCH.common.architecture import Architecture
from QATCH.common.fileManager import FileManager
from QATCH.core.constants import Constants

###############################################################################
# Logging package - All packages can use this module
###############################################################################
class Logger:

    ###########################################################################
    # Creates logging file (.txt)
    ###########################################################################
    @staticmethod
    def create():
        """
        :param level: Level to show in log.
        :type level: int.
        """
        log_format_file = logging.Formatter(
            fmt = '%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
            datefmt = None)
        log_format_console = logging.Formatter(
            fmt = '%(asctime)s\t%(levelname)s\t%(message)s',
            datefmt = '%Y-%m-%d %I:%M:%S %p')

        top_level_logger = logging.getLogger("QATCH")
        top_level_logger.setLevel(logging.DEBUG)

        if (top_level_logger.hasHandlers()):
            #Logger.d("Skipping addHandlers")
            return

        FileManager.create_dir(Constants.log_export_path)
        file_handler = logging.handlers.RotatingFileHandler("{}/{}"
                                                            .format(Constants.log_export_path, Constants.log_filename),
                                                            maxBytes=Constants.log_max_bytes,
                                                            backupCount=0)
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

        #Logger.d("Added handlers successfully")
        top_level_logger.info("Added logging handlers")

    def write(self, message):
        tag = "ERROR"
        msg = message.strip()
        if msg.find("QATCH.logger") >= 0:
            return
        if len(msg) > 0:
            Logger.e(tag, msg)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def __repr__(self):
        return '<%s (%s)>' % (self.__class__.__name__, "QATCH.logger")

    ###########################################################################
    # Closes the enabled loggers.
    ###########################################################################
    @staticmethod
    def close():
        logging.shutdown()

    ###########################################################################
    # Logs at debug level (debug,info,warning and error messages)
    ###########################################################################
    @staticmethod
    def d(tag, msg=""):
        """
        :param tag: TAG to identify the log :type tag: str.
        :param msg: Message to log.         :type msg: str.
        """
        logger = logging.getLogger("QATCH.logger")
        logger.debug("{} {}".format(str(tag), str(msg)))

    ####
    @staticmethod
    def i(tag, msg=""):
        logger = logging.getLogger("QATCH.logger")
        logger.info("{} {}".format(str(tag), str(msg)))

    ####
    @staticmethod
    def w(tag, msg=""):
        logger = logging.getLogger("QATCH.logger")
        logger.warning("{} {}".format(str(tag), str(msg)))

    ####
    @staticmethod
    def e(tag, msg=""):
        logger = logging.getLogger("QATCH.logger")
        logger.error("{} {}".format(str(tag), str(msg)))


    ###########################################################################
    # logs and prints architecture-related informations
    ###########################################################################
    @staticmethod
    def _show_user_info():
        tag = ""#"[User]"
        str = " {} {} ".format(Constants.app_title,Constants.app_version)
        Logger.i("-" * len(str))
        Logger.i(str)
        Logger.i("-" * len(str))
        Logger.i(tag,"Build Date: {}".format(Constants.app_date))
        Logger.i("{} SYSTEM INFORMATIONS:".format(tag))
        Logger.i(tag, "PC Name: {}".format(Architecture.get_os_name()))
        Logger.i(tag, "Platform: {}".format(Architecture.get_os_type()))
        Logger.i(tag, "Python version: {}".format(Architecture.get_python_version()))
        Logger.i(tag, "Path: {}".format(os.getcwd()))
        if getattr(sys, 'frozen', False):
            Logger.d(tag, "_MEIPASS: {}".format(sys._MEIPASS))
        Logger.i("-" * len(str))

        frozen = 'not'
        if getattr(sys, 'frozen', False):
            # we are running in a bundle
            frozen = 'ever so'
            bundle_dir = sys._MEIPASS
        else:
            # we are running in a normal Python environment
            bundle_dir = os.path.dirname(os.path.abspath(__file__))
        Logger.d( "=== DEBUG INFORMATIONS ===")
        Logger.d( f'we are {frozen} frozen')
        Logger.d( 'bundle dir is', bundle_dir )
        Logger.d( 'sys.argv[0] is', sys.argv[0] )
        Logger.d( 'sys.executable is', sys.executable )
        Logger.d( 'os.getcwd is', os.getcwd() )


###############################################################################
# Enumeration for the Logger levels
###############################################################################
class LoggerLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    ANY = logging.NOTSET
