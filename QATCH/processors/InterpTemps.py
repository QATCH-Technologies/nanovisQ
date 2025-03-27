"""
InterpTemps.py

InterpTemps processes pre-compressed raw run data files by searching for "nan"
values in the Temperature data values, replacing them with interpolated real values.

This module provides a multiprocessor process that handles non-blocking real-time
processing to allow this activity to occur in the background while the user is
entering manual inputs to the post-run Run Info dialog window.

The Run Info "Save" action must block the `confirm()` action until this process
has been marked as finished; otherwise, runs may be compressed partially processed.

Author: Alexander Ross
Date: 2025-03-27
"""
from datetime import datetime
from enum import Enum
import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from QATCH.common.logger import Logger as Log
    STANDALONE_MODE = False
except:
    STANDALONE_MODE = True

    class Log():
        @staticmethod
        def d(t, s=""):
            logging.debug(f"{t} {s}")

        def i(t, s=""):
            logging.info(f"{t} {s}")

        def w(t, s=""):
            logging.warning(f"{t} {s}")

        def e(t, s=""):
            logging.error(f"{t} {s}")

TAG = "[InterpTemps]"


class ActionType(Enum):
    """
    Enumeration of action types for queue operations related to temperature data processing.

    This enum defines the possible actions that can be performed on temperature data files:

    - load: Load real temperature values from a file, intended for later interpolation.
    - interp: Interpolate missing (NaN) temperature values using data loaded by the 'load' action.

    Note:
        When using the 'interp' action, include the `flag_missing` key in the command dictionary to mark
        interpolated values as missing (e.g., displayed in red in output tables).
    """

    load = 0  # Load real temperatures from file for later interpolation.
    interp = 1  # Interpolate NaN temperatures from file using prior load.

    def __str__(self) -> str:
        """
        Return the string representation of the ActionType.

        Returns:
            str: The name of the enumeration member.
        """
        return str(self.name)


class QueueCommandFormat(object):
    """
    A container for representing a command for queue operations.

    This class encapsulates the details of a queue command, specifically the filename associated with the command and the
    action to be performed. The action is converted to a string when returning the dictionary representation.

    Attributes:
        filename (str): The name of the file associated with the command.
        action (Any): The action to be performed. Its string representation is used in the dictionary output.
    """

    def __init__(self, filename: str, action: Any) -> None:
        """
        Initialize a new instance of QueueCommandFormat.

        Args:
            filename (str): The name of the file associated with the command.
            action (Any): The action to be performed.
        """
        self.filename = filename
        self.action = action

    def asdict(self) -> Dict[str, str]:
        """
        Convert the instance's attributes to a dictionary.

        Returns:
            Dict[str, str]: A dictionary containing the filename and the string representation of the action.
        """
        return {"filename": self.filename, "action": str(self.action)}


class QueueResultFormat(object):
    """
    A container for storing and formatting the result of a queue operation.

    Attributes:
        filename (str): The name of the file associated with the result.
        result (Any): The result data produced by the operation.
        details (str): Additional details about the result (optional).
    """

    def __init__(self, filename: str, result: Any, details: str = "") -> None:
        """
        Initialize a new instance of QueueResultFormat.

        Args:
            filename (str): The name of the file associated with the result.
            result (Any): The result data produced by the queue operation.
            details (str, optional): Additional details about the result. Defaults to an empty string.
        """
        self.filename = filename
        self.result = result
        self.details = details

    def asdict(self) -> Dict[str, Any]:
        """
        Convert the instance data into a dictionary format.

        Returns:
            Dict[str, Any]: A dictionary containing the filename, result, and details.
        """
        return {"filename": self.filename, "result": self.result, "details": self.details}


class InterpTempsProcess(multiprocessing.Process):
    """
    Process for handling real-time fill state predictions.

    This multiprocess-based class continuously reads incoming data from a queue,
    processes it using a forecaster model, and sends completed prediction results
    to an output queue. It is designed to operate in parallel with the main
    application to provide non-blocking prediction updates.
    """

    def __init__(self,
                 queue_log: multiprocessing.Queue,
                 queue_cmd: multiprocessing.Queue,
                 queue_out: multiprocessing.Queue) -> None:
        """Initializes the InterpTempsProcess.

        Params:
            queue_log (multiprocessing.Queue): Queue used for logging messages.
            queue_cmd (multiprocessing.Queue): Queue for passing in files to be processed.
            queue_out (multiprocessing.Queue): Queue for passing results to caller.

        Command Format:
            Dict[str, Any] = {"filename": str, "action": str(ActionType)}

        Result Format:
            Dict[str, Any] = {"filename": str, "result": bool, "details": str}

        Returns:
            None
        """
        multiprocessing.Process.__init__(self)
        self._queueLog: multiprocessing.Queue = queue_log
        self._queueCmd: multiprocessing.Queue = queue_cmd
        self._queueOut: multiprocessing.Queue = queue_out
        self._started = multiprocessing.Event()
        self._exit = multiprocessing.Event()
        self._done = multiprocessing.Event()

        self._count: int = 0
        self._waiting: bool = False
        self._cache_reset()

    def _cache_reset(self) -> None:
        """Resets the persitient cache.

        Cache start is set to 0 and cache lists, `_cache_xp` and `_cache_fp` are also set to
        empty lists.

        Returns:
            None
        """
        self._cached_start: int = 0
        self._cached_xp: list = []
        self._cached_fp: list = []

    def _cache_set(self, t0: str, xp: np.ndarray, fp: np.ndarray) -> None:
        """
        Cache and preprocess coordinate data for interpolation.

        This internal method verifies that the x-coordinate values (`xp`) are strictly monotonically increasing
        and filters out any NaN values from the corresponding y-coordinate values (`fp`). These preprocessing
        steps ensure that the data meets the requirements for interpolation using `np.interp`.

        If `xp` is not strictly increasing, the method sorts both `xp` and `fp` accordingly. Similarly, if `fp`
        contains any NaN values, those values (and their corresponding `xp` values) are removed.

        The method converts the time string `t0` to seconds (using `self.time_string_to_seconds`) and caches the
        processed data in the instance attributes `_cached_start`, `_cached_xp`, and `_cached_fp`.

        Args:
            t0 (str): A time string representing the starting time to be converted to seconds.
            xp (np.ndarray): An array of x-coordinate values, which should be strictly increasing.
            fp (np.ndarray): An array of y-coordinate values corresponding to `xp`, which may include NaN values.

        Returns:
            None
        """
        # Before setting: confirm `xp` is monotonically increasing, and remove NANs
        # NOTE: These are requirements of what is passed to `np.interp`
        if not np.all(np.diff(xp) > 0):
            Log.w(TAG, "X-coordinate values are not strictly increasing. Sorting...")
            sort_ids = np.argsort(xp)
            xp = xp[sort_ids]
            fp = fp[sort_ids]
        real_ids = [x for x, y in enumerate(fp) if ~np.isnan(y)]
        if not len(real_ids) == len(fp):
            Log.w(TAG, "Y-coordinate values contain one or more NANs. Removing...")
            xp = xp[real_ids]
            fp = fp[real_ids]
        self._cached_start = self.time_string_to_seconds(t0)
        self._cached_xp = xp
        self._cached_fp = fp
        Log.d(TAG, f"Set cache with {len(xp)} entries")

    def _load(self, filename: str) -> None:
        """
        Load temperature data from a CSV file and update the internal cache.

        This method attempts to open a CSV file specified by `filename` and process its content to load
        temperature data. It performs the following steps.

        In case of any exception during the process, the method:
        - Resets the cache.
        - Logs the full traceback.
        - Puts an error message into `_queueOut` containing the exception details.

        Args:
            filename (str): The path to the CSV file containing temperature data.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file does not have the expected format or required data.
            Exception: Propagates any other exceptions after logging the traceback.

        """
        try:
            # Cache is blank if load fails.
            self._cache_reset()

            with open(filename, 'r') as f:
                # Read and validate header row.
                csv_headers = next(f, "").strip()
                if not csv_headers:
                    raise ValueError(
                        f"CSV file '{filename}' is missing header row.")

                # Read and validate first data row.
                csv_firstrow = next(f, "").strip()
                if not csv_firstrow:
                    raise ValueError(
                        f"CSV file '{filename}' is missing data rows.")

                # Extract start time from the first data row.
                parts = csv_firstrow.split(",")
                if len(parts) < 2:
                    raise ValueError(
                        f"CSV file '{filename}' first data row does not contain enough columns to extract start time.")
                time_start = parts[1].strip()
                if not time_start:
                    raise ValueError("Start time is empty.")

                if "Ambient" in csv_headers:
                    csv_cols: Tuple[int, int] = (2, 4)
                else:
                    csv_cols: Tuple[int, int] = (2, 3)
                # Read the rest of the file's lines.
                lines = f.readlines()
                if not lines:
                    raise ValueError(
                        f"CSV file '{filename}' does not contain any data rows after the header.")
                data = np.loadtxt(lines, delimiter=",",
                                  skiprows=0, usecols=csv_cols)
                relative_time = data[:, 0]
                temperature = data[:, 1]

                Log.d(TAG, f"Setting cache with {len(relative_time)} entries")
                self._cache_set(time_start, relative_time, temperature)

            self._queueOut.put(
                QueueResultFormat(
                    filename=filename,
                    result=True,
                    details="Loaded temperature values to cache."
                ).asdict()
            )

        except:
            # Capture and log the traceback in case of an exception.
            limit: Optional[int] = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list += format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

            self._queueOut.put(
                QueueResultFormat(
                    filename=filename,
                    result=False,
                    details=a_list[-1]
                ).asdict()
            )

    def _interp(self, filename: str) -> None:
        """Interpolates missing temperature data in a CSV file and updates the file.

        This method reads the CSV file specified by `filename`, calculates interpolated temperature
        values for rows with missing ('nan') temperatures, and writes the updated temperature data
        back to the file. The CSV file must have a header row and at least one data row. The second
        column of the first data row is used as the starting time for interpolation. The presence of the
        string "Ambient" in the header determines the column indices used for time and temperature data.

        Args:
            filename (str): The path to the CSV file to be processed.

        Raises:
            FileNotFoundError: If the file specified by `filename` does not exist.
            ValueError: If the file is missing expected rows or has an unexpected format.
            Exception: Propagates any other exceptions after logging the traceback.
        """
        try:
            # Initialize interplation array and utilties.
            lines = []
            interp_temps = []  # The array of interpolated temperatures.
            updated_rows = 0

            with open(filename, 'r') as f:
                # Read and validate the CSV file contains a header row.
                csv_headers: str = next(f, "").strip()
                if not csv_headers:
                    raise ValueError("CSV file is missing header row.")

                # Read and validate the CSV file contains a row after the header row.
                csv_firstrow = next(f).strip()
                if not csv_firstrow:
                    raise ValueError("CSV file is missing data rows.")

                # Read and validate that the first row contains at least 2 columns
                parts: List[str] = csv_firstrow.split(",")
                if len(parts) < 2:
                    raise ValueError(
                        "CSV file's first data row does not contain enough columns.")
                time_start = parts[1]

                # Reset file pointer and read lines at file head.
                f.seek(0)
                lines = f.readlines()

                if "Ambient" in csv_headers:
                    csv_cols: Tuple[int, int] = (2, 4)
                else:
                    csv_cols: Tuple[int, int] = (2, 3)

                data = np.loadtxt(
                    lines, delimiter=",", skiprows=1, usecols=csv_cols,
                )
                # Data is formated as (relative_time, temperature)
                relative_time = data[:, 0]
                # temperature = data[:, 1]

                # align relative times based on starting "time" value
                file_start = self.time_string_to_seconds(time_start)
                time_delta = file_start - self._cached_start
                Log.d(TAG, f"time_delta = {time_delta}s")
                Log.d(TAG, "Moved start relative time from " +
                      f"{relative_time[0]} to {relative_time[0]+time_delta} secs")
                relative_time += time_delta

                # Calculate interpolated temperatures for missing ('nan') temperature values,
                # then round to the nearest quarter.
                interp_temps = self.round_to_quarter(
                    np.interp(relative_time,
                              self._cached_xp, self._cached_fp))

                Log.d(
                    TAG, f"Temps({len(interp_temps)}) = [{interp_temps[0]}, ..., {interp_temps[-1]}]")
                Log.d(
                    TAG, f"np.min(Temps) = {np.min(interp_temps)}, np.max(Temps) = {np.max(interp_temps)}")
                Log.d(
                    TAG, f"Propagating temperatures for run \"{filename}\"...")

                # Process each line, replacing occurrences of "nan" with interpolated temperatures.
                for i, line in enumerate(lines):
                    if i == 0:
                        # Skip header row.
                        Log.d(TAG, "Skipping header row")
                    elif "nan" in line:
                        # Missing temperature row.
                        try:
                            replacement_temp = f"{interp_temps[i-1]:2.2f}"
                            lines[i] = line.replace("nan", replacement_temp)
                            updated_rows += 1
                        except IndexError as ie:
                            Log.e(
                                TAG, f"IndexError while processing line {i}: {ie}")
                            raise
                    else:
                        # Empty row, or non-nan entry.
                        Log.w(
                            TAG, f"Skipping row {i}: it does not contain 'nan' temp!")

            # NOTE: uncomment the line below to compare in/out files
            # filename = filename.replace(".csv", "_filled.csv")

            # Re-write the file with the updated temperature data.
            with open(filename, 'w') as w:
                w.writelines(lines)

            Log.d(TAG, f"Written to file: '{filename}'")
            self._count += 1

            # Sanity check: ensure the number of updated rows equals the number of interpolated values.
            if updated_rows != len(interp_temps):
                Log.e(
                    TAG, f"(FAIL): Not all temperatures were propagated in the file! ({updated_rows} != {len(interp_temps)})")
                self._queueOut.put(
                    QueueResultFormat(
                        filename=filename,
                        result=False,
                        details="Not all temperatures were propagated in the file."
                    ).asdict()
                )
            else:
                Log.d(
                    TAG, "(SUCCESS): Propagated all temperature values to file.")
                self._queueOut.put(
                    QueueResultFormat(
                        filename=filename,
                        result=True,
                        details="Successfully propagated temperature values to file."
                    ).asdict()
                )
        except:
            # Capture and log the traceback in case of an exception.
            limit: Optional[int] = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list += format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)
            self._queueOut.put(
                QueueResultFormat(
                    filename=filename,
                    result=False,
                    details=a_list[-1]
                ).asdict()
            )

    def time_string_to_seconds(self, time_string: str, format: str = "%H:%M:%S") -> float:
        """Converts a time string to seconds.

        Args:
            time_string (str): The time string to convert.
            format (str, optional): The format of the time string. Defaults to '%H:%M:%S'.

        Returns:
            float: The number of seconds represented by the time string.
        """
        time_object = datetime.strptime(time_string, format).time()
        seconds = (time_object.hour * 3600) + \
            (time_object.minute * 60) + time_object.second
        Log.d(TAG, f"Time \"{time_string}\" converted to " +
              f"{seconds} seconds (since midnight)")
        return seconds

    def round_to_quarter(self, floats: np.ndarray):
        """Rounds a list of floats to the nearest 0.25.

        Args:
            floats (np.ndarray): The list of floating point values to round.

        Returns:
            np.ndarray: An np.ndarray of rounded float values to the nearest quarter.
        """
        return np.round(np.array(floats) * 4) / 4

    def run(self) -> None:
        """Runs the process to process incoming commands to load and interp temperatures.

        This method expects the following steps:
          1. Redirects stdout and stderr to suppress console output.
          2. Configures the logger for the process and sets up the multiprocessing logger.
          3. Run any one-time action at start of process.
          4. Enters a loop that waits for incoming commands on the input queue:
             - It will process all queued commands, in the order they were received.
             - If the command is valid and non-empty, it processes the command accordingly.
             - An empty command will break the loop, acting as a sentinel to end the process.
          5. The expected workflow order of commands is as follows:
             - Load the primary file into cache.
             - Interp each of the secondaries from cache (in order)
          6. Handles any exceptions by logging a detailed traceback.
          7. Upon termination, logs the process shutdown and sets the done event.

        Raises:
            AttributeError: If required queue attributes (_queueCmd, _queueOut, _queueLog) are not set.
            Exception: If errors occur during stdout/stderr redirection, logger configuration, or command processing,
                    the exceptions are logged and re-raised.
        Returns:
            None
        """
        try:
            # Verify essential attributes exist
            if not hasattr(self, '_queueCmd') or not hasattr(self, '_queueOut') or not hasattr(self, '_queueLog'):
                raise AttributeError(
                    "Required queue attributes '_queueCmd` and '_queueLog' are not set in the InterpTemps object.")
            self._started.set()

            # Redirect stdout and stderr
            try:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
            except Exception as e:
                Log.e(TAG, f"Error redirecting stdout/stderr: {e}")
                raise

            if STANDALONE_MODE:
                logger = logging.getLogger()
            else:
                logger = logging.getLogger("QATCH.logger")

            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            # Configure the multiprocessing logger
            try:
                from multiprocessing.util import get_logger
                multiprocessing_logger = get_logger()
                if not multiprocessing_logger.handlers:
                    raise RuntimeError(
                        "No handlers found in multiprocessing logger.")
                multiprocessing_logger.handlers[0].setStream(sys.stderr)
                multiprocessing_logger.setLevel(logging.WARNING)
            except Exception as e:
                Log.e(TAG, f"Multiprocessing logger configuration failed: {e}")
                raise
            # Run one-time start actions (if any)
            # Place any one-time initialization code here.
            None

            while not self._exit.is_set():
                if not self._waiting:
                    Log.d(TAG, "Waiting for command in queue...")
                if self._queueCmd.empty():
                    self._waiting = True
                    continue
                self._waiting = False

                # Read in the next command from the queue
                command: Optional[Dict[str, Any]] = self._queueCmd.get()

                Log.d(TAG, f"Received CMD: {command}")

                # Process the new data if it exists and is not empty.
                if command is not None:
                    action = command.get("action", None)
                    filename = command.get("filename", None)
                    if action is None or filename is None:
                        Log.e(TAG, "Invalid command, missing required key(s)")
                        self._queueOut.put(
                            QueueResultFormat(
                                filename=filename,
                                result=False,
                                details="Invalid command, missing required key(s)."
                            ).asdict()
                        )
                    elif action == str(ActionType.load):
                        self._load(filename)
                    elif action == str(ActionType.interp):
                        self._interp(filename)
                else:
                    if self._count:
                        Log.i(
                            TAG, f"Propagated temps to {self._count} secondary files (DONE)")
                    else:
                        Log.i(TAG, "No run files modified (DONE)")
                    self.stop()

        except Exception:
            # Capture and log the traceback in case of an exception.
            limit: Optional[int] = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list += format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            # Gracefully end the subprocess and log the shutdown.
            Log.d(TAG, "InterpTempsProcess stopped.")
            self._done.set()

    def is_running(self) -> bool:
        """Checks if the process is still running.

        Returns:
            bool: True if the process is active, False if it has completed or stopped.
        """
        return self._started.is_set() and not self._done.is_set()

    def stop(self) -> None:
        """Signals the process to stop.

        Sets the exit event, which causes the main loop in run() to terminate gracefully.

        Returns:
            None
        """
        self._exit.set()


class InterpTempsTest:

    def __init__(self):
        q_listener, q = self.logger_init()
        # set up logger (only once)
        multiprocessing.log_to_stderr()
        logger = multiprocessing.get_logger()
        logger.setLevel(logging.INFO)
        self._queueLog = q
        self._queueCmd = multiprocessing.Queue()
        self._queueOut = multiprocessing.Queue()
        self._test_process = InterpTempsProcess(
            self._queueLog, self._queueCmd, self._queueOut)

    def run(self):
        # start test process
        Log.i(TAG, "Starting test process...")
        self._test_process.start()
        while not self._test_process.is_running():
            continue  # wait for process to start
        # load test commands
        Log.i(TAG, "Queueing test commands...")
        self._test_process._queueCmd.put(
            QueueCommandFormat(
                r"C:\Users\Alexander J. Ross\Documents\QATCH nanovisQ\logged_data\1_13820290\D250224W3_4CP_B_1_1\capture\D250224W3_4CP_B_1_3rd.csv",
                ActionType.load
            ).asdict()
        )
        self._test_process._queueCmd.put(
            QueueCommandFormat(
                r"C:\Users\Alexander J. Ross\Documents\QATCH nanovisQ\logged_data\2_15548760\D250224W3_4CP_B_1_2\capture\D250224W3_4CP_B_1_3rd.csv",
                ActionType.interp
            ).asdict()
        )
        self._test_process._queueCmd.put(
            QueueCommandFormat(
                r"C:\Users\Alexander J. Ross\Documents\QATCH nanovisQ\logged_data\3_15519000\D250224W3_4CP_B_1_3\capture\D250224W3_4CP_B_1_3rd.csv",
                ActionType.interp
            ).asdict()
        )
        self._test_process._queueCmd.put(
            QueueCommandFormat(
                r"C:\Users\Alexander J. Ross\Documents\QATCH nanovisQ\logged_data\4_15549230\D250224W3_4CP_B_1_4\capture\D250224W3_4CP_B_1_3rd.csv",
                ActionType.interp
            ).asdict()
        )
        # signal finished, ok to end process
        self._test_process._queueCmd.put(None)
        # wait for test process, handle any logging
        Log.i(TAG, "Waiting for test process to finish...")
        while self._test_process.is_running():
            if not self._test_process._queueOut.empty():
                result = self._test_process._queueOut.get()
                Log.i(
                    f"Got Result: '{result['result']}' for filename '{result['filename']}'")
                if len(result['details']):
                    Log.i(f"Result Details: {result['details']}")
        self._test_process.join()

    def logger_init(self):
        q = multiprocessing.Queue()
        # this is the handler for all log records
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(levelname)s: %(asctime)s - %(process)s - %(message)s"))
        # ql gets records from the queue and sends them to the handler
        ql = QueueListener(q, handler)
        ql.start()
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # add the handler to the logger so records from this process are handled
        logger.addHandler(handler)
        return ql, q


if __name__ == '__main__':
    InterpTempsTest().run()
