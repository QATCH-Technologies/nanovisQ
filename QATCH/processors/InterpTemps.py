"""
InterpTemps processes pre-compressed raw run data files by searching for "nan"
values in the Temperature data values, replacing them with interpolated real values.

This module provides a multiprocessor process that handles non-blocking real-time
processing to allow this activity to occur in the background while the user is
entering manual inputs to the post-run Run Info dialog window.

The Run Info "Save" action must block the `confirm()` action until this process
has been marked as finished; otherwise, runs may be compressed partially processed.
"""

import os
import sys
import logging
import multiprocessing
import numpy as np
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Optional, Dict
from enum import Enum
from datetime import datetime

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
    load = 0  # load real temperatures from file in command for later interp
    interp = 1  # interp nan temperatures from file in command using prior load
    # NOTE: use `flag_missing` in command dictionary to flag interp values as missing (red in output table)

    def __str__(self):
        return str(self.name)


class QueueCommandFormat(object):

    def __init__(self, filename, action) -> None:
        self.filename = filename
        self.action = action

    def asdict(self) -> Dict[str, str]:
        return {"filename": self.filename, "action": str(self.action)}


class QueueResultFormat(object):

    def __init__(self, filename, result, details="") -> None:
        self.filename = filename
        self.result = result
        self.details = details

    def asdict(self) -> Dict[str, Any]:
        return {"filename": self.filename, "result": self.result, "details": self.details}


class InterpTempsProcess(multiprocessing.Process):
    """Process for handling real-time fill state predictions.

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

        self._count = 0
        self._waiting = False
        self._cache_reset()

    def _cache_reset(self):
        self._cached_start = 0
        self._cached_xp = []
        self._cached_fp = []

    def _cache_set(self, t0, xp, fp):
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
        try:
            self._cache_reset()  # blank, if load fails

            with open(filename, 'r') as f:
                csv_headers = next(f)

                csv_firstrow = next(f)
                time_start = csv_firstrow.split(",")[1]

                if "Ambient" in csv_headers:
                    csv_cols = (2, 4)
                else:
                    csv_cols = (2, 3)

                data = np.loadtxt(
                    f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols,
                )
                relative_time = data[:, 0]
                temperature = data[:, 1]

                # Log.d(TAG, f"time_start = {time_start}")
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
        try:
            lines = []
            interp_temps = []
            updated_rows = 0

            with open(filename, 'r') as f:
                csv_headers = next(f)

                csv_firstrow = next(f)
                time_start = csv_firstrow.split(",")[1]
                f.seek(0)

                lines = f.readlines()

                if "Ambient" in csv_headers:
                    csv_cols = (2, 4)
                else:
                    csv_cols = (2, 3)

                data = np.loadtxt(
                    lines, delimiter=",", skiprows=1, usecols=csv_cols,
                )
                relative_time = data[:, 0]
                temperature = data[:, 1]

                # align relative times based on starting "time" value
                file_start = self.time_string_to_seconds(time_start)
                time_delta = file_start - self._cached_start
                Log.d(TAG, f"time_delta = {time_delta}s")
                Log.d(TAG, "Moved start relative time from " +
                      f"{relative_time[0]} to {relative_time[0]+time_delta} secs")
                relative_time += time_delta

                # calculate interpolated temperaures for NAN temps
                interp_temps = self.round_to_quarter(
                    np.interp(relative_time,
                              self._cached_xp, self._cached_fp))
                Log.d(TAG, f"Temps({len(interp_temps)}) = " +
                      f"[{interp_temps[0]}, ..., {interp_temps[-1]}]")
                Log.d(TAG, f"np.min(Temps) = {np.min(interp_temps)}, " +
                      f"np.max(Temps) = {np.max(interp_temps)}")

                Log.d(TAG, "Propagating temperatures" +
                      f" for run \"{filename}\"...")
                for i, line in enumerate(lines):
                    if i == 0:  # header row
                        Log.d(TAG, "Skipping header row")
                    elif "nan" in line:  # missing temp row
                        lines[i] = line.replace("nan",
                                                f"{interp_temps[i-1]:2.2f}")
                        updated_rows += 1
                    else:  # empty row, or non-nan entry
                        Log.w(TAG, f"Skipping row {i}: " +
                              "it does not contain 'nan' temp!")

            # NOTE: uncomment the line below to compare in/out files
            # filename = filename.replace(".csv", "_filled.csv")

            # re-write the raw data file with the filled temperature data
            with open(filename, 'w') as w:
                w.writelines(lines)

            Log.d(TAG, f"Written to file: '{filename}'")
            self._count += 1

            # sanity check: all temperature values should have been "nan"
            if updated_rows != len(interp_temps):
                Log.e(TAG, "FAIL: Not all temperatures were propagated in the file!" +
                      f" ({updated_rows} != {len(interp_temps)})")

                self._queueOut.put(
                    QueueResultFormat(
                        filename=filename,
                        result=False,
                        details="Not all temperatures were propagated in the file."
                    ).asdict()
                )
            else:
                Log.d(
                    TAG, "Propagated all temperature values to file (SUCCESS)")

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

    def time_string_to_seconds(self, time_string, format="%H:%M:%S"):
        """Converts a time string to seconds.

        Args:
            time_string: The time string to convert.
            format: The format of the time string.

        Returns:
            The number of seconds represented by the time string.
        """
        time_object = datetime.strptime(time_string, format).time()
        seconds = (time_object.hour * 3600) + \
            (time_object.minute * 60) + time_object.second
        Log.d(TAG, f"Time \"{time_string}\" converted to " +
              f"{seconds} seconds (since midnight)")
        return seconds

    def round_to_quarter(self, floats):
        """Rounds a list of floats to the nearest 0.25."""
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
        """
        try:
            self._started.set()

            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            if STANDALONE_MODE:
                logger = logging.getLogger()
            else:
                logger = logging.getLogger("QATCH.logger")

            logger.addHandler(QueueHandler(self._queueLog))
            logger.setLevel(logging.DEBUG)

            from multiprocessing.util import get_logger
            multiprocessing_logger = get_logger()
            multiprocessing_logger.handlers[0].setStream(sys.stderr)
            multiprocessing_logger.setLevel(logging.WARNING)

            # Run once actions, on start of process:
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
