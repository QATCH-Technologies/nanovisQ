"""File storage utilities for saving and managing measurement data.

Provides :class:`FileStorage` for buffered CSV/TXT export of sensor data and
:class:`SecureOpen` for transparent read/write access to plain files or
records inside AES-encrypted ZIP archives.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)
    Other QATCH Technologies contributors

Date:
    2026-03-20
"""

import csv
import datetime
import hashlib
import json
import os
import sys
from time import localtime, strftime
from traceback import format_tb
from typing import Optional

import numpy as np
import pyzipper

from QATCH.common.fileManager import FileManager
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants

TAG = "[FileStorage]"


class FileStorage:
    """Stores and exports measurement data to CSV/TXT files.

    Manages an internal row buffer (BUFFERED_ROWS) and writes to disk with
    optional downsampling after a configurable time threshold.
    """

    # Global buffer and index helpers
    BUFFERED_ROWS: list[list] = []
    HANDLE = 0
    TOPATH = 1
    BUFFER = 2
    DOWN_SAMPLING = False

    @staticmethod
    def csv_save(
        i: int,
        filename: str,
        path: str,
        data_save0: float,
        data_save1: float,
        data_save2: float,
        data_save3: float,
        data_save4: float,
        data_save5: float,
        writeToFilesystem: bool = True,
    ):
        """Buffer one row of processed sensor data and optionally flush it to a CSV file.

        Args:
            i (int): Device index.
            filename (str): Name for the output file.
            path (str): Directory path for the output file.
            data_save0 (float): Relative time value.
            data_save1 (float): Temperature reading.
            data_save2 (float): Raw peak magnitude.
            data_save3 (float): Resonance frequency.
            data_save4 (float): Dissipation value.
            data_save5 (float): Ambient reading.
            writeToFilesystem (bool): If True, flush the buffer to disk immediately.
        """
        HANDLE = FileStorage.HANDLE
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER

        fix1 = "%Y-%m-%d"
        fix2 = "%H:%M:%S"
        csv_time_prefix1 = strftime(fix1, localtime())
        csv_time_prefix2 = strftime(fix2, localtime())
        d0 = float("{0:.4f}".format(data_save0))
        d1 = float("{0:.2f}".format(data_save1))
        d3 = float("{0:.2f}".format(data_save3))
        d4 = float("{:.15e}".format(data_save4))
        d5 = float("{0:.2f}".format(data_save5))

        # Append device name folder to path
        path = os.path.join(path, FileStorage.dev_get_active(i))
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Find index in buffered row data from file handle (create index if new)
        full_path = FileManager.create_full_path(
            filename, extension=Constants.csv_extension, path=path
        )
        fHashKey = hashlib.sha1(full_path.encode("utf-8")).hexdigest()
        fHandle = len(FileStorage.BUFFERED_ROWS)
        for x in range(fHandle):
            if (
                FileStorage.BUFFERED_ROWS[x][HANDLE] == fHashKey
                and FileStorage.BUFFERED_ROWS[x][TOPATH] == full_path
            ):
                fHandle = x
                break
        if fHandle == len(FileStorage.BUFFERED_ROWS):
            FileStorage.BUFFERED_ROWS.append([fHashKey, full_path, []])

        FileStorage.BUFFERED_ROWS[fHandle][BUFFER].append(
            [csv_time_prefix1, csv_time_prefix2, d0, d5, d1, data_save2, d3, d4]
        )

        if writeToFilesystem:
            FileStorage._correct_for_dup_times(fHandle)
            FileStorage._write_buffered_data(fHandle)

    @staticmethod
    def csv_flush_all():
        """Flush all pending buffered rows to their respective CSV files.

        Called at the end of the SWEEPs loop in Serial.py.
        """
        HANDLE = FileStorage.HANDLE
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER

        # This function (called at end of SWEEPs loop in Serial.py) flushes buffers to filesystem!
        fHandle = len(FileStorage.BUFFERED_ROWS)
        for x in range(fHandle):
            full_path = FileStorage.BUFFERED_ROWS[x][TOPATH]
            fHashKey = hashlib.sha1(full_path.encode("utf-8")).hexdigest()
            if (
                FileStorage.BUFFERED_ROWS[x][HANDLE] == fHashKey
                and FileStorage.BUFFERED_ROWS[x][BUFFER] != []
            ):
                FileStorage._correct_for_dup_times(x)
                FileStorage._write_buffered_data(x)

    @staticmethod
    def _correct_for_dup_times(handle: int):
        """Correct duplicate relative-time values in the buffer for a given handle.

        Duplicate times are replaced by evenly spaced values computed with
        ``numpy.linspace``.

        Args:
            handle (int): Index into BUFFERED_ROWS identifying the target buffer.
        """
        BUFFER = FileStorage.BUFFER
        PRECISION = ".0001"

        # Check for duplicate times, and (if any) evenly space times
        FileStorage.BUFFERED_ROWS[handle][BUFFER] = np.array(
            FileStorage.BUFFERED_ROWS[handle][BUFFER]
        )
        bufferTimes = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 2]
        if FileStorage._check_if_duplicates(bufferTimes):
            bufferTimes = np.linspace(
                float(bufferTimes[0]), float(bufferTimes[-1]) - float(PRECISION), len(bufferTimes)
            )
            bufferTimes = ["{0:.{1}f}".format(x, len(PRECISION) - 1) for x in bufferTimes]
            FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 2] = bufferTimes

    @staticmethod
    def _check_if_duplicates(data: list) -> bool:
        """Return True if the given list contains any duplicate elements.

        Args:
            data (list): The sequence to inspect.

        Returns:
            bool: True if at least one duplicate element is present, False otherwise.
        """
        return len(data) != len(set(data))

    @staticmethod
    def _write_buffered_data(handle: int):
        """Write buffered rows from the internal buffer to the CSV file for the given handle.

        Creates the file and its header row if they do not yet exist. Applies
        downsampling when the relative time in the buffer exceeds
        ``Constants.downsample_after``.

        Args:
            handle (int): Index into BUFFERED_ROWS identifying the target buffer.
        """
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER

        full_path = FileStorage.BUFFERED_ROWS[handle][TOPATH]

        # Creates a directory if the specified path doesn't exist
        # FileManager.create_dir(path)
        # Creates a file full path based on parameters
        if not FileManager.file_exists(full_path):
            Log.i(TAG, "Exporting data to CSV file...")
            Log.i(TAG, "Storing in: {}".format(full_path))

            with open(Constants.new_files_path, "a") as tempFile:
                tempFile.write(full_path + "\n")

            FileStorage.DOWN_SAMPLING = False

        # checks the path for the header insertion
        if os.path.exists(full_path):
            header_exists = True
        else:
            header_exists = False

        # opens the file to write data
        with open(full_path, "a", newline="") as tempFile:
            tempFileWriter = csv.writer(tempFile)
            # inserts the header if it doesn't exist
            if not header_exists:
                tempFileWriter.writerow(
                    [
                        "Date",
                        "Time",
                        "Relative_time",
                        "Ambient",
                        "Temperature",
                        "Peak Magnitude (RAW)",
                        "Resonance_Frequency",
                        "Dissipation",
                    ]
                )

            # peak into buffer times, downsample if greater than 90 seconds in
            bufferTimes = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 2]

            if float(bufferTimes[-1]) < Constants.downsample_after:  # do not downsample
                FileStorage.DOWN_SAMPLING = False
                # write the buffered data
                tempFileWriter.writerows(FileStorage.BUFFERED_ROWS[handle][BUFFER])
                FileStorage.BUFFERED_ROWS[handle][BUFFER] = []  # empty buffer
            # on the edge of downsampling
            elif (
                float(bufferTimes[0]) < Constants.downsample_after
                and float(bufferTimes[-1]) > Constants.downsample_after
            ):
                Log.d(TAG, "Starting to downsample with mixed buffer (on the edge)...")
                for i in range(len(bufferTimes)):
                    if float(bufferTimes[i]) > Constants.downsample_after:
                        # write the non-downsampled buffered data
                        tempFileWriter.writerows(FileStorage.BUFFERED_ROWS[handle][BUFFER][:i])
                        FileStorage.BUFFERED_ROWS[handle][BUFFER] = FileStorage.BUFFERED_ROWS[
                            handle
                        ][BUFFER][i:]
                        # convert partial buffer (now of type 'numpy.ndarray') back to 'list'
                        FileStorage.BUFFERED_ROWS[handle][BUFFER] = FileStorage.BUFFERED_ROWS[
                            handle
                        ][BUFFER].tolist()
                        break
            else:  # downsample
                if not FileStorage.DOWN_SAMPLING:
                    Log.w(
                        TAG,
                        f"Downsampling started after {Constants.downsample_after}"
                        " seconds of measurement capture...",
                    )
                    Log.w(
                        TAG,
                        "Each sample written to file is now an average of "
                        f"{Constants.downsample_file_count} raw data points.",
                    )
                    Log.w(
                        TAG,
                        "Each sample written to plot is now just 1 in"
                        f" every {Constants.downsample_plot_count} raw data points.",
                    )
                    FileStorage.DOWN_SAMPLING = True

                # len(bufferTimes) >= Constants.downsample_file_count:
                if True:
                    mid_interval = int(len(bufferTimes) / 2)
                    date_mid = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 0][mid_interval]
                    time_mid = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 1][mid_interval]

                    ambient = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 3]
                    temperature = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 4]
                    peak_mag = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 5]
                    res_freq = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 6]
                    diss = FileStorage.BUFFERED_ROWS[handle][BUFFER][:, 7]

                    relative_avg = "{0:.4f}".format(
                        float(np.average([float(x) for x in bufferTimes]))
                    )
                    ambient_avg = "{0:.2f}".format(float(np.average([float(x) for x in ambient])))
                    temp_avg = "{0:.2f}".format(float(np.average([float(x) for x in temperature])))
                    peak_avg = "{0:.0f}".format(float(np.average([float(x) for x in peak_mag])))
                    res_avg = "{0:.0f}".format(float(np.average([float(x) for x in res_freq])))
                    diss_avg = "{:.15e}".format(float(np.average([float(x) for x in diss])))

                    downsampled_buffer = [
                        date_mid,
                        time_mid,
                        relative_avg,
                        ambient_avg,
                        temp_avg,
                        peak_avg,
                        res_avg,
                        diss_avg,
                    ]
                    tempFileWriter.writerow(downsampled_buffer)
                    # empty buffer
                    FileStorage.BUFFERED_ROWS[handle][BUFFER] = []

            tempFile.close()

    @staticmethod
    def csv_sweeps_save(
        i,
        filename: str,
        path: str,
        data_save1: float,
        data_save2: float,
        data_save3: Optional[float] = None,
    ):
        """Save a single sweep's frequency/amplitude/phase data as a CSV file.

        Args:
            i: Device index.
            filename (str): Name for the output file.
            path (str): Directory path for the output file.
            data_save1: Array-like of frequency values.
            data_save2: Array-like of amplitude values.
            data_save3: Array-like of phase values, or None to omit phase.
        """
        # Append device name folder to path
        path = os.path.join(path, FileStorage.dev_get_active(i))
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Creates a file full path based on parameters
        full_path = FileManager.create_full_path(
            filename, extension=Constants.csv_extension, path=path
        )
        # creates CSV file
        if data_save3 is None:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2]), delimiter=",")
        else:
            np.savetxt(
                full_path, np.column_stack([data_save1, data_save2, data_save3]), delimiter=","
            )

    @staticmethod
    def txt_sweeps_save(
        i,
        filename: str,
        path: str,
        data_save1,
        data_save2,
        data_save3=None,
        appendNameToPath: bool = True,
    ):
        """Save a single sweep's frequency/amplitude/phase data as a whitespace-delimited TXT file.

        Args:
            i: Device index.
            filename (str): Name for the output file.
            path (str): Directory path for the output file.
            data_save1: Array-like of frequency values.
            data_save2: Array-like of amplitude values.
            data_save3: Array-like of phase values, or None to omit phase.
            appendNameToPath (bool): If True, appends the active device name subfolder to path.
        """
        if appendNameToPath:
            # Append device name folder to path
            path = os.path.join(path, FileStorage.dev_get_active(i))
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Creates a file full path based on parameters
        full_path = FileManager.create_full_path(
            filename, extension=Constants.txt_extension, path=path
        )
        # creates TXT file
        if data_save3 is None:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2]))
        else:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2, data_save3]))

    @staticmethod
    def dev_info_get(i: int, dev_name: str) -> dict:
        """Read device info fields from a TXT file for the given device.

        Args:
            i (int): Device index.
            dev_name (str): Device name used to locate the info file.

        Returns:
            dict: Mapping of label to value pairs, or an empty dict if the file is
            missing, unreadable, or belongs to a different device.
        """
        try:
            # Append device name folder to path (using provided parameter 'dev_name')
            dev_folder = "{}_{}".format(i, dev_name) if i > 0 else dev_name
            path = os.path.join(Constants.csv_calibration_export_path, dev_folder)
            filename = Constants.txt_device_info_filename
            # Creates a file full path based on parameters
            full_path = FileManager.create_full_path(
                filename, extension=Constants.txt_extension, path=path
            )
            # Read in device info from file (if exists)
            if FileManager.file_exists(full_path):
                with open(full_path, "r") as fh:
                    lines = [line.split(": ") for line in fh.read().split("\n") if line]
                    # Parse labels and values from file
                    data = list(zip(*lines, strict=False))
                    labels = data[0]
                    values = data[1]
                    # Generate and return dict of keys and values
                    info = dict(zip(labels, values, strict=False))
                    if info["USB"] == dev_name:
                        return info
                    Log.w(TAG, "Device info for {} is for another device!".format(dev_name))
            else:
                Log.w(TAG, "Device info for {} does not exist.".format(dev_name))
        except Exception:
            Log.w(TAG, "Could not parse device info for {}.".format(dev_name))
        return {}

    @staticmethod
    def _build_device_info_labels(
        port: Optional[str],
        ip: Optional[str],
        uid: Optional[str],
        mac: Optional[str],
        usb: Optional[str],
        pid: Optional[str],
        rev: Optional[str],
        err: Optional[str],
    ) -> list:
        """Build the ordered list of field labels for a device info file.

        Always starts with ["NAME", "FW", "HW"] and appends optional field
        labels for each argument that is not None.

        Args:
            port (Optional[str]): Serial port value.
            ip (Optional[str]): IP address value.
            uid (Optional[str]): UID value.
            mac (Optional[str]): MAC address value.
            usb (Optional[str]): USB identifier value.
            pid (Optional[str]): Product ID value.
            rev (Optional[str]): Revision value.
            err (Optional[str]): Error value.

        Returns:
            list[str]: Ordered list of field label strings.
        """
        labels = ["NAME", "FW", "HW"]
        if port is not None:
            labels.append("PORT")
        if ip is not None:
            labels.append("IP")
        if uid is not None:
            labels.append("UID")
        if mac is not None:
            labels.append("MAC")
        if usb is not None:
            labels.append("USB")
        if pid is not None:
            labels.append("PID")
        if rev is not None:
            labels.append("REV")
        if err is not None:
            labels.append("ERR")
        return labels

    @staticmethod
    def _parse_existing_device_info(
        full_path: str, newVals: list, newLen: int, port: Optional[str]
    ) -> tuple[bool, list, int]:
        """Read an existing device info file and determine whether it needs rewriting.

        Preserves a custom NAME value and restores a missing PORT entry if needed.

        Args:
            full_path (str): Absolute path to the existing device info file.
            newVals (list): List of new values to compare against the file.
            newLen (int): Length of newVals.
            port (Optional[str]): Current port value, used to restore a missing PORT field.

        Returns:
            tuple[bool, list, int]: A 3-tuple of (writeFile, updated newVals, updated newLen)
            where writeFile is True when the file must be rewritten.
        """
        writeFile = False
        try:
            with open(full_path, "r") as fh:
                lines = [line.split(": ") for line in fh.read().split("\n") if line]
            # Generate list and length of old values (no dups)
            oldData = list(zip(*lines, strict=False))
            oldLabels = oldData[0]
            oldVals = oldData[1]
            oldLen = len(oldVals)
            # Do not change device name if custom name value is set in info file
            # NAME must be first line of info file
            if oldLabels[0] == "NAME" and newVals[0] != oldVals[0]:
                # we have a custom "NAME" - do not change name stored in info file
                newVals[0] = oldVals[0]
            # Do not blow away COM port from config file if device is purely
            # IP/NET configured
            if port is None and oldLabels[3] == "PORT":
                port = oldVals[3]
                newVals.insert(3, port)
                newLen += 1
            # Determine whether or not to update the file
            if oldLen != newLen:
                writeFile = True
            else:
                for val in oldVals:
                    if val not in newVals:
                        writeFile = True
                        break
        except Exception:
            Log.w(TAG, "Could not parse existing device info file. Replacing it now.")
            writeFile = True
        return writeFile, newVals, newLen

    @staticmethod
    def dev_info_set(
        i: int,
        filename: str,
        path: str,
        dev_name: str,
        fw: str,
        hw: str,
        port: Optional[str] = None,
        ip: Optional[str] = None,
        uid: Optional[str] = None,
        mac: Optional[str] = None,
        usb: Optional[str] = None,
        pid: Optional[str] = None,
        rev: Optional[str] = None,
        err: Optional[str] = None,
    ) -> bool:
        """Create or update the device info TXT file for the given device.

        Only writes the file when at least one value has changed. Updates the
        active device pointer as a side effect.

        Args:
            i (int): Device index.
            filename (str): Name for the output file.
            path (str): Directory path for the output file.
            dev_name (str): Device name.
            fw (str): Firmware version string.
            hw (str): Hardware version string.
            port (Optional[str]): Serial port identifier.
            ip (Optional[str]): IP address.
            uid (Optional[str]): Unique device identifier.
            mac (Optional[str]): MAC address.
            usb (Optional[str]): USB identifier.
            pid (Optional[str]): Product ID.
            rev (Optional[str]): Revision string.
            err (Optional[str]): Error string.

        Returns:
            bool: True if the file was written, False if no changes were detected.
        """
        # Append device name folder to path (using provided parameter 'dev_name')
        dev_folder = "{}_{}".format(i, dev_name) if i > 0 else dev_name
        path = os.path.join(path, dev_folder)
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Update pointer to active device name
        FileStorage.dev_set_active(i, dev_name)
        # Creates a file full path based on parameters
        full_path = FileManager.create_full_path(
            filename, extension=Constants.txt_extension, path=path
        )
        # Declare vars and put new values in a list
        newVals = [dev_name, fw, hw, port, ip, uid, mac, usb, pid, rev, err]
        while None in newVals:
            newVals.remove(None)
        newLen = len(newVals)
        # Read in device info from file (if exists)
        if FileManager.file_exists(full_path):
            writeFile, newVals, newLen = FileStorage._parse_existing_device_info(
                full_path, newVals, newLen, port
            )
            # if there are no changes, writeFile will still be False
        else:
            # File does not exist, create it now
            writeFile = True
        # Create or update TXT file
        if writeFile:
            Log.i(TAG, "Writing device info file for {}...".format(dev_name))
            labels = FileStorage._build_device_info_labels(port, ip, uid, mac, usb, pid, rev, err)
            with open(full_path, "w") as fh:
                if len(labels) != newLen:
                    Log.w(TAG, "Device info file labels are misaligned!")
                for i in range(newLen):
                    fh.write(labels[i] + ": " + newVals[i] + "\n")
            # Nothing exploded, so we must be OK
            Log.i(TAG, "Device info written!")
        return writeFile

    @staticmethod
    def dev_get_active(i: int) -> str:
        """Return the active device folder name for device index i.

        Reads the active-device index file to determine the current device folder.

        Args:
            i (int): Device index.

        Returns:
            str: Active device folder name, or an empty string on failure.
        """
        try:
            path = Constants.csv_calibration_export_path
            filename = Constants.txt_active_device_filename
            full_path = FileManager.create_full_path(
                filename, extension=Constants.txt_extension, path=path
            )
            if os.path.isdir(path):
                with open(full_path, "r") as fh:
                    _active_device_folders = fh.readlines()
                    idx = max(0, i - 1)
                    if idx < len(_active_device_folders):
                        _active_device_folder = _active_device_folders[idx].strip()
                        if len(_active_device_folder) == 0:
                            raise ValueError(
                                f"Device with PID {i} contains no value in the active list."
                            )
                    else:
                        raise IndexError(f"Device with PID {i} is not in the active list.")
                    _dev_path = os.path.join(path, _active_device_folder)
                    if os.path.isdir(_dev_path):
                        return _active_device_folder
                    else:
                        Log.w(TAG, "Active device does not yet exist.")
                        return _active_device_folder
            else:
                # The config folder does not yet exist (first run)
                pass  # ignore
        except ValueError as e:
            Log.e(TAG, str(e))
            Log.w(TAG, "Failed to get active device name.")
        except IndexError as e:
            Log.e(TAG, str(e))
            Log.w(TAG, "Failed to get active device name.")
        except Exception as e:
            Log.e(TAG, str(e))
            Log.w(TAG, "Failed to get active device name.")

            limit = None
            t, v, tb = sys.exc_info()
            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__ if t is not None else 'Exception'}: {str(v)}")
            for line in a_list:
                Log.e(TAG, line)

        return ""  # default, use root

    @staticmethod
    def dev_set_active(i: int, dev_name: str) -> None:
        """Write or update the active device name for device index i.

        Creates the config directory and active-device index file if they do
        not already exist.

        Args:
            i (int): Device index.
            dev_name (str): Device name to record as active.
        """
        try:
            path = Constants.csv_calibration_export_path
            filename = Constants.txt_active_device_filename
            full_path = FileManager.create_full_path(
                filename, extension=Constants.txt_extension, path=path
            )
            # Creates a directory if the specified path doesn't exist
            FileManager.create_dir(path)
            mode = "w" if i in [None, 0, 1] else "r+"
            with open(full_path, mode) as fh:
                if i is not None and i > 0:
                    f_lines = fh.readlines() if mode == "r+" else []
                    num_devs = len(f_lines) if mode == "r+" else 0
                    if i > num_devs or len(f_lines[i - 1].strip()) == 0:
                        while num_devs < i:
                            f_lines.append("\n")
                            num_devs += 1
                        f_lines[i - 1] = "{}_{}\n".format(i, dev_name)
                    else:
                        Log.d(
                            TAG,
                            f"Device '{i}_{dev_name}' is already in active list."
                            "Ignoring duplicate add request.",
                        )
                    # replace entire file with 'f_lines' contents, but only after reading
                    # it with 'readlines()' earlier
                    fh.seek(0)
                    fh.writelines(f_lines)
                    fh.truncate()
                else:
                    fh.write(dev_name + "\n")
        except Exception:
            Log.w(TAG, "Failed to set active device name.")

    @staticmethod
    def dev_write_default_preferences(save_path: str) -> None:
        """Write the application's default preferences dict to a JSON file.

        Args:
            save_path (str): Absolute path of the JSON file to write.
        """
        with open(save_path, "w") as f:
            json.dump(Constants.default_preferences, f, indent=4)

    @staticmethod
    def dev_write_preferences(save_path: str, preferences: dict) -> None:
        """Write a preferences dict to a JSON file.

        Args:
            save_path (str): Absolute path of the JSON file to write.
            preferences (dict): Preferences data to serialize.
        """
        with open(save_path, "w") as f:
            json.dump(preferences, f, indent=4)

    @staticmethod
    def dev_load_preferences(load_path: str) -> dict | None:
        """Load and return a preferences dict from a JSON file.

        Args:
            load_path (str): Absolute path of the JSON file to read.

        Returns:
            dict | None: Parsed preferences dict, or None if the file is missing,
            not valid JSON, or any other error occurs.
        """
        try:
            with open(load_path, "r") as file:
                preferences = json.load(file)
                return preferences
        except FileNotFoundError:
            Log.e(TAG, f"The file {load_path} was not found.")
        except json.JSONDecodeError:
            Log.e(TAG, f"The file {load_path} is not a valid JSON.")
        except Exception as e:
            Log.e(TAG, f"An unexpected error occurred: {e}")
        return None

    @staticmethod
    def dev_populate_path(path: str, i: int) -> str:
        """Replace the TBD active-device placeholder in path with the actual device folder.

        Args:
            path (str): Path string containing the TBD placeholder.
            i (int): Device index used to look up the active device name.

        Returns:
            str: Path with the placeholder substituted by the active device folder name.
        """
        return path.replace(Constants.tbd_active_device_name_path, FileStorage.dev_get_active(i))

    @staticmethod
    def dev_get_device_list() -> list:
        """Return a list of [index, uid] pairs for every device config folder.

        Scans the calibration export path for device folders. When the same UID
        appears more than once, only the most recently modified folder is kept.

        Returns:
            list: List of [index, uid] pairs, or an empty list on error.
        """
        try:
            path = Constants.csv_calibration_export_path
            devs = [
                device for device in os.listdir(path) if os.path.isdir(os.path.join(path, device))
            ]
            dev_list: list[list] = []
            modified: dict[str, float] = {}
            for d in devs:
                uids = [i[1] for i in dev_list]
                dev_file = os.path.join(
                    path, d, f"{Constants.txt_device_info_filename}.{Constants.txt_extension}"
                )
                if os.path.exists(dev_file):
                    mtime = os.path.getmtime(dev_file)
                else:
                    mtime = -1
                if d.find("_") > 0:
                    s = d.split("_")
                    i = int(s[0], base=16) % 9
                    d = s[1]
                else:
                    i = 0
                if d in uids:
                    if mtime < modified[d]:
                        Log.d(
                            TAG,
                            (
                                "Skipping config folder as older duplicate:" f"{i}_{d}"
                                if i != 0
                                else d
                            ),
                        )
                        Log.d(
                            TAG,
                            "Consider deleting duplicate folder(s) to speed "
                            "up device list parsing.",
                        )
                        continue  # skip folders of matching UIDs that have an older 'modified' time
                    dev_list[uids.index(d)] = [i, d]
                else:
                    dev_list.append([i, d])
                modified[d] = mtime
            return dev_list
        except Exception:
            # most likely cause: config folder does not yet exist (thrown by listdir)
            return []  # empty list

    @staticmethod
    def dev_get_logged_data_folders(dev_name: str) -> list:
        """Return a list of subdirectory names under the logged-data path for the given device.

        Args:
            dev_name (str): Device name used to build the logged-data path.

        Returns:
            list: Subdirectory names found under the device's logged-data path, or
            an empty list if the path does not exist.
        """
        try:
            path = os.path.join(Constants.log_prefer_path, dev_name)
            return [
                folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))
            ]
        except Exception:
            # most likely cause: this device has not yet produced any
            # logged_data (thrown by listdir)
            return []  # empty list

    @staticmethod
    def dev_get_logged_data_files(dev_name: str, data_folder: str) -> list:
        """Return a list of file names in the given logged-data subfolder for the given device.

        Args:
            dev_name (str): Device name used to build the logged-data path.
            data_folder (str): Subfolder name under the device's logged-data path.

        Returns:
            list: File names (not directories) found in the subfolder, or an empty
            list if the path does not exist.
        """
        try:
            path = os.path.join(Constants.log_prefer_path, dev_name, data_folder)
            return [
                file for file in os.listdir(path) if not os.path.isdir(os.path.join(path, file))
            ]
        except Exception:
            # most likely cause: this device has not yet produced
            # any logged_data (thrown by listdir)
            return []  # empty list

    @staticmethod
    def dev_get_all_device_dirs() -> list:
        """Return a list of directory names under the log export path with entirely numeric names.

        Underscores are permitted within the name. The length of the directory
        name is not assumed to be fixed.

        Returns:
            list[str]: Directory names that are numeric (underscores allowed).

        Raises:
            FileNotFoundError: If Constants.log_export_path does not exist.
            PermissionError: If the export path is not readable.
        """
        device_dirs = []
        for dir in os.listdir(Constants.log_export_path):
            dir_path = os.path.join(Constants.log_export_path, dir)
            if os.path.isdir(dir_path) and dir.replace("_", "").isdigit():
                device_dirs.append(dir)
        return device_dirs


class SecureOpen:
    """Context manager for transparent access to plain files or ZIP archive records.

    If the target file exists on disk it is opened normally. Otherwise the record
    is looked up inside an AES-encrypted ZIP archive adjacent to the target path.
    Applies AES password protection when user profiles are active and developer
    mode is off.
    """

    def __init__(self, file: str, mode: str = "r", zipname: Optional[str] = None):
        """Initialise the context manager.

        Args:
            file (str): Full path to the target file or in-ZIP record.
            mode (str): File open mode (e.g. "r", "w", "a").
            zipname (Optional[str]): Base name for the ZIP archive. Defaults to
            the parent directory name when not provided.
        """
        self.file = file
        self.mode = mode
        self.zipname = zipname
        self.zf = None
        self.fh = None

    def _setup_zip_encryption(self, zf: pyzipper.AESZipFile, subDir: str) -> None:
        """Test the ZIP archive for existing encryption and configure it accordingly.

        If the archive is already encrypted, unlocks it with the stored password.
        Otherwise sets a new AES password derived from a friendly name comment.
        Encryption is skipped entirely when developer mode is active.

        Args:
            zf (pyzipper.AESZipFile): Open ZIP file object to configure.
            subDir (str): Leaf directory name used to build the friendly archive name.
        """
        i = 0
        password_protected = False
        while True:
            i += 1
            if i > 3:
                Log.e(TAG, "This ZIP has encrypted files: Try again with a valid password!")
                break
            try:
                zf.testzip()  # will fail if encrypted and no password set
                break  # test pass
            except RuntimeError as e:
                if "encrypted" in str(e):
                    Log.d(TAG, "Accessing secured records...")
                    zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                    password_protected = True
                else:
                    # RuntimeError for other reasons....
                    Log.e(TAG, "ZIP RuntimeError: " + str(e))
                    break
            except Exception as e:
                # other Exception for any reason...
                Log.e(TAG, "ZIP Exception: " + str(e))
                break
        from QATCH.common.userProfiles import UserProfiles

        if (
            UserProfiles.count() > 0
            and password_protected is False
            and UserProfiles.checkDevMode()[0] is False
        ):
            # create a protected archive
            friendly_name = f"{subDir} ({datetime.date.today()})"
            zf.comment = friendly_name.encode()  # run name
            zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
        else:
            zf.setencryption(None)
            if UserProfiles.checkDevMode()[0]:
                Log.w(TAG, "Developer Mode is ENABLED - NOT encrypting ZIP file")

    def _check_record_integrity(self, zf: pyzipper.AESZipFile, record: str, mode: str) -> bool:
        """Verify that a record exists inside the ZIP and that its CRC is valid.

        For CSV records, the stored CRC is compared against the companion .crc file.

        Args:
            zf (pyzipper.AESZipFile): Open ZIP file object to inspect.
            record (str): Name of the record within the ZIP to verify.
            mode (str): File open mode; integrity checks are skipped for write mode.

        Returns:
            bool: True if it is safe to proceed, False if a security check failed.
        """
        proceed = True
        namelist = zf.namelist()
        if "w" not in mode:  # reading or appending
            if record in namelist:
                crc_file = record[:-4] + ".crc"
                if crc_file in namelist and crc_file != record:
                    archive_CRC = str(hex(zf.getinfo(record).CRC))
                    compare_CRC = zf.read(crc_file).decode()
                    Log.d(TAG, f"Archive CRC: {archive_CRC}")
                    Log.d(TAG, f"Compare CRC: {compare_CRC}")
                    if archive_CRC != compare_CRC:
                        Log.e(TAG, f"Record {record} CRC mismatch!")
                        proceed = False
                elif record.endswith(".csv"):
                    Log.e(TAG, f"Record {record} missing CRC file!")
                    proceed = False
                else:
                    Log.d(
                        TAG,
                        f"Record {record} has no CRC file, but it's not a CSV, "
                        "so allow it to proceed...",
                    )
                    # proceed = False
            else:
                Log.e(TAG, f"Record {record} not found!")
                # proceed = False
        return proceed

    def __enter__(self):
        """Open the file or ZIP record and return the file handle.

        Falls back to plain ``open()`` if the file exists on disk.

        Returns:
            IO: An open file handle to the target file or ZIP record.

        Raises:
            Exception: If security checks fail and the file cannot be opened safely.
        """
        file = self.file
        mode = self.mode
        zipname = self.zipname
        # NOTE: Writing a non-existent file will always use encryption... is that fine?
        # It should be fine. Tries to access record without 'pwd' and only encrypts if needed
        if FileManager.file_exists(file):
            self.fh = open(file, mode)  # noqa: SIM115
            return self.fh

        archive, record = os.path.split(file)
        _, subDir = os.path.split(archive)
        if zipname is None:
            zipname = subDir
        zn = os.path.join(archive, f"{zipname}.zip")
        zm = mode.replace("w", "a") if "w" in mode and FileManager.file_exists(zn) else mode
        zf = pyzipper.AESZipFile(
            zn,
            zm,
            compression=pyzipper.ZIP_DEFLATED,
            allowZip64=True,
            encryption=pyzipper.WZ_AES,
        )
        self._setup_zip_encryption(zf, subDir)
        proceed = self._check_record_integrity(zf, record, mode)

        if proceed:
            self.zf = zf  # export to global
            self.fh = zf.open(record, mode, force_zip64=True)
            return self.fh
        else:
            zf.close()
            raise Exception(f"Security checks failed. Cannot open secured file {file}.")

    def __exit__(self, type, value, traceback):
        """Close the file handle and write a companion .crc file for CSV ZIP records.

        For ZIP archives opened in write or append mode, a .crc sidecar file is
        written for any CSV records.

        Args:
            type: Exception type, or None if no exception occurred.
            value: Exception value, or None if no exception occurred.
            traceback: Exception traceback, or None if no exception occurred.
        """
        file = self.file
        mode = self.mode
        zf = self.zf
        fh = self.fh

        if fh is not None:
            fh.close()

        if zf is not None:
            _, record = os.path.split(file)

            archive_file = record
            if "r" not in mode:  # writing or appending
                namelist = zf.namelist()
                if record in namelist and archive_file.endswith(".csv"):
                    crc_file = archive_file[:-4] + ".crc"
                    archive_CRC = str(hex(zf.getinfo(archive_file).CRC))
                    # zf.writestr(crc_file, archive_CRC)
                    with zf.open(crc_file, "w") as crc_fh:  # no append, must 'w'
                        crc_fh.write(archive_CRC.encode())

            zf.close()

    @staticmethod
    def file_exists(file: str, zipname: Optional[str] = None) -> bool:
        """Return True if the file exists on disk or as a record inside the adjacent ZIP archive.

        Args:
            file (str): Full path to the target file or in-ZIP record.
            zipname (Optional[str]): Base name for the ZIP archive. Defaults to
            the parent directory name when not provided.

        Returns:
            bool: True if the file or ZIP record exists, False otherwise.
        """
        if FileManager.file_exists(file):
            return True
        else:

            archive, record = os.path.split(file)
            _, subDir = os.path.split(archive)
            if zipname is None:
                zipname = subDir
            zn = os.path.join(archive, f"{zipname}.zip")

            if not FileManager.file_exists(zn):
                return False

            zf = pyzipper.AESZipFile(
                zn,
                "r",
                compression=pyzipper.ZIP_DEFLATED,
                allowZip64=True,
                encryption=pyzipper.WZ_AES,
            )
            namelist = zf.namelist()

            return record in namelist

    @staticmethod
    def get_namelist(zip_path: str, zip_name: str = "capture") -> list:
        """Return the list of record names inside the ZIP archive at zip_path.

        Args:
            zip_path (str): Full path to the directory containing the ZIP archive.
            zip_name (str): Base name of the ZIP file, without extension.

        Returns:
            list: Record names contained in the ZIP archive.

        Raises:
            FileNotFoundError: If the ZIP file does not exist.
        """
        # Split the provided path into archive and record names
        # 'archive' is the directory, 'record' is the file/leaf
        archive, record = os.path.split(zip_path)
        # Extract parent directory and its leaf name
        folder, subDir = os.path.split(archive)

        # Use `subDir` as the zip file name if `zip_name` is None
        if zip_name is None:
            zip_name = subDir

        # Construct the full path to the zip file using the determined zip name
        zn = os.path.join(archive, f"{zip_name}.zip")

        # Check if the zip file exists. If not, raise FileNotFoundError
        if not FileManager.file_exists(zn):
            raise FileNotFoundError(f"The zip file {zn} does not exist.")

        # Open the zip file using pyzipper for AES encryption handling
        zf = pyzipper.AESZipFile(
            zn,
            "r",  # Open in read mode
            compression=pyzipper.ZIP_DEFLATED,  # Use ZIP_DEFLATED compression
            allowZip64=True,  # Support for files larger than 4GB
            encryption=pyzipper.WZ_AES,  # AES encryption support
        )

        # Extract and return the list of file names in the zip archive
        namelist = zf.namelist()
        return namelist
