import csv
from QATCH.common.fileManager import FileManager
from QATCH.core.constants import Constants
from QATCH.common.logger import Logger as Log
from time import strftime, localtime
import datetime
import numpy as np
import os
import hashlib
import pyzipper

TAG = ""#"[FileStorage]"

###############################################################################
# Stores and exports data to file (CSV/TXT): saves incoming and outcoming data
###############################################################################
class FileStorage:

    # Global buffer and index helpers
    bufferedRows = []
    HANDLE = 0
    TOPATH = 1
    BUFFER = 2
    downsampling = False

    ###########################################################################
    # Saves CSV files of processed data in an assigned directory
    ###########################################################################
    @staticmethod
    def CSVsave(i, filename, path, data_save0, data_save1, data_save2, data_save3, data_save4, data_save5, writeToFilesystem = True):
        """
        :param filename: Name for the file :type filename: str.
        :param path: Path for the file     :type path: str.
        :param data_save0: data to store (w_time)              :type float.
        :param data_save1: data to store (temperature)         :type float.
        :param data_save2: data to store (raw peak magnitude)  :type float.
        :param data_save3: data to store (resonance frequency) :type float.
        :param data_save4: data to store (dissipation)         :type float.
        :param data_save5: data to store (ambient)             :type float.

        """
        HANDLE = FileStorage.HANDLE
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER

        fix1= "%Y-%m-%d"
        fix2= "%H:%M:%S"
        csv_time_prefix1 = (strftime(fix1, localtime()))
        csv_time_prefix2 = (strftime(fix2, localtime()))
        d0=float("{0:.4f}".format(data_save0))
        d1=float("{0:.2f}".format(data_save1))
        d3=float("{0:.2f}".format(data_save3))
        d4=float("{:.15e}".format(data_save4))
        d5=float("{0:.2f}".format(data_save5))

        # Append device name folder to path
        path = os.path.join(path, FileStorage.DEV_get_active(i))
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Find index in buffered row data from file handle (create index if new)
        full_path = FileManager.create_full_path(filename, extension=Constants.csv_extension, path=path)
        fHashKey = hashlib.sha1(full_path.encode('utf-8')).hexdigest()
        fHandle = len(FileStorage.bufferedRows)
        for x in range(fHandle):
            if (FileStorage.bufferedRows[x][HANDLE] == fHashKey and
                FileStorage.bufferedRows[x][TOPATH] == full_path):
                fHandle = x
                break
        if fHandle == len(FileStorage.bufferedRows):
            FileStorage.bufferedRows.append([fHashKey, full_path, []])

        FileStorage.bufferedRows[fHandle][BUFFER].append([csv_time_prefix1,csv_time_prefix2, d0, d5, d1, data_save2, d3, d4])

        if writeToFilesystem:
            FileStorage.__correctForDupTimes__(fHandle)
            FileStorage.__writeBufferedData__(fHandle)


    ###########################################################################
    # Flushes buffered data to pending CSV files in an assigned directory
    ###########################################################################
    @staticmethod
    def CSVflush_all():
        HANDLE = FileStorage.HANDLE
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER

        # This function (called at end of SWEEPs loop in Serial.py) flushes buffers to filesystem!
        fHandle = len(FileStorage.bufferedRows)
        for x in range(fHandle):
            full_path = FileStorage.bufferedRows[x][TOPATH]
            fHashKey = hashlib.sha1(full_path.encode('utf-8')).hexdigest()
            if (FileStorage.bufferedRows[x][HANDLE] == fHashKey and
            not FileStorage.bufferedRows[x][BUFFER] == []):
                FileStorage.__correctForDupTimes__(x)
                FileStorage.__writeBufferedData__(x)


    ###########################################################################
    # Private method for correcting duplicate relative times in data received
    ###########################################################################
    @staticmethod
    def __correctForDupTimes__(handle):
        HANDLE = FileStorage.HANDLE
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER
        PRECISION = ".0001"

        # Check for duplicate times, and (if any) evenly space times
        FileStorage.bufferedRows[handle][BUFFER] = np.array(FileStorage.bufferedRows[handle][BUFFER])
        bufferTimes = FileStorage.bufferedRows[handle][BUFFER][:,2]
        if FileStorage.__checkIfDuplicates__(bufferTimes):
            bufferTimes = np.linspace(float(bufferTimes[0]), float(bufferTimes[-1])-float(PRECISION), len(bufferTimes))
            bufferTimes = ["{0:.{1}f}".format(x, len(PRECISION)-1) for x in bufferTimes]
            FileStorage.bufferedRows[handle][BUFFER][:,2] = bufferTimes

    @staticmethod
    def __checkIfDuplicates__(list):
        ''' Check if given list contains any duplicates '''
        uniques = set()
        for elem in list:
            if elem in uniques:
                return True
            else:
                uniques.add(elem)
        return False


    ###########################################################################
    # Private method for writing buffered data from internal buffer to CSV file
    ###########################################################################
    @staticmethod
    def __writeBufferedData__(handle):
        HANDLE = FileStorage.HANDLE
        TOPATH = FileStorage.TOPATH
        BUFFER = FileStorage.BUFFER

        full_path = FileStorage.bufferedRows[handle][TOPATH]

        # Creates a directory if the specified path doesn't exist
        #FileManager.create_dir(path)
        # Creates a file full path based on parameters
        if not FileManager.file_exists(full_path):
            Log.i(TAG, "Exporting data to CSV file...")
            Log.i(TAG, "Storing in: {}".format(full_path))

            with open(Constants.new_files_path, 'a') as tempFile:
                tempFile.write(full_path + "\n")

            FileStorage.downsampling = False

        # checks the path for the header insertion
        if os.path.exists(full_path):
            header_exists = True
        else:
            header_exists = False

        # opens the file to write data
        with open(full_path,'a', newline='') as tempFile:
            tempFileWriter = csv.writer(tempFile)
            # inserts the header if it doesn't exist
            if not header_exists:
                tempFileWriter.writerow(["Date","Time","Relative_time","Ambient","Temperature","Peak Magnitude (RAW)","Resonance_Frequency","Dissipation"])

            # peak into buffer times, downsample if greater than 90 seconds in
            bufferTimes = FileStorage.bufferedRows[handle][BUFFER][:,2]

            if float(bufferTimes[-1]) < Constants.downsample_after: # do not downsample
                FileStorage.downsampling = False
                # write the buffered data
                tempFileWriter.writerows(FileStorage.bufferedRows[handle][BUFFER])
                FileStorage.bufferedRows[handle][BUFFER] = [] # empty buffer
            elif float(bufferTimes[0]) < Constants.downsample_after and float(bufferTimes[-1]) > Constants.downsample_after: # on the edge of downsampling
                Log.d("Starting to downsample with mixed buffer (on the edge)...")
                for i in range(len(bufferTimes)):
                    if float(bufferTimes[i]) > Constants.downsample_after:
                        # write the non-downsampled buffered data
                        tempFileWriter.writerows(FileStorage.bufferedRows[handle][BUFFER][:i])
                        FileStorage.bufferedRows[handle][BUFFER] = FileStorage.bufferedRows[handle][BUFFER][i:]
                        # convert partial buffer (now of type 'numpy.ndarray') back to 'list'
                        FileStorage.bufferedRows[handle][BUFFER] = FileStorage.bufferedRows[handle][BUFFER].tolist()
                        break
            else: # downsample
                if not FileStorage.downsampling:
                    Log.w(f"Downsampling started after {Constants.downsample_after} seconds of measurement capture...")
                    Log.w(f"Each sample written to file is now an average of {Constants.downsample_file_count} raw data points.")
                    Log.w(f"Each sample written to plot is now just 1 in every {Constants.downsample_plot_count} raw data points.")
                    FileStorage.downsampling = True

                if True: # len(bufferTimes) >= Constants.downsample_file_count:
                    mid_interval = int(len(bufferTimes)/2)
                    date_mid = FileStorage.bufferedRows[handle][BUFFER][:,0][mid_interval]
                    time_mid = FileStorage.bufferedRows[handle][BUFFER][:,1][mid_interval]

                    ambient = FileStorage.bufferedRows[handle][BUFFER][:,3]
                    temperature = FileStorage.bufferedRows[handle][BUFFER][:,4]
                    peak_mag = FileStorage.bufferedRows[handle][BUFFER][:,5]
                    res_freq = FileStorage.bufferedRows[handle][BUFFER][:,6]
                    diss = FileStorage.bufferedRows[handle][BUFFER][:,7]

                    relative_avg =  "{0:.4f}".format(np.average([float(x) for x in bufferTimes]))
                    ambient_avg =   "{0:.2f}".format(np.average([float(x) for x in ambient]))
                    temp_avg =      "{0:.2f}".format(np.average([float(x) for x in temperature]))
                    peak_avg =      "{0:.0f}".format(np.average([float(x) for x in peak_mag]))
                    res_avg =       "{0:.0f}".format(np.average([float(x) for x in res_freq]))
                    diss_avg =      "{:.15e}".format(np.average([float(x) for x in diss]))

                    downsampled_buffer = [date_mid, time_mid, relative_avg, ambient_avg, temp_avg, peak_avg, res_avg, diss_avg]
                    tempFileWriter.writerow(downsampled_buffer)
                    FileStorage.bufferedRows[handle][BUFFER] = [] # empty buffer

            tempFile.close()


    ###########################################################################
    # Saves a CSV-formatted CSV file per sweeps in an assigned directory
    ###########################################################################
    @staticmethod
    def CSV_sweeps_save(i, filename, path, data_save1, data_save2, data_save3 = None):
        """
        :param filename: Name for the file :type filename: str.
        :param path: Path for the file     :type path: str.
        :param data_save1: data to store (frequency) :type float.
        :param data_save2: data to store (Amplitude) :type float.
        :param data_save3: data to store (Phase)     :type float.
        """
        # Append device name folder to path
        path = os.path.join(path, FileStorage.DEV_get_active(i))
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Creates a file full path based on parameters
        full_path = FileManager.create_full_path(filename, extension=Constants.csv_extension, path=path)
        # creates CSV file
        if data_save3 is None:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2]), delimiter=',')
        else:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2, data_save3]), delimiter=',')


    ###########################################################################
    # Saves a CSV-formatted Text file per sweeps in an assigned directory
    ###########################################################################
    @staticmethod
    def TXT_sweeps_save(i, filename, path, data_save1, data_save2, data_save3 = None, appendNameToPath = True):
        """
        :param filename: Name for the file :type filename: str.
        :param path: Path for the file     :type path: str.
        :param path: Path for the file     :type path: str.
        :param data_save1: data to store (frequency) :type float.
        :param data_save2: data to store (Amplitude) :type float.
        :param data_save3: data to store (Phase)     :type float.
        """
        if appendNameToPath:
            # Append device name folder to path
            path = os.path.join(path, FileStorage.DEV_get_active(i))
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Creates a file full path based on parameters
        full_path = FileManager.create_full_path(filename, extension=Constants.txt_extension, path=path)
        # creates TXT file
        if data_save3 is None:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2]))
        else:
            np.savetxt(full_path, np.column_stack([data_save1, data_save2, data_save3]))


    ###########################################################################
    # Get Device Info fields from file for the given device name
    ###########################################################################
    @staticmethod
    def DEV_info_get(i, dev_name):
        try:
            # Append device name folder to path (using provided parameter 'dev_name')
            dev_folder = "{}_{}".format(i, dev_name) if i > 0 else dev_name
            path = os.path.join(Constants.csv_calibration_export_path, dev_folder)
            filename = Constants.txt_device_info_filename
            # Creates a file full path based on parameters
            full_path = FileManager.create_full_path(filename, extension=Constants.txt_extension, path=path)
            # Read in device info from file (if exists)
            if FileManager.file_exists(full_path):
                with open(full_path, "r") as fh:
                    lines = fh.read().split('\n')
                    # Remove blank lines and split on delimiter
                    for i in range(len(lines)):
                        if lines[i] == '':
                            del lines[i]
                        else:
                            lines[i] = lines[i].split(': ')
                    # Parse labels and values from file
                    data = list(zip(*lines))
                    labels = data[0]
                    values = data[1]
                    # Generate and return dict of keys and values
                    info = dict(zip(labels,values))
                    if info['USB'] == dev_name:
                        return info
                    Log.w(TAG, "WARN: Device info for {} is for another device!".format(dev_name))
            else:
                Log.w(TAG, "WARN: Device info for {} does not exist.".format(dev_name))
        except:
            Log.w(TAG, "WARN: Could not parse device info for {}.".format(dev_name))
        return {}


    ###########################################################################
    # Set Device Info fields to file for the given device name
    ###########################################################################
    @staticmethod
    def DEV_info_set(i, filename, path, dev_name, fw, hw, port = None, ip = None,  uid = None, mac = None, usb = None, pid = None, rev = None, err = None):
        # Append device name folder to path (using provided parameter 'dev_name')
        dev_folder = "{}_{}".format(i, dev_name) if i > 0 else dev_name
        path = os.path.join(path, dev_folder)
        # Creates a directory if the specified path doesn't exist
        FileManager.create_dir(path)
        # Update pointer to active device name
        FileStorage.DEV_set_active(i, dev_name)
        # Creates a file full path based on parameters
        full_path = FileManager.create_full_path(filename, extension=Constants.txt_extension, path=path)
        # Declare vars and put new values in a list
        writeFile = False
        newVals = [dev_name,fw,hw,port,ip,uid,mac,usb,pid,rev,err]
        while None in newVals:
            newVals.remove(None)
        newLen = len(newVals)
        # Read in device info from file (if exists)
        if FileManager.file_exists(full_path):
            try:
                with open(full_path, "r") as fh:
                    lines = fh.read().split('\n')
                    for i in range(len(lines)):
                        if lines[i] == '':
                            del lines[i]
                        else:
                            lines[i] = lines[i].split(': ')
                    #Generate list and length of old values (no dups)
                    oldData = list(zip(*lines))
                    oldLabels = oldData[0]
                    oldVals = oldData[1]
                    oldLen = len(oldVals)
                    # Do not change device name if custom name value is set in info file
                    if oldLabels[0] == "NAME" and newVals[0] != oldVals[0]: # NAME must be first line of info file
                        newVals[0] = oldVals[0] # we have a custom "NAME" - do not change name stored in info file
                    # Do not blow away COM port from config file if device is purely IP/NET configured
                    if port == None and oldLabels[3] == "PORT":
                        port = oldVals[3]
                        newVals.insert(3, port)
                        newLen += 1
                    # Determine whether or not to update the file
                    if not oldLen == newLen:
                        writeFile = True
                    else:
                        for val in oldVals:
                            if not val in newVals:
                                writeFile = True
                                break
            except:
                Log.w(TAG, "WARN: Could not parse existing device info file. Replacing it now.")
                writeFile = True
            # if there are no changes, writeFile will still be False
        else:
            # File does not exist, create it now
            writeFile = True
        # Create or update TXT file
        if writeFile:
            Log.i(TAG, "Writing device info file for {}...".format(dev_name))
            with open(full_path, "w") as fh:
                labels = ["NAME","FW","HW"]
                if not port == None: labels.append("PORT")
                if not ip == None: labels.append("IP")
                if not uid == None: labels.append("UID")
                if not mac == None: labels.append("MAC")
                if not usb == None: labels.append("USB")
                if not pid == None: labels.append("PID")
                if not rev == None: labels.append("REV")
                if not err == None: labels.append("ERR")
                if len(labels) != newLen:
                    Log.w(TAG, "WARN: Device info file labels are misaligned!")
                for i in range(newLen):
                    fh.write(labels[i] + ': ' + newVals[i] + '\n')
            # Nothing exploded, so we must be OK
            Log.i(TAG, "Device info written!")
        return writeFile


    ###########################################################################
    # Get Active Device field from file  (USB or PORT field)
    ###########################################################################
    @staticmethod
    def DEV_get_active(i):
        try:
            path = Constants.csv_calibration_export_path
            filename = Constants.txt_active_device_filename
            full_path = FileManager.create_full_path(filename, extension=Constants.txt_extension, path=path)
            if os.path.isdir(path):
                with open(full_path, "r") as fh:
                    _active_device_folders = fh.readlines()
                    idx = max(0, i - 1)
                    if idx < len(_active_device_folders):
                        _active_device_folder = _active_device_folders[idx].strip()
                        if len(_active_device_folder) == 0:
                            raise ValueError(f"Device with PID {i} contains no value in the active list.")
                    else:
                        raise IndexError(f"Device with PID {i} is not in the active list.")
                    _dev_path = os.path.join(path, _active_device_folder)
                    if os.path.isdir(_dev_path):
                        return _active_device_folder
                    else:
                        Log.w(TAG, "WARN: Active device does not yet exist.")
                        return _active_device_folder
            else:
                # The config folder does not yet exist (first run)
                pass # ignore
        except ValueError as e:
            Log.e("ERROR:", str(e)) # Log error to user
            Log.w(TAG, "WARN: Failed to get active device name.")
        except IndexError as e:
            Log.e("ERROR:", str(e)) # Log error to user
            Log.w(TAG, "WARN: Failed to get active device name.")
        except Exception as e:
            Log.w(TAG, "WARN: Failed to get active device name.")

            limit = None
            import sys
            t, v, tb = sys.exc_info()
            from traceback import format_tb
            a_list = ['Traceback (most recent call last):']
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        return '' # default, use root


    ###########################################################################
    # Set Active Device field to file (USB or PORT field)
    ###########################################################################
    @staticmethod
    def DEV_set_active(i, dev_name):
        try:
            path = Constants.csv_calibration_export_path
            filename = Constants.txt_active_device_filename
            full_path = FileManager.create_full_path(filename, extension=Constants.txt_extension, path=path)
            # Creates a directory if the specified path doesn't exist
            FileManager.create_dir(path)
            mode = "w" if i in [None, 0, 1] else "r+"
            with open(full_path, mode) as fh:
                if i != None and i > 0:
                    f_lines = fh.readlines() if mode == "r+" else []
                    num_devs = len(f_lines) if mode == "r+" else 0
                    if i > num_devs or len(f_lines[i-1].strip()) == 0:
                        while num_devs < i:
                            f_lines.append("\n")
                            num_devs += 1
                        f_lines[i-1] = "{}_{}\n".format(i, dev_name)
                    else:
                        Log.d(f"Device \"{i}_{dev_name}\" is already in active list. Ignoring duplicate add request.")
                    # replace entire file with 'f_lines' contents, but only after reading it with 'readlines()' earlier
                    fh.seek(0)
                    fh.writelines(f_lines)
                    fh.truncate()
                else:
                    fh.write(dev_name + '\n')
        except:
            Log.w(TAG, "WARN: Failed to set active device name.")


    ###########################################################################
    # Populate Device Path to insert device folder name in file path
    ###########################################################################
    @staticmethod
    def DEV_populate_path(path, i):
        return path.replace(Constants.tbd_active_device_name_path, FileStorage.DEV_get_active(i))


    ###########################################################################
    # Get Device List of all folders found in the config folder
    ###########################################################################
    @staticmethod
    def DEV_get_device_list():
        try:
            path = Constants.csv_calibration_export_path
            devs = [device for device in os.listdir(path) if os.path.isdir(os.path.join(path, device))]
            dev_list = []
            modified = {}
            for d in devs:
                uids = [i[1] for i in dev_list]
                dev_file = os.path.join(path, d, f"{Constants.txt_device_info_filename}.{Constants.txt_extension}")
                if os.path.exists(dev_file):
                    mtime = os.path.getmtime(dev_file)
                else:
                    mtime = -1
                if d.find('_') > 0:
                    s = d.split("_")
                    i = int(s[0])
                    d = s[1]
                else:
                    i = 0
                if d in uids:
                    if mtime < modified[d]:
                        Log.d("Skipping config folder as older duplicate:", f"{i}_{d}" if i != 0 else d)
                        Log.d("Consider deleting duplicate folder(s) to speed up device list parsing.")
                        continue # skip folders of matching UIDs that have an older 'modified' time
                    dev_list[uids.index(d)] = [i, d]
                else:
                    dev_list.append([i, d])
                modified[d] = mtime
            return dev_list
        except:
            # most likely cause: config folder does not yet exist (thrown by listdir)
            return [] # empty list


    ###########################################################################
    # Get a list of all folders found in the logged_data folder
    ###########################################################################
    @staticmethod
    def DEV_get_logged_data_folders(dev_name):
        try:
            path = os.path.join(Constants.log_export_path, dev_name)
            return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        except:
            # most likely cause: this device has not yet produced any logged_data (thrown by listdir)
            return [] # empty list


    ###########################################################################
    # Get a list of all files found in the given data folder
    ###########################################################################
    @staticmethod
    def DEV_get_logged_data_files(dev_name, data_folder):
        try:
            path = os.path.join(Constants.log_export_path, dev_name, data_folder)
            return [file for file in os.listdir(path) if not os.path.isdir(os.path.join(path, file))]
        except:
            # most likely cause: this device has not yet produced any logged_data (thrown by listdir)
            return [] # empty list


###########################################################################
# Get an IO handle to read/write/append to a secured ZIP archive record
###########################################################################
class secure_open:

    def __init__(self, file, mode='r', zipname=None):
        self.file = file
        self.mode = mode
        self.zipname = zipname
        self.zf = None
        self.fh = None

    def __enter__(self):
        file = self.file
        mode = self.mode
        zipname = self.zipname
        zf = self.zf
        fh = self.fh

        # NOTE: Writing a non-existent file will always use encryption... is that fine?
        # It should be fine. Tries to access record without 'pwd' and only encrypts if needed
        if FileManager.file_exists(file):
            self.fh = open(file, mode)
            return self.fh
        else:
            archive, record = os.path.split(file)
            folder, subDir = os.path.split(archive)
            if zipname == None: zipname = subDir
            zn = os.path.join(archive, f"{zipname}.zip")
            if 'w' in mode and FileManager.file_exists(zn):
                # if archive exists, upgrade 'w' to 'a' to keep other records already in ZIP
                zm = mode.replace('w', 'a')
            else:
                zm = mode
            zf = pyzipper.AESZipFile(zn, zm,
                                     compression=pyzipper.ZIP_DEFLATED,
                                     allowZip64=True,
                                     encryption=pyzipper.WZ_AES)
            if True:
                i = 0
                password_protected = False
                while True:
                    i += 1
                    if i > 3:
                        Log.e("This ZIP has encrypted files: Try again with a valid password!")
                        break
                    try:
                        zf.testzip() # will fail if encrypted and no password set
                        break # test pass
                    except RuntimeError as e:
                        if 'encrypted' in str(e):
                            Log.d('Accessing secured records...')
                            zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                            password_protected = True
                        else:
                            # RuntimeError for other reasons....
                            Log.e("ZIP RuntimeError: " + str(e))
                            break
                    except Exception as e:
                        # other Exception for any reason...
                        Log.e("ZIP Exception: " + str(e))
                        break

                from QATCH.common.userProfiles import UserProfiles
                if UserProfiles.count() > 0 and password_protected == False and UserProfiles.checkDevMode()[0] == False:
                    # create a protected archive
                    friendly_name = f"{subDir} ({datetime.date.today()})"
                    zf.comment = friendly_name.encode() # run name
                    zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                else:
                    zf.setencryption(None)
                    if UserProfiles.checkDevMode()[0]:
                        Log.w("Developer Mode is ENABLED - NOT encrypting ZIP file")

                proceed = True
                archive_file = record
                namelist = zf.namelist()
                if not 'w' in mode: # reading or appending
                    if record in namelist:
                        crc_file = archive_file[:-4] + ".crc"

                        if crc_file in namelist:
                            archive_CRC = str(hex(zf.getinfo(archive_file).CRC))
                            compare_CRC = zf.read(crc_file).decode()

                            Log.d(f"Archive CRC: {archive_CRC}")
                            Log.d(f"Compare CRC: {compare_CRC}")

                            if not archive_CRC == compare_CRC:
                                Log.e(f"Record {record} CRC mismatch!")
                                proceed = False
                        elif archive_file.endswith(".csv"):
                            Log.e(f"Record {record} missing CRC file!")
                            proceed = False
                        else:
                            Log.d(f"Record {record} has no CRC file, but it's not a CSV, so allow it to proceed...")
                            # proceed = False
                    else:
                        Log.e(f"Record {record} not found!")
                        # proceed = False

                if proceed:
                    self.zf = zf # export to global
                    self.fh = zf.open(archive_file, mode, force_zip64 = True)
                    return self.fh
                else:
                    zf.close()
                    raise Exception(f"Security checks failed. Cannot open secured file {file}.")

    def __exit__(self, type, value, traceback):
        file = self.file
        mode = self.mode
        zipname = self.zipname
        zf = self.zf
        fh = self.fh

        if fh != None:
            fh.close()

        if zf != None:
            _, record = os.path.split(file)

            archive_file = record
            if not 'r' in mode: # writing or appending
                namelist = zf.namelist()
                if record in namelist:
                    if archive_file.endswith(".csv"):
                        crc_file = archive_file[:-4] + ".crc"
                        archive_CRC = str(hex(zf.getinfo(archive_file).CRC))
                        # zf.writestr(crc_file, archive_CRC)
                        with zf.open(crc_file, 'w') as crc_fh: # no append, must 'w'
                            crc_fh.write(archive_CRC.encode())

            zf.close()


    ###########################################################################
    # Get an IO handle to read/write/append to a secured ZIP archive record
    ###########################################################################
    @staticmethod
    def file_exists(file, zipname=None):
        if FileManager.file_exists(file):
            return True
        else:

            archive, record = os.path.split(file)
            folder, subDir = os.path.split(archive)
            if zipname == None: zipname = subDir
            zn = os.path.join(archive, f"{zipname}.zip")

            if not FileManager.file_exists(zn):
                return False

            zf = pyzipper.AESZipFile(zn, 'r',
                                     compression=pyzipper.ZIP_DEFLATED,
                                     allowZip64=True,
                                     encryption=pyzipper.WZ_AES)
            namelist = zf.namelist()

            return True if record in namelist else False
