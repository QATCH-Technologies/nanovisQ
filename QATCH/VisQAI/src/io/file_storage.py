"""
file_storage.py

This module provides the `SecureOpen` context manager and utility functions for
securely reading from and writing to ZIP archives, optionally using AES encryption.
It handles CRC validation for CSV records within the archive and supports both
file-based and archive-based I/O operations.

Author:
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.1
"""

import os
import hashlib
import pyzipper
from typing import Optional

TAG = "[FileStorage]"


class SecureOpen:
    """Context manager for secure I/O on files or records within a ZIP archive.

    This class acts like a standard file handle when the target is a regular file.
    If the specified path points to a non-existent file, it interprets the path
    as referencing a record inside a ZIP archive. It will then create or open the
    ZIP (with optional AES encryption), handle CRC checks for CSV files, and return
    a file-like handle to the record. On exit, it writes or updates CRC records as needed.

    Attributes:
        file (str): Full path to the target file or archive record (e.g., "/path/archive/record.csv").
        mode (str): Mode for opening the file or record (e.g., 'r', 'w', 'a').
        zipname (Optional[str]): Base name (without extension) for the ZIP file when operating on archives.
        insecure (bool): If True, bypasses CRC and security checks when opening records.
        zf (Optional[pyzipper.AESZipFile]): The AES-encrypted ZIP file object, if used.
        fh (Optional[file-like]): The file handle returned by open() or zf.open().
    """

    def __init__(self, file: str, mode: str = 'r', zipname: Optional[str] = None, insecure: bool = False):
        """Initialize the SecureOpen context manager.

        Args:
            file (str): Path to the target file or archive record.
            mode (str, optional): Mode for opening ('r', 'w', 'a', etc.). Defaults to 'r'.
            zipname (Optional[str], optional): If provided, overrides the ZIP filename (without extension).
                If None, the ZIP base name is derived from the parent folder of `file`. Defaults to None.
            insecure (bool, optional): If True, skips CRC/security validation when opening records.
                Defaults to False.
        """
        self.file = file
        self.mode = mode
        self.zipname = zipname
        self.insecure = insecure
        self.zf = None
        self.fh = None

    def __enter__(self):
        """Enter the context, opening either the regular file or a record within a ZIP archive.

        If `file` exists as a regular file on disk, it opens and returns a standard file handle.
        Otherwise, it treats `file` as "<archive_dir>/<zipname>.zip/<record_name>" and opens
        or creates the ZIP archive. It performs CRC validation for CSV records before opening.
        If the ZIP is password-protected, it attempts to derive the password from the archive comment
        using SHA-256. If encryption is disabled (development mode), it opens without a password.

        Returns:
            file-like: A file handle for reading/writing/appending to the target.

        Raises:
            Exception: If security checks fail (CRC mismatch) and `insecure` is False.
            RuntimeError: If the ZIP is encrypted and no valid password is provided after retries.
        """
        file_path = self.file
        mode = self.mode
        zipname = self.zipname

        # If the path corresponds to an existing file, open normally
        if os.path.isfile(file_path):
            self.fh = open(file_path, mode)
            return self.fh

        # Otherwise, interpret as an archive record
        archive_dir, record = os.path.split(file_path)
        parent_dir, subdir = os.path.split(archive_dir)
        if zipname is None:
            zipname = subdir
        zip_path = os.path.join(archive_dir, f"{zipname}.zip")

        # Upgrade 'w' to 'a' if archive exists to preserve existing entries
        if 'w' in mode and os.path.isfile(zip_path):
            zip_mode = mode.replace('w', 'a')
        else:
            zip_mode = mode

        # Open (or create) the AES-encrypted ZIP file
        zf = pyzipper.AESZipFile(
            zip_path,
            zip_mode,
            compression=pyzipper.ZIP_DEFLATED,
            allowZip64=True,
            encryption=pyzipper.WZ_AES
        )

        # Check for encryption by attempting a testzip
        i = 0
        password_protected = False
        while True:
            i += 1
            if i > 3:
                print("This ZIP has encrypted files: Try again with a valid password!")
                break
            try:
                zf.testzip()
                break
            except RuntimeError as e:
                # Encrypted ZIP if "encrypted" in exception message
                if 'encrypted' in str(e):
                    print("Accessing secured records...")
                    # Derive password from the archive comment via SHA-256
                    zf.setpassword(hashlib.sha256(
                        zf.comment).hexdigest().encode())
                    password_protected = True
                else:
                    print("ZIP RuntimeError: " + str(e))
                    break
            except Exception as e:
                print("ZIP Exception: " + str(e))
                break

        # Developer mode: disable encryption if no password is needed
        zf.setencryption(None)
        print("Developer Mode is ENABLED - NOT encrypting ZIP file")

        proceed = True
        archive_file = record
        namelist = zf.namelist()

        # If reading or appending, verify CRC for CSV records
        if 'w' not in mode:
            if record in namelist:
                if archive_file.endswith(".csv"):
                    crc_file = archive_file[:-4] + ".crc"
                    if crc_file in namelist and crc_file != record:
                        archive_crc = hex(zf.getinfo(archive_file).CRC)
                        compare_crc = zf.read(crc_file).decode()
                        print(f"Archive CRC: {archive_crc}")
                        print(f"Compare CRC: {compare_crc}")
                        if archive_crc != compare_crc:
                            print(f"Record {record} CRC mismatch!")
                            proceed = False
                    else:
                        print(f"Record {record} missing CRC file!")
                        proceed = False
                else:
                    print(
                        f"Record {record} has no CRC file, but it's not a CSV, so allow it to proceed...")
            else:
                print(f"Record {record} not found!")
                proceed = False

        # If CRC checks pass or insecure is True, open the record
        if proceed or self.insecure:
            self.zf = zf
            self.fh = zf.open(archive_file, mode, force_zip64=True)
            return self.fh

        # Otherwise, close and raise an exception
        zf.close()
        raise Exception(
            f"Security checks failed. Cannot open secured file {file_path}.")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context, closing file handles and updating CRC records if needed.

        On exit, if writing or appending to a CSV record, computes the CRC of the record
        and writes/overwrites a corresponding ".crc" entry in the ZIP. Finally, closes
        both the record handle and the ZIP file.

        Args:
            exc_type: Exception type, if any was raised inside the context.
            exc_value: Exception value, if any was raised inside the context.
            traceback: Traceback, if any exception was raised.
        """
        if self.fh is not None:
            self.fh.close()

        if self.zf is not None:
            _, record = os.path.split(self.file)
            archive_file = record

            # If writing/appending, update CRC for CSV records
            if 'r' not in self.mode:
                namelist = self.zf.namelist()
                if archive_file in namelist and archive_file.endswith(".csv"):
                    crc_file = archive_file[:-4] + ".crc"
                    archive_crc = hex(self.zf.getinfo(archive_file).CRC)
                    # Write or overwrite the CRC file inside the ZIP
                    with self.zf.open(crc_file, 'w') as crc_fh:
                        crc_fh.write(archive_crc.encode())

            self.zf.close()

    @staticmethod
    def file_exists(file: str, zipname: Optional[str] = None) -> bool:
        """Check whether a file or record exists, either on disk or within a ZIP archive.

        If `file` exists as a regular file, returns True. Otherwise, interprets `file`
        as "<archive_dir>/<zipname>.zip/<record>" and checks if the ZIP and record exist.

        Args:
            file (str): Path to the target file or archive record.
            zipname (Optional[str], optional): Base name (without extension) for the ZIP file.
                If None, it is derived from the parent folder of `file`. Defaults to None.

        Returns:
            bool: True if the file or archive record exists; False otherwise.
        """
        if os.path.isfile(file):
            return True

        archive_dir, record = os.path.split(file)
        parent_dir, subdir = os.path.split(archive_dir)
        if zipname is None:
            zipname = subdir
        zip_path = os.path.join(archive_dir, f"{zipname}.zip")

        if not os.path.isfile(zip_path):
            return False

        with pyzipper.AESZipFile(
            zip_path,
            'r',
            compression=pyzipper.ZIP_DEFLATED,
            allowZip64=True,
            encryption=pyzipper.WZ_AES
        ) as zf:
            namelist = zf.namelist()
        return record in namelist

    @staticmethod
    def get_namelist(zip_path: str, zip_name: str = "capture") -> list:
        """Retrieve the list of file names contained in a ZIP archive.

        Opens the specified ZIP (handles AES encryption) and returns its namelist.

        Args:
            zip_path (str): Path to any file or subpath inside the archive (e.g., "/path/to/archive/record.csv").
                             The method extracts the archive directory from this path.
            zip_name (str, optional): Base name of the ZIP file (without extension). If None, derived from the parent folder.
                                     Defaults to "capture".

        Returns:
            list: List of file names (str) present in the ZIP archive.

        Raises:
            FileNotFoundError: If the constructed ZIP path does not exist.
        """
        archive_dir, _ = os.path.split(zip_path)
        parent_dir, subdir = os.path.split(archive_dir)

        if zip_name is None:
            zip_name = subdir

        zip_file_path = os.path.join(archive_dir, f"{zip_name}.zip")
        if not os.path.isfile(zip_file_path):
            raise FileNotFoundError(
                f"The zip file {zip_file_path} does not exist.")

        with pyzipper.AESZipFile(
            zip_file_path,
            'r',
            compression=pyzipper.ZIP_DEFLATED,
            allowZip64=True,
            encryption=pyzipper.WZ_AES
        ) as zf:
            namelist = zf.namelist()

        return namelist
