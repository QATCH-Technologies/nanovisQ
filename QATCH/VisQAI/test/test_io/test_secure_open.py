"""
test_file_storage.py

Unit tests for the SecureOpen class in `src.io.file_storage`, verifying:
    - Reading and writing regular files
    - Creating and accessing records inside a ZIP archive, including CRC generation
    - Handling of missing CRC files (both secure and insecure modes)
    - file_exists behavior for both regular paths and paths inside a ZIP
    - get_namelist behavior for valid and non-existent ZIP archives

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.0
"""

import unittest
import os
import tempfile
import shutil
import pyzipper

# Import the SecureOpen class and its static methods
from src.io.file_storage import SecureOpen


class TestSecureOpen(unittest.TestCase):
    """Unit tests for the `SecureOpen` class that provides IO handles for regular files
    and AES-encrypted ZIP archive records, with CRC integrity checks."""

    def setUp(self):
        """Create a temporary directory and an initial regular file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.regular_file_path = os.path.join(self.temp_dir, "regular.txt")
        with open(self.regular_file_path, "w") as f:
            f.write("initial content")

    def tearDown(self):
        """Remove the temporary directory and all its contents after each test."""
        shutil.rmtree(self.temp_dir)

    def test_open_regular_file_read_write(self):
        """
        Test reading from and writing to an existing regular file using SecureOpen.

        - Open in 'r' mode and verify initial content.
        - Open in 'w' mode, write new content, and verify it was written.
        """
        # Read the existing regular file
        with SecureOpen(self.regular_file_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "initial content")

        # Write new content
        with SecureOpen(self.regular_file_path, "w") as f:
            f.write("new content")

        # Verify the new content
        with open(self.regular_file_path, "r") as f:
            self.assertEqual(f.read(), "new content")

    def test_write_and_read_record_in_zip(self):
        """
        Test writing a CSV record into a new ZIP archive and reading it back securely.

        - Create a new directory and define a record path inside it.
        - Use SecureOpen to write binary CSV data into a record, creating a ZIP.
        - Open the ZIP with pyzipper to verify that:
            * The CSV record is present.
            * A CRC file was generated with matching checksum.
        - Use SecureOpen in 'r' mode with insecure=True to read back the CSV bytes.
        """
        archive_dir = os.path.join(self.temp_dir, "archive_dir")
        os.makedirs(archive_dir, exist_ok=True)
        record_name = "test.csv"
        record_path = os.path.join(archive_dir, record_name)

        # Write CSV data into the ZIP archive
        with SecureOpen(record_path, "w") as f:
            f.write(b"col1,col2\n1,2\n")

        zip_path = os.path.join(archive_dir, "archive_dir.zip")
        self.assertTrue(os.path.isfile(zip_path),
                        "ZIP archive was not created")

        # Verify contents using pyzipper
        with pyzipper.AESZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn(record_name, namelist, "CSV record not found in ZIP")
            crc_file_name = "test.crc"
            self.assertIn(crc_file_name, namelist, "CRC file not found in ZIP")

            info = zf.getinfo(record_name)
            expected_crc = hex(info.CRC)
            actual_crc = zf.read(crc_file_name).decode()
            self.assertEqual(
                actual_crc,
                expected_crc,
                "CRC file content does not match calculated CRC"
            )

        # Read back the CSV record via SecureOpen in insecure mode
        with SecureOpen(record_path, "r", insecure=True) as f:
            content = f.read()
        self.assertEqual(content, b"col1,col2\n1,2\n")

    def test_crc_file_generated(self):
        """
        Test that writing a CSV record results in the generation of a .crc file in the ZIP.

        - Create a new directory and record path.
        - Write binary data via SecureOpen.
        - Re-open the ZIP to verify:
            * The CSV record exists.
            * A .crc file was created.
            * The CRC inside .crc matches the CSV entry's CRC.
        """
        archive_dir = os.path.join(self.temp_dir, "archive2")
        os.makedirs(archive_dir, exist_ok=True)
        record_name = "data.csv"
        record_path = os.path.join(archive_dir, record_name)

        # Write the CSV record
        with SecureOpen(record_path, "w") as f:
            f.write(b"header\nvalue\n")

        zip_path = os.path.join(archive_dir, "archive2.zip")
        self.assertTrue(os.path.isfile(zip_path),
                        "ZIP archive was not created")

        # Verify CRC file was generated correctly
        with pyzipper.AESZipFile(zip_path, "r") as zf:
            self.assertIn("data.csv", zf.namelist(), "CSV record missing")
            self.assertIn("data.crc", zf.namelist(), "CRC file missing")
            info = zf.getinfo("data.csv")
            expected_crc = hex(info.CRC)
            with zf.open("data.crc", "r") as crc_fh:
                actual_crc = crc_fh.read().decode()
            self.assertEqual(
                actual_crc,
                expected_crc,
                "CRC in .crc file does not match expected"
            )

    def test_read_record_missing_crc_raises(self):
        """
        Test that attempting to read a record without a corresponding .crc file raises an Exception.

        - Manually create a ZIP with a CSV entry but omit the .crc file.
        - Use SecureOpen in secure mode (insecure=False) to open the record.
        - Expect an Exception due to failed CRC integrity check.
        """
        archive_dir = os.path.join(self.temp_dir, "archive3")
        os.makedirs(archive_dir, exist_ok=True)
        zip_path = os.path.join(archive_dir, "archive3.zip")
        with pyzipper.AESZipFile(zip_path, "w", compression=pyzipper.ZIP_DEFLATED) as zf:
            zf.writestr("sample.csv", b"abc,123\n")

        record_path = os.path.join(archive_dir, "sample.csv")
        with self.assertRaises(Exception):
            with SecureOpen(record_path, "r") as f:
                _ = f.read()

    def test_read_record_missing_crc_insecure(self):
        """
        Test that using insecure=True allows reading a record even if the .crc is missing.

        - Create a ZIP with a CSV entry but omit the .crc file.
        - Use SecureOpen in insecure mode to open the record.
        - Expect no exception and correct binary content returned.
        """
        archive_dir = os.path.join(self.temp_dir, "archive4")
        os.makedirs(archive_dir, exist_ok=True)
        zip_path = os.path.join(archive_dir, "archive4.zip")
        with pyzipper.AESZipFile(zip_path, "w", compression=pyzipper.ZIP_DEFLATED) as zf:
            zf.writestr("sample2.csv", b"def,456\n")

        record_path = os.path.join(archive_dir, "sample2.csv")
        try:
            with SecureOpen(record_path, "r", insecure=True) as f:
                content = f.read()
            self.assertEqual(content, b"def,456\n")
        except Exception:
            self.fail("SecureOpen raised Exception despite insecure=True")

    def test_file_exists_regular_and_zip(self):
        """
        Test file_exists static method for both regular files and records inside a ZIP.

        - Verify file_exists returns True for an existing regular file.
        - Verify file_exists returns False for a non-existent record path inside a ZIP directory.
        - Create a ZIP containing one record, then verify file_exists returns True for that record.
        """
        # Regular file
        self.assertTrue(SecureOpen.file_exists(self.regular_file_path))

        # Non-existent record in a directory that has no ZIP yet
        archive_dir = os.path.join(self.temp_dir, "archive5")
        os.makedirs(archive_dir, exist_ok=True)
        record_path = os.path.join(archive_dir, "noexist.csv")
        self.assertFalse(SecureOpen.file_exists(record_path))

        # Create a ZIP with a record
        zip_path = os.path.join(archive_dir, "archive5.zip")
        with pyzipper.AESZipFile(zip_path, "w", compression=pyzipper.ZIP_DEFLATED) as zf:
            zf.writestr("exists.csv", b"test\n")

        record_exists_path = os.path.join(archive_dir, "exists.csv")
        self.assertTrue(SecureOpen.file_exists(record_exists_path))

    def test_get_namelist_valid_and_not_found(self):
        """
        Test get_namelist static method for a valid ZIP archive and for a missing ZIP.

        - Create a ZIP with two entries and verify get_namelist returns both names.
        - Call get_namelist on a non-existent ZIP path and expect FileNotFoundError.
        """
        archive_dir = os.path.join(self.temp_dir, "archive6")
        os.makedirs(archive_dir, exist_ok=True)
        zip_path = os.path.join(archive_dir, "archive6.zip")
        with pyzipper.AESZipFile(zip_path, "w", compression=pyzipper.ZIP_DEFLATED) as zf:
            zf.writestr("file1.txt", b"1")
            zf.writestr("file2.txt", b"2")

        namelist = SecureOpen.get_namelist(
            os.path.join(archive_dir, "file1.txt"), "archive6"
        )
        self.assertIn("file1.txt", namelist)
        self.assertIn("file2.txt", namelist)

        with self.assertRaises(FileNotFoundError):
            SecureOpen.get_namelist(
                os.path.join(self.temp_dir, "nonexistent", "no.zip"), "nozip"
            )


if __name__ == "__main__":
    unittest.main()
