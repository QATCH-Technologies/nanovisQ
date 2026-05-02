import os
import zipfile
import csv
import io

from datetime import datetime, timedelta


from PyQt5.QtCore import (
    QThread,
    pyqtSignal,
)

from typing import BinaryIO, cast

from QATCH.common.logger import Logger as Log
from QATCH.common.fileStorage import secure_open

TAG = "[ScanWorker]"


class ScanWorker(QThread):
    """Worker thread that scans the unnamed-runs directory without blocking the UI.

    This thread iterates through a specified directory to find and analyze folders
    containing run data. It parses metadata from internal ZIP/CSV files and
    calculates directory sizes on disk. Results are returned via signals to ensure
    the main GUI thread remains responsive during file I/O.

    Attributes:
        scan_complete (pyqtSignal): Emitted when the scan finishes successfully.
            Carries a list of RunMetadata objects.
        scan_failed (pyqtSignal): Emitted if a critical error occurs during the
            scan. Carries an error message string.
        unnamed_dir (str): The directory path to be scanned.
        _is_cancelled (bool): Internal flag to stop the thread execution prematurely.
    """

    scan_complete = pyqtSignal(list)
    scan_failed = pyqtSignal(str)

    def __init__(self, unnamed_dir):
        """Initializes the ScanWorker with the target directory.

        Args:
            unnamed_dir: The absolute path to the directory containing
                unnamed run folders.
        """
        super().__init__()
        self.unnamed_dir = unnamed_dir
        self._is_cancelled = False

    def cancel(self):
        """Sets the cancellation flag to abort the scanning process."""
        self._is_cancelled = True

    def run(self):
        """Executes the directory scan and metadata extraction.

        Iterates through the directory, identifies run folders, extracts
        CSV metadata, calculates file sizes, computes start/stop times,
        and compiles a list of RunMetadata objects.
        """

        runs = []
        try:
            if not os.path.exists(self.unnamed_dir):
                self.scan_complete.emit(runs)
                return

            for filename in os.listdir(self.unnamed_dir):
                if self._is_cancelled:
                    return

                filepath = os.path.join(self.unnamed_dir, filename)
                if not os.path.isdir(filepath):
                    continue

                if filename.endswith("_BAD"):
                    ruling = "Bad"
                    display_name = filename[:-4]
                else:
                    ruling = "Good"
                    display_name = filename

                (
                    duration,
                    num_points,
                    csv_timestamp,
                    virtual_csv_path,
                ) = self._extract_metadata(filepath)

                # Calculate File Size
                try:
                    total_size = sum(
                        os.path.getsize(os.path.join(filepath, f))
                        for f in os.listdir(filepath)
                        if os.path.isfile(os.path.join(filepath, f))
                    )
                except OSError:
                    total_size = 0
                file_size_mb = round(total_size / (1024 * 1024), 2)

                # Calculate ISO Start and Stop Times
                start_iso = "UNKNOWN"
                stop_iso = "UNKNOWN"

                if csv_timestamp == "Unknown":
                    try:
                        stop_dt = datetime.fromtimestamp(os.stat(filepath).st_mtime)
                        start_dt = stop_dt - timedelta(seconds=float(duration))
                        start_iso = start_dt.isoformat(timespec="seconds")
                        stop_iso = stop_dt.isoformat(timespec="seconds")
                    except OSError:
                        pass
                else:
                    try:
                        if "T" in csv_timestamp:
                            start_dt = datetime.fromisoformat(csv_timestamp)
                        else:
                            start_dt = datetime.strptime(csv_timestamp, "%Y-%m-%d %H:%M:%S")

                        stop_dt = start_dt + timedelta(seconds=float(duration))

                        start_iso = start_dt.isoformat(timespec="seconds")
                        stop_iso = stop_dt.isoformat(timespec="seconds")
                    except (ValueError, TypeError):
                        start_iso = csv_timestamp  # Fallback to raw string if parsing fails

                runs.append(
                    RunMetadata(
                        filepath=filepath,
                        display_name=display_name,
                        start=start_iso,
                        stop=stop_iso,
                        duration=duration,
                        samples=num_points,
                        ruling=ruling,
                        file_size_mb=file_size_mb,
                        virtual_csv_path=virtual_csv_path,
                    )
                )

            self.scan_complete.emit(runs)
        except Exception as e:
            self.scan_failed.emit(str(e))

    @staticmethod
    def _extract_metadata(folderpath):
        """Extracts experimental metadata from files within a run folder.

        Locates a ZIP file within the folder, finds the primary data CSV
        inside that ZIP, and parses it to determine the run duration,
        point count, and start timestamp.

        Args:
            folderpath: The path to the specific run directory to analyze.

        Returns:
            A tuple containing:
                - duration (float): The maximum relative time found in the data.
                - num_points (int): The total number of data rows.
                - timestamp (str): The start date/time string or "Unknown".
                - virtual_csv_path (str | None): The path to the CSV data
                  source, or None if not found.
        """
        duration = 0.0
        num_points = 0
        timestamp = "Unknown"
        virtual_csv_path = None

        try:
            zip_filename = next(
                (name for name in os.listdir(folderpath) if name.lower().endswith(".zip")),
                None,
            )
            if not zip_filename:
                return duration, num_points, timestamp, None

            zip_filepath = os.path.join(folderpath, zip_filename)

            csv_filename = None
            with zipfile.ZipFile(zip_filepath, "r") as z:
                csv_filename = next(
                    (
                        name
                        for name in z.namelist()
                        if name.endswith(".csv") and not name.split("/")[-1].startswith("._")
                    ),
                    None,
                )
            if not csv_filename:
                return duration, num_points, timestamp, None

            virtual_csv_path = os.path.join(folderpath, csv_filename)

            with secure_open(virtual_csv_path, "r") as f:
                binary_f = cast(BinaryIO, f)

                text_io = io.TextIOWrapper(binary_f, encoding="utf-8-sig")
                reader = csv.reader(text_io)

                header_row = None
                rel_time_idx = -1
                date_idx = -1
                time_idx = -1
                max_time = 0.0

                for row in reader:
                    if not row or not any(row):
                        continue

                    if header_row is None:
                        cleaned_row = [str(col).strip() for col in row]
                        if "Relative_time" in cleaned_row or "Date" in cleaned_row:
                            header_row = cleaned_row
                            rel_time_idx = (
                                header_row.index("Relative_time")
                                if "Relative_time" in header_row
                                else -1
                            )
                            date_idx = header_row.index("Date") if "Date" in header_row else -1
                            time_idx = header_row.index("Time") if "Time" in header_row else -1
                        continue

                    num_points += 1

                    if (
                        num_points == 1
                        and date_idx != -1
                        and time_idx != -1
                        and len(row) > max(date_idx, time_idx)
                    ):
                        date_str = str(row[date_idx]).strip()
                        time_str = str(row[time_idx]).strip()
                        if date_str and time_str:
                            timestamp = f"{date_str} {time_str}"

                    if rel_time_idx != -1 and len(row) > rel_time_idx:
                        try:
                            rel_time = float(str(row[rel_time_idx]).strip())
                            if rel_time > max_time:
                                max_time = rel_time
                        except (ValueError, TypeError):
                            pass

                duration = round(max_time, 2)
        except RuntimeError as e:
            if "Bad password" in str(e) or "password" in str(e).lower():
                Log.w(TAG, f"Encrypted file, skipping metadata for {folderpath}")
            else:
                Log.e(TAG, f"Failed to extract metadata from {folderpath}: {str(e)}")
        except Exception as e:
            Log.e(TAG, f"Failed to extract metadata from {folderpath}: {str(e)}")

        return duration, num_points, timestamp, virtual_csv_path
