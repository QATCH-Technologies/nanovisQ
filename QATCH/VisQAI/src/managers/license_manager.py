import json
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import dropbox
from dropbox.exceptions import ApiError


from QATCH.common.logger import Logger as Log
from QATCH.common.deviceFingerprint import DeviceFingerprint

TAG = "[LicenseManager]"


class LicenseStatus:
    """Enum-like class for license statuses"""
    ADMIN = "admin"
    ACTIVE = "active"
    TRIAL = "trial"
    INACTIVE = "inactive"


class LicenseManager:

    def __init__(self, dropbox_token: str, license_directory: str = "/visqai-licenses/active-licenses"):
        self.dbx = dropbox.Dropbox(dropbox_token)
        self.license_directory = license_directory
        self.device_key = DeviceFingerprint.generate_key()
        self.license_filename = f"{self.device_key}.json"
        self.license_filepath = f"{self.license_directory}/{self.license_filename}"

    def _ensure_directory_exists(self) -> bool:
        try:
            self.dbx.files_get_metadata(self.license_directory)
            return True
        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                Log.e(TAG,
                      f"Remote license directory does not exist: {self.license_directory}")
                return False
            Log.e(TAG, f"Error checking remote directory: {e}")
            return False

    def _list_license_files(self) -> List[str]:
        try:
            result = self.dbx.files_list_folder(self.license_directory)
            files = []

            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.json'):
                    files.append(entry.name)
            while result.has_more:
                result = self.dbx.files_list_folder_continue(result.cursor)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith('.json'):
                        files.append(entry.name)

            return files
        except ApiError as e:
            Log.e(f"Error listing license files: {e}")
            return []

    def _download_license_file(self) -> Optional[Dict]:
        try:
            _, response = self.dbx.files_download(self.license_filepath)
            content = response.content.decode('utf-8')
            return json.loads(content)
        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                Log.e(
                    f"License file not found: {self.license_filename}")
            else:
                Log.e(f"Error downloading license file: {e}")
            return None
        except json.JSONDecodeError as e:
            Log.e(f"Error parsing license file: {e}")
            return None

    def validate_license(self) -> Tuple[bool, str, Dict]:
        if not self._ensure_directory_exists():
            return False, "License directory not accessible", {}
        license_data = self._download_license_file()
        if license_data is None:
            available_files = self._list_license_files()
            Log.i(f"Available license files: {available_files}")
            return False, f"No license file found for device: {self.license_filename}", {}

        # Verify the key in the file matches this device
        file_key = license_data.get('license_key', '')
        if file_key != self.device_key:
            return False, f"License key mismatch. File key: {file_key}, Device key: {self.device_key}", {}

        # Check license status
        status = license_data.get('status', LicenseStatus.INACTIVE)

        if status == LicenseStatus.INACTIVE:
            return False, "License is inactive", license_data

        if status == LicenseStatus.ADMIN:
            return True, "Admin license - always valid", license_data

        # Check expiration for trial and active licenses
        if status in [LicenseStatus.TRIAL, LicenseStatus.ACTIVE]:
            expiration_str = license_data.get('expiration')
            if not expiration_str:
                return False, "License has no expiration date set", license_data

            try:
                expiration_date = datetime.fromisoformat(expiration_str)
                now = datetime.now()

                if now > expiration_date:
                    days_expired = (now - expiration_date).days
                    return False, f"License expired {days_expired} days ago", license_data
                else:
                    days_remaining = (expiration_date - now).days
                    hours_remaining = (
                        (expiration_date - now).seconds // 3600) % 24

                    if days_remaining == 0:
                        return True, f"License valid - expires in {hours_remaining} hours", license_data
                    else:
                        return True, f"License valid - {days_remaining} days remaining", license_data

            except (ValueError, TypeError) as e:
                Log.e(f"Error parsing expiration date: {e}")
                return False, "Invalid expiration date format", license_data

        return False, f"Unknown license status: {status}", license_data
