import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import dropbox
from dropbox.exceptions import ApiError


from QATCH.common.logger import Logger as Log
from QATCH.common.deviceFingerprint import DeviceFingerprint

TAG = "[LicenseManager]"


class LicenseStatus:
    ADMIN = "admin"
    ACTIVE = "active"
    TRIAL = "trial"
    INACTIVE = "inactive"


class LicenseManager:
    _TRIAL_PERIOD = 90

    def __init__(self, dbx_conn: dropbox.Dropbox, license_directory: str = "device-licenses",
                 auto_register_trial: bool = True, trial_duration_days: int = _TRIAL_PERIOD):
        self.dbx = dbx_conn
        self.license_directory = license_directory
        self.device_key = DeviceFingerprint.generate_key()
        self.device_summary = DeviceFingerprint.get_device_summary()
        self.license_filename = f"{self.device_key}.json"
        self.license_filepath = f"{self.license_directory}/{self.license_filename}"
        self.auto_register_trial = auto_register_trial
        self.trial_duration_days = trial_duration_days

    def _ensure_directory_exists(self) -> bool:
        try:
            self.dbx.files_get_metadata(self.license_directory)
            return True
        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                try:
                    # Try to create the directory
                    Log.i(
                        TAG, f"Creating license directory: {self.license_directory}")
                    self.dbx.files_create_folder_v2(self.license_directory)
                    return True
                except ApiError as create_error:
                    Log.e(
                        TAG, f"Failed to create license directory: {create_error}")
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
            Log.e(TAG, f"Error listing license files: {e}")
            return []

    def _download_license_file(self) -> Optional[Dict]:
        """Download and parse the license file for this device"""
        try:
            _, response = self.dbx.files_download(self.license_filepath)
            content = response.content.decode('utf-8')
            return json.loads(content)
        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                Log.i(TAG, f"License file not found: {self.license_filename}")
            else:
                Log.e(TAG, f"Error downloading license file: {e}")
            return None
        except json.JSONDecodeError as e:
            Log.e(TAG, f"Error parsing license file: {e}")
            return None

    def _upload_license_file(self, license_data: Dict) -> bool:
        """Upload a license file to Dropbox"""
        try:
            content = json.dumps(license_data, indent=2).encode('utf-8')
            self.dbx.files_upload(
                content,
                self.license_filepath,
                mode=dropbox.files.WriteMode('overwrite'),
                autorename=False
            )
            Log.i(
                TAG, f"Successfully uploaded license file: {self.license_filename}")
            return True
        except ApiError as e:
            Log.e(TAG, f"Error uploading license file: {e}")
            return False

    def register(self, additional_info: Dict = None) -> Tuple[bool, str, Dict]:
        creation_date = datetime.now()
        expiration_date = creation_date + \
            timedelta(days=self.trial_duration_days)

        license_data = {
            'license_key': self.device_key,
            'status': LicenseStatus.TRIAL,
            'creation_date': creation_date.isoformat(),
            'expiration': expiration_date.isoformat(),
            'trial_days': self.trial_duration_days,
            'auto_generated': True,
            'device_info': self.device_summary
        }
        if additional_info:
            license_data['additional_info'] = additional_info
        if self._upload_license_file(license_data):
            return True, f"Trial license created successfully (expires in {self.trial_duration_days} days)", license_data
        else:
            return False, "Failed to create trial license", {}

    def validate_license(self, auto_create_if_missing: Optional[bool] = None) -> Tuple[bool, str, Dict]:
        if not self._ensure_directory_exists():
            return False, "License directory not accessible", {}

        license_data = self._download_license_file()

        # If no license exists and auto-registration is enabled
        if license_data is None:
            should_auto_register = (
                auto_create_if_missing
                if auto_create_if_missing is not None
                else self.auto_register_trial
            )

            if should_auto_register:
                Log.i(
                    TAG, f"No license found for device {self.device_key}. Creating trial license...")
                success, message, new_license_data = self.register()
                if success:
                    return True, message, new_license_data
                else:
                    return False, f"Failed to auto-register trial: {message}", {}
            else:
                available_files = self._list_license_files()
                Log.i(TAG, f"Available license files: {available_files}")
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
                Log.e(TAG, f"Error parsing expiration date: {e}")
                return False, "Invalid expiration date format", license_data

        return False, f"Unknown license status: {status}", license_data

    def extend_license(self, additional_days: int) -> Tuple[bool, str, Dict]:
        license_data = self._download_license_file()
        if license_data is None:
            return False, "No license file found to extend", {}

        status = license_data.get('status', LicenseStatus.INACTIVE)
        if status not in [LicenseStatus.TRIAL, LicenseStatus.ACTIVE]:
            return False, f"Cannot extend license with status: {status}", license_data

        try:
            current_expiration = datetime.fromisoformat(
                license_data.get('expiration'))
            # If already expired, extend from today; otherwise extend from current expiration
            base_date = max(current_expiration, datetime.now())
            new_expiration = base_date + timedelta(days=additional_days)

            license_data['expiration'] = new_expiration.isoformat()
            license_data['last_extended'] = datetime.now().isoformat()
            license_data['extension_days'] = additional_days

            if self._upload_license_file(license_data):
                days_remaining = (new_expiration - datetime.now()).days
                return True, f"License extended by {additional_days} days. New total: {days_remaining} days remaining", license_data
            else:
                return False, "Failed to update license file", license_data

        except (ValueError, TypeError) as e:
            Log.e(TAG, f"Error extending license: {e}")
            return False, "Error processing license extension", license_data

    def get_license_info(self) -> Optional[Dict]:
        return self._download_license_file()


# Example usage
if __name__ == "__main__":
    # Initialize with auto-registration enabled
    license_mgr = LicenseManager(
        dropbox_token="YOUR_DROPBOX_TOKEN",
        auto_register_trial=True,  # Enable automatic trial registration
        trial_duration_days=90      # 90-day trial period
    )

    # Validate license (will auto-create trial if none exists)
    is_valid, message, license_data = license_mgr.validate_license()
    print(f"License valid: {is_valid}")
    print(f"Message: {message}")

    # You can also manually create a trial license with additional info
    success, msg, data = license_mgr.register(
        additional_info={
            'user_email': 'user@example.com',
            'app_version': '1.0.0'
        }
    )

    # Or disable auto-registration for a specific validation
    is_valid, message, license_data = license_mgr.validate_license(
        auto_create_if_missing=False)
