"""
lincenseManager.py

This module provides a comprehensive license management solution that handles trial licenses,
license validation, and caching with Dropbox as the remote storage backend.

Author: 
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date: 
    2025-09-02
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from enum import Enum
import dropbox
from dropbox.exceptions import ApiError
from pathlib import Path
import hashlib
import threading

import pymysql
import cryptography
import zipfile
import base64

try:
    from QATCH.common.logger import Logger as Log
    from QATCH.common.deviceFingerprint import DeviceFingerprint
except:
    print("Running standalone, or import of main app logging and/or other modules failed.")

TAG = "[LicenseManager]"


class LicenseServer(Enum):
    DROPBOX = "dbx"
    AIVENIO = "avn"


USE_SERVER = LicenseServer.AIVENIO


class LicenseStatus(Enum):
    """Constants for license status types.

    Attributes:
        ADMIN (str): Administrator license with unlimited access.
        ACTIVE (str): Active paid license.
        TRIAL (str): Trial license with expiration.
        INACTIVE (str): Inactive or suspended license.
    """
    ADMIN = "admin"
    ACTIVE = "active"
    TRIAL = "trial"
    INACTIVE = "inactive"


class AVN_TEST:

    def __init__(self):

        timeout = 10
        connection = pymysql.connect(**AVN_TEST.load_avn_key_store())

        try:
            cursor = connection.cursor()
            # cursor.execute("CREATE TABLE mytest (id INTEGER PRIMARY KEY)")
            # cursor.execute("INSERT INTO mytest (id) VALUES (1), (2)")
            cursor.execute("SELECT * FROM subscribers")
            print(cursor.fetchall())
            cursor.execute("SELECT * FROM licenses")
            print(cursor.fetchall())
        finally:
            connection.commit()
            connection.close()

    @staticmethod
    def load_avn_key_store():
        DB_CONFIG = {}
        with zipfile.ZipFile("QATCH/resources/avn_key_store.zip", 'r') as zip_key:
            pem_file = zip_key.read("db_config.pem").splitlines()
            pem_file[0] = b""  # remove begin line
            pem_file[-1] = b""  # remove end line
            pem_file[1] = pem_file[1][4:]  # remove "AVN_"
            pem_file = b"".join(pem_file)
            DB_CONFIG = json.loads(base64.b64decode(pem_file).decode()[::2])
            DB_CONFIG['cursorclass'] = pymysql.cursors.DictCursor
        return DB_CONFIG


class LicenseCache:
    """Handles local caching of license data for improved performance.

    This class manages local filesystem caching of license information to reduce
    network requests and improve application responsiveness.
    """

    def __init__(self, cache_dir: str = None, cache_duration_hours: int = 24):
        """Initialize the license cache.

        Args:
            cache_dir (str, optional): Directory for cache files. Defaults to system 
                temp directory with app-specific subdirectory.
            cache_duration_hours (int, optional): How long cache is valid in hours. 
                Defaults to 24.
        """
        if cache_dir is None:
            # Use system temp directory with app-specific subdirectory
            import tempfile
            cache_dir = os.path.join(
                tempfile.gettempdir(), 'qatch_license_cache')

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_duration = timedelta(hours=cache_duration_hours)

    def _get_cache_filepath(self, device_key: str) -> Path:
        """Generate cache file path for a device key.

        Args:
            device_key (str): Unique device identifier.

        Returns:
            Path: Path to the cache file for the given device.
        """
        # Hash the device key for privacy in local cache
        key_hash = hashlib.sha256(device_key.encode()).hexdigest()[:16]
        return self.cache_dir / f"license_cache_{key_hash}.json"

    def save(self, device_key: str, license_data: Dict) -> bool:
        """Save license data to cache.

        Args:
            device_key (str): Unique device identifier.
            license_data (Dict): License information to cache.

        Returns:
            bool: True if successfully cached, False otherwise.
        """
        try:
            cache_filepath = self._get_cache_filepath(device_key)
            cache_data = {
                'license_data': license_data,
                'cached_at': datetime.now().isoformat(),
                'cache_version': '1.0'
            }

            with open(cache_filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)

            Log.i(TAG, f"License cached locally at {cache_filepath}")
            return True

        except Exception as e:
            Log.e(TAG, f"Failed to cache license: {e}")
            return False

    def load(self, device_key: str, ignore_expiry: bool = False) -> Tuple[Optional[Dict], bool]:
        """Load license data from cache.

        Args:
            device_key (str): Unique device identifier.
            ignore_expiry (bool, optional): Whether to ignore cache expiration. 
                Defaults to False.

        Returns:
            Tuple[Optional[Dict], bool]: A tuple containing:
                - license_data: The cached license or None if not found
                - is_expired: True if cache exists but is expired
        """
        try:
            cache_filepath = self._get_cache_filepath(device_key)

            if not cache_filepath.exists():
                Log.i(TAG, "No cached license found")
                return None, False

            with open(cache_filepath, 'r') as f:
                cache_data = json.load(f)

            # Check cache age
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            age = datetime.now() - cached_at
            is_expired = age > self.cache_duration

            if is_expired and not ignore_expiry:
                Log.i(
                    TAG, f"Cache expired (age: {age.total_seconds()/3600:.1f} hours)")
                return cache_data['license_data'], True

            Log.i(
                TAG, f"Using cached license (age: {age.total_seconds()/3600:.1f} hours)")
            return cache_data['license_data'], is_expired

        except Exception as e:
            Log.e(TAG, f"Failed to load cached license: {e}")
            return None, False

    def get_cache_age(self, device_key: str) -> Optional[timedelta]:
        """Get the age of the cached data.

        Args:
            device_key (str): Unique device identifier.

        Returns:
            Optional[timedelta]: Age of cached data or None if cache doesn't exist.
        """
        try:
            cache_filepath = self._get_cache_filepath(device_key)
            if not cache_filepath.exists():
                return None

            with open(cache_filepath, 'r') as f:
                cache_data = json.load(f)

            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            return datetime.now() - cached_at

        except Exception:
            return None

    def clear(self, device_key: str) -> bool:
        """Clear cached license data.

        Args:
            device_key (str): Unique device identifier.

        Returns:
            bool: True if successfully cleared, False otherwise.
        """
        try:
            cache_filepath = self._get_cache_filepath(device_key)
            if cache_filepath.exists():
                cache_filepath.unlink()
                Log.i(TAG, "Cache cleared")
            return True
        except Exception as e:
            Log.e(TAG, f"Failed to clear cache: {e}")
            return False


class LicenseManager:
    """Manages license validation, creation, and caching with Dropbox backend.

    This class provides a comprehensive license management system that supports:
    - Trial license auto-registration
    - License validation with caching
    - Background cache refresh
    - License extension
    - Remote storage via Dropbox

    Attributes:
        _TRIAL_PERIOD (int): Default trial period in days (90).
    """

    _TRIAL_PERIOD = 90

    def __init__(self,
                 dbx_conn: dropbox.Dropbox,
                 license_directory: str = "/device-licenses",
                 auto_register_trial: bool = True,
                 trial_duration_days: int = _TRIAL_PERIOD,
                 cache_enabled: bool = True,
                 cache_duration_hours: int = 24,
                 cache_dir: str = None,
                 background_refresh: bool = True):
        """Initialize License Manager with cache-first approach.

        Args:
            dbx_conn (dropbox.Dropbox): Dropbox connection instance.
            license_directory (str, optional): Remote directory for licenses. 
                Defaults to "/device-licenses".
            auto_register_trial (bool, optional): Auto-create trial licenses for new devices. 
                Defaults to True.
            trial_duration_days (int, optional): Trial period duration in days. 
                Defaults to _TRIAL_PERIOD (90).
            cache_enabled (bool, optional): Enable local caching. Defaults to True.
            cache_duration_hours (int, optional): How long cache is valid in hours. 
                Defaults to 24.
            cache_dir (str, optional): Local cache directory. None uses system temp. 
                Defaults to None.
            background_refresh (bool, optional): Refresh cache in background when expired. 
                Defaults to True.
        """
        self.dbx = dbx_conn
        self.license_directory = license_directory
        self.device_key = DeviceFingerprint.generate_key()
        self.device_summary = DeviceFingerprint.get_device_summary()
        self.license_filename = f"{self.device_key}.json"
        self.license_filepath = f"{self.license_directory}/{self.license_filename}"
        self.auto_register_trial = auto_register_trial
        self.trial_duration_days = trial_duration_days

        # Caching configuration
        self.cache_enabled = cache_enabled
        self.background_refresh = background_refresh
        self.cache = LicenseCache(
            cache_dir, cache_duration_hours) if cache_enabled else None

        # Track refresh status
        self._refresh_thread = None
        self._last_refresh_attempt = None

    def _ensure_directory_exists(self) -> bool:
        """Check if remote license directory exists and create it if needed.

        Returns:
            bool: True if directory exists or was created successfully, False otherwise.
        """
        try:
            self.dbx.files_get_metadata(self.license_directory)
            return True
        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                try:
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

    def _download_license_from_remote(self) -> Optional[Dict]:
        """Download license file from Dropbox without checking cache.

        Returns:
            Optional[Dict]: License data if found and valid, None otherwise.
        """
        try:
            _, response = self.dbx.files_download(self.license_filepath)
            content = response.content.decode('utf-8')
            license_data = json.loads(content)

            Log.i(TAG, "Successfully downloaded license from remote")

            # Update cache with fresh data
            if self.cache_enabled:
                self.cache.save(self.device_key, license_data)

            return license_data

        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                Log.i(
                    TAG, f"License file not found remotely: {self.license_filename}")
            else:
                Log.e(TAG, f"Error downloading license file: {e}")
            return None

        except json.JSONDecodeError as e:
            Log.e(TAG, f"Error parsing license file: {e}")
            return None

    def _refresh_cache_background(self):
        """Background thread to refresh expired cache.

        This method runs in a separate thread to update cache without blocking
        the main application flow.
        """
        Log.i(TAG, "Starting background cache refresh")
        self._last_refresh_attempt = datetime.now()

        try:
            license_data = self._download_license_from_remote()
            if license_data:
                Log.i(TAG, "Background cache refresh successful")
            else:
                Log.w(TAG, "Background cache refresh failed")
        except Exception as e:
            Log.e(TAG, f"Error in background refresh: {e}")
        finally:
            self._refresh_thread = None

    def _get_license_data(self, force_remote: bool = False) -> Tuple[Optional[Dict], str]:
        """Get license data using cache-first approach.

        Args:
            force_remote (bool, optional): Skip cache and go directly to remote. 
                Defaults to False.

        Returns:
            Tuple[Optional[Dict], str]: A tuple containing:
                - license_data: The license data or None if not found
                - source: Data source ('cache', 'remote', 'expired_cache', or 'none')
        """
        # Force remote check if requested
        if force_remote:
            Log.i(TAG, "Forced remote license check")
            data = self._download_license_from_remote()
            return data, 'remote' if data else 'none'

        # ALWAYS check cache first if enabled
        if self.cache_enabled:
            cached_data, is_expired = self.cache.load(self.device_key)

            if cached_data is not None:
                if not is_expired:
                    # Valid cache - return immediately for responsiveness
                    Log.i(TAG, "Returning valid cached license")
                    return cached_data, 'cache'
                else:
                    # Cache exists but expired
                    Log.i(TAG, "Cache expired, will refresh")

                    # Start background refresh if enabled and not already running
                    if self.background_refresh and self._refresh_thread is None:
                        self._refresh_thread = threading.Thread(
                            target=self._refresh_cache_background,
                            daemon=True
                        )
                        self._refresh_thread.start()

                    # Return expired cache for immediate response
                    # The background thread will update it
                    return cached_data, 'expired_cache'

        # No cache available, must check remote
        Log.i(TAG, "No cache available, checking remote")
        data = self._download_license_from_remote()
        return data, 'remote' if data else 'none'

    def _upload_license_file(self, license_data: Dict) -> bool:
        """Upload a license file to Dropbox.

        Args:
            license_data (Dict): License data to upload.

        Returns:
            bool: True if upload successful, False otherwise.
        """
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

            # Update cache with new license data
            if self.cache_enabled:
                self.cache.save(self.device_key, license_data)

            return True

        except ApiError as e:
            Log.e(TAG, f"Error uploading license file: {e}")
            return False

    def register(self, additional_info: Dict = None) -> Tuple[bool, str, Dict]:
        """Register a new trial license for the current device.

        Args:
            additional_info (Dict, optional): Additional information to include 
                in the license. Defaults to None.

        Returns:
            Tuple[bool, str, Dict]: A tuple containing:
                - success: True if registration successful
                - message: Human-readable status message
                - license_data: The created license data (empty dict if failed)
        """
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

    def validate_license(self,
                         auto_create_if_missing: Optional[bool] = None,
                         force_remote: bool = False) -> Tuple[bool, str, Dict]:
        """Validate license with cache-first approach.

        Args:
            auto_create_if_missing (Optional[bool], optional): Override auto-registration 
                setting. Uses instance setting if None. Defaults to None.
            force_remote (bool, optional): Force remote check, bypassing cache. 
                Defaults to False.

        Returns:
            Tuple[bool, str, Dict]: A tuple containing:
                - is_valid: True if license is valid
                - message: Human-readable status message with source info
                - license_data: The license data (empty dict if no license)
        """

        # Get license data (cache-first unless forced)
        license_data, source = self._get_license_data(
            force_remote=force_remote)

        # Add source info to message suffix
        source_msg = ""
        if source == 'cache':
            source_msg = " (cached)"
        elif source == 'expired_cache':
            cache_age = self.cache.get_cache_age(self.device_key)
            if cache_age:
                hours_old = cache_age.total_seconds() / 3600
                source_msg = f" (cached {hours_old:.1f}h old, refreshing)"
        elif source == 'remote':
            source_msg = " (fresh)"

        # If no license exists, need to create one
        if license_data is None:
            # For new licenses, we need to ensure directory exists
            if not self._ensure_directory_exists():
                return False, "Cannot access license directory and no cached license available", {}

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
                return False, f"No license file found for device: {self.license_filename}", {}

        # Verify the key in the file matches this device
        file_key = license_data.get('license_key', '')
        if file_key != self.device_key:
            # Clear bad cache
            if self.cache_enabled:
                self.cache.clear(self.device_key)
            return False, f"License key mismatch. File key: {file_key}, Device key: {self.device_key}", {}

        # Check license status
        status = license_data.get('status', LicenseStatus.INACTIVE)

        if status == LicenseStatus.INACTIVE:
            return False, f"License is inactive{source_msg}", license_data

        if status == LicenseStatus.ADMIN:
            return True, f"Admin license - always valid{source_msg}", license_data

        # Check expiration for trial and active licenses
        if status in [LicenseStatus.TRIAL, LicenseStatus.ACTIVE]:
            expiration_str = license_data.get('expiration')
            if not expiration_str:
                return False, f"License has no expiration date set{source_msg}", license_data

            try:
                expiration_date = datetime.fromisoformat(expiration_str)
                now = datetime.now()

                if now > expiration_date:
                    days_expired = (now - expiration_date).days
                    return False, f"License expired {days_expired} days ago{source_msg}", license_data
                else:
                    days_remaining = (expiration_date - now).days
                    hours_remaining = (
                        (expiration_date - now).seconds // 3600) % 24

                    if days_remaining == 0:
                        return True, f"License valid - expires in {hours_remaining} hours{source_msg}", license_data
                    else:
                        return True, f"License valid - {days_remaining} days remaining{source_msg}", license_data

            except (ValueError, TypeError) as e:
                Log.e(TAG, f"Error parsing expiration date: {e}")
                return False, f"Invalid expiration date format{source_msg}", license_data

        return False, f"Unknown license status: {status}{source_msg}", license_data

    def extend_license(self, additional_days: int) -> Tuple[bool, str, Dict]:
        """Extend license expiration by specified number of days.

        This operation requires a remote connection and will update both remote
        storage and local cache.

        Args:
            additional_days (int): Number of days to extend the license.

        Returns:
            Tuple[bool, str, Dict]: A tuple containing:
                - success: True if extension successful
                - message: Human-readable status message
                - license_data: Updated license data (original data if failed)
        """
        # Force remote check for extension
        license_data, source = self._get_license_data(force_remote=True)

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

    def get_license_info(self, prefer_cache: bool = True) -> Optional[Dict]:
        """Get license information.

        Args:
            prefer_cache (bool, optional): Use cache if available. Set to False 
                to force remote check. Defaults to True.

        Returns:
            Optional[Dict]: License data or None if not found.
        """
        data, _ = self._get_license_data(force_remote=not prefer_cache)
        return data

    def clear_cache(self) -> bool:
        """Clear local license cache.

        Returns:
            bool: True if cache cleared successfully, False if caching disabled 
                or clear failed.
        """
        if self.cache_enabled:
            return self.cache.clear(self.device_key)
        return False

    def refresh_license(self) -> Tuple[bool, str, Dict]:
        """Force refresh license from remote source.

        Returns:
            Tuple[bool, str, Dict]: Same as validate_license() with forced remote check.
        """
        Log.i(TAG, "Forcing license refresh from remote")
        return self.validate_license(force_remote=True)

    def get_cache_status(self) -> Dict:
        """Get information about cache status and configuration.

        Returns:
            Dict: Cache status information including:
                - enabled: Whether caching is enabled
                - cache_exists: Whether cache file exists (if enabled)
                - cache_path: Path to cache file (if enabled)
                - background_refresh: Whether background refresh is enabled
                - refresh_thread_active: Whether refresh thread is currently running
                - last_refresh_attempt: ISO timestamp of last refresh attempt
                - cache_age_hours: Age of cache in hours (if exists)
                - cache_expired: Whether cache is expired (if exists)
        """
        if not self.cache_enabled:
            return {'enabled': False}

        cache_filepath = self.cache._get_cache_filepath(self.device_key)
        status = {
            'enabled': True,
            'cache_exists': cache_filepath.exists(),
            'cache_path': str(cache_filepath),
            'background_refresh': self.background_refresh,
            'refresh_thread_active': self._refresh_thread is not None,
            'last_refresh_attempt': self._last_refresh_attempt.isoformat() if self._last_refresh_attempt else None
        }

        # Get cache age
        cache_age = self.cache.get_cache_age(self.device_key)
        if cache_age:
            status['cache_age_hours'] = cache_age.total_seconds() / 3600
            status['cache_expired'] = cache_age > self.cache.cache_duration

        return status

    def wait_for_background_refresh(self, timeout: float = 5.0) -> bool:
        """Wait for background refresh to complete.

        Args:
            timeout (float, optional): Maximum seconds to wait. Defaults to 5.0.

        Returns:
            bool: True if refresh completed within timeout, False if timeout occurred.
        """
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout)
            return not self._refresh_thread.is_alive()
        return True


# Example usage
if __name__ == "__main__":
    if USE_SERVER == LicenseServer.AIVENIO:
        AVN_TEST()

    else:

        # Initialize with cache-first approach
        license_mgr = LicenseManager(
            dropbox_token="YOUR_DROPBOX_TOKEN",
            auto_register_trial=True,
            trial_duration_days=90,
            cache_enabled=True,           # Enable caching
            cache_duration_hours=24,       # Cache valid for 24 hours
            background_refresh=True        # Refresh in background when expired
        )

        # First call - checks cache first, only goes online if no cache or expired
        # If cache is expired, returns expired data immediately and refreshes in background
        is_valid, message, license_data = license_mgr.validate_license()
        print(f"License valid: {is_valid}")
        # Will show (cached) or (cached Xh old, refreshing)
        print(f"Message: {message}")

        # Subsequent calls within cache period are instant
        is_valid, message, license_data = license_mgr.validate_license()
        print(f"License valid: {is_valid}")
        print(f"Message: {message}")  # Will show (cached)

        # Force fresh check from remote (only when needed)
        is_valid, message, license_data = license_mgr.refresh_license()
        print(f"Fresh license valid: {is_valid}")

        # Check cache status
        cache_status = license_mgr.get_cache_status()
        print(f"Cache status: {json.dumps(cache_status, indent=2)}")

        # Wait for background refresh if one is running
        if license_mgr.wait_for_background_refresh(timeout=5.0):
            print("Background refresh completed")
        else:
            print("Background refresh still running")
