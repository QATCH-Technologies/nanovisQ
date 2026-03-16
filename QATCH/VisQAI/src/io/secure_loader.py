"""
secure_loader.py

A security-conscious module loader designed to verify the integrity of
external model packages. It uses RSA-PSS signature verification and
SHA256 hashing to ensure that manifest files and binary assets have
not been tampered with. It supports dynamic importing of Python
inference logic while allowing configurable security enforcement levels.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    2.0
"""

import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

try:
    TAG = "[SecureLoader (HEADLESS)]"

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

except (ImportError, ModuleNotFoundError):
    TAG = "[SecureLoader]"
    from QATCH.common.logger import Logger as Log


class SecurityError(Exception):
    """Raised when signature verification or integrity checks fail.

    This exception is used throughout the secure loading process to signal that
    a file has been tampered with, a signature is invalid, or a required
    security component is missing.
    """

    pass


class SecureModuleLoader:
    """Handles low-level cryptographic verification and dynamic module loading.

    This class provides utilities for verifying RSA-PSS signatures, calculating
    SHA256 checksums for file integrity, and safely importing Python modules
    from arbitrary filesystem paths.

    Attributes:
        enforce_signatures (bool): Whether to perform cryptographic checks.
            If False, verification methods return early without error.
    """

    def __init__(self, enforce_signatures: bool = True):
        """Initializes the loader with a specific security policy.

        Args:
            enforce_signatures: If True, all verification methods will execute
                strict checks. Defaults to True.
        """
        self.enforce_signatures = enforce_signatures

    def verify_manifest_signature(
        self, manifest_path: Path, signature_path: Path, public_key_pem: bytes
    ) -> None:
        """Verifies the RSA-PSS signature of a manifest file.

        Uses the provided public key to ensure the manifest data matches the
        detached signature. This confirms that the manifest has not been
        modified since it was signed by a trusted authority.

        Args:
            manifest_path: Path to the JSON manifest file.
            signature_path: Path to the detached signature file (.sig).
            public_key_pem: The public key in PEM format as bytes.

        Raises:
            SecurityError: If files are missing, the public key is malformed,
                or the signature verification fails (indicating tampering).
        """
        if not self.enforce_signatures:
            return

        if not manifest_path.exists():
            raise SecurityError(f"Manifest not found: {manifest_path}")
        if not signature_path.exists():
            raise SecurityError(f"Manifest signature not found: {signature_path}")

        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            with open(manifest_path, "rb") as f:
                manifest_data = f.read()

            with open(signature_path, "rb") as f:
                signature = base64.b64decode(f.read())

            public_key.verify(
                signature,
                manifest_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            Log.i(TAG, "Manifest signature verified successfully.")

        except InvalidSignature:
            raise SecurityError(
                "Manifest signature verification FAILED. Package integrity compromised."
            )
        except Exception as e:
            raise SecurityError(f"Error checking manifest signature: {str(e)}")

    def verify_file_integrity(self, file_path: Path, expected_hash: str) -> None:
        """Verifies a file's integrity using SHA256.

        Calculates the hash of the file in 4KB chunks and compares it against
        the provided expected hex string.

        Args:
            file_path: The path to the file to be checked.
            expected_hash: The pre-calculated SHA256 hex string.

        Raises:
            SecurityError: If the file is missing or the calculated hash
                does not match the expected value.
        """
        if not self.enforce_signatures:
            return

        if not file_path.exists():
            raise SecurityError(f"Required file missing: {file_path}")

        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):
                digest.update(chunk)

        calculated_hash = digest.finalize().hex()

        if calculated_hash != expected_hash:
            raise SecurityError(
                f"Hash mismatch for {file_path.name}!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {calculated_hash}"
            )
        Log.i(TAG, f"   Integrity check passed: {file_path.name}")

    def load_module_from_path(self, module_name: str, file_path: Path):
        """Dynamically imports a Python module from a specific file path.

        Creates a module spec, creates a new module object, and executes it
        within the current interpreter's sys.modules.

        Args:
            module_name: The name to assign to the imported module.
            file_path: The filesystem path to the .py file.

        Returns:
            module: The successfully loaded Python module.

        Raises:
            ImportError: If the module cannot be initialized or loaded.
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        raise ImportError(f"Could not load module {module_name} from {file_path}")


class SecurePackageLoader:
    """Orchestrates the verification and loading of an extracted package.

    This class high-levels the loading process by reading the manifest,
    triggering signature and integrity checks, and finally importing the
    required inference modules.

    Note:
        Current implementation downgrades signature and integrity failures
        to warnings to support user-imported (non-official) models.

    Attributes:
        root (Path): The root directory of the extracted package.
        enforce_signatures (bool): Security policy for the loader.
        loader (SecureModuleLoader): The worker instance for crypto operations.
        manifest_data (dict): Cached dictionary of the manifest contents.
    """

    def __init__(self, extracted_dir: Path, enforce_signatures: bool = True):
        """Initializes the package loader.

        Args:
            extracted_dir: The directory where the package files reside.
            enforce_signatures: Whether to perform security checks.
        """
        self.root = extracted_dir
        self.enforce_signatures = enforce_signatures
        self.loader = SecureModuleLoader(enforce_signatures)
        self.manifest_data = {}

    def load_manifest(self) -> Dict[str, Any]:
        """Loads and verifies the package manifest and asset integrity.

        Validates the manifest.json file against a manifest.sig (if present)
        using a public key listed in the manifest. If a 'files' section is
        found, it also verifies the hash of every asset listed.

        Returns:
            Dict[str, Any]: The loaded manifest data.

        Raises:
            SecurityError: If the manifest.json file itself is missing.
        """
        manifest_path = self.root / "manifest.json"
        sig_path = self.root / "manifest.sig"

        if not manifest_path.exists():
            raise SecurityError("Package missing manifest.json")

        with open(manifest_path, "r") as f:
            self.manifest_data = json.load(f)

        # Verify Manifest Signature
        if sig_path.exists():
            pub_key = self.manifest_data.get("public_key")
            if pub_key:
                try:
                    self.loader.verify_manifest_signature(
                        manifest_path, sig_path, pub_key.encode("utf-8")
                    )
                except (SecurityError, Exception) as e:
                    Log.w(TAG, f"Signature Check Failed (Ignored): {e}")
            else:
                Log.w(
                    TAG,
                    "Manifest has signature but no public_key entry. Skipping sig check.",
                )
        elif self.enforce_signatures:
            Log.w(TAG, "No manifest.sig found. Skipping signature verification.")

        # Verify File Integrity
        if "files" in self.manifest_data:
            Log.i(TAG, "Verifying package asset integrity...")
            files_map = self.manifest_data["files"]
            for filename, meta in files_map.items():
                file_path = self.root / filename
                if "sha256" in meta:
                    try:
                        self.loader.verify_file_integrity(file_path, meta["sha256"])
                    except (SecurityError, Exception) as e:
                        Log.w(TAG, f"Integrity Check Failed (Ignored): {e}")

        return self.manifest_data

    def load_inference_module(self, module_filename: str = None):
        """Imports the primary inference logic from the package.

        If a filename is not provided, the method scans the manifest for a file
        of type 'inference_code'.

        Args:
            module_filename: Optional name of the python file to load.

        Returns:
            module: The imported inference module.

        Raises:
            ValueError: If no inference module can be identified in the manifest.
            FileNotFoundError: If the identified module file does not exist.
        """
        if not module_filename:
            files_map = self.manifest_data.get("files", {})
            for fname, meta in files_map.items():
                if "inference" in meta.get("type"):
                    module_filename = fname
                    break

        if not module_filename:
            raise ValueError("Could not determine inference module filename.")

        module_path = self.root / module_filename
        if not module_path.exists():
            raise FileNotFoundError(f"Inference module {module_filename} not found.")

        Log.i(TAG, f"Loading inference logic from: {module_filename}")
        return self.loader.load_module_from_path("visq_inference", module_path)


def create_secure_loader_for_extracted_package(
    extracted_dir: str, enforce_signatures: bool = True
) -> SecurePackageLoader:
    """Factory function to create a SecurePackageLoader instance.

    Args:
        extracted_dir: Path to the directory containing package assets.
        enforce_signatures: Whether to perform cryptographic checks.

    Returns:
        SecurePackageLoader: A prepared loader instance for the package.
    """
    return SecurePackageLoader(Path(extracted_dir), enforce_signatures)
