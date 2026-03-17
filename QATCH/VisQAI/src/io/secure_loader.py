"""
secure_loader.py

<<<<<<< HEAD
A security-conscious module loader designed to verify the integrity of
external model packages. It uses RSA-PSS signature verification and
SHA256 hashing to ensure that manifest files and binary assets have
not been tampered with. It supports dynamic importing of Python
inference logic while allowing configurable security enforcement levels.
=======
Provides cryptographic signature verification for packaged predictor modules.
Works alongside predictor.py to add an additional security layer.

# TODO: This system does not protect against inauthentic keypairs only against 
# direct injection into the current model system.  I need to add an official certificate
# to sign the public key with to protect  against inauthentic key-pair packages.
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
<<<<<<< HEAD
    2026-03-16

Version:
    2.0
=======
    2025-11-12

Version:
    1.0
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
"""
import json
import sys
import importlib.util
from pathlib import Path
<<<<<<< HEAD
from typing import Any, Dict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
=======
from typing import Dict, Any, List
import base64
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

try:
<<<<<<< HEAD
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
=======
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""
        _logger = logging.getLogger("SecureLoader")

        @classmethod
        def i(cls, msg: str) -> None:
            """Log an informational message."""
            cls._logger.info(msg)

        @classmethod
        def w(cls, msg: str) -> None:
            """Log a warning message."""
            cls._logger.warning(msg)

        @classmethod
        def e(cls, msg: str) -> None:
            """Log an error message."""
            cls._logger.error(msg)
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143

        @classmethod
        def d(cls, msg: str) -> None:
            """Log a debug message."""
            cls._logger.debug(msg)


class SecurityError(Exception):
<<<<<<< HEAD
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
=======
    """Raised when security verification fails."""
    pass


class SignatureVerifier:
    """Handles cryptographic signature verification for package files."""

    def __init__(self, public_key_pem: bytes):
        """
        Initialize the signature verifier.

        Args:
            public_key_pem: PEM-encoded public key bytes
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
        """
        self.public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        Log.d("Signature verifier initialized")

    def verify_file(self, file_path: Path, expected_signature: str) -> bool:
        """
        Verify the signature of a single file.

        Args:
            file_path: Path to file to verify
            expected_signature: Base64-encoded expected signature

        Returns:
            True if signature is valid

        Raises:
            SecurityError if signature verification fails
        """
        with open(file_path, 'rb') as f:
            content = f.read()

        return self.verify_bytes(content, expected_signature, str(file_path))

    def verify_bytes(self, content: bytes, expected_signature: str, identifier: str = "content") -> bool:
        """
        Verify the signature of raw bytes.

        Args:
            content: Bytes to verify
            expected_signature: Base64-encoded expected signature
            identifier: Name/path for error messages

        Returns:
            True if signature is valid

        Raises:
            SecurityError if signature verification fails
        """
        try:
            signature_bytes = base64.b64decode(expected_signature)
            self.public_key.verify(
                signature_bytes,
                content,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
<<<<<<< HEAD
            Log.i(TAG, "Manifest signature verified successfully.")
=======

            Log.d(f"Signature verified: {identifier}")
            return True
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143

        except InvalidSignature:
            raise SecurityError(
                f"SECURITY VIOLATION: Invalid signature for {identifier}. "
                "This file may have been tampered with or is from an untrusted source."
            )
        except Exception as e:
<<<<<<< HEAD
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
=======
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
            raise SecurityError(
                f"Signature verification error for {identifier}: {e}"
            )
<<<<<<< HEAD
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
=======


class SecurePackageLoader:
    """
    Verifies cryptographic signatures of packaged predictor files.

    This is a separate loader class for debugging and can be used
    independently or integrated with the main Predictor class.
    """

    def __init__(
        self,
        extracted_dir: Path,
        enforce_signatures: bool = True,
        require_all_signed: bool = True
    ):
        """
        Initialize the secure package loader.

        Args:
            extracted_dir: Path to already-extracted package directory
            enforce_signatures: Whether to enforce signature verification
                               (should always be True in production!)
            require_all_signed: Whether ALL source files must have signatures
                               (recommended: True)
        """
        self.extracted_dir = Path(extracted_dir)
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
        self.enforce_signatures = enforce_signatures
        self.require_all_signed = require_all_signed

<<<<<<< HEAD
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
=======
        self.public_key = None
        self.signatures = {}
        self.verified_files = set()
        self.verifier = None
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143

        if self.enforce_signatures:
            Log.d("Secure package loader initialized (signature verification ENABLED)")
            self._load_security_data()
        else:
            Log.w(
                " WARNING: Signature verification DISABLED - use only for debugging!")

<<<<<<< HEAD
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
=======
    def _load_security_data(self) -> None:
        """Load public key and signatures manifest from security directory."""
        security_dir = self.extracted_dir / 'security'
        if not security_dir.exists():
            if self.enforce_signatures:
                raise SecurityError(
                    " No 'security/' directory found in package. "
                    "This package is not signed and cannot be loaded with "
                    "enforce_signatures=True. Either use a signed package or "
                    "disable signature verification (NOT recommended for production)."
                )
            else:
                Log.w("No security directory found - proceeding without verification")
                return
        public_key_path = security_dir / 'public_key.pem'
        if not public_key_path.exists():
            raise SecurityError(
                " No public key found in security/ directory. "
                "This package may be corrupted."
            )

        with open(public_key_path, 'rb') as f:
            public_key_pem = f.read()
        signatures_path = security_dir / 'signatures.json'
        if not signatures_path.exists():
            raise SecurityError(
                " No signatures manifest found in security/ directory. "
                "This package may be corrupted."
            )
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143

        with open(signatures_path, 'r') as f:
            self.signatures = json.load(f)
        self.verifier = SignatureVerifier(public_key_pem)

<<<<<<< HEAD
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
=======
        Log.d(f"Loaded public key and {len(self.signatures)} signatures")

    def verify_metadata(self, metadata_path: Path) -> bool:
        """
        Verify the metadata.json file signature.

        Args:
            metadata_path: Path to metadata.json

        Returns:
            True if verified (or verification disabled)

        Raises:
            SecurityError if verification fails
        """
        if not self.enforce_signatures:
            return True
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143

        arcname = 'model/metadata.json'

        if arcname not in self.signatures:
            if self.require_all_signed:
                raise SecurityError(
                    f"No signature found for {arcname}. "
                    "This package may be incomplete or corrupted."
                )
            else:
                Log.w(f" No signature for {arcname} - proceeding anyway")
                return True

<<<<<<< HEAD
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
=======
        expected_signature = self.signatures[arcname]
        self.verifier.verify_file(metadata_path, expected_signature)
        self.verified_files.add(arcname)

        return True

    def verify_source_file(self, src_file: Path, relative_name: str) -> bool:
        """
        Verify a single source file signature before it's loaded.

        Args:
            src_file: Path to the source file
            relative_name: Relative name (e.g., 'inference.py')

        Returns:
            True if verified (or verification disabled)

        Raises:
            SecurityError if verification fails
        """
        if not self.enforce_signatures:
            return True

        arcname = f'src/{relative_name}'

        if arcname not in self.signatures:
            if self.require_all_signed:
                raise SecurityError(
                    f"SECURITY VIOLATION: No signature found for {arcname}. "
                    f"Cannot load this module. This could be a code injection attempt."
                )
            else:
                Log.w(f" No signature for {arcname} - proceeding anyway")
                return True

        expected_signature = self.signatures[arcname]
        self.verifier.verify_file(src_file, expected_signature)
        self.verified_files.add(arcname)

        Log.d(f"Module verified: {relative_name}")
        return True

    def verify_all_sources(self, src_dir: Path, source_files: List[str]) -> Dict[str, bool]:
        """
        Verify all source files in batch.

        Args:
            src_dir: Directory containing source files
            source_files: List of source filenames to verify

        Returns:
            Dictionary mapping filenames to verification status

        Raises:
            SecurityError if any verification fails
        """
        results = {}

        Log.d(f"\nVerifying {len(source_files)} source files...")

        for filename in source_files:
            filepath = src_dir / filename
            if not filepath.exists():
                Log.w(f" Source file not found: {filename}")
                results[filename] = False
                continue

            try:
                self.verify_source_file(filepath, filename)
                results[filename] = True
            except SecurityError as e:
                Log.e(str(e))
                raise  # Re-raise to halt loading

        verified_count = sum(results.values())
        Log.d(f"Verified {verified_count}/{len(source_files)} source files")

        return results

    def get_verification_report(self) -> Dict[str, Any]:
        """
        Get a report of verification status.

        Returns:
            Dictionary with verification statistics
        """
        return {
            'enforcement_enabled': self.enforce_signatures,
            'require_all_signed': self.require_all_signed,
            'total_signatures': len(self.signatures),
            'files_verified': len(self.verified_files),
            'verified_files': list(self.verified_files),
            'unverified_signed_files': [
                sig for sig in self.signatures.keys()
                if sig not in self.verified_files
            ]
        }


class SecureModuleLoader:
    """
    Secure wrapper for loading Python modules with signature verification.

    Integrates with the existing Predictor._load_source_modules workflow.
    """

    def __init__(
        self,
        secure_loader: SecurePackageLoader,
        src_dir: Path
    ):
        """
        Initialize the secure module loader.

        Args:
            secure_loader: SecurePackageLoader instance for verification
            src_dir: Directory containing source files
        """
        self.secure_loader = secure_loader
        self.src_dir = Path(src_dir)
        self.loaded_modules = {}

    def load_module_secure(
        self,
        module_name: str,
        verify_first: bool = True
    ) -> Any:
        """
        Load a module with signature verification.

        Args:
            module_name: Name of the module
            verify_first: Whether to verify signature before loading

        Returns:
            Loaded module object

        Raises:
            SecurityError if verification fails
            RuntimeError if module file not found
        """
        filename = f"{module_name}.py"
        filepath = self.src_dir / filename

        if not filepath.exists():
            raise RuntimeError(f"Module file not found: {filepath}")

        if verify_first and self.secure_loader.enforce_signatures:
            self.secure_loader.verify_source_file(filepath, filename)

        spec = importlib.util.spec_from_file_location(
            module_name, str(filepath))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.loaded_modules[module_name] = module
        Log.d(f"Loaded module: {module_name}")

        return module

    def load_all_modules_secure(
        self,
        module_names: List[str],
        verify_batch: bool = True
    ) -> Dict[str, Any]:
        """
        Load multiple modules with batch verification.

        Args:
            module_names: List of module names to load
            verify_batch: Whether to verify all signatures first

        Returns:
            Dict mapping module names to loaded modules

        Raises:
            SecurityError if any verification fails
        """
        if verify_batch and self.secure_loader.enforce_signatures:
            filenames = [f"{name}.py" for name in module_names]
            self.secure_loader.verify_all_sources(self.src_dir, filenames)
        for module_name in module_names:
            verify = not verify_batch
            self.load_module_secure(module_name, verify_first=verify)

        return self.loaded_modules


def create_secure_loader_for_extracted_package(
    extracted_dir: str,
    enforce_signatures: bool = True,
    require_all_signed: bool = True
) -> SecurePackageLoader:
    """
    Factory function to create a secure loader for an already-extracted package.

    Args:
        extracted_dir: Path to extracted package directory
        enforce_signatures: Whether to enforce signature verification
        require_all_signed: Whether all files must have signatures

    Returns:
        SecurePackageLoader instance
    """
    return SecurePackageLoader(
        extracted_dir=Path(extracted_dir),
        enforce_signatures=enforce_signatures,
        require_all_signed=require_all_signed
    )
>>>>>>> c8b8db9a73c06821c07e683989b6114d95b0f143
