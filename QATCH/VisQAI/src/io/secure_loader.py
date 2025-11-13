"""
secure_loader.py

Provides cryptographic signature verification for packaged predictor modules.
Works alongside predictor.py to add an additional security layer.

# TODO: This system does not protect against inauthentic keypairs only against 
# direct injection into the current model system.  I need to add an official certificate
# to sign the public key with to protect  against inauthentic key-pair packages.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-11-12

Version:
    1.0
"""
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, List
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

try:
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

        @classmethod
        def d(cls, msg: str) -> None:
            """Log a debug message."""
            cls._logger.debug(msg)


class SecurityError(Exception):
    """Raised when security verification fails."""
    pass


class SignatureVerifier:
    """Handles cryptographic signature verification for package files."""

    def __init__(self, public_key_pem: bytes):
        """
        Initialize the signature verifier.

        Args:
            public_key_pem: PEM-encoded public key bytes
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

            Log.d(f"Signature verified: {identifier}")
            return True

        except InvalidSignature:
            raise SecurityError(
                f"SECURITY VIOLATION: Invalid signature for {identifier}. "
                "This file may have been tampered with or is from an untrusted source."
            )
        except Exception as e:
            raise SecurityError(
                f"Signature verification error for {identifier}: {e}"
            )


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
        self.enforce_signatures = enforce_signatures
        self.require_all_signed = require_all_signed

        self.public_key = None
        self.signatures = {}
        self.verified_files = set()
        self.verifier = None

        if self.enforce_signatures:
            Log.d("Secure package loader initialized (signature verification ENABLED)")
            self._load_security_data()
        else:
            Log.w(
                " WARNING: Signature verification DISABLED - use only for debugging!")

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

        with open(signatures_path, 'r') as f:
            self.signatures = json.load(f)
        self.verifier = SignatureVerifier(public_key_pem)

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
