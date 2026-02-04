"""
secure_loader.py

Provides cryptographic signature verification for packaged predictor modules.
Supports both legacy file-level signing and new manifest-level signing (CNP models).

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-02-04

Version:
    2.0 (Updated for Manifest-First Security)
"""

import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    import logging

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""

        _logger = logging.getLogger("SecureLoader")

        @classmethod
        def i(cls, msg: str) -> None:
            cls._logger.info(msg)

        @classmethod
        def w(cls, msg: str) -> None:
            cls._logger.warning(msg)

        @classmethod
        def e(cls, msg: str) -> None:
            cls._logger.error(msg)


class SecurityError(Exception):
    """Raised when signature verification or integrity checks fail."""

    pass


class SecureModuleLoader:
    """Verifies and loads Python modules from a secured source directory."""

    def __init__(self, enforce_signatures: bool = True):
        self.enforce_signatures = enforce_signatures

    def verify_manifest_signature(
        self, manifest_path: Path, signature_path: Path, public_key_pem: bytes
    ) -> None:
        """
        Verifies that the manifest.json was signed by the provided public key.
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
            Log.i("Manifest signature verified successfully.")

        except InvalidSignature:
            raise SecurityError(
                "Manifest signature verification FAILED. Package integrity compromised."
            )
        except Exception as e:
            raise SecurityError(f"Error checking manifest signature: {str(e)}")

    def verify_file_integrity(self, file_path: Path, expected_hash: str) -> None:
        """Calculates SHA256 of file and compares to expected hash."""
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
                f"❌ Hash mismatch for {file_path.name}!\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {calculated_hash}"
            )
        Log.i(f"   Integrity check passed: {file_path.name}")

    def load_module_from_path(self, module_name: str, file_path: Path):
        """Dynamically imports a Python module from a file path."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        raise ImportError(f"Could not load module {module_name} from {file_path}")


class SecurePackageLoader:
    """Orchestrates loading for an extracted package directory."""

    def __init__(self, extracted_dir: Path, enforce_signatures: bool = True):
        self.root = extracted_dir
        self.enforce_signatures = enforce_signatures
        self.loader = SecureModuleLoader(enforce_signatures)
        self.manifest_data = {}

    def load_manifest(self) -> Dict[str, Any]:
        """Loads and verifies the manifest."""
        manifest_path = self.root / "manifest.json"
        sig_path = self.root / "manifest.sig"

        if not manifest_path.exists():
            raise SecurityError("Package missing manifest.json")

        with open(manifest_path, "r") as f:
            self.manifest_data = json.load(f)

        # 1. Verify Manifest Signature (if present)
        # Note: New packages put public_key inside manifest.
        # In a strict environment, public_key should be loaded from a trusted local store.
        # Here we trust the key in the manifest for integrity checking (TOFU model).
        if sig_path.exists():
            pub_key = self.manifest_data.get("public_key")
            if pub_key:
                self.loader.verify_manifest_signature(
                    manifest_path, sig_path, pub_key.encode("utf-8")
                )
            else:
                Log.w(
                    "Manifest has signature but no public_key entry. Skipping sig check."
                )
        elif self.enforce_signatures:
            Log.w("No manifest.sig found. Skipping signature verification.")

        # 2. Verify File Integrity (if 'files' section exists - New Format)
        if "files" in self.manifest_data:
            Log.i("Verifying package asset integrity...")
            files_map = self.manifest_data["files"]
            for filename, meta in files_map.items():
                # Some entries might be the inference code, others models
                file_path = self.root / filename
                if "sha256" in meta:
                    self.loader.verify_file_integrity(file_path, meta["sha256"])

        return self.manifest_data

    def load_inference_module(self, module_filename: str = None):
        """
        Loads the entry point python file.
        If module_filename is None, attempts to find it in manifest.
        """
        if not module_filename:
            # Try to guess from manifest files
            files_map = self.manifest_data.get("files", {})
            for fname, meta in files_map.items():
                if meta.get("type") == "inference_code":
                    module_filename = fname
                    break

        if not module_filename:
            raise ValueError("Could not determine inference module filename.")

        module_path = self.root / module_filename
        if not module_path.exists():
            raise FileNotFoundError(f"Inference module {module_filename} not found.")

        Log.i(f"Loading inference logic from: {module_filename}")
        return self.loader.load_module_from_path("visq_inference", module_path)


def create_secure_loader_for_extracted_package(
    extracted_dir: str, enforce_signatures: bool = True, require_all_signed: bool = True
) -> SecurePackageLoader:
    return SecurePackageLoader(Path(extracted_dir), enforce_signatures)
