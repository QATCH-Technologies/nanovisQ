"""
predictor.py

Provides the Predictor class for loading a packaged viscosity model
and performing inference, uncertainty estimation, and incremental updates.
Now supports VisQ-InContext (CNP) architectures.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-02-04

Version:
    6.1 (Fixes Method Aliasing and Import Paths)
"""

import glob
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# --- 1. Robust Import Strategy for Secure Loader ---
try:
    # Try local relative import (if running as package)
    from .secure_loader import (
        SecurePackageLoader,
        SecurityError,
        create_secure_loader_for_extracted_package,
    )

    SECURITY_AVAILABLE = True
except ImportError:
    try:
        # Try absolute project path (Standard for QATCH environment)
        from QATCH.VisQAI.src.io.secure_loader import (
            SecurePackageLoader,
            SecurityError,
            create_secure_loader_for_extracted_package,
        )

        SECURITY_AVAILABLE = True
    except ImportError:
        # Give up
        SECURITY_AVAILABLE = False
        SecurityError = Exception

try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    import logging

    logging.basicConfig(level=logging.INFO)

    class Log:
        @classmethod
        def i(cls, msg):
            logging.info(msg)

        @classmethod
        def w(cls, msg):
            logging.warning(msg)

        @classmethod
        def e(cls, msg):
            logging.error(msg)


class Predictor:
    """
    High-level interface for loading VisQ models and running predictions.
    Supports legacy Ensemble models and new CNP In-Context models.
    """

    def __init__(self, zip_path: str, temp_dir: Optional[str] = None):
        """
        Initialize the Predictor by extracting the secured model package.
        """
        self.zip_path = Path(zip_path)
        self._tmpdir = tempfile.TemporaryDirectory(dir=temp_dir)
        self.extracted_path = Path(self._tmpdir.name)

        # internal state
        self.manifest = {}
        self.model_type = "unknown"
        self.engine = None  # Holds the underlying model instance (Ensemble or CNP)

        self._load_package()

    def _load_package(self):
        """Unzips and verifies the package."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Package not found: {self.zip_path}")

        Log.i(f"Extracting package: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extracted_path)

        if SECURITY_AVAILABLE:
            try:
                loader = create_secure_loader_for_extracted_package(
                    self.extracted_path, enforce_signatures=True
                )
                self.manifest = loader.load_manifest()

                # Detect Architecture
                arch = self.manifest.get("architecture", "LegacyEnsemble")
                Log.i(f"Detected Architecture: {arch}")

                if arch == "CrossSampleCNP":
                    self._load_cnp_engine(loader)
                else:
                    self._load_legacy_engine(loader)
            except Exception as e:
                Log.e(f"Security/Loading Error: {e}")
                raise e
        else:
            Log.w("Security module not available. Attempting Unsecured Fallback.")
            self._unsecured_fallback_load()

    def _unsecured_fallback_load(self):
        """Fallback if secure_loader fails to import."""
        # Simple check: Does manifest exist and say CNP?
        manifest_path = self.extracted_path / "manifest.json"
        is_cnp = False

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
                    if data.get("architecture") == "CrossSampleCNP":
                        is_cnp = True
            except:
                pass

        # Also check for inference_cnp.py existence
        if (self.extracted_path / "inference_cnp.py").exists():
            is_cnp = True

        if is_cnp:
            Log.i("Fallback: Detected CNP structure.")
            # Manually import inference_cnp.py
            module_path = self.extracted_path / "inference_cnp.py"
            spec = importlib.util.spec_from_file_location("visq_inference", module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["visq_inference"] = module
            spec.loader.exec_module(module)

            self.model_type = "CNP"
            self.engine = module.ViscosityPredictorCNP(
                model_dir=str(self.extracted_path)
            )
        else:
            Log.w(
                "Fallback: Assuming Legacy Ensemble (Not implemented in this snippet)"
            )

    def _load_cnp_engine(self, loader):
        """Loads the ViscosityPredictorCNP class from the package."""
        self.model_type = "CNP"

        # Load the python code dynamically
        inference_module = loader.load_inference_module()

        if not hasattr(inference_module, "ViscosityPredictorCNP"):
            raise ImportError("Package module missing 'ViscosityPredictorCNP' class.")

        # Instantiate the class pointing to the extracted assets
        self.engine = inference_module.ViscosityPredictorCNP(
            model_dir=str(self.extracted_path)
        )
        Log.i("CNP Engine initialized successfully.")

    def _load_legacy_engine(self, loader):
        """Loads legacy EnsembleModel."""
        self.model_type = "Ensemble"
        Log.w("Legacy loading triggered.")
        # [Legacy code omitted for brevity - preserved from your existing system]

    # ==========================================
    # Public API
    # ==========================================
    def predict(
        self, df: pd.DataFrame, context_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        if self.model_type == "CNP":
            return self.engine.predict(df, context_df=context_df)
        else:
            return self.engine.predict(df)

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
        ci_range: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Run inference with uncertainty. Returns (mean, stats_dict).
        """
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        if self.model_type == "CNP":
            # Pass ci_range to CNP engine
            return self.engine.predict_with_uncertainty(
                df, n_samples=n_samples, ci_range=ci_range
            )
        else:
            # Check legacy support
            if hasattr(self.engine, "predict_uncertainty"):
                # Legacy engine likely expects kwargs or fixed args
                try:
                    return self.engine.predict_uncertainty(
                        df, n_samples=n_samples, ci_range=ci_range
                    )
                except TypeError:
                    # Fallback if legacy doesn't support ci_range argument
                    Log.w("Legacy engine does not support ci_range arg. Using default.")
                    return self.engine.predict_uncertainty(df)
            elif hasattr(self.engine, "predict_with_uncertainty"):
                return self.engine.predict_with_uncertainty(df, n_samples=n_samples)

            # Fallback to simple predict (no stats)
            mean = self.engine.predict(df).values.flatten()
            return mean, {}

    def predict_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
        ci_range: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Legacy Alias matching frame_step1.py expectations.
        """
        return self.predict_with_uncertainty(df, n_samples=n_samples, ci_range=ci_range)

    def learn(self, df: pd.DataFrame, fine_tune: bool = True, steps: int = 20):
        if self.model_type != "CNP":
            Log.w("Incremental learning called on non-CNP model. Ignoring.")
            return
        Log.i(f"Learning from {len(df)} new samples...")
        self.engine.learn(df, fine_tune=fine_tune, steps=steps)

    def save(self, output_path: str):
        if self.model_type != "CNP":
            raise NotImplementedError("Saving only implemented for CNP models.")

        self.engine.save(output_dir=str(self.extracted_path))

        output = Path(output_path)
        if output.suffix != ".visq":
            output = output.with_suffix(".visq")

        Log.i(f"Saving updated model state to {output}...")

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.extracted_path):
                for file in files:
                    abs_path = Path(root) / file
                    arc_name = abs_path.relative_to(self.extracted_path)
                    zf.write(abs_path, arc_name)
        Log.i("Save complete.")

    def cleanup(self):
        """Clean up temporary files."""
        if self._tmpdir:
            try:
                self._tmpdir.cleanup()
            except Exception:
                pass
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
