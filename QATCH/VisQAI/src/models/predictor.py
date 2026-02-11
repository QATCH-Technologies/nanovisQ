"""
predictor.py

Provides the Predictor class for loading a packaged viscosity model
and performing inference.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-02-09

Version:
    6.5 (Relative Path Search Fix)
"""

import datetime
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ==========================================
# Logger Setup
# ==========================================
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

        @classmethod
        def d(cls, msg):
            logging.debug(msg)


# ==========================================
# Security Imports
# ==========================================
try:
    from .secure_loader import (
        SecurePackageLoader,
        SecurityError,
        create_secure_loader_for_extracted_package,
    )

    SECURITY_AVAILABLE = True
except ImportError:
    try:
        from QATCH.VisQAI.src.io.secure_loader import (
            SecurePackageLoader,
            SecurityError,
            create_secure_loader_for_extracted_package,
        )

        SECURITY_AVAILABLE = True
    except ImportError:
        SECURITY_AVAILABLE = False
        SecurityError = Exception


class Predictor:
    """
    High-level interface for loading VisQ models.
    """

    def __init__(self, zip_path: str, temp_dir: Optional[str] = None):
        self.zip_path = Path(zip_path)
        self._tmpdir = tempfile.TemporaryDirectory(dir=temp_dir)
        self.extracted_path = Path(self._tmpdir.name)

        self.manifest = {}
        self.model_type = "unknown"
        self.engine = None

        self._load_package()

    def _load_package(self):
        """Unzips and dynamically loads the engine."""
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Package not found: {self.zip_path}")

        Log.i(f"Extracting package: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extracted_path)

        # ---------------------------------------------------------
        # 1. ROBUST PATH SEARCH for Local Code
        # ---------------------------------------------------------
        # Instead of guessing "src/...", we look relative to THIS file (predictor.py).
        current_dir = Path(__file__).resolve().parent

        candidate_paths = [
            # 1. Same directory (e.g. if everything is in src/)
            current_dir / "inference_o_net.py",
            # 2. Parent directory (e.g. if predictor is in src/io/)
            current_dir.parent / "inference_o_net.py",
            # 3. Two levels up + src (e.g. if predictor is in src/utils/)
            current_dir.parent.parent / "src" / "inference_o_net.py",
            # 4. Standard Project Root assumption
            Path("src/inference_o_net.py").resolve(),
        ]

        local_code_path = None
        for p in candidate_paths:
            if p.exists():
                local_code_path = p
                break

        # Define where the ZIPPED (stale) code lives
        extracted_code_path = self.extracted_path / "inference_o_net.py"
        inference_module = None

        if local_code_path and local_code_path.exists():
            inference_module = self._dynamic_load_from_path(
                "visq_inference_local", local_code_path
            )
        elif extracted_code_path.exists():
            inference_module = self._dynamic_load_from_path(
                "visq_inference_pkg", extracted_code_path
            )
        else:
            Log.w("No inference logic found. Attempting legacy fallback.")

        # 2. Initialize Engine
        if inference_module and hasattr(inference_module, "ViscosityPredictorCNP"):
            self.model_type = "CNP"
            # Initialize the class with the path to the WEIGHTS (extracted zip)
            # This uses the NEW code (inference_module) with the OLD weights (extracted_path)
            self.engine = inference_module.ViscosityPredictorCNP(
                model_dir=str(self.extracted_path)
            )
            Log.i("CNP Engine initialized successfully.")
            return

        # 3. Fallback to Secure Loader / Legacy if manual dynamic load failed/skipped
        if SECURITY_AVAILABLE:
            try:
                loader = create_secure_loader_for_extracted_package(
                    self.extracted_path, enforce_signatures=True
                )
                self.manifest = loader.load_manifest()
                arch = self.manifest.get("architecture", "LegacyEnsemble")

                if arch == "CrossSampleCNP":
                    # We already tried loading above, but if secure loader does something special:
                    mod = loader.load_inference_module()
                    self.model_type = "CNP"
                    self.engine = mod.ViscosityPredictorCNP(
                        model_dir=str(self.extracted_path)
                    )
                else:
                    self._load_legacy_engine(loader)
            except Exception as e:
                Log.e(f"Loader Error: {e}")
                raise e
        else:
            Log.e("Could not load inference engine.")

    def _dynamic_load_from_path(self, module_name, file_path):
        """Helper to load a python source file by absolute path."""
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                return module
        except Exception as e:
            Log.e(f"Failed to dynamically load {file_path}: {e}")
        return None

    def _load_legacy_engine(self, loader):
        self.model_type = "Ensemble"
        # ... legacy code ...
        pass

    # ==========================================
    # Public API
    # ==========================================
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.engine is None:
            raise RuntimeError("Engine not initialized")
        return self.engine.predict(df)

    def predict_with_uncertainty(
        self,
        df: pd.DataFrame,
        n_samples: int = 20,
        ci_range: Tuple[float, float] = (2.5, 97.5),
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        if self.model_type == "CNP":
            return self.engine.predict_with_uncertainty(
                df, n_samples=n_samples, ci_range=ci_range
            )
        else:
            # Legacy shim
            if hasattr(self.engine, "predict_with_uncertainty"):
                return self.engine.predict_with_uncertainty(df, n_samples=n_samples)

            mean = self.engine.predict(df).values.flatten()
            return mean, {}

    def predict_uncertainty(self, *args, **kwargs):
        return self.predict_with_uncertainty(*args, **kwargs)

    def learn(self, df: pd.DataFrame, fine_tune: bool = True, steps: int = 50):
        if self.model_type != "CNP":
            return
        Log.i(f"Learning from {len(df)} new samples...")
        self.engine.learn(df, steps=50, lr=1e-3)

    def reset(self):
        """
        Resets the model to its original state from the zip package.
        This effectively discards any fine-tuning performed via learn().
        """
        Log.i("Resetting model state to original package version...")
        if self.extracted_path.exists():
            for child in self.extracted_path.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)

        self.engine = None
        self._load_package()

    def save(self, output_path: str):
        if self.model_type != "CNP":
            raise NotImplementedError("Only CNP supports save.")

        if hasattr(self.engine, "save"):
            self.engine.save(output_dir=str(self.extracted_path))

        # We manually zip the EXTRACTED folder, which contains updated weights
        output = Path(output_path).with_suffix(".visq")
        Log.i(f"Saving updated model state to {output}...")

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(self.extracted_path):
                for file in files:
                    abs_path = Path(root) / file
                    arc_name = abs_path.relative_to(self.extracted_path)
                    zf.write(abs_path, arc_name)
        Log.i("Save complete.")

    def cleanup(self):
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
