"""
Module: predictor.py

Provides the Predictor class for loading a packaged viscosity model
and performing inference, uncertainty estimation, and incremental updates.

Now includes integrated cryptographic signature verification to prevent
code injection attacks.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-11-12

Version:
    4.0
"""
import os
import zipfile
import tempfile
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import hashlib
import pyzipper
import zipfile

import numpy as np
import pandas as pd
import importlib.util

# Security imports
try:
    from QATCH.VisQAI.src.io.secure_loader import (
        create_secure_loader_for_extracted_package,
        SecureModuleLoader,
        SecurityError
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    SecurityError = Exception  # Fallback

try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    import logging
    from src.io.secure_loader import (
        create_secure_loader_for_extracted_package,
        SecureModuleLoader,
        SecurityError
    )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""
        _logger = logging.getLogger("Predictor")

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


class Predictor:
    """
    Predictor for loading a packaged VisQAI-base model and performing
    predictions and updates.

    This class extracts a zip archive containing source files and model checkpoint,
    loads required modules dynamically, and instantiates a Predictor for inference
    and incremental updates.

    Security: By default, this class verifies cryptographic signatures of all source
    files before loading them, preventing code injection attacks. This can be disabled
    for debugging by setting verify_signatures=False.
    """
    VISC_100 = "Viscosity_100"
    VISC_1000 = "Viscosity_1000"
    VISC_10000 = "Viscosity_10000"
    VISC_100000 = "Viscosity_100000"
    VISC_15000000 = "Viscosity_15000000"
    ALL_SHEARS = [VISC_100, VISC_1000, VISC_10000, VISC_100000, VISC_15000000]

    def __init__(
        self,
        zip_path: str,
        mc_samples: int = 50,
        verify_signatures: bool = True,
    ):
        """
        Initialize the Predictor by unpacking the archive and loading the model.

        Args:
            zip_path: Path to the zip archive containing model/ and src/ directories.
            mc_samples: Number of Monte Carlo samples for uncertainty estimation.
            verify_signatures: Whether to verify cryptographic signatures of source files.
                             Set to False ONLY for debugging! Production should be True.

        Raises:
            FileNotFoundError: If the archive does not exist.
            SecurityError: If signature verification fails (when verify_signatures=True).
            RuntimeError: If expected folders or files are missing inside the archive.
        """
        self.zip_path = Path(zip_path)
        self.mc_samples = mc_samples
        self.verify_signatures = verify_signatures
        self._saved = False
        self._tmpdir = None
        self.predictor = None
        self.metadata = None
        self.secure_loader = None
        self.verification_report = None
        self.scaler_feature_names = None

        # Check if security module is available
        if self.verify_signatures and not SECURITY_AVAILABLE:
            Log.e("Security verification requested but secure_loader module not found!")
            Log.e("   Install: Place secure_loader.py in the same directory")
            Log.e(
                "   Or disable: set verify_signatures=False (NOT recommended for production)")
            raise RuntimeError(
                "secure_loader module not available. Cannot verify signatures."
            )

        if not self.verify_signatures:
            Log.w(" ========== WARNING =========== ")
            Log.w("Signature verification DISABLED!")

        Log.i(
            f"Predictor.__init__: archive={self.zip_path!r}, mc_samples={mc_samples}, "
            f"verify_signatures={verify_signatures}")
        self._load_zip(self.zip_path)

    def _load_zip(self, zip_path: Path) -> None:
        """
        Unpack the zip file, load source modules, and instantiate the predictor.

        Args:
            zip_path: Path object pointing to the zip archive.

        Raises:
            FileNotFoundError: If the zip_path is not a file.
            SecurityError: If signature verification fails.
            RuntimeError: If required folders or modules are missing.
        """
        # Clean up previous temp directory if exists
        if self._tmpdir:
            self._tmpdir.cleanup()

        if not zip_path.is_file():
            raise FileNotFoundError(f"Archive not found: {zip_path}")

        # Create temp directory and extract
        self._tmpdir = tempfile.TemporaryDirectory()
        extract_dir = Path(self._tmpdir.name)

        with pyzipper.AESZipFile(zip_path, 'r',
                                 compression=pyzipper.ZIP_DEFLATED,
                                 allowZip64=True,
                                 encryption=pyzipper.WZ_AES) as zf:
            try:
                zf.testzip()
            except RuntimeError as e:
                # Encrypted ZIP if "encrypted" in exception message
                if 'encrypted' in str(e):
                    Log.d("Accessing secured records...")
                    # Derive password from the archive comment via SHA-256
                    if len(zf.comment) == 0:
                        Log.e("ZIP archive comment is empty, cannot derive password")
                        raise RuntimeError("Empty ZIP archive comment")
                    zf.setpassword(hashlib.sha256(
                        zf.comment).hexdigest().encode())
                else:
                    Log.e("ZIP RuntimeError: " + str(e))
            except Exception as e:
                Log.e("ZIP Exception: " + str(e))
            zf.extractall(extract_dir)

        Log.d(f"Extracted {zip_path.name} -> {extract_dir}")

        # Verify required directories exist
        model_dir = extract_dir / "model"
        src_dir = extract_dir / "src"

        if not model_dir.is_dir():
            raise RuntimeError("'model/' folder missing in archive")
        if not src_dir.is_dir():
            raise RuntimeError("'src/' folder missing in archive")

        # Initialize security verification if enabled
        if self.verify_signatures:
            Log.i("Initializing security verification...")
            try:
                self.secure_loader = create_secure_loader_for_extracted_package(
                    extracted_dir=extract_dir,
                    enforce_signatures=True,
                    require_all_signed=True
                )
                Log.i("Security system initialized")
            except SecurityError as e:
                Log.e(f"SECURITY INITIALIZATION FAILED: {e}")
                self.cleanup()
                raise

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            # Verify metadata signature if security is enabled
            if self.verify_signatures and self.secure_loader:
                try:
                    self.secure_loader.verify_metadata(metadata_path)
                    Log.d("Metadata signature verified")
                except SecurityError as e:
                    Log.e(f"Metadata verification failed: {e}")
                    self.cleanup()
                    raise

            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)

            # Log signing status
            if self.metadata.get('cryptographically_signed'):
                Log.i(f"Loaded signed package: client={self.metadata.get('client')}, "
                      f"author={self.metadata.get('author')}")
            else:
                Log.i(f"Loaded metadata: client={self.metadata.get('client')}, "
                      f"author={self.metadata.get('author')}")
        else:
            Log.w("No metadata.json found in package")

        # Load source modules dynamically (with security verification)
        self._load_source_modules(src_dir)

        # Import the Predictor class and visq3xConfig
        try:
            from inference import Predictor
            from config import VisQ2xConfig
        except ImportError as e:
            Log.e(f"Failed to import required classes: {e}")
            raise RuntimeError(f"Could not import Predictor or visq2xConfig. "
                               f"Ensure inference.py and config.py are in the src/ directory. Error: {e}")

        # Find the checkpoint file
        checkpoint_path = model_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(f"checkpoint.pt not found in model/ directory")

        # Instantiate the predictor and load the model
        Log.i("Instantiating Predictor...")
        config = VisQ2xConfig()
        self.predictor = Predictor(config)

        Log.i(f"Loading model from {checkpoint_path}...")
        self.predictor.load(str(checkpoint_path))

        self.scaler_feature_names = getattr(
            self.predictor, 'scaler_feature_names', None)
        if self.scaler_feature_names is not None:
            Log.d(
                f"Loaded scaler feature names: {len(self.scaler_feature_names)} features")
        else:
            Log.w("Scaler feature names not available from checkpoint")

        # Generate security report if verification was enabled
        if self.verify_signatures and self.secure_loader:
            self.verification_report = self.secure_loader.get_verification_report()
            Log.i("Security verification complete")
            Log.i(
                f"  Files verified: {self.verification_report['files_verified']}")
            Log.i(
                f"  Total signatures: {self.verification_report['total_signatures']}")

        Log.i("Predictor ready for inference")

    def _load_source_modules(self, src_dir: Path) -> None:
        """
        Load all Python source modules from the src/ directory into sys.modules.

        If verify_signatures is True, verifies cryptographic signatures before loading.

        Args:
            src_dir: Directory containing the source .py files.

        Raises:
            SecurityError: If signature verification fails for any module.
            RuntimeError: If no Python files are found.
        """
        Log.d(f"Loading source modules from {src_dir}")

        # Discover all .py files in the src directory
        py_files = list(src_dir.glob('*.py'))

        if not py_files:
            raise RuntimeError(f"No Python files found in src/ directory")

        Log.i(f"Found {len(py_files)} Python modules to load")

        # Sort files to ensure consistent loading order
        # Load dependencies first (common modules before inference)
        priority_order = ['config', 'encoding', 'common',
                          'model', 'losses', 'continual_learning', 'inference']

        # Get list of actual modules present
        available_modules = [f.stem for f in py_files]

        # Filter priority order to only include available modules
        modules_to_load = [m for m in priority_order if m in available_modules]

        # Add any additional modules not in priority order
        additional_modules = [
            m for m in available_modules if m not in priority_order]
        modules_to_load.extend(additional_modules)

        # Use secure module loader if verification enabled
        if self.verify_signatures and self.secure_loader:
            Log.i("Verifying and loading modules securely...")

            try:
                # Create secure module loader
                module_loader = SecureModuleLoader(self.secure_loader, src_dir)

                # Load all modules with batch verification
                module_loader.load_all_modules_secure(
                    modules_to_load,
                    verify_batch=True
                )

                Log.i(
                    f"All {len(modules_to_load)} modules verified and loaded")

            except SecurityError as e:
                Log.e(f"SECURITY VIOLATION DETECTED: {e}")
                Log.e("Module loading aborted for safety")
                raise
            except Exception as e:
                Log.e(f"Error during secure module loading: {e}")
                raise RuntimeError(f"Failed to load modules securely: {e}")
        else:
            # Original unsecured loading (for backward compatibility or debugging)
            if not self.verify_signatures:
                Log.w(" Loading modules WITHOUT signature verification")

            for module_name in modules_to_load:
                module_file = src_dir / f"{module_name}.py"

                if not module_file.exists():
                    Log.w(f"Module {module_name}.py not found, skipping")
                    continue

                # Load module using importlib
                spec = importlib.util.spec_from_file_location(
                    module_name, str(module_file))

                if spec is None:
                    Log.w(f"Could not load spec for {module_name}")
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                Log.d(f"Loaded: {module_name}")

            Log.i(f"Loaded {len(modules_to_load)} modules")

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Generate point-estimate predictions without uncertainty.

        Args:
            df: Input DataFrame with feature columns.

        Returns:
            Array of model predictions (original scale, not log-transformed).

        Raises:
            RuntimeError: If the predictor is not loaded.
        """
        if self.predictor is None:
            Log.e("predict() called but predictor is not loaded")
            raise RuntimeError("No predictor loaded")
        try:
            Log.d(f"Predictor.predict() with {len(df)} samples")
            return self.predictor.predict(df, return_uncertainty=False)
        except Exception as ex:
            Log.e(f"Error in predict(): {ex}")
            raise

    def predict_uncertainty(
        self,
        df: pd.DataFrame,
        ci_range: tuple = (2.5, 97.5),
        n_samples: int = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate predictions along with uncertainty estimates.

        Args:
            df: Input DataFrame with feature columns.
            n_samples: Number of Monte Carlo samples (default: self.mc_samples).
            ci_range: the confidence interval to report (default: tuple(0.05, 0.95))

        Returns:
            Tuple of (mean predictions, uncertainty dict with std/lower_ci/upper_ci/cv).

        Raises:
            RuntimeError: If the predictor is not loaded.
        """
        if self.predictor is None:
            Log.e("predict_uncertainty() called but predictor is not loaded")
            raise RuntimeError("No predictor loaded")
        if n_samples is None:
            n_samples = self.mc_samples

        try:
            Log.d(f"Predictor.predict_uncertainty() with {len(df)} samples, "
                  f"n_samples={n_samples}")
            return self.predictor.predict(df, return_uncertainty=True, ci_range=ci_range, n_samples=n_samples)
        except Exception as ex:
            Log.e(f"Error in predict_uncertainty(): {ex}")
            raise

    def evaluate(
        self,
        eval_data: pd.DataFrame,
        targets: list = ALL_SHEARS,
        n_samples: int = None,
    ) -> Dict:
        """Evaluate model predictions against actual values with uncertainty metrics.

        This method generates predictions with uncertainty estimates for the evaluation
        data and computes detailed error metrics for each sample and shear rate
        combination. It creates a comprehensive results DataFrame containing actual
        values, predictions, uncertainty bounds, and various error measures.

        Args:
            eval_data (pd.DataFrame): DataFrame containing evaluation data with
                feature columns and target viscosity columns.
            targets (list, optional): List of target shear rate column names to
                evaluate. Only columns present in both targets and eval_data will
                be used. Defaults to ALL_SHEARS.
            n_samples (int, optional): Number of samples to use for uncertainty
                estimation in Monte Carlo predictions. If None, uses the default
                from predict_uncertainty. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with one row per (sample, shear_rate) combination,
                containing the following columns:
                - sample_idx: Index of the sample in eval_data
                - shear_rate: Name of the viscosity column (shear rate)
                - actual: Actual viscosity value
                - predicted: Predicted mean viscosity value
                - std: Standard deviation of predictions
                - lower_ci: Lower bound of 95% confidence interval
                - upper_ci: Upper bound of 95% confidence interval
                - cv: Coefficient of variation
                - residual: Prediction residual (actual - predicted)
                - abs_error: Absolute error
                - pct_error: Percentage error (0 if actual is 0)
                - within_ci: Boolean indicating if actual falls within 95% CI

        Raises:
            RuntimeError: If no predictor has been loaded (self.predictor is None).
            ValueError: If no target columns are found in eval_data.
            Exception: Re-raises any exceptions that occur during predict_uncertainty.

        Note:
            If the number of predicted outputs doesn't match the number of target
            columns, the method adjusts by truncating the viscosity columns to
            match the prediction output shape and logs a warning.
        """

        if self.predictor is None:
            Log.e("evaluate() called but predictor is not loaded")
            raise RuntimeError("No predictor loaded")
        viscosity_cols = [
            col for col in eval_data.columns if col in targets]

        if not viscosity_cols:
            raise ValueError(
                f"No columns starting with targets found in eval_data.")
        viscosity_cols = sorted(viscosity_cols)
        n_outputs = len(viscosity_cols)
        Log.i(
            f"Evaluating on {len(eval_data)} samples with {n_outputs} shear rates output.")
        y_actual = eval_data[viscosity_cols].values
        try:
            y_pred_mean, uncertainty = self.predict_uncertainty(
                eval_data, n_samples=n_samples
            )
        except Exception as ex:
            Log.e(f"Error during evaluation prediction: {ex}")
            raise
        if y_pred_mean.shape[1] != n_outputs:
            Log.w(f"Expected {n_outputs} outputs but got {y_pred_mean.shape[1]}. "
                  f"Adjusting viscosity columns to match.")
            viscosity_cols = viscosity_cols[:y_pred_mean.shape[1]]
            y_actual = y_actual[:, :y_pred_mean.shape[1]]
            n_outputs = y_pred_mean.shape[1]
        results_data = []

        for i in range(len(eval_data)):
            for j, visc_col in enumerate(viscosity_cols):
                results_data.append({
                    'sample_idx': i,
                    'shear_rate': visc_col,
                    'actual': y_actual[i, j],
                    'predicted': y_pred_mean[i, j],
                    'std': uncertainty['std'][i, j],
                    'lower_ci': uncertainty['lower_ci'][i, j],
                    'upper_ci': uncertainty['upper_ci'][i, j],
                    'cv': uncertainty['coefficient_of_variation'][i, j],
                    'residual': y_actual[i, j] - y_pred_mean[i, j],
                    'abs_error': np.abs(y_actual[i, j] - y_pred_mean[i, j]),
                    'pct_error': np.abs((y_actual[i, j] - y_pred_mean[i, j]) / y_actual[i, j]) * 100 if y_actual[i, j] != 0 else 0,
                    'within_ci': (y_actual[i, j] >= uncertainty['lower_ci'][i, j]) and (y_actual[i, j] <= uncertainty['upper_ci'][i, j])
                })

        results_df = pd.DataFrame(results_data)

        return results_df

    def learn(
        self,
        new_df: pd.DataFrame,
        n_epochs: Optional[int] = None,
        save: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        Incrementally update the model with new data.

        Args:
            new_df: DataFrame with both features and target columns.
            n_epochs: Number of training epochs (default: auto from config).
            verbose: Whether to print training progress.

        Returns:
            Dictionary with training info (avg_loss, new_categories_added, n_epochs).

        Raises:
            RuntimeError: If the predictor is not loaded.
        """
        if self.predictor is None:
            Log.e("learn() called but predictor is not loaded")
            raise RuntimeError("No predictor loaded")

        Log.i(f"Predictor.learn(): data.shape={new_df.shape}, "
              f"n_epochs={n_epochs}, verbose={verbose}")

        try:
            result = self.predictor.learn(
                new_df, n_epochs=n_epochs, verbose=verbose)
            Log.i(f"Incremental learning completed successfully, {result}")
            self._saved = save
            return result
        except Exception as ex:
            Log.e(f"Error during learn(): {ex}")
            raise

    def get_metadata(self) -> Optional[Dict]:
        """
        Get the package metadata.

        Returns:
            Dictionary containing metadata or None if not available.
        """
        return self.metadata

    def get_metadata(self) -> Optional[Dict]:
        """
        Get the package metadata.

        Returns:
            Dictionary containing metadata or None if not available.
        """
        return self.metadata

    def get_verification_report(self) -> Optional[Dict]:
        """
        Get detailed security verification report.

        Returns:
            Dictionary containing:
                - enforcement_enabled: Whether signature verification was enforced
                - require_all_signed: Whether all files were required to be signed
                - total_signatures: Total number of signatures in manifest
                - files_verified: Number of files successfully verified
                - verified_files: List of verified file paths
                - unverified_signed_files: Files that have signatures but weren't checked

            Returns None if signature verification was disabled.
        """
        if not self.verify_signatures or self.verification_report is None:
            Log.w("Security report not available (verification may be disabled)")
            return None

        return self.verification_report

    def save_path(self) -> Optional[str]:
        """Path to the temporary directory associated with this Predictor session if a save occurred; otherwise None."""
        return self._tmpdir.name if self._saved else None

    def save(self, path: str) -> None:
        """
        Save the current model state to a file.

        Args:
            path: Path where to save the model checkpoint.

        Raises:
            RuntimeError: If the predictor is not loaded.
        """
        if self.predictor is None:
            Log.e("save() called but predictor is not loaded")
            raise RuntimeError("No predictor loaded")

        Log.i(f"Saving model to {path}")
        self.predictor.save_state(path)
        Log.i("Model saved successfully")

    def add_security_to_zip(self, zip_path) -> bool:
        """Replace an existing ZIP in-place with a secured ZIP"""
        try:
            comment = self._calculate_sha256_safe(zip_path)
            if not comment:
                Log.e("Could not compute SHA-256 of zip; aborting encryption")
                return False
            password = hashlib.sha256(comment.encode()).hexdigest()
            # replace ZIP with encrypted version
            self._add_password_and_comment(
                zip_path, zip_path, password, comment)
            Log.i("Updated ensemble saved to archive")
            return True
        except Exception as e:
            Log.e(f"Error saving updated ensemble: {e}")
            return False

    def _calculate_sha256_safe(self, file_path):
        """Calculate SHA256 with error handling"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            sha256_hash = hashlib.sha256()

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            return sha256_hash.hexdigest()

        except Exception as e:
            Log.e(f"Error calculating hash: {e}")
            return None

    def _add_password_and_comment(self, input_zip, output_zip, password, comment):
        """Create a new encrypted zip file with a password and comment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract input zip
            with zipfile.ZipFile(input_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Create new encrypted zip with pyzipper
            with pyzipper.AESZipFile(output_zip, 'w',
                                     compression=pyzipper.ZIP_DEFLATED,
                                     encryption=pyzipper.WZ_AES) as zf:
                # Set password
                zf.setpassword(password.encode('utf-8'))
                zf.setencryption(pyzipper.WZ_AES, nbits=256)

                # Add comment
                zf.comment = comment.encode('utf-8')

                # Add all files recursively
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate archive path (relative to temp_dir)
                        arc_path = os.path.relpath(file_path, temp_dir)
                        # Normalize path separators for cross-platform compatibility
                        arc_path = arc_path.replace('\\', '/')
                        zf.write(file_path, arc_path)

    def reload_archive(self, new_zip: str) -> None:
        """
        Reload a different archive, cleaning up the previous one.

        Args:
            new_zip: Path to the new zip archive.
        """
        Log.i(f"reload_archive: {new_zip!r}")
        self.zip_path = Path(new_zip)
        self._load_zip(self.zip_path)

    def cleanup(self) -> None:
        """
        Clean up temporary files and drop the loaded ensemble.
        """
        if self._tmpdir:
            Log.i("Cleaning up temp directory and ensemble")
            try:
                self._tmpdir.cleanup()
            except Exception as ex:
                Log.w(f"Issue during cleanup: {ex}")
            finally:
                self._tmpdir = None
                self.predictor = None
                self.secure_loader = None
                self.verification_report = None
                self.scaler_feature_names = None

    def __enter__(self) -> 'Predictor':
        """Enter context manager, returning the Predictor instance."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit context manager, performing cleanup."""
        self.cleanup()

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass
