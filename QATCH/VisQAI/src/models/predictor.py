"""
Module: predictor.py

Provides the Predictor class for loading a packaged viscosity model
and performing inference, uncertainty estimation, and incremental updates.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-15

Version:
    3.1
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
    """

    def __init__(
        self,
        zip_path: str,
        mc_samples: int = 50,
    ):
        """
        Initialize the Predictor by unpacking the archive and loading the model.

        Args:
            zip_path: Path to the zip archive containing model/ and src/ directories.
            mc_samples: Number of Monte Carlo samples for uncertainty estimation.

        Raises:
            FileNotFoundError: If the archive does not exist.
            RuntimeError: If expected folders or files are missing inside the archive.
        """
        self.zip_path = Path(zip_path)
        self.mc_samples = mc_samples
        self._saved = False
        self._tmpdir = None
        self.predictor = None
        self.metadata = None

        Log.i(
            f"Predictor.__init__: archive={self.zip_path!r}, mc_samples={mc_samples}")
        self._load_zip(self.zip_path)

    def _load_zip(self, zip_path: Path) -> None:
        """
        Unpack the zip file, load source modules, and instantiate the predictor.

        Args: 
            zip_path: Path object pointing to the zip archive.

        Raises:
            FileNotFoundError: If the zip_path is not a file.
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

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            Log.i(f"Loaded metadata: client={self.metadata.get('client')}, "
                  f"author={self.metadata.get('author')}")
        else:
            Log.w("No metadata.json found in package")

        # Load source modules dynamically
        self._load_source_modules(src_dir)

        # Import the Predictor class and visq3xConfig
        try:
            from inference import Predictor
            from config import Visq3xConfig
        except ImportError as e:
            Log.e(f"Failed to import required classes: {e}")
            raise RuntimeError(f"Could not import Predictor or visq3xConfig. "
                               f"Ensure inference.py and config.py are in the src/ directory. Error: {e}")

        # Find the checkpoint file
        checkpoint_path = model_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(f"checkpoint.pt not found in model/ directory")

        # Instantiate the predictor and load the model
        Log.i("Instantiating Predictor...")
        config = Visq3xConfig()
        self.predictor = Predictor(config)

        Log.i(f"Loading model from {checkpoint_path}...")
        self.predictor.load(str(checkpoint_path))

        Log.i("Predictor ready for inference")

    def _load_source_modules(self, src_dir: Path) -> None:
        """
        Load all Python source modules from the src/ directory into sys.modules.

        Args:
            src_dir: Directory containing the source .py files.

        Raises:
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

        # Create a sorted list: priority modules first, then others
        sorted_files = []
        remaining_files = []

        for py_file in py_files:
            module_name = py_file.stem
            if module_name in priority_order:
                sorted_files.append(
                    (priority_order.index(module_name), py_file))
            else:
                remaining_files.append(py_file)

        # Sort priority files by their order
        sorted_files.sort(key=lambda x: x[0])
        final_files = [f[1] for f in sorted_files] + sorted(remaining_files)

        # Load each module
        for py_file in final_files:
            module_name = py_file.stem

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                Log.d(f"Loaded {py_file.name}")
            except Exception as e:
                Log.e(f"Failed to load {py_file.name}: {e}")
                raise RuntimeError(f"Failed to load module {module_name}: {e}")

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
        n_samples: int = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate predictions along with uncertainty estimates.

        Args:
            df: Input DataFrame with feature columns.
            n_samples: Number of Monte Carlo samples (default: self.mc_samples).

        Returns:
            Tuple of (mean predictions, uncertainty dict with std/lower_95/upper_95/cv).

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
            return self.predictor.predict(df, return_uncertainty=True, n_samples=n_samples)
        except Exception as ex:
            Log.e(f"Error in predict_uncertainty(): {ex}")
            raise

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
                self.ensemble = None

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


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    sample_data = pd.DataFrame({
        'Protein_type': ['IgG1'],
        'MW': [150000],
        'Protein_conc': [145],
        'Temperature': [25],
        'Buffer_type': ['PBS'],
        'Buffer_pH': [7.4],
        'Buffer_conc': [10],
        'Salt_type': ['NaCl'],
        'Salt_conc': [140],
        'Stabilizer_type': ['Sucrose'],
        'Stabilizer_conc': [1],
        'Surfactant_type': ['none'],
        'Surfactant_conc': [0],
        'Protein_class_type': ['mAb'],
    })

    # Load packaged predictor
    with Predictor("packages/VisQAI-base.zip") as predictor:
        # Get metadata
        metadata = predictor.get_metadata()
        print("Metadata:", metadata)

        # Make predictions
        predictions = predictor.predict(sample_data)
        print("\nPredictions:", predictions)

        # Make predictions with uncertainty
        mean_pred, uncertainty = predictor.predict_uncertainty(sample_data)
        print("\nMean predictions:", mean_pred)
        print("Uncertainty (std):", uncertainty['std'])
        print("95% CI:", uncertainty['lower_95'],
              "to", uncertainty['upper_95'])
