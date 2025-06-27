"""
Module: predictor

Provides the Predictor class for loading a packaged viscosity model ensemble
and performing inference, uncertainty estimation, and incremental updates.
Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-27

Version:
    2.1
"""
import zipfile
import tempfile
import sys
import logging
from pathlib import Path
from typing import Union, Dict, List, Tuple

import numpy as np
import pandas as pd
import importlib.util
import importlib.machinery
try:
    from QATCH.common.logger import Logger as Log
except (ImportError, ModuleNotFoundError):
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """
        Logging utility for standardized log messages.

        Provides shorthand methods for different log levels tied to a "Predictor" logger.
        """
        _logger = logging.getLogger("Predictor")

        @classmethod
        def i(cls, msg: str) -> None:
            """
            Log an informational message.

            Args:
                msg: Message string to log at INFO level.
            """
            cls._logger.info(msg)

        @classmethod
        def w(cls, msg: str) -> None:
            """
            Log a warning message.

            Args:
                msg: Message string to log at WARNING level.
            """
            cls._logger.warning(msg)

        @classmethod
        def e(cls, msg: str) -> None:
            """
            Log an error message.

            Args:
                msg: Message string to log at ERROR level.
            """
            cls._logger.error(msg)

        @classmethod
        def d(cls, msg: str) -> None:
            """
            Log a debug message.

            Args:
                msg: Message string to log at DEBUG level.
            """
            cls._logger.debug(msg)


class Predictor:
    """
    Predictor for loading a packaged viscosity model ensemble and performing
    predictions and updates.

    This class extracts a zip archive containing a model folder, loads required
    modules, and instantiates an EnsembleViscosityPredictor for inference and
    incremental updates.
    """

    def __init__(
        self,
        zip_path: str,
        mc_samples: int = 50,
        model_filename: str = "model.h5",
        preprocessor_filename: str = "preprocessor.pkl",
    ):
        """
        Initialize the Predictor by unpacking the archive and loading the ensemble.

        Args:
            zip_path: Path to the zip archive containing the 'model/' directory.
            mc_samples: Number of Monte Carlo samples for uncertainty estimation.
            model_filename: Filename of the saved model inside each member folder.
            preprocessor_filename: Filename of the preprocessor pickle inside each member folder.

        Raises:
            FileNotFoundError: If the archive does not exist.
            RuntimeError: If expected folders or files are missing inside the archive.
        """
        self.zip_path = Path(zip_path)
        self.mc_samples = mc_samples
        self.model_filename = model_filename
        self.preprocessor_filename = preprocessor_filename

        self._tmpdir = None
        self.ensemble = None

        Log.i(f"Predictor.__init__: archive={self.zip_path!r}, "
              f"mc_samples={mc_samples}, model_fn={model_filename!r}, "
              f"pre_fn={preprocessor_filename!r}")
        self._load_zip(self.zip_path)

    def _load_zip(self, zip_path: Path) -> None:
        """
        Unpack the zip file, load custom modules, and instantiate the ensemble.

        Args:
            zip_path: Path object pointing to the zip archive.

        Raises:
            FileNotFoundError: If the zip_path is not a file.
            RuntimeError: If required folders or compiled modules are missing.
        """
        if self._tmpdir:
            self._tmpdir.cleanup()
        if not zip_path.is_file():
            raise FileNotFoundError(f"Archive not found: {zip_path}")
        self._tmpdir = tempfile.TemporaryDirectory()
        extract_dir = Path(self._tmpdir.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        Log.i(f"Extracted {zip_path.name} → {extract_dir}")

        # Locate model/
        model_dir = extract_dir / "model"
        if not model_dir.is_dir():
            raise RuntimeError("'model/' folder missing in archive")

        # Load custom_layers.pyc
        self._load_module(extract_dir, "custom_layers")

        # Load data processor module (contains DataProcessor class)
        self._load_module(extract_dir, "data_processor")

        # Load predictor.pyc (depends on above)
        predictor_pyc = extract_dir / "predictor.pyc"
        if not predictor_pyc.exists():
            Log.e("predictor.pyc not found")
            raise RuntimeError("predictor.pyc missing")
        loader = importlib.machinery.SourcelessFileLoader(
            "predictor", str(predictor_pyc)
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules["predictor"] = mod
        Log.i("Loaded predictor.pyc into sys.modules")

        # Instantiate EnsembleViscosityPredictor
        from predictor import EnsembleViscosityPredictor
        self.ensemble = EnsembleViscosityPredictor(
            base_dir=str(model_dir),
            mc_samples=self.mc_samples,
            model_filename=self.model_filename,
            preprocessor_filename=self.preprocessor_filename,
        )
        Log.i(f"Ensemble loaded with {len(self.ensemble.members)} members")

    def _load_module(self, base_dir: Path, name: str) -> None:
        """
        Load a compiled .pyc module into sys.modules.

        Args:
            base_dir: Directory containing the .pyc file.
            name: Base name of the module to load (without extension).

        Raises:
            RuntimeError: If the .pyc file is not found.
        """
        pyc = base_dir / f"{name}.pyc"
        if not pyc.exists():
            Log.e(f"{name}.pyc not found in {base_dir}")
            raise RuntimeError(f"{name}.pyc missing")
        loader = importlib.machinery.SourcelessFileLoader(name, str(pyc))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules[name] = mod
        Log.d(f"Loaded module '{name}' from {pyc.name}")

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, List], List[Dict[str, float]]]
    ) -> np.ndarray:
        """
        Generate point-estimate predictions without uncertainty.

        Args:
            data: Input features in one of several supported formats.

        Returns:
            Array of model predictions.

        Raises:
            RuntimeError: If the ensemble is not loaded.
        """
        if self.ensemble is None:
            Log.e("predict() called but ensemble is not loaded")
            raise RuntimeError("No ensemble loaded")
        try:
            Log.d("Predictor.predict()")
            return self.ensemble.predict(data, return_confidence=False)
        except Exception as ex:
            Log.e(f"Error in predict(): {ex}")
            raise

    def predict_uncertainty(
        self,
        data: Union[pd.DataFrame, Dict[str, List], List[Dict[str, float]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions along with uncertainty estimates.

        Args:
            data: Input features in one of several supported formats.

        Returns:
            Tuple of (mean predictions, total uncertainty).

        Raises:
            RuntimeError: If the ensemble is not loaded.
        """
        if self.ensemble is None:
            Log.e("predict_uncertainty() called but ensemble is not loaded")
            raise RuntimeError("No ensemble loaded")
        try:
            Log.d("Predictor.predict_uncertainty()")
            return self.ensemble.predict(data, return_confidence=True)
        except Exception as ex:
            Log.e(f"Error in predict_uncertainty(): {ex}")
            raise

    def update(
        self,
        new_data: pd.DataFrame,
        new_targets: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        save: bool = True,
    ) -> None:
        """
        Incrementally update the ensemble with new data.

        Args:
            new_data: DataFrame of new feature samples.
            new_targets: NumPy array of new target values.
            epochs: Number of training epochs for the update.
            batch_size: Batch size for the training.
            save: Whether to save updated models back to disk.

        Raises:
            RuntimeError: If the ensemble is not loaded.
        """
        if self.ensemble is None:
            Log.e("update() called but ensemble is not loaded")
            raise RuntimeError("No ensemble loaded")
        Log.i(
            f"Predictor.update(): data.shape={getattr(new_data, 'shape', None)}, "
            f"targets.shape={getattr(new_targets, 'shape', None)}, "
            f"epochs={epochs}, batch_size={batch_size}, save={save}"
        )
        try:
            self.ensemble.update(
                new_data,
                new_targets,
                epochs=epochs,
                batch_size=batch_size,
                save=save,
            )
            Log.i("Ensemble update completed successfully")
        except Exception as ex:
            Log.e(f"Error during ensemble.update(): {ex}")
            raise

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
        """
        Enter context manager, returning the Predictor instance.
        """
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ) -> None:
        """
        Exit context manager, performing cleanup.
        """
        self.cleanup()

    def __del__(self) -> None:
        """
        Ensure cleanup on garbage collection.
        """
        try:
            self.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    import pandas as pd
    from io import StringIO

    csv = """ID,Protein_type,MW,PI_mean,PI_range,Protein_conc,Temperature,Buffer_type,Buffer_ph,Buffer_conc,Salt_type,Salt_conc,Stabilizer_type,Stabilizer_conc,Surfactant_type,Surfactant_conc,Viscosity_100,Viscosity_1000,Viscosity_10000,Viscosity_100000,Viscosity_15000000,
F1,poly-hIgG,150,7.6,1,145,25,PBS,7.4,10,NaCl,140,Sucrose,1,none,0,12.5,11.5,9.8,8.8,6.92,
    """
    df_test = pd.read_csv(StringIO(csv))
    executor = Predictor("visQAI/packaged/VisQAI-base.zip")
    executor.update(
        df_test, np.array([[12.5, 11.5, 9.8, 8.8, 6.92]])
    )
    mean, std = executor.predict_uncertainty(df_test)
    print("Mean predictions:", mean)
    print("Std. deviations:", std)
    executor.cleanup()
