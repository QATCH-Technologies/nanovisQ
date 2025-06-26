# predictor.py

"""
# TODO: Describe module and save structure

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-26

Version:
    2.2
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
    from src.controller.formulation_controller import FormulationController
    from src.db.db import Database
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        _logger = logging.getLogger("Predictor")
        @classmethod
        def i(cls, msg: str): cls._logger.info(msg)
        @classmethod
        def w(cls, msg: str): cls._logger.warning(msg)
        @classmethod
        def e(cls, msg: str): cls._logger.error(msg)
        @classmethod
        def d(cls, msg: str): cls._logger.debug(msg)
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.common.logger import Logger as Log


class Predictor:

    def __init__(
        self,
        zip_path: str,
        mc_samples: int = 50,
        model_filename: str = "model.keras",
        preprocessor_filename: str = "preprocessor.pkl",
    ):
        self.zip_path = Path(zip_path)
        self.mc_samples = mc_samples
        self.model_filename = model_filename
        self.preprocessor_filename = preprocessor_filename

        self._tmpdir = None
        self.model = None

        Log.i(f"Predictor.__init__: archive={self.zip_path!r}, "
              f"mc_samples={mc_samples}, model_fn={model_filename!r}, "
              f"pre_fn={preprocessor_filename!r}")
        self._load_zip(self.zip_path)

    def _load_zip(self, zip_path: Path):
        if self._tmpdir:
            self._tmpdir.cleanup()
        if not zip_path.is_file():
            raise FileNotFoundError(f"Archive not found: {zip_path}")
        self._tmpdir = tempfile.TemporaryDirectory()
        extract_dir = Path(self._tmpdir.name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        Log.i(f"Extracted {zip_path.name} → {extract_dir}")

        # Locate cnn/
        cnn_dir = extract_dir / "cnn"
        if not cnn_dir.is_dir():
            raise RuntimeError("'cnn/' folder missing in archive")

        # Load custom_layers.pyc
        self._load_module(extract_dir, "custom_layers")

        # Load visQ_data_processor.pyc
        self._load_module(extract_dir, "visQ_data_processor")

        # Load predictor.pyc (which depends on the two above)
        predictor_pyc = extract_dir / "predictor.pyc"
        if not predictor_pyc.exists():
            Log.e("predictor.pyc not found")
            raise RuntimeError("predictor.pyc missing")
        loader = importlib.machinery.SourcelessFileLoader(
            "predictor", str(predictor_pyc))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        sys.modules["predictor"] = mod
        Log.i("Loaded predictor.pyc into sys.modules")

        # Instantiate EnsembleViscosityPredictor
        from predictor import EnsembleViscosityPredictor
        self.model = EnsembleViscosityPredictor(
            base_dir=str(cnn_dir),
            mc_samples=self.mc_samples,
            model_filename=self.model_filename,
            preprocessor_filename=self.preprocessor_filename,
        )
        Log.i(f"Ensemble loaded with {len(self.model.members)} members")

    def _load_module(self, base_dir: Path, name: str):
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

    def predict(self, data: Union[pd.DataFrame, Dict[str, List], List[Dict[str, float]]]) -> np.ndarray:
        if self.model is None:
            Log.e("predict() called but ensemble is not loaded")
            raise RuntimeError("No ensemble loaded")
        try:
            Log.d("Executor.predict()")
            return self.model.predict(data, return_confidence=False)
        except Exception as ex:
            Log.e(f"Error in predict(): {ex}")
            raise

    def predict_uncertainty(self, data: Union[pd.DataFrame, Dict[str, List], List[Dict[str, float]]]) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            Log.e("predict_uncertainty() called but ensemble is not loaded")
            raise RuntimeError("No ensemble loaded")
        try:
            Log.d("Executor.predict_uncertainty()")
            return self.model.predict(data, return_confidence=True)
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
    ):
        if self.model is None:
            Log.e("update() called but ensemble is not loaded")
            raise RuntimeError("No ensemble loaded")
        Log.i(f"Executor.update(): data.shape={getattr(new_data, 'shape', None)}, "
              f"targets.shape={getattr(new_targets, 'shape', None)}, "
              f"epochs={epochs}, batch_size={batch_size}, save={save}")
        try:
            self.model.update(
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

    def reload_archive(self, new_zip: str):
        Log.i(f"reload_archive: {new_zip!r}")
        self.zip_path = Path(new_zip)
        self._load_zip(self.zip_path)

    def cleanup(self):
        if self._tmpdir:
            Log.i("Cleaning up temp directory and ensemble")
            try:
                self._tmpdir.cleanup()
            except Exception as ex:
                Log.w(f"Issue during cleanup: {ex}")
            finally:
                self._tmpdir = None
                self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
