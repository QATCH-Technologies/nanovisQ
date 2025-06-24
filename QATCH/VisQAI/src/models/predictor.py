# predictor.py

"""
This module provides a generic `Predictor` class that dynamically loads and executes a pipeline of components
for data processing, transformation, and prediction. The expected bundle structure is:

    <root>/
    └── package/
        ├── __init__.py
        ├── __pycache__/
        │   ├── dataprocessor.cpython-xx.pyc
        │   └── predictor.cpython-xx.pyc
        ├── model/
        │   ├── assets/
        │   ├── variables/
        │   └── saved_model.pb
        └── transformer.pkl

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-24

Version:
    2.1
"""

import tempfile
import zipfile
import pandas as pd
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Union, Tuple
import numpy as np
import importlib.util
from importlib.machinery import SourcelessFileLoader
from src.controller.formulation_controller import FormulationController
from src.db.db import Database
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False


class ComponentLoadError(Exception):
    """Raised when a required pipeline component cannot be found or validated."""
    pass


class Predictor:

    """
    Dynamically load and execute a pipeline of components for data processing, transformation, and prediction.

    The bundle must follow this layout under the root directory ( or ZIP):
        <root > /
        └── package/
            ├── __init__.py
            ├── __pycache__/
            │   ├── dataprocessor.cpython-xx.pyc
            │   └── predictor.cpython-xx.pyc
            ├── model/
            │   ├── assets/
            │   ├── variables/
            │   └── saved_model.pb
            └── transformer.pkl

    Attributes:
        data_processor: Instance of the DataProcessor class (loaded from compiled module).
        transformer:     Fitted scikit transformer(must implement .transform(...) and .fit_transform(...)).
        model:           A namespace wrapping the SavedModel's .predict(...) function.
        predictor:       Instance of the executor's Predictor class (loaded from compiled module).
        _tempdir:        TemporaryDirectory(if source was a ZIP).
    """

    def __init__(self, source: Union[str, Path]) -> None:
        """
        Initialize a Predictor by loading all required components from either:
          - A directory containing `package /`, or
          - A ZIP archive with the same structure.

        Args:
            source: Path to the directory or ZIP archive.

        Raises:
            ComponentLoadError: If any component(directory, modules, or files) cannot be found, loaded, or validated.
        """
        self._tempdir = None
        extracted_root = self._extract_if_archive(Path(source))
        pkg_dir = extracted_root / "package"
        if not pkg_dir.is_dir():
            raise ComponentLoadError(
                f"Expected 'package/' folder under '{extracted_root}', but it was not found."
            )

        data_processor_name = "dataprocessor"
        transformer_filename = "transformer.pkl"
        model_dir_name = "model"
        predictor_name = "predictor"

        # Load DataProcessor from compiled .pyc in package/__pycache__/
        dp_module = self._load_pyc_module(pkg_dir, data_processor_name)
        if not hasattr(dp_module, "DataProcessor") or not callable(dp_module.DataProcessor):
            raise ComponentLoadError(
                f"Module '{data_processor_name}' does not define a callable DataProcessor class."
            )
        try:
            self.data_processor = dp_module.DataProcessor()
        except Exception as e:
            raise ComponentLoadError(
                f"Failed to instantiate DataProcessor: {e!s}")

        # Instantiate TransformerPipeline with the saved scaler
        transformer_path = pkg_dir / transformer_filename
        if not transformer_path.is_file():
            raise ComponentLoadError(
                f"Missing transformer file '{transformer_filename}' under '{pkg_dir}'."
            )

        # Load the compiled transformer module so we can grab the class
        tp_module = self._load_pyc_module(pkg_dir, "transformer")
        if not hasattr(tp_module, "TransformerPipeline"):
            raise ComponentLoadError(
                "Module 'transformer' does not define TransformerPipeline."
            )
        try:
            TransformerPipeline = tp_module.TransformerPipeline
            self.transformer = TransformerPipeline(
                scaler_path=str(transformer_path))
        except Exception as e:
            raise ComponentLoadError(
                f"Failed to instantiate TransformerPipeline with '{transformer_path}': {e!s}"
            )
        if not (
            hasattr(self.transformer, "transform") and callable(
                self.transformer.transform)
            and hasattr(self.transformer, "fit_transform") and callable(self.transformer.fit_transform)
        ):
            raise ComponentLoadError(
                "TransformerPipeline must implement callable transform(...) and fit_transform(...)."
            )

        # Load TensorFlow SavedModel directory ("model/")
        saved_model_dir = pkg_dir / model_dir_name
        if not saved_model_dir.is_dir():
            raise ComponentLoadError(
                f"Missing SavedModel directory '{model_dir_name}/' under '{pkg_dir}'."
            )
        if not _TF_AVAILABLE:
            raise ComponentLoadError(
                "TensorFlow is not installed; cannot load model."
            )
        try:
            self.model_path = str(saved_model_dir)
            self.model = tf.saved_model.load(str(self.model_path))
        except Exception as e:
            raise ComponentLoadError(
                f"Failed to load SavedModel from '{saved_model_dir}': {e!s}"
            )
        if "serving_default" not in self.model.signatures:
            raise ComponentLoadError(
                f"SavedModel at '{saved_model_dir}' has no 'serving_default' signature."
            )

        # Load executor (predictor) module from compiled .pyc in package/__pycache__/
        pred_module = self._load_pyc_module(pkg_dir, predictor_name)
        if not hasattr(pred_module, "Predictor") or not callable(pred_module.Predictor):
            raise ComponentLoadError(
                f"Module '{predictor_name}' does not define a callable Predictor class."
            )
        try:
            self.predictor = pred_module.Predictor()
        except Exception as e:
            raise ComponentLoadError(
                f"Failed to instantiate executor Predictor: {e!s}"
            )
        self._wire_model_to_executor()
        self.database = Database(parse_file_key=True)
        self.form_ctrl = FormulationController(db=self.database)

    def _extract_if_archive(self, path: Path) -> Path:
        """
        If `path` is a ZIP file, extract it to a temporary directory and return that path.
        If `path` is a directory, return it unchanged.

        Args:
            path: Path to a directory or ZIP file.

        Returns:
            Path to the directory containing extracted contents or the original directory.

        Raises:
            ComponentLoadError: If `path` is neither a directory nor a valid ZIP file, or extraction fails.
        """
        if path.is_dir():
            return path.resolve()

        if path.is_file() and zipfile.is_zipfile(path):
            try:
                temp_dir = tempfile.TemporaryDirectory()
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(temp_dir.name)
                self._tempdir = temp_dir
                return Path(temp_dir.name).resolve()
            except Exception as e:
                raise ComponentLoadError(
                    f"Failed to extract ZIP '{path}': {e!s}")

        raise ComponentLoadError(
            f"Path '{path}' is neither a directory nor a valid ZIP file.")

    def _load_pyc_module(self, pkg_dir: Path, module_name: str) -> types.ModuleType:
        """
        Locate `< pkg_dir > /__pycache__/{module_name}.*.pyc` and load it via SourcelessFileLoader.

        Args:
            pkg_dir: Path to the package directory containing __pycache__.
            module_name: Base name of the module to load(without extension).

        Returns:
            ModuleType: The loaded Python module.

        Raises:
            ComponentLoadError: If no .pyc file is found or if loading fails.
        """
        pyc_dir = pkg_dir / "__pycache__"
        if not pyc_dir.is_dir():
            raise ComponentLoadError(
                f"Missing '__pycache__' under '{pkg_dir}'.")

        for entry in pyc_dir.iterdir():
            if (
                entry.is_file()
                and entry.name.startswith(f"{module_name}.")
                and entry.suffix == ".pyc"
            ):
                try:
                    loader = SourcelessFileLoader(module_name, str(entry))
                    spec = importlib.util.spec_from_loader(module_name, loader)
                    if spec is None:
                        raise ComponentLoadError(
                            f"Cannot create spec for '{entry}'.")
                    module = importlib.util.module_from_spec(spec)
                    loader.exec_module(module)
                    return module
                except Exception as e:
                    raise ComponentLoadError(
                        f"Failed to load '{entry}': {e!s}")

        raise ComponentLoadError(
            f"No compiled module '{module_name}.pyc' found under '{pyc_dir}'."
        )

    def _wire_model_to_executor(self) -> None:
        """
        Wires a TensorFlow SavedModel signature to the executor’s predictor, ensuring
        that `executor.predict(...)` can be called without attribute or dtype/signature mismatches.

        Raises:
            ComponentLoadError: If the “serving_default” signature is not found,
                if retrieving or wrapping the signature fails, or if the executor
                does not support setting the model and the attribute assignment fails.
        """
        try:
            sig = self.model.signatures.get("serving_default", None)
            if sig is None:
                raise ComponentLoadError(
                    "SavedModel signature 'serving_default' not found.")
        except Exception as e:
            raise ComponentLoadError(
                f"Failed to get serving_default signature: {e!s}")

        class _SignatureModelAdapter:
            """
            Adapter for TensorFlow SavedModel signature functions.

            Wraps a TensorFlow signature function to provide a Keras-like interface for prediction.

            Args:
                signature_fn (Callable): A TensorFlow signature function obtained from a SavedModel, typically
                    via `model.signatures['serving_default']`.
            """

            def __init__(self, signature_fn):
                """
                Initializes the SignatureModelAdapter.

                Args:
                    signature_fn (Callable): The signature function to wrap.
                """
                self._sig = signature_fn

            def predict(self, x):
                """
                Runs inference on input data and returns a NumPy array.

                This method converts the input to a TensorFlow tensor of type float32,
                calls the wrapped signature function, extracts the first output tensor,
                and returns its NumPy value.

                Args:
                    x (array-like or tf.Tensor): Input data to predict on. Can be any object
                        convertible to a TensorFlow tensor of dtype float32.

                Returns:
                    numpy.ndarray: The output of the model as a NumPy array.
                """
                import tensorflow as tf
                x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
                result = self._sig(keras_tensor=x_tf)
                out = next(iter(result.values()))
                return out.numpy()

            def __call__(self, x, training=False):
                """
                Calls the adapter like a Keras model, returning a TensorFlow tensor.

                This method converts the input to a TensorFlow tensor of type float32,
                calls the wrapped signature function, and returns the raw output tensor.

                Args:
                    x (array-like or tf.Tensor): Input data for the model. Must be convertible
                        to a TensorFlow tensor of dtype float32.
                    training (bool, optional): Whether to run in training mode. This adapter
                        does not currently utilize the `training` flag; it is included for
                        interface compatibility. Defaults to False.

                Returns:
                    tf.Tensor: The output tensor from the signature function.
                """
                import tensorflow as tf
                x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
                result = self._sig(keras_tensor=x_tf)
                out = next(iter(result.values()))
                return out

        adapter = _SignatureModelAdapter(sig)
        if hasattr(self.predictor, "set_model") and callable(self.predictor.set_model):
            try:
                self.predictor.set_model(adapter)
            except Exception as e:
                raise ComponentLoadError(
                    f"Executor.set_model(...) failed: {e!s}")
        else:
            try:
                setattr(self.predictor, "model", adapter)
            except Exception as e:
                raise ComponentLoadError(
                    f"Executor does not support setting `.model`: {e!s}")

    def predict(self, data: Any) -> Any:
        """
        Execute the full inference pipeline on raw input data and return a prediction.

        Args:
            data (Any): Raw input data as expected by the data_processor. This can be
                a file path, array, or any format the processor supports.

        Returns:
            Any: The output of the model's `predict` method, typically a NumPy array or
                tensor containing the prediction results.

        Raises:
            ComponentLoadError: If the data_processor fails to process or load the data,
                if the transformer fails to transform the data, or if the predictor
                fails to produce a prediction.
        """
        try:
            if hasattr(self.data_processor, "process_predict") and callable(self.data_processor.process_predict):
                data = self.data_processor.process_predict(data)
            elif hasattr(self.data_processor, "load") and callable(self.data_processor.load):
                data = self.data_processor.load(data)
            else:
                raise ComponentLoadError(
                    "DataProcessor does not implement `process_predict(...)` or `load(...)`."
                )
        except ComponentLoadError:
            raise
        except Exception as e:
            raise ComponentLoadError(f"DataProcessor failed: {e!s}")
        try:
            data = self.transformer.transform(data)
        except Exception as e:
            raise ComponentLoadError(f"Transformer failed: {e!s}")
        try:
            return self.predictor.predict(data)
        except Exception as e:
            raise ComponentLoadError(
                f"Executor.predict(...) failed: {e!s}")

    def predict_with_uncertainty(
        self,
        data: Any,
        n_samples: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the full inference pipeline on input data, returning mean predictions, confidence scores, and raw uncertainty.

        Args:
            data (Any): Raw input to the inference pipeline. Must be accepted by the data_processor.
            n_samples (int, optional): Number of stochastic samples for uncertainty estimation. Defaults to 50.

        Raises:
            ComponentLoadError: If the data_processor or transformer fails, or if the predictor
                does not support uncertainty estimation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                mean_pred (np.ndarray): The mean prediction across `n_samples` runs.
                confidence (np.ndarray): Normalized confidence scores in [0, 1],
                    computed as 1 - (std_pred - std_min) / (std_max - std_min).
                std_pred (np.ndarray): Raw standard deviation of predictions across samples.
        """
        try:
            if hasattr(self.data_processor, "process_predict") and callable(self.data_processor.process_predict):
                processed = self.data_processor.process_predict(data)
            elif hasattr(self.data_processor, "load") and callable(self.data_processor.load):
                processed = self.data_processor.load(data)
            else:
                raise ComponentLoadError(
                    "DataProcessor does not implement `process_predict(...)` or `load(...)`."
                )
        except ComponentLoadError:
            raise
        except Exception as e:
            raise ComponentLoadError(f"DataProcessor failed: {e!s}")

        try:
            transformed = self.transformer.transform(processed)
        except Exception as e:
            raise ComponentLoadError(f"Transformer failed: {e!s}")

        try:
            mean_pred, std_pred = self.predictor.predict(
                transformed, return_uncertainty=True, n_samples=n_samples
            )
        except AttributeError as e:
            raise ComponentLoadError(f"Failed to compute uncertainty: {e!s}")

        std_min, std_max = std_pred.min(), std_pred.max()
        denom = std_max - std_min if std_max > std_min else 1.0
        confidence = 1.0 - (std_pred - std_min) / denom

        return mean_pred, confidence, std_pred

    def update(self, new_data: pd.DataFrame, train_full: bool = False) -> dict:
        """
        Incorporate new training data into the pipeline via transfer learning or incremental updates.

        Args:
            new_data (pd.DataFrame): The new X and y components.
        Returns:
            A dict containing updated components and a timestamp:
              {
                "dataprocessor": < DataProcessor instance > ,
                "transformer":   < Transformer instance > ,
                "model":         < TF model namespace > ,
                "predictor":     < Executor instance > ,
                "updated_at": < ISO8601 timestamp str >
              }

        Raises:
            ComponentLoadError: If any step fails, or if required methods are missing/invalid.
        """
        # Data processing
        if train_full:
            data = self.form_ctrl.get_all_as_dataframe(encoded=True)
        else:
            data = new_data
        try:
            processed = self.data_processor.process_train(data)
        except ComponentLoadError:
            raise
        except Exception as e:
            raise ComponentLoadError(f"DataProcessor.update failed: {e!s}")

        if not (isinstance(processed, tuple) and len(processed) == 2):
            raise ComponentLoadError(
                "DataProcessor.process_train(...) must return (features, labels).")

        features, labels = processed

        # Transformer fit_transform
        if not hasattr(self.transformer, "fit_transform") or not callable(self.transformer.fit_transform):
            raise ComponentLoadError(
                "Transformer does not implement `fit_transform(...)`.")

        try:
            scaled_features = self.transformer.fit_transform(
                features, labels)
        except Exception as e:
            raise ComponentLoadError(
                f"Transformer.fit_transform(...) failed: {e!s}")

        if not hasattr(self.predictor, "update") or not callable(self.predictor.update):
            raise ComponentLoadError(
                "Executor does not implement `update(...)`.")
        updated_model = self.model
        try:
            updated_model = self.predictor.update(
                scaled_features, labels, self.model_path)
        except Exception as e:
            raise ComponentLoadError(f"Executor.update(...) failed: {e!s}")

        return {
            "dataprocessor": self.data_processor,
            "transformer":   self.transformer,
            "model":         updated_model,
            "predictor":     self.predictor,
            "updated_at":    datetime.now().isoformat(),
        }

    def __del__(self):
        """
        Clean up the temporary directory if this instance created one.
        """
        try:
            if self._tempdir is not None:
                self._tempdir.cleanup()
        except Exception:
            pass
