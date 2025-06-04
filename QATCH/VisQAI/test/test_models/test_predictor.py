from src.models.predictor import Predictor, ComponentLoadError
import os
import sys
import tempfile
import zipfile
import shutil
import py_compile
import unittest
from pathlib import Path
from unittest import mock

import cloudpickle
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def _create_dummy_dataprocessor(pkg_dir: Path, module_name: str):
    """
    Create a dummy dataprocessor.py, compile it to .pyc, and place it under __pycache__.
    The DataProcessor class provides:
      - process(raw_input) -> raw_input * 2  (for predict)
      - process(X, y) -> (X + 1, y + 1)     (for update)
    """
    src_file = pkg_dir / f"{module_name}.py"
    pycache_dir = pkg_dir / "__pycache__"
    pycache_dir.mkdir(exist_ok=True)

    content = '''
class DataProcessor:
    def __init__(self):
        pass

    def process(self, *args):
        # If one argument, double it for prediction
        if len(args) == 1:
            raw = args[0]
            try:
                return raw * 2
            except Exception:
                return raw
        # If two arguments, treat as (X, y) and return (X+1, y+1)
        elif len(args) == 2:
            X, y = args
            return (X + 1, y + 1)
        else:
            raise ValueError("Unexpected number of arguments")
'''
    src_file.write_text(content)
    compiled_path = pycache_dir / \
        f"{module_name}.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    py_compile.compile(str(src_file), cfile=str(compiled_path))


def _create_dummy_predictor(pkg_dir: Path, module_name: str):
    """
    Create a dummy predictor.py, compile it to .pyc, and place it under __pycache__.
    The Predictor class provides:
      - set_model(model): stores model
      - predict(data): returns model.predict(data) + 5
      - update(X, y, model): stores last_update = (X, y, model)
    """
    src_file = pkg_dir / f"{module_name}.py"
    pycache_dir = pkg_dir / "__pycache__"
    pycache_dir.mkdir(exist_ok=True)

    content = '''
class Predictor:
    def __init__(self):
        self._model = None
        self.last_update = None

    def set_model(self, model):
        self._model = model

    def predict(self, data):
        # Emulate calling the wrapped model: base + 5
        if self._model is not None and hasattr(self._model, "predict"):
            base = self._model.predict(data)
            try:
                return base + 5
            except Exception:
                return base
        return data + 5

    def update(self, X, y, model=None):
        self.last_update = (X, y, model)
'''
    src_file.write_text(content)
    compiled_path = pycache_dir / \
        f"{module_name}.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
    py_compile.compile(str(src_file), cfile=str(compiled_path))


def _create_dummy_transformer(pkg_dir: Path, name: str):
    """
    Create a dummy transformer object with transform and fit_transform methods.
    - transform(X) returns X * 10
    - fit_transform(X, y) returns X * 20
    Serialize to <name>.pkl via cloudpickle.
    """
    class DummyTransformer:
        def transform(self, data):
            return data * 10

        def fit_transform(self, X, y=None):
            return X * 20

    transformer = DummyTransformer()
    pkl_path = pkg_dir / f"{name}.pkl"
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(transformer, f)


def _create_dummy_saved_model(pkg_dir: Path, name: str):
    """
    Create a dummy SavedModel directory structure. We'll monkeypatch
    tf.saved_model.load(...) in tests to return a stub object.
    """
    sm_dir = pkg_dir / name
    sm_dir.mkdir(exist_ok=True)
    (sm_dir / "assets").mkdir(exist_ok=True)
    (sm_dir / "variables").mkdir(exist_ok=True)
    (sm_dir / "saved_model.pb").write_text("")


class TestPredictorModule(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.pkg_dir = self.root / "package"
        self.pkg_dir.mkdir()
        (self.pkg_dir / "__init__.py").write_text("# Init")
        _create_dummy_dataprocessor(self.pkg_dir, "dataprocessor")
        _create_dummy_predictor(self.pkg_dir, "predictor")
        (self.pkg_dir / "__pycache__").mkdir(exist_ok=True)
        _create_dummy_transformer(self.pkg_dir, "transformer")
        _create_dummy_saved_model(self.pkg_dir, "model")
        self.zip_path = self.root / "bundle.zip"
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder, _, files in os.walk(self.pkg_dir):
                for file in files:
                    full_path = Path(folder) / file
                    rel_path = full_path.relative_to(self.root)
                    zf.write(full_path, rel_path)
        patch_load = mock.patch(
            "src.models.predictor.tf.saved_model.load", new=self._dummy_tf_load
        )
        patch_tf_flag = mock.patch(
            "src.models.predictor._TF_AVAILABLE", new=True)
        patch_load.start()
        patch_tf_flag.start()
        self.addCleanup(patch_load.stop)
        self.addCleanup(patch_tf_flag.stop)

    def tearDown(self):
        self.tempdir.cleanup()

    def _dummy_tf_load(self, path):
        """
        Returns a stub with:
          signatures = {"serving_default": function}
        The function returns a DummyTensor whose numpy() method returns input + 3.
        """
        class DummyTensor:
            def __init__(self, arr):
                self._arr = arr

            def numpy(self):
                return self._arr

        class DummySignature:
            def __call__(self, tensor):
                try:
                    val = tensor.numpy()
                except Exception:
                    val = tensor
                return {"output": DummyTensor(val + 3)}

        class DummySavedModel:
            def __init__(self):
                self.signatures = {"serving_default": DummySignature()}

        return DummySavedModel()

    def test_successful_load_from_directory_and_predict(self):
        predictor = Predictor(self.root)
        result = predictor.predict(2)
        self.assertEqual(result, 48)

    def test_successful_load_from_zip_and_predict(self):
        predictor = Predictor(self.zip_path)
        result = predictor.predict(2)
        self.assertEqual(result, 48)

    def test_missing_package_dir_raises(self):
        shutil.rmtree(self.pkg_dir)
        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn("Expected 'package/'", str(ctx.exception))

    def test_missing_pyc_raises(self):
        shutil.rmtree(self.pkg_dir / "__pycache__")
        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn("Missing '__pycache__'", str(ctx.exception))

    def test_pyc_without_DataProcessor_class_raises(self):
        no_dp_src = self.pkg_dir / "dataprocessor_no_dp.py"
        no_dp_src.write_text("x = 1")
        compiled_path = self.pkg_dir / "__pycache__" / \
            f"dataprocessor_no_dp.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        py_compile.compile(str(no_dp_src), cfile=str(compiled_path))
        (self.pkg_dir / "__pycache__" / f"dataprocessor_no_dp.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc").replace(
            self.pkg_dir / "__pycache__" /
            f"dataprocessor.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        )

        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn(
            "does not define a callable DataProcessor class", str(ctx.exception))

    def test_missing_transformer_file_raises(self):
        os.remove(self.pkg_dir / "transformer.pkl")
        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn("Missing transformer file 'transformer.pkl'",
                      str(ctx.exception))

    def test_transformer_without_transform_method_raises(self):
        class BadTransformer:
            pass

        with open(self.pkg_dir / "transformer.pkl", "wb") as f:
            cloudpickle.dump(BadTransformer(), f)

        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn(
            "Transformer must implement callable transform(...) and fit_transform(...).", str(ctx.exception))

    def test_missing_saved_model_dir_raises(self):
        shutil.rmtree(self.pkg_dir / "model")
        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn(
            "Missing SavedModel directory 'model/'", str(ctx.exception))

    def test_tf_not_available_raises(self):
        import src.models.predictor as mod
        orig_flag = mod._TF_AVAILABLE
        try:
            mod._TF_AVAILABLE = False
            with self.assertRaises(ComponentLoadError) as ctx:
                Predictor(self.root)
            self.assertIn("TensorFlow is not installed", str(ctx.exception))
        finally:
            mod._TF_AVAILABLE = orig_flag

    def test_saved_model_no_signature_raises(self):
        def no_sig_load(path):
            class DummyNoSig:
                def __init__(self):
                    self.signatures = {}
            return DummyNoSig()

        with mock.patch("src.models.predictor.tf.saved_model.load", new=no_sig_load):
            with self.assertRaises(ComponentLoadError) as ctx:
                Predictor(self.root)
            self.assertIn("has no 'serving_default' signature",
                          str(ctx.exception))

    def test_missing_executor_class_raises(self):
        no_exec_src = self.pkg_dir / "predictor_no_exec.py"
        no_exec_src.write_text("x = 2")
        compiled_path = self.pkg_dir / "__pycache__" / \
            f"predictor_no_exec.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        py_compile.compile(str(no_exec_src), cfile=str(compiled_path))
        (self.pkg_dir / "__pycache__" / f"predictor_no_exec.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc").replace(
            self.pkg_dir / "__pycache__" /
            f"predictor.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        )

        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn("does not define a callable Predictor class",
                      str(ctx.exception))

    def test_executor_missing_predict_raises(self):
        bad_exec_src = self.pkg_dir / "predictor_bad_exec.py"
        bad_exec_src.write_text('''
class Predictor:
    def __init__(self):
        pass
''')
        compiled_path = self.pkg_dir / "__pycache__" / \
            f"predictor_bad_exec.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        py_compile.compile(str(bad_exec_src), cfile=str(compiled_path))
        # Replace the original predictor pyc
        (self.pkg_dir / "__pycache__" / f"predictor_bad_exec.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc").replace(
            self.pkg_dir / "__pycache__" /
            f"predictor.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        )
        predictor = Predictor(self.root)
        with self.assertRaises(ComponentLoadError) as ctx:
            predictor.predict(1)
        self.assertIn("does not implement `predict(...)`", str(ctx.exception))

    def test_update_success(self):
        predictor = Predictor(self.root)
        result = predictor.update(1, 2)
        self.assertIn("updated_at", result)
        self.assertIsInstance(result["dataprocessor"], object)
        self.assertIsInstance(result["transformer"], object)
        self.assertEqual(result["predictor"].last_update[0], 40)
        self.assertEqual(result["predictor"].last_update[1], 3)
        self.assertEqual(result["predictor"].last_update[2], result["model"])

    def test_update_missing_fit_transform_raises(self):
        class BadTransformer2:
            def transform(self, data):
                return data

        with open(self.pkg_dir / "transformer.pkl", "wb") as f:
            cloudpickle.dump(BadTransformer2(), f)

        with self.assertRaises(ComponentLoadError) as ctx:
            Predictor(self.root)
        self.assertIn(
            "Transformer must implement callable transform(...) and fit_transform(...).", str(ctx.exception))

    def test_update_executor_missing_update_raises(self):
        no_upd_src = self.pkg_dir / "predictor_no_update.py"
        no_upd_src.write_text('''
class Predictor:
    def __init__(self):
        pass

    def predict(self, data):
        return data
''')
        compiled_path = self.pkg_dir / "__pycache__" / \
            f"predictor_no_update.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        py_compile.compile(str(no_upd_src), cfile=str(compiled_path))
        (self.pkg_dir / "__pycache__" / f"predictor_no_update.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc").replace(
            self.pkg_dir / "__pycache__" /
            f"predictor.cpython-{sys.version_info.major}{sys.version_info.minor}.pyc"
        )

        predictor = Predictor(self.root)
        with self.assertRaises(ComponentLoadError) as ctx:
            predictor.update(1, 2)
        self.assertIn("Executor does not implement `update(...)`",
                      str(ctx.exception))

    def test_load_real_bundle_methods_callable(self):
        bundle_path = Path(__file__).parent / "assets" / "VisQAI-base.zip"
        self.assertTrue(bundle_path.is_file(),
                        f"Bundle not found: {bundle_path}")

        predictor = Predictor(bundle_path)

        self.assertTrue(hasattr(predictor, "data_processor"))
        self.assertTrue(callable(predictor.data_processor.process))

        self.assertTrue(hasattr(predictor, "transformer"))
        self.assertTrue(callable(predictor.transformer.transform))
        self.assertTrue(callable(predictor.transformer.fit_transform))

        self.assertTrue(hasattr(predictor, "model"))
        self.assertTrue(hasattr(predictor.model, "predict"))
        self.assertTrue(callable(predictor.model.predict))

        self.assertTrue(hasattr(predictor, "predictor"))
        self.assertTrue(hasattr(predictor.predictor, "predict"))
        self.assertTrue(callable(predictor.predictor.predict))
        self.assertTrue(hasattr(predictor.predictor, "update"))
        self.assertTrue(callable(predictor.predictor.update))

        self.assertTrue(callable(predictor.predict))
        self.assertTrue(callable(predictor.update))


if __name__ == "__main__":
    unittest.main()
