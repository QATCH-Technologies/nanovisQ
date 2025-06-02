"""
test_asset_controller.py

This module validates that AssetController correctly handles creating, reading,
updating, and deleting named assets in a filesystem directory. Each test case
uses temporary directories to simulate real-world usage and ensure that errors
are raised appropriately when invalid operations are attempted.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2025-06-02

Version:
    1.0
"""


import unittest
import tempfile
from pathlib import Path
from src.controller.asset_controller import AssetController, AssetError


class TestAssetController(unittest.TestCase):
    """
    Test suite for AssetController.

    Verifies that:
      - Storing from an existing file works correctly (and errors on missing extension or existing files).
      - Storing raw bytes works correctly (and errors on invalid extension or existing file).
      - Retrieving an existing asset path and loading bytes returns the correct content.
      - asset_exists and list_assets reflect the current directory contents.
      - Updating from a file only succeeds when an existing asset with matching extension is present.
      - Updating raw bytes only succeeds when an existing asset is present with the given extension.
      - Deleting an asset removes the file, and errors when the asset does not exist.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create a temporary project directory and initialize an AssetController
        pointing to a nested 'VisQAI/assets' folder for all tests in this class .
        """
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.project_root = Path(cls._tmpdir.name)
        cls.assets_dir = cls.project_root / "VisQAI" / "assets"
        cls.assets_dir.mkdir(parents=True, exist_ok=True)
        cls.ac = AssetController(cls.assets_dir)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the temporary directory after all tests have run.
        """
        cls._tmpdir.cleanup()

    def setUp(self):
        """
        Before each test, remove any existing files in the assets directory
        to ensure a clean state.
        """
        for f in self.assets_dir.iterdir():
            if f.is_file():
                f.unlink()

    def test_store_from_file_success(self):
        """
        Test that store_from_file copies a valid source file into assets_dir,
        preserves its extension, and returns the correct destination Path.
        """
        src = self.project_root / "tempfile.pkl"
        data = b"binary data 123"
        src.write_bytes(data)

        dest = self.ac.store_from_file("my_model", src, overwrite=False)
        expected = self.assets_dir / "my_model.pkl"

        self.assertEqual(dest, expected)
        self.assertTrue(expected.exists())
        self.assertEqual(expected.read_bytes(), data)

    def test_store_from_file_no_extension(self):
        """
        Test that store_from_file raises AssetError when the source file has no extension.
        """
        src = self.project_root / "noextfile"
        src.write_text("some text")

        with self.assertRaises(AssetError) as cm:
            self.ac.store_from_file("no_ext", src, overwrite=False)
        self.assertIn("no extension", str(cm.exception).lower())

    def test_store_from_file_already_exists_without_overwrite(self):
        """
        Test that store_from_file raises AssetError if the destination already exists
        and overwrite = False.
        """
        src1 = self.project_root / "temp1.pkl"
        src1.write_bytes(b"first")
        self.ac.store_from_file("dup_model", src1, overwrite=False)

        src2 = self.project_root / "temp2.pkl"
        src2.write_bytes(b"second")
        with self.assertRaises(AssetError) as cm:
            self.ac.store_from_file("dup_model", src2, overwrite=False)
        self.assertIn("already exists", str(cm.exception).lower())

    def test_store_from_file_overwrite(self):
        """
        Test that store_from_file replaces an existing asset when overwrite = True.
        """
        src1 = self.project_root / "orig.pkl"
        src1.write_bytes(b"orig data")
        self.ac.store_from_file("overwrite_model", src1, overwrite=False)

        src2 = self.project_root / "newdata.pkl"
        src2.write_bytes(b"new data")
        dest = self.ac.store_from_file("overwrite_model", src2, overwrite=True)

        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_bytes(), b"new data")

    def test_store_bytes_success(self):
        """
        Test that store_bytes writes raw bytes to a new file in assets_dir
        and returns the correct Path.
        """
        content = b"hello world"
        dest = self.ac.store_bytes(
            "greeting", content, ".bin", overwrite=False)
        expected = self.assets_dir / "greeting.bin"

        self.assertEqual(dest, expected)
        self.assertTrue(expected.exists())
        self.assertEqual(expected.read_bytes(), content)

    def test_store_bytes_invalid_extension(self):
        """
        Test that store_bytes raises AssetError when the extension does not start with a dot.
        """
        with self.assertRaises(AssetError) as cm:
            self.ac.store_bytes("bad_ext", b"data", "bin", overwrite=False)
        self.assertIn("invalid extension", str(cm.exception).lower())

    def test_store_bytes_already_exists_without_overwrite(self):
        """
        Test that store_bytes raises AssetError if the destination file already exists
        and overwrite = False.
        """
        self.ac.store_bytes("dup_bytes", b"a", ".dat", overwrite=False)
        with self.assertRaises(AssetError) as cm:
            self.ac.store_bytes("dup_bytes", b"b", ".dat", overwrite=False)
        self.assertIn("already exists", str(cm.exception).lower())

    def test_get_asset_path_and_load_bytes_success(self):
        """
        Test that get_asset_path returns the correct Path and load_bytes returns the file’s content.
        """
        path = self.assets_dir / "example.txt"
        content = b"example content"
        path.write_bytes(content)

        found = self.ac.get_asset_path("example", [".txt", ".bin"])
        self.assertEqual(found, path)

        loaded = self.ac.load_bytes("example", [".txt", ".bin"])
        self.assertEqual(loaded, content)

    def test_get_asset_path_not_found(self):
        """
        Test that get_asset_path raises AssetError when no matching file is found.
        """
        with self.assertRaises(AssetError) as cm:
            self.ac.get_asset_path("no_such", [".pkl", ".joblib"])
        self.assertIn("no asset found", str(cm.exception).lower())

    def test_load_bytes_not_found(self):
        """
        Test that load_bytes raises AssetError when no matching file is found.
        """
        with self.assertRaises(AssetError) as cm:
            self.ac.load_bytes("missing", [".dat", ".bin"])
        self.assertIn("no asset found", str(cm.exception).lower())

    def test_asset_exists_and_list_assets(self):
        """
        Test that asset_exists correctly identifies existing files and that
        list_assets returns all logical names present in the directory.
        """
        self.assertFalse(self.ac.asset_exists("nothing", [".pkl", ".joblib"]))
        self.assertEqual(self.ac.list_assets(), [])

        self.ac.store_bytes("first", b"1", ".a", overwrite=False)
        self.ac.store_bytes("second", b"2", ".b", overwrite=False)

        self.assertTrue(self.ac.asset_exists("first", [".a", ".x"]))
        self.assertTrue(self.ac.asset_exists("second", [".b", ".y"]))
        self.assertFalse(self.ac.asset_exists("first", [".x", ".y"]))

        self.assertEqual(sorted(self.ac.list_assets()), ["first", "second"])

    def test_update_from_file_success(self):
        """
        Test that update_from_file replaces the existing asset file when the extensions match.
        """
        src1 = self.project_root / "initial.pcl"
        src1.write_bytes(b"init")
        self.ac.store_from_file("to_update", src1, overwrite=False)

        src2 = self.project_root / "new.pcl"
        src2.write_bytes(b"updated")
        updated_path = self.ac.update_from_file("to_update", src2)

        self.assertTrue(updated_path.exists())
        self.assertEqual(updated_path.read_bytes(), b"updated")

    def test_update_from_file_no_existing(self):
        """
        Test that update_from_file raises AssetError when no existing asset is found.
        """
        src = self.project_root / "whatever.pkl"
        src.write_bytes(b"data")

        with self.assertRaises(AssetError) as cm:
            self.ac.update_from_file("noexist", src)
        self.assertIn("no asset found", str(cm.exception).lower())

    def test_update_from_file_extension_mismatch(self):
        """
        Test that update_from_file raises AssetError when the source file’s extension
        does not match the existing asset’s extension.
        """
        bin_src = self.project_root / "f1.bin"
        bin_src.write_bytes(b"x")
        self.ac.store_from_file("mismatch", bin_src, overwrite=False)

        txt_src = self.project_root / "f2.txt"
        txt_src.write_bytes(b"y")
        with self.assertRaises(AssetError) as cm:
            self.ac.update_from_file("mismatch", txt_src)
        self.assertIn("no asset found", str(cm.exception).lower())

    def test_update_bytes_success(self):
        """
        Test that update_bytes correctly overwrites the existing asset’s contents.
        """
        self.ac.store_bytes("upd_bytes", b"one", ".upd", overwrite=False)
        updated_path = self.ac.update_bytes("upd_bytes", b"two", ".upd")

        self.assertTrue(updated_path.exists())
        self.assertEqual(updated_path.read_bytes(), b"two")

    def test_update_bytes_no_existing(self):
        """
        Test that update_bytes raises AssetError when no existing asset is found to update.
        """
        with self.assertRaises(AssetError) as cm:
            self.ac.update_bytes("absent", b"data", ".zzz")
        self.assertIn("no asset found", str(cm.exception).lower())

    def test_update_bytes_invalid_extension(self):
        """
        Test that update_bytes raises AssetError when the provided extension does not start with a dot.
        """
        with self.assertRaises(AssetError) as cm:
            self.ac.update_bytes("anyname", b"data", "zzz")
        self.assertIn("invalid extension", str(cm.exception).lower())

    def test_delete_asset_success(self):
        """
        Test that delete_asset removes the specified file and does not affect other assets.
        """
        self.ac.store_bytes("delme", b"1", ".a", overwrite=False)
        self.ac.store_bytes("delme_alias", b"2", ".b", overwrite=False)

        self.ac.delete_asset("delme", [".a", ".x"])
        self.assertFalse(self.ac.asset_exists("delme", [".a", ".x"]))
        self.assertTrue(self.ac.asset_exists("delme_alias", [".b"]))

    def test_delete_asset_not_found(self):
        """
        Test that delete_asset raises AssetError when attempting to delete a non-existent asset.
        """
        with self.assertRaises(AssetError) as cm:
            self.ac.delete_asset("nope", [".a", ".b"])
        self.assertIn("no asset found", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
