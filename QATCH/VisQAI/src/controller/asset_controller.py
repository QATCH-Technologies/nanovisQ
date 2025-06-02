"""
asset_controller.py

This module defines the AssetController class, which manages named assets on disk
under a single assets directory. Each asset is identified by a logical name
(basename without extension) and one or more possible file extensions
(e.g., ".pkl", ".joblib", ".h5", ".json", etc.). The module provides CRUD
operations for assets:

  - Create:
        store_from_file(logical_name, source_path, overwrite=False)
        store_bytes(logical_name, data, extension, overwrite=False)

  - Read:
        get_asset_path(logical_name, extensions)
        load_bytes(logical_name, extensions)
        asset_exists(logical_name, extensions)
        list_assets()

  - Update:
        update_from_file(logical_name, source_path)
        update_bytes(logical_name, data, extension)

  - Delete:
        delete_asset(logical_name, extensions)

Any failure to locate, read, write, or delete an asset raises AssetError.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2025-06-02

Version:
    1.0
"""
from pathlib import Path
from typing import Optional, Union


class AssetError(Exception):
    """
    Raised whenever an asset cannot be found, loaded, stored, updated, or deleted.
    """
    pass


class AssetController:
    """
    Manages “named” assets on disk under a single assets directory.

    Each asset is identified by a logical name (basename without extension)
    and one or more possible file extensions (e.g. ".pkl", ".joblib", ".h5", ".json", etc.).

    CRUD methods provided:
      - Create: store_from_file(), store_bytes()
      - Read:   get_asset_path(), load_bytes(), asset_exists(), list_assets()
      - Update: update_from_file(), update_bytes()
      - Delete: delete_asset()

    Usage sketch:
        acct = AssetController("/path/to/assets_dir")

        # Create from an existing file:
        acct.store_from_file("my_model", "/tmp/temp_model.pkl", overwrite=False)
        # Create from raw bytes:
        acct.store_bytes("greeting", b"hello", ".bin", overwrite=False)

        # Read:
        asset_path = acct.get_asset_path("my_model", [".pkl", ".joblib"])
        contents   = acct.load_bytes("greeting", [".bin"])

        # Update (overwrite only if an existing asset is already present):
        acct.update_from_file("my_model", "/tmp/new_model.pkl")
        acct.update_bytes("greeting", b"updated bytes", ".bin")

        # Delete:
        acct.delete_asset("my_model", [".pkl", ".joblib"])

        # List all logical names:
        all_names = acct.list_assets()
    """

    def __init__(self, assets_dir: Union[str, Path]) -> None:
        """
        Initialize the AssetController with a given assets directory.

        Args:
            assets_dir: Path to (or to-be-created) directory where assets will live.

        Raises:
            AssetError: If the assets directory cannot be created or accessed.
        """
        self.assets_dir = Path(assets_dir)
        try:
            self.assets_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise AssetError(
                f"Could not create or access assets directory '{self.assets_dir}': {e!s}"
            )

    def _resolve_asset_path(
        self,
        logical_name: str,
        extensions: list[str],
        must_exist: bool = True
    ) -> Optional[Path]:
        """
        Internal helper: locate a file under self.assets_dir matching logical_name + one of the extensions.

        Comparison is case-sensitive on most OSes, but matching is done against lowercase names.

        Args:
            logical_name: Filename without extension (e.g. "model_v1").
            extensions: List of extensions to try, e.g., [".pkl", ".joblib"].
            must_exist: 
                - If True, raises AssetError when no matching file is found.
                - If False, returns the candidate Path for the first extension (even if it does not exist).

        Returns:
            Path: Path to the first matching file if must_exist=True and found.
            Path: Path to the candidate file using the first extension if must_exist=False.
            None: If extensions list is empty and must_exist=False.

        Raises:
            AssetError: If must_exist=True and no matching file is found.
        """
        lower_name = logical_name.lower()
        for ext in extensions:
            candidate = self.assets_dir / f"{logical_name}{ext}"
            if candidate.exists() and candidate.is_file():
                return candidate

        if must_exist:
            raise AssetError(
                f"No asset found for name '{logical_name}' with extensions {extensions} "
                f"in '{self.assets_dir}'."
            )

        if extensions:
            return self.assets_dir / f"{logical_name}{extensions[0]}"
        return None

    def store_from_file(
        self,
        logical_name: str,
        source_path: Union[str, Path],
        overwrite: bool = False
    ) -> Path:
        """
        Copy an existing file on disk into the assets directory, preserving its extension.

        Args:
            logical_name: New name (without extension) under assets_dir.
            source_path: Path to an existing file; its extension will be preserved.
            overwrite: 
                - If False, raises AssetError if the destination already exists.
                - If True, replaces any existing file with the same name and extension.

        Returns:
            Path: Path to the newly stored asset file.

        Raises:
            AssetError:
                - If source_path is invalid or not a file.
                - If source has no extension.
                - If destination exists and overwrite=False.
                - If the copy operation fails.
        """
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise AssetError(
                f"Source asset '{src}' does not exist or is not a file."
            )

        ext = src.suffix
        if ext == "":
            raise AssetError(f"Source '{src}' has no extension to preserve.")

        dest = self.assets_dir / f"{logical_name}{ext}"
        if dest.exists() and not overwrite:
            raise AssetError(
                f"Asset '{logical_name}{ext}' already exists at '{dest}'. "
                "Set overwrite=True to replace."
            )

        try:
            if dest.exists() and overwrite:
                dest.unlink()
            data = src.read_bytes()
            dest.write_bytes(data)
            return dest
        except Exception as e:
            raise AssetError(f"Failed to copy '{src}' to '{dest}': {e!s}")

    def store_bytes(
        self,
        logical_name: str,
        data: bytes,
        extension: str,
        overwrite: bool = False
    ) -> Path:
        """
        Write raw bytes into a new file named <logical_name><extension> in assets_dir.

        Args:
            logical_name: Filename (without extension) under assets_dir.
            data: Bytes to write to the file.
            extension: File extension (e.g., ".bin", ".json", ".pkl"). Must start with a dot.
            overwrite: 
                - If False, raises AssetError if a file with the same name already exists.
                - If True, replaces any existing file.

        Returns:
            Path: Path to the newly written asset.

        Raises:
            AssetError:
                - If extension does not start with a dot.
                - If destination exists and overwrite=False.
                - If writing fails.
        """
        if not extension.startswith("."):
            raise AssetError(
                f"Invalid extension '{extension}'. Must begin with a period, e.g. '.bin'."
            )

        dest = self.assets_dir / f"{logical_name}{extension}"
        if dest.exists() and not overwrite:
            raise AssetError(
                f"Asset '{logical_name}{extension}' already exists. Set overwrite=True to replace."
            )

        try:
            dest.write_bytes(data)
            return dest
        except Exception as e:
            raise AssetError(f"Failed to write bytes to '{dest}': {e!s}")

    def get_asset_path(
        self,
        logical_name: str,
        extensions: list[str]
    ) -> Path:
        """
        Retrieve the Path of an existing asset by logical name, trying each extension until one is found.

        Args:
            logical_name: Base filename without extension (e.g., "model_v2").
            extensions: List of allowed file extensions (e.g., [".pkl", ".joblib", ".h5"]).

        Returns:
            Path: Path to the first existing file that matches logical_name + ext.

        Raises:
            AssetError: If no matching file is found.
        """
        return self._resolve_asset_path(logical_name, extensions, must_exist=True)

    def load_bytes(
        self,
        logical_name: str,
        extensions: list[str]
    ) -> bytes:
        """
        Load and return the raw bytes of an existing asset file.

        Args:
            logical_name: Filename without extension (e.g., "config").
            extensions: List of allowed file extensions to try, in order of preference.

        Returns:
            bytes: Contents of the first found file matching logical_name + ext.

        Raises:
            AssetError:
                - If no matching file is found.
                - If reading the file fails.
        """
        path = self.get_asset_path(logical_name, extensions)
        try:
            return path.read_bytes()
        except Exception as e:
            raise AssetError(f"Failed to read bytes from '{path}': {e!s}")

    def asset_exists(
        self,
        logical_name: str,
        extensions: list[str]
    ) -> bool:
        """
        Check if any file matching logical_name + ext exists under assets_dir.

        Args:
            logical_name: Filename without extension.
            extensions: List of extensions to try.

        Returns:
            bool: True if a matching file is found, False otherwise.
        """
        try:
            self._resolve_asset_path(logical_name, extensions, must_exist=True)
            return True
        except AssetError:
            return False

    def list_assets(self) -> list[str]:
        """
        List all logical asset names stored in the assets directory.

        Returns:
            list[str]: Sorted list of unique logical names (filenames without extensions)
                       for all files directly under assets_dir.
        """
        names = set()
        for child in self.assets_dir.iterdir():
            if child.is_file():
                names.add(child.stem)
        return sorted(names)

    def update_from_file(
        self,
        logical_name: str,
        source_path: Union[str, Path]
    ) -> Path:
        """
        Replace an existing asset (with the same extension) using a new file.

        This method only succeeds if an asset with the same logical_name and extension
        already exists in assets_dir. Otherwise, it raises an AssetError.

        Args:
            logical_name: Base filename (without extension) to update (e.g., "model_v1").
            source_path: Path to a new file whose extension must match the existing asset’s extension.

        Returns:
            Path: Path to the updated asset file (<assets_dir>/<logical_name><ext>).

        Raises:
            AssetError:
                - If no existing asset is found under logical_name.
                - If source_path is invalid or not a file.
                - If source_path has no extension.
                - If source_path’s extension does not match the existing asset’s extension.
                - If the write operation fails.
        """
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise AssetError(
                f"Source asset '{src}' does not exist or is not a file."
            )

        ext = src.suffix
        if ext == "":
            raise AssetError(f"Source '{src}' has no extension to preserve.")

        existing = self._resolve_asset_path(
            logical_name, [ext], must_exist=True)

        try:
            existing.unlink()
            data = src.read_bytes()
            existing.write_bytes(data)
            return existing
        except Exception as e:
            raise AssetError(
                f"Failed to update '{existing}' with '{src}': {e!s}")

    def update_bytes(
        self,
        logical_name: str,
        data: bytes,
        extension: str
    ) -> Path:
        """
        Replace an existing asset’s contents by writing new bytes.

        This only works if <assets_dir>/<logical_name><extension> already exists.

        Args:
            logical_name: Base filename (without extension) under assets_dir.
            data: New bytes to write.
            extension: File extension (must begin with a dot, e.g., ".bin", ".pkl").

        Returns:
            Path: Path to the updated asset file.

        Raises:
            AssetError:
                - If extension does not start with a dot.
                - If no existing file matching <logical_name><extension> is found.
                - If writing fails.
        """
        if not extension.startswith("."):
            raise AssetError(
                f"Invalid extension '{extension}'. Must begin with a period, e.g., '.bin'."
            )

        existing = self._resolve_asset_path(
            logical_name, [extension], must_exist=True)

        try:
            existing.unlink()
            existing.write_bytes(data)
            return existing
        except Exception as e:
            raise AssetError(f"Failed to update bytes in '{existing}': {e!s}")

    def delete_asset(
        self,
        logical_name: str,
        extensions: list[str]
    ) -> None:
        """
        Delete an existing asset by logical name, trying each extension until found.

        Args:
            logical_name: Filename without extension.
            extensions: List of extensions to try.

        Raises:
            AssetError:
                - If no matching file is found.
                - If file deletion fails.
        """
        path = self.get_asset_path(logical_name, extensions)
        try:
            path.unlink()
        except Exception as e:
            raise AssetError(f"Failed to delete asset '{path}': {e!s}")
