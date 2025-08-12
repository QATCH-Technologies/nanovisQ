from typing import Any, Dict
import os
import json
import hashlib
import shutil
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
from datetime import datetime
from datetime import timezone as tz

_INDEX_FILENAME = "index.json"
_OBJECTS_DIR = "objects"
_DEFAULT_RETENTION = 50
_SHA256_HEX_RE = re.compile(r'^[0-9a-f]{64}$')

# NOTE: VisQAI-base.zip library will be pruned with this class after the retention period.
#       Software should invoke front-end protections to automatically pin the base library
#       and prevent it from being unpinned by the user.


class VersionManager:
    def __init__(self, repo_dir: str, retention: int = _DEFAULT_RETENTION) -> None:
        """Initializes a content-addressed snapshot repository.

        Creates the necessary directory structure and loads (or creates) the index file.

        Args:
            repo_dir (str): Path to the root of the version repository.
            retention (int): Maximum number of snapshots to keep. Must be >= 1.

        Raises:
            TypeError: If `repo_dir` is not a string.
            ValueError: If `retention` is not a positive integer.
        """
        if not isinstance(repo_dir, str):
            raise TypeError(
                f"repo_dir must be a str, got {type(repo_dir).__name__}")
        if not isinstance(retention, int) or retention < 1:
            raise ValueError(
                f"retention must be a positive integer, got {retention}")

        # Core paths
        self.repo_dir = Path(repo_dir)
        self.objects_dir = self.repo_dir / _OBJECTS_DIR
        self.index_path = self.repo_dir / _INDEX_FILENAME

        # Settings
        self.retention = retention
        self._lock = threading.Lock()

        # Ensure directories and index exist
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({})

        # Load into memory
        self._load_index()

    def _init_dirs(self) -> None:
        """Ensure the repository structure exists and initialize the index.

        This method will create the root repo directory (if missing), the
        `objects/` subdirectory, and an empty index.json file if it doesn’t yet
        exist.

        Raises:
            OSError: If any of the required directories cannot be created.
        """
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self._write_index({})

    def _load_index(self) -> Dict[str, Any]:
        """Load the index.json file into memory.

        Reads the index file mapping SHA hashes to their metadata and
        stores it in `self._index`.

        Returns:
            Dict[str, Any]: The in-memory index mapping snapshot SHA to metadata.

        Raises:
            FileNotFoundError: If the index file does not exist.
            json.JSONDecodeError: If the index file contains invalid JSON.
        """
        with self.index_path.open("r") as f:
            self._index = json.load(f)
        return self._index

    def _write_index(self, idx: Dict[str, Any]) -> None:
        """Atomically write the index dictionary to a JSON file on disk.


        Args:
            idx: The index data to serialize and write.

        Raises:
            ValueError: If `idx` cannot be serialized to JSON.
            OSError: If an I/O error occurs during write, fsync, or replace.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + '.tmp')
        try:
            # Write JSON to temporary file
            with tmp_path.open('w', encoding='utf-8') as f:
                json.dump(idx, f, indent=2, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            # Atomically replace the old index file
            tmp_path.replace(self.index_path)
            # Update in-memory index only on success
            self._index = idx

        except (TypeError, ValueError) as exc:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"Failed to serialize index to JSON: {exc}") from exc
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise OSError(f"Failed to write index file: {exc}") from exc

    def _hash_file(self, file_path: Path) -> str:
        """Compute the SHA-256 hash of a file's contents in a memory-efficient way.

        This method reads the file in fixed-size chunks to avoid loading
        the entire file into memory, and returns the hexadecimal digest.

        Args:
            file_path: Path to the file to be hashed.

        Returns:
            The SHA-256 hex digest of the file's contents.

        Raises:
            FileNotFoundError: If `file_path` does not exist or is not a regular file.
            PermissionError: If the file cannot be opened for reading.
            OSError: For other I/O errors during read.
            ValueError: If `file_path` is not a Path instance.
        """
        # Validate type
        if not isinstance(file_path, Path):
            raise ValueError(
                f"`file_path` must be a pathlib.Path, got {type(file_path)}")

        # Ensure the file exists and is a regular file
        if not file_path.is_file():
            raise FileNotFoundError(f"No such file: {file_path}")

        hasher = hashlib.sha256()
        chunk_size = 8192

        try:
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hasher.update(chunk)
        except PermissionError as exc:
            raise PermissionError(
                f"Permission denied when reading {file_path}") from exc
        except OSError as exc:
            raise OSError(
                f"I/O error while reading {file_path}: {exc}") from exc

        return hasher.hexdigest()

    def _object_path(self, sha: str) -> Path:
        """Compute the on-disk path for a content-addressed object.

        Args:
            sha: A 64-character, lowercase hexadecimal SHA-256 digest.

        Returns:
            A `Path` pointing at `<objects_dir>/<first_two_hex>/<full_digest>`.

        Raises:
            ValueError: If `sha` is not a valid 64-char lowercase hex string.
        """
        if not isinstance(sha, str):
            raise ValueError(f"Expected `sha` to be a str, got {type(sha)}")
        if not _SHA256_HEX_RE.match(sha):
            raise ValueError(f"Invalid SHA-256 hex digest: {sha!r}")

        # Shared directory by first two hex characters
        shared = sha[:2]
        return self.objects_dir / shared / sha

    def commit(self, model_file: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new model snapshot to the repository, automically storing the binary and metadata.

        This method computes the SHA-256 of the given model file, copies it into a
        content-addressed shared directory, writes metadata (including timestamp) via
        an atomic write, updates the central index atomically, and prunes old snapshots
        if retention limits are exceeded.

        Args:
            model_file: Path to the binary model file to commit.
            metadata: Optional dict of JSON-serializable metadata.

        Returns:
            The SHA-256 hex digest (ID) of the committed snapshot.

        Raises:
            FileNotFoundError: If `model_file` does not exist or is not a file.
            ValueError: If `metadata` is not JSON-serializable.
            OSError: For I/O errors during file operations.
        """
        model_path = Path(model_file)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Compute content hash
        sha = self._hash_file(model_path)
        obj_dir = self._object_path(sha)
        metadata = metadata or {}
        committed_at = datetime.now(tz.utc).replace(
            microsecond=0).isoformat() + "Z"
        meta = {
            "sha": sha,
            "committed_at": committed_at,
            "filename": model_path.name,
            "metadata": metadata,
        }

        # Ensure only one thread writes this SHA at once
        with self._lock:
            # If already committed, skip
            if obj_dir.exists():
                return sha

            try:
                # Create shared directory
                obj_dir.mkdir(parents=True, exist_ok=False)
                dest = obj_dir / model_path.name
                shutil.copy2(model_path, dest)
                meta_path = obj_dir / "meta.json"
                tmp_meta = obj_dir / "meta.json.tmp"
                with tmp_meta.open("w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2, sort_keys=True)
                    mf.flush()
                    mf.fileno() and os.fsync(mf.fileno())
                tmp_meta.replace(meta_path)

                # Update index
                index = self._load_index()
                index[sha] = meta
                self._write_index(index)

                # Prune old snapshots
                self.prune()

            except (TypeError, ValueError) as exc:
                shutil.rmtree(obj_dir, ignore_errors=True)
                raise ValueError(
                    f"Metadata serialization failed: {exc}") from exc

            except OSError as exc:
                shutil.rmtree(obj_dir, ignore_errors=True)
                raise OSError(
                    f"Failed to commit snapshot {sha}: {exc}") from exc
        return sha

    def get(self, sha: str, dest_dir: str) -> Path:
        """Retrieve a snapshot by its hash, copying it into a destination directory.

        This method looks up the snapshot metadata by SHA-256 hash in the central index,
        ensures the stored binary exists, creates the destination directory if needed,
        and copies the model file there.

        Args:
            sha: The SHA-256 hex digest of a committed snapshot.
            dest_dir: Path to the directory where the snapshot file should be copied.

        Returns:
            A `Path` to the copied model file in `dest_dir`.

        Raises:
            KeyError: If no snapshot with the given `sha` exists.
            FileNotFoundError: If the snapshot binary is missing on disk.
            PermissionError: If unable to read the source or write to the destination.
            OSError: For other I/O errors during directory creation or file copy.
        """
        index: Dict[str, Any] = self._load_index()
        if sha not in index:
            raise KeyError(f"No snapshot with hash {sha}")

        meta = index[sha]
        src_dir = self._object_path(sha)
        src_file = src_dir / meta["filename"]

        if not src_file.is_file():
            raise FileNotFoundError(f"Snapshot binary not found at {src_file}")
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        dest_file = dest_path / meta["filename"]
        shutil.copy2(src_file, dest_file)

        return dest_file

    def list(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List committed snapshots in reverse chronological order.

        Args:
            limit: Maximum number of entries to return. If None, returns all.

        Returns:
            A list of metadata dicts (each containing keys like 'sha',
            'committed_at', 'filename', and 'metadata').

        Raises:
            ValueError: If `limit` is not None or a non-negative integer.
        """
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError(
                f"limit must be a non-negative integer or None, got {limit!r}")

        index = self._load_index()
        items = sorted(
            index.values(),
            key=lambda m: m["committed_at"],
            reverse=True
        )
        return items if limit is None else items[:limit]

    def prune(self) -> None:
        """Enforce retention policy by removing oldest unpinned snapshots.

        Retains at most `self.retention` snapshots (most recent), never removing
        snapshots whose metadata contains `"pin": True`.

        Raises:
            ValueError: If `self.retention` is not a positive integer.
            OSError: If an error occurs while deleting object directories.
        """
        if not isinstance(self.retention, int) or self.retention < 1:
            raise ValueError(
                f"retention must be a positive integer, got {self.retention!r}")

        index = self._load_index()
        # Sort ascending by commit time
        items = sorted(index.values(), key=lambda m: m["committed_at"])
        original_count = len(index)

        for meta in items:
            if len(index) <= self.retention:
                break
            if not meta.get("metadata", {}).get("pin", False):
                sha = meta["sha"]
                obj_dir = self._object_path(sha)
                try:
                    if obj_dir.exists():
                        shutil.rmtree(obj_dir)
                except OSError as exc:
                    raise OSError(
                        f"Failed to remove object dir {obj_dir}: {exc}") from exc
                index.pop(sha, None)

        if len(index) != original_count:
            self._write_index(index)

    def pin(self, sha: str) -> None:
        """Pin a snapshot so it will not be removed by pruning.

        Args:
            sha: 64-character lowercase SHA-256 hex digest of the snapshot.

        Raises:
            ValueError: If `sha` is not a valid SHA-256 hex digest.
            KeyError: If no snapshot with the given `sha` exists.
        """
        if not isinstance(sha, str) or not _SHA256_HEX_RE.match(sha):
            raise ValueError(f"Invalid SHA-256 hex digest: {sha!r}")

        index = self._load_index()
        if sha not in index:
            raise KeyError(f"No such snapshot to pin: {sha}")

        meta = index[sha].setdefault("metadata", {})
        if not meta.get("pin", False):
            meta["pin"] = True
            self._write_index(index)

    def unpin(self, sha: str) -> None:
        """Unpin a snapshot so it becomes subject to pruning again.

        Args:
            sha: 64-character lowercase SHA-256 hex digest of the snapshot.

        Raises:
            ValueError: If `sha` is not a valid SHA-256 hex digest.
            KeyError: If no snapshot with the given `sha` exists.
        """
        if not isinstance(sha, str) or not _SHA256_HEX_RE.match(sha):
            raise ValueError(f"Invalid SHA-256 hex digest: {sha!r}")

        index = self._load_index()
        if sha not in index:
            raise KeyError(f"No such snapshot to unpin: {sha}")

        meta = index[sha].get("metadata", {})
        if "pin" in meta:
            meta.pop("pin")
            self._write_index(index)
