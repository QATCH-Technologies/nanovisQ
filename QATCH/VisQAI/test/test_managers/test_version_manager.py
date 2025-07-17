# test_model_version_controller.py

import unittest
import tempfile
import os
import json
import time
import shutil
import threading
import random
from pathlib import Path

from src.managers.version_manager import VersionManager


class TestVersionManager(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tempdir.name) / "repo"
        self.repo_dir.mkdir()
        self.src_model = Path("test/assets/VisQAI-base.zip")
        self.assertTrue(self.src_model.exists(),
                        f"Actual model not found at {self.src_model}")
        self.model_file = Path(self.tempdir.name) / self.src_model.name
        shutil.copy2(self.src_model, self.model_file)
        self.mvc = VersionManager(str(self.repo_dir), retention=2)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_commit_and_get(self):
        sha = self.mvc.commit(str(self.model_file), metadata={"test": True})
        idx = json.loads((self.repo_dir / "index.json").read_text())
        self.assertIn(sha, idx)

        blob_dir = self.repo_dir / "objects" / sha[:2] / sha
        self.assertTrue(blob_dir.exists())

        out_dir = Path(self.tempdir.name) / "restored"
        restored_path = self.mvc.get(sha, str(out_dir))
        self.assertTrue(restored_path.exists())
        self.assertEqual(restored_path.read_bytes(),
                         self.model_file.read_bytes())

    def test_list_order(self):
        sha1 = self.mvc.commit(str(self.model_file), metadata={"seq": 1})
        time.sleep(1.1)
        with self.model_file.open("wb") as f:
            f.write(os.urandom(2048))
        sha2 = self.mvc.commit(str(self.model_file), metadata={"seq": 2})
        lst = self.mvc.list()
        seqs = [item["metadata"].get("seq") for item in lst]
        self.assertEqual(seqs, [2, 1])

    def test_deduplication(self):
        sha1 = self.mvc.commit(str(self.model_file))
        time.sleep(1.1)
        sha2 = self.mvc.commit(str(self.model_file))
        self.assertEqual(sha1, sha2)
        idx = json.loads((self.repo_dir / "index.json").read_text())
        self.assertEqual(len(idx), 1)

    def test_retention_policy(self):
        file2 = Path(self.tempdir.name) / "dummy2.h5"
        file3 = Path(self.tempdir.name) / "dummy3.h5"
        for p in (file2, file3):
            with p.open("wb") as f:
                f.write(os.urandom(512))
        sha1 = self.mvc.commit(str(self.model_file))
        sha2 = self.mvc.commit(str(file2))
        time.sleep(1.1)
        sha3 = self.mvc.commit(str(file3))
        idx = json.loads((self.repo_dir / "index.json").read_text())
        self.assertNotIn(sha1, idx)
        self.assertIn(sha2, idx)
        self.assertIn(sha3, idx)

    def test_pin_and_unpin(self):
        file2 = Path(self.tempdir.name) / "dummy2.bin"
        file3 = Path(self.tempdir.name) / "dummy3.bin"
        for p in (file2, file3):
            with p.open("wb") as f:
                f.write(os.urandom(256))
        # Commit two snapshots
        sha1 = self.mvc.commit(str(self.model_file))
        time.sleep(1.1)
        sha2 = self.mvc.commit(str(file2))
        self.mvc.pin(sha1)
        time.sleep(1.1)
        sha3 = self.mvc.commit(str(file3))
        idx = json.loads((self.repo_dir / "index.json").read_text())
        # Should keep pinned sha1 and the new sha3, sha2 should be pruned
        self.assertIn(sha1, idx)
        self.assertIn(sha3, idx)
        self.assertNotIn(sha2, idx)

        # Unpin sha1 and commit a fourth snapshot
        self.mvc.unpin(sha1)
        time.sleep(1.1)
        file4 = Path(self.tempdir.name) / "dummy4.bin"
        with file4.open("wb") as f:
            f.write(os.urandom(128))
        sha4 = self.mvc.commit(str(file4))
        idx = json.loads((self.repo_dir / "index.json").read_text())
        # Now should keep the two most recent: sha3 and sha4
        self.assertIn(sha3, idx)
        self.assertIn(sha4, idx)
        self.assertNotIn(sha1, idx)
        self.assertNotIn(sha2, idx)

    def test_get_nonexistent_raises(self):
        with self.assertRaises(KeyError):
            self.mvc.get("nonexistent" * 8, str(self.tempdir.name))

    def test_concurrent_commits(self):
        results = []

        def worker(i):
            fp = self.model_file if i % 2 == 0 else self.model_file
            sha = self.mvc.commit(str(fp), metadata={"thread": i})
            results.append(sha)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        evens = [results[i] for i in range(10) if i % 2 == 0]
        self.assertTrue(all(s == evens[0] for s in evens))

    def test_stress_random_operations(self):
        shas = []
        for i in range(200):
            if random.random() < 0.5:
                fn = Path("test/assets/VisQAI-base.zip")
                with fn.open("wb") as f:
                    f.write(os.urandom(random.randint(100, 1000)))
                sha = self.mvc.commit(str(fn), metadata={"i": i})
                shas.append(sha)
            if shas and random.random() < 0.3:
                try:
                    self.mvc.pin(random.choice(shas))
                except KeyError:
                    pass
            if shas and random.random() < 0.2:
                try:
                    self.mvc.unpin(random.choice(shas))
                except KeyError:
                    pass
            _ = self.mvc.list()
            if shas and random.random() < 0.4:
                choice = random.choice(shas)
                out = Path(self.tempdir.name) / f"out{i}"
                try:
                    snapshot = self.mvc.get(choice, str(out))
                except KeyError:
                    pass
        idx = json.loads((self.repo_dir / "index.json").read_text())
        self.assertIsInstance(idx, dict)


if __name__ == "__main__":
    unittest.main()
