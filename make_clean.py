import os
import shutil
from QATCH.core.constants import Constants

CLEAN_PATH = os.getcwd()
CLEAN_PATTERNS = ["build", "dist", "data", "__pycache__"]


def delete_folders_by_patterns(root_path, patterns):
    """
    Recursively walk through `root_path`, and delete all directories 
    where the full path ends with one of the given `patterns`.

    NOTE: Why `topdown=False`?
        This ensures that subdirectories are visited before their parents - 
        important when deleting parent folders and not break the traversal.

    Args:
        root_path (str): The root directory to start the search.
        patterns (str): The patterns to match folder names against.
    """
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            for pattern in patterns:
                if dirname == pattern:
                    full_path = os.path.join(dirpath, dirname)
                    print(f"Deleting folder: {full_path}")
                    shutil.rmtree(full_path)


delete_folders_by_patterns(CLEAN_PATH, CLEAN_PATTERNS)
