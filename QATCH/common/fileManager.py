import os
from QATCH.common.architecture import Architecture, OSType

###############################################################################
# File operations: create directory, full path and check if the existing file
###############################################################################


class FileManager:
    """
    Utility class for file and directory operations, including directory creation,
    file path construction, and file existence checks.
    All methods are static and do not require instantiation.
    """

    ###########################################################################
    # Creates a directory if the specified path doesn't exist.
    ###########################################################################
    @staticmethod
    def create_dir(path=None):
        """
        Creates a directory if the specified path does not exist.
        Args:
            path (str, optional): Directory name or full path.
        Returns:
            bool: True if the specified directory exists after creation.
        """
        if path is not None:
            if not os.path.isdir(path):
                os.makedirs(path)
        return os.path.isdir(path)

    ###########################################################################
    # Creates a file full path based on parameters
    ###########################################################################

    @staticmethod
    def create_full_path(filename, extension="txt", path=None):
        """
        Constructs a full file path from filename, extension, and optional directory path.
        Args:
            filename (str): Name for the file (without extension).
            extension (str, optional): Extension for the file. Defaults to "txt".
            path (str, optional): Directory path for the file.
        Returns:
            str: Full path for the specified file.
        """
        full_path = str("{}.{}".format(filename, extension))
        if not path == None:
            full_path = os.path.join(path, full_path)
        return full_path

    ###########################################################################
    # Checks if a file exists (True if file exists)
    ###########################################################################

    @staticmethod
    def file_exists(filename):
        """
        Checks if a file exists at the specified path.
        Args:
            filename (str): Name of the file, including path.
        Returns:
            bool: True if the file exists, False otherwise.
        """
        if filename is not None:
            return os.path.isfile(filename)
