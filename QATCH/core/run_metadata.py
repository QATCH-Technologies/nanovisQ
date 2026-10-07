class RunMetadata:
    """A data class representing information for a run that has not been named.

    This class serves as a container for metadata extracted from raw run files
    located in the unnamed recovery directory.

    Attributes:
        filepath (str): The absolute path to the directory containing the run data.
        display_name (str): The name of the run derived from the folder name.
        start (str): The date and time the run started in ISO format.
        stop (str): The date and time the run ended in ISO format.
        duration (float): The total length of the run in seconds.
        samples (int): The total count of data points recorded in the run.
        ruling (str): The classification of the run (e.g., "Good" or "Bad").
        file_size_mb (float): The total size of the run folder in megabytes.
        virtual_csv_path (str | None): The path to the CSV file within a ZIP archive.
    """

    def __init__(
        self,
        filepath: str,
        display_name: str,
        start: str,
        stop: str,
        duration: float,
        samples: int,
        ruling: str,
        file_size_mb: float,
        virtual_csv_path: str | None = None,
    ):
        self.filepath = filepath
        self.display_name = display_name
        self.start = start
        self.stop = stop
        self.duration = duration
        self.samples = samples
        self.ruling = ruling
        self.file_size_mb = file_size_mb
        self.virtual_csv_path = virtual_csv_path
