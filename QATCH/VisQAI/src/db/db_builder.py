import os
import pandas as pd
from db import Database
try:
    from src.controller.formulation_controller import FormulationController
except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.controller.formulation_controller import FormulationController


class DBBuilder:
    """Utility to initialize the Database from a CSV of formulations."""

    def __init__(self):
        self._db = Database()
        self.form_ctrl = FormulationController(db=self._db)

    def init_database(self, csv_path: str):
        """Read formulations from CSV and populate the database.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            List[Formulation]: The list of formulations added.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV is missing required columns.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        added_forms = self.form_ctrl.add_all_from_dataframe(df)
        return added_forms


if __name__ == "__main__":
    csv_path = os.path.join(
        "VisQAI", "assets", "formulation_data_10212025.csv")
    builder = DBBuilder()
    builder.init_database(csv_path=csv_path)
