"""
formulation_controller.py

This module defines the `FormulationController` class for managing CRUD operations
on `Formulation` objects, interfacing with a `Database` and an `IngredientController`
to persist formulations and their components. It provides methods to add, retrieve,
update, and delete formulations, as well as bulk import/export to and from pandas
DataFrame representations.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-18

Version:
    1.4
"""

from typing import List

import pandas as pd

try:
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        ProteinClass,
        Salt,
        Stabilizer,
        Surfactant,
    )
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.models.formulation import (
        Formulation,
        ViscosityProfile,
    )
    from QATCH.VisQAI.src.models.ingredient import (
        Buffer,
        Excipient,
        Ingredient,
        Protein,
        ProteinClass,
        Salt,
        Stabilizer,
        Surfactant,
    )


class FormulationController:
    """Controller for managing `Formulation` persistence and retrieval.

    This class wraps a `Database` instance and an `IngredientController` to handle
    CRUD operations on `Formulation` objects. It supports adding, retrieving by ID,
    updating, deleting, and bulk import/export of formulations via pandas DataFrames.

    Attributes:
        db (Database): The database instance used for persistence.
        ingredient_controller (IngredientController): The controller responsible for
            adding or retrieving `Ingredient` objects referenced by formulations.
    """

    def __init__(self, db: Database) -> None:
        """Initialize a FormulationController with the given database.

        Args:
            db (Database): An initialized `Database` instance for storing formulations
                and their related ingredients.
        """
        self.db: Database = db
        self.ingredient_controller: IngredientController = IngredientController(self.db)

    def get_all_formulations(self) -> List[Formulation]:
        """Retrieve all stored formulations.

        Returns:
            List[Formulation]: A list of all `Formulation` objects persisted in the database.
        """
        return self.db.get_all_formulations()

    def get_formulation_by_id(self, id: int) -> Formulation:
        """Retrieve a single formulation by its database ID.

        Args:
            id (int): The primary key of the formulation to fetch.

        Returns:
            Formulation: The `Formulation` object with the given ID, or `None` if not found.
        """
        return self.db.get_formulation(id)

    def get_formulation_by_signature(self, signature: str) -> Formulation:
        """Retrieve a formulation by its unique SHA256 signature.

        Args:
            signature (str): The SHA256 signature string.

        Returns:
            Formulation: The matching Formulation object, or None if not found.
        """
        return self.db.get_formulation_by_signature(signature)

    def delete_formulation_by_signature(self, signature: str) -> bool:
        """Delete a formulation identified by its signature.

        Args:
            signature (str): The SHA256 signature string.

        Returns:
            bool: True if deleted, False if not found.
        """
        return self.db.delete_formulation_by_signature(signature)

    def update_formulation_name_by_signature(self, signature: str, name: str) -> bool:
        """Update the name of a formulation identified by its signature.

        Args:
            signature (str): The SHA256 signature.
            name (str): The new name.

        Returns:
            bool: True if updated, False if signature not found.
        """
        return self.db.update_formulation_name_by_signature(signature, name)

    def find_id(self, formulation: Formulation) -> int:
        """Find the database ID of a formulation matching the given object's data.

        Iterates through all stored formulations and returns the ID of the formulation
        that compares equal (ignoring ID) to the provided `formulation`.

        Args:
            formulation (Formulation): The `Formulation` instance to match.

        Returns:
            int: The ID of the matching formulation in the database.

        Raises:
            ValueError: If no matching formulation is found.
        """
        formulations = self.get_all_formulations()
        for f in formulations:
            if f == formulation:
                return f.id
        raise ValueError(
            f"Formulation with params\n\t'{formulation.to_dict()}'\nnot found."
        )

    def _ensure_ingredient(self, ing: Ingredient) -> Ingredient:
        """Return `ing` as-is if already persisted, otherwise add it via the ingredient controller.

        Args:
            ing (Ingredient): The ingredient to persist if needed.

        Returns:
            Ingredient: The persisted ingredient with a valid database ID.
        """
        if ing.id is not None:
            return ing
        return self.ingredient_controller.add(ing)

    def add_formulation(self, formulation: Formulation) -> Formulation:
        """Add a new formulation to the database if it does not already exist.

        This method ensures that all component ingredients are added via
        `IngredientController` before inserting the formulation. If an identical
        formulation (ignoring ID) already exists, returns the existing one.

        Args:
            formulation (Formulation): The `Formulation` instance to add. Its
                `buffer`, `protein`, `salt`, `surfactant`, `stabilizer`, and `excipient`
                attributes must be set to `Component` objects.

        Returns:
            Formulation: The persisted `Formulation` instance with its database-assigned ID.

        Raises:
            ValueError: If any required component is missing or invalid.
        """
        # Ensure each ingredient is persisted and update the formulation's component references
        formulation.buffer.ingredient = self._ensure_ingredient(
            formulation.buffer.ingredient
        )
        formulation.protein.ingredient = self._ensure_ingredient(
            formulation.protein.ingredient
        )
        formulation.salt.ingredient = self._ensure_ingredient(
            formulation.salt.ingredient
        )
        formulation.surfactant.ingredient = self._ensure_ingredient(
            formulation.surfactant.ingredient
        )
        formulation.stabilizer.ingredient = self._ensure_ingredient(
            formulation.stabilizer.ingredient
        )
        formulation.excipient.ingredient = self._ensure_ingredient(
            formulation.excipient.ingredient
        )

        # If an identical formulation already exists, return it
        if formulation.signature is not None:
            existing = self.get_formulation_by_signature(formulation.signature)
            if existing is not None:
                return existing

        # Otherwise, add a new formulation record
        self.db.add_formulation(formulation)
        return formulation

    def delete_formulation_by_id(self, id: int) -> Formulation:
        """Delete a formulation by its database ID.

        Retrieves the formulation first (to return it), then deletes the record.

        Args:
            id (int): The ID of the formulation to delete.

        Returns:
            Formulation: The deleted `Formulation` object.

        Raises:
            ValueError: If no formulation with the given ID exists.
        """
        formulation = self.get_formulation_by_id(id)
        if formulation is None:
            raise ValueError(f"Formulation with id '{id}' does not exist.")
        self.db.delete_formulation(id)
        return formulation

    def update_formulation(self, id: int, f_new: Formulation) -> Formulation:
        """Update an existing formulation's data by replacing its record.

        This implementation deletes the old record and inserts a new one with
        the data from `f_new`. The ID of `f_new` is not preserved.

        Args:
            id (int): The ID of the formulation to update.
            f_new (Formulation): A `Formulation` instance containing the new data.
                Its components and viscosity profile should be set.

        Returns:
            Formulation: The updated `Formulation` instance.

        Raises:
            ValueError: If no formulation with the given ID exists.
        """
        f_fetch = self.get_formulation_by_id(id)
        if f_fetch is None:
            raise ValueError(f"Formulation with id '{id}' does not exist.")
        if f_fetch == f_new:
            return f_new

        # Ensure each ingredient is persisted before saving the updated formulation
        f_new.buffer.ingredient = self._ensure_ingredient(f_new.buffer.ingredient)
        f_new.protein.ingredient = self._ensure_ingredient(f_new.protein.ingredient)
        f_new.salt.ingredient = self._ensure_ingredient(f_new.salt.ingredient)
        f_new.surfactant.ingredient = self._ensure_ingredient(
            f_new.surfactant.ingredient
        )
        f_new.stabilizer.ingredient = self._ensure_ingredient(
            f_new.stabilizer.ingredient
        )
        f_new.excipient.ingredient = self._ensure_ingredient(f_new.excipient.ingredient)

        # Delete the old formulation and re-add the new data
        self.db.delete_formulation(id)
        self.db.add_formulation(f_new)
        return f_new

    def update_formulation_metadata(
        self, id: int, icl: bool = None, last_model: str = None
    ) -> bool:
        """Update lightweight metadata fields for a formulation without touching its components.

        Delegates to `Database.update_formulation_metadata`. Only non-``None`` arguments
        will trigger a database write, so callers may pass a single field to update.

        Args:
            id (int): The primary key of the formulation to update.
            icl (bool, optional): New value for the In-Concentration Loading flag.
                If ``None``, the field is left unchanged.
            last_model (str, optional): Identifier of the last predictive model run
                against this formulation. If ``None``, the field is left unchanged.

        Returns:
            bool: ``True`` if at least one field was updated successfully, ``False``
                if the formulation was not found or an error occurred.
        """
        return self.db.update_formulation_metadata(id, icl, last_model)

    def add_all_from_dataframe(
        self, df: pd.DataFrame, verbose_print: bool = False
    ) -> List[Formulation]:
        """Bulk import multiple formulations from a pandas DataFrame.

        The DataFrame must contain specific columns for each ingredient type,
        concentration, and viscosity measurements at predefined shear rates. Each
        row corresponds to a single formulation.

        Expected columns:
            - Protein_type, Protein_class_type, MW, PI_mean, PI_range, Protein_conc
            - Temperature
            - Buffer_type, Buffer_pH, Buffer_conc
            - Salt_type, Salt_conc
            - Stabilizer_type, Stabilizer_conc
            - Surfactant_type, Surfactant_conc
            - Excipient_type, Excipient_conc
            - Viscosity_100, Viscosity_1000, Viscosity_10000, Viscosity_100000, Viscosity_15000000

        Args:
            df (pd.DataFrame): DataFrame containing formulation data row-wise.
            verbose_print (bool, False): If True, display a `tqdm` progress bar for status.

        Returns:
            List[Formulation]: A list of all `Formulation` instances added to the database.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        shear_rates = [100, 1000, 10000, 100000, 15000000]
        expected = {
            "Protein_type",
            "MW",
            "PI_mean",
            "PI_range",
            "Protein_conc",
            "Protein_class_type",
            "Temperature",
            "Buffer_type",
            "Buffer_pH",
            "Buffer_conc",
            "Salt_type",
            "Salt_conc",
            "Stabilizer_type",
            "Stabilizer_conc",
            "Surfactant_type",
            "Surfactant_conc",
            "Excipient_type",
            "Excipient_conc",
            *{f"Viscosity_{r}" for r in shear_rates},
        }

        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: `{missing}`")

        has_name = "name" in df.columns
        has_sig = "signature" in df.columns
        has_icl = "icl" in df.columns
        has_model = "last_model" in df.columns

        self.db._defer_backup = True
        self.db._defer_commit = True
        self.db.begin_bulk()

        pending_forms: List[Formulation] = []

        if verbose_print:
            from tqdm import tqdm

            p_bar = tqdm(total=len(df))

        try:
            for row in df.itertuples(index=False):
                if verbose_print:
                    p_bar.update()

                f_sig = getattr(row, "signature", None) if has_sig else None
                if f_sig is not None and not pd.notna(f_sig):
                    f_sig = None

                # Skip rows whose signature is already present in the database
                if (
                    f_sig is not None
                    and self.get_formulation_by_signature(f_sig) is not None
                ):
                    continue

                f_name = getattr(row, "name", None) if has_name else None
                if f_name is not None and not pd.notna(f_name):
                    f_name = None

                form = Formulation(name=f_name, signature=f_sig)

                if has_icl and pd.notna(row.icl):
                    form.icl = bool(row.icl)

                if has_model and pd.notna(row.last_model):
                    form.last_model = str(row.last_model)

                protein = self.ingredient_controller.add_protein(
                    Protein(
                        enc_id=0,
                        name=str(row.Protein_type),
                        molecular_weight=float(row.MW),
                        pI_mean=float(row.PI_mean),
                        pI_range=float(row.PI_range),
                        class_type=(
                            ProteinClass.from_value(row.Protein_class_type)
                            if pd.notna(row.Protein_class_type)
                            else None
                        ),
                    )
                )
                buffer = self.ingredient_controller.add_buffer(
                    Buffer(enc_id=0, name=str(row.Buffer_type), pH=row.Buffer_pH)
                )
                stabilizer = self.ingredient_controller.add_stabilizer(
                    Stabilizer(enc_id=0, name=str(row.Stabilizer_type))
                )
                surfactant = self.ingredient_controller.add_surfactant(
                    Surfactant(enc_id=0, name=str(row.Surfactant_type))
                )
                salt = self.ingredient_controller.add_salt(
                    Salt(enc_id=0, name=str(row.Salt_type))
                )
                excipient = self.ingredient_controller.add_excipient(
                    Excipient(enc_id=0, name=str(row.Excipient_type))
                )

                # BUILD VISCOSITY PROFILE
                vis_values = [getattr(row, f"Viscosity_{r}") for r in shear_rates]
                if any(pd.notna(v) for v in vis_values):
                    vp = ViscosityProfile(
                        shear_rates=shear_rates, viscosities=vis_values, units="cP"
                    )
                else:
                    vp = ViscosityProfile(
                        shear_rates=shear_rates,
                        viscosities=[-1, -1, -1, -1, -1],
                        units="unset",
                    )

                # SET COMPONENTS
                form.set_buffer(
                    buffer=buffer, concentration=row.Buffer_conc, units="mM"
                )
                form.set_protein(
                    protein=protein, concentration=row.Protein_conc, units="mg/mL"
                )
                form.set_stabilizer(
                    stabilizer=stabilizer, concentration=row.Stabilizer_conc, units="M"
                )
                form.set_salt(salt=salt, concentration=row.Salt_conc, units="mM")
                form.set_surfactant(
                    surfactant=surfactant, concentration=row.Surfactant_conc, units="%w"
                )
                form.set_excipient(
                    excipient=excipient, concentration=row.Excipient_conc, units="mM"
                )
                form.set_temperature(temp=row.Temperature)
                form.set_viscosity_profile(profile=vp)

                pending_forms.append(form)

        finally:
            self.db.add_formulations_batch(pending_forms)
            self.db._defer_commit = False
            self.db._defer_backup = False
            self.db.end_bulk()
            self.db.flush()
        if verbose_print:
            p_bar.close()

        return pending_forms

    def get_all_as_dataframe(self, encoded: bool = True) -> pd.DataFrame:
        """Export all stored formulations to a pandas DataFrame.

        Args:
            encoded (bool): If endocded is set to true, the enc_id's are returned
                for each categorical feature of the dataframe.  If encoded is false, the
                name is returned instead of the enc_id (Optional, Default=True).

        The resulting DataFrame contains one row per formulation, with columns:
            - ID
            - Protein_type, MW, PI_mean, PI_range, Protein_conc
            - Temperature
            - Buffer_type, Buffer_pH, Buffer_conc
            - Salt_type, Salt_conc
            - Stabilizer_type, Stabilizer_conc
            - Surfactant_type, Surfactant_conc
            - Excipient_type, Excipient_conc
            - Viscosity_100, Viscosity_1000, Viscosity_10000, Viscosity_100000, Viscosity_15000000

        Missing viscosity values default to NaN.

        Returns:
            pd.DataFrame: DataFrame containing all formulation data with the specified columns.
        """
        expected = [
            "ID",
            "Protein_type",
            "Protein_class_type",
            "kP",
            "MW",
            "PI_mean",
            "PI_range",
            "Protein_conc",
            "Temperature",
            "Buffer_type",
            "Buffer_pH",
            "Buffer_conc",
            "Salt_type",
            "Salt_conc",
            "Stabilizer_type",
            "Stabilizer_conc",
            "Surfactant_type",
            "Surfactant_conc",
            "Excipient_type",
            "Excipient_conc",
            "C_Class",
            "HCI",
            "Viscosity_100",
            "Viscosity_1000",
            "Viscosity_10000",
            "Viscosity_100000",
            "Viscosity_15000000",
        ]

        rows = []
        for f in self.get_all_formulations():
            single_df = f.to_dataframe(encoded=encoded)
            rows.append(single_df)

        if not rows:
            return pd.DataFrame(columns=expected)

        df = pd.concat(rows, ignore_index=True)

        # Ensure all expected columns exist, filling missing ones with NaN
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA

        # Return the dataframe with columns strictly enforced in the 'expected' order
        return df[expected]
