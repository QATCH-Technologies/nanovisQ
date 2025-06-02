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
    2025-06-02

Version:
    1.3
"""

from typing import List
import pandas as pd

try:
    from src.db.db import Database
    from src.controller.ingredient_controller import IngredientController
    from src.models.formulation import Formulation, Component, ViscosityProfile
    from src.models.ingredient import Protein, Buffer, Stabilizer, Salt, Surfactant
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.formulation import Formulation, Component, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import Protein, Buffer, Stabilizer, Salt, Surfactant


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
        self.ingredient_controller: IngredientController = IngredientController(
            self.db)

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

    def add_formulation(self, formulation: Formulation) -> Formulation:
        """Add a new formulation to the database if it does not already exist.

        This method ensures that all component ingredients are added via
        `IngredientController` before inserting the formulation. If an identical
        formulation (ignoring ID) already exists, returns the existing one.

        Args:
            formulation (Formulation): The `Formulation` instance to add. Its
                `buffer`, `protein`, `salt`, `surfactant`, and `stabilizer`
                attributes must be set to `Component` objects.

        Returns:
            Formulation: The persisted `Formulation` instance with its database-assigned ID.

        Raises:
            ValueError: If any required component is missing or invalid.
        """
        # Ensure each ingredient is persisted and update the formulation's component references
        buffer_ing = formulation.buffer.ingredient
        formulation.buffer.ingredient = self.ingredient_controller.add(
            buffer_ing)
        protein_ing = formulation.protein.ingredient
        formulation.protein.ingredient = self.ingredient_controller.add(
            protein_ing)
        salt_ing = formulation.salt.ingredient
        formulation.salt.ingredient = self.ingredient_controller.add(salt_ing)

        surfactant_ing = formulation.surfactant.ingredient
        formulation.surfactant.ingredient = self.ingredient_controller.add(
            surfactant_ing)

        stabilizer_ing = formulation.stabilizer.ingredient
        formulation.stabilizer.ingredient = self.ingredient_controller.add(
            stabilizer_ing)

        # If an identical formulation already exists, return it
        existing = self.get_all_formulations()
        for f in existing:
            if f == formulation:
                return f

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
            Formulation: The updated `Formulation` instance (whose `.id` is set to `id`).

        Raises:
            ValueError: If no formulation with the given ID exists.
        """
        f_fetch = self.get_formulation_by_id(id)
        if f_fetch is None:
            raise ValueError(f"Formulation with id '{id}' does not exist.")
        if f_fetch == f_new:
            return f_new

        # Delete the old formulation and re-add the new data
        self.db.delete_formulation(id)
        self.db.add_formulation(f_new)
        return f_new

    def add_all_from_dataframe(self, df: pd.DataFrame) -> List[Formulation]:
        """Bulk import multiple formulations from a pandas DataFrame.

        The DataFrame must contain specific columns for each ingredient type,
        concentration, and viscosity measurements at predefined shear rates. Each
        row corresponds to a single formulation.

        Expected columns:
            - Protein_type, MW, PI_mean, PI_range, Protein_conc
            - Temperature
            - Buffer_type, Buffer_pH, Buffer_conc
            - Salt_type, Salt_conc
            - Stabilizer_type, Stabilizer_conc
            - Surfactant_type, Surfactant_conc
            - Viscosity_100, Viscosity_1000, Viscosity_10000, Viscosity_100000, Viscosity_15000000

        Args:
            df (pd.DataFrame): DataFrame containing formulation data row-wise.

        Returns:
            List[Formulation]: A list of all `Formulation` instances added to the database.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        added_forms: List[Formulation] = []
        shear_rates = [100, 1000, 10000, 100000, 15000000]
        expected = {
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Stabilizer_type", "Stabilizer_conc",
            "Surfactant_type", "Surfactant_conc",
            *{f"Viscosity_{r}" for r in shear_rates},
        }
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: `{missing}`")

        for _, row in df.iterrows():
            protein = self.ingredient_controller.add_protein(
                Protein(
                    enc_id=0,
                    name=str(row["Protein_type"]),
                    molecular_weight=float(row["MW"]),
                    pI_mean=float(row["PI_mean"]),
                    pI_range=float(row["PI_range"])
                )
            )
            buffer = self.ingredient_controller.add_buffer(
                Buffer(
                    enc_id=0,
                    name=str(row["Buffer_type"]),
                    pH=row["Buffer_pH"]
                )
            )
            stabilizer = self.ingredient_controller.add_stabilizer(
                Stabilizer(enc_id=0, name=str(row["Stabilizer_type"]))
            )
            surfactant = self.ingredient_controller.add_surfactant(
                Surfactant(enc_id=0, name=str(row["Surfactant_type"]))
            )
            salt = self.ingredient_controller.add_salt(
                Salt(enc_id=0, name=str(row["Salt_type"]))
            )

            vis_values = [row[f"Viscosity_{r}"] for r in shear_rates]
            if any(pd.notna(v) for v in vis_values):
                vp = ViscosityProfile(
                    shear_rates=shear_rates,
                    viscosities=vis_values,
                    units="cP"
                )
            else:
                vp = ViscosityProfile(
                    shear_rates=shear_rates,
                    viscosities=[-1, -1, -1, -1, -1],
                    units="unset"
                )

            form = Formulation()
            form.set_buffer(
                buffer=buffer,
                concentration=row["Buffer_conc"],
                units="mg/mL"
            )
            form.set_protein(
                protein=protein,
                concentration=row["Protein_conc"],
                units="mg/mL"
            )
            form.set_stabilizer(
                stabilizer=stabilizer,
                concentration=row["Stabilizer_conc"],
                units="M"
            )
            form.set_salt(
                salt=salt,
                concentration=row["Salt_conc"],
                units="mg/mL"
            )
            form.set_surfactant(
                surfactant=surfactant,
                concentration=row["Surfactant_conc"],
                units="%w"
            )
            form.set_temperature(temp=row["Temperature"])
            form.set_viscosity_profile(profile=vp)

            saved = self.add_formulation(form)
            added_forms.append(saved)

        return added_forms

    def get_all_as_dataframe(self) -> pd.DataFrame:
        """Export all stored formulations to a pandas DataFrame.

        The resulting DataFrame contains one row per formulation, with columns:
            - ID
            - Protein_type, MW, PI_mean, PI_range, Protein_conc
            - Temperature
            - Buffer_type, Buffer_pH, Buffer_conc
            - Salt_type, Salt_conc
            - Stabilizer_type, Stabilizer_conc
            - Surfactant_type, Surfactant_conc
            - Viscosity_100, Viscosity_1000, Viscosity_10000, Viscosity_100000, Viscosity_15000000

        Missing viscosity values default to NaN.

        Returns:
            pd.DataFrame: DataFrame containing all formulation data with the specified columns.
        """
        rows = []
        for f in self.get_all_formulations():
            row = {
                "ID":              f.id,
                "Protein_type":    f.protein.ingredient.enc_id,
                "MW":              f.protein.ingredient.molecular_weight,
                "PI_mean":         f.protein.ingredient.pI_mean,
                "PI_range":        f.protein.ingredient.pI_range,
                "Protein_conc":    f.protein.concentration,
                "Temperature":     getattr(f, "temperature", pd.NA),
                "Buffer_type":     f.buffer.ingredient.enc_id,
                "Buffer_pH":       f.buffer.ingredient.pH,
                "Buffer_conc":     f.buffer.concentration,
                "Salt_type":       f.salt.ingredient.enc_id,
                "Salt_conc":       f.salt.concentration,
                "Stabilizer_type": f.stabilizer.ingredient.enc_id,
                "Stabilizer_conc": f.stabilizer.concentration,
                "Surfactant_type": f.surfactant.ingredient.enc_id,
                "Surfactant_conc": f.surfactant.concentration,
            }

            shear_rates = [100, 1000, 10000, 100000, 15000000]
            if f.viscosity_profile is not None:
                for r in shear_rates:
                    row[f"Viscosity_{int(r)}"] = f.viscosity_profile.get_viscosity(
                        r)

            rows.append(row)

        df = pd.DataFrame(rows)
        expected = [
            "ID",
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Stabilizer_type", "Stabilizer_conc",
            "Surfactant_type", "Surfactant_conc",
            "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
            "Viscosity_100000", "Viscosity_15000000"
        ]
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA

        return df[expected]
