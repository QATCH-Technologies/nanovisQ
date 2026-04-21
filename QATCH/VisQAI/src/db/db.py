"""
database.py

This module provides the `Database` class for managing persistence of ingredients,
formulations, components, and viscosity profiles using SQLite. It supports optional
in-memory encryption via a simple XOR+Caesar cipher. The database file can support
an encoded metadata on the first line for storing relevant information including an
 `app_key` which can be automatically parsed on load. The schema includes tables for
generic ingredients, subclass-specific ingredient details, formulation records, and
associated component and viscosity profile data. Utility methods handle CRUD operations,
schema initialization, and optional encryption/decryption of the entire database.

Author:
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-04-14

Version:
    1.9.1
"""

import json
import os
import random
import sqlite3
import tempfile
from pathlib import Path
from typing import List, Optional, Union
import time
import numpy as np

try:
    TAG = "[Database (HEADLESS)]"
    from src.models.formulation import Component, Formulation, ViscosityProfile
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

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

except (ModuleNotFoundError, ImportError):
    TAG = "[Database]"
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.models.formulation import (
        Component,
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

DB_PATH = Path(
    os.path.join(os.path.expandvars(r"%LOCALAPPDATA%"), "QATCH", "nanovisQ", "database", "app.db")
)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

allowed_vals = ",".join(f"'{v}'" for v in ProteinClass.all_strings())


class Database:
    """Manages SQLite persistence for ingredients, formulations, and viscosity profiles.

    This class initializes the required tables for storing generic ingredients
    and their subclass-specific attributes, as well as formulations with associated
    components and viscosity profiles. It supports optional in-memory encryption
    and decryption of the database file using a Caesar+XOR cipher. CRUD methods
    handle insertion, retrieval, updating, and deletion of ingredients and formulations.

    Attributes:
        db_path (Path): Filesystem path to the SQLite database file.
        file_handle (Optional[IO]): Handle used when an encrypted database is opened.
        encryption_key (Optional[str]): Key used for Caesar+XOR encryption.
        use_encryption (bool): Whether to use in-memory encryption.
        conn (sqlite3.Connection): SQLite connection object.
        init_changes (int): Initial change count to detect unsaved changes.
    """

    def __init__(
        self,
        path: Union[str, Path] = DB_PATH,
        encryption_key: Union[str, None] = None,
        parse_file_key: bool = False,
    ) -> None:
        """Initialize the Database, apply encryption if requested, and create tables.

        Args:
            path (Union[str, Path]): Filesystem path to the SQLite database file.
                Defaults to a standard `app.db` location under %LOCALAPPDATA%.
            encryption_key (Union[str, None], optional): If provided, uses
                in-memory Caesar+XOR encryption/decryption. If None, plain SQLite
                is used. Defaults to None.
            parse_file_key (bool): If True, the `encryption_key` will be loaded
                from the `app_key` that is embedded in the `path` database file.
                Defaults to False. Flag ignored if `encryption_key` is provided.

        Raises:
            ValueError: If `encryption_key` is None but `use_encryption` is True
                (e.g., encryption_key provided as empty).
        """
        self.db_path = Path(path)
        self.file_handle = None  # used for locking when encrypted
        self.encryption_key = encryption_key
        self.use_encryption = parse_file_key if self.encryption_key is None else True
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_metadata()

        if self.use_encryption:
            if parse_file_key and not self.encryption_key:
                self.set_encryption_key(self.metadata.get("app_key", None))
            if not self.encryption_key:
                Log.e(TAG, "Encryption key missing for encrypted database")
            self.conn = self._open_cdb(str(self.db_path), self.encryption_key)
        else:
            self.conn = sqlite3.connect(str(self.db_path))

        # Enforce foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_changes = self.conn.total_changes
        self._defer_backup: bool = False
        self._defer_commit: bool = False
        self._create_tables()
        self.is_open = True

    def _create_tables(self) -> None:
        """Create all necessary tables if they do not already exist.

        The tables include:
            - ingredient: core table with common fields (id, enc_id, name, type, is_user).
            - protein, buffer, stabilizer, surfactant, salt, excipient: subclass tables referencing ingredient.
            - formulation: core formulation table (id, temperature).
            - formulation_component: linking table between formulations and ingredient components.
            - viscosity_profile: table storing JSON-serialized shear_rates and viscosities.

        This method commits the changes to the database.
        """
        c = self.conn.cursor()
        # Ingredient core table
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS ingredient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                enc_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                is_user INTEGER NOT NULL DEFAULT 0
            )
        """
        )
        # Subclass tables
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS protein (
                ingredient_id    INTEGER PRIMARY KEY,
                class_type       TEXT NOT NULL DEFAULT 'None'
                                CHECK (class_type IN ({allowed_vals})),
                molecular_weight REAL,
                pI_mean          REAL,
                pI_range         REAL,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS buffer (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS stabilizer (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS surfactant (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS salt (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS excipient (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS formulation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                signature TEXT UNIQUE,
                temperature REAL,
                icl INTEGER DEFAULT 1,
                last_model TEXT
            )
        """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_formulation_signature ON formulation(signature)")
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS formulation_component (
                formulation_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,
                ingredient_id INTEGER NOT NULL,
                concentration REAL NOT NULL,
                units TEXT NOT NULL,
                pH REAL,
                PRIMARY KEY (formulation_id, component_type),
                FOREIGN KEY (formulation_id) REFERENCES formulation(id) ON DELETE CASCADE,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """
        )
        c.execute(
            rf"""
            CREATE TABLE IF NOT EXISTS viscosity_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                formulation_id INTEGER NOT NULL UNIQUE,
                shear_rates TEXT NOT NULL,
                viscosities TEXT NOT NULL,
                units TEXT NOT NULL,
                is_measured INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (formulation_id) REFERENCES formulation(id) ON DELETE CASCADE
            )
        """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_ingredient_type ON ingredient(type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ingredient_name_type ON ingredient(name, type)")
        self._commit()

    def add_ingredient(self, ing: Ingredient) -> int:
        """Insert a new ingredient and its subclass-specific details into the database.

        Args:
            ing (Ingredient): An instance of a subclass of `Ingredient` (Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient).
                The `enc_id` and `name` fields must be set, and subclass-specific attributes populated.

        Returns:
            int: The database-assigned primary key (`id`) of the newly inserted ingredient.

        Raises:
            ValueError: If `ing.enc_id` is None (indicating improper usage).
        """
        if ing.enc_id is None:
            raise ValueError(
                "ing.enc_id is None: you must call IngredientController.add_* so that enc_id is auto-assigned."
            )
        c = self.conn.cursor()
        c.execute(
            "SELECT id FROM ingredient WHERE name = ? AND type = ?",
            (ing.name, type(ing).__name__),
        )
        existing = c.fetchone()
        if existing:
            ing.id = existing[0]
            return existing[0]

        # Only insert if no match found
        c.execute(
            "INSERT INTO ingredient (enc_id, name, type, is_user) VALUES (?, ?, ?, ?)",
            (ing.enc_id, ing.name, type(ing).__name__, int(ing.is_user)),
        )
        db_id = c.lastrowid

        # Insert into the subclass table based on runtime type
        if isinstance(ing, Protein):
            if isinstance(ing.class_type, ProteinClass):
                class_val = ing.class_type.value
            else:
                class_val = ing.class_type or ProteinClass.NONE.value

            c.execute(
                """
                INSERT INTO protein
                    (ingredient_id, class_type, molecular_weight, pI_mean, pI_range)
                VALUES (?, ?, ?, ?, ?)
                """,
                (db_id, class_val, ing.molecular_weight, ing.pI_mean, ing.pI_range),
            )
        elif isinstance(ing, Buffer):
            c.execute("INSERT INTO buffer (ingredient_id) VALUES (?)", (db_id,))
        elif isinstance(ing, Stabilizer):
            c.execute("INSERT INTO stabilizer VALUES (?)", (db_id,))
        elif isinstance(ing, Surfactant):
            c.execute("INSERT INTO surfactant VALUES (?)", (db_id,))
        elif isinstance(ing, Salt):
            c.execute("INSERT INTO salt VALUES (?)", (db_id,))
        elif isinstance(ing, Excipient):
            c.execute("INSERT INTO excipient VALUES (?)", (db_id,))

        self._commit()
        if self.use_encryption and not self._defer_backup:
            self.backup()
        ing.id = db_id
        return db_id

    def get_ingredient(self, id: int) -> Optional[Ingredient]:
        """Retrieve an ingredient by its database ID, reconstructing the correct subclass.

        Args:
            id (int): The primary key of the ingredient to fetch.

        Returns:
            Optional[Ingredient]: An instance of the appropriate `Ingredient` subclass
                with all properties populated, or `None` if no ingredient exists with this ID.
        """
        c = self.conn.cursor()
        c.execute("SELECT enc_id, name, type, is_user FROM ingredient WHERE id = ?", (id,))
        row = c.fetchone()
        if not row:
            return None
        enc_id, name, typ, is_user = row

        # Reconstruct subclass-specific object
        if typ == "Protein":
            row = c.execute(
                """
                SELECT class_type, molecular_weight, pI_mean, pI_range
                FROM protein
                WHERE ingredient_id = ?
            """,
                (id,),
            ).fetchone()

            if row is None:
                raise LookupError(f"No protein row for ingredient_id {id}")

            class_str, mw, mean, rng = row
            ing = Protein(
                enc_id=enc_id,
                name=name,
                molecular_weight=mw,
                pI_mean=mean,
                pI_range=rng,
                class_type=(ProteinClass.from_value(class_str) if class_str is not None else None),
                id=id,
            )
        elif typ == "Buffer":
            ing = Buffer(enc_id=enc_id, name=name, id=id)
        elif typ == "Stabilizer":
            ing = Stabilizer(enc_id, name)

        elif typ == "Surfactant":
            ing = Surfactant(enc_id, name)

        elif typ == "Excipient":
            ing = Excipient(enc_id, name)

        elif typ == "Salt":
            ing = Salt(enc_id, name)

        else:
            return None

        ing.id = id
        ing.is_user = bool(is_user)
        return ing

    def get_all_ingredients(self) -> List[Ingredient]:
        """Retrieve all ingredients stored in the database.

        Returns:
            List[Ingredient]: A list of `Ingredient` subclass instances, one per row
                in the `ingredient` table, reconstructed using `get_ingredient`.
        """
        c = self.conn.cursor()
        c.execute("SELECT id FROM ingredient")
        ids = [r[0] for r in c.fetchall()]
        return [self.get_ingredient(i) for i in ids]

    def get_ingredients_by_type(self, ing_type: str) -> List[Ingredient]:
        """Retrieve all ingredients of a specific subclass type.

        Args:
            ing_type (str): The ingredient type name to filter by (e.g. ``"Protein"``,
                ``"Buffer"``, ``"Salt"``).

        Returns:
            List[Ingredient]: A list of `Ingredient` subclass instances matching
                the given type, reconstructed via `get_ingredient`.
        """
        c = self.conn.cursor()
        c.execute("SELECT id FROM ingredient WHERE type = ?", (ing_type,))
        return [self.get_ingredient(r[0]) for r in c.fetchall()]

    def get_max_enc_id(self, ing_type: str, min_enc_id: int, max_enc_id: int) -> Optional[int]:
        """Return the highest ``enc_id`` for a given ingredient type within a range.

        Args:
            ing_type (str): The ingredient type name to query (e.g. ``"Protein"``).
            min_enc_id (int): Lower bound of the ``enc_id`` range (inclusive).
            max_enc_id (int): Upper bound of the ``enc_id`` range (inclusive).

        Returns:
            Optional[int]: The maximum ``enc_id`` found within the range, or ``None``
                if no matching rows exist.
        """
        c = self.conn.cursor()
        c.execute(
            "SELECT MAX(enc_id) FROM ingredient WHERE type = ? AND enc_id BETWEEN ? AND ?",
            (
                ing_type,
                min_enc_id,
                max_enc_id,
            ),
        )
        row = c.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_ingredient_by_name_type(self, name: str, ing_type: str) -> Optional[Ingredient]:
        """Retrieve an ingredient by its name and subclass type.

        Args:
            name (str): The exact name of the ingredient to look up.
            ing_type (str): The ingredient type name to match (e.g. ``"Protein"``).

        Returns:
            Optional[Ingredient]: The matching `Ingredient` subclass instance, or
                ``None`` if no ingredient with the given name and type exists.
        """
        c = self.conn.cursor()
        c.execute(
            "SELECT id, enc_id, is_user FROM ingredient WHERE name = ? AND type = ?",
            (name, ing_type),
        )
        row = c.fetchone()
        if not row:
            return None
        return self.get_ingredient(row[0])

    def update_ingredient(self, id: int, ing: Ingredient) -> bool:
        """Update an existing ingredient record and its subclass-specific details.

        Args:
            id (int): The primary key of the ingredient to update.
            ing (Ingredient): An instance of an `Ingredient` subclass containing new data.
                The `id` of `ing` is not used; only `enc_id`, `name`, `is_user`, and subclass fields are updated.

        Returns:
            bool: True if the update succeeded (ingredient existed), False if no ingredient with `id` existed.
        """
        c = self.conn.cursor()
        c.execute("SELECT 1 FROM ingredient WHERE id = ?", (id,))
        if not c.fetchone():
            return False

        c.execute(
            "UPDATE ingredient SET enc_id = ?, name = ?, type = ?, is_user = ? WHERE id = ?",
            (ing.enc_id, ing.name, type(ing).__name__, int(ing.is_user), id),
        )
        # Clear any existing subclass rows for this ingredient
        c.execute("DELETE FROM protein WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM buffer WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM stabilizer WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM surfactant WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM salt WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM excipient WHERE ingredient_id = ?", (id,))
        # Re-insert appropriate subclass row
        if isinstance(ing, Protein):
            class_val = (
                ing.class_type.value
                if isinstance(ing.class_type, ProteinClass)
                else (
                    (ing.class_type or ProteinClass.NONE).value
                    if isinstance(ing.class_type, ProteinClass) is False
                    and ing.class_type is not None
                    else ProteinClass.NONE.value
                )
            )

            c.execute(
                """
                INSERT INTO protein
                    (ingredient_id, class_type, molecular_weight, pI_mean, pI_range)
                VALUES (?, ?, ?, ?, ?)
            """,
                (id, class_val, ing.molecular_weight, ing.pI_mean, ing.pI_range),
            )
        elif isinstance(ing, Buffer):
            c.execute("INSERT INTO buffer (ingredient_id) VALUES (?)", (id,))
        elif isinstance(ing, Stabilizer):
            c.execute("INSERT INTO stabilizer VALUES (?)", (id,))
        elif isinstance(ing, Surfactant):
            c.execute("INSERT INTO surfactant VALUES (?)", (id,))
        elif isinstance(ing, Salt):
            c.execute("INSERT INTO salt VALUES (?)", (id,))
        elif isinstance(ing, Excipient):
            c.execute("INSERT INTO excipient VALUES (?)", (id,))

        self._commit()
        if self.use_encryption:
            self.backup()
        return True

    def delete_ingredient(self, id: int) -> bool:
        """Delete an ingredient and all related subclass data from the database.

        Args:
            id (int): The primary key of the ingredient to delete.

        Returns:
            bool: True if a row was deleted (ingredient existed), False otherwise.
        """
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredient WHERE id = ?", (id,))
        self._commit()
        return c.rowcount > 0

    def delete_all_ingredients(self) -> None:
        """Remove all ingredients and their subclass-specific data from the database."""
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredient")
        self._commit()

    def add_formulations_batch(self, forms: List[Formulation]) -> None:
        """Batch insert all formulations, components, and viscosity profiles in 3 SQL calls."""
        if not forms:
            return
        c = self.conn.cursor()
        for f in forms:
            c.execute(
                "INSERT INTO formulation (name, signature, temperature, icl, last_model) VALUES (?, ?, ?, ?, ?)",
                (f.name, f.signature, f.temperature, int(f.icl), f.last_model),
            )
            f.id = c.lastrowid  # type: ignore[assignment]  # lastrowid is non-None after a successful INSERT
        comp_rows = [
            (f.id, comp_type, comp.ingredient.id, comp.concentration, comp.units, comp.pH)
            for f in forms
            for comp_type, comp in f._components.items()
            if comp is not None
        ]
        c.executemany(
            "INSERT INTO formulation_component "
            "(formulation_id, component_type, ingredient_id, concentration, units, pH) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            comp_rows,
        )
        vp_rows = [
            (
                f.id,
                json.dumps(f.viscosity_profile.shear_rates),
                json.dumps(f.viscosity_profile.viscosities),
                f.viscosity_profile.units,
                int(f.viscosity_profile.is_measured),
            )
            for f in forms
            if f.viscosity_profile
        ]
        if vp_rows:
            c.executemany(
                "INSERT INTO viscosity_profile "
                "(formulation_id, shear_rates, viscosities, units, is_measured) "
                "VALUES (?, ?, ?, ?, ?)",
                vp_rows,
            )

        self._commit()
        if self.use_encryption and not self._defer_backup:
            self.backup()

    def add_formulation(self, form: Formulation) -> int:
        """Insert a new formulation, its components, and viscosity profile into the database.

        Args:
            form (Formulation): A `Formulation` instance with its `_components` dictionary
                populated (Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient as `Component`),
                and `viscosity_profile` optionally set.

        Returns:
            int: The database-assigned primary key (`id`) of the newly inserted formulation.
        """
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO formulation (name, signature, temperature, icl, last_model) VALUES (?, ?, ?, ?, ?)",
            (
                form.name,
                form.signature,
                form.temperature,
                int(form.icl),
                form.last_model,
            ),
        )
        fid = c.lastrowid

        # Insert each non-None component; ensure ingredient is also persisted
        for comp_type, comp in form._components.items():
            if comp is None:
                continue
            iid = (
                comp.ingredient.id
                if comp.ingredient.id is not None
                else self.add_ingredient(comp.ingredient)
            )
            c.execute(
                "INSERT INTO formulation_component "
                "(formulation_id, component_type, ingredient_id, concentration, units, pH) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fid, comp_type, iid, comp.concentration, comp.units, comp.pH),
            )

        # Insert viscosity profile if present
        if form.viscosity_profile:
            vp = form.viscosity_profile
            c.execute(
                "INSERT INTO viscosity_profile "
                "(formulation_id, shear_rates, viscosities, units, is_measured) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    fid,
                    json.dumps(vp.shear_rates),
                    json.dumps(vp.viscosities),
                    vp.units,
                    int(vp.is_measured),
                ),
            )

        self._commit()
        if self.use_encryption and not self._defer_backup:
            self.backup()
        form.id = fid
        return fid

    def get_formulation(self, id: int) -> Optional[Formulation]:
        """Retrieve a formulation by ID, reconstructing its components and viscosity profile.

        Args:
            id (int): The primary key of the formulation to fetch.

        Returns:
            Optional[Formulation]: A `Formulation` instance with populated components,
                temperature, and viscosity profile, or `None` if not found.
        """
        c = self.conn.cursor()

        try:
            c.execute(
                "SELECT name, signature, temperature, icl, last_model FROM formulation WHERE id = ?",
                (id,),
            )
            row = c.fetchone()
        except sqlite3.OperationalError:
            c.execute(
                "SELECT name, signature, temperature FROM formulation WHERE id = ?",
                (id,),
            )
            row = c.fetchone()
            if row:
                row = row + (1, None)

        if not row:
            return None

        name, signature, temp, icl, last_model = row

        form = Formulation(id=id, name=name, signature=signature)
        form.set_temperature(temp)
        form.icl = bool(icl) if icl is not None else True
        form.last_model = last_model

        c.execute(
            "SELECT component_type, ingredient_id, concentration, units, pH "
            "FROM formulation_component WHERE formulation_id = ?",
            (id,),
        )
        rows = c.fetchall()
        for r in rows:
            comp_type, iid, conc, units, ph = r
            ingredient = self.get_ingredient(iid)
            if ingredient and comp_type in form._components:
                form._components[comp_type] = Component(ingredient, conc, units, pH=ph)

        c.execute(
            "SELECT shear_rates, viscosities, units, is_measured "
            "FROM viscosity_profile WHERE formulation_id = ?",
            (id,),
        )
        row = c.fetchone()
        if row:
            srs, vs, units, meas = row
            vp = ViscosityProfile(json.loads(srs), json.loads(vs), units)
            vp.is_measured = bool(meas)
            form.set_viscosity_profile(vp)

        return form

    def get_all_formulations(self) -> List[Formulation]:
        """Retrieve all formulations in the database.

        Returns:
            List[Formulation]: A list of `Formulation` instances, each reconstructed via `get_formulation`.
        """
        c = self.conn.cursor()
        c.execute("SELECT id FROM formulation")
        rows = c.fetchall()

        formulations: List[Formulation] = []
        for (fid,) in rows:
            form = self.get_formulation(fid)
            if form is not None:
                formulations.append(form)

        return formulations

    def update_formulation_metadata(
        self, f_id: int, icl: bool = None, last_model: str = None
    ) -> bool:
        """Updates lightweight metadata for a formulation without affecting components.

        This method performs an in-place update of the 'icl' flag and the
        'last_model' identifier. Unlike a full record replacement, this approach
        preserves the primary key (ID) and any existing foreign key links to
        formulation components. The update is conditional; only non-None
        arguments will trigger a database change.

        Args:
            f_id: The unique primary key ID of the formulation to update.
            icl: An optional boolean flag indicating In-Concentration Loading status.
                Converted to 1 (True) or 0 (False) for SQL storage.
            last_model: An optional string identifier for the last predictive
                model run against this formulation.

        Returns:
            bool: True if at least one row was successfully modified and committed;
                False if the ID was not found or an error occurred.

        Side Effects:
            - Calls `_commit()` if changes were made.
            - Triggers `backup()` if `self.use_encryption` is True to ensure
              the encrypted disk state is synchronized.
        """
        try:
            c = self.conn.cursor()

            if icl is not None:
                c.execute(
                    "UPDATE formulation SET icl = ? WHERE id = ?",
                    (1 if icl else 0, f_id),
                )

            if last_model is not None:
                c.execute(
                    "UPDATE formulation SET last_model = ? WHERE id = ?",
                    (last_model, f_id),
                )

            if c.rowcount > 0:
                self._commit()
                if self.use_encryption:
                    self.backup()
                return True
            return False
        except Exception as e:
            Log.e(TAG, f"Error updating metadata: {e}")
            return False

    def get_formulation_by_signature(self, signature: str) -> Optional[Formulation]:
        """Retrieve a formulation by its SHA256 signature.

        Args:
            signature (str): The unique SHA256 signature string.

        Returns:
            Optional[Formulation]: The matching Formulation object, or None if not found.
        """
        c = self.conn.cursor()
        c.execute("SELECT id FROM formulation WHERE signature = ?", (signature,))
        row = c.fetchone()

        if row:
            return self.get_formulation(row[0])
        return None

    def delete_formulation_by_signature(self, signature: str) -> bool:
        """Delete a formulation identified by its signature.

        Args:
            signature (str): The unique SHA256 signature string.

        Returns:
            bool: True if a formulation was deleted, False if no match was found.
        """
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation WHERE signature = ?", (signature,))

        if c.rowcount > 0:
            self._commit()
            if self.use_encryption:
                self.backup()
            return True
        return False

    def update_formulation_name_by_signature(self, signature: str, new_name: str) -> bool:
        """Update the name of a formulation identified by its signature.

        Args:
            signature (str): The unique SHA256 signature of the formulation to update.
            new_name (str): The new name to assign.

        Returns:
            bool: True if the update was successful, False if the signature was not found.
        """
        c = self.conn.cursor()
        c.execute("UPDATE formulation SET name = ? WHERE signature = ?", (new_name, signature))

        if c.rowcount > 0:
            self._commit()
            if self.use_encryption:
                self.backup()
            return True
        return False

    def delete_formulation(self, id: int) -> bool:
        """Delete a formulation and all linked components and viscosity profile.

        Args:
            id (int): The primary key of the formulation to delete.

        Returns:
            bool: True if a row was deleted, False if no such formulation existed.
        """
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation WHERE id = ?", (id,))
        self._commit()
        if self.use_encryption:
            self.backup()
        return c.rowcount > 0

    def delete_all_formulations(self) -> None:
        """Remove all formulations, their components, and viscosity profiles from the database."""
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation")
        self._commit()
        if self.use_encryption:
            self.backup()

    def reset(self) -> None:
        """Reset the database by deleting the file (if exists) and recreating tables.

        This closes any existing connection, removes the physical file, reopens a new
        SQLite connection at the same path, and reinitializes the schema.
        """
        try:
            self.conn.close()
        except Exception:
            pass

        if self.db_path.exists():
            self.db_path.unlink()
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def set_encryption_key(self, key: Union[str, None]) -> None:
        """Set or clear the encryption key for future database saves.

        Args:
            key (Union[str, None]): New encryption key. If None, disables encryption.
        """
        self.encryption_key = key

    def _load_metadata(self) -> None:
        """Reads the JSON metadata embedded in the database file.

        Any metadata found will be parsed and stored to `self.metadata`.
        """
        try:
            self._enc_metadata = b"\x45{}\n"  # default: nop seed with empty dict
            self.metadata = {}
            self._has_metadata = False

            if not self.db_path.exists():
                Log.e(TAG, "Database file does not exist. Creating empty metadata.")
                return

            with open(self.db_path, "rb") as f:
                magic_header = f.read(16)
                if magic_header == b"SQLite format 3\x00":
                    Log.i(TAG, "Standard SQLite binary format detected. Bypassing encryption.")
                    self.use_encryption = False
                    return

                # Check if it looks like a raw SQL script instead of custom DB
                f.seek(0)
                first_chars = f.read(20).upper()
                if first_chars.startswith(b"BEGIN TRANSACTION") or first_chars.startswith(
                    b"PRAGMA"
                ):
                    Log.i(
                        TAG,
                        "Plain SQL script detected. Treating as unencrypted DB without metadata.",
                    )
                    return

                f.seek(0)
                self._enc_metadata = f.readline()
                enc_metadata = self._enc_metadata.decode(
                    self.metadata.get("app_encoding", "utf-8")
                ).rsplit("\n", 1)[
                    0
                ]  # remove at most 1 trailing '\n'

                if not enc_metadata:
                    Log.w(TAG, "Database file is empty. Bypassing encryption load.")
                    self.use_encryption = False
                    return

                seed = ord(enc_metadata[0])
                shuffled = enc_metadata[1:]
                enc_metadata = self._shuffle_text(shuffled, seed)
                shift_by = -len(enc_metadata)
                str_metadata = self._caesar_cipher(enc_metadata, shift_by)
                self.metadata = json.loads(str_metadata)
                self._has_metadata = True

        except Exception as e:
            Log.e(TAG, f"No readable metadata found in database file. Error: {e}")
            self._has_metadata = False
            self._enc_metadata = b"\x45{}\n"  # Restore default so we don't write garbage later

    def _shuffle_text(self, text: str, seed: Union[int, None] = None) -> str:
        """Shuffles the characters of a given `text` string in a pseudo-random order.

        Args:
            text (str): The string to be shuffled (or unshuffled, perhaps).
            seed (Union[int, None]): The seed for `random` to use (for repeatability).

        Returns:
            str: The character shuffled string.
        """
        if seed != None:
            random.seed(seed)
        indices = list(range(len(text)))
        random.shuffle(indices)
        original = [""] * len(text)
        for i, orig_index in enumerate(indices):
            original[orig_index] = text[i]
        return "".join(original)

    def _commit(self) -> None:
        """Commits the current transaction and synchronizes the disk state.

        This method performs a standard SQL commit on the active connection. If
        encryption is enabled, it goes a step further by performing an immediate
        persistence operation to the filesystem. To maintain data integrity during
        the encrypted write, it safely rotates the file handle—closing the current
        read handle, executing the save, and re-establishing the read handle.

        Side Effects:
            - Finalizes the current SQL transaction via `self.conn.commit()`.
            - If `self.use_encryption` is True:
                - Closes and re-opens `self.file_handle`.
                - Overwrites the file at `self.db_path` via `_save_cdb`.
                - Resets `self.init_changes` to match the connection's
                  total change count to prevent redundant backups.
        """
        if not self._defer_commit:
            self.conn.commit()
        if self.use_encryption and not self._defer_backup:
            if self.file_handle is not None:
                self.file_handle.close()
            self._save_cdb(str(self.db_path), self.encryption_key)
            self.file_handle = open(self.db_path, "rb")
            self.init_changes = self.conn.total_changes

    def flush(self) -> None:
        """Force a commit and optional backup — call at end of bulk operations."""
        self.conn.commit()
        if self.use_encryption:
            if self.file_handle is not None:
                self.file_handle.close()
            self._save_cdb(str(self.db_path), self.encryption_key)
            self.file_handle = open(self.db_path, "rb")
            self.init_changes = self.conn.total_changes

    def _caesar_cipher(self, text: str, shift: int = 0) -> str:
        """Apply a Caesar cipher shift to a text string, rotating alphanumeric characters.

        Args:
            text (str): The plaintext string to encode.
            shift (int, optional): Shift amount. If 0, defaults to `len(text)`. Defaults to 0.

        Returns:
            str: The Caesar-ciphered string.
        """
        result = []
        if shift == 0:
            shift = len(text)
        for char in text:
            if char.isdigit():
                base = ord("0")
                result.append(chr((ord(char) - base + shift) % 10 + base))
            elif char.isupper():
                base = ord("A")
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif char.islower():
                base = ord("a")
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif ord(char) in range(32, 48):
                base = 32
                result.append(chr((ord(char) - base + shift) % len(range(32, 48)) + base))
            elif ord(char) in range(58, 65):
                base = 58
                result.append(chr((ord(char) - base + shift) % len(range(58, 65)) + base))
            elif ord(char) in range(91, 97):
                base = 91
                result.append(chr((ord(char) - base + shift) % len(range(91, 97)) + base))
            elif ord(char) in range(123, 127):
                base = 123
                result.append(chr((ord(char) - base + shift) % len(range(123, 127)) + base))
        return "".join(result)

    def _xor_cipher(self, data: bytes, key: str) -> bytes:
        """Apply an XOR cipher to a byte sequence using a repeating key.

        Args:
            data (bytes): Bytes to encrypt or decrypt.
            key (str): String key used for XOR; repeated as needed.

        Returns:
            bytes: Resulting encrypted or decrypted bytes.
        """
        key_bytes = key.encode(self.metadata.get("app_encoding", "utf-8"))
        data_arr = np.frombuffer(data, dtype=np.uint8)
        # Tile key to match data length, then XOR in one vectorized op
        repeats = len(data) // len(key_bytes) + 1
        key_arr = np.frombuffer(key_bytes * repeats, dtype=np.uint8)[: len(data)]
        return (data_arr ^ key_arr).tobytes()

    def _open_cdb(self, filepath: str, password: str) -> sqlite3.Connection:
        """Open an AES+Caesar+XOR encrypted database file into an in-memory SQLite connection.

        Reads the encrypted file from disk, applies Caesar+XOR decryption using `password`,
        and executes the SQL script into a new in-memory database.

        Args:
            filepath (str): Path to the encrypted database file.
            password (str): Encryption key used to decrypt.

        Returns:
            sqlite3.Connection: An in-memory connection populated with decrypted schema and data.

        Raises:
            IOError: If the file cannot be read.
        """
        con = sqlite3.connect(":memory:")
        if os.path.isfile(filepath):
            self.file_handle = open(filepath, "rb")

            # Only consume the first line if we successfully validated it as metadata
            if getattr(self, "_has_metadata", False):
                self._enc_metadata = self.file_handle.readline()

            secure_bytes = self.file_handle.read()
            if password:
                # Decrypt the file content
                decrypted_text = self._xor_cipher(secure_bytes, self._caesar_cipher(password))
            else:
                decrypted_text = secure_bytes
            decrypted_text = decrypted_text.decode(self.metadata.get("app_encoding", "utf-8"))
            con.executescript(decrypted_text)
            con.commit()
        return con

    def _save_cdb(self, filepath: str, password: str):
        """Encrypt and save the in-memory SQLite database to disk.

        Dumps the in-memory database to SQL text, applies Caesar+XOR encryption using `password`,
        and writes the result to `filepath`.

        Args:
            filepath (str): Path to write the encrypted database.
            password (str): Encryption key used to encrypt.
        """
        encoding = self.metadata.get("app_encoding", "utf-8")
        dump_bytes = b"".join((line + "\n").encode(encoding) for line in self.conn.iterdump())
        if password:
            encrypted = self._xor_cipher(dump_bytes, self._caesar_cipher(password))
        else:
            encrypted = dump_bytes
        with open(filepath, "wb") as f:
            f.write(self._enc_metadata)  # must end with "\n"
            f.write(encrypted)

    def close(self) -> None:
        """Close the database connection, saving to disk if changes occurred.

        If `use_encryption` is True, writes the encrypted dump back to `db_path`.
        Otherwise, defaults to the inline commits already performed. Ensures any
        open file_handle is closed to release locks.
        """
        if self.file_handle is not None:
            self.file_handle.close()
        if self.is_open:
            if self.conn.total_changes > self.init_changes or not os.path.isfile(self.db_path):
                if self.use_encryption:
                    self._save_cdb(self.db_path, self.encryption_key)
                else:
                    pass
            self.conn.close()
            self.is_open = False

    def create_temp_decrypt(self) -> Optional[Path]:
        """Create a temporary decrypted copy of the current database.

        This method generates a standard SQLite database file in the system's
        temporary directory that contains all the same data as the current
        database, but without encryption. This is useful for operations that
        require direct SQLite access or third-party tools.

        The temporary file is tracked internally and should be removed using
        `cleanup_temp_decrypt()` when no longer needed.

        Returns:
            Optional[Path]: Path to the temporary decrypted database file, or
            None if the operation failed.

        Notes:
            - The temporary file will have a '.db' extension.
            - The temporary file contains sensitive data in plaintext.
            - The file descriptor is closed immediately after creation.
        """
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".db", prefix="app_temp_")
            temp_path = Path(temp_path)
            os.close(temp_fd)
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                # Best-effort; continue on platforms where chmod is a no-op
                pass
            temp_conn = sqlite3.connect(str(temp_path))
            sql_script = "\n".join(self.conn.iterdump())
            temp_conn.executescript(sql_script)
            temp_conn.commit()
            temp_conn.close()
            if not hasattr(self, "_temp_db_paths"):
                self._temp_db_paths = []
            self._temp_db_paths.append(temp_path)

            return temp_path

        except Exception as e:
            Log.e(f"Failed to create temporary decrypted database: {e}")
            # Best-effort cleanup of partially created file
            try:
                if "temp_path" in locals() and Path(temp_path).exists():
                    Path(temp_path).unlink()
            except OSError as oe:
                Log.e(f"Failed to remove temp file {temp_path}: {oe}")
            return None

    def update_metadata_version(self, new_version: int) -> None:
        """Update the db_version in the encrypted metadata header."""
        self.metadata["db_version"] = new_version
        dumpster = json.dumps(self.metadata)
        ciphered = self._caesar_cipher(dumpster)
        seed = int(time.time() * 1000) % 255
        if seed == 10:
            seed += 1
        random.seed(seed)
        indices = list(range(len(ciphered)))
        random.shuffle(indices)
        shuffled = "".join(ciphered[i] for i in indices)
        enc_encoding = self.metadata.get("app_encoding", "utf-8")
        self._enc_metadata = (chr(seed) + shuffled).encode(enc_encoding) + b"\n"

    def cleanup_temp_decrypt(self, temp_path: Optional[Path] = None) -> bool:
        """Remove one or all temporary decrypted database files.

        This method deletes temporary decrypted database files created by
        `create_temp_decrypt()`. You can specify a particular file to remove,
        or omit the argument to clean up all temporary files associated with
        this instance.

        Args:
            temp_path (Optional[Path]): Path to a specific temporary database
                to remove. If None, all temporary databases created by this
                instance will be removed.

        Returns:
            bool: True if the cleanup operation completed successfully for all
            targeted files, False if any deletion failed.

        Notes:
            - Attempting to delete a non-existent file will be ignored.
            - All files tracked internally in `_temp_db_paths` are candidates
            for removal when `temp_path` is None.
        """
        try:
            if not hasattr(self, "_temp_db_paths"):
                return True

            if temp_path is not None:
                if temp_path.exists():
                    temp_path.unlink()
                if temp_path in self._temp_db_paths:
                    self._temp_db_paths.remove(temp_path)
            else:
                for path in self._temp_db_paths[:]:
                    try:
                        if path.exists():
                            path.unlink()
                        self._temp_db_paths.remove(path)
                    except Exception as e:
                        Log.e(f"Failed to remove temp file {path}: {e}")

            return True

        except Exception as e:
            Log.e(f"Failed to cleanup temporary database: {e}")
            return False

    def __del__(self):
        """Destructor to ensure temporary decrypted files are cleaned up.

        When the object is destroyed, this method automatically calls
        `cleanup_temp_decrypt()` to remove any temporary decrypted database
        files created during the instance's lifetime. This helps prevent
        sensitive data from persisting on disk.
        """
        if hasattr(self, "_temp_db_paths"):
            self.cleanup_temp_decrypt()

    def backup(self) -> None:
        """Persists the in-memory database state to the physical disk.

        This method synchronizes the current connection state with the filesystem.
        Unlike the `close()` method, `backup()` ensures the database connection
        remains active and open for subsequent operations. It evaluates whether
        changes have occurred since the last initialization or if the database
        file is missing before triggering a save.

        The method automatically handles two storage modes:
            1. Encrypted: Saves the state using a custom CDB implementation
               with the provided encryption key.
            2. Standard: Performs a standard SQL commit to the disk.

        Side Effects:
            - Closes and re-opens `self.file_handle` to ensure the disk state
              is current.
            - Updates `self.init_changes` to the current total change count
              upon a successful save.

        Notes:
            This is particularly useful for long-running processes where
            periodic data safety is required without interrupting the
            active session.
        """
        if self.file_handle is not None:
            self.file_handle.close()
        if self.conn.total_changes > self.init_changes or not os.path.isfile(self.db_path):
            if self.use_encryption:
                self._save_cdb(self.db_path, self.encryption_key)
            else:
                self._commit()
            self.init_changes = self.conn.total_changes
        self.file_handle = open(self.db_path, "rb")

    def begin_bulk(self) -> None:
        """Apply performance PRAGMAs to speed up bulk insert operations.

        Switches the journal mode to in-memory and relaxes synchronization
        constraints so that large batches of inserts complete faster. Call
        `end_bulk()` when the batch is finished to restore safe write settings.
        """
        self.conn.execute("PRAGMA journal_mode = MEMORY")
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA cache_size = -64000")

    def end_bulk(self) -> None:
        """Restore safe write settings after bulk imports.

        This method only resets the connection PRAGMAs; it does **not** commit
        the current transaction.  The caller is responsible for committing (via
        ``_commit()``) or rolling back after calling this method.
        """
        self.conn.execute("PRAGMA journal_mode = DELETE")
        self.conn.execute("PRAGMA synchronous = FULL")
