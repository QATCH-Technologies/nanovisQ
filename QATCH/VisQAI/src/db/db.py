"""
database.py

This module provides the `Database` class for managing persistence of ingredients,
formulations, components, and viscosity profiles using SQLite. It supports optional
in-memory encryption via a simple XOR+Caesar cipher. The schema includes tables for
generic ingredients, subclass-specific ingredient details, formulation records, and
associated component and viscosity profile data. Utility methods handle CRUD operations,
schema initialization, and optional encryption/decryption of the entire database.

Author:
    Alexander Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.4
"""

import sqlite3
import json
from typing import List, Optional, Union
from pathlib import Path
import os

try:
    from src.models.ingredient import (
        Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt
    )
    from src.models.formulation import Formulation, Component, ViscosityProfile
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import (
        Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt
    )
    from QATCH.VisQAI.src.models.formulation import Formulation, Component, ViscosityProfile

DB_PATH = Path(
    os.path.join(
        os.path.expandvars(r"%LOCALAPPDATA%"),
        "QATCH", "nanovisQ", "data", "app.db"
    )
)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


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
        encryption_key: Union[str, None] = None
    ) -> None:
        """Initialize the Database, apply encryption if requested, and create tables.

        Args:
            path (Union[str, Path]): Filesystem path to the SQLite database file.
                Defaults to a standard `app.db` location under %LOCALAPPDATA%.
            encryption_key (Union[str, None], optional): If provided, uses
                in-memory Caesar+XOR encryption/decryption. If None, plain SQLite
                is used. Defaults to None.

        Raises:
            ValueError: If `encryption_key` is None but `use_encryption` is True
                (e.g., encryption_key provided as empty).
        """
        self.db_path = Path(path)
        self.file_handle = None  # used for locking when encrypted
        self.encryption_key = encryption_key
        self.use_encryption = False if self.encryption_key is None else True
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_encryption:
            if not self.encryption_key:
                raise ValueError(
                    "Encryption key required for encrypted database")
            self.conn = self._open_cdb(str(self.db_path), self.encryption_key)
        else:
            self.conn = sqlite3.connect(str(self.db_path))

        # Enforce foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_changes = self.conn.total_changes
        self._create_tables()

    def _create_tables(self) -> None:
        """Create all necessary tables if they do not already exist.

        The tables include:
            - ingredient: core table with common fields (id, enc_id, name, type, is_user).
            - protein, buffer, stabilizer, surfactant, salt: subclass tables referencing ingredient.
            - formulation: core formulation table (id, temperature).
            - formulation_component: linking table between formulations and ingredient components.
            - viscosity_profile: table storing JSON-serialized shear_rates and viscosities.

        This method commits the changes to the database.
        """
        c = self.conn.cursor()
        # Ingredient core table
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS ingredient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                enc_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                is_user INTEGER NOT NULL DEFAULT 0
            )
        """)
        # Subclass tables
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS protein (
                ingredient_id   INTEGER PRIMARY KEY,
                molecular_weight REAL,
                pI_mean          REAL,
                pI_range         REAL,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """)
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS buffer (
                ingredient_id INTEGER PRIMARY KEY,
                pH REAL,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """)
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS stabilizer (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """)
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS surfactant (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """)
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS salt (
                ingredient_id INTEGER PRIMARY KEY,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        """)
        # Formulation and components
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS formulation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature REAL
            )
        """)
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS formulation_component (
                formulation_id INTEGER NOT NULL,
                component_type TEXT NOT NULL,
                ingredient_id INTEGER NOT NULL,
                concentration REAL NOT NULL,
                units TEXT NOT NULL,
                PRIMARY KEY (formulation_id, component_type),
                FOREIGN KEY (formulation_id) REFERENCES formulation(id) ON DELETE CASCADE,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id)
            )
        """)
        c.execute(r"""
            CREATE TABLE IF NOT EXISTS viscosity_profile (
                formulation_id INTEGER PRIMARY KEY,
                shear_rates TEXT NOT NULL,
                viscosities TEXT NOT NULL,
                units TEXT NOT NULL,
                is_measured INTEGER NOT NULL,
                FOREIGN KEY (formulation_id) REFERENCES formulation(id) ON DELETE CASCADE
            )
        """)
        self.conn.commit()

    def add_ingredient(self, ing: Ingredient) -> int:
        """Insert a new ingredient and its subclass-specific details into the database.

        Args:
            ing (Ingredient): An instance of a subclass of `Ingredient` (Protein, Buffer, Stabilizer, Surfactant, Salt).
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
            "INSERT INTO ingredient (enc_id, name, type, is_user) VALUES (?, ?, ?, ?)",
            (ing.enc_id, ing.name, type(ing).__name__, int(ing.is_user))
        )
        db_id = c.lastrowid

        # Insert into the subclass table based on runtime type
        if isinstance(ing, Protein):
            c.execute(
                "INSERT INTO protein VALUES (?, ?, ?, ?)",
                (db_id, ing.molecular_weight, ing.pI_mean, ing.pI_range)
            )
        elif isinstance(ing, Buffer):
            c.execute(
                "INSERT INTO buffer VALUES (?, ?)",
                (db_id, ing.pH)
            )
        elif isinstance(ing, Stabilizer):
            c.execute("INSERT INTO stabilizer VALUES (?)", (db_id,))
        elif isinstance(ing, Surfactant):
            c.execute("INSERT INTO surfactant VALUES (?)", (db_id,))
        elif isinstance(ing, Salt):
            c.execute("INSERT INTO salt VALUES (?)", (db_id,))

        self.conn.commit()
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
        c.execute(
            "SELECT enc_id, name, type, is_user FROM ingredient WHERE id = ?",
            (id,)
        )
        row = c.fetchone()
        if not row:
            return None
        enc_id, name, typ, is_user = row

        # Reconstruct subclass-specific object
        if typ == "Protein":
            c.execute(
                "SELECT molecular_weight, pI_mean, pI_range FROM protein WHERE ingredient_id = ?",
                (id,)
            )
            mw, mean, rng = c.fetchone()
            ing = Protein(enc_id, name, mw, mean, rng)

        elif typ == "Buffer":
            c.execute(
                "SELECT pH FROM buffer WHERE ingredient_id = ?",
                (id,)
            )
            (pH,) = c.fetchone()
            ing = Buffer(enc_id, name, pH)

        elif typ == "Stabilizer":
            ing = Stabilizer(enc_id, name)

        elif typ == "Surfactant":
            ing = Surfactant(enc_id, name)

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
            (ing.enc_id, ing.name, type(ing).__name__, int(ing.is_user), id)
        )
        # Clear any existing subclass rows for this ingredient
        c.execute("DELETE FROM protein WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM buffer WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM stabilizer WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM surfactant WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM salt WHERE ingredient_id = ?", (id,))

        # Re-insert appropriate subclass row
        if isinstance(ing, Protein):
            c.execute(
                "INSERT INTO protein VALUES (?, ?, ?, ?)",
                (id, ing.molecular_weight, ing.pI_mean, ing.pI_range)
            )
        elif isinstance(ing, Buffer):
            c.execute("INSERT INTO buffer VALUES (?, ?)", (id, ing.pH))
        elif isinstance(ing, Stabilizer):
            c.execute("INSERT INTO stabilizer VALUES (?)", (id,))
        elif isinstance(ing, Surfactant):
            c.execute("INSERT INTO surfactant VALUES (?)", (id,))
        elif isinstance(ing, Salt):
            c.execute("INSERT INTO salt VALUES (?)", (id,))

        self.conn.commit()
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
        self.conn.commit()
        return c.rowcount > 0

    def delete_all_ingredients(self) -> None:
        """Remove all ingredients and their subclass-specific data from the database."""
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredient")
        self.conn.commit()

    def add_formulation(self, form: Formulation) -> int:
        """Insert a new formulation, its components, and viscosity profile into the database.

        Args:
            form (Formulation): A `Formulation` instance with its `_components` dictionary
                populated (Protein, Buffer, Stabilizer, Surfactant, Salt as `Component`),
                and `viscosity_profile` optionally set.

        Returns:
            int: The database-assigned primary key (`id`) of the newly inserted formulation.
        """
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO formulation (temperature) VALUES (?)",
            (form.temperature,)
        )
        fid = c.lastrowid

        # Insert each non-None component; ensure ingredient is also persisted
        for comp_type, comp in form._components.items():
            if comp is None:
                continue
            iid = self.add_ingredient(comp.ingredient)
            c.execute(
                "INSERT INTO formulation_component "
                "(formulation_id, component_type, ingredient_id, concentration, units) "
                "VALUES (?, ?, ?, ?, ?)",
                (fid, comp_type, iid, comp.concentration, comp.units)
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
                    int(vp.is_measured)
                )
            )

        self.conn.commit()
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
        c.execute("SELECT temperature FROM formulation WHERE id = ?", (id,))
        row = c.fetchone()
        if not row:
            return None
        (temp,) = row

        form = Formulation(id)
        form.id = id
        form.set_temperature(temp)

        # Load component rows and set each on the Formulation
        c.execute(
            "SELECT component_type, ingredient_id, concentration, units "
            "FROM formulation_component WHERE formulation_id = ?",
            (id,)
        )
        for comp_type, iid, conc, units in c.fetchall():
            ing = self.get_ingredient(iid)
            if ing:
                setter = getattr(form, f"set_{comp_type}")
                setter(ing, conc, units)

        # Load viscosity profile if present
        c.execute(
            "SELECT shear_rates, viscosities, units, is_measured "
            "FROM viscosity_profile WHERE formulation_id = ?",
            (id,)
        )
        row = c.fetchone()
        if row:
            srs, vs, units, meas = row
            vp = ViscosityProfile(
                json.loads(srs), json.loads(vs), units
            )
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

    def delete_formulation(self, id: int) -> bool:
        """Delete a formulation and all linked components and viscosity profile.

        Args:
            id (int): The primary key of the formulation to delete.

        Returns:
            bool: True if a row was deleted, False if no such formulation existed.
        """
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation WHERE id = ?", (id,))
        self.conn.commit()
        return c.rowcount > 0

    def delete_all_formulations(self) -> None:
        """Remove all formulations, their components, and viscosity profiles from the database."""
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation")
        self.conn.commit()

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
            else:
                result.append(char)
        return "".join(result)

    def _xor_cipher(self, data: bytes, key: str) -> bytes:
        """Apply an XOR cipher to a byte sequence using a repeating key.

        Args:
            data (bytes): Bytes to encrypt or decrypt.
            key (str): String key used for XOR; repeated as needed.

        Returns:
            bytes: Resulting encrypted or decrypted bytes.
        """
        key_bytes = key.encode("utf-8")
        key_length = len(key_bytes)
        return bytes([b ^ key_bytes[i % key_length] for i, b in enumerate(data)])

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
            secure = self.file_handle.read()
            # Decrypt the file content
            decrypted_text = self._xor_cipher(
                secure, self._caesar_cipher(password)
            ).decode("utf-8")
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
        dump_bytes = b"".join((line + "\n").encode("utf-8")
                              for line in self.conn.iterdump())
        encrypted = self._xor_cipher(dump_bytes, self._caesar_cipher(password))
        with open(filepath, "wb") as f:
            f.write(encrypted)

    def close(self) -> None:
        """Close the database connection, saving to disk if changes occurred.

        If `use_encryption` is True, writes the encrypted dump back to `db_path`.
        Otherwise, defaults to the inline commits already performed. Ensures any
        open file_handle is closed to release locks.
        """
        if self.file_handle is not None:
            self.file_handle.close()
        # If there are changes or the file does not exist, save or commit
        if self.conn.total_changes > self.init_changes or not os.path.isfile(self.db_path):
            if self.use_encryption:
                self._save_cdb(self.db_path, self.encryption_key)
            else:
                # No additional action needed; changes have been committed inline.
                pass
        self.conn.close()
