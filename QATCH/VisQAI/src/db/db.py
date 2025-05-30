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
    def __init__(self, path: Union[str, Path] = DB_PATH, encryption_key: Union[str, None] = None) -> None:
        self.db_path = Path(path)
        self.file_handle = None  # used for locking db file when encrypted
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
        self.conn.execute('PRAGMA foreign_keys = ON')
        self.init_changes = self.conn.total_changes
        self._create_tables()

    def _create_tables(self) -> None:
        c = self.conn.cursor()
        # Ingredient core table
        c.execute(r'''
            CREATE TABLE IF NOT EXISTS ingredient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                enc_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                is_user INTEGER NOT NULL DEFAULT 0
            )
        ''')
        # Subclass tables
        c.execute(r'''
            CREATE TABLE IF NOT EXISTS protein (
                ingredient_id   INTEGER PRIMARY KEY,
                molecular_weight REAL,
                pI_mean          REAL,
                pI_range         REAL,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        ''')
        c.execute(r'''
            CREATE TABLE IF NOT EXISTS buffer (
                ingredient_id INTEGER PRIMARY KEY,
                pH REAL,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        ''')
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

     # Ingredient CRUD
    def add_ingredient(self, ing: Ingredient) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO ingredient (enc_id, name, type, is_user) VALUES (?, ?, ?, ?)",
            (
                ing.enc_id,
                ing.name,
                type(ing).__name__,
                int(ing.is_user)
            )
        )
        db_id = c.lastrowid
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
        c = self.conn.cursor()
        c.execute(
            "SELECT enc_id, name, type, is_user FROM ingredient WHERE id = ?",
            (id,)
        )
        row = c.fetchone()
        if not row:
            return None
        enc_id, name, typ, is_user = row

        if typ == 'Protein':
            c.execute(
                "SELECT molecular_weight, pI_mean, pI_range FROM protein WHERE ingredient_id = ?",
                (id,)
            )
            mw, mean, rng = c.fetchone()
            ing = Protein(enc_id, name, mw, mean, rng)

        elif typ == 'Buffer':
            c.execute(
                "SELECT pH FROM buffer WHERE ingredient_id = ?",
                (id,)
            )
            (pH,) = c.fetchone()
            ing = Buffer(enc_id, name, pH)

        elif typ == 'Stabilizer':
            ing = Stabilizer(enc_id, name)

        elif typ == 'Surfactant':
            ing = Surfactant(enc_id, name)

        elif typ == 'Salt':
            ing = Salt(enc_id, name)

        else:
            return None

        ing.id = id
        ing.is_user = bool(is_user)
        return ing

    def get_all_ingredients(self) -> List[Ingredient]:
        c = self.conn.cursor()
        c.execute("SELECT id FROM ingredient")
        ids = [r[0] for r in c.fetchall()]
        return [self.get_ingredient(i) for i in ids]

    def update_ingredient(self, id: int, ing: Ingredient) -> bool:
        c = self.conn.cursor()
        c.execute("SELECT 1 FROM ingredient WHERE id = ?", (id,))
        if not c.fetchone():
            return False
        c.execute(
            "UPDATE ingredient SET enc_id = ?, name = ?, type = ?, is_user = ? WHERE id = ?",
            (
                ing.enc_id,
                ing.name,
                type(ing).__name__,
                int(ing.is_user),
                id
            )
        )
        c.execute("DELETE FROM protein WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM buffer WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM stabilizer WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM surfactant WHERE ingredient_id = ?", (id,))
        c.execute("DELETE FROM salt WHERE ingredient_id = ?", (id,))
        # reinsert
        if isinstance(ing, Protein):
            c.execute("INSERT INTO protein VALUES (?, ?, ?, ?)",
                      (id, ing.molecular_weight, ing.pI_mean, ing.pI_range))
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
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredient WHERE id = ?", (id,))
        self.conn.commit()
        return c.rowcount > 0

    def delete_all_ingredients(self) -> None:
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredient")
        self.conn.commit()

    # Formulation CRUD
    def add_formulation(self, form: Formulation) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO formulation (temperature) VALUES (?)",
            (form.temperature,)
        )
        fid = c.lastrowid

        # components
        for comp_type, comp in form._components.items():
            if comp is None:
                continue
            # ensure ingredient is in DB (and its .id is set)
            iid = self.add_ingredient(comp.ingredient)
            c.execute(
                "INSERT INTO formulation_component "
                "(formulation_id, component_type, ingredient_id, concentration, units) "
                "VALUES (?,?,?,?,?)",
                (fid, comp_type, iid, comp.concentration, comp.units)
            )

        # viscosity
        if form.viscosity_profile:
            vp = form.viscosity_profile
            c.execute(
                "INSERT INTO viscosity_profile "
                "(formulation_id, shear_rates, viscosities, units, is_measured) "
                "VALUES (?,?,?,?,?)",
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
        c = self.conn.cursor()
        c.execute("SELECT temperature FROM formulation WHERE id = ?", (id,))
        row = c.fetchone()
        if not row:
            return None

        (temp,) = row
        form = Formulation(id)
        form.id = id

        form.set_temperature(temp)

        # load components
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

        # load viscosity
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
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation WHERE id = ?", (id,))
        self.conn.commit()
        return c.rowcount > 0

    def delete_all_formulations(self) -> None:
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation")
        self.conn.commit()

    def reset(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

        if self.db_path.exists():
            self.db_path.unlink()
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute('PRAGMA foreign_keys = ON')
        self._create_tables()

    def set_encryption_key(self, key: Union[str, None]) -> None:
        self.encryption_key = key

    def _caesar_cipher(self, text: str, shift: int = 0) -> str:
        result = []
        if shift == 0:
            shift = len(text)
        for char in text:
            if char.isdigit():
                base = ord('0')
                result.append(chr((ord(char) - base + shift) % 10 + base))
            elif char.isupper():
                base = ord('A')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            elif char.islower():
                base = ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
        return ''.join(result)

    def _xor_cipher(self, data: bytes, key: str) -> bytes:
        key_bytes = key.encode('utf-8')
        key_length = len(key_bytes)
        return bytes([b ^ key_bytes[i % key_length] for i, b in enumerate(data)])

    def _open_cdb(self, filepath: str, password: str) -> sqlite3.Connection:
        con = sqlite3.connect(':memory:')
        if os.path.isfile(filepath):
            self.file_handle = open(filepath, 'rb')
            secure = self.file_handle.read()
            # NOTE: Leave file_handle open, keeping it locked by process
            # self.file_handle.close()
            decrypted = self._xor_cipher(
                secure, self._caesar_cipher(password)).decode('utf-8')
            con.executescript(decrypted)
            con.commit()
        return con

    def _save_cdb(self, filepath: str, password: str):
        dump_bytes = b''.join((line + '\n').encode('utf-8')
                              for line in self.conn.iterdump())
        encrypted = self._xor_cipher(dump_bytes, self._caesar_cipher(password))
        with open(filepath, 'wb') as f:
            f.write(encrypted)

    def close(self) -> None:
        if self.file_handle is not None:
            self.file_handle.close()  # release file lock
        if self.conn.total_changes > self.init_changes or not os.path.isfile(self.db_path):
            if self.use_encryption:
                self._save_cdb(self.db_path, self.encryption_key)
            else:
                pass  # self.conn.commit() called inline on execute()
        self.conn.close()
