import sqlite3
import json
from typing import List, Optional, Union
from pathlib import Path
import os
from src.models.ingredient import (
    Ingredient, Buffer, Protein, Stabilizer, Surfactant, Salt
)
from src.models.formulation import Formulation, Component, ViscosityProfile

DB_PATH = Path(
    os.path.join(
        os.path.expandvars(r"%LOCALAPPDATA%"),
        "QATCH", "nanovisQ", "data", "app.db"
    )
)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class Database:
    def __init__(self, path: Union[str, Path] = DB_PATH) -> None:
        self.db_path = Path(path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute('PRAGMA foreign_keys = ON')
        self._create_tables()

    def _create_tables(self) -> None:
        c = self.conn.cursor()
        # Ingredient core table
        c.execute(r'''
            CREATE TABLE IF NOT EXISTS ingredient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                enc_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL
            )
        ''')
        # Subclass tables
        c.execute(r'''
            CREATE TABLE IF NOT EXISTS protein (
                ingredient_id INTEGER PRIMARY KEY,
                molecular_weight REAL NOT NULL,
                pI_mean REAL NOT NULL,
                pI_range REAL NOT NULL,
                FOREIGN KEY (ingredient_id) REFERENCES ingredient(id) ON DELETE CASCADE
            )
        ''')
        c.execute(r'''
            CREATE TABLE IF NOT EXISTS buffer (
                ingredient_id INTEGER PRIMARY KEY,
                pH REAL NOT NULL,
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
            "INSERT INTO ingredient (enc_id, name, type) VALUES (?, ?, ?)",
            (ing.enc_id, ing.name, type(ing).__name__)
        )
        db_id = c.lastrowid
        # subclass details
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
        return db_id

    def get_ingredient(self, id: int) -> Optional[Ingredient]:
        c = self.conn.cursor()
        c.execute("SELECT enc_id, name, type FROM ingredient WHERE id = ?", (id,))
        row = c.fetchone()
        if not row:
            return None
        enc_id, name, typ = row
        # fetch subclass
        if typ == 'Protein':
            c.execute(
                "SELECT molecular_weight, pI_mean, pI_range FROM protein WHERE ingredient_id = ?", (id,))
            mw, mean, rng = c.fetchone()
            return Protein(enc_id, name, mw, mean, rng)
        if typ == 'Buffer':
            c.execute("SELECT pH FROM buffer WHERE ingredient_id = ?", (id,))
            (pH,) = c.fetchone()
            return Buffer(enc_id, name, pH)
        if typ == 'Stabilizer':
            return Stabilizer(enc_id, name)
        if typ == 'Surfactant':
            return Surfactant(enc_id, name)
        if typ == 'Salt':
            return Salt(enc_id, name)
        return None

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
            "UPDATE ingredient SET enc_id = ?, name = ?, type = ? WHERE id = ?",
            (ing.enc_id, ing.name, type(ing).__name__, id)
        )
        # clear subclass
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
            # assume ingredient already in DB and comp.ingredient._db_id set
            iid = self.add_ingredient(comp.ingredient)
            c.execute(
                "INSERT INTO formulation_component VALUES (?,?,?,?,?)",
                (fid, comp_type, iid, comp.concentration, comp.units)
            )
        # viscosity
        if form.viscosity_profile:
            vp = form.viscosity_profile
            c.execute("INSERT INTO viscosity_profile VALUES (?,?,?,?,?)",
                      (fid,
                       json.dumps(vp.shear_rates),
                       json.dumps(vp.viscosities),
                       vp.units,
                       int(vp.is_measured)))
        self.conn.commit()
        return fid

    def get_formulation(self, id: int) -> Optional[Formulation]:
        c = self.conn.cursor()
        c.execute("SELECT temperature FROM formulation WHERE id = ?", (id,))
        row = c.fetchone()
        if not row:
            return None
        temp, = row
        form = Formulation(id)
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
        ids = [r[0] for r in c.fetchall()]
        return [self.get_formulation(i) for i in ids]

    def delete_formulation(self, id: int) -> bool:
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation WHERE id = ?", (id,))
        self.conn.commit()
        return c.rowcount > 0

    def delete_all_formulations(self) -> None:
        c = self.conn.cursor()
        c.execute("DELETE FROM formulation")
        self.conn.commit()
