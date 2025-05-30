
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


class FormulationController():
    def __init__(self, db: Database):
        self.db: Database = db
        self.ingredient_controller: IngredientController = IngredientController(
            self.db)

    def get_all_formulations(self) -> List[Formulation]:
        return self.db.get_all_formulations()

    def get_formulation_by_id(self, id: int) -> Formulation:
        return self.db.get_formulation(id)

    def find_id(self, formulation: Formulation) -> Formulation:
        formulations = self.get_all_formulations()
        for f in formulations:
            if f == formulation:
                return f.id
        raise ValueError(
            f"Formulation with params\n\t'{formulation.to_dict()}'\nnot found.")

    def add_formulation(self, formulation: Formulation) -> Formulation:

        try:
            buffer = formulation.buffer.ingredient
            self.ingredient_controller.add(buffer)
        except ValueError:
            pass
        try:
            protein = formulation.protein.ingredient
            self.ingredient_controller.add(protein)
        except ValueError:
            pass
        try:
            salt = formulation.salt.ingredient
            self.ingredient_controller.add(salt)
        except ValueError:
            pass
        try:
            surfactant = formulation.surfactant.ingredient
            self.ingredient_controller.add(surfactant)
        except ValueError:
            pass
        try:
            stabilizer = formulation.stabilizer.ingredient
            self.ingredient_controller.add(stabilizer)
        except ValueError:
            pass
        formulations = self.get_all_formulations()
        for f in formulations:
            print(f, formulation)
            if f == formulation:
                return f
        self.db.add_formulation(formulation)
        return formulation

    def delete_formulation_by_id(self, id: int) -> Formulation:
        formulation = self.get_formulation_by_id(id)
        if formulation is None:
            raise ValueError(f"Formulation with id '{id}' does not exist.")
        self.db.delete_formulation(id)
        return formulation

    def update_formulation(self, id: int, f_new: Formulation) -> Formulation:
        f_fetch = self.get_formulation_by_id(id)
        if f_fetch is None:
            raise ValueError(f"Formulation with id '{id}' does not exist.")
        if f_fetch == f_new:
            return f_new

        self.db.delete_formulation(id)
        self.db.add_formulation(f_new)
        return f_new

    def add_all_from_dataframe(self, df: pd.DataFrame) -> List[Formulation]:
        added_forms: List[Formulation] = []
        shear_rates = [100, 1000, 10000, 100000, 15000000]
        expected = {
            "Protein_type", "MW", "PI_mean", "PI_range", "Protein_conc",
            "Temperature",
            "Buffer_type", "Buffer_pH", "Buffer_conc",
            "Salt_type", "Salt_conc",
            "Sugar_type", "Sugar_conc",
            "Surfactant_type", "Surfactant_conc",
            *{f"Viscosity_{r}" for r in shear_rates},
        }
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: `{missing}`")

        for _, row in df.iterrows():
            protein = self.ingredient_controller.add_protein(
                Protein(enc_id=0, name=str(row["Protein_type"]), molecular_weight=float(
                    row["Molecular_weight"]), pI_mean=float(row["pI_mean"]), pI_range=float(row["pI_range"])))
            buffer = self.ingredient_controller.add_buffer(
                Buffer(enc_id=0, name=str(row['Buffer_type']), pH=row["Buffer_pH"]))
            stabilizer = self.ingredient_controller.add_stabilizer(
                Stabilizer(enc_id=0, name=str(row['Stabilizer_type'])))
            surfactant = self.ingredient_controller.add_surfactant(
                Surfactant(enc_id=0, name=str(row["Surfactant_type"])))
            salt = self.ingredient_controller.add_salt(
                Salt(enc_id=0, name=str(row["Salt_type"])))

            vp = None
            vis_values = [row[f"Viscosity_{r}"] for r in shear_rates]
            if any(pd.notna(v) for v in vis_values):
                vp = ViscosityProfile(
                    shear_rates=shear_rates, viscosities=vis_values, units="cP")
            if vp is None:
                vp = ViscosityProfile(
                    shear_rates=shear_rates, viscosities=[-1, -1, -1, -1, -1], units="unset")

            form = Formulation()
            form.set_buffer(
                buffer=buffer, concentration=row["Buffer_conc"], units="mg/mL")
            form.set_protein(
                protein=protein, concentration=row["Protein_conc"], units="mg/mL")
            form.set_stabilizer(stabilizer=stabilizer,
                                concentration=row["Stabilizer_conc"], units="M")
            form.set_salt(
                salt=salt, concentration=row["Salt_conc"], units="mg/mL")
            form.set_surfactant(surfactant=surfactant,
                                concentration=row["Surfactant_conc"], units="%w")
            form.set_temperature(temp=row["Temperature"])
            form.set_viscosity_profile(profile=vp)
            saved = self.add_formulation(form)
            added_forms.append(saved)
        return added_forms

    def get_all_as_dataframe(self) -> pd.DataFrame:
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
                "Sugar_type":      f.stabilizer.ingredient.enc_id,
                "Sugar_conc":      f.stabilizer.concentration,
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
            "Sugar_type", "Sugar_conc",
            "Surfactant_type", "Surfactant_conc",
            "Viscosity_100", "Viscosity_1000", "Viscosity_10000",
            "Viscosity_100000", "Viscosity_15000000"
        ]
        for col in expected:
            if col not in df.columns:
                df[col] = pd.NA

        return df[expected]
