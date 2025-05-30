
from typing import List

try:
    from src.db.db import Database
    from src.controller.ingredient_controller import IngredientController
    from src.models.formulation import Formulation
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.formulation import Formulation


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
        formulations = self.get_all_formulations()
        for f in formulations:
            if f == formulation:
                return f
        try:
            buffer = formulation.buffer.ingredient
            self.ingredient_controller.add(buffer)
            protein = formulation.protein.ingredient
            self.ingredient_controller.add(protein)
            salt = formulation.salt.ingredient
            self.ingredient_controller.add(salt)
            surfactant = formulation.surfactant.ingredient
            self.ingredient_controller.add(surfactant)
            stabilizer = formulation.stabilizer.ingredient
            self.ingredient_controller.add(stabilizer)
        except ValueError:
            pass

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
