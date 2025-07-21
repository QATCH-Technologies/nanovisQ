try:
    from src.controller.ingredient_controller import IngredientController
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController


class ListUtils:

    @staticmethod
    def load_all_excipient_types(ing_ctrl: IngredientController):
        proteins: list[str] = []
        buffers: list[str] = []
        surfactants: list[str] = []
        stabilizers: list[str] = []
        salts: list[str] = []
        ingredients = ing_ctrl.get_all_ingredients()

        for i in ingredients:
            if i.name.casefold() == "none":
                continue  # skip "none"
            if i.type == "Protein" and i.is_user:
                proteins.append(i.name)  # only show user proteins, hide core
            elif i.type == "Buffer":
                buffers.append(i.name)
            elif i.type == "Surfactant":
                surfactants.append(i.name)
            elif i.type == "Stabilizer":
                stabilizers.append(i.name)
            elif i.type == "Salt":
                salts.append(i.name)

        # use unique, case-insensitive sorting method:
        proteins = ListUtils.unique_case_insensitive_sort(proteins)
        buffers = ListUtils.unique_case_insensitive_sort(buffers)
        surfactants = ListUtils.unique_case_insensitive_sort(
            surfactants)
        stabilizers = ListUtils.unique_case_insensitive_sort(
            stabilizers)
        salts = ListUtils.unique_case_insensitive_sort(salts)

        return proteins, buffers, surfactants, stabilizers, salts

    @staticmethod
    def unique_case_insensitive_sort(list):
        """
        Returns a sorted list with unique items, ignoring case.
        """
        seen = set()
        result = []
        for item in list:
            lower_item = item.lower()
            if lower_item not in seen:
                seen.add(lower_item)
                result.append(item)

        # Sort case-insensitive
        result.sort(key=str.lower)
        return result
