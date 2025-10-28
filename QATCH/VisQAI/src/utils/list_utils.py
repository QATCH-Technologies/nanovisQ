try:
    from src.controller.ingredient_controller import IngredientController
    from src.models.ingredient import ProteinClass
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.ingredient import ProteinClass


class ListUtils:

    @staticmethod
    def load_all_ingredient_types(ing_ctrl: IngredientController):
        proteins: list[str] = []
        buffers: list[str] = []
        surfactants: list[str] = []
        stabilizers: list[str] = []
        salts: list[str] = []
        excipients: list[str] = []
        class_types: list[str] = []
        proteins_by_class: dict[str, str] = {}

        # fixed list of supported protein class types:
        class_types = list(ProteinClass.all_strings())

        for type in class_types:
            proteins_by_class[type] = []

        ingredients = ing_ctrl.get_all_ingredients()

        for i in ingredients:
            if i.name.casefold() == "none":
                continue  # skip "none"
            if i.type == "Protein" and i.is_user:
                proteins.append(i.name)  # only show user proteins, hide core
                # NOTE: If multiple proteins of the same name have conflicting
                # class types, this lookup table will contain the same protein
                # name under the keys of all matching class types, not just 1.
                # This could lead to unexpected behaviors if a protein ends up
                # in both the allowed and the not allowed class type category.
                # TODO: Add filter to restrict protein names to a single type!
                # NOTE: `i.class_type` will be ProteinClass or None; never str
                if not isinstance(i.class_type, ProteinClass):
                    class_type = "None"  # must be None
                elif i.class_type.value not in class_types:
                    # Something is off; value is not in `all_strings`
                    class_type = "Other"  # mark unknown as Other
                else:  # class_type must be a ProteinClass object
                    class_type = i.class_type.value
                if i.name not in proteins_by_class[class_type]:
                    proteins_by_class[class_type].append(i.name)
            elif i.type == "Buffer":
                buffers.append(i.name)
            elif i.type == "Surfactant":
                surfactants.append(i.name)
            elif i.type == "Stabilizer":
                stabilizers.append(i.name)
            elif i.type == "Salt":
                salts.append(i.name)
            elif i.type == "Excipient":
                excipients.append(i.name)

        # use unique, case-insensitive sorting method:
        proteins = ListUtils.unique_case_insensitive_sort(proteins)
        buffers = ListUtils.unique_case_insensitive_sort(buffers)
        surfactants = ListUtils.unique_case_insensitive_sort(
            surfactants)
        stabilizers = ListUtils.unique_case_insensitive_sort(
            stabilizers)
        salts = ListUtils.unique_case_insensitive_sort(salts)
        excipients = ListUtils.unique_case_insensitive_sort(excipients)

        return proteins, buffers, surfactants, stabilizers, salts, excipients, class_types, proteins_by_class

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
