"""
Data Processing and Ingredient Categorization Utilities.

This module provides static utility methods for managing lists of ingredients
and protein classifications. It acts as a bridge between the raw data returned
by database controllers and the filtered, sorted formats required by
formulation design interfaces and the search space constraints.

Key features include:
- Type-safe ingredient categorization.
- Case-insensitive deduplication and sorting.
- Protein classification mapping.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.1
"""

try:
    from src.controller.ingredient_controller import IngredientController
    from src.models.ingredient import ProteinClass
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.models.ingredient import ProteinClass


class ListUtils:
    """Utilities for processing and sorting ingredient data."""

    @staticmethod
    def load_all_ingredient_types(ing_ctrl: IngredientController):
        """Retrieves and categorizes all ingredients from the database.

        This method fetches the entire ingredient library and organizes it into
        specific categories (Proteins, Buffers, Salts, etc.). It applies
        business logic to filter out system-level ingredients, ensuring only
        user-defined proteins are exposed.

        It also performs a nested categorization of proteins based on their
        `ProteinClass`, which is used to drive conditional
        logic in the sampler and optimizer search spaces.

        Args:
            ing_ctrl: An instance of IngredientController used to interface
                with the backend database.

        Returns:
            Tuple: A complex tuple containing:
                - proteins (List[str]): Unique, sorted names of user proteins.
                - buffers (List[str]): Unique, sorted names of buffer types.
                - surfactants (List[str]): Unique, sorted names of surfactants.
                - stabilizers (List[str]): Unique, sorted names of stabilizers.
                - salts (List[str]): Unique, sorted names of salts.
                - excipients (List[str]): Unique, sorted names of excipients.
                - class_types (List[str]): Supported protein class strings.
                - proteins_by_class (Dict[str, List[str]]): A mapping of
                  protein class names to the protein names belonging to them.

        Note:
            Placeholder ingredients named "None" (case-insensitive) are
            automatically excluded from all returned lists.
        """
        proteins: list[str] = []
        buffers: list[str] = []
        surfactants: list[str] = []
        stabilizers: list[str] = []
        salts: list[str] = []
        excipients: list[str] = []
        class_types: list[str] = []
        proteins_by_class: dict[str, list[str]] = {}

        # fixed list of supported protein class types:
        class_types = list(ProteinClass.all_strings())

        for class_name in class_types:
            proteins_by_class[class_name] = []

        ingredients = ing_ctrl.get_all_ingredients()

        for i in ingredients:
            if i is None or i.name.casefold() == "none":
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
        surfactants = ListUtils.unique_case_insensitive_sort(surfactants)
        stabilizers = ListUtils.unique_case_insensitive_sort(stabilizers)
        salts = ListUtils.unique_case_insensitive_sort(salts)
        excipients = ListUtils.unique_case_insensitive_sort(excipients)

        return (
            proteins,
            buffers,
            surfactants,
            stabilizers,
            salts,
            excipients,
            class_types,
            proteins_by_class,
        )

    @staticmethod
    def unique_case_insensitive_sort(list):
        """Returns a deduplicated and sorted list, ignoring character case.

        This is a critical utility for ensuring deterministic behavior in the
        Optimizer and Sampler. By enforcing a case-insensitive sort, it
        guarantees that categorical indices (e.g., 'Arg' vs 'arg') remain
        consistent regardless of how they were entered into the database.

        Args:
            list_in: The raw list of strings to be processed.

        Returns:
            list: A new list containing unique items, sorted alphabetically
                in a case-insensitive manner.

        Example:
            >>> ListUtils.unique_case_insensitive_sort(["citrate", "Acetate", "citrate"])
            ['Acetate', 'citrate']
        """
        seen = set()
        result = []
        for item in list:
            lower_item = item.casefold()
            if lower_item not in seen:
                seen.add(lower_item)
                result.append(item)

        # Sort case-insensitive
        result.sort(key=str.casefold)
        return result
