class ListUtils:

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
