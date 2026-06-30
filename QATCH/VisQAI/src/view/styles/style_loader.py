"""Style Loader Utility for Qt Applications (VisQAI).

This module is now a thin VisQAI-specific shim over the shared
`QATCH.ui.styles.style_loader.StyleLoader`, which generalized this class's
icon-path placeholder substitution to also support color-token substitution
for app-wide light/dark theming. This module keeps its prior defaults (icon
paths, "theme.qss" filename, "styles" subdirectory) so nothing importing
from here needs to change.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os
from pathlib import Path

try:
    from architecture import Architecture
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture

from QATCH.ui.styles.style_loader import StyleLoader as _BaseStyleLoader

DEFAULT_ICONS = {
    "ICON_DOWN": os.path.join(
        Architecture.get_path(),
        "QATCH",
        "VisQAI",
        "src",
        "view",
        "icons",
        "down-chevron-svgrepo-com.svg",
    ),
    "ICON_UP": os.path.join(
        Architecture.get_path(),
        "QATCH",
        "VisQAI",
        "src",
        "view",
        "icons",
        "up-chevron-svgrepo-com.svg",
    ),
    "ICON_BROWSE_MODEL": os.path.join(
        Architecture.get_path(),
        "QATCH",
        "VisQAI",
        "src",
        "view",
        "icons",
        "file-plus-2-svgrepo-com.svg",
    ),
}


class StyleLoader(_BaseStyleLoader):
    """VisQAI-specific StyleLoader: same API as the shared base, with this
    module's icon paths and "theme.qss" as the defaults.

    Attributes:
        DEFAULT_ICONS (dict): Default mapping of icon placeholders to their
            relative paths within the project architecture.
    """

    DEFAULT_ICONS = DEFAULT_ICONS

    def __init__(self, base_dir=None, theme_file="theme.qss", icon_paths=None):
        """Initializes the StyleLoader with path configurations.

        Args:
            base_dir (str or Path, optional): Base directory for resolving
                relative icon paths. If None, defaults to the parent of this
                module's parent directory (`.../VisQAI/src/view`).
            theme_file (str): Name of the QSS file. Defaults to 'theme.qss'.
            icon_paths (dict, optional): Dictionary mapping placeholders to
                paths. If None, uses `DEFAULT_ICONS`.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        super().__init__(
            base_dir=base_dir,
            theme_file=theme_file,
            icon_paths=icon_paths,
            styles_subdir="styles",
        )


def load_stylesheet(base_dir=None, icon_paths=None):
    """Convenience function to quickly load the default stylesheet.

    Args:
        base_dir (str or Path, optional): Base directory for path resolution.
        icon_paths (dict, optional): Custom icon placeholder mappings.

    Returns:
        str: The complete processed QSS stylesheet.
    """
    loader = StyleLoader(base_dir=base_dir, icon_paths=icon_paths)
    return loader.get_stylesheet()
