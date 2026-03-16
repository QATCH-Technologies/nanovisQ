"""
Style Loader Utility for Qt Applications

This module provides a centralized way to load and apply QSS stylesheets
with dynamic icon path substitution.

Usage:
    from styles.style_loader import StyleLoader

    # Initialize the loader
    loader = StyleLoader()

    # Apply to your application
    app.setStyleSheet(loader.get_stylesheet())

    # Or apply to a specific widget
    my_widget.setStyleSheet(loader.get_stylesheet())
"""

import os
from ast import Import
from pathlib import Path

try:
    from architecture import Architecture
except (ImportError, ModuleNotFoundError):
    from QATCH.common.architecture import Architecture


class StyleLoader:
    """Loads and manages QSS stylesheets with icon path substitution.

    This class handles reading QSS files from disk, caching their content,
    and substituting specific markers with absolute file system paths. This
    is particularly useful for ensuring icons load correctly across different
    installation environments.

    Attributes:
        DEFAULT_ICONS (dict): Default mapping of icon placeholders to their
            relative paths within the project architecture.
        base_dir (Path): The root directory used to resolve relative icon paths.
        theme_file (str): The filename of the QSS theme to load.
        icon_paths (dict): The active mapping of placeholders to file paths.
        _stylesheet_cache (str, optional): Cached version of the processed
            stylesheet to prevent redundant disk I/O.
    """

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

    def __init__(self, base_dir=None, theme_file="theme.qss", icon_paths=None):
        """Initializes the StyleLoader with path configurations.

        Args:
            base_dir (str or Path, optional): Base directory for resolving relative
                paths. If None, defaults to the parent of the current file's directory.
            theme_file (str): Name of the QSS file. Defaults to 'theme.qss'.
            icon_paths (dict, optional): Dictionary mapping placeholders to paths.
                If None, uses `DEFAULT_ICONS`.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.theme_file = theme_file
        self.icon_paths = icon_paths or self.DEFAULT_ICONS.copy()
        self._stylesheet_cache = None

    def _resolve_icon_path(self, relative_path):
        """Resolves an icon path to an absolute path formatted for Qt.

        Qt requires forward slashes for file paths within QSS, even on Windows
        operating systems.

        Args:
            relative_path (str): Path relative to the `base_dir`.

        Returns:
            str: An absolute path string using forward slashes.
        """
        abs_path = (self.base_dir / relative_path).resolve()
        return str(abs_path).replace("\\", "/")

    def _load_qss_file(self):
        """Loads the raw QSS file content from the filesystem.

        Returns:
            str: The raw content of the QSS file.

        Raises:
            FileNotFoundError: If the theme file cannot be found at the
                calculated path.
        """
        qss_path = self.base_dir / "styles" / self.theme_file

        if not qss_path.exists():
            raise FileNotFoundError(
                f"QSS file not found: {qss_path}\n"
                f"Expected location: {qss_path.absolute()}"
            )

        with open(qss_path, "r", encoding="utf-8") as f:
            return f.read()

    def _substitute_icons(self, qss_content):
        """Replaces icon placeholders with absolute file paths.

        Supports both the new `{{PLACEHOLDER}}` and legacy `__PLACEHOLDER__`
        formats.

        Args:
            qss_content (str): Raw QSS content containing placeholder markers.

        Returns:
            str: Processed QSS content with absolute file paths.
        """
        for placeholder, relative_path in self.icon_paths.items():
            abs_path = self._resolve_icon_path(relative_path)
            qss_content = qss_content.replace(f"{{{{{placeholder}}}}}", abs_path)
            qss_content = qss_content.replace(f"__{placeholder}__", abs_path)

        return qss_content

    def get_stylesheet(self, use_cache=True):
        """Retrieves the processed stylesheet with substituted paths.

        Args:
            use_cache (bool): If True, returns the previously processed
                stylesheet if available. Defaults to True.

        Returns:
            str: The final QSS stylesheet ready for use in `setStyleSheet()`.
        """
        if use_cache and self._stylesheet_cache is not None:
            return self._stylesheet_cache

        qss_content = self._load_qss_file()
        qss_content = self._substitute_icons(qss_content)

        if use_cache:
            self._stylesheet_cache = qss_content

        return qss_content

    def reload(self):
        """Forces a reload of the stylesheet from disk.

        Useful during development to see QSS changes without restarting the app.

        Returns:
            str: The freshly loaded and processed QSS stylesheet.
        """
        self._stylesheet_cache = None
        return self.get_stylesheet(use_cache=False)

    def set_icon_path(self, placeholder, path):
        """Updates or adds a new icon path mapping.

        Calling this method automatically invalidates the stylesheet cache.

        Args:
            placeholder (str): The placeholder name used in the QSS (e.g., 'ICON_LOGO').
            path (str): The file path relative to `base_dir`.
        """
        self.icon_paths[placeholder] = path
        self._stylesheet_cache = None

    @classmethod
    def create_default(cls):
        """Creates a StyleLoader instance with default settings.

        Returns:
            StyleLoader: A new instance initialized with default icons and theme.
        """
        return cls()


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
