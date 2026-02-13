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

from pathlib import Path


class StyleLoader:
    """
    Loads and manages QSS stylesheets with icon path substitution.
    """

    # Default icon paths (relative to the base directory)
    DEFAULT_ICONS = {
        "ICON_DOWN": "icons/down-chevron-svgrepo-com.svg",
        "ICON_UP": "icons/up-chevron-svgrepo-com.svg",
        "ICON_BROWSE_MODEL": "icons/machine-learning-01-svgrepo-com.svg",
    }

    def __init__(self, base_dir=None, theme_file="theme.qss", icon_paths=None):
        """
        Initialize the style loader.

        Args:
            base_dir: Base directory for resolving relative paths.
                     If None, uses the directory of this file.
            theme_file: Name of the QSS file (default: 'theme.qss')
            icon_paths: Dictionary of icon placeholder -> path mappings.
                       If None, uses DEFAULT_ICONS.
        """
        if base_dir is None:
            # Get the directory where this file is located
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.theme_file = theme_file
        self.icon_paths = icon_paths or self.DEFAULT_ICONS.copy()
        self._stylesheet_cache = None

    def _resolve_icon_path(self, relative_path):
        """
        Resolve an icon path to an absolute path with forward slashes.

        Args:
            relative_path: Path relative to base_dir

        Returns:
            Absolute path string with forward slashes
        """
        abs_path = (self.base_dir / relative_path).resolve()
        # Qt requires forward slashes even on Windows
        return str(abs_path).replace("\\", "/")

    def _load_qss_file(self):
        """
        Load the QSS file from disk.

        Returns:
            Raw QSS content as string
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
        """
        Replace icon placeholders with actual file paths.

        Args:
            qss_content: Raw QSS content with {{PLACEHOLDER}} markers

        Returns:
            QSS content with substituted paths
        """
        for placeholder, relative_path in self.icon_paths.items():
            abs_path = self._resolve_icon_path(relative_path)
            # Replace both {{PLACEHOLDER}} and old __PLACEHOLDER__ formats
            qss_content = qss_content.replace(f"{{{{{placeholder}}}}}", abs_path)
            qss_content = qss_content.replace(f"__{placeholder}__", abs_path)

        return qss_content

    def get_stylesheet(self, use_cache=True):
        """
        Get the complete stylesheet with icon paths substituted.

        Args:
            use_cache: If True, uses cached stylesheet after first load

        Returns:
            Complete QSS stylesheet as string
        """
        if use_cache and self._stylesheet_cache is not None:
            return self._stylesheet_cache

        qss_content = self._load_qss_file()
        qss_content = self._substitute_icons(qss_content)

        if use_cache:
            self._stylesheet_cache = qss_content

        return qss_content

    def reload(self):
        """
        Force reload of the stylesheet from disk.
        Useful during development.
        """
        self._stylesheet_cache = None
        return self.get_stylesheet(use_cache=False)

    def set_icon_path(self, placeholder, path):
        """
        Update or add an icon path mapping.

        Args:
            placeholder: The placeholder name (e.g., 'ICON_DOWN')
            path: Path relative to base_dir
        """
        self.icon_paths[placeholder] = path
        self._stylesheet_cache = None  # Invalidate cache

    @classmethod
    def create_default(cls):
        """
        Create a StyleLoader with default settings.
        Convenience method for typical usage.
        """
        return cls()


# Convenience function for quick usage
def load_stylesheet(base_dir=None, icon_paths=None):
    """
    Quick function to load the default stylesheet.

    Args:
        base_dir: Base directory for resolving paths
        icon_paths: Custom icon path mappings

    Returns:
        Complete QSS stylesheet string
    """
    loader = StyleLoader(base_dir=base_dir, icon_paths=icon_paths)
    return loader.get_stylesheet()
