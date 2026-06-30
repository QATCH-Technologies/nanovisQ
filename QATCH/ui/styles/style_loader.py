"""
QATCH.ui.styles.style_loader

Shared utility to load and apply QSS (Qt Style Sheets) with placeholder
substitution - both for icon paths (so QSS never embeds an absolute,
install-specific path) and, now, for theme color tokens (so a QSS file can
say `{{ACCENT}}` instead of a hardcoded `rgba(10, 163, 230, 255)`).

This is a generalized promotion of the original
QATCH.VisQAI.src.view.styles.style_loader.StyleLoader, which now subclasses
this one to keep its existing VisQAI-specific defaults (icon paths, default
theme filename) without duplicating the substitution logic.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from pathlib import Path


class StyleLoader:
    """Loads and manages QSS stylesheets with icon-path and color-token
    placeholder substitution.

    Attributes:
        DEFAULT_ICONS (dict): Default mapping of icon placeholders to their
            relative paths. Empty by default - subclasses (e.g. VisQAI's
            StyleLoader) provide their own.
        base_dir (Path): The root directory used to resolve relative icon
            paths and to locate the QSS file itself.
        theme_file (str): The filename of the QSS file to load.
        styles_subdir (str): Subdirectory under `base_dir` containing
            `theme_file` (e.g. "styles" for VisQAI's layout, "" to load
            directly from `base_dir`).
        icon_paths (dict): The active mapping of icon placeholders to paths.
        tokens (dict): The active mapping of color-token placeholders (e.g.
            "ACCENT") to literal CSS color strings (e.g. "rgba(10, 163, 230, 255)").
        _stylesheet_cache (str, optional): Cached processed stylesheet.
    """

    DEFAULT_ICONS: dict = {}

    def __init__(
        self,
        base_dir=None,
        theme_file="theme.qss",
        icon_paths=None,
        tokens=None,
        styles_subdir="styles",
    ):
        """Initializes the StyleLoader with path configurations.

        Args:
            base_dir (str or Path, optional): Base directory for resolving
                relative icon paths and locating the QSS file. If None,
                defaults to the parent of the current file's directory.
            theme_file (str): Name of the QSS file. Defaults to 'theme.qss'.
            icon_paths (dict, optional): Mapping of icon placeholders to
                paths. If None, uses `DEFAULT_ICONS`.
            tokens (dict, optional): Mapping of color-token placeholders to
                literal CSS color strings. Empty by default - callers set
                this via the constructor or `set_tokens()`.
            styles_subdir (str): Subdirectory under `base_dir` containing
                `theme_file`. Pass "" to load `theme_file` directly from
                `base_dir` with no subdirectory.
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.theme_file = theme_file
        self.styles_subdir = styles_subdir
        self.icon_paths = icon_paths if icon_paths is not None else self.DEFAULT_ICONS.copy()
        self.tokens = tokens or {}
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
        qss_path = (
            self.base_dir / self.styles_subdir / self.theme_file
            if self.styles_subdir
            else self.base_dir / self.theme_file
        )

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

    def _substitute_tokens(self, qss_content):
        """Replaces color-token placeholders with their literal CSS values.

        Same `{{PLACEHOLDER}}` / `__PLACEHOLDER__` dual format as icons.

        Args:
            qss_content (str): QSS content containing placeholder markers.

        Returns:
            str: Processed QSS content with literal color values.
        """
        for placeholder, value in self.tokens.items():
            qss_content = qss_content.replace(f"{{{{{placeholder}}}}}", value)
            qss_content = qss_content.replace(f"__{placeholder}__", value)

        return qss_content

    def get_stylesheet(self, use_cache=True):
        """Retrieves the processed stylesheet with substituted paths/tokens.

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
        qss_content = self._substitute_tokens(qss_content)

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

    def set_tokens(self, tokens):
        """Replaces the active color-token mapping.

        Calling this method automatically invalidates the stylesheet cache.

        Args:
            tokens (dict): Mapping of color-token placeholders (e.g.
                "ACCENT") to literal CSS color strings.
        """
        self.tokens = tokens
        self._stylesheet_cache = None

    @classmethod
    def create_default(cls):
        """Creates a StyleLoader instance with default settings.

        Returns:
            StyleLoader: A new instance initialized with default icons and theme.
        """
        return cls()


def load_stylesheet(base_dir=None, icon_paths=None, tokens=None):
    """Convenience function to quickly load the default stylesheet.

    Args:
        base_dir (str or Path, optional): Base directory for path resolution.
        icon_paths (dict, optional): Custom icon placeholder mappings.
        tokens (dict, optional): Custom color-token placeholder mappings.

    Returns:
        str: The complete processed QSS stylesheet.
    """
    loader = StyleLoader(base_dir=base_dir, icon_paths=icon_paths, tokens=tokens)
    return loader.get_stylesheet()
