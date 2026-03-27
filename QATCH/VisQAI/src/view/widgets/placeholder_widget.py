"""
placeholder_widget.py

Empty-state indicator widget for the VisQAI prediction UI.

Displays a tinted icon and a short instructional message whenever a panel has
no data to present.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-03-16

Version:
    1.0
"""

import os

from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from architecture import Architecture
except (ModuleNotFoundError, ImportError):
    from QATCH.common.architecture import Architecture


class PlaceholderWidget(QtWidgets.QWidget):
    """A centred empty-state widget composed of a tinted icon and a message label.

    Renders a single SVG icon scaled to 64 x 64 px with a cyan tint applied at
    construction time, followed by a word-wrapped message label.  Both elements
    can be replaced at runtime via ``set_icon`` and ``set_message`` without
    rebuilding the widget.

    Styling is delegated entirely to ``theme.qss`` via the object-name
    selectors ``#placeholderWidget``, ``#placeholderIcon``, and
    ``#placeholderText``, so visual changes require no Python edits.

    Attributes:
        lbl_icon (QtWidgets.QLabel): Label that holds the scaled, tinted icon
            pixmap.  Object name ``"placeholderIcon"``.
        lbl_text (QtWidgets.QLabel): Centre-aligned, word-wrapping label that
            displays the instructional message.  Object name
            ``"placeholderText"``.
    """

    def __init__(self, parent=None):
        """Construct the placeholder, load the default icon, and build the layout.

        Sets up a vertically centred ``QVBoxLayout`` containing ``lbl_icon``
        and ``lbl_text``.  The default icon is ``info-circle-svgrepo-com.svg``
        resolved through ``Architecture.get_path()``.  If the pixmap loads
        successfully it is tinted via ``_apply_icon_tint`` and scaled to
        64 x 64 px with smooth transformation; a null pixmap is silently
        skipped so the widget still renders with the message alone.

        Args:
            parent (QtWidgets.QWidget | None): Optional Qt parent widget.
                Defaults to ``None``.
        """
        super().__init__(parent)
        self.setObjectName("placeholderWidget")

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # Icon
        self.lbl_icon = QtWidgets.QLabel()
        self.lbl_icon.setObjectName("placeholderIcon")
        self.lbl_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pixmap = QtGui.QPixmap(
            os.path.join(
                Architecture.get_path(),
                "QATCH",
                "VisQAI",
                "src",
                "view",
                "icons",
                "info-circle-svgrepo-com.svg",
            )
        )

        if not pixmap.isNull():
            self._apply_icon_tint(pixmap)
            self.lbl_icon.setPixmap(
                pixmap.scaled(
                    64,
                    64,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

        layout.addWidget(self.lbl_icon)
        self.lbl_text = QtWidgets.QLabel(
            "No data yet.\nClick the + button to add new data."
        )
        self.lbl_text.setObjectName("placeholderText")
        self.lbl_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_text.setWordWrap(True)

        layout.addWidget(self.lbl_text)

    def _apply_icon_tint(self, pixmap):
        """Paint a flat cyan overlay onto *pixmap* in-place using ``SourceIn`` blending.

        ``CompositionMode_SourceIn`` replaces every non-transparent pixel of
        the destination with the solid fill colour while preserving the source
        alpha channel.  The result is that the SVG silhouette retains its
        shape and transparency but adopts the theme's mid-cyan colour
        (``#4EC4EB``).

        Args:
            pixmap (QtGui.QPixmap): The pixmap to tint.  Modified in-place;
                must not be null.
        """
        painter = QtGui.QPainter(pixmap)
        painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_SourceIn
        )
        painter.fillRect(pixmap.rect(), QtGui.QColor("#4EC4EB"))  # Cyan medium
        painter.end()

    def set_message(self, message):
        """Replace the instructional text shown below the icon.

        Args:
            message (str): New plain-text string to display.  The label has
                word-wrap enabled, so long strings reflow automatically.
        """
        self.lbl_text.setText(message)

    def set_icon(self, icon_path, size=64):
        """Replace the placeholder icon with a new image file.

        Loads the file at *icon_path* into a ``QPixmap``, applies the cyan
        tint via ``_apply_icon_tint``, then scales the result to *size* x
        *size* px using ``KeepAspectRatio`` and smooth transformation before
        updating ``lbl_icon``.  If the pixmap is null (file not found or
        unsupported format) the existing icon is left unchanged.

        Args:
            icon_path (str): Absolute or relative path to the icon file.
                SVG and all raster formats supported by Qt are accepted.
            size (int): Target width and height in pixels.  Defaults to ``64``.
        """
        pixmap = QtGui.QPixmap(icon_path)
        if not pixmap.isNull():
            self._apply_icon_tint(pixmap)
            self.lbl_icon.setPixmap(
                pixmap.scaled(
                    size,
                    size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )
