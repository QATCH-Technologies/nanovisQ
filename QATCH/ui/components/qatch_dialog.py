"""
QATCH.ui.components.qatch_dialog.py

Provides the application's shared flat-styled modal dialog components.

This module replaces the use of `QMessageBox` throughout the application
with themed dialogs that match the QATCH flat control system. Dialogs render
a semi-transparent backdrop over the application's root window and present
their content in a centered `DialogCard` with theme-aware styling.

The module provides:

* `DialogCard`: Shared flat-surface container used by modal dialogs.
* `DialogBase`: Base dialog providing frameless window setup, application-
  wide backdrop rendering, root-window resolution, and Escape handling.
* `QATCHDialog`: Standard message dialog supporting titles, semantic icons,
  body text, expandable details, and configurable `QATCHPushButton` actions.
* `tinted_icon`: Utility for applying a theme color to an SVG or other
  rasterized Qt icon.

Dialog appearance is resolved from `ThemeManager` tokens at runtime so
dialogs automatically follow light/dark theme changes without requiring
separate style definitions for each theme.

Typical usage::

    dialog = QATCHDialog(
        parent,
        "Confirm Delete",
        "Are you sure you want to delete this run?",
        buttons=[
            ("Cancel", "secondary", 0),
            ("Delete", "destructive", 1),
        ],
        icon_type="warning",
    )

    if dialog.exec_() and dialog.result_value() == 1:
        delete_run()

Semantic dialog icons are loaded from `QATCH/icons` and tinted using the
active theme's semantic color tokens. If an expected SVG icon is unavailable,
`QATCHDialog` falls back to a programmatically rendered circular glyph.

Attributes:
    ButtonSpec: Type alias describing a dialog action as
        `(label, variant, return_value)`.
    _ICONS_DIR: Absolute path to the application's shared icon directory.
    _BADGE_TOKENS: Mapping from dialog semantic type to its theme color token.
    _ICON_FILES: Mapping from dialog semantic type to its SVG icon filename.
    _CARD_W: Default width of the dialog card in pixels.
    _CARD_RADIUS: Default corner radius of the dialog card.
    _HEADER_H: Fixed height of the dialog header.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-05
"""

from __future__ import annotations

import os

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.common.architecture import Architecture
from QATCH.ui.components.flat_paint import paint_flat_surface
from QATCH.ui.components.qatch_push_button import QATCHPushButton
from QATCH.ui.components.window_utils import find_app_window
from QATCH.ui.styles.theme_manager import (
    ThemeManager,
    dialog_message_qss,
    dialog_title_qss,
)

# (button_label, QATCHPushButton_variant, return_value)
ButtonSpec = tuple[str, str, int]

_ICONS_DIR = os.path.join(Architecture.get_path(), "QATCH", "icons")

# Which palette token colors the icon badge for each dialog type.
_BADGE_TOKENS = {
    "information": "accent",
    "question": "accent",
    "warning": "warning",
    "critical": "danger",
}

_ICON_FILES = {
    "information": "info-circle.svg",
    "question": "question-circle.svg",
    "warning": "warning-circle.svg",
    "critical": "critical-circle.svg",
}

_CARD_W = 440
_CARD_RADIUS = 18.0
_HEADER_H = 52


def tinted_icon(path: str, color: QtGui.QColor, size: int = 22) -> QtGui.QPixmap:
    """Creates a theme-tinted pixmap from an icon file.

    Loads the specified icon through Qt's `QIcon` pipeline, rasterizes it
    at the requested size, and applies the supplied color using Qt's
    `SourceAtop` composition mode. The resulting pixmap preserves the
    source icon's transparency while replacing its visible pixels with the
    requested color.

    Args:
        path: Filesystem path to the source icon.
        color: Color to apply to the visible icon pixels.
        size: Target width and height of the rasterized icon in pixels.
            Defaults to 22.

    Returns:
        A transparent `QPixmap` containing the tinted icon.

    Notes:
        The function is primarily used by `QATCHDialog` for semantic
        dialog icons whose colors are resolved from the active
        `ThemeManager` tokens.
    """
    src = QtGui.QIcon(path).pixmap(size, size)
    dst = QtGui.QPixmap(src.size())
    dst.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(dst)
    p.drawPixmap(0, 0, src)
    p.setCompositionMode(QtGui.QPainter.CompositionMode_SourceAtop)
    p.fillRect(dst.rect(), color)
    p.end()
    return dst


class DialogCard(QtWidgets.QFrame):
    """Flat card container used as the shared surface for modal dialogs.

    Provides a theme-aware card surface for dialogs built on `DialogBase`,
    including `QATCHDialog` and `SignatureDialog`. The card's visual
    chrome is rendered through the shared flat-paint helper so its surface,
    border, and geometry remain consistent with other flat controls such as
    `QATCHLineEdit` and `QATCHPanel`.

    The card disables Qt's automatic background filling so the shared custom
    painting logic remains responsible for rendering the surface. An
    optional horizontal header separator can also be positioned within the
    card.

    Attributes:
        _radius (float): Corner radius used when painting the card surface.
        _header_line_y (float | None): Vertical position of the optional
            header separator line. `None` disables the separator.

    Args:
        parent: Parent widget that owns the dialog card.
        radius: Corner radius used for the card surface. Defaults to
            `_CARD_RADIUS`.
        header_line_y: Vertical position of the header separator. Set to
            `None` to disable the header line. Defaults to `_HEADER_H`.

    Example:
        Create a standard dialog card::

            card = DialogCard(parent)

        Create a card without a header separator::

            card = DialogCard(parent, header_line_y=None)
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        radius: float = _CARD_RADIUS,
        header_line_y: float | None = _HEADER_H,
    ) -> None:
        """Initialize the dialog card.

        Configures the card to use custom flat-surface painting rather than
        Qt's automatic background rendering.

        Args:
            parent: Parent widget that owns the card.
            radius: Corner radius used for the card surface.
            header_line_y: Vertical position of the optional header separator,
                or `None` to disable it.

        Returns:
            None.
        """
        super().__init__(parent)
        self._radius = radius
        self._header_line_y = header_line_y
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the dialog card surface and optional header separator.

        Resolves the card surface and border colors from the active theme and
        renders the rounded flat surface using :func:`paint_flat_surface`.
        When `_header_line_y` is configured, a one-pixel horizontal separator
        is drawn at the specified vertical position.

        Args:
            event: Qt paint event describing the region that needs to be
                repainted.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        paint_flat_surface(
            self,
            radius=self._radius,
            fill=QtGui.QColor(*tok["flat_surface"]),
            border=QtGui.QColor(*tok["flat_border"]),
            painter=p,
        )
        if self._header_line_y is not None:
            p.setPen(QtGui.QPen(QtGui.QColor(*tok["flat_border"]), 1.0))
            p.drawLine(0, int(self._header_line_y), self.width(), int(self._header_line_y))
        p.end()


class DialogBase(QtWidgets.QDialog):
    """Shared modal window chrome and behavior for QATCH application dialogs.

    Provides the common window configuration used by modal dialogs built on
    the application's `DialogCard` surface. The base class creates a
    frameless, translucent, application-modal `QDialog` and resolves the
    appropriate root window so the dialog overlay can cover the full
    application window.

    Subclasses are responsible for constructing their dialog-specific
    `DialogCard` content in their own `_build_ui` or `__init__`
    implementations. `DialogBase` supplies the shared modal behavior,
    including Escape-to-cancel handling and the translucent overlay used
    behind the dialog content.

    Subclasses:
        QATCHDialog: Dialog used for title/message/button-based modal
            interactions.
        SignatureDialog: Dialog providing a custom signature-capture form.

    Args:
        parent: Parent widget from which the application's root window is
            resolved. May be `None` when the dialog has no explicit parent.

    Attributes:
        The base class manages the dialog window configuration internally;
        subclasses provide the dialog-specific content and controls.
    """

    def __init__(self, parent: QtWidgets.QWidget | None) -> None:
        """Initialize the shared modal dialog configuration.

        Resolves the application's root window from `parent` before
        initializing the underlying `QDialog`. Configures the dialog as a
        frameless, translucent, application-modal window and prevents Qt from
        automatically deleting the dialog when it closes.

        Args:
            parent: Parent widget used to determine the root application
                window, or `None` if no parent is available.

        Returns:
            None.
        """
        root = self._find_root(parent)
        super().__init__(root)

        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)  # type: ignore[arg-type]
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, False)

    @staticmethod
    def _find_root(widget: QtWidgets.QWidget | None) -> QtWidgets.QWidget | None:
        """Resolve the root application window for the dialog overlay.

        Attempts to identify the application's true top-level window using
        :func:`find_app_window` before falling back to Qt's active window and
        finally the widget's parent hierarchy. This ordering is intentional:
        `QApplication.activeWindow()` can resolve to a smaller embedded
        `QMainWindow` during UI initialization or focus transitions, which
        would cause the modal dim overlay to cover only part of the
        application.

        Dialog windows are excluded from the application-window search so an
        existing `DialogBase` instance cannot accidentally be selected as
        the root.

        Args:
            widget: Widget associated with the dialog, used as a final fallback
                when no suitable application or active window can be found.

        Returns:
            The visible root application widget, or `None` if no suitable
            root can be resolved.
        """
        root = find_app_window(exclude_types=(DialogBase,))
        if root is not None:
            return root
        active = QtWidgets.QApplication.activeWindow()
        if active and active.isVisible():
            return active
        if not isinstance(widget, QtWidgets.QWidget):
            return None
        w = widget
        while w.parent() and isinstance(w.parent(), QtWidgets.QWidget):
            w = w.parent()
        return w

    def _on_escape(self) -> None:
        """Handle an Escape key press by rejecting the dialog.

        Provides the default Escape-key behavior for dialogs derived from
        :class:`DialogBase`. Subclasses that require Escape to trigger a
        specific action, such as a cancel or "No" button, should override
        this method.

        Returns:
            None.
        """
        self.reject()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Size the dialog to the current application root before showing.

        Re-resolves the root window at show time because the active application
        window may have changed since the dialog was constructed. When a valid
        root widget is found, the dialog geometry is matched to that window.
        Otherwise, the available desktop geometry is used as a fallback.

        Args:
            event: Qt show event generated when the dialog is about to become
                visible.

        Returns:
            None.
        """
        root = self._find_root(None)  # re-resolve at show time in case active window changed
        if isinstance(root, QtWidgets.QWidget):
            self.setGeometry(root.geometry())
        else:
            self.setGeometry(QtWidgets.QDesktopWidget().availableGeometry())
        super().showEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the modal backdrop dimming overlay.

        Resolves the backdrop color from the active theme and fills the entire
        dialog surface with the configured dim color. The overlay is rendered
        behind the dialog's card-based content to visually separate the modal
        interaction from the underlying application window.

        Args:
            event: Qt paint event describing the region that needs to be
                repainted.

        Returns:
            None.
        """
        tok = ThemeManager.instance().tokens()
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(*tok["backdrop_dim"]))
        p.end()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle keyboard input for the modal dialog.

        Intercepts the Escape key and routes it through the dialog's
        `_on_escape` handler. All other key events are delegated to the
        standard `QDialog` implementation.

        Args:
            event: Qt key event generated by keyboard input.

        Returns:
            None.
        """
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._on_escape()
            return
        super().keyPressEvent(event)


class QATCHDialog(DialogBase):
    """Styled card used as the visual container for modal dialogs.

    The card delegates its surface rendering to the shared
    `paint_flat_surface` helper so that modal dialogs use the same fill,
    border, radius, and theme-token styling as the application's other flat
    controls.

    A horizontal divider can optionally be rendered beneath the header area.
    This class is shared by dialogs derived from `DialogBase`, including
    `QATCHDialog` and `SignatureDialog`.

    Args:
        parent: Parent widget that owns the dialog card.
        radius: Corner radius used when painting the card surface.
        header_line_y: Vertical position of the optional header divider in
            widget coordinates. If `None`, no divider is drawn.

    Attributes:
        _radius: Corner radius used for the card surface.
        _header_line_y: Vertical coordinate of the optional header divider,
            or `None` when the divider is disabled.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        title: str,
        message: str,
        details: str = "",
        buttons: list[ButtonSpec] | None = None,
        icon_type: str = "information",
    ) -> None:
        """Initializes and constructs the standard modal dialog.

        Args:
            parent: Ancestor widget used to resolve the application root
                window.
            title: Dialog title displayed in the header.
            message: Main informational or warning message.
            details: Optional expandable details text.
            buttons: Optional action specifications in
                `(label, variant, return_value)` form. Defaults to a single
                `("OK", "primary", 1)` action.
            icon_type: Semantic icon category used to select the dialog icon.
        """
        super().__init__(parent)

        self._result_value: int = 0
        self._icon_type = icon_type

        if buttons is None:
            buttons = [("OK", "primary", 1)]

        self._build_ui(title, message, details, buttons)

        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def result_value(self) -> int:
        """Returns the value associated with the selected dialog action.

        Returns:
            The integer result value assigned to the button that most recently
            completed the dialog. Returns `0` when no action has been selected.
        """
        return self._result_value

    def _on_escape(self) -> None:
        """Handles Escape-key cancellation for the dialog.

        Escape is treated as the dialog's cancel action and routed through
        `_on_button()` with a result value of `0`, allowing cancellation to follow
        the same result-handling path as an explicitly configured cancel button.
        """
        self._on_button(0)

    def _build_ui(
        self,
        title: str,
        message: str,
        details: str,
        buttons: list[ButtonSpec],
    ) -> None:
        """Builds the dialog card and its contents.

        Constructs the complete modal layout, including the centered
        `DialogCard`, semantic header icon, title, message body, optional
        expandable details section, and configurable action buttons.

        The final button in `buttons` is treated as the primary/default action
        and receives keyboard focus when the dialog is displayed.

        Args:
            title: Text displayed in the dialog header.
            message: Main body text displayed beneath the header.
            details: Optional expandable details content. An empty string omits
                the details section entirely.
            buttons: Button specifications defining the available actions. Each
                entry contains a label, `QATCHPushButton` variant, and integer
                result value in the form `(label, variant, return_value)`.

        Notes:
            The card width, header height, spacing, margins, and button dimensions
            are intentionally centralized here so all standard dialogs share the
            same visual geometry.
        """
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

        card_row = QtWidgets.QHBoxLayout()
        card_row.setContentsMargins(0, 0, 0, 0)
        card_row.addStretch()

        self._card = DialogCard(self)
        self._card.setFixedWidth(_CARD_W)

        card_v = QtWidgets.QVBoxLayout(self._card)
        card_v.setContentsMargins(0, 0, 0, 0)
        card_v.setSpacing(0)

        # Header
        header_w = QtWidgets.QWidget()
        header_w.setFixedHeight(_HEADER_H)
        header_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        header_layout = QtWidgets.QHBoxLayout(header_w)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(10)

        self._icon_label = QtWidgets.QLabel()
        self._icon_label.setFixedSize(24, 24)
        self._icon_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._refresh_icon()
        header_layout.addWidget(self._icon_label)

        self._title_label = QtWidgets.QLabel(title)
        self._title_label.setObjectName("QATCHDialogTitle")
        self._title_label.setWordWrap(True)
        self._apply_title_style()
        header_layout.addWidget(self._title_label, 1)

        card_v.addWidget(header_w)

        # Body
        body_w = QtWidgets.QWidget()
        body_w.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        body_layout = QtWidgets.QVBoxLayout(body_w)
        body_layout.setContentsMargins(20, 16, 20, 0)
        body_layout.setSpacing(10)

        self._msg_label = QtWidgets.QLabel(message)
        self._msg_label.setObjectName("QATCHDialogMessage")
        self._msg_label.setWordWrap(True)
        self._msg_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop
        )
        self._apply_body_style()
        body_layout.addWidget(self._msg_label)

        if details:
            body_layout.addWidget(self._build_details(details))

        card_v.addWidget(body_w)

        # Buttons
        btn_row = QtWidgets.QWidget()
        btn_row.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        btn_layout = QtWidgets.QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(20, 16, 20, 20)
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        for i, (label, variant, val) in enumerate(buttons):
            btn = QATCHPushButton(label, variant=variant)
            btn.setFixedHeight(34)
            btn.setMinimumWidth(90)
            btn.clicked.connect(lambda checked=False, r=val: self._on_button(r))
            btn_layout.addWidget(btn)
            if i == len(buttons) - 1:
                btn.setDefault(True)
                btn.setFocus()

        card_v.addWidget(btn_row)

        card_row.addWidget(self._card)
        card_row.addStretch()
        outer.addLayout(card_row)
        outer.addStretch()

    def _build_details(self, details: str) -> QtWidgets.QWidget:
        """Builds the expandable details section for the dialog.

        Creates a compact toggle button and a read-only text area containing the
        supplied details. The text area is hidden initially and is shown or
        hidden when the toggle button is clicked. The parent dialog card is
        resized after each visibility change so the card expands and contracts
        with the details content.

        Args:
            details: Detailed diagnostic or supplemental text to display in the
                expandable text area.

        Returns:
            A QWidget containing the details toggle and expandable text area.

        Notes:
            The details text area uses theme tokens for its background, border,
            and text colors. The toggle is rendered as a borderless
            `QATCHPushButton` using the neutral button variant.
        """
        container = QtWidgets.QWidget()
        container.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        toggle_btn = QATCHPushButton("Show Details ▾", variant="neutral")
        toggle_btn.setFixedHeight(28)
        toggle_btn.set_border_visible(False)
        lay.addWidget(toggle_btn)

        text_area = QtWidgets.QPlainTextEdit(details)
        text_area.setReadOnly(True)
        text_area.setMaximumHeight(100)
        tok = ThemeManager.instance().tokens()
        fill = tok["ctrl_input_bg"]
        border = tok["ctrl_input_border"]
        txt = tok["plot_text_normal"]
        text_area.setStyleSheet(
            "QPlainTextEdit {"
            f"  background: rgba({fill[0]},{fill[1]},{fill[2]},{fill[3]});"
            f"  border: 1px solid rgba({border[0]},{border[1]},{border[2]},{border[3]});"
            "  border-radius: 6px;"
            f"  color: rgba({txt[0]},{txt[1]},{txt[2]},{txt[3]});"
            "  font-size: 11px;"
            "}"
        )
        text_area.hide()
        lay.addWidget(text_area)

        def _toggle() -> None:
            visible = text_area.isVisible()
            text_area.setVisible(not visible)
            toggle_btn.setText("Hide Details ▴" if not visible else "Show Details ▾")
            self._card.adjustSize()

        toggle_btn.clicked.connect(_toggle)
        return container

    def _apply_title_style(self) -> None:
        """Applies the current theme's styling to the dialog title.

        Resolves the title stylesheet through `dialog_title_qss()` so the title
        typography and colors remain consistent with the active application
        theme.
        """
        self._title_label.setStyleSheet(dialog_title_qss())

    def _apply_body_style(self) -> None:
        """Applies the current theme's styling to the dialog message body.

        Resolves the message stylesheet through `dialog_message_qss()` so the
        body text remains synchronized with the active application theme.
        """
        self._msg_label.setStyleSheet(dialog_message_qss())

    def _refresh_icon(self) -> None:
        """Refreshes the dialog icon using the active theme tokens.

        Resolves the icon's semantic color from the dialog's `icon_type` and
        loads the corresponding SVG from the application's icon directory.
        When the expected SVG is unavailable, a simple circular fallback glyph
        is generated programmatically using the resolved theme color.

        The resulting icon is assigned to the dialog's header icon label.

        Notes:
            This method is also called when the application theme changes so
            semantic dialog icons are recolored immediately without requiring
            the dialog to be reconstructed.
        """
        tok = ThemeManager.instance().tokens()
        badge_key = _BADGE_TOKENS.get(self._icon_type, "accent")
        color = QtGui.QColor(*tok[badge_key])
        svg = os.path.join(_ICONS_DIR, _ICON_FILES.get(self._icon_type, "info-circle.svg"))
        if os.path.isfile(svg):
            pm = tinted_icon(svg, color, size=22)
        else:
            pm = QtGui.QPixmap(22, 22)
            pm.fill(QtCore.Qt.GlobalColor.transparent)
            painter = QtGui.QPainter(pm)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
            painter.drawEllipse(2, 2, 18, 18)
            painter.end()
        self._icon_label.setPixmap(pm)

    def _on_theme_changed(self, _mode: str) -> None:
        """Refreshes dialog styling after an application theme change.

        Reapplies the title and body text styles, regenerates the semantic icon
        using the new theme tokens, and schedules the dialog card for repainting.
        This keeps all dialog chrome synchronized with the active light/dark
        theme without rebuilding the dialog.

        Args:
            _mode: Theme mode identifier emitted by `ThemeManager`. The mode is
                not needed directly because all visual values are resolved from
                the current theme tokens.
        """
        self._apply_title_style()
        self._apply_body_style()
        self._refresh_icon()
        self._card.update()

    def _on_button(self, result: int) -> None:
        """Completes the dialog with the specified action result.

        Stores the application-defined result value and accepts the dialog,
        causing the modal operation to return control to its caller.

        Args:
            result: Integer value associated with the selected dialog action.
        """
        self._result_value = result
        self.accept()
