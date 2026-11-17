"""
QATCH.ui.styles.typography

Centralized text-styling constants for the whole app - the typography
counterpart to `QATCH.ui.styles.tokens` (colors). `FONT_SANS_STACK` /
`FONT_MONO_STACK` are the single source of truth for the CSS font-family
lists used in QSS and in inline HTML (pyqtgraph `TextItem`s, `QTextDocument`
content) - both accept identical `font-family:` syntax, so one literal
string serves both consumers.

This is now the single font system for the whole app - including the "flat
control" component family (buttons, line edits, combo/spin boxes, toggle,
option card), which previously bundled its own IBM Plex Sans/Mono faces.
Those faces registered static weights (Medium, SemiBold) as separate Qt
family names rather than true weight variants, which is why weight is now
expressed via `TypeSpec.weight` / `make_qfont(weight=...)` against the one
shared family list instead of selecting a different family string per
weight - real installed fonts (Segoe UI, Arial, ...) support `QFont`/CSS
weight directly.

Design note
-----------
Unlike `tokens.py`'s color ramps, there's no natural mathematical
relationship among the app's existing text sizes (8px-16pt), so this module
doesn't invent a modular scale - it names *today's* actual values so every
call site that wants "the dialog title size" or "the muted caption style"
can say so instead of repeating a magic number. `TypeSpec.weight` defaults
to 400 (regular); `font_css()` omits `font-weight` entirely when it's 400,
matching the existing convention where regular-weight QSS rules simply
don't declare a weight.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

from PyQt5 import QtGui

# =====================================================================
# Font-family stacks (CSS text, for QSS `font-family:` / inline HTML style=)
# =====================================================================
FONT_SANS_STACK = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
FONT_MONO_STACK = 'Consolas, "Courier New", monospace'

# Qt-actionable subset of FONT_SANS_STACK for QFont.setFamilies(). "system-ui"
# and "-apple-system" are CSS/WebKit pseudo-keywords, not real installed font
# names - QFont would look for a literal font called "system-ui", fail, and
# silently fall through, so make_qfont() uses this explicit list instead.
QT_SANS_FAMILIES = ["Segoe UI", "Roboto", "Helvetica", "Arial"]


class TypeSpec(NamedTuple):
    """A named text-size/weight/decoration role. `size` keeps its unit
    attached ("12px" / "14pt") since existing call sites mix both."""

    size: str
    weight: int = 400
    letter_spacing: Optional[str] = None
    transform: Optional[str] = None


# One constant per existing text role, values audited to reproduce today's
# rendered output exactly (see theme_manager.py's *_qss() helpers and
# main_window.py's plot-overlay text).
TYPE_DISPLAY = TypeSpec("16pt", 600, "0.5px")       # main_window welcome title
TYPE_TITLE = TypeSpec("14px", 700)                  # dialog_title_qss
TYPE_CARD_TITLE = TypeSpec("14pt", 700)              # card_title_qss
TYPE_SECTION_TITLE = TypeSpec("13px", 600)           # component/section titles
TYPE_BODY = TypeSpec("13px", 400)                    # dialog_message_qss
TYPE_FIELD = TypeSpec("12.5px", 600)                 # field_label_qss
TYPE_MONO_PREVIEW = TypeSpec("12.5px", 400)          # mono_preview_qss (pairs with FONT_MONO_STACK)
TYPE_DESC = TypeSpec("12px", 400)                    # desc_label_qss
TYPE_CAPTION = TypeSpec("10px", 600, "0.5px", "uppercase")  # caption_label_qss
TYPE_ERROR = TypeSpec("8.5pt", 600)                  # error_label_qss
TYPE_TOOLTIP = TypeSpec("9pt", 400)                  # plot hover tooltip title
TYPE_TOOLTIP_SUB = TypeSpec("8pt", 400)              # plot hover tooltip sub-line


def font_css(spec: TypeSpec, *, family: Optional[str] = None) -> str:
    """Renders a `TypeSpec` (plus an optional font-family) into a literal
    CSS fragment - valid inside a QSS rule body or a pyqtgraph/QTextDocument
    inline HTML `style="..."` attribute. Mirrors what `tok_css()`
    (theme_manager.py) does for color tokens.

    `family`, if given, is inserted verbatim (callers are responsible for
    any quoting a single font name might need) - usually `FONT_SANS_STACK`
    or `FONT_MONO_STACK`, but left generic in case a call site ever needs a
    one-off family.
    """
    parts = []
    if family is not None:
        parts.append(f"font-family: {family};")
    parts.append(f"font-size: {spec.size};")
    if spec.weight != 400:
        parts.append(f"font-weight: {spec.weight};")
    if spec.transform:
        parts.append(f"text-transform: {spec.transform};")
    if spec.letter_spacing:
        parts.append(f"letter-spacing: {spec.letter_spacing};")
    return " ".join(parts)


def make_qfont(
    *,
    families: Sequence[str] = QT_SANS_FAMILIES,
    style_hint: QtGui.QFont.StyleHint = QtGui.QFont.SansSerif,
    pixel_size: Optional[int] = None,
    point_size: Optional[float] = None,
    weight: Optional[int] = None,
    italic: bool = False,
) -> QtGui.QFont:
    """Builds a `QFont` with a real ordered-family fallback list via
    `QFont.setFamilies()` (Qt 5.13+; this repo pins PyQt5 5.15.11) plus a
    `setStyleHint()` generic fallback (the `QFont` equivalent of CSS
    `sans-serif`/`monospace`) that kicks in if none of `families` is
    installed. Defaults to `QFont.SansSerif` for `QT_SANS_FAMILIES`; pass
    `style_hint=QtGui.QFont.Monospace` when `families` is a monospace list
    (e.g. Consolas) so Qt substitutes another monospace face, not a sans one,
    if the family is missing. For raw-`QFont` call sites that bypass QSS
    entirely - pyqtgraph `setTickFont`, `setFont` on non-QSS-styled items.

    `weight` uses Qt's own `QFont.Weight` scale (e.g. `QtGui.QFont.Bold`),
    not the CSS 100-900 scale used by `TypeSpec.weight` - the two aren't
    interchangeable, so don't pass a `TypeSpec.weight` value straight in.
    """
    font = QtGui.QFont()
    font.setFamilies(list(families))
    font.setStyleHint(style_hint)
    if pixel_size is not None:
        font.setPixelSize(pixel_size)
    if point_size is not None:
        font.setPointSizeF(point_size)
    if weight is not None:
        font.setWeight(weight)
    if italic:
        font.setItalic(True)
    return font
