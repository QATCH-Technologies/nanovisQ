"""
QATCH.ui.components.glass_panel

Static frosted-glass content panel matching the app's glass surface family
(see QATCH.ui.components.glass_paint). Renders a QFrame via the shared
`paint_glass_surface` recipe with an opaque base fill, so panel content
never bleeds through onto whatever sits behind the panel (the same
translucency-visibility fix applied to GlassDialog's card).

This replaces the several independently hand-rolled "frosted card" QFrame
subclasses/QSS blocks that used to live inside the data-management mode
widgets (Import's `_card`, Export's `_card`/`_glass_panel_qss`, Advanced's
`_panel`) - all of which hardcoded light-mode-only colors and none of which
refreshed on a theme change.

Usage
-----
    panel = GlassPanel()
    layout = QtWidgets.QVBoxLayout(panel)
    ...

    danger_panel = GlassPanel(danger=True)   # tinted toward the danger token
"""

from __future__ import annotations

from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from QATCH.ui.components.glass_paint import paint_glass_surface
from QATCH.ui.styles.theme_manager import ThemeManager

_RADIUS = 12.0


class QATCHPanel(QtWidgets.QFrame):
    """A static frosted-glass content card.

    Attributes:
        _danger (bool): When True, the base fill is tinted toward the
            `danger` token instead of the neutral glass tint.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, *, danger: bool = False) -> None:
        super().__init__(parent)
        self._danger = danger
        self.setAutoFillBackground(False)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        ThemeManager.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, _mode: str) -> None:
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        tokens = ThemeManager.instance().tokens()

        if self._danger:
            # Tint the base fill toward the danger token instead of the
            # neutral glass base, keeping the same shimmer/rim/vignette recipe.
            base = tokens["plot_glass_base"]
            danger = tokens["danger"]
            tokens = dict(tokens)
            tokens["plot_glass_base"] = (
                (base[0] + danger[0]) // 2,
                (base[1] + danger[1]) // 2,
                (base[2] + danger[2]) // 2,
                base[3],
            )

        p = QtGui.QPainter(self)
        p.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        paint_glass_surface(
            self,
            radius=_RADIUS,
            tokens=tokens,
            painter=p,
            shimmer_height=40.0,
            draw_vignette=False,
            opaque_base=True,
        )
        p.end()
