"""
QATCH.ui.labels.temperature_label.py

Temperature label with text change notifications.

This module provides :class:`TemperatureLabel`, a small extension of
:class:`QtWidgets.QLabel` that emits a signal whenever its displayed text is
updated.

The widget behaves identically to a standard QLabel while adding a convenient
observer mechanism for other UI components. Rather than polling the label for
changes, consumers can connect to the emitted signal to react immediately
when the displayed temperature or status text changes.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-08-04
"""

from PyQt5 import QtCore, QtWidgets


class TemperatureLabel(QtWidgets.QLabel):
    """A QLabel that emits a signal whenever its text changes.

    This class overrides :meth:`setText` to preserve the standard QLabel
    behavior while notifying interested objects that the displayed text has
    been updated. It provides a lightweight publish-subscribe mechanism for
    synchronizing dependent UI components without requiring polling or direct
    coupling.

    Attributes:
        text_updated (QtCore.pyqtSignal): Signal emitted after the label text
            has been updated. The new text is provided as the signal
            argument.
    """

    text_updated = QtCore.pyqtSignal(str)

    def setText(self, text: str) -> None:
        """Sets the label text and emits the update notification.

        The method first delegates to the base QLabel implementation to
        update the displayed text and then emits :attr:`text_updated`,
        allowing connected slots to react to the new value.

        Args:
            text (str): New text to display in the label.
        """
        super().setText(text)
        self.text_updated.emit(text)
