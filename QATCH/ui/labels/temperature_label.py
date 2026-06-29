from PyQt5 import QtCore, QtWidgets


class TemperatureLabel(QtWidgets.QLabel):
    """QLabel that fires a signal whenever setText() is called.

    This class extends the standard QLabel to provide an observer mechanism.
    It maintains full backward compatibility, allowing callers to use
    `setText()` as normal, while simultaneously emitting a signal that
    allows other UI components to react to text changes without requiring
    expensive polling.

    Attributes:
        text_updated (QtCore.pyqtSignal): A signal emitted with the new
            text string whenever `setText` is invoked.
    """

    text_updated = QtCore.pyqtSignal(str)

    def setText(self, text: str) -> None:  # noqa
        """Sets the label text and notifies observers of the change.

        Overrides the base `setText` method to emit the `text_updated`
        signal after updating the display.

        Args:
            text (str): The new string to display on the label.
        """
        super().setText(text)
        self.text_updated.emit(text)
