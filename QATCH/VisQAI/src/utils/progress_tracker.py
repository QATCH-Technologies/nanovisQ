"""
progress_tracker.py

Provides a lightweight, cross-process progress dialog class for tracking progress in multiprocessing scenarios.
"""
import multiprocessing


class Lite_QProgressDialog:
    """
    A lightweight, cross-process progress dialog for tracking progress in multiprocessing scenarios.
    Mimics the interface of QProgressDialog but is not a GUI widget.
    """

    def __init__(self, labelText, cancelButtonText, minimum: int, maximum: int, parent) -> None:
        """
        Initialize the Lite_QProgressDialog.

        Args:
            labelText (str): The label text for the dialog (not used).
            cancelButtonText (str): The cancel button text (not used).
            minimum (int): The minimum progress value.
            maximum (int): The maximum progress value.
            parent: The parent object (not used).
        """
        self._minimum = minimum
        self._maximum = maximum
        self._value = minimum
        self._canceled = False
        self._queue = multiprocessing.Queue()

    def queue(self):
        """
        Returns the multiprocessing.Queue used for communicating progress updates.
        """
        return self._queue

    def setMinimum(self, minimum):
        """
        Set the minimum progress value.
        """
        self._minimum = minimum

    def setMaximum(self, maximum):
        """
        Set the maximum progress value.
        """
        self._maximum = maximum

    def setValue(self, value):
        """
        Set the current progress value, clamped to [minimum, maximum].
        If the value changes, put the new value on the queue.
        """
        old_value = self._value
        if value < self._minimum:
            self._value = self._minimum
        elif value > self._maximum:
            self._value = self._maximum
        else:
            self._value = value
        if self._value != old_value:
            self._queue.put(self._value)

    def value(self):
        """
        Get the current progress value.
        """
        return self._value

    def minimum(self):
        """
        Get the minimum progress value.
        """
        return self._minimum

    def maximum(self):
        """
        Get the maximum progress value.
        """
        return self._maximum

    def cancel(self):
        """
        Mark the dialog as canceled.
        """
        self._canceled = True

    def wasCanceled(self):
        """
        Check if the dialog was canceled.
        """
        return self._canceled

    def reset(self):
        """
        Reset the progress value to minimum and clear the canceled state.
        Put the minimum value on the queue.
        """
        self._value = self._minimum
        self._canceled = False
        self._queue.put(self._value)
