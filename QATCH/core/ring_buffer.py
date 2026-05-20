"""
ring_buffer.py

This module provides a high-performance ``RingBuffer`` class backed by NumPy.
This buffer maintains a fixed-size backing array and an internal write head.
This guarantees ``O(1)`` time complexity for data insertion, preventing process stalls
or synchronization issues during tight data-ingestion loops.

References:
    - http://code.activestate.com/recipes/68429-ring-buffer/
        ^ ISSUE: This is 22 years old designed for Python 2.x.x

    - http://stackoverflow.com/questions/4151320/efficient-circular-buffer
        ^ ISSUE: https://stackoverflow.com/questions/4151320/efficient-circular-buffer#comment75816568_19157187
        ^ Also see: https://numpy.org/doc/2.1/reference/generated/numpy.roll.html

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2026-05-20
"""

import warnings
import numpy as np
import numpy.typing as npt
from typing import Any, Union

warnings.filterwarnings("ignore", category=RuntimeWarning)


class RingBuffer:
    """An ``O(1)`` circular-buffer implementation backed by a numpy array.

    This replaces previous `np.roll`-based designs to avoid ``O(N)`` copy operations
    per append. It maintains a fixed-size backing array and a write head, never
    reallocating after construction.

    Attributes:
        size_max (int): The maximum capacity of the buffer.
        size (int): The current number of valid samples written to the buffer.
    """

    def __init__(self, size_max: int, default_value: Any = 0, dtype: npt.DTypeLike = float) -> None:
        """Initializes the RingBuffer.

        Args:
            size_max (int): The maximum number of elements the buffer can hold.
            default_value (Any, optional): The value used to pre-fill the buffer. Defaults to 0.
            dtype (npt.DTypeLike, optional): The numpy data type of the array. Defaults to float.

        Raises:
            TypeError: If ``size_max`` is not an integer.
            ValueError: If ``size_max`` is less than or equal to zero.
        """
        if not isinstance(size_max, int):
            raise TypeError(f"size_max must be an integer, got {type(size_max).__name__}")
        if size_max <= 0:
            raise ValueError(f"size_max must be strictly positive, got {size_max}")

        self.size_max: int = size_max
        self._default_value: Any = default_value
        self._dtype: npt.DTypeLike = dtype

        # NOTE: _head points at the slot the NEXT append will write into.
        # NOTE: The most-recently-written sample is therefore at _head - 1.
        self._head: int = 0
        self._full: bool = False
        self.size: int = 0
        self._data: np.ndarray = np.full(size_max, default_value, dtype=dtype)

    def append(self, value: Any) -> None:
        """Appends new data to the ring buffer in ``O(1)`` time.

        If the buffer is full, the oldest sample is overwritten.

        Args:
            value (Any): The value to append to the buffer.
        """
        self._data[self._head] = value
        self._head = (self._head + 1) % self.size_max

        if not self._full:
            self.size += 1
            if self.size == self.size_max:
                self._full = True

    def get_all(self) -> np.ndarray:
        """Returns all elements from the buffer, preserving capacity size.

        When the buffer is not yet full, it returns the written chronological
        region followed by default-filled values to preserve the old contract.

        Returns:
            np.ndarray: A copy of the array from oldest to newest.
        """
        if not self._full:
            return self._data.copy()

        # Oldest sample sits at _head, newest at _head - 1.
        return np.concatenate((self._data[self._head :], self._data[: self._head]))

    def get_partial(self) -> np.ndarray:
        """Returns only the valid, written samples in chronological order.

        If no valid samples exist ``(size == 0)``, returns the default-filled array.

        Returns:
            np.ndarray: A slice of the buffer containing only written data from oldest to newest.
        """
        if self.size == 0 or self._full:
            return self.get_all()

        return self._data[: self.size].copy()

    def get_newest(self) -> Any:
        """Retrieves the most recently added element in the buffer.

        Raises:
            IndexError: If the buffer is empty.

        Returns:
            Any: The newest value.
        """
        if self.size == 0:
            raise ValueError("RingBuffer is empty")
        return self._data[(self._head - 1) % self.size_max]

    def get_oldest(self) -> Any:
        """Retrieves the oldest valid element remaining in the buffer.

        Raises:
            IndexError: If the buffer is empty.

        Returns:
            Any: The oldest value.
        """
        if self.size == 0:
            raise ValueError("RingBuffer is empty")
        # When full, oldest sits at the write head; otherwise at index 0
        return self._data[self._head] if self._full else self._data[0]

    def __getitem__(self, key: Union[int, slice]) -> Any:
        """Indexes into the chronological, oldest-to-newest view of the valid data.

        Args:
            key (Union[int, slice]): The index or slice to retrieve.

        Returns:
            Any: The requested element or slice from the valid samples.
        """
        return self.get_partial()[key]

    def __len__(self) -> int:
        """Returns the number of valid items currently in the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.size

    def __repr__(self) -> str:
        """Returns a string representation of the valid buffer state.

        Returns:
            str: The string representation detailing the partial view, size, and reversed view.
        """
        partial = self.get_partial()
        return f"{repr(partial)}\t{self.size}\t{repr(partial[::-1])}"
