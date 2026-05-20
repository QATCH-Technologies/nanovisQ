"""
ring_buffer.py

This module provides a high-performance ``RingBuffer`` class backed by NumPy.
It maintains a contiguous linear arena and a moving write head, guaranteeing
amortized ``O(1)`` insertion AND ``O(1)`` reads. This prevents process stalls or
synchronization issues during tight data-ingestion loops, and -- critically --
during the much hotter read path that drains the buffer many times per UI tick.

Design history (why this is arena-backed rather than wrapped):
    - A ``np.roll`` design is ``O(N)`` on every append (it copies the whole
      array each insertion), which stalls tight ingestion loops.
    - A wrapped head/tail design fixes append to ``O(1)`` but forces reads to
      ``np.concatenate`` the two halves once the buffer wraps -- a fresh
      ``O(N)`` allocation on every ``get_partial``/``get_all`` call. Because the
      render path reads the buffer dozens of times per tick (per channel, per
      curve, per detector), that merely relocates the ``O(N)`` cost from the
      producer to the hotter consumer side.
    - This arena design keeps the live samples ALWAYS contiguous in a backing
      array sized ``2 * size_max``. Appends advance a tail; the live window is
      slid back to the front only when the tail reaches the end of the arena
      (at most once per ``size_max`` appends -> amortized ``O(1)``). Reads then
      return a plain contiguous slice -- a view, with no allocation or copy.

Consumer contract:
    ``get_partial`` and ``get_all`` (when full) return a VIEW into the internal
    backing store for speed, not a snapshot. Consume the result (slice it, copy
    it, feed it to ``setData`` / ``np.average`` / ``np.convolve``) before the
    next ``append`` on the same buffer. Do not retain a returned array across
    appends and expect it to stay frozen. All current callers read-then-discard
    within a single tick, which is safe.

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
    """An amortized-``O(1)`` circular buffer backed by a contiguous numpy arena.

    The backing array has room for ``2 * size_max`` elements so that the live
    sample window can grow rightward before being slid back to the front. The
    live region (``[_start:_end]``) is always contiguous, so reads return a
    zero-copy view. See the module docstring for the consumer contract regarding
    returned views.

    Attributes:
        size_max (int): The maximum capacity (window length) of the buffer.
        size (int): The current number of valid samples held in the window.
    """

    def __init__(self, size_max: int, default_value: Any = 0, dtype: npt.DTypeLike = float) -> None:
        """Initializes the RingBuffer.

        Args:
            size_max (int): The maximum number of elements the buffer can hold.
            default_value (Any, optional): The value used to pre-fill the arena. Defaults to 0.
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

        # The arena holds 2x the window. _start indexes the oldest live sample;
        # _end is one past the newest. The live region is always contiguous and
        # satisfies the invariant: _end - _start == size.
        self._cap: int = 2 * size_max
        self._start: int = 0
        self._end: int = 0
        self.size: int = 0
        self._data: np.ndarray = np.full(self._cap, default_value, dtype=dtype)

    def append(self, value: Any) -> None:
        """Appends new data to the ring buffer in amortized ``O(1)`` time.

        If the window is full, the oldest sample is dropped by advancing the
        window start. No per-append reallocation or full-array copy occurs; the
        live window is slid back to the front of the arena at most once every
        ``size_max`` appends.

        Args:
            value (Any): The value to append to the buffer.
        """
        if self._end >= self._cap:
            # Slide the live window back to the front. The source and
            # destination ranges do not overlap (cap == 2 * size_max), so a
            # plain block copy is safe. Amortized O(1) over size_max appends.
            np.copyto(self._data[: self.size], self._data[self._start : self._end])
            self._start = 0
            self._end = self.size

        self._data[self._end] = value
        self._end += 1

        if self.size < self.size_max:
            self.size += 1
        else:
            # Window already full: drop the oldest sample.
            self._start += 1

    def get_all(self) -> np.ndarray:
        """Returns the buffer contents from oldest to newest.

        When the buffer is full, this returns a zero-copy VIEW of the live
        window (see the module-level consumer contract). When it is not yet
        full, it returns a freshly allocated ``size_max``-length array with the
        written samples followed by ``default_value`` padding, preserving the
        legacy capacity-sized contract.

        Returns:
            np.ndarray: Elements ordered oldest to newest.
        """
        if self.size >= self.size_max:
            return self._data[self._start : self._end]

        out = np.full(self.size_max, self._default_value, dtype=self._dtype)
        out[: self.size] = self._data[self._start : self._end]
        return out

    def get_partial(self) -> np.ndarray:
        """Returns only the valid, written samples in chronological order.

        This is the hot-path accessor. When at least one sample is present it
        returns a zero-copy VIEW of the live window (oldest to newest); see the
        module-level consumer contract. If no valid samples exist
        ``(size == 0)`` it returns the default-filled capacity array.

        Returns:
            np.ndarray: The valid samples from oldest to newest.
        """
        if self.size == 0:
            return self.get_all()

        return self._data[self._start : self._end]

    def get_newest(self) -> Any:
        """Retrieves the most recently added element in the buffer.

        Raises:
            ValueError: If the buffer is empty.

        Returns:
            Any: The newest value.
        """
        if self.size == 0:
            raise ValueError("RingBuffer is empty")
        return self._data[self._end - 1]

    def get_oldest(self) -> Any:
        """Retrieves the oldest valid element remaining in the buffer.

        Raises:
            ValueError: If the buffer is empty.

        Returns:
            Any: The oldest value.
        """
        if self.size == 0:
            raise ValueError("RingBuffer is empty")
        return self._data[self._start]

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
