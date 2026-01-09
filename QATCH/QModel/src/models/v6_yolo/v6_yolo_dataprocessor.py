# module: QModelV6Dataprocessor.py
"""
This module provides the data preprocessing and visualization logic required for the
QModel V6 YOLO pipeline. It handles the transformation of raw sensor CSV data into
interpolated time-series data, computes derived features (like the Difference curve),
and renders the signals into multi-channel images suitable for YOLO object detection
and classification.

Dependencies:
- opencv-python (cv2)
- pandas, numpy, matplotlib
- scipy.signal (medfilt)

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)
Date:
    2026-01-09

Version:
    6.0.1
"""

import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt


class QModelV6YOLO_DataProcessor:
    """
    A utility class for preprocessing sensor data and generating image inputs for YOLO.

    This class handles the end-to-end data pipeline from raw CSV to model input:
    1. Cleaning and interpolating raw sensor data.
    2. Computing the 'Difference' curve based on Dissipation and Frequency.
    3. Applying median filtering to smooth signal noise.
    4. Rendering time-series data into stacked RGB images for classification
       or detection tasks.

    Attributes:
        TAG (str): Log tag for the class.
        COL_TIME (str): Column name for relative time.
        COL_DISS (str): Column name for dissipation.
        COL_FREQ (str): Column name for resonance frequency.
        COL_DIFF (str): Column name for the calculated difference curve.
        TIME_STEP (float): The time interval for interpolation grid (seconds).
        MEDIAN_KERNEL (int): Kernel size for median filtering.
        DIFF_FACTOR (float): Scaling factor for the difference calculation.
        BASELINE_START_TIME (float): Start time for baseline averaging (seconds).
        BASELINE_END_TIME (float): End time for baseline averaging (seconds).
        IMG_CHANNELS (int): Number of image channels (3 for RGB/BGR).
    """

    TAG = "[QModelV6YOLO_DataProcessor]"
    # Column Names
    COL_TIME = "Relative_time"
    COL_DISS = "Dissipation"
    COL_FREQ = "Resonance_Frequency"
    COL_DIFF = "Difference"

    # Preprocessing Settings
    DROP_COLS = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]
    TIME_STEP = 0.005
    MEDIAN_KERNEL = 5

    # Difference Curve Calculation Settings
    DIFF_FACTOR = 2.0
    BASELINE_START_TIME = 0.5
    BASELINE_END_TIME = 2.5
    BASELINE_WINDOW_OFFSET = 2.0

    # Visualization Settings
    IMG_CHANNELS = 3
    EPSILON = 1e-9
    PADDING = 5

    # Color Palettes (BGR)
    COLOR_RED = (0, 0, 255)
    COLOR_GREEN = (0, 255, 0)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)

    # Map signals to standard colors/channels
    SIGNAL_CONFIG = {
        0: {"col": COL_DISS, "color": COLOR_RED, "ch_idx": 2},  # Red Channel
        1: {"col": COL_FREQ, "color": COLOR_GREEN, "ch_idx": 1},  # Green Channel
        2: {"col": COL_DIFF, "color": COLOR_BLUE, "ch_idx": 0},  # Blue Channel
    }

    @classmethod
    def preprocess_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans, interpolates, and enriches the raw sensor dataframe.

        Performs the following steps:
        1. Drops unnecessary columns (Ambient, Temperature, etc.).
        2. Reindexes the dataframe to a fixed time grid defined by `TIME_STEP`.
        3. Interpolates missing values.
        4. Computes and appends the 'Difference' curve.
        5. Applies a median filter to smooth numeric columns.

        Args:
            df (pd.DataFrame): The raw input dataframe containing sensor data.

        Returns:
            pd.DataFrame: The processed dataframe with interpolated time and
            smoothed signals, or None if the required time column is missing.
        """
        cols_to_drop = [c for c in cls.DROP_COLS if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        if cls.COL_TIME not in df.columns:
            return None
        t_min = df[cls.COL_TIME].min()
        t_max = df[cls.COL_TIME].max()
        new_time_grid = np.arange(t_min, t_max, cls.TIME_STEP)
        df = df.set_index(cls.COL_TIME)
        combined_index = df.index.union(new_time_grid).sort_values()
        df = df.reindex(combined_index).interpolate(method="index").loc[new_time_grid]
        df = df.reset_index().rename(columns={"index": cls.COL_TIME})
        diff_series = cls._compute_difference_curve(df)
        df[cls.COL_DIFF] = diff_series if diff_series is not None else 0.0
        for col in df.columns:
            if col != cls.COL_TIME and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = medfilt(df[col], kernel_size=cls.MEDIAN_KERNEL)

        return df

    @classmethod
    def _compute_difference_curve(cls, df: pd.DataFrame) -> pd.Series:
        """
        Computes the 'Difference' signal derived from Dissipation and Resonance Frequency.

        The calculation uses a baseline window (defined by `BASELINE_START_TIME` and
        `BASELINE_END_TIME`) to normalize the signals before computing the difference.

        Args:
            df (pd.DataFrame): The dataframe containing `Dissipation`,
                `Resonance_Frequency`, and `Relative_time`.

        Returns:
            pd.Series: A pandas Series containing the computed difference values,
            or None if required columns are missing or data is insufficient.
        """
        required = [cls.COL_FREQ, cls.COL_DISS, cls.COL_TIME]
        if not all(col in df.columns for col in required):
            return None

        xs = df[cls.COL_TIME].values
        if len(xs) == 0:
            return None
        i = np.searchsorted(xs, cls.BASELINE_START_TIME)
        j = np.searchsorted(xs, cls.BASELINE_END_TIME)

        if i == j and j < len(xs):
            j = np.searchsorted(xs, xs[j] + cls.BASELINE_WINDOW_OFFSET)

        if i >= len(df) or j > len(df) or i == j:
            i, j = 0, min(100, len(df))

        avg_res_freq = df[cls.COL_FREQ].iloc[i:j].mean()
        avg_diss = df[cls.COL_DISS].iloc[i:j].mean()

        ys_diss = (df[cls.COL_DISS].values - avg_diss) * avg_res_freq / 2
        ys_freq = avg_res_freq - df[cls.COL_FREQ].values

        return pd.Series(ys_freq - cls.DIFF_FACTOR * ys_diss, index=df.index)

    @classmethod
    def _get_signal_points(
        cls, values, img_w, strip_h, strip_idx, scaling_limits=None, col_name=None
    ):
        """
        Normalizes signal values and converts them to pixel coordinates.

        This helper method maps a 1D array of signal values to (x, y) coordinates
        suitable for drawing on an image strip.

        Args:
            values (np.array): The signal values to plot.
            img_w (int): The width of the target image in pixels.
            strip_h (int): The height of a single signal strip in pixels.
            strip_idx (int): The index of the strip (0, 1, or 2) to calculate vertical offset.
            scaling_limits (dict, optional): A dictionary of {col_name: (min, max)} for
                fixed scaling. Defaults to None (auto-scaling based on data min/max).
            col_name (str, optional): The name of the column being plotted, used for
                looking up scaling limits.

        Returns:
            np.ndarray: An (N, 2) array of integer coordinates (x, y), or None if
            values are insufficient.
        """
        if len(values) < 2:
            return None
        v_min, v_max = np.nanmin(values), np.nanmax(values)

        if scaling_limits and col_name and col_name in scaling_limits:
            v_min, v_max = scaling_limits[col_name]

        diff = v_max - v_min
        if diff == 0:
            diff = cls.EPSILON
            v_min -= cls.EPSILON

        norm = (values - v_min) / diff
        norm = np.clip(norm, 0, 1)
        x_points = np.linspace(0, img_w - 1, len(values)).astype(np.int32)
        draw_h = strip_h - (2 * cls.PADDING)
        y_rel = (strip_h - cls.PADDING) - (norm * draw_h)
        y_offset = strip_idx * strip_h
        y_points = (y_offset + y_rel).astype(np.int32)
        return np.stack((x_points, y_points), axis=1)

    @classmethod
    def generate_fill_cls(
        cls, df: pd.DataFrame, img_h: int, img_w: int, scaling_limits: dict = None
    ) -> np.ndarray:
        """
        Generates a stacked visualization of the signals for human validation/classification.

        This method produces an image with 3 horizontal strips, one for each signal
        (Dissipation, Frequency, Difference). It uses standard BGR colors for visualization.

        Args:
            df (pd.DataFrame): The preprocessed dataframe.
            img_h (int): The height of a *single* strip. The total image height
                will be 3 * img_h.
            img_w (int): The width of the image.
            scaling_limits (dict, optional): Fixed scaling limits for signal normalization.

        Returns:
            np.ndarray: A numpy array representing the generated image (Total_H, W, 3).
        """
        strip_h = img_h  # Input H is treated as height per strip
        total_h = 3 * strip_h
        img = np.zeros((total_h, img_w, cls.IMG_CHANNELS), dtype=np.uint8)

        if df.empty or len(df) < 2:
            return img

        for idx, config in cls.SIGNAL_CONFIG.items():
            col_name = config["col"]
            if col_name not in df.columns:
                continue

            pts = cls._get_signal_points(
                df[col_name].values, img_w, strip_h, idx, scaling_limits, col_name
            )

            if pts is None:
                continue

            strip_bottom_y = (idx + 1) * strip_h - cls.PADDING
            poly_pts = np.concatenate(
                [pts, [[pts[-1][0], strip_bottom_y]], [[pts[0][0], strip_bottom_y]]]
            )
            cv2.fillPoly(img, [poly_pts], color=config["color"])
            edge_color = tuple([min(c + 50, 255) for c in config["color"]])
            cv2.polylines(
                img,
                [pts.reshape((-1, 1, 2))],
                isClosed=False,
                color=edge_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return img

    @classmethod
    def generate_channel_det(
        cls, df: pd.DataFrame, img_w: int, img_h: int
    ) -> np.ndarray:
        """
        Generates the visualization input for the YOLO Detection Model.

        This method generates a stacked image where signals are encoded into specific
        RGB channels to act as masks.
        - Red Channel: Dissipation
        - Green Channel: Resonance Frequency
        - Blue Channel: Difference Curve

        Args:
            df (pd.DataFrame): The preprocessed dataframe.
            img_w (int): The total width of the image.
            img_h (int): The *total* height of the image (will be divided by 3 internally).

        Returns:
            np.ndarray: A numpy array representing the model input image (img_h, img_w, 3).
        """
        img = np.zeros((img_h, img_w, cls.IMG_CHANNELS), dtype=np.uint8)
        strip_h = img_h // 3  # Divide total height by 3 strips

        if df.empty or len(df) < 2:
            return img

        for idx, config in cls.SIGNAL_CONFIG.items():
            col_name = config["col"]
            if col_name not in df.columns:
                continue

            pts = cls._get_signal_points(
                df[col_name].values,
                img_w,
                strip_h,
                idx,
                scaling_limits=None,
                col_name=None,
            )

            if pts is None:
                continue

            strip_bottom_y = (idx + 1) * strip_h - cls.PADDING
            poly_pts = np.concatenate(
                [pts, [[pts[-1][0], strip_bottom_y]], [[pts[0][0], strip_bottom_y]]]
            )
            mask_color = [0, 0, 0]
            mask_color[config["ch_idx"]] = 255

            cv2.fillPoly(img, [poly_pts], tuple(mask_color))
            cv2.polylines(
                img,
                [pts.reshape((-1, 1, 2))],
                isClosed=False,
                color=cls.COLOR_WHITE,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return img
