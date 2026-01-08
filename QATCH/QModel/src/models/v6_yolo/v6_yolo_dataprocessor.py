import cv2
import numpy as np
import pandas as pd
from scipy.signal import medfilt


class QModelV6YOLO_DataProcessor:

    TAG = "QModelV6YOLO_DataProcessor"

    # --- Configuration / Constants ---
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
        Shared Math: Normalizes data values into pixel coordinates (x, y).
        Returns an array of points (N, 2) suitable for cv2 functions.
        """
        if len(values) < 2:
            return None

        # Determine limits
        v_min, v_max = np.nanmin(values), np.nanmax(values)

        if scaling_limits and col_name and col_name in scaling_limits:
            v_min, v_max = scaling_limits[col_name]

        diff = v_max - v_min
        if diff == 0:
            diff = cls.EPSILON
            # Slight buffer if flatline to avoid div/0 issues
            v_min -= cls.EPSILON

        # Normalize 0..1
        norm = (values - v_min) / diff
        norm = np.clip(norm, 0, 1)

        # Map to X coordinates
        x_points = np.linspace(0, img_w - 1, len(values)).astype(np.int32)

        # Map to Y coordinates (Inverted for image coordinates: 0 is top)
        # Apply padding to keep line fully inside strip
        draw_h = strip_h - (2 * cls.PADDING)
        y_rel = (strip_h - cls.PADDING) - (norm * draw_h)

        # Apply Strip Offset
        y_offset = strip_idx * strip_h
        y_points = (y_offset + y_rel).astype(np.int32)

        return np.stack((x_points, y_points), axis=1)

    @classmethod
    def generate_fill_cls(
        cls, df: pd.DataFrame, img_h: int, img_w: int, scaling_limits: dict = None
    ) -> np.ndarray:
        """
        Generates the visualization for human validation (Standard colors, edges).
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

            # Create Polygon for fill (Signal line + bottom corners)
            poly_pts = np.concatenate(
                [pts, [[pts[-1][0], strip_bottom_y]], [[pts[0][0], strip_bottom_y]]]
            )

            # Draw Fill
            cv2.fillPoly(img, [poly_pts], color=config["color"])

            # Draw Lighter Edge
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
        Generates the visualization for Model Input (Masks in RGB channels).
        Note: img_h here is usually the TOTAL height.
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

            # Create Polygon for fill
            poly_pts = np.concatenate(
                [pts, [[pts[-1][0], strip_bottom_y]], [[pts[0][0], strip_bottom_y]]]
            )

            # Generate Mask Color: 255 in the specific channel, 0 elsewhere
            mask_color = [0, 0, 0]
            mask_color[config["ch_idx"]] = 255

            # Draw Fill
            cv2.fillPoly(img, [poly_pts], tuple(mask_color))

            # Draw Edge (White)
            cv2.polylines(
                img,
                [pts.reshape((-1, 1, 2))],
                isClosed=False,
                color=cls.COLOR_WHITE,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return img
