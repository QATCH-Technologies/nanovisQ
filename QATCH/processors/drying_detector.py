from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from QATCH.common.logger import Logger as Log


class DryingDetection:
    """State machine for detecting when a sensor has dried.

    Criteria for drying:
      - Stability: stddev of each normalized window < threshold.
      - Flatness: absolute slope of each normalized window < threshold.

    The first call that meets both criteria returns True; thereafter,
    update() will always return False until you call reset().

    Args:
        window_size: number of samples in each rolling window.
        sigma_stable_freq: max allowed stddev of normalized frequency.
        sigma_stable_diss: max allowed stddev of normalized dissipation.
        flat_slope_eps: max allowed abs(slope) of normalized windows.
    """

    def __init__(
        self,
        window_size: int,
        snr_stable_freq: float,
        snr_stable_diss: float,
        flat_slope_eps_diss: float,
        flat_slope_eps_freq: float,
        debug_plot: bool = False,
    ) -> None:
        self.win_n = int(window_size)
        self.freq_w = deque(maxlen=self.win_n)
        self.diss_w = deque(maxlen=self.win_n)
        self.time_w = deque(maxlen=self.win_n)
        self.snr_stable_freq = float(snr_stable_freq)
        self.snr_stable_diss = float(snr_stable_diss)
        self.flat_eps_diss = float(flat_slope_eps_diss)
        self.flat_eps_freq = float(flat_slope_eps_freq)
        self._detection_index = None
        self._last_message: str = ""
        self._init_exec: bool = True
        self._start_time = 0.0
        self.debug_plot = debug_plot
        if self.debug_plot:
            self.plotter = LivePlotHelper()

        self.reset()

    def reset(self) -> None:
        """Reset the drying detection state.

        Clears all internal buffers and resets detection flags and counters.

        Returns:
            None
        """
        self.freq_w.clear()
        self.diss_w.clear()
        self.time_w.clear()
        self._last_message = ""
        self._dried = False
        self._init_exec = True
        self._sample_count = 0
        self._dry_time = 0.0

    @property
    def is_dry(self) -> bool:
        """bool: Whether a drying event has been detected.

        Returns:
            bool: True if drying has been detected at least once; False otherwise.
        """
        return self._dried

    @property
    def dry_time(self) -> float:
        """float: The relative time at which the drying event was detected.

        Returns:
            float: The time value from the input time array where drying was first detected,
                or None if no drying has been detected.
        """
        return self._dry_time

    def update(
        self,
        resonance_frequency: np.ndarray,
        dissipation: np.ndarray,
        relative_time: np.ndarray,
    ):
        """
        Feed in arrays of newest-first samples; process them in one batch.

        Args:
            resonance_frequency: 1D array of newest-first frequency samples.
            dissipation:        1D array of newest-first dissipation samples.
            relative_time:      1D array of newest-first time samples.

        Returns:
            Tuple[bool, str]:
                dried:  True if drying has just been detected (only once).
                status: A newline-separated string of one or more status messages.
        """
        if self._dried:
            self._last_message = "Dried"
            return True, self._last_message

        f_arr = np.asarray(resonance_frequency)[::-1]
        d_arr = np.asarray(dissipation)[::-1]
        t_arr = np.asarray(relative_time)[::-1]
        if f_arr.shape != d_arr.shape or f_arr.shape != t_arr.shape:
            return False, self._last_message

        valid = t_arr > 0.0
        f_arr, d_arr, t_arr = f_arr[valid], d_arr[valid], t_arr[valid]
        if f_arr.size == 0:
            return False, self._last_message

        t_arr_uniq = np.unique(t_arr)
        batch_size = t_arr_uniq.size
        self._sample_count = batch_size
        self.freq_w.extend(f_arr)
        self.diss_w.extend(d_arr)
        self.time_w.extend(t_arr)
        current_time = float(t_arr.max())
        arr_f = np.array(self.freq_w, dtype=float)
        arr_d = np.array(self.diss_w, dtype=float)
        snr_f = self._compute_snr(arr_f)
        snr_d = self._compute_snr(arr_d)
        slope_f = self._compute_slope(arr_f)
        slope_d = self._compute_slope(arr_d)
        if getattr(self, "debug_plot", False):
            current_stats = {"snr_f": snr_f, "slope_f": slope_f, "snr_d": snr_d, "slope_d": slope_d}
            thresholds = {
                "snr_f": self.snr_stable_freq,
                "slope_f": self.flat_eps_freq,
                "snr_d": self.snr_stable_diss,
                "slope_d": self.flat_eps_diss,
            }
            self.plotter.render_frame(
                self.time_w, self.freq_w, self.diss_w, current_stats, thresholds
            )
        if np.max(t_arr) < 10.0:
            self._last_message = "Calibrating..."
            return False, self._last_message

        sensor_not_dry = (
            snr_f >= self.snr_stable_freq
            or abs(slope_f) >= self.flat_eps_freq
            or snr_d >= self.snr_stable_diss
            or abs(slope_d) >= self.flat_eps_diss
        )

        if sensor_not_dry:
            elapsed = current_time - self._start_time
            if elapsed >= 45.0:
                self._last_message = "Sensor not dry, please restart"
                return False, self._last_message
            elif not self._init_exec:
                self._last_message = "Sensor not ready, please wait.."
                return False, self._last_message
            else:
                self._init_exec = False
                return False, self._last_message

        # If we reach here, all stability conditions are met
        self._dried = True
        self._dry_time = current_time
        Log.i(f"Dry time was {self._dry_time}")
        self._last_message = "Dried"
        return True, self._last_message

    def _compute_slope(self, arr: np.ndarray) -> float:
        """Compute the slope of a linear fit to the array.

        Fits a first-degree polynomial to the data points (x, arr) with x values
        from 0 to N-1 and returns the slope.

        Args:
            arr (np.ndarray): 1D array of values for slope computation.

        Returns:
            float: Slope of the fitted line. Returns 0.0 if the array has fewer than
            two points.
        """
        if arr.size < 2:
            return 0.0
        x = np.arange(arr.size)
        m, _ = np.polyfit(x, arr, 1)
        return float(m)

    def _compute_snr(self, arr: np.ndarray) -> float:
        """Currently just comutes standard deviation"""
        # sigma = np.power(float(np.nanstd(arr)), 2)
        # if sigma == 0:
        #     return np.inf
        # return np.power(float(np.nanmean(arr)), 2) / sigma
        sigma = float(np.nanstd(arr))
        return sigma


class LivePlotHelper:

    def __init__(self):

        plt.ion()
        self.fig, (self.ax_freq, self.ax_diss) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        self.fig.canvas.manager.set_window_title("Drying Detection Live View")
        self.fig.suptitle("Live View")

        (self.line_freq,) = self.ax_freq.plot([], [], "b-", linewidth=1.5, label="Resonance Freq")
        (self.line_diss,) = self.ax_diss.plot([], [], "r-", linewidth=1.5, label="Dissipation")

        self.ax_freq.set_ylabel("Frequency")
        self.ax_freq.legend(loc="upper right")
        self.ax_freq.grid(True, linestyle="--", alpha=0.6)

        self.ax_diss.set_ylabel("Dissipation")
        self.ax_diss.set_xlabel("Relative Time")
        self.ax_diss.legend(loc="upper right")
        self.ax_diss.grid(True, linestyle="--", alpha=0.6)
        box_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray")
        self.text_freq = self.ax_freq.text(
            0.02,
            0.95,
            "",
            transform=self.ax_freq.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=box_props,
            family="monospace",
        )
        self.text_diss = self.ax_diss.text(
            0.02,
            0.95,
            "",
            transform=self.ax_diss.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=box_props,
            family="monospace",
        )

        self.fig.tight_layout()

    def render_frame(self, time_w, freq_w, diss_w, stats: dict, thresholds: dict):
        if not time_w:
            return
        t_arr = np.array(time_w)
        self.line_freq.set_data(t_arr, np.array(freq_w))
        self.line_diss.set_data(t_arr, np.array(diss_w))
        f_snr_ok = "✗" if stats["snr_f"] >= thresholds["snr_f"] else "✓"
        f_slp_ok = "✗" if abs(stats["slope_f"]) >= thresholds["slope_f"] else "✓"

        freq_str = (
            f"StdDev: {stats['snr_f']:.3e} / {thresholds['snr_f']:.3e} [{f_snr_ok}]\n"
            f"Slope (Abs):  {abs(stats['slope_f']):.3e} / {thresholds['slope_f']:.3e} [{f_slp_ok}]"
        )
        self.text_freq.set_text(freq_str)

        d_snr_ok = "✗" if stats["snr_d"] >= thresholds["snr_d"] else "✓"
        d_slp_ok = "✗" if abs(stats["slope_d"]) >= thresholds["slope_d"] else "✓"

        diss_str = (
            f"StdDev: {stats['snr_d']:.3e} / {thresholds['snr_d']:.3e} [{d_snr_ok}]\n"
            f"Slope (Abs):  {abs(stats['slope_d']):.3e} / {thresholds['slope_d']:.3e} [{d_slp_ok}]"
        )
        self.text_diss.set_text(diss_str)
        self.ax_freq.relim()
        self.ax_freq.autoscale_view(scalex=True, scaley=True)
        self.ax_diss.relim()
        self.ax_diss.autoscale_view(scalex=True, scaley=True)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.ioff()
        plt.close(self.fig)
