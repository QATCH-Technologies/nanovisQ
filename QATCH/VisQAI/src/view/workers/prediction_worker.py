from PyQt5 import QtCore


class PredictionThread(QtCore.QThread):
    """
    A robust QThread subclass.
    Combines the thread and logic into one object to prevent Garbage Collection errors.
    """

    # Define the signal that sends data back to the main UI
    data_ready = QtCore.pyqtSignal(dict)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True  # Logic flag

    def run(self):
        """
        The code that runs in the background.
        """
        import time

        import numpy as np

        # 1. Simulate Calculation (Safe to sleep here)
        time.sleep(0.8)

        # 2. Check if we were told to stop during the sleep
        if not self._is_running:
            return

        # 3. The Math Logic
        shear_rates = [100, 1000, 10000, 100000, 15000000]
        viscosity = np.array([10, 9.8, 9.8, 6, 2])

        measured_y = None
        if self.config and self.config.get("measured", False):
            noise = np.random.normal(0, 2, len(shear_rates))
            measured_y = viscosity + noise

        data_package = {
            "x": shear_rates,
            "y": viscosity,
            "upper": viscosity * 1.15,
            "lower": viscosity * 0.85,
            "measured_y": measured_y,
            "config_name": (
                self.config.get("name", "Unknown") if self.config else "Unknown"
            ),
        }

        # 4. Emit Data
        if self._is_running:
            self.data_ready.emit(data_package)

    def stop(self):
        """Call this to stop the thread gracefully."""
        self._is_running = False
        self.quit()
        self.wait(1000)
