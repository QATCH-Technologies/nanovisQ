import hashlib
import os
import sys
from datetime import date, datetime
from time import localtime, monotonic, strftime
from xml.dom import minidom

import numpy as np
import pyzipper
from numpy import loadtxt
from PyQt5 import QtCore, QtWidgets
from scipy.interpolate import interp1d

from QATCH.common.fileStorage import secure_open
from QATCH.common.logger import Logger as Log
from QATCH.common.userProfiles import UserProfiles
from QATCH.core.constants import Constants
from QATCH.ui.widgets.table_view_widget import TableView

TAG = "[AnalyzeWorker]"
USE_NEW_FILL_METHOD = True


class AnalyzeWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(int, str)

    def __init__(self, parent, data_path, xml_path, poi_vals, diff_factor):
        super().__init__()
        self.parent = parent
        self._exitSuccess = False

        # set global expectations
        self.loaded_datapath = data_path
        self.xml_path = xml_path
        self.poi_vals = poi_vals
        if diff_factor != None:
            self.diff_factor = diff_factor
        # else: self.diff_factor not set

        self._running = False
        self.progress.connect(self._started)
        self.progress.connect(QtCore.QCoreApplication.processEvents)
        self.finished.connect(self._stopped)

    def _started(self, val, status):
        self._running = True

    def _stopped(self):
        self._running = False

    def is_running(self):
        return self._running

    def exitCode(self):
        return self._exitSuccess

    def run(self):
        try:
            # self.progress.emit(0, "Analyzing...")
            status_label = "Analyzing..."
            self.update(status_label)

            # lazy load required modules
            import matplotlib.backends.backend_pdf
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg,
            )
            from matplotlib.backends.backend_qt5agg import (
                NavigationToolbar2QT as NavigationToolbar,
            )
            from scipy.optimize import curve_fit
            from scipy.signal import argrelextrema, savgol_filter

            matplotlib.use("Qt5Agg")

            # data, rows, cols = [
            #     {
            #         "A": ["", "", "", ""],
            #         "B": ["", "", "", ""],
            #         "C": ["", "", "", ""],
            #         "D": ["", "", "", ""],
            #     },
            #     4,
            #     4,
            # ]
            # results_table = TableView(data, rows, cols)
            # results_figure = pg.PlotWidget()
            # results_figure.setBackground("w")
            # plot_text = pg.TextItem("", (51, 51, 51), anchor=(0.5, 0.5))
            # plot_text.setHtml("<span style='font-size: 10pt'>Analyze in-progress...</span>")
            # plot_text.setPos(0.5, 0.5)
            # results_figure.addItem(plot_text, ignoreBounds=True)
            # self.parent.results_split.replaceWidget(0, results_table)
            # self.parent.results_split.replaceWidget(1, results_figure)
            # self.parent.results_split.setEnabled(False)

            # self.progress.emit(50, "Analyzing...")

            confirm_envelopeSize = False
            confirm_startIndex = False
            confirm_stopIndex = False
            confirm_postIndex = False
            confirm_blipIndices = False

            poi_vals = self.poi_vals
            # poi_vals = np.insert(poi_vals, 2, poi_vals[1]+2)

            data_path = self.loaded_datapath
            data_title = os.path.splitext(os.path.basename(data_path))[0]
            Log.i(TAG, "Starting Analysis process of file: {}".format(data_path))

            self.update(status_label)

            batch_input_type = "none"
            batch = "N/A"
            xml_path = data_path[0:-4] + ".xml" if self.xml_path == None else self.xml_path
            xml_params = {}
            if os.path.exists(xml_path):
                doc = minidom.parse(xml_path)
                params = doc.getElementsByTagName("params")[-1]  # most recent element

                for p in params.childNodes:
                    if p.nodeType == p.TEXT_NODE:
                        continue  # only process elements

                    name = p.getAttribute("name")
                    value = p.getAttribute("value")

                    # Normalize file encoding so it can be logged as UTF-8
                    xml_params[name] = value.encode(
                        encoding="ascii", errors="xmlcharrefreplace"
                    ).decode(encoding="utf-8", errors="ignore")

                    if name == "batch_number" and p.hasAttribute("input"):
                        batch_input_type = p.getAttribute("input")

                    # if name == "bioformulation":
                    # if name == "protein":
                    # if name == "surfactant":
                    # if name == "concentration":
                    # if name == "surface_tension":
                    # if name == "contact_angle":
                    # if name == "density":

                    # if name == "protein_type":
                    # if name == "protein_concentration":
                    # if name == "buffer_type":
                    # if name == "buffer_concentration":
                    # if name == "surfactant_type":
                    # if name == "surfactant_concentration":
                    # if name == "stabilizer_type":
                    # if name == "stabilizer_concentration":

                batch = str(
                    xml_params.get("batch_number", "N/A")
                )  # used later on to pull batch params during analysis

                # START BATCH PARAMS INSERT #
                batch_params_old = {}
                batch_params_xml = doc.getElementsByTagName("batch_params")
                if len(batch_params_xml) > 0:
                    # most recent element
                    batch_params_xml = batch_params_xml[-1]
                    for p in batch_params_xml.childNodes:
                        if p.nodeType == p.TEXT_NODE:
                            continue  # only process elements

                        name = p.getAttribute("name")
                        value = p.getAttribute("value")
                        batch_params_old[name] = value
                else:
                    batch_params_xml = None

                batch_found = Constants.get_batch_param(batch)
                batch_params_all = Constants.get_batch_param(
                    batch, "ALL"
                )  # dictionary of {param_names:param_vals}
                batch_params_now = {}
                batch_params_now["BATCH"] = batch
                batch_params_now.update(
                    batch_params_all
                )  # update dict so that "BATCH" comes first, followed by other params

                # Look for changes
                changes = True
                if batch_params_xml != None:
                    changes = False
                    for key, val in batch_params_old.items():
                        if key in batch_params_now.keys():
                            if batch_params_now[key] != val:
                                changes = True
                                break
                        else:
                            changes = True
                            break
                    if not changes:
                        for key, val in batch_params_now.items():
                            if key in batch_params_old.keys():
                                if batch_params_old[key] != val:
                                    changes = True
                                    break
                            else:
                                changes = True
                                break

                # Add changed <batch_params> to XML
                if changes:
                    xml = doc.documentElement

                    # create new batch_params element
                    recorded_at = datetime.now().isoformat()
                    batch_params = doc.createElement("batch_params")
                    batch_params.setAttribute("recorded", recorded_at)
                    xml.appendChild(batch_params)

                    # param = doc.createElement('batch_param')
                    # param.setAttribute('name', str("BATCH"))
                    # param.setAttribute('value', str(batch))
                    # batch_params.appendChild(param)

                    for k, v in batch_params_now.items():
                        param = doc.createElement("batch_param")
                        param.setAttribute("name", str(k))
                        param.setAttribute("value", str(v))
                        if k.upper() == "BATCH":
                            param.setAttribute("found", str(batch_found))
                        batch_params.appendChild(param)

                    hash = hashlib.sha256()
                    for p in batch_params.childNodes:
                        for name, value in p.attributes.items():
                            hash.update(name.encode())
                            hash.update(value.encode())
                    signature = hash.hexdigest()
                    batch_params.setAttribute("signature", signature)

                    try:
                        with open(self.xml_path, "w", encoding="utf-8") as f:
                            xml_str = doc.toxml(encoding="ascii").decode(
                                encoding="utf-8", errors="ignore"
                            )
                            f.write(xml_str)
                            Log.d(f"Added <batch_params> to XML file: {self.xml_path}")
                    except OSError as ose:  # FileNotFoundError
                        Log.e(f"Filesystem error writing XML: {self.xml_path}")
                        Log.e("Error Details:", ose.strerror)
                    except UnicodeError as ue:  # UnicodeEncodeError, UnicodeDecodeError
                        Log.e(f"Unicode error writing XML: {self.xml_path}")
                        Log.e("Error Details:", ue.reason)
                # END BATCH PARAMS INSERT #

            self.update(status_label)

            Log.d(f"xml_path = {xml_path}")
            Log.d(f"xml_found = {os.path.exists(xml_path)}")
            Log.d(xml_params)

            BIOFORMULATION = xml_params.get("bioformulation", "False") == "True"
            ST = float(xml_params.get("surface_tension", 69.0))
            CA = float(xml_params.get("contact_angle", 55.0))
            DENSITY = float(xml_params.get("density", 1.2))

            # only do this if "contact_angle" is auto-calculated (NOT if 'manual')
            if batch_input_type == "auto" or True:  # Per Zehra 2023-10-09, do this ALWAYS
                CA += float(Constants.get_batch_param(batch, "CA_offset"))

            self.update(status_label)

            if True:
                with secure_open(data_path, "r", "capture") as f:
                    csv_headers = next(f)

                    if isinstance(csv_headers, bytes):
                        csv_headers = csv_headers.decode()

                    if "Ambient" in csv_headers:
                        csv_cols = (2, 4, 6, 7)
                    else:
                        csv_cols = (2, 3, 5, 6)

                    data = loadtxt(f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols)

            self.update(status_label)

            relative_time = data[:, 0]
            temperature = data[:, 1]
            resonance_frequency = data[:, 2]
            dissipation = data[:, 3]

            self.update(status_label)

            # check for and remove time jumps that would break analysis
            t_last = 0
            rows_to_toss = []
            for x, t in enumerate(relative_time):
                if t < t_last:
                    rows_to_toss.append(x - 1)
                t_last = t
            if len(rows_to_toss) > 0:
                Log.w(f"Warning: time jump(s) observed at the following indices: {rows_to_toss}")
                relative_time = np.delete(relative_time, rows_to_toss)
                temperature = np.delete(temperature, rows_to_toss)
                resonance_frequency = np.delete(resonance_frequency, rows_to_toss)
                dissipation = np.delete(dissipation, rows_to_toss)
                Log.w(
                    "Time jumps removed from dataset for analysis purposes (original file unchanged)"
                )

            self.update(status_label)

            poi_path = os.path.join(os.path.split(data_path)[0], f"{data_title}_poi.csv")
            cal_path = os.path.join(os.path.split(data_path)[0], f"{data_title}_cal.csv")

            # NOTE: Temp CA offset removed from support as of 2025-03-17
            # # calculate and apply temperature adjusted contact angle offset
            # real_temps = [x for x in temperature if ~np.isnan(x)]
            # avg_run_temp = round(np.average(real_temps),
            #                      1) if len(real_temps) else 25.0
            # CA_temp_factor = round(
            #     (avg_run_temp - 25.0) * Constants.temp_adjusted_CA_factor, 1
            # )
            # Log.d(f"Applying temperature adjusted CA offset:")
            # Log.d(
            #     f"Temp CA offset = ({avg_run_temp}-25.0)*{Constants.temp_adjusted_CA_factor} = {CA_temp_factor}"
            # )
            # Log.d(
            #     f"Changing CA from {CA} to {CA + CA_temp_factor} with temperature offset {CA_temp_factor}"
            # )
            # CA += CA_temp_factor

            START_IDX = 0  # start-of-fill
            FILL_IDX = 1  # end-of-fill
            # NORMAL_PTS: 2-5 # 20%, 40%, 60%, 80%
            BLIP1_IDX = 6  # ch 1 fill
            # MIDP2_IDX = 6   # not used
            BLIP2_IDX = 7  # ch 2 fill
            # MIDP3_IDX = 8   # not used
            BLIP3_IDX = 8  # ch 3 fill

            # NOTE: start, eof, mid1, blip1, mid2, blip2, mid3, blip3
            # Support flexible array formatting in batch params lookup file:
            # [1.15, 1.61, 2.17, 2.67, 3.23, 5.00, 10.90, 16.2]  -or-
            # [1.15,1.61,2.17,2.67,3.23,5.00,10.90,16.2] -or-
            # [1.15 1.61 2.17 2.67 3.23 5.00 10.90 16.2]
            distances = str(Constants.get_batch_param(batch, "distances"))
            distances = (
                distances.replace("[", "").replace("]", "").replace(",", " ").replace("  ", " ")
            )  # remove array chars: '[],'
            distances = np.fromstring(
                distances, sep=" "
            ).tolist()  # convert string to numpy array and then to a list
            normal_pts = [0.2, 0.4, 0.6, 0.8]

            self.update(status_label)

            # Computes initial difference cancelations for difference, resonance frequency
            # and dissipation and applies them to the UI curves.
            canceled_diss, canceled_rf = None, None
            if self.parent.drop_effect_cancelation_checkbox.isChecked():
                canceled_diss, canceled_rf = self.parent._correct_drop_effect(
                    self.loaded_datapath, poi_vals, "worker"
                )
                if canceled_diss is not None:
                    dissipation = canceled_diss
                if canceled_rf is not None:
                    resonance_frequency = canceled_rf

            # raw data
            xs = relative_time
            ys = dissipation

            self.update(status_label)

            # use rough smoothing based on total runtime to figure start/stop
            total_runtime = xs[-1]
            smooth_factor = total_runtime * Constants.smooth_factor_ratio
            smooth_factor = int(smooth_factor) + (int(smooth_factor + 1) % 2)
            if smooth_factor < 3:
                smooth_factor = 3
            if smooth_factor > 69:
                smooth_factor = 69
            Log.i(TAG, f"Total run time: {total_runtime} secs")
            # the nearest odd number of seconds (runtime)
            Log.d(TAG, f"Smoothing: {smooth_factor}")
            Log.d(TAG, f"Applying smooth factor for first 90s ONLY.")

            t_first_90_split = (
                len(xs) if total_runtime <= 90 else next(x for x, t in enumerate(xs) if t > 90)
            )
            extend_data = True if total_runtime > 90 else False
            # downsample factor for extended data > 90s
            extend_smf = int(smooth_factor / 20)
            extend_smf += int(extend_smf + 1) % 2  # force to odd number

            if extend_data and len(xs) < t_first_90_split + 2 * extend_smf:
                Log.w(
                    "Not enough points after 90s to downsample effectively when analyzing. Not downsampling this dataset!"
                )
                t_first_90_split = len(xs)
                extend_data = False

            ys_fit = savgol_filter(ys[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_fit_ext = savgol_filter(
                    ys[t_first_90_split:],
                    min(len(ys[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_fit = np.concatenate((ys_fit, ys_fit_ext))

            ys_diss_diff = savgol_filter(ys_fit[:t_first_90_split], smooth_factor, 1, 1)
            if extend_data:
                ys_diss_diff_ext = savgol_filter(
                    ys_fit[t_first_90_split:],
                    min(len(ys_fit[t_first_90_split:]), extend_smf),
                    1,
                    1,
                )
                ys_diss_diff = np.concatenate((ys_diss_diff, ys_diss_diff_ext))

            ys_diss_2ndd = savgol_filter(ys_diss_diff[:t_first_90_split], smooth_factor, 1, 1)
            if extend_data:
                ys_diss_2ndd_ext = savgol_filter(
                    ys_diss_diff[t_first_90_split:],
                    min(len(ys_diss_diff[t_first_90_split:]), extend_smf),
                    1,
                    1,
                )
                ys_diss_2ndd = np.concatenate((ys_diss_2ndd, ys_diss_2ndd_ext))

            ys_diss_diff_avg = np.average(
                ys_diss_diff
            )  # AJR TODO 4/14: pick up here, this line is too high for the 109cp run
            ys_diss_diff_offset = ys_diss_diff - ys_diss_diff_avg
            zeros3 = np.where(np.diff(np.sign(ys_diss_diff_offset)))[0]
            while len(zeros3) < 2:
                zeros3 = np.append(zeros3, 100)
            ys_diss_diff_avg = np.average(ys_diss_diff[zeros3[1] :])

            self.update(status_label)

            minima_idx = argrelextrema(ys_diss_2ndd, np.less)[0]
            minima_val = ys_diss_2ndd[minima_idx]
            minima_dict = {minima_idx[i]: minima_val[i] for i in range(len(minima_idx))}
            minima_sort = sorted(minima_dict.items(), key=lambda kv: (kv[0]))

            maxima_idx = argrelextrema(ys_diss_2ndd, np.greater)[0]
            maxima_val = ys_diss_2ndd[maxima_idx]
            maxima_dict = {maxima_idx[i]: maxima_val[i] for i in range(len(maxima_idx))}
            maxima_sort = sorted(maxima_dict.items(), key=lambda kv: (kv[1], kv[0]))

            self.update(status_label)

            start_stop = sorted(maxima_sort[-2:])
            start_stop = [start_stop[0][0], start_stop[1][0]]
            t_start = np.amin(start_stop)
            t_stop = np.amax(start_stop) + (3 * smooth_factor)
            if t_stop < len(xs) / 2 or t_stop >= len(xs):
                Log.d(
                    f"Warning: t_stop was {t_stop} out of {len(xs)} but that was deemed too big/small! (This can usually be ignored.)"
                )
                t_stop = len(xs) - 1
            if t_stop - t_start < len(xs) / 3 or t_start > len(xs) / 2:
                Log.d(
                    f"Warning: t_start was {t_start} out of {len(xs)} but that was deemed too big/small! (This can usually be ignored.)"
                )
                t_start = 100

            if total_runtime < 3:
                Log.e("ERROR: Data run must be at least 3 seconds in total runtime to analyze.")
                return

            self.update(status_label)

            # get indices for 0.5 seconds to start of run
            t_0p5 = 0 if xs[t_start] < 0.5 else next((x for x, t in enumerate(xs) if t > 0.5), 0)
            t_1p0 = (
                t_start if xs[t_start] < 2.0 else next((x for x, t in enumerate(xs) if t > 2.0), 1)
            )
            if t_0p5 == t_1p0:
                t_1p0 = next((x for x, t in enumerate(xs) if t > xs[t_1p0] + 1.5), t_1p0 + 1)

            # new maths for resonance and dissipation (scaled)
            avg = np.average(resonance_frequency[t_0p5:t_1p0])
            ys = ys * avg / 2

            ys_fit = ys_fit * avg / 2
            ys = ys - np.amin(ys_fit)
            ys_fit = ys_fit - np.amin(ys_fit)
            ys_freq = avg - resonance_frequency

            ys_freq_fit = savgol_filter(ys_freq[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_freq_fit_ext = savgol_filter(
                    ys_freq[t_first_90_split:],
                    min(len(ys_freq[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_freq_fit = np.concatenate((ys_freq_fit, ys_freq_fit_ext))

            self.update(status_label)

            # # APPLY DROP EFFECT VECTORS
            # drop_offsets = np.zeros(ys.shape)
            # try:
            #     if self.parent.correct_drop_effect.isChecked():
            #         # baseline = np.average(ys[t_0p5:t_1p0])
            #         # base_std = np.std(ys[t_0p5:t_1p0])
            #         # next(x - 1 for x,y in enumerate(ys) if y > baseline + 4*base_std and x > t_1p0)
            #         drop_start = poi_vals[0]
            #         # next(ys[x + 2] for x,y in enumerate(ys) if y > Constants.drop_effect_cutoff_freq / 2 and x > t_1p0)
            #         drop_diss = ys[drop_start]
            #         if drop_diss > Constants.drop_effect_cutoff_freq:
            #             self.diff_factor = Constants.drop_effect_multiplier_high
            #         else:
            #             self.diff_factor = Constants.drop_effect_multiplier_low
            #         with open("QATCH/resources/lookup_drop_effect.csv", "r") as f:
            #             data = np.loadtxt(
            #                 f.readlines(), delimiter=",", skiprows=1)
            #             col = (
            #                 1
            #                 if self.diff_factor == Constants.drop_effect_multiplier_low
            #                 else 2
            #             )
            #             RR_offset = data[:, col]
            #             if drop_start + len(RR_offset) > len(drop_offsets):
            #                 # RR vector is longer than the actual run data, truncate it
            #                 drop_offsets[drop_start:] = RR_offset[
            #                     : len(drop_offsets) - drop_start
            #                 ]
            #             else:
            #                 # RR vector is shorter and needs to be padded with the final value
            #                 drop_offsets[drop_start: drop_start + len(RR_offset)] = (
            #                     RR_offset
            #                 )
            #                 drop_offsets[drop_start +
            #                              len(RR_offset):] = RR_offset[-1]
            #         Log.d(
            #             f"Applying vectors starting at time 't = {xs[drop_start]:1.3f}s'"
            #         )
            #         Log.d(
            #             f"Drop effect 'cutoff' dissipation frequency is {drop_diss:1.1f}Hz"
            #         )
            #         Log.d(
            #             f"Using {'low' if col == 1 else 'high'} viscosity drop effect 'diff_factor' and vector"
            #         )
            # except Exception as e:
            #     Log.e("ERROR:", e)

            # Automatically compute optimal difference factor
            if self.parent.difference_factor_optimizer_checkbox.isChecked():
                self.diff_factor = self.parent._optimize_curve(self.loaded_datapath)

            baseline = np.average(dissipation[t_0p5:t_1p0])
            diff_factor = Constants.default_diff_factor  # 1.0 if baseline < 50e-6 else 1.5
            if hasattr(self, "diff_factor"):
                diff_factor = self.diff_factor
            ys_diff = ys_freq - (diff_factor * ys)

            # Invert difference curve if drop applied to outlet
            if np.average(np.abs(ys_freq_fit)) < np.average(np.abs(diff_factor * ys_fit)) and abs(
                ys_diff[t_1p0:].min()
            ) > 5 * abs(ys_diff[t_1p0:].max()):
                Log.w("Inverting DIFFERENCE curve due to negative initial fill deltas")
                ys_diff *= -1

            ys_diff_fit = savgol_filter(ys_diff[:t_first_90_split], smooth_factor, 1)
            if extend_data:
                ys_diff_fit_ext = savgol_filter(
                    ys_diff[t_first_90_split:],
                    min(len(ys_diff[t_first_90_split:]), extend_smf),
                    1,
                )
                ys_diff_fit = np.concatenate((ys_diff_fit, ys_diff_fit_ext))
            Log.d(f"Difference factor: {diff_factor:1.3f}x")

            self.update(status_label)

            smf = max(3, int(smooth_factor / 10))
            if smf % 2 == 0:
                smf += 1  # force odd number
            ys_diff_fine = savgol_filter(ys_diff, smf, 1)
            ys_diff_diff = savgol_filter(
                ys_diff, smf, 1, 1
            )  # difference derivatives, not dissipation
            ys_diff_2ndd = savgol_filter(ys_diff_diff, smf, 1, 1)

            self.update(status_label)

            # plt.ion()
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(2, 3, (1, 3))
            ax2 = fig.add_subplot(234)
            ax3 = fig.add_subplot(235)
            ax4 = fig.add_subplot(236)
            ax.set_title(f"Confirm: {data_title}")

            self.update(status_label)

            mask = np.arange(0, len(xs), 1)

            ax.legend(["Resonance", "Difference", "Dissipation"])

            d_avg = np.average(ys_diff[t_0p5:t_1p0])
            d_max = np.amax(ys_diff[t_0p5:t_1p0])
            d_min = np.amin(ys_diff[t_0p5:t_1p0])
            envelope_size = int(d_max - d_min)

            start_stop.clear()

            self.update(status_label)

            if len(poi_vals) > 0:
                t0 = poi_vals[0]
            t0_was = t0
            cw = max(10, int(smooth_factor / 10))  # context width
            while confirm_startIndex:
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                ax2.plot(
                    xs[t0 - cw : t0 + cw],
                    ys_freq[t0 - cw : t0 + cw],
                    "g.",
                    label="freq",
                )
                ax2.scatter(xs[t0], ys_freq[t0], marker="*", s=75, c="black", zorder=10)
                ax3.plot(
                    xs[t0 - cw : t0 + cw],
                    ys_diff[t0 - cw : t0 + cw],
                    "b.",
                    label="diff",
                )
                ax3.scatter(xs[t0], ys_diff[t0], marker="*", s=75, c="black", zorder=10)
                ax4.plot(xs[t0 - cw : t0 + cw], ys[t0 - cw : t0 + cw], "r.", label="diss")
                ax4.scatter(xs[t0], ys[t0], marker="*", s=75, c="black", zorder=10)
                t0, done = QtWidgets.QInputDialog.getDouble(
                    None, "Input Dialog", "Confirm precise start index:", value=t0
                )
                if t0.is_integer() and int(t0) in [-1] + list(range(t0_was - cw, t0_was + cw)):
                    t0 = int(t0)
                else:
                    try:
                        t0 = next(x for x, t in enumerate(xs) if t > t0)
                    except StopIteration:
                        Log.d("Re-interpreting user input as an index, not a timestamp")
                        t0 = int(t0)
                if not done:
                    return
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                if t0_was == t0:
                    break
                t0_was = t0
            start_stop.append(t0)

            self.update(status_label)

            if len(poi_vals) > 1:
                t1 = poi_vals[1]
            t1_was = t1
            cw = max(10, int(smooth_factor / 2))  # context width
            while confirm_stopIndex:
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                ax2.plot(
                    xs[t1 - cw : t1 + cw],
                    ys_freq[t1 - cw : t1 + cw],
                    "g.",
                    label="freq",
                )
                ax2.scatter(xs[t1], ys_freq[t1], marker="*", s=75, c="black", zorder=10)
                ax3.plot(
                    xs[t1 - cw : t1 + cw],
                    ys_diff[t1 - cw : t1 + cw],
                    "b.",
                    label="diff",
                )
                ax3.scatter(xs[t1], ys_diff[t1], marker="*", s=75, c="black", zorder=10)
                ax4.plot(xs[t1 - cw : t1 + cw], ys[t1 - cw : t1 + cw], "r.", label="diss")
                ax4.scatter(xs[t1], ys[t1], marker="*", s=75, c="black", zorder=10)
                t1, done = QtWidgets.QInputDialog.getDouble(
                    None, "Input Dialog", "Confirm precise stop index:", value=t1
                )
                if t1.is_integer() and int(t1) in [-1] + list(range(t1_was - cw, t1_was + cw)):
                    t1 = int(t1)
                else:
                    try:
                        t1 = next(x for x, t in enumerate(xs) if t > t1)
                    except StopIteration:
                        Log.d("Re-interpreting user input as an index, not a timestamp")
                        t1 = int(t1)
                if not done:
                    return
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                if t1_was == t1:
                    break
                t1_was = t1
            start_stop.append(t1)

            self.update(status_label)

            tp = t1 + 2
            if len(poi_vals) > 2:
                tp = poi_vals[2]
            tp_was = tp
            while confirm_postIndex:
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                ax2.plot(
                    xs[tp - cw : tp + cw],
                    ys_freq[tp - cw : tp + cw],
                    "g.",
                    label="freq",
                )
                ax2.scatter(xs[tp], ys_freq[tp], marker="*", s=75, c="black", zorder=10)
                ax3.plot(
                    xs[tp - cw : tp + cw],
                    ys_diff[tp - cw : tp + cw],
                    "b.",
                    label="diff",
                )
                ax3.scatter(xs[tp], ys_diff[tp], marker="*", s=75, c="black", zorder=10)
                ax4.plot(xs[tp - cw : tp + cw], ys[tp - cw : tp + cw], "r.", label="diss")
                ax4.scatter(xs[tp], ys[tp], marker="*", s=75, c="black", zorder=10)
                tp, done = QtWidgets.QInputDialog.getDouble(
                    None, "Input Dialog", "Confirm precise post index:", value=tp
                )
                if tp.is_integer() and int(tp) in [-1] + list(range(tp_was - cw, tp_was + cw)):
                    tp = int(tp)
                else:
                    try:
                        tp = next(x for x, t in enumerate(xs) if t > tp)
                    except StopIteration:
                        Log.d("Re-interpreting user input as an index, not a timestamp")
                        tp = int(tp)
                if not done:
                    return
                ax2.cla()  # clear axis state without closing it
                ax3.cla()
                ax4.cla()
                if tp_was == tp:
                    break
                tp_was = tp

            self.update(status_label)

            # offset time by start point
            xs -= xs[t0]

            # zero all three datasets (and their fits) at start point
            ys_fit -= ys[t0]
            ys_freq_fit -= ys_freq[t0]
            ys_diff_fit -= ys_diff[t0]
            ys -= ys[t0]
            ys_freq -= ys_freq[t0]
            ys_diff -= ys_diff[t0]

            self.update(status_label)

            def monoLine(x, m, b):
                return m * x + b

            def monoCube(x, a, b, c):
                return a * (x**3) + b * (x**2) + c * x

            def monoCurve(x, a, b, c, d):
                return a * np.exp(b * x + c) + d

            # calculate normalized curve
            normal_x = xs[t0 : t1 + 1]
            normal_y = ys_freq[t0 : t1 + 1]
            n_max = np.amax(normal_y)
            n_min = np.amin(normal_y)

            self.update(status_label)

            if len(normal_y) <= 5:
                Log.w("Initial fill region contains too few points to apply smoothing.")

            sm1 = min(len(normal_y), max(5, int(len(normal_y) / 2)))
            if sm1 % 2 == 0:
                sm1 -= 1  # force odd number
            initial_fill = normal_y  # save for later plot
            initial_smooth = savgol_filter(initial_fill, sm1, 1) if sm1 > 1 else initial_fill

            # approximate linear fit
            n_slope = 1 / (normal_x[-1] - normal_x[0])
            n_offset = -n_slope * normal_x[0]

            self.update(status_label)

            p0 = (0, 0, n_slope)  # start with values near those we expect
            a, b, n_slope = p0  # default, not yet optimized
            best_fit_pts = normal_y  # default, not yet optimized
            try:
                fit_ignore = 0  # int((t1 - t0) / 4)
                params, cv = curve_fit(monoCube, normal_x[fit_ignore:], normal_y[fit_ignore:], p0)
                a, b, n_slope = params
                best_fit_pts = monoCube(normal_x, a, b, n_slope)
                Log.d(f"Normalized fit coeffs: {params}")
            except:
                Log.w(
                    'Curve fit 1 failed to find optimal parameters for Figure 1 "Normalized" curve.'
                )
                Log.w("Using raw points in place of fit line.")

            self.update(status_label)

            Df = n_max - n_min

            # normalize both raw data and best fit points
            y_max = np.amax(normal_y)
            y_offset = np.amin(normal_y)
            normal_y = (normal_y - y_offset) / (y_max - y_offset)
            y_max = np.amax(best_fit_pts)
            y_offset = np.amin(best_fit_pts)
            best_fit_pts = (best_fit_pts - y_offset) / (y_max - y_offset)

            #############################################################
            ### TODO: THIS IS A "BAND-AID" IMPLEMENTATION - REMOVE IT ###
            ### PURPOSE: APPLY POLYNOMIAL CORRECTION TO INITIAL FILL  ###
            ###          WHEN FILLING TIME IS GREATER THAN 1 SECOND.  ###
            ### DATE ADDED: 2024-01-14                                ###
            #############################################################
            enable_bandaid_code = True  # Use to disable modified behavior
            line1_x = normal_x
            t_filling = line1_x[-1]
            Log.i(f"t_filling = {t_filling} secs")
            if enable_bandaid_code and t_filling > 1.5:  # t_filling > 1 sec
                Log.w("Applying polynomial correction to initial fill region (for long runs)")
                line1_y = np.sqrt(np.polyval([0.1, 0.9, 0], normal_y)) * distances[0]
                line1_y_fit = np.sqrt(np.polyval([0.1, 0.9, 0], best_fit_pts)) * distances[0]
            else:
                line1_y = np.sqrt(normal_y) * distances[0]
                line1_y_fit = np.sqrt(best_fit_pts) * distances[0]
            line1_y[np.isnan(line1_y)] = 0
            line1_y_fit[np.isnan(line1_y_fit)] = 0
            ### END OF CODE BLOCK: "BAND-AID" IMPLEMENTATION ############

            self.update(status_label)

            line1_smooth = savgol_filter(line1_y, sm1, 3) if sm1 > 3 else line1_y
            line1_smooth[0] = 0  # force first value to zero
            mask = np.where(line1_smooth < 0)
            line1_smooth[mask] = 0

            self.update(status_label)

            # start with values near those we expect
            p0 = (-1, -1, 1, distances[0])
            a, b, c, d = p0  # default, not yet optimized
            line1_curve = line1_y  # default, not yet optimized
            try:
                fit_ignore = 0  # int((t1 - t0) / 4)
                params, cv = curve_fit(monoCurve, line1_x[fit_ignore:], line1_y[fit_ignore:], p0)
                a, b, c, d = params
                line1_curve = monoCurve(line1_x, a, b, c, d)
            except:
                Log.w(
                    'Curve fit 2 failed to find optimal parameters for Figure 1 "Position" curve.'
                )
                Log.w("Using raw points in place of fit line.")

            self.update(status_label)

            x_fit_pts = 0
            for x in range(len(line1_x)):
                if line1_curve[x] < line1_y[x]:
                    x_fit_pts = x + 1
                    break
            x_fit_val = np.linspace(0, line1_curve[x_fit_pts], x_fit_pts)
            for x in range(x_fit_pts - 1):
                line1_curve[x] = x_fit_val[x]

            self.update(status_label)

            # normalize endpoint to max of 1.15mm
            y_ratio = np.amax(line1_smooth) / distances[0]
            line1_y /= y_ratio
            line1_smooth /= y_ratio
            a_max = np.amax(line1_curve)
            line1_curve -= np.amin(line1_curve)
            line1_curve *= a_max / np.amax(line1_curve)
            y_ratio = np.amax(line1_curve) / distances[0]
            line1_curve /= y_ratio

            self.update(status_label)

            Log.d(TAG, f"Df = {Df}")
            Log.d(TAG, f"# pts = {t1+1-t0}")

            # search for locations of blips @ 5.6mm, 11.3mm, 15.7mm
            times = []

            # find zero2 crossings
            zeros2 = np.where(np.diff(np.sign(ys_diss_2ndd)))[0]

            # find zero3 crossings
            ys_diss_diff_offset = ys_diss_diff - ys_diss_diff_avg
            zeros3 = np.where(np.diff(np.sign(ys_diss_diff_offset)))[0]
            Log.d(zeros3)

            self.update(status_label)

            # define rough blip zones
            # t0 = next(t for t in zeros if t > t0) # first zero crossing to right of max value
            # t3r = t0 + np.argmin(ys_diss_diff[t0:])
            t0 = t1  # t1 is from different context, refers to end of initial fill period
            t3 = t_stop
            td = int((t3 - t0) / 3)
            t1 = t0 + td
            t2 = t1 + td
            t0r = float(t0)
            t1r = float(t1)
            t2r = float(t2)
            t3r = float(t3)

            self.update(status_label)

            # search for precise blips
            blips = [1, 2, 3]
            range_list = []
            # Log.d(zeros2)
            for key, val in maxima_sort[::-1]:  # iterate from big to small
                this_max = key
                try:
                    this_min = next(t for t, y in minima_sort if t > key)
                    this_zero = next(t for t in zeros2 if t > key)
                except StopIteration:
                    this_min = len(ys_diss_2ndd) - 1
                    this_zero = len(ys_diss_2ndd) - 2
                if this_max > this_zero or this_min < this_zero:
                    if ys_diss_2ndd[this_max] > 0 and ys_diss_2ndd[this_min] < 0:
                        # Log.w("Warning: ")
                        Log.d(
                            TAG,
                            "Something is off! The maxima MUST come before the minima, with the zero crossing in between. (This can usually be ignored.)",
                        )
                    continue
                x_range = this_min - this_max
                y_range = ys_diss_2ndd[this_max] - ys_diss_2ndd[this_min]
                # Log.d((this_max, this_zero, this_min), xs[key], ys_diss_2ndd[this_max], ys_diss_2ndd[this_min], y_range)
                zone = -1
                if key > t0r and key < t1r:
                    zone = 1
                if key > t1r and key < t2r:
                    zone = 2
                if key > t2r and key <= t3r:
                    zone = 3
                range_list.append((this_zero, zone, x_range, y_range))
            range_sort = sorted(range_list, key=lambda kv: (kv[3], kv[1]))

            self.update(status_label)

            # Log.d(range_sort)
            # iterate from big to small
            for key, zone, x, y in range_sort[::-1]:
                # Log.d(xs[key], zone, y)
                if zone in blips:
                    blips.remove(zone)
                    # Log.d("Using:", zone, key, xs[key])
                    if zone == 1:
                        t1 = key
                    if zone == 2:
                        t2 = key
                    if zone == 3:
                        t3 = key
                        t3r = min(len(xs) - 1, int(key + x))
                if len(blips) == 0:
                    break

            self.update(status_label)

            # overload blips for new method (test)
            idx = 0
            t_num = 0
            t_len = 0
            t_minima = []
            t_size = []
            while True:
                if len(zeros3) > idx + 1:
                    mid_val = ys_diss_diff_offset[int((zeros3[idx] + zeros3[idx + 1]) / 2)]
                    min_pt = zeros3[idx] + np.argmin(
                        ys_diss_diff_offset[zeros3[idx] : zeros3[idx + 1]]
                    )
                    this_size = zeros3[idx + 1] - zeros3[idx]
                    idx += 1
                    if mid_val > ys_diss_diff_offset[zeros3[idx]]:
                        continue  # skip if this interval is a maximum, not a minimum
                    t_minima.append(min_pt)
                    t_size.append(this_size)
                else:
                    break

            self.update(status_label)

            v_minima = []
            v_size = []
            for i in range(len(t_minima)):
                if t_size[i] > t_len:
                    v_minima.append(t_minima[i])
                    v_size.append(t_size[i])
                    t_len = t_size[i]

            self.update(status_label)

            if len(v_minima) == 3:
                t1 = int((v_minima[0] + v_minima[1]) / 2)
                t2 = v_minima[2]
            if len(v_minima) == 2:
                t1 = v_minima[0]
                t2 = v_minima[1]
            if len(v_minima) == 1:
                t1 = v_minima[0]
                t2 = t_minima[-1] + np.argmin(ys_diss_diff_offset[t_minima[-1] :])

            self.update(status_label)

            np.asarray(t_minima)

            start_idx = int(zeros3[0]) if len(zeros3) else 0
            if len(zeros3) == 0:
                Log.w("No zero-crossings found in ys_diss_diff_offset; plotting full range.")
            plot_len = min(len(xs), len(ys_diss_diff_offset))
            ax2.plot(xs[start_idx:plot_len], ys_diss_diff_offset[start_idx:plot_len], "b:")
            ax2.plot(xs[t_minima], ys_diss_diff_offset[t_minima], "rx")
            ax2.plot(xs[t1], ys_diss_diff_offset[t1], "gx")
            ax2.plot(xs[t2], ys_diss_diff_offset[t2], "gx")
            ax2.axhline(y=0)  # ys_diss_diff_avg)

            self.update(status_label)

            if len(poi_vals) > 5:
                # write prior blips
                times.append(poi_vals[3])
                times.append(poi_vals[4])
                times.append(poi_vals[5])
            else:
                # write found blips
                times.append(t1)
                times.append(t2)
                times.append(t3r)
            Log.d(times)

            self.update(status_label)

            bounds = [int(t0r), int(t1r), int(t2r), int(t3r)]
            for b in range(len(times)):
                # confirm blips (if desired)
                time = times[b]
                time_was = time
                cw = max(10, int(smooth_factor * 3))  # context width
                while confirm_blipIndices:
                    ax.cla()  # clear axis state without closing it
                    num_points = int((bounds[b + 1] - bounds[b]) / 2)
                    mask = np.arange(
                        max(0, time - num_points),
                        min(len(xs) - 1, time + num_points),
                        1,
                    ).astype(
                        int
                    )  # keep centered in wide-context window
                    time = int(time)
                    Log.d(mask)
                    ax.plot(xs[mask], ys_freq_fit[mask], ":", color="green", label="fit")
                    ax.plot(xs[mask], ys_diff_fit[mask], ":", color="blue", label="fit")
                    ax.plot(xs[mask], ys_fit[mask], ":", color="red", label="fit")
                    ax.plot(xs[mask], ys_freq[mask], "g,", label="freq")
                    ax.scatter(
                        xs[time],
                        ys_freq_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax.plot(xs[mask], ys_diff[mask], "b,", label="diff")
                    ax.scatter(
                        xs[time],
                        ys_diff_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax.plot(xs[mask], ys[mask], "r,", label="diss")
                    ax.scatter(xs[time], ys_fit[time], marker="*", s=75, c="black", zorder=10)
                    ax.legend(["Resonance", "Difference", "Dissipation"])
                    ax2.cla()  # clear axis state without closing it
                    ax3.cla()
                    ax4.cla()
                    ax2.plot(
                        xs[time - cw : time + cw],
                        ys_freq[time - cw : time + cw],
                        "g,",
                        label="freq",
                    )
                    ax2.plot(
                        xs[time - cw : time + cw],
                        ys_freq_fit[time - cw : time + cw],
                        "g.",
                        label="freq",
                    )
                    ax2.scatter(
                        xs[time],
                        ys_freq_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax3.plot(
                        xs[time - cw : time + cw],
                        ys_diff[time - cw : time + cw],
                        "b,",
                        label="diff",
                    )
                    ax3.plot(
                        xs[time - cw : time + cw],
                        ys_diff_fit[time - cw : time + cw],
                        "b.",
                        label="diff",
                    )
                    ax3.scatter(
                        xs[time],
                        ys_diff_fit[time],
                        marker="*",
                        s=75,
                        c="black",
                        zorder=10,
                    )
                    ax4.plot(
                        xs[time - cw : time + cw],
                        ys[time - cw : time + cw],
                        "r,",
                        label="diss",
                    )
                    ax4.plot(
                        xs[time - cw : time + cw],
                        ys_fit[time - cw : time + cw],
                        "r.",
                        label="diss",
                    )
                    ax4.scatter(xs[time], ys_fit[time], marker="*", s=75, c="black", zorder=10)
                    time, done = QtWidgets.QInputDialog.getDouble(
                        None,
                        "Input Dialog",
                        f"Confirm precise blip_{b+1} index:",
                        value=time,
                    )
                    if time.is_integer():
                        time = int(time)
                    else:
                        try:
                            time = next(x for x, t in enumerate(xs) if t > time)
                        except StopIteration:
                            Log.d("Re-interpreting user input as an index, not a timestamp")
                            time = int(time)
                    if not done:
                        return
                    ax2.cla()  # clear axis state without closing it
                    ax3.cla()
                    ax4.cla()
                    if time_was == time:
                        break
                    time_was = time
                times[b] = time

                self.update(status_label)

            # display and export selected points
            points_of_interest = np.concatenate((start_stop, [tp], times))
            Log.d(f"POIs (index only): {points_of_interest}")
            Log.i(f"POIs: {points_of_interest} {xs[points_of_interest]}")
            np.savetxt(poi_path, points_of_interest, fmt="%i")

            self.update(status_label)

            # pop blips if user input -1
            while True:
                if times[-1] == -1:
                    times.pop(-1)
                else:
                    break

            # Log.d(times)
            # Log.d("times:", xs[times]) #-xs[times[0]])

            self.update(status_label)

            t_start = max(0, np.amin(start_stop) - 50)
            t_stop = min(len(xs), np.amax(times) + 50)

            self.update(status_label)

            ax.cla()  # clear axis state without closing it
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax.set_title(f"Raw Data: {data_title}")
            ax.grid(axis="y", which="major")
            mask = np.arange(t_start, t_stop, 1)
            ax.plot(xs[mask], ys_freq_fit[mask], "--", color="green", label="freq fit")
            ax.plot(xs[mask], ys_diff_fit[mask], "--", color="blue", label="diff fit")
            ax.plot(xs[mask], ys_fit[mask], "--", color="red", label="diss fit")

            self.update(status_label)

            ax.plot(xs[mask], ys_freq[mask], "g,", label="freq")
            ax.plot(xs[mask], ys[mask], "r,", label="diss")
            ax.plot(xs[mask], ys_diff[mask], "b,", label="diff")

            # ax.plot(xs[mask], ys_diff_fine[mask], ":", color="blue", label="diff fine")
            # ax.plot(xs[mask], ys_diff_diff[mask], ':', color="orange", label="diss diff")
            # ax.plot(xs[mask], ys_diff_2ndd[mask], ':', color="red", label="freq fit")
            # ax.plot([0, xs[times[-1]]], [0, 0], "-", color="black", markersize=0)

            ax.plot(xs[start_stop], ys_diff[start_stop], "d", color="black")
            ax.plot(xs[times], ys_fit[times], "d", color="black")
            ax.legend(["Resonance", "Difference", "Dissipation"])

            ax2.plot(normal_x, initial_fill, "r.", label="init")
            ax2.plot(normal_x, initial_smooth, "-", label="fit")
            leg = ax2.legend(["Initial Fill"], handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legend_handles:
                item.set_visible(False)

            self.update(status_label)

            mask = np.where(normal_y >= 0)
            ax3.plot(normal_x[mask], normal_y[mask], "r.", label="normal")
            ax3.plot(normal_x, best_fit_pts, "-", label="fit")
            leg = ax3.legend(["Normalized"], handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legend_handles:
                item.set_visible(False)

            mask = np.where(line1_y >= 0)
            ax4.plot(line1_x[mask], line1_y[mask], "r.", label="line1")
            ax4.plot(line1_x, line1_y_fit, "-", label="curve")
            # ax4.plot(line1_x, line1_smooth, ':', label="fit")
            leg = ax4.legend(["Position"], handlelength=0, handletextpad=0, fancybox=True)
            for item in leg.legend_handles:
                item.set_visible(False)

            self.update(status_label)

            # show final constructed distance vs time curve
            times.append(t0)  # overloaded: end of inital fill
            times.sort()

            # insert midpoints into "times" array (to match length of "distances" array)
            Log.i("(The following midpoints are shown as blue Xs on Figure 1):")

            if len(times) >= 2:
                # NOTE: Channel 1 midpoint uses dissipation deliberately
                #       whereas 2nd and 3rd channels are frequency derived
                midpoint_ch1_y = (ys_fit[times[1]] + ys_fit[times[0]]) / 2
                midpoint_ch1_i = next(
                    (x for x, y in enumerate(ys_fit) if y > midpoint_ch1_y),
                    int(np.average([times[0:2]])),
                )
                midpoint_ch1_x = xs[midpoint_ch1_i]
                Log.i(
                    f"1st channel dissipation midpoint = {midpoint_ch1_y:2.2f} Hz @ {midpoint_ch1_x:2.2f} secs"
                )
                # ax.plot(midpoint_ch1_x, midpoint_ch1_y, "bd")
                # times.append(midpoint_ch1_i)
            else:
                Log.w(
                    "1st channel midpoint not available from dataset. Confirm Precise Fill Points 3 and 4 for accuracy."
                )
                Log.w(
                    "See Figure 2 to check if one of these points is being dropped due to time delta not being 2x last."
                )

            self.update(status_label)

            if len(times) >= 4:
                midpoint_ch2_y = (ys_freq_fit[times[2]] + ys_freq_fit[times[1]]) / 2
                midpoint_ch2_i = next(
                    (x for x, y in enumerate(ys_freq_fit) if y > midpoint_ch2_y),
                    int(np.average([times[1:3]])),
                )
                midpoint_ch2_x = xs[midpoint_ch2_i]
                Log.i(
                    f"2nd channel frequency midpoint = {midpoint_ch2_y:2.2f} Hz @ {midpoint_ch2_x:2.2f} secs"
                )
                # ax.plot(midpoint_ch2_x, midpoint_ch2_y, "bd")  # MIDP2
                # times.append(midpoint_ch2_i)                   # mid2
            else:
                Log.w(
                    "2nd channel midpoint not available from dataset. Confirm Precise Fill Points 4 and 5 for accuracy."
                )
                Log.w(
                    "See Figure 2 to check if one of these points is being dropped due to time delta not being 2x last."
                )

            self.update(status_label)

            if len(times) >= 6:
                midpoint_ch3_y = (ys_freq_fit[times[3]] + ys_freq_fit[times[2]]) / 2
                midpoint_ch3_i = next(
                    (x for x, y in enumerate(ys_freq_fit) if y > midpoint_ch3_y),
                    int(np.average([times[2:4]])),
                )
                midpoint_ch3_x = xs[midpoint_ch3_i]
                Log.i(
                    f"3rd channel frequency midpoint = {midpoint_ch3_y:2.2f} Hz @ {midpoint_ch3_x:2.2f} secs"
                )
                # ax.plot(midpoint_ch3_x, midpoint_ch3_y, "bd")  # MIDP3
                # times.append(midpoint_ch3_i)                   # mid3
            else:
                Log.w(
                    "3rd channel midpoint not available from dataset. Confirm Precise Fill Points 5 and 6 for accuracy."
                )
                Log.w(
                    "See Figure 2 to check if one of these points is being dropped due to time delta not being 2x last."
                )

            self.update(status_label)

            # downsample factor for extended data > 90s
            fine_smf = int(smooth_factor / 2.5)
            fine_smf += int(fine_smf + 1) % 2  # force to odd number
            ys_fit_fine = savgol_filter(ys, fine_smf, 1)

            # fine fit points: smoothed, but only a little for accurate fast fill
            ys_normal = ys_fit_fine - ys_fit_fine[tp]
            ys_normal /= ys_fit_fine[times[1]] - ys_fit_fine[tp]
            ys_normal = ys_normal[tp : times[1]]

            debug = False
            if debug:
                import matplotlib.pyplot as plt

                fig_dbg = plt.figure(figsize=(12, 9))
                ax_dbg = fig_dbg.add_subplot(111)

                # raw points
                xs_dbg = xs[tp : times[1]]
                ys_normal2 = ys - ys_fit_fine[tp]
                ys_normal2 /= ys_fit_fine[times[1]] - ys_fit_fine[tp]
                ys_normal2 = ys_normal2[tp : times[1]]

                Log.i("ys_normal: ")
                Log.i(ys_normal)
                ax_dbg.plot(xs_dbg, ys_normal, color="red", marker=",")
                ax_dbg.plot(xs_dbg, ys_normal2, color="green", marker=",")
                # ax_dbg.plot(xs, ys_fit_fine, color="blue", marker=",")

                Log.i("times:")
                Log.i(times)
                Log.i(xs[times])
                ax_dbg.plot(xs[times[0]], 0, color="black", marker="X")
                ax_dbg.plot(xs[times[1]], 1, color="black", marker="X")

                fig_dbg.show()

            idx_of_normal_pts_to_remove = []
            idx_of_normal_pts_to_retain = []
            for p in normal_pts:
                try:
                    midpoint_p_i = next(x for x, y in enumerate(ys_normal) if y >= p) + tp
                except StopIteration:
                    Log.w(f"Failed to find 1st channel dissipation @ {p:0.1f}")
                    continue
                midpoint_p_x = xs[midpoint_p_i]
                midpoint_p_y = ys_fit[midpoint_p_i]
                Log.i(
                    f"1st channel dissipation @ {p:0.1f} = {midpoint_p_y:2.2f} Hz @ {midpoint_p_x:2.2f} secs"
                )
                ax.plot(midpoint_p_x, midpoint_p_y, color="blue", marker="d", markersize=4)
                if debug:
                    ax_dbg.plot(midpoint_p_x, p, color="blue", marker="X")
                times.append(midpoint_p_i)
                # Issue #392: Remove 80% too; only 60% used (if not too off)
                if p == 0.2 or p == 0.4 or p == 0.8:
                    idx_of_normal_pts_to_remove.append(midpoint_p_i)
                else:
                    idx_of_normal_pts_to_retain.append(midpoint_p_i)
            times.sort()  # sort again, so midpoints are in proper order

            self.update(status_label)

            last_window_size = 0
            last_x = 0
            bad_idx = []
            bad_times = []
            bad_distances = []
            for x in range(1, len(times)):
                this_window_size = xs[times[x]] - xs[times[last_x]]
                # Log.e(f"Compare {times[x]} to {len(xs)-1}...")
                if this_window_size < 0.75 * last_window_size or times[x] == len(xs) - 1:
                    bad_x = x
                    if bad_x == 5:  # trust channel 1 pt more than estimated 80% point
                        bad_x = 4
                    Log.w(f"Point {bad_x} @ {xs[times[bad_x]]}s is 'bad' and will be ignored!")
                    bad_idx.append(bad_x)
                    bad_times.append(times[bad_x])
                    bad_distances.append(distances[bad_x])
                if x == 5:  # set ch1 window size compared to start, not estimated points
                    this_window_size = xs[times[x]] - xs[times[0]]
                last_window_size = this_window_size
                last_x = x

            self.update(status_label)

            # hide bad points from Figure 3 and Figure 4
            for x in bad_idx[::-1]:
                times.pop(x)
                distances.pop(x)

            np.asarray(times)
            np.asarray(bad_times)
            distances = np.asarray(distances)[0 : len(times)]

            self.update(status_label)

            ext_line1_x = np.linspace(0, xs[times[-1]], 1000)
            ext_index = np.concatenate(([start_stop[0]], times))
            ext_times = np.concatenate(([0], xs[times]))
            ext_dists = np.concatenate(([0], distances))
            all_times = np.sort(np.concatenate([[points_of_interest[0]], times, bad_times])).astype(
                int
            )
            Log.i("times and distances:")
            Log.d("indexes: {}".format(ext_index))
            Log.i(ext_times)
            Log.i(ext_dists)
            ext_line1_curve = np.interp(ext_line1_x, ext_times, ext_dists)

            self.update(status_label)

            fig2 = plt.figure(figsize=(12, 6))
            ax5 = fig2.add_subplot(111)
            ax5.plot(line1_x[0], 0, "d", color="black")
            ax5.plot(ext_line1_x, ext_line1_curve, ":", color="orange")
            for i in range(len(distances)):
                if not distances[i] in bad_distances:
                    ax5.plot(xs[times[i]], distances[i], "d", color="black")
            ax5.plot(xs[bad_times], bad_distances, "x", color="red")
            ax5.set_title(f"Position: {data_title}")
            ax5.set_xlabel("Time (s)")
            ax5.set_ylabel("Position (mm)")

            self.update(status_label)

            norm_fit_xs = xs[start_stop[1] : times[-1]]
            norm_fit_dists = ys_fit[start_stop[1] : times[-1]]
            norm_fit_dists -= norm_fit_dists[0]
            norm_fit_dists /= norm_fit_dists[-1]
            norm_fit_dists *= distances[-1] - distances[0]
            norm_fit_dists += distances[0]

            # Generate log() plot curves for velocity and position^-1
            log_velocity = np.concatenate((line1_y / line1_x, distances / xs[times]))
            log_position = np.concatenate((1 / line1_y, 1 / distances))

            raw_velocity = norm_fit_dists / norm_fit_xs
            raw_position = 1 / norm_fit_dists

            self.update(status_label)

            log_velocity = np.log10(log_velocity)
            log_position = np.log10(log_position)

            raw_velocity = np.log10(raw_velocity)
            raw_position = np.log10(raw_position)

            self.update(status_label)

            log_velocity[np.isnan(log_velocity)] = 0
            log_position[np.isnan(log_position)] = 0

            raw_velocity[np.isnan(raw_velocity)] = 0
            raw_position[np.isnan(raw_position)] = 0

            log_velocity[np.isinf(log_velocity)] = 0
            log_position[np.isinf(log_position)] = 0

            raw_velocity[np.isinf(raw_velocity)] = 0
            raw_position[np.isinf(raw_position)] = 0

            # fit_ignore = next(x for x,y in enumerate(log_velocity) if y > 0)
            # log_velocity = log_velocity[fit_ignore:]
            # log_position = log_position[fit_ignore:]

            self.update(status_label)

            # Parallel view of initial_fill that will undergo the same filter pipeline
            # as the initial-fill portion of log_velocity / log_velocity_46.
            initial_fill_tracked = np.asarray(initial_fill, dtype=float).copy()
            dropUnder2 = 0
            ####################################
            # NEW CODE for 2023-11-07 TESTING:
            # Drop 5 Hz or 3.33% of initial fill (whichever is larger)
            # default, if/when Band-Aid #2 disabled
            # dropBelowPct = float(1 / 30)
            #######################################
            # Band-Aid #2: Drop more initial fill
            # To disable: Comment out line below:
            # dropBelowPct = 0.10  # 0.15 if BIOFORMULATION else 0.40
            # Log.w(
            #     f"Dropping {int(dropBelowPct * 100)}% of initial fill region...")
            ### END Band-Aid #2 ###################
            dropFreqBelow = 2  # max(5, initial_fill[-1] * dropBelowPct)
            ### END NEW CODE ###################
            for i in range(len(initial_fill)):
                if initial_fill[i] > dropFreqBelow:
                    Log.d(f"Dropped {i} initial samples under 2 Hz threshold")
                    log_velocity = log_velocity[i:]
                    log_position = log_position[i:]
                    initial_fill_tracked = initial_fill_tracked[i:]
                    dropUnder2 = i
                    ####################################
                    # NEW CODE for 02/03/2023 TESTING:
                    # REMOVED 2023-11-07:
                    # if initial_fill[-1] > 1.1*DENSITY*300:
                    #     dropUnder2 = int(np.floor(len(initial_fill)/5))
                    # else:
                    #     dropUnder2 = i
                    ### END NEW CODE ###################
                    Log.d(f"dropUnder2 = {dropUnder2}")
                    break

            self.update(status_label)

            log_velocity_46 = log_velocity
            log_position_46 = log_position
            Log.d(f"log_velocity = {log_velocity}")
            Log.d(f"initial_fill_tracked = {initial_fill_tracked}")
            if initial_fill_tracked[-1] < 90:
                log_velocity_46 = log_velocity_46[-len(distances) :]
                log_position_46 = log_position_46[-len(distances) :]
                cp = xs[all_times[FILL_IDX]] / 2
                ax2.annotate("Not Analyzed", (cp, 0), ha="center")
                ax3.annotate("Not Analyzed", (cp, 0), ha="center")
                ax4.annotate("Not Analyzed", (cp, 0), ha="center")
            else:
                drop_46 = 0
                for i in range(len(initial_fill_tracked)):
                    if initial_fill_tracked[i] > 46:
                        drop_46 = i
                        Log.d(f"Dropped {i} initial samples under 46 Hz threshold.")
                        break

                # Never trim into the distances region. The subsequent
                # np.delete calls assume at least len(distances) samples remain.
                max_drop = max(0, len(log_velocity_46) - len(distances))
                if drop_46 > max_drop:
                    Log.w(
                        f"46 Hz threshold at tracked index {drop_46} would leave "
                        f"fewer than len(distances)={len(distances)} samples; "
                        f"clamping to {max_drop}"
                    )
                    drop_46 = max_drop

                log_velocity_46 = log_velocity_46[drop_46:]
                log_position_46 = log_position_46[drop_46:]

            self.update(status_label)

            def reject_outliers(data, m):
                d = np.abs(data - np.median(data))
                mdev = np.median(d)
                s = d / mdev if mdev else 0.0
                # Log.d(f"reject_s = {s}")
                return s < m

            self.update(status_label)

            keep_ids = reject_outliers(log_velocity[0 : -len(distances)], 11.0)
            if isinstance(keep_ids, bool):
                keep_ids = [keep_ids]
            trues = [True for x in distances]
            keep_ids = np.concatenate((keep_ids, trues))
            if len(keep_ids) != len(log_velocity):
                Log.w("Mismatched array lengths when rejecting initial fill outliers!")
                while len(keep_ids) < len(log_velocity):
                    keep_ids = np.concatenate((keep_ids, [True]))  # lengthen, if needed
                keep_ids = keep_ids[: len(log_velocity)].astype("bool")  # shorten, if needed
            log_velocity_skip = log_velocity[~keep_ids]
            log_position_skip = log_position[~keep_ids]
            log_velocity = log_velocity[keep_ids]
            log_position = log_position[keep_ids]
            # Apply the same filter to the tracked initial-fill view. keep_ids is sized
            # to log_velocity (initial fill + distances)
            initial_fill_keep = keep_ids[: len(initial_fill_tracked)]
            initial_fill_tracked = initial_fill_tracked[initial_fill_keep]
            Log.d(
                f"Rejected {len(log_position_skip)} samples as initial outliers: {log_position_skip}"
            )

            self.update(status_label)

            # See issue #410: Reduce Initial Fill Point Weighting in n_coeff Calculation
            end_fill_idx = max(0, len(log_velocity_46) - len(distances))
            fill_velocity = log_velocity_46[:end_fill_idx]
            fill_position = log_position_46[:end_fill_idx]
            best_fit_idx = []
            best_fit_pts = []
            try:
                if len(fill_velocity) and len(fill_position):
                    # Shown as black squares on "velocity vs position" plot
                    num_fill_pts = min(5, len(fill_velocity))
                    target_offsets = np.linspace(
                        1, len(fill_velocity), num_fill_pts, endpoint=False
                    )
                    selected_offsets = np.unique(
                        np.clip(np.rint(target_offsets).astype(int) - 1, 0, len(fill_velocity) - 1)
                    )
                    selected_offsets.sort()
                    best_fit_idx.extend(fill_velocity[selected_offsets].tolist())
                    best_fit_pts.extend(fill_position[selected_offsets].tolist())
                best_fill_idx = np.asarray(best_fit_idx, dtype=float)  # copy for later plotting
                best_fill_pts = np.asarray(best_fit_pts, dtype=float)  # copy for later plotting
                best_fit_idx.extend(log_velocity_46[end_fill_idx:])
                best_fit_pts.extend(log_position_46[end_fill_idx:])
                # kludgy code to remove the 20% and 40% fill points from fit
                best_fit_idx = np.delete(best_fit_idx, len(best_fill_idx) + 2)
                best_fit_idx = np.delete(best_fit_idx, len(best_fill_idx) + 1)
                best_fit_pts = np.delete(best_fit_pts, len(best_fill_pts) + 2)
                best_fit_pts = np.delete(best_fit_pts, len(best_fill_pts) + 1)
            except Exception as e:
                Log.e("An error occurred while finding the initial fill points median value")
                best_fit_idx = log_velocity_46
                best_fit_pts = log_position_46

            p0 = (1, 0)  # start with values near those we expect
            n_slope, n_offset = p0  # default, not yet optimized
            try:
                params, cv = curve_fit(monoLine, best_fit_idx, best_fit_pts, p0)
                n_slope, n_offset = params
                best_fit_pts = monoLine(log_velocity_46, n_slope, n_offset)
            except:
                Log.w('Curve fit 3 failed to find optimal parameters for Figure 3 "slope" fit.')
                Log.w('Using raw points in place of fit line (assuming "slope = 1").')
                best_fit_pts = log_position_46

            n_rounded = max(
                0.05, min(1, round((n_slope + 0.05) * 20) / 20)
            )  # round up to nearest 0.05, max of 1
            n = n_rounded

            self.update(status_label)

            ####################################
            # NEW CODE for 2022-12-06 TESTING:
            # ADDED `try-except` protection on 2025-02-17
            # Initialize result velocity and position lists.
            log_velocity_20p = []
            log_position_20p = []
            try:
                # Ensure that log_velocity_46 and log_position_46 have the same length
                if len(log_velocity_46) != len(log_position_46):
                    raise ValueError("log_velocity_46 and log_position_46 must be the same length.")

                # Compute m based on ==> m = |initial_fill| - |log_velocity| - |lov_velocity_46|.
                m = len(initial_fill) - (len(log_velocity) - len(log_velocity_46))
                if m < 5:
                    raise ValueError(
                        "Not enough valid data points based on initial_fill and log_velocity lengths."
                    )

                # Calculate the chunk length (mlen) and ensure it is positive.
                mlen = int(np.floor(m / 5))
                if mlen <= 0:
                    raise ValueError("Calculated mlen is not positive. Check input lengths.")

                # Ensure that the maximum index needed is within bounds.
                max_required_index = 5 * mlen  # roughly the maximum index accessed
                if max_required_index > len(log_velocity_46):
                    raise ValueError("Not enough entries in log_velocity_46 for the computed mlen.")

                # Process the first mlen elements.
                for hh in range(mlen):
                    if hh >= len(log_velocity_46) or hh >= len(log_position_46):
                        raise IndexError(
                            f"Index {hh} out of range for log_velocity_46 or log_position_46."
                        )
                    log_velocity_20p.append(log_velocity_46[hh])
                    log_position_20p.append(log_position_46[hh])

                # Process the remaining four chunks using slicing and averaging.
                for hh in range(1, 4):
                    start_index = hh * mlen
                    end_index = (hh + 1) * mlen - 1

                    # Clamp the end_index to the length of the list if necessary.
                    if end_index > len(log_velocity_46):
                        end_index = len(log_velocity_46)
                    if end_index > len(log_position_46):
                        end_index = len(log_position_46)

                    # Create velocity & position slices.
                    slice_vel = log_velocity_46[start_index:end_index]
                    slice_pos = log_position_46[start_index:end_index]

                    # Check that the velocity and position slices are not empty.
                    if len(slice_vel) == 0 or len(slice_pos) == 0:
                        raise ValueError(
                            f"Slice from {start_index} to {end_index} resulted in an empty list."
                        )

                    # Compute the averages of velocity and position slices.
                    avg_vel = np.average(slice_vel)
                    avg_pos = np.average(slice_pos)

                    log_velocity_20p.append(avg_vel)
                    log_position_20p.append(avg_pos)
            except Exception as e:
                Log.w(TAG, f"Bad initial fill region, skipping analysis of this reigon.")
                Log.d(TAG, f"With error: {e}")
            ### END NEW CODE ###################

            self.update(status_label)

            fig3 = plt.figure(figsize=(12, 6))
            ax6 = fig3.add_subplot(111)
            ax6.plot(log_velocity_20p, log_position_20p, ".", color="red")
            ax6.plot(log_velocity_46, log_position_46, ":", color="orange")
            ax6.plot(log_velocity_46, best_fit_pts, "-", color="blue")
            ax6.plot(best_fill_idx, best_fill_pts, "s", color="black")  # initial fill (avg)
            try:
                for i in range(-len(distances), 0):
                    ax6.plot(log_velocity_46[i], log_position_46[i], "d", color="black")
                # mark the 20% and 40% points as not being included in the fit approximation
                ax6.plot(
                    log_velocity_46[end_fill_idx + 2],
                    log_position_46[end_fill_idx + 2],
                    "x",
                    color="red",
                )
                ax6.plot(
                    log_velocity_46[end_fill_idx + 1],
                    log_position_46[end_fill_idx + 1],
                    "x",
                    color="red",
                )
                ax6.set_title(
                    f"Power log coefficient: {data_title}\nn = {n:.2f}" + r"$ \pm $" + "0.05"
                )
            except:
                Log.e(TAG, "An error occurred while annotating Figure 3")
            ax6.set_xlabel("Log(velocity) (mm/s)")
            ax6.set_ylabel("Log(1/position) (1/mm)")

            self.update(status_label)

            Log.d("the distances were:", distances)
            Log.d("the times were:", times)
            Log.d("the times to remove are:", idx_of_normal_pts_to_remove)
            for i in idx_of_normal_pts_to_remove:
                try:
                    if i not in times:
                        Log.w(f"Midpoint @ {i} already removed, skipping removal of bad point...")
                        continue
                    idx = times.index(i)
                    Log.d(f"Removing index {idx} from distances with value {distances[idx]}.")
                    distances = np.delete(distances, idx)
                    Log.d(f"Removing index {idx} from times with value {i}.")
                    times.remove(i)
                except Exception as e:
                    Log.e("Error removing midpoint from dataset:", str(e))
            Log.d("the distances are now:", distances)
            Log.d("the times are now:", times)

            all_pos = np.concatenate((line1_y[dropUnder2:], distances))
            all_time = np.concatenate((line1_x[dropUnder2:], xs[times]))

            lb = t0
            ub = t0 + len(line1_x[dropUnder2:])
            # safety check: prevent upper bound larger than array end
            if ub > len(temperature):
                overflow = ub - len(temperature)
                lb = max(0, lb - overflow)
                ub = len(temperature)
            # safety check: prevent lower bound less than array start
            if lb < 0:
                lb = 0
                ub = min(len(temperature), len(line1_x[dropUnder2:]))

            all_temp = np.concatenate((temperature[lb:ub], temperature[times]))

            len_pos = len(all_pos)
            len_time = len(all_time)
            len_temp = len(all_temp)
            if len_pos == len_time == len_temp:
                Log.d("CHECK PASS: ALL arrays are the same length!")
            else:
                Log.w("CHECK FAIL: ALL arrays are different lengths. Truncating to shortest one.")
                len_req = min(len_pos, len_time, len_temp)
                if len_pos != len_req:
                    Log.w(f"Array `all_pos` resized from {len_pos} to {len_req}")
                    all_pos = all_pos[:len_req]
                if len_time != len_req:
                    Log.w(f"Array `all_time` resized from {len_time} to {len_req}")
                    all_time = all_time[:len_req]
                if len_temp != len_req:
                    Log.w(f"Array `all_temp` resized from {len_temp} to {len_req}")
                    all_temp = all_temp[:len_req]

            avg_temp = np.average(temperature[t0 : times[-1]])
            all_velocity = all_pos / all_time
            # all_velocity[-7] /= 2 # 1.61 (#2 in distances)
            # all_velocity[-6] /= 1.5 # 2.17 (#3 in distances)
            # all_velocity[-5] /= 1 # 2.67 (#4 in distances)

            fill_pos = line1_y_fit[dropUnder2:]
            fill_time = line1_x[dropUnder2:]
            fill_velocity = fill_pos / fill_time

            self.update(status_label)

            Log.d(f"Channel thickness = {Constants.channel_thickness}")
            viscosity = (
                ST
                * np.cos(np.radians(CA))
                * all_time
                * Constants.channel_thickness
                / 6
                / (all_pos**2)
                * 1e6
                * (3 * (n + 1) / (2 * n + 1))
            )
            shear_rate = (
                6
                * all_velocity
                / Constants.channel_thickness
                * (2 / 3 + 1 / 3 / n)
                * 1e-3
                / (n + 1)
                * n
            )

            fill_visc = (
                ST
                * np.cos(np.radians(CA))
                * fill_time
                * Constants.channel_thickness
                / 6
                / (fill_pos**2)
                * 1e6
                * (3 * (n + 1) / (2 * n + 1))
            )
            fill_shear = (
                6
                * fill_velocity
                / Constants.channel_thickness
                * (2 / 3 + 1 / 3 / n)
                * 1e-3
                / (n + 1)
                * n
            )

            self.update(status_label)

            fig4 = plt.figure(figsize=(12, 6))
            fig4.set_layout_engine(None)  # full control, no auto-layout adjustments
            fig4.subplots_adjust(
                left=0.10, right=0.99, top=0.92, bottom=0.22, wspace=0.0, hspace=0.0
            )
            ax7 = fig4.add_subplot(111)

            # Create the annotation object (hidden by default)
            self.annot = ax7.annotate(
                "",
                xy=(0, 0),
                xytext=(-40, 20),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
                picker=True,
            )
            self.annot.set_visible(False)
            self.annot.get_bbox_patch().set_alpha(0.8)

            # Connect the event listeners for click, hover and pick
            fig4.canvas.mpl_connect("button_press_event", self._click)
            fig4.canvas.mpl_connect("motion_notify_event", self.hover)
            fig4.canvas.mpl_connect("pick_event", self.on_annot_click)

            high_shear_5x = 0
            high_shear_15x = 0

            self.update(status_label)

            if len(all_times) > BLIP1_IDX:
                f0 = ys_freq[all_times[FILL_IDX]]
                d0 = dissipation[all_times[FILL_IDX]]
                f2 = ys_freq[all_times[BLIP1_IDX]]
                d2 = dissipation[all_times[BLIP1_IDX]]
                Log.i(f"f0 = {f0:2.2f} Hz")
                Log.i(f"f2 = {f2:2.2f} Hz")
                Log.i(f"f2-f0 = {f2-f0} Hz")

                # PR #377: Guardrail: Check Frequency/Dissipation Ratio for High Shear-Rate Calculation
                frequency_shift = f2 - f0
                dissipation_shift = d2 - d0
                ratio = frequency_shift / dissipation_shift * 1e-6

                # Adjust limits for the guardrails as needed
                freq_limit = 1000
                ratio_limit = 40
                if abs(frequency_shift) < freq_limit and abs(ratio) < ratio_limit:

                    # Calculate high shear-rate viscosity
                    high_shear_15x = 15e6

                    self.update(status_label)

                    if frequency_shift > float(
                        Constants.get_batch_param(batch, "freq_delta_15MHz")
                    ):
                        freq_factor_15MHz = float(
                            Constants.get_batch_param(batch, "freq_factor_15MHz")
                        )
                        high_shear_15y = ((frequency_shift * freq_factor_15MHz) ** 2) / DENSITY
                        Log.i(
                            f"15MHz High shear = ((f2-f0) * {freq_factor_15MHz})^2 / {DENSITY} = {high_shear_15y:2.2f} cP"
                        )
                    else:
                        diss_factor1_15MHz = float(
                            Constants.get_batch_param(batch, "diss_factor1_15MHz")
                        )
                        diss_factor2_15MHz = float(
                            Constants.get_batch_param(batch, "diss_factor2_15MHz")
                        )
                        bandaid_compensate_high_shear_viscosity = False
                        if bandaid_compensate_high_shear_viscosity:
                            E3 = (
                                ys_freq[all_times[FILL_IDX]] - ys_freq[all_times[START_IDX]]
                            )  # from CAL file (Freq_fill)
                            D = dissipation_shift - ((0.023112 * (E3) / DENSITY - 4.6868) * 1e-6)
                            high_shear_15y = (
                                (D * diss_factor1_15MHz - diss_factor2_15MHz) ** 2
                            ) / DENSITY
                        else:
                            high_shear_15y = (
                                (dissipation_shift * diss_factor1_15MHz - diss_factor2_15MHz) ** 2
                            ) / DENSITY
                        Log.i(f"d0 = {d0:1.4E}")
                        Log.i(f"d2 = {d2:1.4E}")
                        Log.i(f"d2-d0 = {dissipation_shift:1.4E}")
                        if bandaid_compensate_high_shear_viscosity:
                            Log.i(f"E3 = {E3}")
                            Log.i(f"D = {D}")
                            Log.i(
                                f"15MHz High shear = ({D} * {diss_factor1_15MHz}-{diss_factor2_15MHz})^2 / {DENSITY} = {high_shear_15y:2.2f} cP"
                            )
                        else:
                            Log.i(
                                f"15MHz High shear = ((d2-d0) * {diss_factor1_15MHz}-{diss_factor2_15MHz})^2 / {DENSITY} = {high_shear_15y:2.2f} cP"
                            )
                    high_shear_15x = self.correctHighShear(high_shear_15x, high_shear_15y)
                    ax7.plot(high_shear_15x, high_shear_15y, "bd")
                    ax7.errorbar(
                        high_shear_15x,
                        high_shear_15y,
                        0.30 * high_shear_15y,
                        fmt="b.",
                        ecolor="blue",
                        capsize=3,
                    )

                    self.update(status_label)

                    if True:
                        data_path_fun = data_path.replace("_3rd.csv", "_lower.csv")
                        fun_file_exists = secure_open.file_exists(data_path_fun, "capture")

                    if (
                        frequency_shift < 900 and fun_file_exists
                    ):  # frequency check added 2023-02-01
                        if True:
                            with secure_open(data_path_fun, "r", "capture") as f:
                                csv_headers_fun = next(f)

                                if isinstance(csv_headers_fun, bytes):
                                    csv_headers_fun = csv_headers_fun.decode()

                                if "Ambient" in csv_headers_fun:
                                    csv_cols_fun = (2, 4, 6, 7)
                                else:
                                    csv_cols_fun = (2, 3, 5, 6)

                                data_fun = loadtxt(
                                    f.readlines(),
                                    delimiter=",",
                                    skiprows=0,
                                    usecols=csv_cols_fun,
                                )

                        self.update(status_label)

                        relative_time_fun = data_fun[:, 0]
                        temperature_fun = data_fun[:, 1]
                        resonance_frequency_fun = data_fun[:, 2]
                        dissipation_fun = data_fun[:, 3]

                        self.update(status_label)

                        Log.i("Analyzing fundamental frequency dataset...")
                        times_fun = []
                        for i in range(len(all_times)):
                            t_fun = 0
                            try:
                                t_fun = (
                                    next(
                                        x
                                        for x, t in enumerate(relative_time_fun)
                                        if t >= xs[all_times[i]] - xs[0]
                                    )
                                    - 1
                                )
                            except StopIteration:
                                Log.e(
                                    f"Failed to locate POI_{i} @ timestamp {xs[all_times[i]] - xs[0]} from fundamental dataset. Attempting to proceed with index 0..."
                                )
                            Log.d(f"time[{i}] must be >= {xs[all_times[i]] - xs[0]}")
                            Log.d(f"time[{i}] = {relative_time_fun[t_fun]}, index {t_fun}")
                            times_fun.append(t_fun)
                        ys_freq_fun = (
                            np.average(resonance_frequency_fun[0 : times_fun[FILL_IDX]])
                            - resonance_frequency_fun
                        )
                        high_shear_5x = 5e6
                        xp = relative_time_fun
                        fp = ys_freq_fun
                        # absolute time of 15MHz start idx
                        t0 = xs[all_times[FILL_IDX]] - xs[0]
                        # absolute time of 15MHz blip1 idx
                        t2 = xs[all_times[BLIP1_IDX]] - xs[0]
                        f0 = np.interp(t0, xp, fp)
                        d0 = dissipation_fun[10]
                        f2 = np.interp(t2, xp, fp)
                        d2 = dissipation_fun[times_fun[BLIP1_IDX]]
                        Log.d(f"fun values to interpolate: [{t0}, {t2}]")
                        Log.d(f"{0}: ({xp[0]}, {fp[0]})")
                        Log.d("...")
                        for i in range(len(xp)):
                            if xp[i - 1] <= t0 and xp[i] >= t0:
                                Log.d(f"{i-1}: ({xp[i-1]}, {fp[i-1]})")
                                Log.d(f"## INTERP t0 HERE: ({t0}, {f0})")
                                Log.d(f"{i+1}: ({xp[i+1]}, {fp[i+1]})")
                                Log.d("...")
                            if xp[i - 1] <= t2 and xp[i] >= t2:
                                Log.d(f"{i-1}: ({xp[i-1]}, {fp[i-1]})")
                                Log.d(f"## INTERP t2 HERE: ({t2}, {f2})")
                                Log.d(f"{i+1}: ({xp[i+1]}, {fp[i+1]})")
                                Log.d("...")
                            # Log.d(f"{i}: ({xp[i]}, {fp[i]})")
                        # ending 'i' from last 'for' loop
                        Log.d(f"{i}: ({xp[i]}, {fp[i]})")
                        Log.i(f"f0 = {f0:2.2f} Hz")
                        Log.i(f"f2 = {f2:2.2f} Hz")
                        Log.i(f"f2-f0 = {f2-f0} Hz")
                        if f2 - f0 > float(Constants.get_batch_param(batch, "freq_delta_5MHz")):
                            freq_factor_5MHz = float(
                                Constants.get_batch_param(batch, "freq_factor_5MHz")
                            )
                            high_shear_5y = (((f2 - f0) * freq_factor_5MHz) ** 2) / DENSITY
                            Log.i(
                                f"5MHz High shear = ((f2-f0) * {freq_factor_5MHz})^2 / {DENSITY} = {high_shear_5y:2.2f} cP"
                            )
                        else:
                            diss_factor1_5MHz = float(
                                Constants.get_batch_param(batch, "diss_factor1_5MHz")
                            )
                            diss_factor2_5MHz = float(
                                Constants.get_batch_param(batch, "diss_factor2_5MHz")
                            )
                            high_shear_5y = (
                                ((d2 - d0) * diss_factor1_5MHz - diss_factor2_5MHz) ** 2
                            ) / DENSITY
                            Log.i(f"d0 = {d0:1.4E}")
                            Log.i(f"d2 = {d2:1.4E}")
                            Log.i(f"d2-d0 = {d2-d0:1.4E}")
                            Log.i(
                                f"5MHz High shear = ((d2-d0) * {diss_factor1_5MHz}-{diss_factor2_5MHz})^2 / {DENSITY} = {high_shear_5y:2.2f} cP"
                            )
                        high_shear_5x = self.correctHighShear(high_shear_5x, high_shear_5y)
                        ax7.plot(high_shear_5x, high_shear_5y, "bd")
                        ax7.errorbar(
                            high_shear_5x,
                            high_shear_5y,
                            0.30 * high_shear_5y,
                            fmt="b.",
                            ecolor="blue",
                            capsize=3,
                        )
                    else:
                        Log.w("5 MHz high-shear calculation not available from dataset.")
                        if not fun_file_exists:
                            Log.w(
                                "The 5 MHz mode does not exist in the dataset for this captured run."
                            )
                        else:
                            Log.w(
                                "The frequency shift of the initial fill region is too small (<900 Hz) for high-shear calculation accuracy."
                            )
                else:
                    Log.w("5 MHz high-shear calculation not available from dataset.")
                    Log.w("15 MHz high-shear calculation not available from dataset.")

                    if not frequency_shift < freq_limit:
                        Log.w("Reason: Frequency shift limit exceeded.")
                        Log.d(
                            f"Detail: Must be less than {freq_limit}. Actual: {frequency_shift:2.2f}."
                        )

                    if not ratio < ratio_limit:
                        Log.w("Reason: Ratio threshold limit exceeded.")
                        Log.d(f"Detail: Must be less than {ratio_limit}. Actual: {ratio:2.2f}.")
            else:
                Log.w("5 MHz high-shear calculation not available from dataset.")
                Log.w("15 MHz high-shear calculation not available from dataset.")

                Log.w(
                    "Too few valid time points are available in Figure 2 for any high-shear calculation accuracy."
                )
                Log.w(
                    "See Figure 2 to check if any of these points is being dropped due to time delta not being 2x last."
                )
                Log.w(
                    "If so, please adjust the Precise Fill Points for this run accordingly and try this analysis again."
                )

            self.update(status_label)

            # viscosity = ST*np.cos(np.radians(CA))*all_time*Constants.channel_thickness/6/(all_pos**2)*1e3*(3*(n+1)/(2*n+1))
            # viscosity = viscosity * 1000

            # viscosity_2 = ST*np.cos(np.radians(CA))*line1_x*Constants.channel_thickness/6/(line1_curve**2)*1e3*(3*(n+1)/(2*n+1))
            # viscosity_2 = viscosity_2 * 1000

            # keep_ids = reject_outliers(viscosity, 11.)
            # out_shear_rate = shear_rate[~keep_ids]
            # out_viscosity = viscosity[~keep_ids]
            in_shear_rate = shear_rate  # [keep_ids]
            in_viscosity = viscosity  # [keep_ids]
            in_temp = all_temp
            # outliers = np.where(keep_ids == False)
            # Log.d(f"in_visc = {viscosity}")
            # Log.d(f"outliers = {outliers}")

            self.update(status_label)

            if len(in_shear_rate) == 0 or len(in_viscosity) == 0:
                in_shear_rate = shear_rate
                in_viscosity = viscosity
                in_temp = all_temp
                Log.w(
                    "WARN: Initial fill region contains nothing but outlier. Attempting to continue with outliers."
                )
                Log.w("Please check the first 2 POIs for accuracy.")

            # remove NANs from datasets
            to_remove = np.isnan(in_shear_rate)
            to_remove |= np.isnan(in_viscosity)
            to_remove |= np.isinf(in_shear_rate)
            to_remove |= np.isinf(in_viscosity)
            in_shear_rate = in_shear_rate[~to_remove]
            in_viscosity = in_viscosity[~to_remove]
            in_temp = in_temp[~to_remove]

            self.update(status_label)

            if len(in_shear_rate) == 0 or len(in_viscosity) == 0:
                in_shear_rate = shear_rate
                in_viscosity = viscosity
                in_temp = all_temp
                Log.w(
                    "WARN: Initial fill region contains nothing but inf/nan. Attempting to continue with outliers."
                )
                Log.w("Please check the first 2 POIs for accuracy.")

            viscosity_at_1p15 = viscosity[-len(distances)]

            # PURPOSE: Hide 60% and/or 80% points when trending outside +/- 5% of POI2 and POI4
            # NOTE: Historically, this used to be +/- 10%, but was changed with issue #314.
            try:
                normal_idxs = []
                percent_pts = {}
                for i in idx_of_normal_pts_to_retain:
                    if i in times:
                        normal_idxs.append(-len(distances) + times.index(i))
                    else:
                        Log.w(f"Index for {i} in `times` cannot be found in list. Skipping point")
                if len(normal_idxs) == 0:
                    raise Exception("Empty list cannot be reduced further")
                idx0 = np.min(normal_idxs) - 1  # POI2
                idx1 = np.max(normal_idxs) + 1  # POI4
                # avg_viscosity = np.average(
                #     [in_viscosity[idx0], in_viscosity[idx3]])
                # std_viscosity = np.std(
                #     np.delete(in_viscosity, [idx1, idx2])
                # )  # all of in_viscosity, just not 2 points
                min_visc = 0.95 * min(viscosity[idx0], viscosity[idx1])
                max_visc = 1.05 * max(viscosity[idx0], viscosity[idx1])
                Log.i(f"Expected normal viscosity range = (min = {min_visc}, max = {max_visc})")
                # Log.d("Indices 0-3 are:", [idx0, idx1, idx2, idx3])
                for x, i in enumerate(normal_idxs):
                    pt = "60%" if x == 0 else "80%"
                    percent_pts[pt] = (in_shear_rate[i], in_viscosity[i])
                    if min_visc <= viscosity[i] <= max_visc:
                        continue
                    Log.w(
                        f"Removed {pt} point '{viscosity[i]}' for being outside the standard deviation of expected viscosity."
                    )
                    flag_warn = False
                    arrays_to_check = [
                        in_shear_rate,
                        in_viscosity,
                        in_temp,
                        viscosity,
                        shear_rate,
                        fill_visc,
                        fill_shear,
                        distances,
                    ]
                    if any(len(arr) < abs(i) for arr in arrays_to_check):
                        Log.w("Unable to remove outlier consistently; leaving dataset unchanged.")
                        continue
                    if len(in_shear_rate) >= abs(i):
                        in_shear_rate = np.delete(in_shear_rate, i)
                    else:
                        flag_warn = True
                    if len(in_viscosity) >= abs(i):
                        in_viscosity = np.delete(in_viscosity, i)
                    else:
                        flag_warn = True
                    if len(in_temp) >= abs(i):
                        in_temp = np.delete(in_temp, i)
                    else:
                        flag_warn = True
                    if len(viscosity) >= abs(i):
                        viscosity = np.delete(viscosity, i)
                    else:
                        flag_warn = True
                    if len(shear_rate) >= abs(i):
                        shear_rate = np.delete(shear_rate, i)
                    else:
                        flag_warn = True
                    if len(fill_visc) >= abs(i):
                        fill_visc = np.delete(fill_visc, i)
                    else:
                        flag_warn = True
                    if len(fill_shear) >= abs(i):
                        fill_shear = np.delete(fill_shear, i)
                    else:
                        flag_warn = True
                    if len(distances) >= abs(i):
                        distances = np.delete(distances, -i)
                    else:
                        flag_warn = True
                    if flag_warn:
                        Log.w("WARNING: Unable to remove all outliers from the dataset.")
            except Exception as e:
                Log.e("ERROR:", e)
                Log.e("Unable to remove outliers from the dataset prior to plotting.")

            self.update(status_label)

            # np.linspace(lin_shear_rate[0], lin_shear_rate[-1]) # default: 50 points
            fit_shear = fill_shear
            fit_visc = (
                # cube_fit(fit_shear) # plot this one, evenly spaced points
                fill_visc
            )
            lin_viscosity = fill_visc

            debug = False
            if debug:
                raw_shear = in_shear_rate[0 : -len(distances)]
                raw_visc = in_viscosity[0 : -len(distances)]
                import matplotlib.pyplot as plt

                fig_dbg = plt.figure(figsize=(12, 9))
                ax_dbg = fig_dbg.add_subplot(111)
                # ax_dbg.scatter(raw_shear_out, raw_visc_out, color="red", marker="x")
                ax_dbg.scatter(raw_shear, raw_visc, color="blue", marker=".")
                ax_dbg.plot(fit_shear, fit_visc, color="black", marker=",")
                fig_dbg.show()

            self.update(status_label)

            # ax7.annotate("START", (shear_rate[0],viscosity[0]),
            #    textcoords="offset points", xytext=(0,-15), ha='center')
            # ax7.plot(out_shear_rate, out_viscosity, 'rx')
            # ax7.plot(shear_rate, viscosity, 'r:')

            ### BANDAID #3 ###
            # PURPOSE: Hide initial fill points when trending in the wrong direction of high-shear
            enable_bandaid_3 = True
            # NOTE: This was only enabled for production builds, not dev/nightly builds
            #       As of issue #314 (2026-03-19): Band-Aid is enabled in all contexts.
            # if "_dev" in Constants.app_version or "_nightly" in Constants.app_version:
            #    enable_bandaid_3 = False
            hide_initial_fill = False  # if disabled, never force hide initial fill
            remove_initial_fill = False

            # New trendline variable for smoothing the initial fill small blue diamond points
            try:
                sm_x = in_viscosity[: -len(distances)]
                sm_wl = len(in_viscosity) - len(distances)
                if len(sm_x) == 0:
                    raise ValueError("No initial fill points present in dataset.")
                if sm_wl <= 1:
                    raise ValueError(
                        "Too few points for smoothing: `wl` must be greater than `polyorder`."
                    )
                sm_trendline = savgol_filter(sm_x, sm_wl, 1)
            except (ValueError, IndexError) as e:
                Log.e(
                    "Failed to generate initial fill region trendline. Skipping initial fill region analysis."
                )
                Log.d(f"Error Details: {e}")
                hide_initial_fill = True

            point_factor_limit = 0.25
            if point_factor_limit < 0 or point_factor_limit > 1:
                Log.e(
                    f"Invalid 'point_factor_limit' set: {point_factor_limit:2.2f} (Must be between zero and one)"
                )
                Log.e(
                    "Disabling initial fill limit check due to invalid parameter specified: 'point_factor_limit'"
                )
                enable_bandaid_3 = False

            # Checking for bandaid #3 moved ~100 lines lower to accommodate USE_NEW_FILL_METHOD
            ##################

            ### BANDAID #4 ###
            # PURPOSE: Hide 60% and/or 80% points when trending in the wrong direction surrounding POI2 and POI4
            # enable_bandaid_4 = True
            # if enable_bandaid_4:
            #     for i in idx_of_normal_pts_to_retain:
            #         try:
            #             idx = times.index(i)
            #             Log.d(
            #                 f"Removing index {idx} from distances with value {distances[idx]}."
            #             )
            #             distances = np.delete(distances, idx)
            #             Log.d(f"Removing index {idx} from times with value {i}.")
            #             times.remove(i)
            #         except Exception as e:
            #             Log.e("Error removing midpoint from dataset:", str(e))
            ##################

            if initial_fill[-1] >= 90 and not hide_initial_fill:
                # Truncate the initial fill region to just a few evenly spaced points
                # mlen = int(np.floor((len(in_shear_rate) - len(distances)) / 5))
                # See issue #256 for details on why use dynamic number of fill points
                min_fill_pts = 3
                max_fill_pts = 8
                target_num_pts = 10
                num_fill_pts = max(min_fill_pts, min(max_fill_pts, target_num_pts - len(distances)))
                shear_at_fill_start = in_shear_rate[0]
                shear_at_fill_end = in_shear_rate[-len(distances) - 1]
                shear_points = np.geomspace(  # like `linspace` but for log10
                    shear_at_fill_end, shear_at_fill_start, num_fill_pts
                )
                local_shear = []
                local_visc = []
                local_linv = []
                local_temp = []
                last_idx = -1
                # skip first point, reverse order
                for pt in shear_points[1:][::-1]:
                    try:
                        idx = next(x for x, y in enumerate(in_shear_rate) if y <= pt)
                    except StopIteration:
                        Log.e(
                            f"Failed to find index for shear point {pt:2.2f}. Cannot plot this index."
                        )
                        continue

                    if USE_NEW_FILL_METHOD:
                        # NEW METHOD:
                        if last_idx == -1:
                            mv = fill_pos[idx] / fill_time[idx]
                            mp = fill_pos[idx] / 2
                        else:
                            mv = (fill_pos[idx] - fill_pos[last_idx]) / (
                                fill_time[idx] - fill_time[last_idx]
                            )
                            mp = (fill_pos[idx] + fill_pos[last_idx]) / 2
                        last_idx = idx
                        mid_visc = (
                            ST
                            * np.cos(np.radians(CA))
                            * Constants.channel_thickness
                            * 1e6
                            / ((mp * mv * 6) * (2 / 3 + 1 / 3 / n))
                        )
                        mid_shear = (
                            6 * mv / Constants.channel_thickness * (2 / 3 + 1 / 3 / n) * 1e-3
                        )
                        # Use to show the old positions:
                        # ax7.scatter(
                        #     in_shear_rate[idx],
                        #     sm_trendline[idx],
                        #     marker="d",
                        #     s=15,
                        #     c="red",
                        # )
                        sm_trendline[idx] = mid_visc
                        in_shear_rate[idx] = mid_shear

                    local_shear.append(in_shear_rate[idx])
                    local_visc.append(sm_trendline[idx])
                    local_linv.append(lin_viscosity[idx])
                    local_temp.append(in_temp[idx])
                if enable_bandaid_3 and high_shear_15x:
                    P1_value = local_visc[-1]
                    P2_value = high_shear_15y  # exists only if high_Shear_15x is not zero
                    lower_factor = 1 - point_factor_limit
                    upper_factor = 1 + point_factor_limit
                    min_fit_end = min(P1_value, P2_value) * lower_factor
                    max_fit_end = max(P1_value, P2_value) * upper_factor
                    local_visc_array = np.array(local_visc)
                    Log.d(f"Point Factor Limit for Initial Fill is: {point_factor_limit:2.2f}x")
                    Log.d(
                        f"Trendline must be within range from {min_fit_end:2.2f} to {max_fit_end:2.2f}"
                    )
                    Log.d(
                        f"Initial Fill Trendline ranges from {local_visc_array.min():2.2f} to {local_visc_array.max():2.2f}"
                    )
                    if (
                        min_fit_end > local_visc_array.min() or max_fit_end < local_visc_array.max()
                    ):  # Trendline is outside the allowable range
                        Log.w(
                            f"Dropping initial fill region due to being outside of the accepted limits (see Debug for more info)"
                        )
                        remove_initial_fill = True
                if not remove_initial_fill:
                    ax7.scatter(
                        local_shear,
                        local_visc,
                        marker="d",
                        s=15,
                        c="blue",
                    )
                for idx in range(len(distances)):
                    local_shear.append(in_shear_rate[-len(distances) + idx])
                    local_visc.append(in_viscosity[-len(distances) + idx])
                    # local_linv.append(lin_viscosity[-len(distances)+idx])
                    local_temp.append(in_temp[-len(distances) + idx])
                in_shear_rate = local_shear
                in_viscosity = local_visc
                lin_viscosity = local_linv
                in_temp = local_temp
                # These plots are useful for debugging, but are not usually shown:
                # ax7.scatter(
                #     in_shear_rate, in_viscosity, marker="d", s=1, c="blue"
                # )
                # ax7.plot(in_shear_rate[:-len(distances)], sm_trendline,
                #          color="black", marker=",")
                # ax7.plot(fit_shear, fit_visc, color="black", marker=",")
                # for hh in range(1, 5):
                #     xp = np.average(
                #         in_shear_rate[hh * mlen: (hh + 1) * mlen - 1])
                #     yp = np.average(
                #         in_viscosity[hh * mlen: (hh + 1) * mlen - 1])
                #     stdev = np.std(
                #         in_viscosity[hh * mlen: (hh + 1) * mlen - 1])
                #     # ax7.plot(xp, yp, 'b.')
                #     ax7.errorbar(xp, yp, stdev, fmt="b.",
                #                  ecolor="blue", capsize=3)
            else:
                # Remove initial fill points from output table later
                remove_initial_fill = True

            self.update(status_label)

            avg_viscosity = np.average(in_viscosity)
            std_viscosity = np.std(in_viscosity)
            # lin_viscosity = np.flip(lin_viscosity)
            for i in range(-len(distances), 0):
                percent_error = (
                    abs((viscosity[i] - viscosity[-len(distances)]) / viscosity[-len(distances)])
                    * 100
                )
                Log.d(f"Percent error for calculated viscosity is: {percent_error}")
                if percent_error < 20.0:
                    # show bigly if it's not an outlier
                    ax7.plot(shear_rate[i], viscosity[i], "bd")
                else:
                    ax7.plot(
                        shear_rate[i], viscosity[i], "bd"
                    )  # show bigly even if it is (for now)

                if lin_viscosity[-1] == viscosity[i] and i == -len(distances):
                    continue  # skip adding the first viscosity point if it is a duplicate
                lin_viscosity = np.append(lin_viscosity, viscosity[i])
            if len(lin_viscosity) == len(in_shear_rate) - 1:
                # re-add the skipped viscosity point if the lengths do not match
                lin_viscosity = np.insert(
                    lin_viscosity, -len(distances), viscosity[-len(distances)]
                )

            if remove_initial_fill:
                # Remove initial fill points from output table
                in_shear_rate = in_shear_rate[-len(distances) :]
                in_viscosity = in_viscosity[-len(distances) :]
                lin_viscosity = lin_viscosity[-len(distances) :]
                in_temp = in_temp[-len(distances) :]

            in_shear_rate = np.flip(in_shear_rate)
            in_viscosity = np.flip(in_viscosity)
            lin_viscosity = np.flip(lin_viscosity)
            in_temp = np.flip(in_temp)
            if high_shear_5x != 0:
                in_shear_rate = np.append(in_shear_rate, high_shear_5x)
                in_viscosity = np.append(in_viscosity, high_shear_5y)
                lin_viscosity = np.append(lin_viscosity, high_shear_5y)
                in_temp = np.append(in_temp, avg_temp)
            if high_shear_15x != 0:
                in_shear_rate = np.append(in_shear_rate, high_shear_15x)
                in_viscosity = np.append(in_viscosity, high_shear_15y)
                lin_viscosity = np.append(lin_viscosity, high_shear_15y)
                in_temp = np.append(in_temp, avg_temp)

            self.update(status_label)

            ax7.set_title(f"Shear-rate vs. Viscosity: {data_title}")
            ax7.set_xlabel("Shear-rate (s⁻¹)")
            ax7.set_ylabel("Viscosity (cP)")
            lower_limit = np.amin(in_viscosity) / 1.5
            power = 1
            while power > -5:
                if lower_limit > 10**power:
                    lower_limit = 10**power
                    break
                power -= 1
            upper_limit = np.amax(in_viscosity) * 1.5
            power = 0
            while power < 5:
                if upper_limit < 10**power:
                    upper_limit = 10**power
                    break
                power += 1
            if lower_limit >= upper_limit:
                Log.w(
                    "Limits were auto-calculated but are in an invalid range! Using ylim [0, 1000]."
                )
                ax7.set_ylim([0, 1000])
            elif np.isfinite(lower_limit) and np.isfinite(upper_limit):
                Log.d(
                    f"Auto-calculated y-range limits for Figure 4 are: [{lower_limit}, {upper_limit}]"
                )
                ax7.set_ylim([lower_limit, upper_limit])
            else:
                Log.w(
                    "Limits were auto-calculated but were not finite values! Using ylim [0, 1000]."
                )
                ax7.set_ylim([0, 1000])

            ax7.set_xscale("log")
            ax7.set_yscale("log")

            self.update(status_label)

            err_viscosity = []
            str_viscosity = []
            for i in range(len(in_shear_rate)):
                err_viscosity.append(in_viscosity[i] * 0.10)
                str_viscosity.append(
                    f"{in_viscosity[i]:2.2f} \u00b1 {err_viscosity[i]:2.2f}"
                )  # plus-or-minus = \u00b1

            # On multiplex systems, all `in_temp` will be NaN
            # NOTE: Move this check to before marking "*...*" error cells
            real_temps = [x for x in in_temp if ~np.isnan(x)]

            # Annotate average viscosity and standard deviation on plot and in output CSV
            if len(log_velocity_46) == len(distances):  # not checked
                Log.w(
                    "WARNING: Initial fill values are not considered to be reliably accurate for this run."
                )
                Log.w(
                    "Initial fill values are marked as 'light red' in the tabular data for reference only."
                )

                in_shear_rate = np.array(in_shear_rate, dtype=str)
                in_viscosity = np.array(in_viscosity, dtype=str)
                # str_viscosity = np.array(str_viscosity, dtype=str) # NOTE: Numpy doesn't handle unicode chars in array strings
                in_temp = np.array(in_temp, dtype=str)

                pts_to_modify = range(len(distances), len(in_shear_rate))
                for i in range(len(in_shear_rate)):
                    is_error_cell = i in pts_to_modify
                    if in_shear_rate[i] in [str(5e6), str(15e6)]:
                        is_error_cell = False

                    if is_error_cell:
                        # Log.i(f"Converting {in_shear_rate[i]}")
                        # Log.i(f"into *{in_shear_rate[i]:2.2f}*")
                        in_shear_rate[i] = f"*{float(in_shear_rate[i]):2.2f}*"
                        in_viscosity[i] = f"*{float(in_viscosity[i]):2.2f}*"
                        str_viscosity[i] = f"*{str_viscosity[i]}*"
                        in_temp[i] = f"*{float(in_temp[i]):2.2f}*"
                    else:
                        in_shear_rate[i] = f"{float(in_shear_rate[i]):2.2f}"
                        in_viscosity[i] = f"{float(in_viscosity[i]):2.2f}"
                        # str_viscosity[i] = f"{str_viscosity[i]}"
                        in_temp[i] = f"{float(in_temp[i]):2.2f}"

                in_shear_rate = in_shear_rate.tolist()
                in_viscosity = in_viscosity.tolist()
                # str_viscosity = str_viscosity.tolist()
                in_temp = in_temp.tolist()

            # add data to table view of results
            data = {
                "Shear Rate (s⁻¹)": in_shear_rate,
                "Raw Viscosity (cP)": in_viscosity,
                "Avg Viscosity (cP)": str_viscosity,
                "Temperature (C)": in_temp,
            }
            rows = len(in_shear_rate)
            cols = len(data)

            # Store to global for hover/pick actions
            self.last_shear_rates = in_shear_rate
            self.last_distances = distances

            # On multiplex systems, all `in_temp` will be NaN
            if len(real_temps) == 0:
                Log.w("Hiding \"Temperature (C)\" column, as all temperature values are 'nan'.")
                data.pop("Temperature (C)")
                cols -= 1
            # data, rows, cols = [{"col1": ["Hello", "This"], "col2": ["World", "Is"], "col3": ["Foo", "A"], "col4": ["Bar", "Test"]}, 2, 4]

            self.update(status_label)

            # Create all result objects
            res_shear_rate = []
            res_viscosity = []
            res_percent_err = []
            res_temp = []
            res_n_coeff = []

            try:

                if True:
                    # Highly non-Newtonian or Newtonian: use interpolated value at 1000 s⁻¹
                    # Interpolate across the entire dataset from POI1 (start-of-fill) to POI6 (ch3)
                    # excluding the 60% and 80% points from the initial fill region (if present)
                    shear_interp = 1000
                    in_shear_san_60_80: list = in_shear_rate.tolist()
                    in_visco_san_60_80: list = in_viscosity.tolist()
                    if "percent_pts" in locals():
                        # Remove 60% and 80% points from data for interpolation
                        for shear, visco in percent_pts.values():
                            if shear in in_shear_san_60_80:
                                in_shear_san_60_80.remove(shear)
                            if visco in in_visco_san_60_80:
                                in_visco_san_60_80.remove(visco)
                    interp_func = interp1d(
                        in_shear_san_60_80, in_visco_san_60_80, fill_value="extrapolate"
                    )
                    visc_interp = float(interp_func(shear_interp))
                    i_l, i_r = next(
                        ((i - 1, i) for i, s in enumerate(in_shear_san_60_80) if s > shear_interp),
                        (-1, len(in_shear_san_60_80)),
                    )
                    if i_l == -1 or i_r == len(in_shear_san_60_80):
                        # indicate 10% error when extrapolating beyond left or right of the shear array
                        visc_error = visc_interp / 10
                    else:
                        # indicate half of absolute difference for left/right points when interpolating
                        visc_error = np.abs(in_visco_san_60_80[i_l] - in_visco_san_60_80[i_r]) / 2

                    summary_text = "Interpolated viscosity is {:2.2f} \u00b1 {:2.2f} cP for shear rate {:2.0f} s⁻¹.".format(
                        visc_interp, visc_error, shear_interp
                    )
                    plot_text = "{:2.2f} \u00b1 {:2.2f} cP @ {:2.0f} s⁻¹".format(
                        visc_interp, visc_error, shear_interp
                    )

                    res_shear_rate.append(f"{shear_interp:2.2f}")
                    res_viscosity.append(visc_interp)
                    res_percent_err.append(visc_error)
                    res_temp.append(avg_temp)
                    res_n_coeff.append(n)

                if abs(n - 1.0) <= Constants.shear_interp_threshold:
                    # Nearly Newtonian: use current average method
                    # Calculate the average viscosity and standard deviation from POI2 (end-of-fill) to POI6 (ch3)
                    values_to_average = len(distances)
                    # high_shear_counts = np.count_nonzero(
                    #     [high_shear_5x, high_shear_15x])
                    idx_start = 0
                    # keep within current arrays (handles extra high-shear rows appended at end)
                    idx_end = min(
                        len(in_viscosity) - 1,
                        len(in_shear_rate) - 1,
                        max(2, values_to_average - 1),
                    )
                    # Make sure NOT to include high-shear(s) in average viscosity calculation
                    visc_subset = in_viscosity[idx_start : idx_end + 1]
                    if high_shear_15x != 0:
                        if high_shear_15y in visc_subset:
                            # assumes 15MHz is last in list
                            visc_subset = visc_subset[:-1]
                            idx_end -= 1
                    if high_shear_5x != 0:
                        if high_shear_5y in visc_subset:
                            # assumes 5MHz is last in list
                            visc_subset = visc_subset[:-1]
                            idx_end -= 1
                    # Calculate average +/- deviation from viscosity subset
                    visc_avg = np.average(visc_subset)
                    visc_std = np.std(visc_subset)
                    shear_min = in_shear_rate[idx_start]
                    shear_max = in_shear_rate[idx_end]

                    summary_text += "\nAverage viscosity is {:2.2f} \u00b1 {:2.2f} cP for shear rates {:2.0f} - {:2.0f} s⁻¹.".format(
                        visc_avg, visc_std, shear_min, shear_max
                    )
                    plot_text += "\n\n{:2.2f} \u00b1 {:2.2f} cP\n({:2.0f} - {:2.0f}) s⁻¹".format(
                        visc_avg, visc_std, shear_min, shear_max
                    )

                    res_shear_rate.append(f"{shear_min:2.2f}-{shear_max:2.2f}")
                    res_viscosity.append(visc_avg)
                    res_percent_err.append(visc_std)
                    res_temp.append(avg_temp)
                    res_n_coeff.append(n)

                # Add optimally positioned label to plot data
                self.plot_ax = ax7
                self.plot_text = plot_text
                self.place_text_avoiding_data()

                # # Convert `in_shear_rate` to formatted strings
                # for i in range(len(in_shear_rate)):
                #     if type(in_shear_rate[i]) is not str:
                #         in_shear_rate[i] = f"{in_shear_rate[i]:2.2f}"

            except Exception as e:
                Log.e("Failed to calculate average viscosity summary.", str(e))

            status_label = "Saving Results..."
            self.update(status_label)

            try:
                # Convert all output data to list type
                if type(in_shear_rate) is not list:
                    in_shear_rate = in_shear_rate.tolist()
                if type(in_viscosity) is not list:
                    in_viscosity = in_viscosity.tolist()
                if type(lin_viscosity) is not list:
                    lin_viscosity = lin_viscosity.tolist()
                if type(err_viscosity) is not list:
                    err_viscosity = err_viscosity.tolist()
                if type(in_temp) is not list:
                    in_temp = in_temp.tolist()

                # export output data to csv
                export_path = data_path
                export_path = export_path.replace(".csv", Constants.export_file_format)
                export_path = export_path.replace("_fundamental", "")
                export_path = export_path.replace("_3rd", "")
                Log.i(f"Exporting analyze output to:\n\t{export_path}")
                np.savetxt(
                    export_path,
                    np.column_stack(
                        [
                            in_shear_rate,
                            lin_viscosity,
                            lin_viscosity,
                            err_viscosity,
                            in_temp,
                        ]
                    ),
                    fmt="%.2f",
                    delimiter=",",
                    header="shear_rate,viscosity_raw,viscosity_avg,percent_error,temperature",
                )
            except Exception as e:
                Log.e("Error generating output file: " + str(e))
                if not os.path.exists(export_path):
                    with open(export_path, "w") as f:
                        f.write(str(e))
                # raise e # Debug only!

            self.update(status_label)

            try:
                # Convert all result data to list type
                if type(res_shear_rate) is not list:
                    res_shear_rate = res_shear_rate.tolist()
                if type(res_viscosity) is not list:
                    res_viscosity = res_viscosity.tolist()
                if type(res_percent_err) is not list:
                    res_percent_err = res_percent_err.tolist()
                if type(res_temp) is not list:
                    res_temp = res_temp.tolist()
                if type(res_n_coeff) is not list:
                    res_n_coeff = res_n_coeffs_temp.tolist()

                # export result data to csv
                result_path = data_path
                result_path = result_path.replace(".csv", Constants.result_file_format)
                result_path = result_path.replace("_fundamental", "")
                result_path = result_path.replace("_3rd", "")
                Log.i(f"Exporting analyze result to:\n\t{result_path}")
                np.savetxt(
                    result_path,
                    np.column_stack(
                        [
                            # Force to object
                            np.asarray(res_shear_rate, dtype=object),
                            # Force to object
                            np.asarray(res_viscosity, dtype=object),
                            # Force to object
                            np.asarray(res_percent_err, dtype=object),
                            # Force to object
                            np.asarray(res_temp, dtype=object),
                            # Force to object
                            np.asarray(res_n_coeff, dtype=object),
                        ]
                    ),
                    # First column as string, rest as floats
                    fmt=["%s", "%.2f", "%.2f", "%.2f", "%.02f"],
                    delimiter=",",
                    header="shear_rate,viscosity_avg,percent_error,temperature_avg,n_coeff",
                )
            except Exception as e:
                Log.e("Error generating result file: " + str(e))
                if not os.path.exists(result_path):
                    with open(result_path, "w") as f:
                        f.write(str(e))
                # raise e # Debug only!

            self.update(status_label)

            # Add a footnote below and to the right side of the chart
            footnote = "Generated by {} {} ({}) at {}."
            for axis in [ax4, ax5, ax6, ax7]:
                axis.annotate(
                    footnote.format(
                        Constants.app_title,
                        Constants.app_version,
                        Constants.app_date,
                        strftime("%Y-%m-%d %I:%M:%S %p", localtime()),
                    ),
                    xy=(0.5, 1),
                    xycoords=("figure fraction", "figure pixels"),
                    ha="center",
                    va="bottom",
                    color="dimgray",
                    fontsize=8,
                )

            self.update(status_label)

            # def export_figures(fig1, fig2, fig3, fig4):
            # export figures to file
            Log.i("Saving figures to file...")
            # self.progress.emit(75, "Saving Results...")
            Log.i(f'Exporting Figure 1 to:\n\t{export_path.replace(".csv", "_1.pdf")}')
            fig.savefig(export_path.replace(".csv", "_1.pdf"))
            self.update(status_label)
            # self.progress.emit(80, "Saving Results...")
            Log.i(f'Exporting Figure 2 to:\n\t{export_path.replace(".csv", "_2.pdf")}')
            fig2.savefig(export_path.replace(".csv", "_2.pdf"))
            self.update(status_label)
            # self.progress.emit(85, "Saving Results...")
            Log.i(f'Exporting Figure 3 to:\n\t{export_path.replace(".csv", "_3.pdf")}')
            fig3.savefig(export_path.replace(".csv", "_3.pdf"))
            self.update(status_label)
            # self.progress.emit(90, "Saving Results...")
            Log.i(f'Exporting Figure 4 to:\n\t{export_path.replace(".csv", "_4.pdf")}')
            fig4.savefig(export_path.replace(".csv", "_4.pdf"))
            self.update(status_label)

            enabled, error, expires = UserProfiles.checkDevMode()
            if enabled == False and (error == True or expires != ""):
                PopUp.warning(
                    self,
                    "Developer Mode Expired",
                    "Developer Mode has expired and these analysis results will now be encrypted.\n"
                    + 'An admin must renew or disable "Developer Mode" to suppress this warning.',
                )

            self.update(status_label)

            # Generate CAL file if dev mode is enabled, and not expired
            try:
                if enabled:

                    Log.i("Generating CAL file in output folder for manual analysis...")
                    cal_pts = np.array(
                        [
                            "start",
                            "fill",
                            "20%",
                            "40%",
                            "60%",
                            "80%",
                            "ch1",
                            "ch2",
                            "ch3",
                        ],
                        dtype=str,
                    )
                    cal_idxs = np.unique(
                        np.sort(
                            np.concatenate(
                                [
                                    [points_of_interest[0]],
                                    times,
                                    bad_times,
                                    idx_of_normal_pts_to_remove,
                                ]
                            )
                        ).astype(int)
                    )
                    na_val = len(xs) - 1
                    while len(cal_idxs) < len(cal_pts):  # extend until at required size
                        cal_idxs = np.append(cal_idxs, na_val)
                    cal_times = np.array(np.round(xs[cal_idxs], 4), dtype=float)
                    cal_disss = np.array(
                        np.round(ys[cal_idxs] - ys[points_of_interest[0]], 4),
                        dtype=float,
                    )
                    cal_freqs = np.array(
                        np.round(ys_freq[cal_idxs] - ys_freq[points_of_interest[0]], 4),
                        dtype=float,
                    )
                    cal_notes = ["Not Analyzed" if x in bad_times else "" for x in cal_idxs]

                    len_pts = len(cal_pts)
                    len_idxs = len(cal_idxs)
                    len_times = len(cal_times)
                    len_disss = len(cal_disss)
                    len_freqs = len(cal_freqs)
                    len_notes = len(cal_notes)

                    if len_pts == len_idxs == len_times == len_disss == len_freqs == len_notes:
                        Log.d("CHECK PASS: CAL arrays are the same length!")
                    else:
                        Log.w(
                            "CHECK FAIL: CAL arrays are different lengths. Truncating to shortest one."
                        )
                        len_req = min(
                            len_pts,
                            len_idxs,
                            len_times,
                            len_disss,
                            len_freqs,
                            len_notes,
                        )
                        if len_pts != len_req:
                            Log.w(f"Array `cal_pts` resized from {len_pts} to {len_req}")
                            cal_pts = cal_pts[:len_req]
                        if len_idxs != len_req:
                            Log.w(f"Array `cal_idxs` resized from {len_idxs} to {len_req}")
                            cal_idxs = cal_idxs[:len_req]
                        if len_times != len_req:
                            Log.w(f"Array `cal_times` resized from {len_times} to {len_req}")
                            cal_times = cal_times[:len_req]
                        if len_disss != len_req:
                            Log.w(f"Array `cal_disss` resized from {len_disss} to {len_req}")
                            cal_disss = cal_disss[:len_req]
                        if len_freqs != len_req:
                            Log.w(f"Array `cal_freqs` resized from {len_freqs} to {len_req}")
                            cal_freqs = cal_freqs[:len_req]
                        if len_notes != len_req:
                            Log.w(f"Array `cal_notes` resized from {len_notes} to {len_req}")
                            cal_notes = cal_notes[:len_req]

                    cal_data = np.column_stack(
                        [cal_pts, cal_idxs, cal_times, cal_disss, cal_freqs, cal_notes]
                    )

                    np.savetxt(
                        cal_path,
                        cal_data,
                        delimiter=",",
                        fmt="%s",
                        header="Point,Index,Time,Diss,Freq,Notes",
                    )

                    Log.i("Successfully generated CAL file: " + cal_path)

            except Exception as e:
                Log.e("Error generating CAL file: " + str(e))
                if not os.path.exists(cal_path):
                    with open(cal_path, "w") as f:
                        f.write(str(e))
                # raise e # Debug only!

            self.update(status_label)

            # Move generated PDFs (and CSVs) to secure ZIP folder
            Log.i(f"Compressing exported files to ZIP...")
            QtCore.QCoreApplication.processEvents()

            num = 1
            fullpath = os.path.split(export_path)[0]
            folder = os.path.split(fullpath)[1]
            last_zn = None
            while True:
                zn = os.path.join(fullpath, f"analyze-{num}.zip")
                if os.path.exists(zn):
                    last_zn = zn
                    num += 1
                else:
                    break
            with pyzipper.AESZipFile(
                zn,
                "w",
                compression=pyzipper.ZIP_DEFLATED,
                allowZip64=True,
                encryption=pyzipper.WZ_AES,
            ) as zf:
                # Add a protected file to the zip archive
                friendly_name = f"{folder} ({date.today()})"
                zf.comment = friendly_name.encode()  # run name
                if (
                    False and UserProfiles.count() > 0 and enabled == False
                ):  # NEVER do this (is this still CFR compliant?)
                    # create a protected archive
                    zf.setpassword(hashlib.sha256(zf.comment).hexdigest().encode())
                else:
                    zf.setencryption(None)
                    if enabled:
                        Log.w("Developer Mode is ENABLED - NOT encrypting ZIP file")

                copy_file = poi_path
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                # os.remove(copy_file) # do not remove POIs file

                copy_file = result_path
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_0.pdf")
                if os.path.exists(copy_file):  # Drop effect figure (optional)
                    zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                    os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_1.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_2.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_3.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                copy_file = export_path.replace(".csv", "_4.pdf")
                zf.write(copy_file, arcname=os.path.split(copy_file)[1])
                os.remove(copy_file)

                this_poi_csv_crc = str(hex(zf.getinfo(os.path.split(poi_path)[1]).CRC))
                this_out_csv_crc = str(hex(zf.getinfo(os.path.split(export_path)[1]).CRC))
                this_files_count = len(zf.namelist())

                Log.i(f"Compressed exported files to ZIP:\n\t{zn}")

            self.update(status_label)

            if last_zn != None and self.parent.option_remove_dups.isChecked():
                Log.d("Checking for duplicate analysis output file...")
                with pyzipper.AESZipFile(
                    last_zn,
                    "r",
                    compression=pyzipper.ZIP_DEFLATED,
                    allowZip64=True,
                    encryption=pyzipper.WZ_AES,
                ) as zf:
                    try:
                        last_poi_csv_crc = str(hex(zf.getinfo(os.path.split(poi_path)[1]).CRC))
                        last_out_csv_crc = str(hex(zf.getinfo(os.path.split(export_path)[1]).CRC))
                        last_files_count = len(zf.namelist())
                    except Exception as e:
                        Log.w(f"Error checking prior archive: {str(e)}")
                        last_poi_csv_crc = 0
                        last_out_csv_crc = 0
                        last_files_count = 0

                    Log.d(f"Last poi.csv CRC: {last_poi_csv_crc}")
                    Log.d(f"This poi.csv CRC: {this_poi_csv_crc}")
                    Log.d(f"Last out.csv CRC: {last_out_csv_crc}")
                    Log.d(f"This out.csv CRC: {this_out_csv_crc}")
                    Log.d(f"Last files count: {last_files_count}")
                    Log.d(f"This files count: {this_files_count}")

                    if (
                        last_poi_csv_crc == this_poi_csv_crc
                        and last_out_csv_crc == this_out_csv_crc
                        and last_files_count == this_files_count
                    ):
                        Log.w("Removing duplicate analysis output file.")
                        Log.w(f"See prior analyze output file:\n\t{last_zn}")
                        os.remove(zn)  # duplicate

            # self.progress.emit(99, "Showing Results...")
            status_label = "Showing Results..."
            self.update(status_label)
            Log.i("\tDONE!")
            # sub = threading.Thread(target=export_figures, args=(fig,fig2,fig3,fig4,))
            # sub.start()
            # sub.join()

            i = 0
            Log.d("Waiting for progressbar...")
            while (
                self.parent.progress_value_scanning and i < 300
            ):  # wait for progressBar to reach 99%
                i += 1
                QtCore.QCoreApplication.processEvents()

            Log.i("Showing results...")
            # sleep(1)

            self.parent.results_split.setEnabled(True)
            # self.parent.widget_h4.setStyleSheet("background-color: #ffffff; color: #515151")

            try:
                # NOTE: Creation of `data`, `rows`, `cols`, `summary_text` and `plot_text` moved to section
                #       "Annotate average viscosity and standard deviation on plot and in output CSV" above

                # Add summary text to bottom of table data
                table_layout = QtWidgets.QVBoxLayout()
                tableWidgetWithFooter = QtWidgets.QWidget()
                tableWidgetWithFooter.setLayout(table_layout)
                tableWidget = TableView(data, rows, cols)
                # tableWidget.setStyleSheet("QScrollBar:vertical { width: 15px; }")
                # tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                table_layout.addWidget(tableWidget)
                tableLabel = QtWidgets.QLabel(summary_text)
                tableLabel.setStyleSheet(
                    "font-family: Roboto, Arial, Calibri, sans-serif; font-size: 12pt; font-weight: bold;"
                )
                tableLabel.setWordWrap(True)
                table_layout.addWidget(tableLabel)
                self.parent.results_split.replaceWidget(0, tableWidgetWithFooter)
                self.parent.results_split.setSizes(self.parent.get_results_split_auto_sizes())

            except Exception as e:
                Log.e("Failed to show average viscosity summary.", str(e))

            # add figure to plot view of results
            sc = FigureCanvasQTAgg(fig4)
            mp_toolbar = NavigationToolbar(sc, self.parent)
            mp_layout = QtWidgets.QVBoxLayout()
            mp_layout.addWidget(mp_toolbar)
            mp_layout.addWidget(sc)
            plotWidget = QtWidgets.QWidget()
            plotWidget.setLayout(mp_layout)
            self.parent.results_split.replaceWidget(1, plotWidget)

            # force layout redraw now, and on any resize event
            sc.draw_idle()
            self._results_resize_filter = ResizeFilter(self, sc)
            sc.installEventFilter(self._results_resize_filter)

            # all_figs = [fig4] # [fig,fig2,fig3,fig4]
            # for f in all_figs:
            #     f.show()
            #     mngr = f.canvas.manager     # plt.get_current_fig_manager()
            #     mngr.window.showMaximized() # setGeometry(7, 30, 1503, 742)

            self.progress.emit(
                100, "Finishing..."
            )  # will change to "Progress: Finished" once finished() handlers fire
            Log.i("Analyze process finished.")
            self._exitSuccess = True

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

        finally:
            self.finished.emit()  # queue callback

    def place_text_avoiding_data(
        self,
        ax=None,
        text=None,
        candidates=None,
        pad=None,
        color=None,
        fontsize=None,
        bbox=False,
        return_all_scores=False,
    ):
        """
        Place text in an Axes while avoiding overlap with plotted data.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        text : str
        candidates : list of dict
            Each dict must contain:
                {
                    "pos": (x, y) in axes-fraction coords,
                    "ha": horizontal alignment,
                    "va": vertical alignment,
                    "priority": lower = more preferred
                }
        pad : int or None
            pixel padding around detected overlaps
        color : int or None
            color to use when drawing the overlay text
        fontsize : int or None
            fontsize to use when drawing the overlay text
        bbox : bool
            draw background box around overlay text
        return_all_scores : bool
            debug option

        Returns
        -------
        matplotlib.text.Text
        """

        if ax is None:
            ax = getattr(self, "plot_ax", None)

        if text is None:
            text = getattr(self, "plot_text", None)

        if None in [ax, text]:
            return None

        if candidates is None:
            candidates = [
                {"pos": (0.50, 0.95), "ha": "center", "va": "top", "priority": 0},
                {"pos": (0.50, 0.05), "ha": "center", "va": "bottom", "priority": 1},
                {"pos": (0.95, 0.95), "ha": "right", "va": "top", "priority": 2},
                {"pos": (0.05, 0.05), "ha": "left", "va": "bottom", "priority": 3},
                {"pos": (0.95, 0.05), "ha": "right", "va": "bottom", "priority": 4},
                {"pos": (0.05, 0.95), "ha": "left", "va": "top", "priority": 5},
            ]

        # Sort candidates by priority (ascending)
        candidates = sorted(candidates, key=lambda x: x["priority"])

        if pad is None:
            pad = 10

        if color is None:
            color = "blue"

        if fontsize is None:
            fontsize = 10

        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # ---- collect data in display coords ----
        data_pts = []

        for line in ax.lines:
            # scatter plots are NOT lines; lines are lines
            try:
                x, y = line.get_data()
                if len(x) == 0:
                    continue

                pts = ax.transData.transform(np.column_stack([x, y]))
                data_pts.append(pts)

            except Exception:
                # ignore things that aren't lines or otherwise fail
                pass

        for col in ax.collections:
            # scatter plots are PathCollections
            try:
                offsets = col.get_offsets()
                if len(offsets) == 0:
                    continue

                pts = ax.transData.transform(offsets)
                data_pts.append(pts)

            except Exception:
                # ignore non-scatter collections (e.g., contours, bars, etc.)
                pass

        def score_bbox(bbox):
            """Count how many data points fall inside bbox."""
            if not data_pts:
                return 0

            pts = np.vstack(data_pts)
            x0, y0, x1, y1 = bbox

            inside = (
                (pts[:, 0] >= x0 - pad)
                & (pts[:, 0] <= x1 + pad)
                & (pts[:, 1] >= y0 - pad)
                & (pts[:, 1] <= y1 + pad)
            )
            return int(np.sum(inside))

        # delete best artist from prior redraws
        if getattr(self, "plot_text_obj", None):
            self.plot_text_obj.remove()

        # ---- selection tracking ----
        best_artist = None
        best_score = float("inf")
        best_priority = float("inf")
        debug_scores = []

        for c in candidates:
            x, y = c["pos"]
            ha = c.get("ha", "center")
            va = c.get("va", "center")
            priority = c.get("priority", 0)

            t = ax.text(
                x,
                y,
                text,
                transform=ax.transAxes,
                ha=ha,
                va=va,
                color=color,
                fontsize=fontsize,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none") if bbox else None,
                zorder=10,
            )

            # force render so bbox is correct
            fig.canvas.draw()
            bb = t.get_window_extent(renderer)

            score = score_bbox((bb.x0, bb.y0, bb.x1, bb.y1))

            debug_scores.append((c, score))

            # ---- lexicographic selection ----
            if (score < best_score) or (score == best_score and priority < best_priority):
                if best_artist is not None:
                    best_artist.remove()
                best_artist = t
                best_score = score
                best_priority = priority
            else:
                t.remove()

            # stop early if nothing better can be found
            if not return_all_scores and score == 0:
                break

        # force render so final candidate is removed
        fig.canvas.draw()

        # store best artist for later redraws
        self.plot_text_obj = best_artist

        if return_all_scores:
            return best_artist, debug_scores
        return best_artist

    def get_point_index_from_shear_rate(self, shear_rate):
        idx = 2
        try:
            if hasattr(self, "last_shear_rates") and hasattr(self, "last_distances"):
                # initial_fill_pts = len(self.last_shear_rates) - len(self.last_distances)
                shear_index = np.where(self.last_shear_rates == shear_rate)[0][0]
                if shear_index != len(self.last_shear_rates) - 1:
                    idx = max(3, 7 - shear_index)
        except:
            Log.w("An exception occurred while updating the index on annotation text.")
        return idx - 2

    def update_annot(self, child, ind):
        """Define the update function"""
        ax = getattr(self, "plot_ax", None)
        if ax is None:
            Log.w("No points to annotate.")
            return
        point_labels = {
            0: "High-Shear",
            1: "Initial Fill",
            2: "End of Initial",
            3: "Channel 1 Fill",
            4: "Channel 2 Fill",
            5: "Channel 3 Fill",
        }
        # If the child is a Scatter plot (PathCollection)
        if hasattr(child, "get_offsets"):
            pos = child.get_offsets()[ind["ind"][0]]
        # If the child is a Line plot (Line2D)
        elif hasattr(child, "get_data"):
            xdata, ydata = child.get_data()
            idx = ind["ind"][0]
            pos = (xdata[idx], ydata[idx])
        else:
            return
        if tuple(pos) == (0, 0):
            # Log.d("Suppressed annotation update on errorbars hover event.")
            return
        idx = self.get_point_index_from_shear_rate(pos[0])
        self.annot.xy = pos
        self.annot.set_text(
            f"POI: {idx:.0f}\n{point_labels[idx]}\n{pos[0]:.2f} S⁻¹\n{pos[1]:.2f} cP\n(Click to Modify)"
        )
        self.annot.get_bbox_patch().set_facecolor("lightblue")

    def _click(self, event):
        ax = getattr(self, "plot_ax", None)
        if ax is None:
            Log.w("No points to annotate.")
            return
        fig = ax.figure
        vis = self.annot.get_visible()
        self.annot.set_visible(not vis)
        fig.canvas.draw_idle()

    def hover(self, event):
        """Define the hover event listener"""
        ax = getattr(self, "plot_ax", None)
        if ax is None:
            Log.w("No points to annotate.")
            return
        fig = ax.figure
        vis = self.annot.get_visible()
        if event.inaxes == ax:
            # Loop through all children elements on the axes
            for child in ax.get_children():
                # Check if the child supports the 'contains' method
                if hasattr(child, "contains") and child != self.annot:
                    cont, ind = child.contains(event)
                    if cont:
                        # Update your annotation using the found child
                        self.update_annot(child, ind)
                        # self.annot.set_visible(True)
                        if vis:
                            fig.canvas.draw_idle()
                        return  # Exit once you find the hovered item
        # Hide annotation if mouse is not over any valid child
        if vis:
            self.annot.set_visible(False)
            fig.canvas.draw_idle()

    def on_annot_click(self, event):
        ax = getattr(self, "plot_ax", None)
        if ax is None:
            Log.w("No points to annotate.")
            return
        fig = ax.figure
        # Verify the clicked item is our specific annotation box
        if event.artist == self.annot:
            # Log.i(f"Success! You clicked the label. Current text: '{self.annot.get_text()}'")
            # Modify the label to visually show it was clicked
            # self.annot.get_bbox_patch().set_facecolor("lightgreen")
            # fig.canvas.draw_idle()
            self.parent.allow_modify = self.parent.tool_Modify.isChecked()
            if not self.parent.allow_modify:
                self.parent.tool_Modify.setChecked(True)
                self.parent.allow_modify = self.parent.tool_Modify.isChecked()
            idx = int(self.annot.get_text().splitlines()[0].split()[1]) + 2
            self.parent.gotoStepNum(None, idx)

    def update(self, status):
        try:
            from inspect import currentframe, getframeinfo

            frameinfo = getframeinfo(currentframe().f_back)
            # print(frameinfo.filename, frameinfo.lineno)
            start = self.run.__code__.co_firstlineno
            stop = self.update.__code__.co_firstlineno
            pct = 100 * (frameinfo.lineno - start) / (stop - start)
            # Log.i(f"line #: {start}, {frameinfo.lineno}, {stop}, {pct}%")
            self.progress.emit(int(pct), status)

        except:
            limit = None
            t, v, tb = sys.exc_info()
            from traceback import format_tb

            a_list = ["Traceback (most recent call last):"]
            a_list = a_list + format_tb(tb, limit)
            a_list.append(f"{t.__name__}: {str(v)}")
            for line in a_list:
                Log.e(line)

    def correctHighShear(self, initial, visc):
        output = initial
        try:
            # NOTE: Even when frozen the working path will have resource files
            with open(os.path.join("QATCH", "resources", "lookup_shear_correction.csv"), "r") as f:
                data = np.loadtxt(f.readlines(), delimiter=",", skiprows=1)
                col = 1 if initial == 5e6 else 2
                lookup_visc = data[:, 0]
                lookup_freq = data[:, col]
                nearest_idx = (np.abs(lookup_visc - visc)).argmin()
                correction_factor = lookup_freq[nearest_idx]
                output *= correction_factor
        except Exception as e:
            Log.e("ERROR:", e)
        return np.round(output, 2)

class ResizeFilter(QtCore.QObject):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self._draw_pending = False
        self._draw_delay = 250  # ms

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Resize:
            self._resize_time = monotonic()
            if not self._draw_pending:
                self._draw_pending = True
                QtCore.QTimer.singleShot(self._draw_delay, self._draw_idle)
        return super().eventFilter(obj, event)

    def _draw_idle(self):
        # convert secs -> ms: compare ms to ms
        if (monotonic() - self._resize_time) * 1000 < self._draw_delay:
            # resize event still occurring, try again later
            QtCore.QTimer.singleShot(self._draw_delay, self._draw_idle)
        else:
            # resize event finished, hysteresis elapsed: redraw!
            self.worker.place_text_avoiding_data()
            self._draw_pending = False
