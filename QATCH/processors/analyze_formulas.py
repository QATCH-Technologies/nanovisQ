import numpy as np
from numpy import loadtxt

from QATCH.common.architecture import Architecture
from QATCH.common.fileStorage import secure_open
from QATCH.common.logger import Logger as Log
from QATCH.core.constants import Constants
from QATCH.QModel import QModelTweed


class AnalyzeFormulas:

    @staticmethod
    def Lookup_ST(surfactant, concentration):
        ST1 = 72

        if concentration <= 123:  # mg/mL
            return ST1

        # See issue #383: Concentration-Based Surface Tension Correction Formula
        correction = np.polyval([0.0016, 0.8026], concentration)
        return round(ST1 / correction, 1)

        # NOTE: The rest of function is not currently used; returning corrected value above
        if concentration > 2:  # mg/mL
            ST1 = 57.5
        return ST1  # always

        if concentration < 0.01:
            ST1 = 71
        else:
            X1 = np.log10(concentration)
            ST1 = np.polyval([-0.9092, -3.5982, 67], X1)
        X2 = np.log10(surfactant / 125)  # NOTE: np.log10(0) = -INF
        if X2 < -6:
            ST2 = ST1
        elif X2 < -5:
            ST2 = ST1 - 1
        elif -5 <= X2 <= -2.8:
            ST2 = ST1 - np.polyval([11.5, 59.5], X2)
        else:  # X2 > -2.8:
            ST2 = ST1 - 27
        # AnalyzeProcess.Lookup_Table("QATCH/resources/lookup_ST.csv", surfactant, concentration)
        return ST2

    @staticmethod
    def Lookup_CA(surfactant, concentration):
        CA = 55
        if concentration > 10:
            CA = CA - 0
        elif concentration >= 1:
            CA = CA - 0
        # AnalyzeProcess.Lookup_Table("QATCH/resources/lookup_CA.csv", surfactant, concentration)
        return CA

    @staticmethod
    def Lookup_DN(surfactant, concentration, stabilizer_type="none", stabilizer_concentration=0):
        stabilizer_offset = 0
        if stabilizer_type == "sucrose":  # expect caller `casefold()` stabilizer type
            stabilizer_offset = 0.13 * stabilizer_concentration
        return 1 + 2.62e-4 * concentration + stabilizer_offset

    @staticmethod
    def Lookup_Table(table_path, surfactant, concentration):
        debug = False
        table = loadtxt(table_path, delimiter="\t")
        log_surfactant = np.log10(surfactant)
        # first row values, without A1 cell (empty)
        pcts = np.log10(table[0][1:])
        cons = table[:, 0][1:]  # first col values, without A1 cell (empty)
        data = table[1:, 1:]  # data, without header row/col (for indexing)
        if debug:
            Log.d(data)

        # find surfactant position in table
        s_idx = [None]
        if surfactant in pcts:
            s_idx = np.where(pcts == log_surfactant)
        elif log_surfactant < pcts[0]:
            s_idx = [0, 1]
        elif log_surfactant > pcts[-1]:
            tmp = len(pcts) - 1
            s_idx = [tmp - 1, tmp]
        else:
            tmp = next(i for i, t in enumerate(pcts) if t > log_surfactant)
            s_idx = [tmp - 1, tmp]
        if debug:
            Log.d(s_idx)

        # find concentration position in table
        c_idx = [None]
        if concentration in cons:
            c_idx = np.where(cons == concentration)
        elif concentration < cons[0]:
            c_idx = [0, 1]
        elif concentration > cons[-1]:
            tmp = len(cons) - 1
            c_idx = [tmp - 1, tmp]
        else:
            tmp = next(i for i, t in enumerate(cons) if t > concentration)
            c_idx = [tmp - 1, tmp]
        if debug:
            Log.d(c_idx)

        ret = 0
        if len(c_idx) == 2 and len(s_idx) == 2:
            # most complex case, extrapolate both ways, then average together
            row1 = data[c_idx[0]]
            if debug:
                Log.d(f"row1 = {row1}")
            s_ratio = (log_surfactant - pcts[s_idx[0]]) / (pcts[s_idx[1]] - pcts[s_idx[0]])
            if debug:
                Log.d(f"s_ratio = {s_ratio}")
            val1 = row1[s_idx[0]] + (row1[s_idx[1]] - row1[s_idx[0]]) * s_ratio
            if debug:
                Log.d(f"val1 = {val1}")
            row2 = data[c_idx[1]]
            if debug:
                Log.d(f"row2 = {row2}")
            val2 = row2[s_idx[0]] + (row2[s_idx[1]] - row2[s_idx[0]]) * s_ratio
            if debug:
                Log.d(f"val2 = {val2}")
            c_ratio = (concentration - cons[c_idx[0]]) / (cons[c_idx[1]] - cons[c_idx[0]])
            if debug:
                Log.d(f"ratio = {c_ratio}")
            ret1 = val1 + (val2 - val1) * c_ratio
            if debug:
                Log.d(f"ret1 = {ret1}")
            col1 = data[:, s_idx[0]]
            if debug:
                Log.d(f"col1 = {col1}")
            val1 = col1[c_idx[0]] + (col1[c_idx[1]] - col1[c_idx[0]]) * c_ratio
            if debug:
                Log.d(f"val1 = {val1}")
            col2 = data[:, s_idx[1]]
            if debug:
                Log.d(f"col2 = {col2}")
            val2 = col2[c_idx[0]] + (col2[c_idx[1]] - col2[c_idx[0]]) * c_ratio
            if debug:
                Log.d(f"val2 = {val2}")
            ret2 = val1 + (val2 - val1) * s_ratio
            if debug:
                Log.d(f"ret2 = {ret2}")
            ret = (ret1 + ret2) / 2
        if len(c_idx) == 2 and len(s_idx) == 1:
            col = data[:, s_idx]
            if debug:
                Log.d(col)
            ratio = (concentration - cons[c_idx[0]]) / (cons[c_idx[1]] - cons[c_idx[0]])
            if debug:
                Log.d(ratio)
            ret = (col[c_idx[0]] + (col[c_idx[1]] - col[c_idx[0]]) * ratio)[0][0]
        if len(c_idx) == 1 and len(s_idx) == 2:
            row = data[c_idx][0]
            if debug:
                Log.d(row)
            ratio = (log_surfactant - pcts[s_idx[0]]) / (pcts[s_idx[1]] - pcts[s_idx[0]])
            if debug:
                Log.d(ratio)
            ret = row[s_idx[0]] + (row[s_idx[1]] - row[s_idx[0]]) * ratio
        if len(c_idx) == 1 and len(s_idx) == 1:
            ret = data[c_idx, s_idx][0][0]

        lookup_type = table_path[table_path.rindex("/") + 1 : -4]
        Log.d(f"{lookup_type}({surfactant:1.3f}, {concentration:3.0f}) = {ret:2.2f}")
        return ret

    @staticmethod
    def run_qmodel_tweed(data_path):
        val = False
        try:
            with secure_open(data_path, "r", "capture") as f:
                csv_headers = next(f)

                if "Ambient" in csv_headers:
                    csv_cols = (2, 4, 6, 7)
                else:
                    csv_cols = (2, 3, 5, 6)

                data = np.loadtxt(f.readlines(), delimiter=",", skiprows=0, usecols=csv_cols)
                relative_time = data[:, 0]
                temperature = data[:, 1]
                resonance_frequency = data[:, 2]
                dissipation = data[:, 3]

                if Constants.qmodel_tweed_predict:
                    try:
                        dataModel = QModelTweed()
                        model_result = dataModel.IdentifyPoints(
                            data_path=data_path,
                            times=relative_time,
                            freq=resonance_frequency,
                            diss=dissipation,
                        )
                        if model_result != -1:
                            val = True

                        return val  # if we got here, skip the rest of this method, return now
                    except:
                        Log.e("Error modeling data... Using 'tensorflow' as a backup (slow).")

                if Constants.TensorFlow_predict:
                    # raw data
                    xs = relative_time
                    ys = dissipation

                    t_0p5 = (
                        0 if (xs[-1] < 0.5) else next((x for x, t in enumerate(xs) if t > 0.5), 0)
                    )
                    t_1p0 = (
                        100
                        if (len(xs) < 500)
                        else next((x for x, t in enumerate(xs) if t > 1.0), 1)
                    )
                    if t_0p5 == t_1p0:
                        t_1p0 = next(
                            (x for x, t in enumerate(xs) if t > xs[t_1p0] + 0.5),
                            t_1p0 + 1,
                        )

                    # t_1p0, done = QtWidgets.QInputDialog.getDouble(None, 'Input Dialog', 'Confirm rough start index:', value=t_1p0)

                    # new maths for resonance and dissipation (scaled)
                    avg = np.average(resonance_frequency[t_0p5:t_1p0])
                    ys = ys * avg / 2
                    # ys_fit = ys_fit * avg / 2
                    ys = ys - np.amin(ys)
                    # ys_fit = ys_fit - np.amin(ys_fit)
                    ys_freq = avg - resonance_frequency
                    # ys_freq_fit = savgol_filter(ys_freq, smooth_factor, 1)

                    # needed for invert curve detection, but use unfitted curves (raw is fine)
                    ys_fit = ys.copy()
                    ys_freq_fit = ys_freq.copy()

                    baseline = np.average(dissipation[t_0p5:t_1p0])
                    diff_factor = Constants.default_diff_factor  # 1.0 if baseline < 50e-6 else 1.5
                    # if hasattr(self, "diff_factor"):
                    #     diff_factor = self.diff_factor
                    ys_diff = (
                        ys_freq - diff_factor * ys
                    )  # NOTE: For temporary testing as of Pi Day 2023! (3 places in this file)

                    # Invert difference curve if drop applied to outlet
                    if np.average(np.abs(ys_freq_fit)) < np.average(
                        np.abs(diff_factor * ys_fit)
                    ) and abs(ys_diff[t_1p0:].min()) > 5 * abs(ys_diff[t_1p0:].max()):
                        Log.w("Inverting DIFFERENCE curve due to negative initial fill deltas")
                        ys_diff *= -1

                    # ys_diff_fit = savgol_filter(ys_diff, smooth_factor, 1)
                    Log.d(f"Difference factor: {diff_factor:1.3f}x")

                    # model trained with 1000 points
                    lin_xs = np.linspace(xs[0], xs[-1], 1000)
                    lin_ys = np.interp(lin_xs, xs, ys)
                    lin_ys_freq = np.interp(lin_xs, xs, ys_freq)
                    lin_ys_diff = np.interp(lin_xs, xs, ys_diff)

                    # lazy load tensorflow module
                    # hide info/warning logs from tf
                    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
                    import tensorflow as tf

                    # import tensorflow, load model, and predict good or bad
                    model_path = os.path.join(
                        Architecture.get_path(), "QATCH", "QModel", "assets", "qmodel_tweed"
                    )
                    time_model = tf.keras.models.load_model(os.path.join(model_path, "time_model"))
                    diss_model = tf.keras.models.load_model(os.path.join(model_path, "diss_model"))
                    freq_model = tf.keras.models.load_model(os.path.join(model_path, "freq_model"))
                    diff_model = tf.keras.models.load_model(os.path.join(model_path, "diff_model"))

                    data_time = lin_xs
                    data_diss = lin_ys
                    data_freq = lin_ys_freq
                    data_diff = lin_ys_diff

                    predict_time = (
                        # max(0, min(1, time_model([data_time]).numpy()[0][0]))
                        0
                    )
                    predict_diss = max(0, min(1, diss_model([data_diss]).numpy()[0][0]))
                    predict_freq = max(0, min(1, freq_model([data_freq]).numpy()[0][0]))
                    predict_diff = max(0, min(1, diff_model([data_diff]).numpy()[0][0]))

                    predictors_count = 3  # ignore time
                    predict_data = (
                        predict_time + predict_diss + predict_freq + predict_diff
                    ) / predictors_count
                    val = max(0, min(1, np.round(predict_data).astype(int)))
        except Exception as e:
            # raise e
            Log.e("ERROR: Model encountered an exception while analyzing run data.")
        return val  # true if good
