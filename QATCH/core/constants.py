from enum import Enum
from time import strftime, localtime
import os

from QATCH.common.architecture import Architecture,OSType

###############################################################################
# Enum for the types of sources. Indices MUST match app_sources constant
###############################################################################
class OperationType(Enum):
    calibration = 0
    measurement = 1


###############################################################################
# Specifies the minimal Python version required
###############################################################################
class MinimalPython:
    major = 3
    minor = 10
    release = 0


###############################################################################
# Common constants and parameters for the application.
###############################################################################
class Constants:

    ##########################
    # APPLICATION parameters #
    ##########################
    app_title = "QATCH nanovisQ Real-Time GUI"
    app_version = "v2.6b36"
    app_date = "2024-05-23"
    app_sources = ["Calibration Qatch Q-1 Device", "Measurement Qatch Q-1 Device"]
    app_publisher = "QATCH"
    app_name = "nanovisQ"
    app_encoding = "utf-8"

    ########################
    # RECOMMENDED firmware #
    ########################
    # best_fw_version = "v2.6b32"
    best_fw_version = app_version # may specify an exact version if needed
    do_legacy_updates = False # only use on FW v2.5b23 or older; will break newer devices!

    ###########################
    # FREQ HOPPING parameters #
    ###########################
    base_overtone_freq = 25 # Number of baseline frequencies to collect before collecting upper and lower freqs.
    initial_settle_samples = 100 # Throw away this many samples before allowing calculated data to be displayed.
    downsample_after = 90 # Downsample captures after this many seconds
    downsample_file_count = 20 # Downsample at least this many sample captures (with averaging)
    downsample_plot_count = 3  # Only plot every X samples to the real-time view (no averaging)

    ###########################
    # DISSIPATION conversions #
    ###########################
    dissipation_factor_1st_mode = 2.4942e-4
    dissipation_factor_3rd_mode = 1.4400e-4
    dissipation_factor_5th_mode = 1.1154e-4

    ###########################
    # FW FILTERING parameters #
    ###########################
    # NOTE: avg_in & avg_out  #
    # indicate 'no averaging' #
    # if/when value is '<= 5' #
    # *** Min value is 5! *** #
    ###########################
    avg_in = 35               # raw input averaging (value too low will cause false peaks)
    avg_out = 5               # raw averaging of peak, left, right and magnitude outputs
    step_size = 2             # frequency sweep step size (smaller = more accurate, but slower)   step 3: 4 Hz noise, 4 ms speed
    max_drift_l_hz = 10000    # allowed frequency drift LEFT from initial peak detection (in Hz)
    max_drift_r_hz = 10000    # allowed frequency drift RIGHT from initial peak detection (in Hz)
    track_width_db = -1.50    # how far left-and-right of peak to track (in dB) before switching

    ###########################
    # MAX dissipation by mode #
    ###########################
    max_dissipation_1st_mode = 1000e-6
    max_dissipation_3rd_mode = 0650e-6
    max_dissipation_5th_mode = 1000e-6

    ###################
    # TEMP parameters #
    ###################
    temp_offset_both = +0.00 # deg C
    temp_offset_heat = -0.00 # deg C
    temp_offset_cool = +0.00 # deg C

    #######################
    # Tune PID parameters #
    #######################
    # NOTE: 0 means 'not used' for all PID params
    # cool mode PID:
    tune_pid_cp = 30
    tune_pid_ci = 0.12
    tune_pid_cd = 0
    # heat mode PID:
    tune_pid_hp = 20
    tune_pid_hi = 0.045
    tune_pid_hd = 0

    ###################
    # PLOT parameters #
    ###################
    plot_update_ms = 100#16
    plot_colors = ['#ff0000', '#0072bd', '#4d90ee', '#edb120', '#7e2f8e', '#77ac30', '#4dbeee', '#a2142f']
    plot_max_lines = len(plot_colors)
    plot_min_range_freq = 1000 # Hz
    plot_min_range_diss = 10e-6 # Hz

    # SHOW PHASE [None/True/False]
    # None = let FW decide (see #define REPORT_PHASE)
    # True = force phase to be shown
    # False = force phase to be hidden
    plot_show_phase = False

    ######################
    #  SAMPLE parameters #
    ######################
    argument_default_samples = 51 #1001 63#
    # Savitzky-Golay order of the polynomial fit (common for all)
    SG_order = 3
    # Savitzky-Golay size of the data window (common for all)
    SG_window_size = 25
    # Spline smoothing factor (common for all)
    Spline_factor = 0.01

    ##########################
    # SERIAL PORT parameters #
    ##########################
    serial_default_speed = 2000000
    serial_default_overtone = None
    serial_default_QCS = "@5MHz"
    serial_writetimeout_ms = 3
    serial_timeout_ms = 10#0.01
    serial_simulate_device = False

    ########################
    # MAX SPEED parameters #
    ########################
    max_speed_single = 3000 # 3000 = 3 ms; 0 means full speed ahead, fast as you can
    max_speed_multi4 = 25000 # 30 ms for multi (4x) systems

    ##################
    # LOG parameters #
    ##################
    log_export_path = "logged_data"
    log_filename = "{}.log".format(app_title)
    log_max_bytes = 5120
    log_default_console_log = True

    ######################################
    # FILE parameters for exporting data #
    ######################################
    # sets the slash depending on the OS types
    if Architecture.get_os() is (OSType.macosx or OSType.linux):
       slash="/"
    else:
       slash="\\"

    txt_device_info_filename    = "Device_Info"
    txt_active_device_filename  = "Last_Used"
    tbd_active_device_name_path = "[ACTIVE]"

    csv_delimiter = "," # for splitting data of the serial port and CSV file storage
    csv_default_prefix = "%y%m%d_%H%M%S"
    csv_extension = "csv"
    txt_extension = "txt"
    csv_export_path = "logged_data"
    csv_filename = (strftime(csv_default_prefix, localtime()))
    csv_sweeps_export_path = "{}{}{}{}{}".format(csv_export_path,slash,tbd_active_device_name_path,slash,csv_filename)
    csv_sweeps_filename = "sweep"

    # Calibration: scan (WRITE for @5MHz and @10MHz QCS) path: 'common\'
    csv_calibration_filename    = "Calibration_5MHz"
    csv_calibration_filename10  = "Calibration_10MHz"

    local_app_data_path =  os.path.expandvars(r"%LOCALAPPDATA%{0}{1}{0}{2}".format(slash,app_publisher,app_name))
    csv_calibration_export_path = os.path.join(local_app_data_path, "config")
    user_profiles_path = os.path.join(local_app_data_path, "profiles", "users")
    run_profiles_path = os.path.join(local_app_data_path, "profiles", "runs") # future use
    query_info_recall_path = os.path.join(local_app_data_path, "recall.xml")
    user_constants_path = os.path.join(local_app_data_path, "settings", "userConstants.py")

    ##################
    # Calibration: baseline correction (READ for @5MHz and @10MHz QCS) path: 'common\'
    csv_calibration_path   = "{}{}{}{}{}.{}".format(csv_calibration_export_path,slash,tbd_active_device_name_path,slash,csv_calibration_filename,txt_extension)
    csv_calibration_path10 = "{}{}{}{}{}.{}".format(csv_calibration_export_path,slash,tbd_active_device_name_path,slash,csv_calibration_filename10,txt_extension)

    # Frequencies: Fundamental and overtones (READ and WRITE for @5MHz and @10MHz QCS)
    csv_peakfrequencies_filename   = "PeakFrequencies"
    cvs_peakfrequencies_path    = "{}{}{}{}{}.{}".format(csv_calibration_export_path,slash,tbd_active_device_name_path,slash,csv_peakfrequencies_filename,txt_extension)
    #########################

    # Temp file to store paths of all log files created during a run
    new_files_path = csv_export_path + slash + "new_files.txt"

    # Log file for storing the output of the TEC temperature controller
    tec_log_path = csv_export_path + slash + tbd_active_device_name_path + slash + "output_tec.csv"

    ##########################
    # CALIBRATION parameters #
    ##########################
    calibration_default_samples = 100001
    calibration_frequency_start =  2500000 # 1000000
    calibration_frequency_stop  = 17500000 # 51000000
    calibration_fStep = (calibration_frequency_stop - calibration_frequency_start) / (calibration_default_samples-1)
    calibration_readFREQ = list(range(calibration_frequency_start, calibration_frequency_stop+1, int(calibration_fStep)))
    # Peak Detection - distance in samples between neighbouring peaks
    dist5  =  int(3000000/calibration_fStep) # for @5MHz
    dist10 =  10000 # for @10MHz

    ##########################
    # RING BUFFER parameters #
    ##########################
    ring_buffer_samples = 6000 # @ 50 ms/sample = 5 mins history

    ########################
    # AVERAGING parameters #
    ########################
    environment = 2
    SG_order_environment = 1
    SG_window_environment = 3

    ###################
    # ADC conversions #
    ###################
    vmax = 3.3
    bitmax = 8191 #1023 # 10-bit ADC
    ADCtoVolt = vmax / bitmax
    VCP = 0.9

    ######################
    # ANALYZE parameters #
    ######################
    export_file_format = "_analyze_out.csv"
    channel_thickness = 2.25e-6
    smooth_factor_ratio = 0.75
    super_smooth_factor_ratio = 25.0
    baseline_smooth = 9 # must be an odd integer, greater than 1
    consider_points_above_pct = 0.60
    default_diff_factor = 2.0
    temp_adjusted_CA_factor = -1.15

    ######################
    # BATCH # parameters #
    ######################
    CA_offset = 0
    freq_delta_15MHz = 400
    freq_factor_15MHz = 0.0147
    diss_factor1_15MHz = 0.1934e6
    diss_factor2_15MHz = 0
    freq_delta_5MHz = calibration_frequency_stop # super large number (always use dissipation method)
    freq_factor_5MHz = 0.0147 # never used
    diss_factor1_5MHz = 0.2222e6
    diss_factor2_5MHz = 1.67
    distances = "[1.15, 1.61, 2.17, 2.67, 3.23, 5.00, 10.90, 16.2]"

    @staticmethod
    def get_batch_param(batch, param = ""):
        # Returns the found parameter for the (batch,param) pair as a string.
        # Unless, if you only supply a 'batch', it returns True/False if found.
        # Caller MUST cast the returned string to the desired data type!
        from QATCH.common.logger import Logger as Log
        import numpy as np

        params = [""]
        batches = [""]
        try:
            # pull in lookup table from resources
            working_resource_path = os.path.join(os.getcwd(), "QATCH/resources/") # prefer working resource path, if exists
            # bundled_resource_path = os.path.join(Architecture.get_path(), "QATCH/resources/") # otherwise, use bundled resource path
            resource_path = working_resource_path # if os.path.exists(working_resource_path) else bundled_resource_path
            batch_params_file = os.path.join(resource_path, "lookup_batch_params.csv")
            detected_delimeter = '\t'
            with open(batch_params_file, 'r') as f:
                first_line = f.readline()
                tab_count = first_line.count('\t')
                comma_count = first_line.count(',')
                if comma_count > tab_count:
                    detected_delimeter = ','
            table = np.genfromtxt(batch_params_file, dtype = 'str', delimiter = detected_delimeter, skip_header = 0)
            params = table[0][1:] # first row values, without A1 cell (empty)
            batches = table[:,0][1:] # first col values, without A1 cell (empty)
            data = table[1:,1:] # data, without header row/col (for indexing)
        except Exception as e:
            Log.e(f"get_batch_param(): ERROR {e}")

        # transform all lookup entries to UPPERCASE
        param_upper = param.upper()
        batch_upper = batch.upper()
        params_orig_case = params
        params = np.char.upper(params)
        batches = np.char.upper(batches)

        # figure out where to do the lookup
        param_in_py_file = False
        param_in_csv_file = False
        batch_in_csv_file = False

        # is the param defined in Constants.py?
        if hasattr(Constants, param):
            param_in_py_file = True
        # is the param defined in lookup_BATCH#.csv?
        if param_upper in params and param != "":
            param_in_csv_file = True
        # is the batch defined in lookup_BATCH#.csv?
        if batch_upper in batches and batch != "":
            batch_in_csv_file = True
        # if only given 'batch', return True/False if it's found in CSV
        if param == "":
            return batch_in_csv_file
        # if given 'param' ALL, return entire list of params for 'batch'
        if param == "ALL":
            if batch_in_csv_file:
                idx_of_batch = np.where(batches == batch_upper)[0] # batches.index(batch)
                all_params = data[idx_of_batch[0]]
                return dict(zip(params_orig_case, all_params))
            else:
                defaults = {}
                for p in params_orig_case:
                    defaults[p] = getattr(Constants, p, 0)
                return defaults

        # print(f"param_in_py_file: {param_in_py_file}")
        # print(f"param_in_csv_file: {param_in_csv_file}")
        # print(f"batch_in_csv_file: {batch_in_csv_file}")

        # param does not exist, anywhere
        if param_in_py_file == False and (param_in_csv_file == False or batch_in_csv_file == False):
            default_val = 0 # getattr(Constants, param)
            Log.e(f"get_batch_param(): PARAM '{param}' is not found (using {default_val}). Please specify a default value in Constants.py.")
            return str(default_val)

        # param only exists in Constants.py
        if param_in_py_file == True and (param_in_csv_file == False or batch_in_csv_file == False):
            default_val = getattr(Constants, param)
            Log.w(f"get_batch_param(): PARAM '{param}' is not found for BATCH '{batch}' (using {default_val}). Please add the batch/param to lookup_BATCH#.csv.")
            return str(default_val)

        # batch and param exists in lookup_BATCH, so return the lookup pair
        idx_of_param = np.where(params == param_upper)[0] # params.index(param)
        idx_of_batch = np.where(batches == batch_upper)[0] # batches.index(batch)
        found_param = data[idx_of_batch[0], idx_of_param[0]]

        # check for dups and report the warning if so
        if len(idx_of_param) != 1:
            Log.w(f"get_batch_param(): More than one PARAM col {param} found in lookup_BATCH#.py! Using first column.")
        if len(idx_of_batch) != 1:
            Log.w(f"get_batch_param(): More than one BATCH col {batch} found in lookup_BATCH#.py! Using first row.")
        Log.d(f"Found param '{found_param}' from ('{batch}', '{param}') pair in lookup table at idx ({idx_of_batch}, {idx_of_param}).")

        # return the found param as a string
        # caller must cast to desired type
        return str(found_param)


# Moved to mainWindow.py:
# ###############################################################################
# #  Provides a date-time aware axis
# ###############################################################################
# class DateAxis(AxisItem):
#     def __init__(self, *args, **kwargs):
#         super(DateAxis, self).__init__(*args, **kwargs)
#
#     def tickStrings(self, values, scale, spacing):
#         try:
#             # If less than 1 hour: display as "MM:SS" format.
#             # If equal or over 1 hour: display as "HH:MM:SS".
#             z= [datetime.datetime.utcfromtimestamp(float(value)).strftime("%M:%S")
#             if datetime.datetime.utcfromtimestamp(float(value)).strftime("%H") == "00"
#             else datetime.datetime.utcfromtimestamp(float(value)).strftime("%H:%M:%S")
#             for value in values]
#         except:
#             z= ''
#         return z
