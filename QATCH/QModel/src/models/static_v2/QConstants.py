import multiprocessing

NUM_THREADS = multiprocessing.cpu_count()
print(f"[INFO] Available {NUM_THREADS} threads.")
""" The percentage of data to include in the validation set. """
VALID_SIZE = 0.20
""" The percentage of data to include in the test set. """
TEST_SIZE = 0.20
""" The number of folds to train. """
NUMBER_KFOLDS = 5
""" A random seed to set the state of QModel to. """
SEED = 42
""" The number of rounds to boost for. """
MAX_ROUNDS = 1000
OPT_ROUNDS = 1000
""" Acceptable number of early stopping rounds. """
EARLY_STOP = 50
""" The number of rounds after which to print a verbose model evaluation. """
VERBOSE_EVAL = 50
""" The target supervision feature. """
DISTANCES = [1.15, 1.61, 2.17, 5.00, 10.90, 16.2]
""" The maximum number of candidates to return points and confidences for. """
MAX_GUESSES = 10
""" The label=0 is a 'good' target label. """
GOOD_LABEL = 0
""" The label=1 is a 'bad' target label. """
BAD_LABEL = 1
""" Flag for plotting results. """
PLOTTING = True
""" Flag for running training. """
TRAINING = False
TESTING = True

""" Size of the batch to train/test on. """
BATCH_SIZE = 0.8

DROPBOX_DUMP_PATH = "content/dropbox_dump"
""" Path to 'good' training datasets. """

""" QModel and QMultiModel features """
FEATURES = [
    "Relative_time",
    "Resonance_Frequency",
    "Dissipation",
    "Difference",
    "Cumulative",
    "Dissipation_super",
    "Difference_super",
    "Cumulative_super",
    "Resonance_Frequency_super",
    "Dissipation_gradient",
    "Difference_gradient",
    "Resonance_Frequency_gradient",
    "Cumulative_detrend",
    "Dissipation_detrend",
    "Resonance_Frequency_detrend",
    "Difference_detrend",
]
""" QGBClassifier features """
GB_FEATURES = [
    "Approx_Entropy",
    "Autocorrelation",
    "First_Derivative_Mean",
    "Max_Amp",
    "Max_Time",
    "Mean_Absolute_Deviation",
    "Min_Amp",
    "N_Peaks",
    "PTP_Jitter",
    "RMS_Jitter",
    "Second_Derivative_Mean",
    "Shannon_Entropy",
    "Signal_Energy",
    "Variance",
    "Wavelet_Energy",
    # "Zero_Crossing_Rate",
]
""" QGB Target class. """
GB_TARGET = "Class"
""" Multi-model target class"""
M_TARGET = "Class"
""" Single target classes. """
S_TARGETS = ["Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]
