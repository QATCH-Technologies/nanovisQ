"""
hypothesis_testing.py

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-28

Version:
   1.0
"""
import numpy as np
from typing import Union
try:
    from src.models.predictor import Predictor
    from src.models.formulation import Formulation
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""
        _logger = logging.getLogger("Predictor")

        @classmethod
        def i(cls, msg: str) -> None:
            """Log an informational message."""
            cls._logger.info(msg)

        @classmethod
        def w(cls, msg: str) -> None:
            """Log a warning message."""
            cls._logger.warning(msg)

        @classmethod
        def e(cls, msg: str) -> None:
            """Log an error message."""
            cls._logger.error(msg)

        @classmethod
        def d(cls, msg: str) -> None:
            """Log a debug message."""
            cls._logger.debug(msg)
except (ImportError, ModuleNotFoundError):
    from QATCH.VisQAI.src.models.predictor import Predictor
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.common.logger import Logger as Log

TAG = "[HypothesisTesting]"


class HypothesisTesting:
    def __init__(self, model_path: str):
        self._predictor = Predictor(model_path)

    def evaluate_hypothesis(self,
                            formulation: Formulation,
                            hypothesis_type: str,
                            shear_rates: list[int],
                            bounds: tuple,
                            ci_range: tuple):
        mean_pred, pred_stats = self._predictor.predict_uncertainty(
            df=formulation.to_dataframe(encoded=False, training=False),
            ci_range=ci_range)
        lower_ci = pred_stats.get("lower_ci", None)
        if lower_ci is None:
            Log.e(
                TAG, f"Lower {ci_range[0]} could not be retrieved from predictor.")

        upper_ci = pred_stats.get("upper_ci", None)
        if upper_ci is None:
            Log.e(
                TAG, f"Upper {ci_range[1]} could not be retrieved from predictor.")

        if hypothesis_type == "upper" and bounds[0] == -np.inf:
            return self._upper_bound_test(mean_pred=mean_pred,
                                          upper_ci=upper_ci,
                                          bounds=bounds,
                                          shear_rates=shear_rates)
        elif hypothesis_type == "lower" and bounds[1] == np.inf:
            return self._upper_bound_test(mean_pred=mean_pred,
                                          upper_ci=upper_ci,
                                          lower_bound=bounds,
                                          shear_rates=shear_rates)
        elif hypothesis_type == "between" and bounds[0] > -np.inf and bounds[1] < np.inf:
            return self._between_bound_test(mean_pred=mean_pred,
                                            upper_ci=upper_ci,
                                            lower_ci=lower_ci,
                                            bounds=bounds,
                                            shear_rates=shear_rates)
        else:
            msg = f"Hypothesis type {hypothesis_type} with CI=({bounds[1]}, {bounds[0]}) is unsupported."
            Log.e(msg)
            raise ValueError(msg)

    def _upper_bound_test(self, mean_pred: Union[list, np.ndarray],
                          upper_ci: Union[list, np.ndarray],
                          bounds: tuple, shear_rates: list) -> dict:

        mean_pred = np.asarray(mean_pred)
        upper_ci = np.asarray(upper_ci)

        results = {}
        upper_threshold = bounds[1]

        for i, sr in enumerate(shear_rates):
            mean_val = mean_pred[i]
            upper_val = upper_ci[i]

            ci_range = abs(upper_val - mean_val)

            if ci_range == 0:
                pct_below = 100.0 if mean_val <= upper_threshold else 0.0
            elif upper_val <= upper_threshold:
                pct_below = 100.0
            elif mean_val >= upper_threshold:
                pct_below = 0.0
            else:
                below_threshold = upper_threshold - mean_val
                pct_below = (below_threshold / ci_range) * 100.0

            results[sr] = {"pct": pct_below,
                           "thresh": upper_threshold,
                           "pred": mean_pred
                           }

        return results

    def _lower_bound_test(self, mean_pred: Union[list, np.ndarray],
                          lower_ci: Union[list, np.ndarray],
                          bounds: tuple, shear_rates: list) -> dict:
        mean_pred = np.asarray(mean_pred)
        lower_ci = np.asarray(lower_ci)

        results = {}
        lower_threshold = bounds[0]

        for i, sr in enumerate(shear_rates):
            mean_val = mean_pred[i]
            lower_val = lower_ci[i]

            ci_range = abs(lower_val - mean_val)

            if ci_range == 0:
                pct_above = 100.0 if mean_val >= lower_threshold else 0.0
            elif lower_val >= lower_threshold:
                pct_above = 100.0
            elif mean_val <= lower_threshold:
                pct_above = 0.0
            else:
                below_threshold = lower_threshold - mean_val
                pct_above = (below_threshold / ci_range) * 100.0

            results[sr] = {"pct": pct_above,
                           "thresh": lower_threshold,
                           "pred": mean_pred
                           }

        return results

    def _between_bound_test(self,  mean_pred: Union[list, np.ndarray],
                            upper_ci: Union[list, np.ndarray],
                            lower_ci: Union[list, np.ndarray],
                            bounds: tuple, shear_rates: list) -> dict:
        pass
