"""
hypothesis_testing.py

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-28

Version:
   2.0 - Refactored to use area-based CI polygon containment
"""

from typing import Dict, Union

import numpy as np
from shapely.geometry import Polygon, box

try:
    import logging

    from src.models.formulation import Formulation
    from src.models.predictor import Predictor

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )

    class Log:
        """Logging utility for standardized log messages."""

        _logger = logging.getLogger("HypothesisTesting")

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
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.models.formulation import Formulation
    from QATCH.VisQAI.src.models.predictor import Predictor

TAG = "[HypothesisTesting]"


class HypothesisTesting:
    def __init__(self, model_path: str):
        self._predictor = Predictor(model_path)

    def evaluate_hypothesis(
        self,
        formulation: Formulation,
        hypothesis_type: str,
        shear_rates: list[int],
        bounds: tuple,
        ci_range: tuple,
    ) -> Dict:
        """
        Evaluate hypothesis by calculating the area of the CI polygon contained within bounds.

        Args:
            formulation: Formulation object to predict
            hypothesis_type: Type of test - "upper", "lower", or "between"
            shear_rates: List of shear rates (typically [100, 1000, 10000, 100000, 15000000])
            bounds: Tuple of (lower_bound, upper_bound) for the hypothesis test
            ci_range: Confidence interval range for predictions

        Returns:
            Dictionary containing:
                - mean_predictions: Dict mapping shear rate to mean prediction
                - pct_contained: Percentage of CI polygon area within bounds
                - area_contained: Actual area of CI polygon within bounds
                - total_area: Total area of CI polygon
                - bounds: The bounds used for testing
        """
        mean_pred, pred_stats = self._predictor.predict_uncertainty(
            df=formulation.to_dataframe(encoded=False, training=False),
            ci_range=ci_range,
        )
        mean_pred = mean_pred.flatten()
        lower_ci = pred_stats.get("lower_ci", None)
        if lower_ci is None:
            msg = f"Lower {ci_range[0]} could not be retrieved from predictor."
            Log.e(msg)
            raise ValueError(msg)

        upper_ci = pred_stats.get("upper_ci", None)
        if upper_ci is None:
            msg = f"Upper {ci_range[1]} could not be retrieved from predictor."
            Log.e(msg)
            raise ValueError(msg)

        # Convert to numpy arrays
        mean_pred = np.asarray(mean_pred)
        upper_ci = np.asarray(upper_ci)
        lower_ci = np.asarray(lower_ci)
        mean_pred = mean_pred.flatten()
        upper_ci = upper_ci.flatten()
        lower_ci = lower_ci.flatten()
        # Create mean predictions dictionary
        mean_predictions = {sr: mean_pred[i] for i, sr in enumerate(shear_rates)}

        # Route to appropriate test method
        if hypothesis_type == "upper" and bounds[0] == -np.inf:
            result = self._upper_bound_test(
                mean_pred=mean_pred,
                upper_ci=upper_ci,
                lower_ci=lower_ci,
                bounds=bounds,
                shear_rates=shear_rates,
            )
        elif hypothesis_type == "lower" and bounds[1] == np.inf:
            result = self._lower_bound_test(
                mean_pred=mean_pred,
                upper_ci=upper_ci,
                lower_ci=lower_ci,
                bounds=bounds,
                shear_rates=shear_rates,
            )
        elif (
            hypothesis_type == "between" and bounds[0] > -np.inf and bounds[1] < np.inf
        ):
            result = self._between_bound_test(
                mean_pred=mean_pred,
                upper_ci=upper_ci,
                lower_ci=lower_ci,
                bounds=bounds,
                shear_rates=shear_rates,
            )
        else:
            msg = f"Hypothesis type '{hypothesis_type}' with bounds ({bounds[0]}, {bounds[1]}) is unsupported."
            Log.e(msg)
            raise ValueError(msg)

        # Add mean predictions to result
        result["mean_predictions"] = mean_predictions

        return result

    def _calculate_ci_polygon_area(
        self,
        shear_rates: list,
        upper_ci: np.ndarray,
        lower_ci: np.ndarray,
        bounds: tuple,
    ) -> Dict:
        """
        Calculate the area of the CI polygon contained within the specified bounds.

        Uses log10-scale for shear rates to handle the large range (100 to 15,000,000).

        Args:
            shear_rates: List of shear rates
            upper_ci: Upper confidence interval values
            lower_ci: Lower confidence interval values
            bounds: (lower_bound, upper_bound) for clipping

        Returns:
            Dictionary with pct_contained, area_contained, and total_area
        """
        # Convert shear rates to log10 scale for more balanced area calculation
        log_shear_rates = np.log10(shear_rates)

        # Create CI polygon
        # Points go clockwise: upper boundary (left to right), then lower boundary (right to left)
        upper_points = [
            (log_shear_rates[i], upper_ci[i]) for i in range(len(shear_rates))
        ]
        lower_points = [
            (log_shear_rates[i], lower_ci[i])
            for i in range(len(shear_rates) - 1, -1, -1)
        ]

        try:
            ci_polygon = Polygon(upper_points + lower_points)

            # Validate polygon
            if not ci_polygon.is_valid:
                Log.w("CI polygon is invalid, attempting to fix with buffer(0)")
                ci_polygon = ci_polygon.buffer(0)

            total_area = ci_polygon.area

            if total_area == 0:
                Log.w("CI polygon has zero area")
                return {"pct_contained": 0.0, "area_contained": 0.0, "total_area": 0.0}

            # Create bounding box for clipping
            lower_bound, upper_bound = bounds
            min_x = min(log_shear_rates)
            max_x = max(log_shear_rates)

            # Handle infinite bounds
            if lower_bound == -np.inf:
                lower_bound = min(lower_ci) - abs(min(lower_ci)) * 0.1  # Extend below
            if upper_bound == np.inf:
                upper_bound = max(upper_ci) + abs(max(upper_ci)) * 0.1  # Extend above

            # Create bounding box and find intersection
            bounding_box = box(min_x, lower_bound, max_x, upper_bound)
            clipped_polygon = ci_polygon.intersection(bounding_box)

            # Calculate clipped area
            if clipped_polygon.is_empty:
                clipped_area = 0.0
            else:
                clipped_area = clipped_polygon.area

            pct_contained = (
                (clipped_area / total_area * 100.0) if total_area > 0 else 0.0
            )

            return {
                "pct_contained": pct_contained,
                "area_contained": clipped_area,
                "total_area": total_area,
            }

        except Exception as e:
            Log.e(f"Error calculating polygon area: {str(e)}")
            return {"pct_contained": 0.0, "area_contained": 0.0, "total_area": 0.0}

    def _upper_bound_test(
        self,
        mean_pred: np.ndarray,
        upper_ci: np.ndarray,
        lower_ci: np.ndarray,
        bounds: tuple,
        shear_rates: list,
    ) -> Dict:
        """
        Test if CI polygon is below the upper bound threshold.

        Args:
            mean_pred: Mean predictions
            upper_ci: Upper confidence interval
            lower_ci: Lower confidence interval
            bounds: (-inf, upper_threshold)
            shear_rates: List of shear rates

        Returns:
            Dictionary with test results
        """
        upper_threshold = bounds[1]

        # For upper bound test, we clip from -inf to upper_threshold
        # Use the minimum lower_ci as the effective lower bound
        effective_lower = min(lower_ci) - abs(min(lower_ci)) * 0.1
        test_bounds = (effective_lower, upper_threshold)

        result = self._calculate_ci_polygon_area(
            shear_rates=shear_rates,
            upper_ci=upper_ci,
            lower_ci=lower_ci,
            bounds=test_bounds,
        )

        result["bounds"] = bounds
        result["test_type"] = "upper"

        return result

    def _lower_bound_test(
        self,
        mean_pred: np.ndarray,
        upper_ci: np.ndarray,
        lower_ci: np.ndarray,
        bounds: tuple,
        shear_rates: list,
    ) -> Dict:
        """
        Test if CI polygon is above the lower bound threshold.

        Args:
            mean_pred: Mean predictions
            upper_ci: Upper confidence interval
            lower_ci: Lower confidence interval
            bounds: (lower_threshold, inf)
            shear_rates: List of shear rates

        Returns:
            Dictionary with test results
        """
        lower_threshold = bounds[0]

        # For lower bound test, we clip from lower_threshold to inf
        # Use the maximum upper_ci as the effective upper bound
        effective_upper = max(upper_ci) + abs(max(upper_ci)) * 0.1
        test_bounds = (lower_threshold, effective_upper)

        result = self._calculate_ci_polygon_area(
            shear_rates=shear_rates,
            upper_ci=upper_ci,
            lower_ci=lower_ci,
            bounds=test_bounds,
        )

        result["bounds"] = bounds
        result["test_type"] = "lower"

        return result

    def _between_bound_test(
        self,
        mean_pred: np.ndarray,
        upper_ci: np.ndarray,
        lower_ci: np.ndarray,
        bounds: tuple,
        shear_rates: list,
    ) -> Dict:
        """
        Test if CI polygon is between the lower and upper bound thresholds.

        Args:
            mean_pred: Mean predictions
            upper_ci: Upper confidence interval
            lower_ci: Lower confidence interval
            bounds: (lower_threshold, upper_threshold)
            shear_rates: List of shear rates

        Returns:
            Dictionary with test results
        """
        result = self._calculate_ci_polygon_area(
            shear_rates=shear_rates, upper_ci=upper_ci, lower_ci=lower_ci, bounds=bounds
        )

        result["bounds"] = bounds
        result["test_type"] = "between"

        return result
