"""
test_metrics.py

Comprehensive unit tests for the Metrics class.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-21

Version:
    1.0
"""
import unittest
import pandas as pd
import numpy as np
from src.utils.metrics import Metrics


class TestMetrics(unittest.TestCase):
    """Test suite for the Metrics class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a sample DataFrame with known values for testing
        self.sample_df = pd.DataFrame({
            'actual': [10, 20, 30, 40, 50],
            'predicted': [12, 18, 32, 38, 51],
            'residual': [2, -2, 2, -2, 1],
            'abs_error': [2, 2, 2, 2, 1],
            'pct_error': [20, 10, 6.67, 5, 2],
            'within_ci': [True, True, False, True, True],
            'cv': [0.1, 0.2, 0.15, 0.25, 0.3],
            'std': [1.0, 1.5, 2.0, 2.5, 3.0],
            'shear_rate': [100, 100, 200, 200, 300]
        })

        # Create a perfect prediction DataFrame
        self.perfect_df = pd.DataFrame({
            'actual': [10, 20, 30],
            'predicted': [10, 20, 30],
            'residual': [0, 0, 0],
            'abs_error': [0, 0, 0],
            'pct_error': [0, 0, 0],
            'within_ci': [True, True, True],
            'cv': [0.0, 0.0, 0.0],
            'std': [0.0, 0.0, 0.0],
            'shear_rate': [100, 100, 100]
        })

        self.metrics = Metrics()

    def test_mae_calculation(self):
        """Test Mean Absolute Error calculation."""
        result = Metrics._mae(self.sample_df)
        expected = 1.8  # (2+2+2+2+1)/5
        self.assertAlmostEqual(result, expected, places=5)

    def test_mae_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        result = Metrics._mae(self.perfect_df)
        self.assertEqual(result, 0.0)

    def test_rmse_calculation(self):
        """Test Root Mean Squared Error calculation."""
        result = Metrics._rmse(self.sample_df)
        # sqrt((4+4+4+4+1)/5) = sqrt(17/5) = sqrt(3.4) ≈ 1.844
        expected = np.sqrt(3.4)
        self.assertAlmostEqual(result, expected, places=5)

    def test_rmse_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        result = Metrics._rmse(self.perfect_df)
        self.assertEqual(result, 0.0)

    def test_mse_calculation(self):
        """Test Mean Squared Error calculation."""
        result = Metrics._mse(self.sample_df)
        expected = 3.4  # (4+4+4+4+1)/5
        self.assertAlmostEqual(result, expected, places=5)

    def test_mape_calculation(self):
        """Test Mean Absolute Percentage Error calculation."""
        result = Metrics._mape(self.sample_df)
        expected = 8.734  # (20+10+6.67+5+2)/5
        self.assertAlmostEqual(result, expected, places=2)

    def test_median_ae_calculation(self):
        """Test Median Absolute Error calculation."""
        result = Metrics._median_ae(self.sample_df)
        expected = 2.0  # median of [2, 2, 2, 2, 1]
        self.assertEqual(result, expected)

    def test_r2_calculation(self):
        """Test R-squared calculation."""
        result = Metrics._r2(self.sample_df)
        # R² = 1 - (SS_res / SS_tot)
        actual = self.sample_df['actual']
        predicted = self.sample_df['predicted']
        ss_res = ((actual - predicted) ** 2).sum()  # 17
        ss_tot = ((actual - actual.mean()) ** 2).sum()  # 500
        expected = 1 - (ss_res / ss_tot)
        self.assertAlmostEqual(result, expected, places=5)

    def test_r2_perfect_predictions(self):
        """Test R² with perfect predictions."""
        result = Metrics._r2(self.perfect_df)
        self.assertEqual(result, 1.0)

    def test_r2_zero_variance(self):
        """Test R² when actual values have zero variance."""
        zero_var_df = pd.DataFrame({
            'actual': [10, 10, 10],
            'predicted': [12, 8, 10]
        })
        result = Metrics._r2(zero_var_df)
        self.assertEqual(result, 0.0)

    def test_coverage_calculation(self):
        """Test confidence interval coverage calculation."""
        result = Metrics._coverage(self.sample_df)
        expected = 80.0  # 4 out of 5 are True
        self.assertEqual(result, expected)

    def test_coverage_full(self):
        """Test coverage with 100% coverage."""
        result = Metrics._coverage(self.perfect_df)
        self.assertEqual(result, 100.0)

    def test_coverage_zero(self):
        """Test coverage with 0% coverage."""
        zero_coverage_df = pd.DataFrame({
            'within_ci': [False, False, False]
        })
        result = Metrics._coverage(zero_coverage_df)
        self.assertEqual(result, 0.0)

    def test_mean_cv_calculation(self):
        """Test mean coefficient of variation calculation."""
        result = Metrics._mean_cv(self.sample_df)
        expected = 0.2  # (0.1+0.2+0.15+0.25+0.3)/5
        self.assertAlmostEqual(result, expected, places=5)

    def test_median_cv_calculation(self):
        """Test median coefficient of variation calculation."""
        result = Metrics._median_cv(self.sample_df)
        expected = 0.2  # median of [0.1, 0.2, 0.15, 0.25, 0.3]
        self.assertEqual(result, expected)

    def test_max_error_calculation(self):
        """Test maximum absolute error calculation."""
        result = Metrics._max_error(self.sample_df)
        expected = 2.0
        self.assertEqual(result, expected)

    def test_std_error_calculation(self):
        """Test standard deviation of errors calculation."""
        result = Metrics._std_error(self.sample_df)
        expected = self.sample_df['abs_error'].std()
        self.assertAlmostEqual(result, expected, places=5)

    def test_mean_std_calculation(self):
        """Test mean standard deviation calculation."""
        result = Metrics._mean_std(self.sample_df)
        expected = 2.0  # (1.0+1.5+2.0+2.5+3.0)/5
        self.assertEqual(result, expected)

    def test_count_calculation(self):
        """Test count calculation."""
        result = Metrics._count(self.sample_df)
        self.assertEqual(result, 5)

    def test_count_empty_dataframe(self):
        """Test count with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = Metrics._count(empty_df)
        self.assertEqual(result, 0)

    def test_init_default(self):
        """Test initialization without custom metrics."""
        metrics = Metrics()
        self.assertEqual(len(metrics.metrics), len(Metrics.METRIC_FUNCTIONS))
        self.assertIn('mae', metrics.metrics)
        self.assertIn('rmse', metrics.metrics)

    def test_init_with_custom_metrics(self):
        """Test initialization with custom metrics."""
        def custom_metric(df):
            return 42.0

        custom_metrics = {'custom': custom_metric}
        metrics = Metrics(custom_metrics=custom_metrics)

        self.assertIn('custom', metrics.metrics)
        self.assertEqual(metrics.metrics['custom'](self.sample_df), 42.0)
        # Verify default metrics are still present
        self.assertIn('mae', metrics.metrics)

    def test_compute_overall_all_metrics(self):
        """Test computing all metrics overall."""
        result = self.metrics.compute_overall(self.sample_df)

        # Verify all default metrics are computed
        expected_metrics = list(Metrics.METRIC_FUNCTIONS.keys())
        for metric in expected_metrics:
            self.assertIn(metric, result)
            self.assertIsInstance(
                result[metric], (int, float, np.int64, np.float64))

    def test_compute_overall_specific_metrics(self):
        """Test computing specific metrics overall."""
        metrics_to_compute = ['mae', 'rmse', 'r2']
        result = self.metrics.compute_overall(
            self.sample_df,
            metrics=metrics_to_compute
        )

        self.assertEqual(len(result), 3)
        for metric in metrics_to_compute:
            self.assertIn(metric, result)

    def test_compute_overall_unknown_metric(self):
        """Test that unknown metric raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.metrics.compute_overall(
                self.sample_df,
                metrics=['unknown_metric']
            )
        self.assertIn('Unknown metric', str(context.exception))

    def test_compute_per_shear_rate_all_metrics(self):
        """Test computing all metrics per shear rate."""
        result = self.metrics.compute_per_shear_rate(self.sample_df)

        # Should have 3 unique shear rates
        self.assertEqual(len(result), 3)
        self.assertIn('shear_rate', result.columns)

        # Verify shear rates are present
        self.assertListEqual(
            sorted(result['shear_rate'].tolist()),
            [100, 200, 300]
        )

        # Verify all metrics are computed
        for metric in Metrics.METRIC_FUNCTIONS.keys():
            self.assertIn(metric, result.columns)

    def test_compute_per_shear_rate_specific_metrics(self):
        """Test computing specific metrics per shear rate."""
        metrics_to_compute = ['mae', 'count']
        result = self.metrics.compute_per_shear_rate(
            self.sample_df,
            metrics=metrics_to_compute
        )

        # Should have shear_rate column plus specified metrics
        self.assertEqual(len(result.columns), len(metrics_to_compute) + 1)
        self.assertIn('shear_rate', result.columns)
        self.assertIn('mae', result.columns)
        self.assertIn('count', result.columns)

    def test_compute_per_shear_rate_correct_grouping(self):
        """Test that per-shear-rate metrics are computed correctly."""
        result = self.metrics.compute_per_shear_rate(
            self.sample_df,
            metrics=['count', 'mae']
        )

        # Check count for each shear rate
        shear_100 = result[result['shear_rate'] == 100].iloc[0]
        shear_200 = result[result['shear_rate'] == 200].iloc[0]
        shear_300 = result[result['shear_rate'] == 300].iloc[0]

        self.assertEqual(shear_100['count'], 2)
        self.assertEqual(shear_200['count'], 2)
        self.assertEqual(shear_300['count'], 1)

        # Check MAE for shear rate 100 (abs_errors are [2, 2])
        self.assertEqual(shear_100['mae'], 2.0)

    def test_compute_per_shear_rate_unknown_metric(self):
        """Test that unknown metric raises ValueError in per-shear-rate."""
        with self.assertRaises(ValueError) as context:
            self.metrics.compute_per_shear_rate(
                self.sample_df,
                metrics=['nonexistent_metric']
            )
        self.assertIn('Unknown metric', str(context.exception))

    def test_add_metric(self):
        """Test adding a custom metric."""
        def new_metric(df):
            return len(df) * 2

        self.metrics.add_metric('double_count', new_metric)

        self.assertIn('double_count', self.metrics.metrics)
        result = self.metrics.compute_overall(
            self.sample_df,
            metrics=['double_count']
        )
        self.assertEqual(result['double_count'], 10)

    def test_add_metric_overwrites_existing(self):
        """Test that adding a metric with existing name overwrites it."""
        def new_mae(df):
            return 999.0

        original_mae = self.metrics.compute_overall(
            self.sample_df,
            metrics=['mae']
        )['mae']

        self.metrics.add_metric('mae', new_mae)

        new_result = self.metrics.compute_overall(
            self.sample_df,
            metrics=['mae']
        )['mae']

        self.assertNotEqual(original_mae, new_result)
        self.assertEqual(new_result, 999.0)

    def test_get_available_metrics(self):
        """Test getting list of available metrics."""
        available = self.metrics.get_available_metrics()

        self.assertIsInstance(available, list)
        self.assertEqual(len(available), len(Metrics.METRIC_FUNCTIONS))

        # Check some expected metrics
        for metric in ['mae', 'rmse', 'r2', 'count']:
            self.assertIn(metric, available)

    def test_get_available_metrics_with_custom(self):
        """Test getting available metrics includes custom metrics."""
        def custom(df):
            return 1.0

        metrics = Metrics(custom_metrics={'my_custom': custom})
        available = metrics.get_available_metrics()

        self.assertIn('my_custom', available)
        self.assertIn('mae', available)

    def test_metric_with_single_value(self):
        """Test metrics with DataFrame containing single value."""
        single_df = pd.DataFrame({
            'actual': [10],
            'predicted': [12],
            'residual': [2],
            'abs_error': [2],
            'pct_error': [20],
            'within_ci': [True],
            'cv': [0.1],
            'std': [1.0],
            'shear_rate': [100]
        })

        result = self.metrics.compute_overall(
            single_df,
            metrics=['mae', 'count', 'coverage']
        )

        self.assertEqual(result['mae'], 2.0)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['coverage'], 100.0)

    def test_metric_with_nan_values(self):
        """Test metric behavior with NaN values."""
        nan_df = pd.DataFrame({
            'abs_error': [1.0, 2.0, np.nan, 3.0]
        })

        result = Metrics._mae(nan_df)
        # pandas mean() skips NaN by default
        expected = 2.0  # (1+2+3)/3
        self.assertAlmostEqual(result, expected, places=5)

    def test_compute_overall_preserves_metric_order(self):
        """Test that compute_overall returns metrics in requested order."""
        metrics_list = ['count', 'mae', 'rmse']
        result = self.metrics.compute_overall(
            self.sample_df,
            metrics=metrics_list
        )

        # Dictionary order is preserved in Python 3.7+
        self.assertListEqual(list(result.keys()), metrics_list)

    def test_per_shear_rate_sorted_output(self):
        """Test that per-shear-rate results are sorted by shear rate."""
        result = self.metrics.compute_per_shear_rate(
            self.sample_df,
            metrics=['count']
        )

        shear_rates = result['shear_rate'].tolist()
        self.assertEqual(shear_rates, sorted(shear_rates))

    def test_metric_functions_immutability(self):
        """Test that METRIC_FUNCTIONS class attribute is not modified."""
        original_keys = set(Metrics.METRIC_FUNCTIONS.keys())

        metrics = Metrics()
        metrics.add_metric('new_metric', lambda df: 1.0)

        # METRIC_FUNCTIONS should remain unchanged
        self.assertEqual(set(Metrics.METRIC_FUNCTIONS.keys()), original_keys)


if __name__ == '__main__':
    unittest.main()
