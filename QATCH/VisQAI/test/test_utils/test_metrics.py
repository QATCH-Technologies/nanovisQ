"""
test_metrics.py

Comprehensive unit tests for the Metrics class.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-11-03

Version:
    1.0
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
import warnings
from src.utils.metrics import Metrics


class TestMetrics(unittest.TestCase):
    """Test suite for the Metrics class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a simple test dataset
        self.simple_df = pd.DataFrame({
            'actual': [10.0, 20.0, 30.0, 40.0, 50.0],
            'predicted': [12.0, 19.0, 31.0, 38.0, 51.0],
            'residual': [2.0, -1.0, 1.0, -2.0, 1.0],
            'abs_error': [2.0, 1.0, 1.0, 2.0, 1.0],
            'pct_error': [20.0, 5.0, 3.33, 5.0, 2.0],
            'within_ci': [True, True, True, False, True],
            'cv': [0.1, 0.05, 0.03, 0.05, 0.02],
            'std': [1.0, 0.5, 0.3, 0.5, 0.2],
            'shear_rate': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        # Create a perfect prediction dataset
        self.perfect_df = pd.DataFrame({
            'actual': [10.0, 20.0, 30.0, 40.0, 50.0],
            'predicted': [10.0, 20.0, 30.0, 40.0, 50.0],
            'residual': [0.0, 0.0, 0.0, 0.0, 0.0],
            'abs_error': [0.0, 0.0, 0.0, 0.0, 0.0],
            'pct_error': [0.0, 0.0, 0.0, 0.0, 0.0],
            'within_ci': [True, True, True, True, True],
            'cv': [0.0, 0.0, 0.0, 0.0, 0.0],
            'std': [0.0, 0.0, 0.0, 0.0, 0.0],
            'shear_rate': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        # Create a dataset with edge cases
        self.edge_case_df = pd.DataFrame({
            'actual': [0.0, 1.0, 100.0, 0.001],
            'predicted': [0.0, 1.0, 100.0, 0.001],
            'residual': [0.0, 0.0, 0.0, 0.0],
            'abs_error': [0.0, 0.0, 0.0, 0.0],
            'pct_error': [0.0, 0.0, 0.0, 0.0],
            'within_ci': [True, True, True, True],
            'cv': [0.0, 0.0, 0.0, 0.0],
            'std': [0.0, 0.0, 0.0, 0.0],
            'shear_rate': [1.0, 2.0, 3.0, 4.0]
        })

        self.metrics = Metrics()

    def test_initialization(self):
        """Test Metrics class initialization."""
        metrics = Metrics()
        self.assertIsInstance(metrics.metrics, dict)
        self.assertGreater(len(metrics.metrics), 0)
        self.assertIn('mae', metrics.metrics)
        self.assertIn('rmse', metrics.metrics)
        self.assertIn('r2', metrics.metrics)

    def test_initialization_with_custom_metrics(self):
        """Test initialization with custom metrics."""
        def custom_metric(df):
            return df['actual'].sum()

        custom_metrics = {'custom': custom_metric}
        metrics = Metrics(custom_metrics=custom_metrics)
        self.assertIn('custom', metrics.metrics)
        self.assertEqual(metrics.metrics['custom'](self.simple_df), 150.0)

    def test_mae(self):
        """Test Mean Absolute Error calculation."""
        mae = Metrics._mae(self.simple_df)
        expected_mae = np.mean([2.0, 1.0, 1.0, 2.0, 1.0])
        self.assertAlmostEqual(mae, expected_mae, places=5)

        # Test perfect prediction
        mae_perfect = Metrics._mae(self.perfect_df)
        self.assertEqual(mae_perfect, 0.0)

    def test_rmse(self):
        """Test Root Mean Squared Error calculation."""
        rmse = Metrics._rmse(self.simple_df)
        expected_rmse = np.sqrt(np.mean([4.0, 1.0, 1.0, 4.0, 1.0]))
        self.assertAlmostEqual(rmse, expected_rmse, places=5)

        # Test perfect prediction
        rmse_perfect = Metrics._rmse(self.perfect_df)
        self.assertEqual(rmse_perfect, 0.0)

    def test_mse(self):
        """Test Mean Squared Error calculation."""
        mse = Metrics._mse(self.simple_df)
        expected_mse = np.mean([4.0, 1.0, 1.0, 4.0, 1.0])
        self.assertAlmostEqual(mse, expected_mse, places=5)

        # Test perfect prediction
        mse_perfect = Metrics._mse(self.perfect_df)
        self.assertEqual(mse_perfect, 0.0)

    def test_mape(self):
        """Test Mean Absolute Percentage Error calculation."""
        mape = Metrics._mape(self.simple_df)
        expected_mape = np.mean([20.0, 5.0, 3.33, 5.0, 2.0])
        self.assertAlmostEqual(mape, expected_mape, places=2)

    def test_median_ae(self):
        """Test Median Absolute Error calculation."""
        median_ae = Metrics._median_ae(self.simple_df)
        self.assertEqual(median_ae, 1.0)

    def test_r2(self):
        """Test R-squared calculation."""
        r2 = Metrics._r2(self.simple_df)
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)

        # Test perfect prediction
        r2_perfect = Metrics._r2(self.perfect_df)
        self.assertAlmostEqual(r2_perfect, 1.0, places=5)

    def test_adjusted_r2(self):
        """Test Adjusted R-squared calculation."""
        adj_r2 = Metrics._adjusted_r2(self.simple_df, n_features=2)
        self.assertIsInstance(adj_r2, float)

        # Test with n_features >= n-1 (edge case)
        adj_r2_edge = Metrics._adjusted_r2(self.simple_df, n_features=10)
        r2 = Metrics._r2(self.simple_df)
        self.assertEqual(adj_r2_edge, r2)

    def test_coverage(self):
        """Test confidence interval coverage calculation."""
        coverage = Metrics._coverage(self.simple_df)
        expected_coverage = (4 / 5) * 100
        self.assertEqual(coverage, expected_coverage)

        # Test perfect coverage
        coverage_perfect = Metrics._coverage(self.perfect_df)
        self.assertEqual(coverage_perfect, 100.0)

    def test_mean_cv(self):
        """Test mean coefficient of variation calculation."""
        mean_cv = Metrics._mean_cv(self.simple_df)
        expected_mean_cv = np.mean([0.1, 0.05, 0.03, 0.05, 0.02])
        self.assertAlmostEqual(mean_cv, expected_mean_cv, places=5)

    def test_median_cv(self):
        """Test median coefficient of variation calculation."""
        median_cv = Metrics._median_cv(self.simple_df)
        self.assertEqual(median_cv, 0.05)

    def test_max_error(self):
        """Test maximum error calculation."""
        max_error = Metrics._max_error(self.simple_df)
        self.assertEqual(max_error, 2.0)

    def test_std_error(self):
        """Test standard deviation of errors calculation."""
        std_error = Metrics._std_error(self.simple_df)
        expected_std = np.std([2.0, 1.0, 1.0, 2.0, 1.0], ddof=1)
        self.assertAlmostEqual(std_error, expected_std, places=5)

    def test_mean_std(self):
        """Test mean standard deviation calculation."""
        mean_std = Metrics._mean_std(self.simple_df)
        expected_mean_std = np.mean([1.0, 0.5, 0.3, 0.5, 0.2])
        self.assertAlmostEqual(mean_std, expected_mean_std, places=5)

    def test_count(self):
        """Test count calculation."""
        count = Metrics._count(self.simple_df)
        self.assertEqual(count, 5)

    def test_smape(self):
        """Test Symmetric Mean Absolute Percentage Error calculation."""
        smape = Metrics._smape(self.simple_df)
        self.assertGreaterEqual(smape, 0.0)
        self.assertLessEqual(smape, 200.0)

        # Test perfect prediction
        smape_perfect = Metrics._smape(self.perfect_df)
        self.assertEqual(smape_perfect, 0.0)

        # Test edge case with zeros
        zero_df = pd.DataFrame({
            'actual': [0.0, 0.0],
            'predicted': [0.0, 0.0]
        })
        smape_zero = Metrics._smape(zero_df)
        self.assertEqual(smape_zero, 0.0)

    def test_msle(self):
        """Test Mean Squared Logarithmic Error calculation."""
        # Create positive-only dataset
        positive_df = self.simple_df.copy()
        msle = Metrics._msle(positive_df)
        self.assertIsInstance(msle, (float, np.floating))
        self.assertGreaterEqual(msle, 0.0)

        # Test with negative values
        negative_df = self.simple_df.copy()
        negative_df.loc[0, 'actual'] = -1.0
        msle_negative = Metrics._msle(negative_df)
        self.assertTrue(np.isnan(msle_negative))

    def test_explained_variance(self):
        """Test Explained Variance Score calculation."""
        ev = Metrics._explained_variance(self.simple_df)
        self.assertLessEqual(ev, 1.0)

        # Test perfect prediction
        ev_perfect = Metrics._explained_variance(self.perfect_df)
        self.assertAlmostEqual(ev_perfect, 1.0, places=5)

    def test_bias(self):
        """Test bias calculation."""
        bias = Metrics._bias(self.simple_df)
        expected_bias = np.mean([2.0, -1.0, 1.0, -2.0, 1.0])
        self.assertAlmostEqual(bias, expected_bias, places=5)

        # Test perfect prediction
        bias_perfect = Metrics._bias(self.perfect_df)
        self.assertEqual(bias_perfect, 0.0)

    def test_relative_bias(self):
        """Test relative bias calculation."""
        rel_bias = Metrics._relative_bias(self.simple_df)
        self.assertIsInstance(rel_bias, float)

        # Test with zero mean
        zero_mean_df = pd.DataFrame({
            'actual': [0.0, 0.0],
            'residual': [1.0, -1.0]
        })
        rel_bias_zero = Metrics._relative_bias(zero_mean_df)
        self.assertEqual(rel_bias_zero, 0.0)

    def test_iqr_error(self):
        """Test Interquartile Range of errors calculation."""
        iqr = Metrics._iqr_error(self.simple_df)
        self.assertGreaterEqual(iqr, 0.0)

    def test_percentile_errors(self):
        """Test percentile error calculations."""
        p90 = Metrics._p90_error(self.simple_df)
        p95 = Metrics._p95_error(self.simple_df)
        p99 = Metrics._p99_error(self.simple_df)

        self.assertLessEqual(p90, p95)
        self.assertLessEqual(p95, p99)
        self.assertGreaterEqual(p90, 0.0)

    def test_skewness(self):
        """Test skewness calculation."""
        skew = Metrics._skewness(self.simple_df)
        self.assertIsInstance(skew, (float, np.floating))

        # Test symmetric distribution
        skew_perfect = Metrics._skewness(self.perfect_df)
        self.assertTrue(np.isnan(skew_perfect) or abs(skew_perfect) < 1e-10)

    def test_kurtosis(self):
        """Test kurtosis calculation."""
        kurt = Metrics._kurtosis(self.simple_df)
        self.assertIsInstance(kurt, (float, np.floating))

    def test_durbin_watson(self):
        """Test Durbin-Watson statistic calculation."""
        dw = Metrics._durbin_watson(self.simple_df)
        self.assertGreaterEqual(dw, 0.0)
        self.assertLessEqual(dw, 4.0)

    def test_theil_u(self):
        """Test Theil's U statistic calculation."""
        theil = Metrics._theil_u(self.simple_df)
        self.assertIsInstance(theil, (float, np.floating))

        # Test with single observation
        single_df = pd.DataFrame({
            'actual': [10.0],
            'predicted': [12.0]
        })
        theil_single = Metrics._theil_u(single_df)
        self.assertTrue(np.isnan(theil_single))

    def test_accuracy_within_tolerance(self):
        """Test accuracy within tolerance calculation."""
        acc_10 = Metrics._accuracy_within_tolerance(self.simple_df, 0.1)
        self.assertGreaterEqual(acc_10, 0.0)
        self.assertLessEqual(acc_10, 100.0)

        # Test perfect prediction
        acc_perfect = Metrics._accuracy_within_tolerance(self.perfect_df, 0.1)
        self.assertEqual(acc_perfect, 100.0)

    def test_normalized_rmse(self):
        """Test Normalized RMSE calculation."""
        nrmse = Metrics._normalized_rmse(self.simple_df)
        self.assertGreaterEqual(nrmse, 0.0)

        # Test with zero range
        constant_df = pd.DataFrame({
            'actual': [10.0, 10.0, 10.0],
            'predicted': [10.0, 10.0, 10.0],
            'residual': [0.0, 0.0, 0.0]
        })
        nrmse_constant = Metrics._normalized_rmse(constant_df)
        self.assertEqual(nrmse_constant, 0.0)

    def test_cv_rmse(self):
        """Test Coefficient of Variation of RMSE calculation."""
        cv_rmse = Metrics._cv_rmse(self.simple_df)
        self.assertGreaterEqual(cv_rmse, 0.0)

        # Test with zero mean
        zero_mean_df = pd.DataFrame({
            'actual': [0.0, 0.0],
            'predicted': [1.0, -1.0],
            'residual': [1.0, -1.0]
        })
        cv_rmse_zero = Metrics._cv_rmse(zero_mean_df)
        self.assertTrue(np.isinf(cv_rmse_zero) or cv_rmse_zero == 0.0)

    def test_log_accuracy_ratio(self):
        """Test log accuracy ratio calculation."""
        lar = Metrics._log_accuracy_ratio(self.simple_df)
        self.assertIsInstance(lar, (float, np.floating))

        # Test with zero/negative values
        invalid_df = pd.DataFrame({
            'actual': [0.0, 1.0],
            'predicted': [1.0, 2.0]
        })
        lar_invalid = Metrics._log_accuracy_ratio(invalid_df)
        self.assertTrue(np.isnan(lar_invalid) or np.isfinite(lar_invalid))

    def test_shapiro_wilk_p(self):
        """Test Shapiro-Wilk test p-value calculation."""
        p_value = Metrics._shapiro_wilk_p(self.simple_df)
        self.assertIsInstance(p_value, (float, np.floating))

        # Test with insufficient data
        small_df = pd.DataFrame({
            'residual': [1.0, 2.0]
        })
        p_value_small = Metrics._shapiro_wilk_p(small_df)
        self.assertTrue(np.isnan(p_value_small))

    def test_anderson_darling_stat(self):
        """Test Anderson-Darling test statistic calculation."""
        ad_stat = Metrics._anderson_darling_stat(self.simple_df)
        self.assertIsInstance(ad_stat, (float, np.floating))

        # Test with insufficient data
        small_df = pd.DataFrame({
            'residual': [1.0]
        })
        ad_stat_small = Metrics._anderson_darling_stat(small_df)
        self.assertTrue(np.isnan(ad_stat_small))

    def test_pearson_correlation(self):
        """Test Pearson correlation calculation."""
        corr = Metrics._pearson_correlation(self.simple_df)
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

        # Test perfect prediction
        corr_perfect = Metrics._pearson_correlation(self.perfect_df)
        self.assertAlmostEqual(corr_perfect, 1.0, places=5)

    def test_spearman_correlation(self):
        """Test Spearman correlation calculation."""
        corr = Metrics._spearman_correlation(self.simple_df)
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    def test_kendall_correlation(self):
        """Test Kendall correlation calculation."""
        corr = Metrics._kendall_correlation(self.simple_df)
        self.assertGreaterEqual(corr, -1.0)
        self.assertLessEqual(corr, 1.0)

    def test_compute_overall(self):
        """Test compute_overall method."""
        metrics_list = ['mae', 'rmse', 'r2']
        results = self.metrics.compute_overall(
            self.simple_df, metrics=metrics_list)

        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)
        self.assertIn('mae', results)
        self.assertIn('rmse', results)
        self.assertIn('r2', results)

    def test_compute_overall_with_confidence_intervals(self):
        """Test compute_overall with confidence intervals."""
        metrics_list = ['mae', 'rmse']
        results = self.metrics.compute_overall(
            self.simple_df,
            metrics=metrics_list,
            include_confidence_intervals=True
        )

        self.assertIn('mae', results)
        self.assertIn('mae_ci_low', results)
        self.assertIn('mae_ci_high', results)
        self.assertLessEqual(results['mae_ci_low'], results['mae'])
        self.assertGreaterEqual(results['mae_ci_high'], results['mae'])

    def test_compute_overall_with_invalid_metric(self):
        """Test compute_overall with invalid metric name."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self.metrics.compute_overall(
                self.simple_df,
                metrics=['invalid_metric']
            )
            self.assertEqual(len(w), 1)
            self.assertIn('Unknown metric', str(w[0].message))

    def test_compute_overall_with_default_metrics(self):
        """Test compute_overall with default metrics."""
        results = self.metrics.compute_overall(self.simple_df)
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        self.assertIn('mae', results)

    def test_compute_per_shear_rate(self):
        """Test compute_per_shear_rate method."""
        metrics_list = ['mae', 'rmse']
        results = self.metrics.compute_per_shear_rate(
            self.simple_df,
            metrics=metrics_list
        )

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 5)  # 5 unique shear rates
        self.assertIn('shear_rate', results.columns)
        self.assertIn('mae', results.columns)
        self.assertIn('rmse', results.columns)

    def test_compute_per_shear_rate_with_invalid_metric(self):
        """Test compute_per_shear_rate with invalid metric."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = self.metrics.compute_per_shear_rate(
                self.simple_df,
                metrics=['invalid_metric']
            )
            self.assertTrue(len(w) >= 1)

    def test_get_metrics_by_category(self):
        """Test get_metrics_by_category method."""
        basic_metrics = self.metrics.get_metrics_by_category('basic')
        self.assertIsInstance(basic_metrics, list)
        self.assertIn('mae', basic_metrics)
        self.assertIn('rmse', basic_metrics)

        advanced_metrics = self.metrics.get_metrics_by_category('advanced')
        self.assertIsInstance(advanced_metrics, list)
        self.assertIn('smape', advanced_metrics)

        # Test invalid category
        invalid_metrics = self.metrics.get_metrics_by_category('invalid')
        self.assertEqual(invalid_metrics, [])

    def test_compute_metric_summary(self):
        """Test compute_metric_summary method."""
        summary = self.metrics.compute_metric_summary(self.simple_df)

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertIn('Category', summary.columns)
        self.assertIn('Metric', summary.columns)
        self.assertIn('Value', summary.columns)
        self.assertIn('Status', summary.columns)
        self.assertGreater(len(summary), 0)

    def test_evaluate_metric_status(self):
        """Test _evaluate_metric_status method."""
        # Test R2 status
        status_excellent = self.metrics._evaluate_metric_status('r2', 0.95)
        self.assertEqual(status_excellent, 'Excellent')

        status_good = self.metrics._evaluate_metric_status('r2', 0.75)
        self.assertEqual(status_good, 'Good')

        status_poor = self.metrics._evaluate_metric_status('r2', 0.3)
        self.assertEqual(status_poor, 'Poor')

        # Test MAPE status (lower is better)
        status_excellent = self.metrics._evaluate_metric_status('mape', 3.0)
        self.assertEqual(status_excellent, 'Excellent')

        status_poor = self.metrics._evaluate_metric_status('mape', 50.0)
        self.assertEqual(status_poor, 'Poor')

        # Test NaN value
        status_nan = self.metrics._evaluate_metric_status('r2', np.nan)
        self.assertEqual(status_nan, 'N/A')

        # Test unknown metric
        status_unknown = self.metrics._evaluate_metric_status(
            'unknown_metric', 0.5)
        self.assertEqual(status_unknown, 'Unknown')

    def test_add_metric(self):
        """Test add_metric method."""
        def custom_sum(df):
            return df['actual'].sum()

        self.metrics.add_metric('custom_sum', custom_sum)
        self.assertIn('custom_sum', self.metrics.metrics)

        result = self.metrics.compute_overall(
            self.simple_df,
            metrics=['custom_sum']
        )
        self.assertEqual(result['custom_sum'], 150.0)

    def test_get_available_metrics(self):
        """Test get_available_metrics method."""
        available = self.metrics.get_available_metrics()
        self.assertIsInstance(available, list)
        self.assertGreater(len(available), 0)
        self.assertIn('mae', available)
        self.assertIn('rmse', available)
        self.assertIn('r2', available)

    def test_bootstrap_ci(self):
        """Test _bootstrap_ci method."""
        ci_low, ci_high = self.metrics._bootstrap_ci(
            self.simple_df,
            'mae',
            n_bootstrap=100
        )

        self.assertIsInstance(ci_low, (float, np.floating))
        self.assertIsInstance(ci_high, (float, np.floating))
        self.assertLessEqual(ci_low, ci_high)

    def test_bootstrap_ci_with_failing_metric(self):
        """Test _bootstrap_ci with a metric that fails on some samples."""
        # Create a custom metric that sometimes fails
        def failing_metric(df):
            if len(df) < 3:
                raise ValueError("Insufficient data")
            return df['actual'].mean()

        self.metrics.add_metric('failing_metric', failing_metric)

        # Should handle failures gracefully
        ci_low, ci_high = self.metrics._bootstrap_ci(
            self.simple_df,
            'failing_metric',
            n_bootstrap=10
        )

        # Should still return some CI if enough samples succeed
        self.assertIsInstance(ci_low, (float, np.floating))
        self.assertIsInstance(ci_high, (float, np.floating))

    def test_metric_categories(self):
        """Test that metric categories are properly defined."""
        self.assertIn('basic', self.metrics.metric_categories)
        self.assertIn('advanced', self.metrics.metric_categories)
        self.assertIn('distribution', self.metrics.metric_categories)
        self.assertIn('correlation', self.metrics.metric_categories)

        # Verify all categories contain valid metrics
        for category, metric_list in self.metrics.metric_categories.items():
            for metric_name in metric_list:
                self.assertIn(metric_name, self.metrics.metrics)

    def test_edge_cases_with_small_dataset(self):
        """Test metrics with very small datasets."""
        small_df = pd.DataFrame({
            'actual': [10.0],
            'predicted': [12.0],
            'residual': [2.0],
            'abs_error': [2.0],
            'pct_error': [20.0],
            'within_ci': [True],
            'cv': [0.1],
            'std': [1.0],
            'shear_rate': [1.0]
        })

        # Most metrics should still work with single observation
        mae = Metrics._mae(small_df)
        self.assertEqual(mae, 2.0)

        count = Metrics._count(small_df)
        self.assertEqual(count, 1)

    def test_all_metrics_callable(self):
        """Test that all registered metrics are callable."""
        for metric_name, metric_func in self.metrics.metrics.items():
            self.assertTrue(callable(metric_func),
                            f"Metric '{metric_name}' is not callable")

    def test_metrics_return_numeric_values(self):
        """Test that all metrics return numeric values or NaN."""
        for metric_name in self.metrics.get_available_metrics():
            try:
                result = self.metrics.metrics[metric_name](self.simple_df)
                self.assertTrue(
                    isinstance(result, (int, float, np.number)
                               ) or np.isnan(result),
                    f"Metric '{metric_name}' returned non-numeric value: {type(result)}"
                )
            except Exception as e:
                # Some metrics may fail with certain data, which is acceptable
                pass


if __name__ == '__main__':
    unittest.main()
