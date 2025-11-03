"""
metrics.py

Extended version of the Metrics class with additional statistical and performance metrics
for comprehensive model evaluation.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-22

Version:
    1.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_log_error, explained_variance_score
import warnings


class Metrics:

    @staticmethod
    def _mae(df: pd.DataFrame) -> float:
        """Calculate Mean Absolute Error."""
        return df['abs_error'].mean()

    @staticmethod
    def _rmse(df: pd.DataFrame) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt((df['residual'] ** 2).mean())

    @staticmethod
    def _mse(df: pd.DataFrame) -> float:
        """Calculate Mean Squared Error."""
        return (df['residual'] ** 2).mean()

    @staticmethod
    def _mape(df: pd.DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return df['percentage_error'].mean()

    @staticmethod
    def _median_ae(df: pd.DataFrame) -> float:
        """Calculate Median Absolute Error."""
        return df['abs_error'].median()

    @staticmethod
    def _r2(df: pd.DataFrame) -> float:
        """Calculate R-squared (coefficient of determination)."""
        actual = df['actual']
        predicted = df['predicted']
        ss_res = ((actual - predicted) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    @staticmethod
    def _adjusted_r2(df: pd.DataFrame, n_features: int = 14) -> float:
        """Calculate Adjusted R-squared.

        Args:
            df: DataFrame with actual and predicted columns.
            n_features: Number of features used in the model.

        Returns:
            Adjusted R-squared value.
        """
        n = len(df)
        r2 = Metrics._r2(df)

        if n - n_features - 1 <= 0:
            return r2

        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
        return adj_r2

    @staticmethod
    def _coverage(df: pd.DataFrame) -> float:
        """Calculate confidence interval coverage percentage."""
        return (df['within_ci'].sum() / len(df)) * 100

    @staticmethod
    def _mean_cv(df: pd.DataFrame) -> float:
        """Calculate mean coefficient of variation."""
        return df['cv'].mean()

    @staticmethod
    def _median_cv(df: pd.DataFrame) -> float:
        """Calculate median coefficient of variation."""
        return df['cv'].median()

    @staticmethod
    def _max_error(df: pd.DataFrame) -> float:
        """Calculate maximum absolute error."""
        return df['abs_error'].max()

    @staticmethod
    def _std_error(df: pd.DataFrame) -> float:
        """Calculate standard deviation of absolute errors."""
        return df['abs_error'].std()

    @staticmethod
    def _mean_std(df: pd.DataFrame) -> float:
        """Calculate mean standard deviation."""
        return df['std'].mean()

    @staticmethod
    def _count(df: pd.DataFrame) -> float:
        """Calculate number of observations."""
        return len(df)

    # Additional advanced metrics

    @staticmethod
    def _smape(df: pd.DataFrame) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error.

        SMAPE = 100 * mean(2 * |actual - predicted| / (|actual| + |predicted|))
        """
        actual = df['actual']
        predicted = df['predicted']
        denominator = np.abs(actual) + np.abs(predicted)
        # Avoid division by zero
        mask = denominator != 0
        if not mask.any():
            return 0.0
        smape = 100 * \
            np.mean(2 * np.abs(actual[mask] -
                    predicted[mask]) / denominator[mask])
        return smape

    @staticmethod
    def _msle(df: pd.DataFrame) -> float:
        """Calculate Mean Squared Logarithmic Error.

        Useful when targets have exponential growth.
        """
        actual = df['actual']
        predicted = df['predicted']

        # Check for negative values
        if (actual < 0).any() or (predicted < 0).any():
            return np.nan

        try:
            return mean_squared_log_error(actual, predicted)
        except:
            return np.nan

    @staticmethod
    def _explained_variance(df: pd.DataFrame) -> float:
        """Calculate Explained Variance Score.

        Best possible score is 1.0, lower values are worse.
        """
        actual = df['actual']
        predicted = df['predicted']
        return explained_variance_score(actual, predicted)

    @staticmethod
    def _bias(df: pd.DataFrame) -> float:
        """Calculate mean bias (systematic error)."""
        return df['residual'].mean()

    @staticmethod
    def _relative_bias(df: pd.DataFrame) -> float:
        """Calculate relative bias as percentage."""
        actual_mean = df['actual'].mean()
        if actual_mean == 0:
            return 0.0
        return (df['residual'].mean() / actual_mean) * 100

    @staticmethod
    def _iqr_error(df: pd.DataFrame) -> float:
        """Calculate Interquartile Range of errors."""
        q75 = df['abs_error'].quantile(0.75)
        q25 = df['abs_error'].quantile(0.25)
        return q75 - q25

    @staticmethod
    def _p90_error(df: pd.DataFrame) -> float:
        """Calculate 90th percentile of absolute errors."""
        return df['abs_error'].quantile(0.90)

    @staticmethod
    def _p95_error(df: pd.DataFrame) -> float:
        """Calculate 95th percentile of absolute errors."""
        return df['abs_error'].quantile(0.95)

    @staticmethod
    def _p99_error(df: pd.DataFrame) -> float:
        """Calculate 99th percentile of absolute errors."""
        return df['abs_error'].quantile(0.99)

    @staticmethod
    def _skewness(df: pd.DataFrame) -> float:
        """Calculate skewness of residuals."""
        return stats.skew(df['residual'])

    @staticmethod
    def _kurtosis(df: pd.DataFrame) -> float:
        """Calculate kurtosis of residuals."""
        return stats.kurtosis(df['residual'])

    @staticmethod
    def _durbin_watson(df: pd.DataFrame) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation in residuals.

        Values around 2 suggest no autocorrelation.
        Values approaching 0 indicate positive autocorrelation.
        Values approaching 4 indicate negative autocorrelation.
        """
        residuals = df['residual'].values
        diff_resid = np.diff(residuals)
        dw = np.sum(diff_resid**2) / np.sum(residuals**2)
        return dw

    @staticmethod
    def _theil_u(df: pd.DataFrame) -> float:
        """Calculate Theil's U statistic (uncertainty coefficient).

        U < 1: Better than naive forecast
        U = 1: As good as naive forecast
        U > 1: Worse than naive forecast
        """
        actual = df['actual'].values
        predicted = df['predicted'].values

        if len(actual) < 2:
            return np.nan

        # Calculate numerator (RMSE of prediction)
        rmse_pred = np.sqrt(np.mean((actual - predicted)**2))

        # Calculate denominator (RMSE of naive forecast)
        naive_forecast = actual[:-1]  # Use previous value as forecast
        actual_shifted = actual[1:]
        rmse_naive = np.sqrt(np.mean((actual_shifted - naive_forecast)**2))

        if rmse_naive == 0:
            return np.inf if rmse_pred > 0 else 0.0

        return rmse_pred / rmse_naive

    @staticmethod
    def _accuracy_within_tolerance(df: pd.DataFrame, tolerance: float = 0.1) -> float:
        """Calculate percentage of predictions within a given tolerance.

        Args:
            df: DataFrame with actual and predicted columns.
            tolerance: Fractional tolerance (e.g., 0.1 for 10%).

        Returns:
            Percentage of predictions within tolerance.
        """
        actual = df['actual']
        predicted = df['predicted']

        lower_bound = actual * (1 - tolerance)
        upper_bound = actual * (1 + tolerance)

        within_tolerance = ((predicted >= lower_bound) &
                            (predicted <= upper_bound)).sum()
        return (within_tolerance / len(df)) * 100

    @staticmethod
    def _normalized_rmse(df: pd.DataFrame) -> float:
        """Calculate Normalized RMSE (NRMSE) using range normalization."""
        rmse = Metrics._rmse(df)
        actual_range = df['actual'].max() - df['actual'].min()

        if actual_range == 0:
            return 0.0 if rmse == 0 else np.inf

        return rmse / actual_range

    @staticmethod
    def _cv_rmse(df: pd.DataFrame) -> float:
        """Calculate Coefficient of Variation of RMSE."""
        rmse = Metrics._rmse(df)
        mean_actual = df['actual'].mean()

        if mean_actual == 0:
            return 0.0 if rmse == 0 else np.inf

        return (rmse / mean_actual) * 100

    @staticmethod
    def _log_accuracy_ratio(df: pd.DataFrame) -> float:
        """Calculate mean of log(predicted/actual).

        Values close to 0 indicate good predictions.
        Positive values indicate overprediction.
        Negative values indicate underprediction.
        """
        actual = df['actual']
        predicted = df['predicted']

        # Avoid log of zero or negative
        mask = (actual > 0) & (predicted > 0)
        if not mask.any():
            return np.nan

        log_ratio = np.log(predicted[mask] / actual[mask])
        return log_ratio.mean()

    @staticmethod
    def _shapiro_wilk_p(df: pd.DataFrame) -> float:
        """Calculate Shapiro-Wilk test p-value for normality of residuals.

        p-value > 0.05 suggests residuals are normally distributed.
        """
        if len(df) < 3:
            return np.nan

        try:
            _, p_value = stats.shapiro(df['residual'])
            return p_value
        except:
            return np.nan

    @staticmethod
    def _anderson_darling_stat(df: pd.DataFrame) -> float:
        """Calculate Anderson-Darling test statistic for normality.

        Lower values indicate better fit to normal distribution.
        """
        if len(df) < 2:
            return np.nan

        try:
            result = stats.anderson(df['residual'])
            return result.statistic
        except:
            return np.nan

    # Correlation metrics

    @staticmethod
    def _pearson_correlation(df: pd.DataFrame) -> float:
        """Calculate Pearson correlation coefficient between actual and predicted."""
        return df['actual'].corr(df['predicted'], method='pearson')

    @staticmethod
    def _spearman_correlation(df: pd.DataFrame) -> float:
        """Calculate Spearman rank correlation between actual and predicted."""
        return df['actual'].corr(df['predicted'], method='spearman')

    @staticmethod
    def _kendall_correlation(df: pd.DataFrame) -> float:
        """Calculate Kendall Tau correlation between actual and predicted."""
        return df['actual'].corr(df['predicted'], method='kendall')

    # Metric collection

    BASIC_METRICS = {
        'mae': _mae.__func__,
        'rmse': _rmse.__func__,
        'mse': _mse.__func__,
        'mape': _mape.__func__,
        'median_ae': _median_ae.__func__,
        'r2': _r2.__func__,
        'coverage': _coverage.__func__,
        'mean_cv': _mean_cv.__func__,
        'median_cv': _median_cv.__func__,
        'max_error': _max_error.__func__,
        'std_error': _std_error.__func__,
        'mean_std': _mean_std.__func__,
        'count': _count.__func__,
    }

    ADVANCED_METRICS = {
        'adjusted_r2': _adjusted_r2.__func__,
        'smape': _smape.__func__,
        'msle': _msle.__func__,
        'explained_variance': _explained_variance.__func__,
        'bias': _bias.__func__,
        'relative_bias': _relative_bias.__func__,
        'iqr_error': _iqr_error.__func__,
        'p90_error': _p90_error.__func__,
        'p95_error': _p95_error.__func__,
        'p99_error': _p99_error.__func__,
        'normalized_rmse': _normalized_rmse.__func__,
        'cv_rmse': _cv_rmse.__func__,
        'accuracy_within_10pct': lambda df: Metrics._accuracy_within_tolerance(df, 0.1),
        'accuracy_within_20pct': lambda df: Metrics._accuracy_within_tolerance(df, 0.2),
        'theil_u': _theil_u.__func__,
        'log_accuracy_ratio': _log_accuracy_ratio.__func__,
    }

    DISTRIBUTION_METRICS = {
        'skewness': _skewness.__func__,
        'kurtosis': _kurtosis.__func__,
        'durbin_watson': _durbin_watson.__func__,
        'shapiro_wilk_p': _shapiro_wilk_p.__func__,
        'anderson_darling_stat': _anderson_darling_stat.__func__,
    }

    CORRELATION_METRICS = {
        'pearson_correlation': _pearson_correlation.__func__,
        'spearman_correlation': _spearman_correlation.__func__,
        'kendall_correlation': _kendall_correlation.__func__,
    }

    def __init__(self, custom_metrics: Optional[Dict[str, Callable]] = None):
        """Initialize the Metrics class.

        Args:
            custom_metrics: Dictionary of custom metric functions.
        """
        # Combine all metric categories
        self.metrics = {}
        self.metrics.update(self.BASIC_METRICS)
        self.metrics.update(self.ADVANCED_METRICS)
        self.metrics.update(self.DISTRIBUTION_METRICS)
        self.metrics.update(self.CORRELATION_METRICS)

        if custom_metrics:
            self.metrics.update(custom_metrics)

        # Define metric categories for organization
        self.metric_categories = {
            'basic': list(self.BASIC_METRICS.keys()),
            'advanced': list(self.ADVANCED_METRICS.keys()),
            'distribution': list(self.DISTRIBUTION_METRICS.keys()),
            'correlation': list(self.CORRELATION_METRICS.keys()),
        }

    def compute_overall(
        self,
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        include_confidence_intervals: bool = False
    ) -> Dict[str, float]:
        """Compute overall metrics across all data.

        Args:
            results_df: DataFrame containing prediction results.
            metrics: List of metric names to compute.
            include_confidence_intervals: Whether to include bootstrap CIs.

        Returns:
            Dictionary mapping metric names to their computed values.
        """
        if metrics is None:
            metrics = list(self.BASIC_METRICS.keys())

        overall_metrics = {}

        for metric_name in metrics:
            if metric_name not in self.metrics:
                warnings.warn(f"Unknown metric: {metric_name}")
                continue

            try:
                value = self.metrics[metric_name](results_df)
                overall_metrics[metric_name] = value

                if include_confidence_intervals and not np.isnan(value):
                    # Bootstrap confidence intervals
                    ci_low, ci_high = self._bootstrap_ci(
                        results_df, metric_name, n_bootstrap=1000
                    )
                    overall_metrics[f"{metric_name}_ci_low"] = ci_low
                    overall_metrics[f"{metric_name}_ci_high"] = ci_high

            except Exception as e:
                warnings.warn(f"Failed to compute {metric_name}: {str(e)}")
                overall_metrics[metric_name] = np.nan

        return overall_metrics

    def compute_per_shear_rate(
        self,
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compute metrics separately for each shear rate.

        Args:
            results_df: DataFrame containing prediction results.
            metrics: List of metric names to compute.

        Returns:
            DataFrame with one row per shear rate.
        """
        if metrics is None:
            metrics = list(self.BASIC_METRICS.keys())

        shear_rates = sorted(results_df['shear_rate'].unique())
        per_shear_metrics = []

        for shear_rate in shear_rates:
            shear_df = results_df[results_df['shear_rate'] == shear_rate]
            shear_metrics = {'shear_rate': shear_rate}

            for metric_name in metrics:
                if metric_name not in self.metrics:
                    warnings.warn(f"Unknown metric: {metric_name}")
                    continue

                try:
                    shear_metrics[metric_name] = self.metrics[metric_name](
                        shear_df)
                except Exception as e:
                    warnings.warn(
                        f"Failed to compute {metric_name} for shear rate {shear_rate}: {str(e)}")
                    shear_metrics[metric_name] = np.nan

            per_shear_metrics.append(shear_metrics)

        return pd.DataFrame(per_shear_metrics)

    def _bootstrap_ci(
        self,
        df: pd.DataFrame,
        metric_name: str,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals for a metric.

        Args:
            df: DataFrame with data.
            metric_name: Name of the metric.
            n_bootstrap: Number of bootstrap samples.
            confidence_level: Confidence level (default 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        metric_func = self.metrics[metric_name]
        bootstrap_values = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_df = df.sample(n=len(df), replace=True)
            try:
                value = metric_func(sample_df)
                if not np.isnan(value):
                    bootstrap_values.append(value)
            except:
                continue

        if not bootstrap_values:
            return np.nan, np.nan

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_values, lower_percentile)
        upper_bound = np.percentile(bootstrap_values, upper_percentile)

        return lower_bound, upper_bound

    def get_metrics_by_category(self, category: str) -> List[str]:
        """Get list of metrics in a specific category.

        Args:
            category: Category name ('basic', 'advanced', 'distribution', 'correlation').

        Returns:
            List of metric names in the category.
        """
        return self.metric_categories.get(category, [])

    def compute_metric_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute a comprehensive summary of all metric categories.

        Args:
            results_df: DataFrame containing prediction results.

        Returns:
            DataFrame with metric summaries organized by category.
        """
        summary_data = []

        for category, metric_names in self.metric_categories.items():
            for metric_name in metric_names:
                try:
                    value = self.metrics[metric_name](results_df)
                    summary_data.append({
                        'Category': category.title(),
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': value,
                        'Status': self._evaluate_metric_status(metric_name, value)
                    })
                except:
                    summary_data.append({
                        'Category': category.title(),
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': np.nan,
                        'Status': 'N/A'
                    })

        return pd.DataFrame(summary_data)

    def _evaluate_metric_status(self, metric_name: str, value: float) -> str:
        """Evaluate if a metric value is good, acceptable, or poor.

        Args:
            metric_name: Name of the metric.
            value: Metric value.

        Returns:
            Status string ('Excellent', 'Good', 'Fair', 'Poor').
        """
        if np.isnan(value):
            return 'N/A'

        # Define thresholds for different metrics
        thresholds = {
            'r2': [(0.9, 'Excellent'), (0.7, 'Good'), (0.5, 'Fair'), (0, 'Poor')],
            'mape': [(5, 'Excellent'), (10, 'Good'), (20, 'Fair'), (100, 'Poor')],
            'coverage': [(95, 'Excellent'), (90, 'Good'), (80, 'Fair'), (0, 'Poor')],
            # Simplified
            'durbin_watson': [(1.5, 'Good'), (1, 'Fair'), (0, 'Poor')],
            'shapiro_wilk_p': [(0.1, 'Good'), (0.05, 'Fair'), (0, 'Poor')],
        }

        if metric_name in thresholds:
            for threshold, status in thresholds[metric_name]:
                if metric_name in ['mape']:  # Lower is better
                    if value <= threshold:
                        return status
                elif metric_name in ['durbin_watson']:  # Around 2 is best
                    if abs(value - 2) <= threshold:
                        return status
                else:  # Higher is better
                    if value >= threshold:
                        return status

        return 'Unknown'

    def add_metric(self, name: str, func: Callable[[pd.DataFrame], float]):
        """Add a custom metric to the available metrics.

        Args:
            name: Name of the metric.
            func: Function that computes the metric.
        """
        self.metrics[name] = func

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metric names.

        Returns:
            List of metric names.
        """
        return list(self.metrics.keys())
