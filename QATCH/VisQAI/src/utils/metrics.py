"""
metrics.py

This module provides the `Metrics` class for generating evaluation metrics
for model fit on an oaverall or per shear-rate basis.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-21

Version:
    1.1
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional


class Metrics:
    """A class for computing various metrics on prediction results.

    This class provides static methods for calculating different error metrics
    and can compute these metrics both overall and per shear rate. It supports
    custom metrics through initialization.

    Attributes:
        metrics (Dict[str, Callable]): Dictionary mapping metric names to their
            computation functions.
        METRIC_FUNCTIONS (Dict[str, Callable]): Class-level dictionary of default
            metric computation functions.
    """

    @staticmethod
    def _mae(df: pd.DataFrame) -> float:
        """Calculate Mean Absolute Error.

        Args:
            df (pd.DataFrame): DataFrame containing an 'abs_error' column.

        Returns:
            float: Mean absolute error value.
        """
        return df['abs_error'].mean()

    @staticmethod
    def _rmse(df: pd.DataFrame) -> float:
        """Calculate Root Mean Squared Error.

        Args:
            df (pd.DataFrame): DataFrame containing a 'residual' column.

        Returns:
            float: Root mean squared error value.
        """
        return np.sqrt((df['residual'] ** 2).mean())

    @staticmethod
    def _mse(df: pd.DataFrame) -> float:
        """Calculate Mean Squared Error.

        Args:
            df (pd.DataFrame): DataFrame containing a 'residual' column.

        Returns:
            float: Mean squared error value.
        """
        return (df['residual'] ** 2).mean()

    @staticmethod
    def _mape(df: pd.DataFrame) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            df (pd.DataFrame): DataFrame containing a 'pct_error' column.

        Returns:
            float: Mean absolute percentage error value.
        """
        return df['pct_error'].mean()

    @staticmethod
    def _median_ae(df: pd.DataFrame) -> float:
        """Calculate Median Absolute Error.

        Args:
            df (pd.DataFrame): DataFrame containing an 'abs_error' column.

        Returns:
            float: Median absolute error value.
        """
        return df['abs_error'].median()

    @staticmethod
    def _r2(df: pd.DataFrame) -> float:
        """Calculate R-squared (coefficient of determination).

        Args:
            df (pd.DataFrame): DataFrame containing 'actual' and 'predicted' columns.

        Returns:
            float: R-squared value. Returns 0.0 if total sum of squares is zero.
        """
        actual = df['actual']
        predicted = df['predicted']
        ss_res = ((actual - predicted) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    @staticmethod
    def _coverage(df: pd.DataFrame) -> float:
        """Calculate confidence interval coverage percentage.

        Args:
            df (pd.DataFrame): DataFrame containing a 'within_ci' column with
                boolean values indicating if predictions fall within confidence
                intervals.

        Returns:
            float: Percentage of predictions within confidence intervals.
        """
        return (df['within_ci'].sum() / len(df)) * 100

    @staticmethod
    def _mean_cv(df: pd.DataFrame) -> float:
        """Calculate mean coefficient of variation.

        Args:
            df (pd.DataFrame): DataFrame containing a 'cv' column.

        Returns:
            float: Mean coefficient of variation value.
        """
        return df['cv'].mean()

    @staticmethod
    def _median_cv(df: pd.DataFrame) -> float:
        """Calculate median coefficient of variation.

        Args:
            df (pd.DataFrame): DataFrame containing a 'cv' column.

        Returns:
            float: Median coefficient of variation value.
        """
        return df['cv'].median()

    @staticmethod
    def _max_error(df: pd.DataFrame) -> float:
        """Calculate maximum absolute error.

        Args:
            df (pd.DataFrame): DataFrame containing an 'abs_error' column.

        Returns:
            float: Maximum absolute error value.
        """
        return df['abs_error'].max()

    @staticmethod
    def _std_error(df: pd.DataFrame) -> float:
        """Calculate standard deviation of absolute errors.

        Args:
            df (pd.DataFrame): DataFrame containing an 'abs_error' column.

        Returns:
            float: Standard deviation of absolute errors.
        """
        return df['abs_error'].std()

    @staticmethod
    def _mean_std(df: pd.DataFrame) -> float:
        """Calculate mean standard deviation.

        Args:
            df (pd.DataFrame): DataFrame containing a 'std' column.

        Returns:
            float: Mean standard deviation value.
        """
        return df['std'].mean()

    @staticmethod
    def _count(df: pd.DataFrame) -> float:
        """Calculate number of observations.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            float: Number of rows in the DataFrame.
        """
        return len(df)

    METRIC_FUNCTIONS = {
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

    def __init__(self, custom_metrics: Optional[Dict[str, Callable]] = None):
        """Initialize the Metrics class.

        Args:
            custom_metrics (Optional[Dict[str, Callable]]): Dictionary of custom
                metric functions to add to the default metrics. Keys are metric
                names and values are callables that take a DataFrame and return
                a float. Defaults to None.
        """
        self.metrics = self.METRIC_FUNCTIONS.copy()
        if custom_metrics:
            self.metrics.update(custom_metrics)

    def compute_overall(
        self,
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute overall metrics across all data.

        Args:
            results_df (pd.DataFrame): DataFrame containing prediction results
                with appropriate columns for the requested metrics.
            metrics (Optional[List[str]]): List of metric names to compute.
                If None, computes all available metrics. Defaults to None.

        Returns:
            Dict[str, float]: Dictionary mapping metric names to their computed
                values.

        Raises:
            ValueError: If an unknown metric name is provided.
        """
        if metrics is None:
            metrics = list(self.metrics.keys())

        overall_metrics = {}
        for metric_name in metrics:
            if metric_name not in self.metrics:
                raise ValueError(f"Unknown metric: {metric_name}")
            overall_metrics[metric_name] = self.metrics[metric_name](
                results_df)

        return overall_metrics

    def compute_per_shear_rate(
        self,
        results_df: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compute metrics separately for each shear rate.

        Args:
            results_df (pd.DataFrame): DataFrame containing prediction results
                with a 'shear_rate' column and appropriate columns for the
                requested metrics.
            metrics (Optional[List[str]]): List of metric names to compute.
                If None, computes all available metrics. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with one row per shear rate, containing
                the shear rate and all computed metric values.

        Raises:
            ValueError: If an unknown metric name is provided.
        """
        if metrics is None:
            metrics = list(self.metrics.keys())

        shear_rates = sorted(results_df['shear_rate'].unique())
        per_shear_metrics = []

        for shear_rate in shear_rates:
            shear_df = results_df[results_df['shear_rate'] == shear_rate]
            shear_metrics = {'shear_rate': shear_rate}

            for metric_name in metrics:
                if metric_name not in self.metrics:
                    raise ValueError(f"Unknown metric: {metric_name}")
                shear_metrics[metric_name] = self.metrics[metric_name](
                    shear_df)

            per_shear_metrics.append(shear_metrics)

        return pd.DataFrame(per_shear_metrics)

    def add_metric(self, name: str, func: Callable[[pd.DataFrame], float]):
        """Add a custom metric to the available metrics.

        Args:
            name (str): Name of the metric to add.
            func (Callable[[pd.DataFrame], float]): Function that takes a
                DataFrame and returns a float metric value.
        """
        self.metrics[name] = func

    def get_available_metrics(self) -> List[str]:
        """Get list of all available metric names.

        Returns:
            List[str]: List of metric names that can be computed.
        """
        return list(self.metrics.keys())
