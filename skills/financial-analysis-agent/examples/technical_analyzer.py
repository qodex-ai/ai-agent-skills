"""
Technical Analysis Module

Implements technical analysis techniques for stock price analysis including
moving averages, RSI, and support/resistance level identification.
"""

import pandas as pd
from typing import Dict, List


class TechnicalAnalyzer:
    """Performs technical analysis on price data."""

    def calculate_moving_averages(self, prices: pd.Series, windows: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate moving averages for given windows.

        Args:
            prices: Series of price data
            windows: List of window sizes (default: [20, 50, 200])

        Returns:
            Dictionary with moving average series
        """
        if windows is None:
            windows = [20, 50, 200]

        mas = {}
        for window in windows:
            mas[f"ma_{window}"] = prices.rolling(window=window).mean()
        return mas

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI).

        Args:
            prices: Series of price data
            period: RSI calculation period (default: 14)

        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gains = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        losses = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def identify_support_resistance(self, prices: pd.Series) -> tuple:
        """
        Identify support and resistance levels.

        Args:
            prices: Series of price data

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        resistance_levels = prices.rolling(window=5, center=True).max()
        support_levels = prices.rolling(window=5, center=True).min()
        return support_levels, resistance_levels
