"""
Portfolio Management Module

Handles portfolio management operations including calculation of portfolio value,
rebalancing, and risk assessment.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict


class PortfolioManager:
    """Manages investment portfolio operations."""

    def __init__(self, portfolio: Dict[str, float]):
        """
        Initialize portfolio manager.

        Args:
            portfolio: Dictionary with ticker symbols and share counts
        """
        self.portfolio = portfolio

    def calculate_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.

        Returns:
            Total portfolio value in currency units
        """
        total_value = 0
        for ticker, shares in self.portfolio.items():
            price = yf.Ticker(ticker).info.get("currentPrice", 0)
            total_value += price * shares
        return total_value

    def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate rebalancing trades needed.

        Args:
            target_allocation: Target allocation percentages

        Returns:
            Dictionary with ticker symbols and shares to buy/sell
        """
        current_value = self.calculate_portfolio_value()
        rebalancing_trades = {}

        for ticker, target_pct in target_allocation.items():
            target_value = current_value * target_pct
            price = yf.Ticker(ticker).info.get("currentPrice", 0)
            current_value_held = self.portfolio.get(ticker, 0) * price
            shares_needed = (target_value - current_value_held) / price

            if shares_needed != 0:
                rebalancing_trades[ticker] = shares_needed

        return rebalancing_trades

    def calculate_portfolio_risk(self) -> float:
        """
        Calculate portfolio volatility/risk.

        Returns:
            Portfolio volatility
        """
        returns = pd.DataFrame()
        for ticker in self.portfolio.keys():
            data = yf.download(ticker, period="1y")
            returns[ticker] = data["Close"].pct_change()

        covariance = returns.cov()
        weights = np.array(list(self.portfolio.values()))
        portfolio_variance = np.dot(weights, np.dot(covariance, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)

        return portfolio_volatility
