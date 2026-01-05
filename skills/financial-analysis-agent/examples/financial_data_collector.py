"""
Financial Data Collector Module

Handles integration with financial data APIs and collection of stock data,
financial statements, and key metrics.
"""

import yfinance as yf
import pandas as pd
from alpha_vantage.fundamentaldata import FundamentalData
from typing import Dict, List


class FinancialDataCollector:
    """Collects financial data from various APIs."""

    def __init__(self, api_key: str = None):
        """
        Initialize the financial data collector.

        Args:
            api_key: Alpha Vantage API key for fundamental data
        """
        self.stock_data = yf.Ticker
        if api_key:
            self.fundamental_data = FundamentalData(api_key=api_key)
        else:
            self.fundamental_data = None

    def get_stock_data(self, ticker: str, period="1y") -> pd.DataFrame:
        """
        Retrieve stock price data for a given ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period for data (default: 1 year)

        Returns:
            DataFrame with stock price data
        """
        data = yf.download(ticker, period=period)
        return data

    def get_financial_statements(self, ticker: str) -> Dict:
        """
        Retrieve financial statements for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with income statement, balance sheet, and cash flow
        """
        company = yf.Ticker(ticker)
        return {
            "income_statement": company.financials,
            "balance_sheet": company.balance_sheet,
            "cash_flow": company.cashflow
        }

    def get_key_metrics(self, ticker: str) -> Dict:
        """
        Retrieve key financial metrics for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with key metrics like market cap, PE ratio, etc.
        """
        company = yf.Ticker(ticker)
        return {
            "market_cap": company.info.get("marketCap"),
            "pe_ratio": company.info.get("trailingPE"),
            "pb_ratio": company.info.get("priceToBook"),
            "dividend_yield": company.info.get("dividendYield"),
            "52_week_high": company.info.get("fiftyTwoWeekHigh"),
            "52_week_low": company.info.get("fiftyTwoWeekLow")
        }
