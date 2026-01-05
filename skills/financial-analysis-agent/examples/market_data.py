"""Market data collection and financial data integration."""

import yfinance as yf
import pandas as pd
from typing import Dict, Any


class FinancialDataCollector:
    """Collects financial data from multiple sources."""

    def __init__(self, alpha_vantage_key: str = None):
        """Initialize the financial data collector.

        Args:
            alpha_vantage_key: API key for Alpha Vantage (optional)
        """
        self.stock_data = yf.Ticker
        self.alpha_vantage_key = alpha_vantage_key
        if alpha_vantage_key:
            from alpha_vantage.fundamentaldata import FundamentalData
            self.fundamental_data = FundamentalData(api_key=alpha_vantage_key)

    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Retrieve historical stock data.

        Args:
            ticker: Stock ticker symbol
            period: Time period for data (default: 1y)

        Returns:
            DataFrame with historical OHLCV data
        """
        data = yf.download(ticker, period=period)
        return data

    def get_financial_statements(self, ticker: str) -> Dict[str, Any]:
        """Retrieve financial statements for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing income statement, balance sheet, and cash flow
        """
        company = yf.Ticker(ticker)
        return {
            "income_statement": company.financials,
            "balance_sheet": company.balance_sheet,
            "cash_flow": company.cashflow
        }

    def get_key_metrics(self, ticker: str) -> Dict[str, Any]:
        """Retrieve key financial metrics for a company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with key metrics like PE ratio, market cap, etc.
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
