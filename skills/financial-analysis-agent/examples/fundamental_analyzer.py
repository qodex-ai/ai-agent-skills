"""
Fundamental Analysis Module

Implements fundamental analysis techniques including profitability ratios,
valuation ratios, and liquidity ratios.
"""

from typing import Dict


class FundamentalAnalyzer:
    """Performs fundamental analysis on financial data."""

    def calculate_profitability_ratios(self, financials: Dict) -> Dict[str, float]:
        """
        Calculate profitability ratios.

        Args:
            financials: Dictionary with financial data

        Returns:
            Dictionary with profitability ratios
        """
        return {
            "gross_margin": (
                financials["revenue"] - financials["cost_of_goods"]
            ) / financials["revenue"],
            "operating_margin": (
                financials["operating_income"] / financials["revenue"]
            ),
            "net_margin": (
                financials["net_income"] / financials["revenue"]
            ),
            "roa": financials["net_income"] / financials["total_assets"],
            "roe": financials["net_income"] / financials["equity"]
        }

    def calculate_valuation_ratios(self, financials: Dict, market_cap: float) -> Dict[str, float]:
        """
        Calculate valuation ratios.

        Args:
            financials: Dictionary with financial data
            market_cap: Current market capitalization

        Returns:
            Dictionary with valuation ratios
        """
        return {
            "pe_ratio": market_cap / financials["net_income"],
            "pb_ratio": market_cap / financials["book_value"],
            "peg_ratio": (market_cap / financials["net_income"]) / (
                financials["earnings_growth_rate"] * 100
            ),
            "price_to_sales": market_cap / financials["revenue"]
        }

    def calculate_liquidity_ratios(self, financials: Dict) -> Dict[str, float]:
        """
        Calculate liquidity ratios.

        Args:
            financials: Dictionary with financial data

        Returns:
            Dictionary with liquidity ratios
        """
        return {
            "current_ratio": (
                financials["current_assets"] / financials["current_liabilities"]
            ),
            "quick_ratio": (
                (financials["current_assets"] - financials["inventory"]) /
                financials["current_liabilities"]
            ),
            "debt_to_equity": (
                financials["total_debt"] / financials["equity"]
            )
        }
