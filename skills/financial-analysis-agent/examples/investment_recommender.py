"""
Investment Recommendation Module

Generates investment recommendations based on technical and fundamental analysis.
"""

from typing import Dict


class InvestmentRecommender:
    """Generates investment recommendations based on analysis."""

    def generate_recommendation(self, analysis_results: Dict) -> Dict:
        """
        Generate investment recommendation.

        Args:
            analysis_results: Dictionary with analysis results

        Returns:
            Dictionary with recommendation, confidence, reasoning, price target, and risk level
        """
        score = self._calculate_investment_score(analysis_results)

        if score >= 8:
            action = "STRONG BUY"
            reason = analysis_results["bullish_factors"]
        elif score >= 6:
            action = "BUY"
            reason = analysis_results["bullish_factors"]
        elif score >= 4:
            action = "HOLD"
            reason = "Mixed signals"
        elif score >= 2:
            action = "SELL"
            reason = analysis_results["bearish_factors"]
        else:
            action = "STRONG SELL"
            reason = analysis_results["bearish_factors"]

        return {
            "action": action,
            "confidence": score / 10,
            "reasoning": reason,
            "price_target": self._calculate_price_target(analysis_results),
            "risk_level": self._assess_risk_level(analysis_results)
        }

    def _calculate_investment_score(self, results: Dict) -> float:
        """
        Calculate investment score.

        Args:
            results: Analysis results

        Returns:
            Score between 0 and 10
        """
        score = 5  # Start at neutral

        # Technical signals
        if results.get("technical_signal") == "bullish":
            score += 1.5
        elif results.get("technical_signal") == "bearish":
            score -= 1.5

        # Fundamental strength
        if results.get("pe_ratio_attractive"):
            score += 1
        if results.get("strong_cash_flow"):
            score += 1
        if results.get("dividend_growth"):
            score += 0.5

        # Risk factors
        if results.get("high_debt"):
            score -= 1
        if results.get("declining_revenue"):
            score -= 1.5

        return max(0, min(10, score))

    def _calculate_price_target(self, analysis_results: Dict) -> float:
        """
        Calculate price target.

        Args:
            analysis_results: Analysis results

        Returns:
            Estimated price target
        """
        # Placeholder for price target calculation
        return 0.0

    def _assess_risk_level(self, analysis_results: Dict) -> str:
        """
        Assess risk level.

        Args:
            analysis_results: Analysis results

        Returns:
            Risk level (Low, Medium, High)
        """
        # Placeholder for risk assessment
        return "Medium"
