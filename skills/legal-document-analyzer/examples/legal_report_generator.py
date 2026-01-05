"""
Legal Report Generation Module

Generates comprehensive legal analysis reports for contracts.
"""

from typing import Dict, List


class LegalReportGenerator:
    """Generates legal analysis reports."""

    def generate_analysis_report(self, analysis: Dict, risks: Dict) -> str:
        """
        Generate comprehensive contract analysis report.

        Args:
            analysis: Contract analysis results
            risks: Risk assessment results

        Returns:
            Formatted report as string
        """
        report = f"""
# Contract Analysis Report

## Contract Overview
- Type: {analysis.get('contract_type')}
- Parties: {analysis.get('parties_involved')}

## Key Terms
{self._format_key_terms(analysis.get('key_terms', {}))}

## Obligations & Rights
### Obligations
{self._format_list(analysis.get('obligations', []))}

### Rights
{self._format_list(analysis.get('rights', []))}

## Risk Assessment
{self._format_risks(risks)}

## Recommendations
{self._generate_recommendations(analysis, risks)}
        """
        return report

    def _format_key_terms(self, key_terms: Dict) -> str:
        """Format key terms for report."""
        formatted = ""
        for term, value in key_terms.items():
            formatted += f"- **{term.replace('_', ' ').title()}**: {value}\n"
        return formatted

    def _format_list(self, items: List) -> str:
        """Format list items for report."""
        if not items:
            return "None identified"
        return "\n".join([f"- {item}" for item in items])

    def _format_risks(self, risks: Dict) -> str:
        """Format risks for report."""
        formatted = ""
        for risk_type, risk_list in risks.items():
            formatted += f"\n### {risk_type.replace('_', ' ').title()}\n"
            for risk in risk_list:
                formatted += f"- **[{risk.get('severity')}]** {risk.get('description')}\n"
                formatted += f"  Recommendation: {risk.get('recommendation')}\n"
        return formatted

    def _generate_recommendations(self, analysis: Dict, risks: Dict) -> str:
        """Generate overall recommendations."""
        # Placeholder for recommendation generation
        return "Review all identified risks and implement recommended actions."
