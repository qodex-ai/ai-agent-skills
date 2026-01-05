"""
Contract Analysis Module

Analyzes contracts to extract key terms, obligations, rights, and liabilities.
"""

import re
from typing import Dict, List


class ContractAnalyzer:
    """Analyzes contract documents for key information."""

    def analyze_contract(self, contract_data: Dict) -> Dict:
        """
        Perform comprehensive contract analysis.

        Args:
            contract_data: Dictionary with parsed contract data

        Returns:
            Dictionary with contract analysis
        """
        analysis = {
            "contract_type": self._identify_contract_type(contract_data),
            "parties_involved": contract_data.get("parties"),
            "key_terms": self._extract_key_terms(contract_data),
            "obligations": self._extract_obligations(contract_data),
            "rights": self._extract_rights(contract_data),
            "liabilities": self._extract_liabilities(contract_data),
            "termination_conditions": self._extract_termination(contract_data),
            "dispute_resolution": self._extract_dispute_resolution(contract_data)
        }
        return analysis

    def _identify_contract_type(self, contract_data: Dict) -> str:
        """Identify the type of contract."""
        keywords = {
            "nda": ["confidential", "non-disclosure", "proprietary"],
            "employment": ["employment", "salary", "position", "employee"],
            "service": ["services", "service provider", "agreement"],
            "purchase": ["purchase", "sale", "buyer", "seller"],
            "lease": ["lease", "tenant", "landlord", "rent"]
        }

        text = contract_data.get("full_text", "").lower()
        scores = {}

        for contract_type, keywords_list in keywords.items():
            scores[contract_type] = sum(
                1 for keyword in keywords_list if keyword in text
            )

        return max(scores, key=scores.get) if scores else "unknown"

    def _extract_key_terms(self, contract_data: Dict) -> Dict:
        """Extract key terms from contract."""
        key_terms = {
            "effective_date": self._find_date(contract_data, "effective|commencement"),
            "expiration_date": self._find_date(contract_data, "expiration|end|termination"),
            "renewal_terms": self._find_renewal_terms(contract_data),
            "payment_terms": self._find_payment_terms(contract_data),
            "performance_metrics": self._find_performance_metrics(contract_data)
        }
        return key_terms

    def _extract_obligations(self, contract_data: Dict) -> List[str]:
        """Extract obligations from contract."""
        text = contract_data.get("full_text", "")
        pattern = r'(?:shall|must|agree to|required to)\s+(.+?)(?:\.|;|,)'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        return [match.group(1) for match in matches]

    def _extract_rights(self, contract_data: Dict) -> List[str]:
        """Extract rights from contract."""
        text = contract_data.get("full_text", "")
        pattern = r'(?:may|can|has the right to|entitled to)\s+(.+?)(?:\.|;|,)'
        matches = re.finditer(pattern, text, re.IGNORECASE)
        return [match.group(1) for match in matches]

    def _extract_liabilities(self, contract_data: Dict) -> List[str]:
        """Extract liability clauses from contract."""
        # Placeholder for liability extraction
        return []

    def _extract_termination(self, contract_data: Dict) -> Dict:
        """Extract termination conditions from contract."""
        # Placeholder for termination extraction
        return {}

    def _extract_dispute_resolution(self, contract_data: Dict) -> Dict:
        """Extract dispute resolution clauses."""
        # Placeholder for dispute resolution extraction
        return {}

    def _find_date(self, contract_data: Dict, pattern: str) -> str:
        """Find date matching pattern."""
        # Placeholder for date finding
        return ""

    def _find_renewal_terms(self, contract_data: Dict) -> Dict:
        """Find renewal terms."""
        # Placeholder for renewal terms extraction
        return {}

    def _find_payment_terms(self, contract_data: Dict) -> Dict:
        """Find payment terms."""
        # Placeholder for payment terms extraction
        return {}

    def _find_performance_metrics(self, contract_data: Dict) -> List[str]:
        """Find performance metrics."""
        # Placeholder for performance metrics extraction
        return []
