"""
Source Evaluation Module

Evaluates and scores information sources for credibility.
"""

from datetime import datetime
from typing import Dict


class SourceEvaluator:
    """Evaluates credibility of information sources."""

    def evaluate_source(self, source: Dict) -> int:
        """
        Evaluate source credibility.

        Args:
            source: Source object with attributes

        Returns:
            Credibility score (0-100)
        """
        score = 0

        # Authority: Is author qualified?
        if hasattr(source, 'author_expertise_level') and source.author_expertise_level > 0.7:
            score += 25

        # Credibility: Is publisher trusted?
        if hasattr(source, 'publisher_reputation') and source.publisher_reputation > 0.8:
            score += 25

        # Recency: Is information current?
        if hasattr(source, 'publication_date'):
            days_old = (datetime.now() - source.publication_date).days
            if days_old < 365:
                score += 25

        # Bias: Are there clear biases?
        if hasattr(source, 'bias_score') and source.bias_score < 0.3:
            score += 25

        return score
