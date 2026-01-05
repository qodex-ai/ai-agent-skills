"""
RAG Evaluation Metrics

Evaluate retrieval quality and answer generation quality.
"""

from typing import List, Dict, Tuple, Optional
import math


class RetrievalMetrics:
    """Calculate retrieval quality metrics."""

    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate precision@k.

        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Cutoff position

        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0

        top_k_retrieved = set(retrieved[:k])
        relevant_set = set(relevant)
        matches = len(top_k_retrieved & relevant_set)
        return matches / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate recall@k.

        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents
            k: Cutoff position

        Returns:
            Recall@k score
        """
        if not relevant:
            return 0.0

        top_k_retrieved = set(retrieved[:k])
        relevant_set = set(relevant)
        matches = len(top_k_retrieved & relevant_set)
        return matches / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            retrieved: List of retrieved documents
            relevant: List of relevant documents

        Returns:
            MRR score
        """
        relevant_set = set(relevant)
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg(retrieved: List[Tuple[str, float]], relevant: List[str], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).

        Args:
            retrieved: List of (document, score) tuples
            relevant: List of relevant documents
            k: Cutoff position

        Returns:
            NDCG score
        """
        relevant_set = set(relevant)
        docs = [doc for doc, _ in retrieved[:k]]

        # Calculate DCG
        dcg = 0.0
        for rank, doc in enumerate(docs, 1):
            if doc in relevant_set:
                dcg += 1.0 / math.log2(rank + 1)

        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for rank in range(1, min(k, len(relevant_set)) + 1):
            idcg += 1.0 / math.log2(rank + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def compute_retrieval_scores(
        retrieved: List[Tuple[str, float]], relevant: List[str]
    ) -> Dict[str, float]:
        """
        Compute all retrieval metrics.

        Args:
            retrieved: List of (document, score) tuples
            relevant: List of relevant documents

        Returns:
            Dict of metric names to scores
        """
        docs = [doc for doc, _ in retrieved]

        return {
            "precision_at_5": RetrievalMetrics.precision_at_k(docs, relevant, 5),
            "precision_at_10": RetrievalMetrics.precision_at_k(docs, relevant, 10),
            "recall_at_5": RetrievalMetrics.recall_at_k(docs, relevant, 5),
            "recall_at_10": RetrievalMetrics.recall_at_k(docs, relevant, 10),
            "mrr": RetrievalMetrics.mean_reciprocal_rank(docs, relevant),
            "ndcg_5": RetrievalMetrics.ndcg(retrieved, relevant, k=5),
            "ndcg_10": RetrievalMetrics.ndcg(retrieved, relevant, k=10),
        }


class AnswerQualityMetrics:
    """Evaluate answer quality."""

    @staticmethod
    def has_source_citations(answer: str, source_documents: List[str]) -> bool:
        """
        Check if answer cites sources.

        Args:
            answer: Generated answer
            source_documents: Retrieved source documents

        Returns:
            Whether answer contains source citations
        """
        # Simple check - looks for document references
        citations = ["source", "document", "according to", "cited from", "[1]"]
        return any(citation in answer.lower() for citation in citations)

    @staticmethod
    def answer_length_quality(answer: str, min_length: int = 50) -> Dict:
        """
        Evaluate answer length quality.

        Args:
            answer: Generated answer
            min_length: Minimum acceptable length

        Returns:
            Length quality assessment
        """
        length = len(answer)
        word_count = len(answer.split())

        if length < min_length:
            quality = "too_short"
        elif length > 5000:
            quality = "too_long"
        else:
            quality = "appropriate"

        return {
            "quality": quality,
            "character_count": length,
            "word_count": word_count,
        }

    @staticmethod
    def answer_coherence_score(answer: str) -> float:
        """
        Estimate answer coherence (0-1).

        Args:
            answer: Generated answer

        Returns:
            Coherence score
        """
        # Simple heuristic - score based on structure
        sentences = answer.split(".")
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not valid_sentences:
            return 0.0

        # More sentences = better coherence (up to a point)
        coherence = min(len(valid_sentences) / 5.0, 1.0)
        return coherence

    @staticmethod
    def grounding_in_context(answer: str, context: str) -> float:
        """
        Estimate how well answer is grounded in context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Grounding score (0-1)
        """
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        overlap = len(answer_words & context_words)
        grounding = overlap / len(answer_words)

        return min(grounding, 1.0)


class RAGEvaluator:
    """Comprehensive RAG evaluation."""

    def __init__(self):
        """Initialize evaluator."""
        self.evaluation_history = []

    def evaluate_retrieval(
        self, retrieved: List[Tuple[str, float]], relevant: List[str], query: str
    ) -> Dict:
        """
        Evaluate retrieval step.

        Args:
            retrieved: List of (document, score) tuples
            relevant: List of relevant documents
            query: Original query

        Returns:
            Evaluation results
        """
        scores = RetrievalMetrics.compute_retrieval_scores(retrieved, relevant)

        result = {
            "step": "retrieval",
            "query": query,
            "metrics": scores,
            "retrieved_count": len(retrieved),
            "relevant_count": len(relevant),
        }

        self.evaluation_history.append(result)
        return result

    def evaluate_answer(self, answer: str, context: str, query: str) -> Dict:
        """
        Evaluate answer generation step.

        Args:
            answer: Generated answer
            context: Retrieved context
            query: Original query

        Returns:
            Evaluation results
        """
        has_citations = AnswerQualityMetrics.has_source_citations(answer, [context])
        length_quality = AnswerQualityMetrics.answer_length_quality(answer)
        coherence = AnswerQualityMetrics.answer_coherence_score(answer)
        grounding = AnswerQualityMetrics.grounding_in_context(answer, context)

        result = {
            "step": "answer",
            "query": query,
            "has_citations": has_citations,
            "length_quality": length_quality,
            "coherence": coherence,
            "grounding": grounding,
            "overall_quality": (coherence + grounding) / 2,
        }

        self.evaluation_history.append(result)
        return result

    def evaluate_rag_pipeline(
        self,
        query: str,
        retrieved: List[Tuple[str, float]],
        relevant_documents: List[str],
        answer: str,
        context: str,
    ) -> Dict:
        """
        Evaluate complete RAG pipeline.

        Args:
            query: User query
            retrieved: Retrieved documents
            relevant_documents: Relevant documents
            answer: Generated answer
            context: Retrieved context

        Returns:
            Complete evaluation
        """
        retrieval_eval = self.evaluate_retrieval(retrieved, relevant_documents, query)
        answer_eval = self.evaluate_answer(answer, context, query)

        return {
            "query": query,
            "retrieval": retrieval_eval,
            "answer": answer_eval,
            "overall_score": (
                retrieval_eval["metrics"]["ndcg_5"]
                + answer_eval["overall_quality"]
            )
            / 2,
        }

    def get_evaluation_summary(self) -> Dict:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {"evaluations_count": 0}

        retrieval_evals = [e for e in self.evaluation_history if e["step"] == "retrieval"]
        answer_evals = [e for e in self.evaluation_history if e["step"] == "answer"]

        avg_precision = (
            sum(e["metrics"]["precision_at_5"] for e in retrieval_evals)
            / len(retrieval_evals)
            if retrieval_evals
            else 0
        )

        avg_quality = (
            sum(e["overall_quality"] for e in answer_evals) / len(answer_evals)
            if answer_evals
            else 0
        )

        return {
            "total_evaluations": len(self.evaluation_history),
            "retrieval_evaluations": len(retrieval_evals),
            "answer_evaluations": len(answer_evals),
            "avg_retrieval_precision": avg_precision,
            "avg_answer_quality": avg_quality,
        }
