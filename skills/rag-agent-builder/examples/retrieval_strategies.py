"""
Advanced Retrieval Strategies for RAG

Implements hybrid search, reranking, and filtering techniques.
"""

from typing import List, Tuple, Optional
import math


class HybridRetriever:
    """Combines keyword (BM25) and semantic (embedding) search."""

    def __init__(self, bm25_retriever, vector_db, alpha: float = 0.5):
        """
        Initialize hybrid retriever.

        Args:
            bm25_retriever: BM25 keyword search instance
            vector_db: Vector database for semantic search
            alpha: Weight for combining results (0-1)
                  0 = pure keyword, 1 = pure semantic
        """
        self.bm25 = bm25_retriever
        self.vector_db = vector_db
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve using both keyword and semantic search.

        Args:
            query: User query
            k: Number of results

        Returns:
            Ranked list of (document, score) tuples
        """
        # Keyword search results
        keyword_results = self.bm25.search(query, k=k)
        keyword_scores = {doc: score for doc, score in keyword_results}

        # Semantic search results
        semantic_results = self.vector_db.query(query, k=k)
        semantic_scores = {doc: score for doc, score in semantic_results}

        # Combine scores
        combined_scores = {}
        all_docs = set(keyword_scores.keys()) | set(semantic_scores.keys())

        for doc in all_docs:
            keyword_score = keyword_scores.get(doc, 0.0)
            semantic_score = semantic_scores.get(doc, 0.0)
            combined = self.alpha * semantic_score + (1 - self.alpha) * keyword_score
            combined_scores[doc] = combined

        # Sort and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


class DocumentReranker:
    """Rerank retrieved documents by relevance."""

    def __init__(self, reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        """
        Initialize reranker.

        Args:
            reranker_model: Cross-encoder model name
        """
        self.model = reranker_model

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: User query
            documents: List of candidate documents
            top_k: Number of results to return

        Returns:
            Reranked list of (document, score) tuples
        """
        # Placeholder - use sentence_transformers.CrossEncoder in production
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder(self.model)
        # scores = model.predict([(query, doc) for doc in documents])

        # Simple placeholder: rank by query-document overlap
        scores = []
        query_words = set(query.lower().split())
        for doc in documents:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words) / len(query_words)
            scores.append(overlap)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class RelevanceFilter:
    """Filter retrieved documents by relevance threshold."""

    def __init__(self, threshold: float = 0.7):
        """
        Initialize filter.

        Args:
            threshold: Minimum relevance score (0-1)
        """
        self.threshold = threshold

    def filter_results(
        self, results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Filter results by relevance threshold.

        Args:
            results: List of (document, score) tuples

        Returns:
            Filtered results above threshold
        """
        return [
            (doc, score) for doc, score in results if score >= self.threshold
        ]

    def filter_and_log(
        self, results: List[Tuple[str, float]], query: str = ""
    ) -> Tuple[List[Tuple[str, float]], dict]:
        """
        Filter and return statistics.

        Args:
            results: List of (document, score) tuples
            query: Original query (for logging)

        Returns:
            Tuple of (filtered_results, statistics)
        """
        original_count = len(results)
        filtered = self.filter_results(results)
        filtered_count = len(filtered)

        stats = {
            "original_count": original_count,
            "filtered_count": filtered_count,
            "filtered_out": original_count - filtered_count,
            "threshold": self.threshold,
        }

        return filtered, stats


class ContextWindowManager:
    """Manage retrieved documents to fit within LLM context window."""

    def __init__(self, max_tokens: int = 3000, tokens_per_char: float = 0.25):
        """
        Initialize context manager.

        Args:
            max_tokens: Maximum tokens for context
            tokens_per_char: Approximate tokens per character
        """
        self.max_tokens = max_tokens
        self.tokens_per_char = tokens_per_char

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) * self.tokens_per_char)

    def fit_documents(self, documents: List[str]) -> Tuple[List[str], dict]:
        """
        Fit retrieved documents into context window.

        Args:
            documents: List of document chunks

        Returns:
            Tuple of (selected_documents, statistics)
        """
        selected = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = self.estimate_tokens(doc)
            if total_tokens + doc_tokens <= self.max_tokens:
                selected.append(doc)
                total_tokens += doc_tokens
            else:
                break

        stats = {
            "total_tokens_available": self.max_tokens,
            "total_tokens_used": total_tokens,
            "documents_selected": len(selected),
            "documents_truncated": len(documents) - len(selected),
        }

        return selected, stats

    def prepare_context(self, documents: List[str]) -> str:
        """
        Prepare context string from documents.

        Args:
            documents: List of document chunks

        Returns:
            Formatted context string
        """
        selected, stats = self.fit_documents(documents)
        context = "\n\n".join(selected)
        return context
