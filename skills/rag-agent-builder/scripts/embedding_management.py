"""
Embedding Management for RAG

Handles embedding generation, normalization, and caching.
"""

from typing import List, Dict, Optional, Tuple
import math


class EmbeddingManager:
    """Manages embedding generation and storage."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_embeddings: bool = True):
        """
        Initialize embedding manager.

        Args:
            model_name: Name of embedding model
            cache_embeddings: Whether to cache generated embeddings
        """
        self.model_name = model_name
        self.cache = {} if cache_embeddings else None

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text segments

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            if self.cache is not None and text in self.cache:
                embeddings.append(self.cache[text])
            else:
                # Placeholder - use actual embedding model
                # from sentence_transformers import SentenceTransformer
                # model = SentenceTransformer(self.model_name)
                # embedding = model.encode(text)

                embedding = self._dummy_embedding(text)
                if self.cache is not None:
                    self.cache[text] = embedding
                embeddings.append(embedding)

        return embeddings

    def _dummy_embedding(self, text: str) -> List[float]:
        """Generate placeholder embedding."""
        # Create deterministic dummy embedding based on text
        seed = sum(ord(c) for c in text[:10])
        return [math.sin(seed + i) for i in range(384)]

    def normalize_embeddings(
        self, embeddings: List[List[float]]
    ) -> List[List[float]]:
        """
        Normalize embeddings to unit vectors.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Normalized embeddings
        """
        normalized = []
        for embedding in embeddings:
            norm = math.sqrt(sum(x**2 for x in embedding))
            if norm > 0:
                normalized_emb = [x / norm for x in embedding]
            else:
                normalized_emb = embedding
            normalized.append(normalized_emb)

        return normalized

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(x**2 for x in embedding1))
        norm2 = math.sqrt(sum(x**2 for x in embedding2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_cache_stats(self) -> Dict:
        """Get embedding cache statistics."""
        if self.cache is None:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "cached_embeddings": len(self.cache),
            "cache_size_estimate_mb": len(self.cache) * 384 * 4 / (1024 * 1024),
        }

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache is not None:
            self.cache.clear()


class EmbeddingQualityAssessment:
    """Assess quality of embeddings."""

    @staticmethod
    def embedding_variance(embeddings: List[List[float]]) -> float:
        """
        Calculate variance across embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Variance metric
        """
        if not embeddings:
            return 0.0

        # Calculate mean embedding
        dim = len(embeddings[0])
        mean_embedding = [
            sum(emb[i] for emb in embeddings) / len(embeddings) for i in range(dim)
        ]

        # Calculate variance
        variance = sum(
            sum((emb[i] - mean_embedding[i]) ** 2 for i in range(dim))
            for emb in embeddings
        ) / len(embeddings)

        return variance

    @staticmethod
    def embedding_distribution_quality(embeddings: List[List[float]]) -> Dict:
        """
        Assess quality of embedding distribution.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Quality assessment dict
        """
        if not embeddings:
            return {"quality": "unknown", "metrics": {}}

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = EmbeddingManager.compute_similarity(
                    embeddings[i], embeddings[j]
                )
                similarities.append(sim)

        if not similarities:
            return {"quality": "unknown", "metrics": {}}

        avg_similarity = sum(similarities) / len(similarities)
        max_similarity = max(similarities)
        min_similarity = min(similarities)

        # Assess quality
        if avg_similarity < 0.3:
            quality = "good"  # Diverse embeddings
        elif avg_similarity < 0.6:
            quality = "moderate"
        else:
            quality = "poor"  # Too similar

        return {
            "quality": quality,
            "metrics": {
                "avg_similarity": avg_similarity,
                "max_similarity": max_similarity,
                "min_similarity": min_similarity,
                "diversity": 1 - avg_similarity,
            },
        }

    @staticmethod
    def detect_dead_dimensions(
        embeddings: List[List[float]], threshold: float = 1e-6
    ) -> List[int]:
        """
        Detect dimensions with low variance (dead dimensions).

        Args:
            embeddings: List of embedding vectors
            threshold: Variance threshold

        Returns:
            List of dead dimension indices
        """
        if not embeddings:
            return []

        dim = len(embeddings[0])
        dead_dims = []

        for d in range(dim):
            values = [emb[d] for emb in embeddings]
            variance = sum((v - sum(values) / len(values)) ** 2 for v in values) / len(
                values
            )
            if variance < threshold:
                dead_dims.append(d)

        return dead_dims
