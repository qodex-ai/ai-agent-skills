"""
Vector Database Manager

Abstracts interactions with different vector database backends.
"""

from typing import List, Tuple, Dict, Optional
from abc import ABC, abstractmethod


class VectorDBBackend(ABC):
    """Abstract base for vector database backends."""

    @abstractmethod
    def add_text(self, text: str, embedding: List[float], metadata: Optional[Dict] = None) -> str:
        """Add text with embedding to database."""
        pass

    @abstractmethod
    def query(self, embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Query database for similar embeddings."""
        pass

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete document from database."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        """Get database statistics."""
        pass


class InMemoryVectorDB(VectorDBBackend):
    """Simple in-memory vector database for prototyping."""

    def __init__(self):
        """Initialize in-memory database."""
        self.documents = {}  # id -> (text, embedding, metadata)
        self.next_id = 0

    def add_text(
        self, text: str, embedding: List[float], metadata: Optional[Dict] = None
    ) -> str:
        """
        Add text to database.

        Args:
            text: Text content
            embedding: Embedding vector
            metadata: Optional metadata dict

        Returns:
            Document ID
        """
        doc_id = f"doc_{self.next_id}"
        self.documents[doc_id] = {
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self.next_id += 1
        return doc_id

    def query(self, embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Query for similar documents.

        Args:
            embedding: Query embedding
            k: Number of results

        Returns:
            List of (text, similarity_score) tuples
        """
        if not self.documents:
            return []

        # Calculate similarities
        similarities = []
        for doc_id, doc_data in self.documents.items():
            similarity = self._cosine_similarity(embedding, doc_data["embedding"])
            similarities.append((doc_data["text"], similarity))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def delete(self, document_id: str) -> bool:
        """Delete document."""
        if document_id in self.documents:
            del self.documents[document_id]
            return True
        return False

    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            "backend": "in_memory",
            "document_count": len(self.documents),
            "next_id": self.next_id,
        }

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(x**2 for x in vec1) ** 0.5
        norm2 = sum(x**2 for x in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class VectorDBManager:
    """Unified interface for vector database operations."""

    def __init__(self, backend: VectorDBBackend):
        """
        Initialize manager with backend.

        Args:
            backend: Vector database backend instance
        """
        self.backend = backend
        self.operation_count = 0

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
    ) -> List[str]:
        """
        Add multiple documents.

        Args:
            texts: List of text contents
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dicts

        Returns:
            List of document IDs
        """
        doc_ids = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else None
            doc_id = self.backend.add_text(text, embedding, metadata)
            doc_ids.append(doc_id)
            self.operation_count += 1

        return doc_ids

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results

        Returns:
            List of (text, score) tuples
        """
        self.operation_count += 1
        return self.backend.query(query_embedding, k=k)

    def remove_document(self, document_id: str) -> bool:
        """Remove document."""
        result = self.backend.delete(document_id)
        if result:
            self.operation_count += 1
        return result

    def get_database_info(self) -> Dict:
        """Get database information."""
        stats = self.backend.get_stats()
        stats["manager_operations"] = self.operation_count
        return stats

    def health_check(self) -> bool:
        """Check if database is healthy."""
        try:
            stats = self.backend.get_stats()
            return stats is not None
        except Exception:
            return False


class VectorDBFactory:
    """Factory for creating vector database instances."""

    @staticmethod
    def create_db(db_type: str = "in_memory", **kwargs) -> VectorDBBackend:
        """
        Create vector database instance.

        Args:
            db_type: Type of database ("in_memory", "chroma", "pinecone", etc.)
            **kwargs: Database-specific arguments

        Returns:
            Vector database instance
        """
        if db_type == "in_memory":
            return InMemoryVectorDB()

        elif db_type == "chroma":
            # from chromadb import Client
            # return ChromaVectorDB(**kwargs)
            raise NotImplementedError("Chroma backend not implemented")

        elif db_type == "pinecone":
            # from pinecone import Index
            # return PineconeVectorDB(**kwargs)
            raise NotImplementedError("Pinecone backend not implemented")

        else:
            raise ValueError(f"Unknown database type: {db_type}")
