"""
Basic RAG Implementation

Simplest RAG implementation demonstrating document chunking,
embedding generation, vector storage, and retrieval.
"""

from typing import List, Tuple


class BasicRAG:
    """Basic RAG pipeline for document Q&A."""

    def __init__(self, embedding_model="all-MiniLM-L6-v2", vector_db=None):
        """
        Initialize basic RAG system.

        Args:
            embedding_model: Name of embedding model to use
            vector_db: Vector database instance (Chroma, Pinecone, etc.)
        """
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.chunks = []

    def chunk_documents(
        self, documents: List[str], chunk_size: int = 1000, overlap: int = 100
    ) -> List[str]:
        """
        Split documents into overlapping chunks.

        Args:
            documents: List of document texts
            chunk_size: Size of each chunk in characters
            overlap: Overlap between consecutive chunks

        Returns:
            List of text chunks
        """
        chunks = []
        for doc in documents:
            # Split each document into chunks
            for i in range(0, len(doc), chunk_size - overlap):
                chunk = doc[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)

        self.chunks = chunks
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text segments

        Returns:
            List of embedding vectors
        """
        # Placeholder - implement with actual embedding model
        # In practice, use: from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer(self.embedding_model)
        # embeddings = model.encode(texts)
        return [[0.1] * 384 for _ in texts]  # Dummy embeddings

    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents into vector database.

        Args:
            documents: List of document texts
        """
        chunks = self.chunk_documents(documents)
        embeddings = self.generate_embeddings(chunks)

        # Store in vector database
        for chunk, embedding in zip(chunks, embeddings):
            self.vector_db.add_text(chunk, embedding=embedding)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k relevant chunks for query.

        Args:
            query: User query
            k: Number of results to retrieve

        Returns:
            List of (chunk, similarity_score) tuples
        """
        query_embedding = self.generate_embeddings([query])[0]
        results = self.vector_db.query(query_embedding, k=k)
        return results

    def generate_answer(self, query: str, context: str, llm_client) -> str:
        """
        Generate answer using retrieved context.

        Args:
            query: User query
            context: Retrieved context
            llm_client: LLM client instance

        Returns:
            Generated answer
        """
        prompt = f"""You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

        response = llm_client.generate(prompt)
        return response

    def query(self, query: str, llm_client, k: int = 5) -> Tuple[str, List[str]]:
        """
        Complete RAG pipeline: retrieve and generate.

        Args:
            query: User query
            llm_client: LLM client instance
            k: Number of chunks to retrieve

        Returns:
            Tuple of (answer, retrieved_chunks)
        """
        # Retrieve relevant chunks
        retrieved = self.retrieve(query, k=k)
        chunks = [item[0] for item in retrieved]
        context = "\n\n".join(chunks)

        # Generate answer
        answer = self.generate_answer(query, context, llm_client)

        return answer, chunks
