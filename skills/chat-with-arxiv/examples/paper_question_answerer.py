"""
Paper Question Answerer Module

Handles Q&A about research papers using RAG.
"""

from typing import Dict, List
import numpy as np


class PaperQuestionAnswerer:
    """Answers questions about research papers."""

    def __init__(self):
        """Initialize answerer."""
        self.retriever = None  # Will be initialized with ArXivPaperRetriever
        self.processor = None  # Will be initialized with PaperContentProcessor
        self.embeddings = None  # Will be initialized with embedding model
        self.llm = None  # Will be initialized with LLM

    def answer_question(self, question: str, top_k_papers: int = 5) -> Dict:
        """Answer question using papers from ArXiv."""
        # Step 1: Search for relevant papers
        papers = self.retriever.search_papers(question, max_results=top_k_papers)

        # Step 2: Process papers and chunk content
        paper_chunks = []
        paper_sources = []

        for paper in papers:
            try:
                content = self.processor.download_and_process_paper(
                    paper["pdf_url"],
                    paper["arxiv_id"]
                )

                if content:
                    chunks = self.processor.chunk_paper_for_rag(content["text"])
                    paper_chunks.extend(chunks)
                    paper_sources.append(paper)

            except Exception as e:
                print(f"Error processing {paper['arxiv_id']}: {e}")

        # Step 3: Retrieve most relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(question, paper_chunks)

        # Step 4: Generate answer
        context = "\n\n".join(relevant_chunks)
        answer = self.generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "sources": paper_sources,
            "relevant_chunks": relevant_chunks
        }

    def retrieve_relevant_chunks(self, query: str, chunks: List[str], top_k: int = 5) -> List[str]:
        """Retrieve most relevant chunks for query."""
        if not chunks or not self.embeddings:
            return chunks[:top_k]

        query_embedding = self.embeddings.encode(query)

        chunk_embeddings = [self.embeddings.encode(chunk) for chunk in chunks]

        similarities = [
            np.dot(query_embedding, chunk_emb) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            for chunk_emb in chunk_embeddings
        ]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [chunks[i] for i in top_indices if similarities[i] > 0.3]

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        prompt = f"""Based on the following research paper excerpts, answer the question.

Question: {question}

Context from research papers:
{context}

Answer: """

        if self.llm:
            answer = self.llm.generate(prompt, max_tokens=500)
            return answer.strip()
        else:
            return "LLM not configured"

    def answer_with_citations(self, question: str) -> Dict:
        """Answer question with proper citations."""
        result = self.answer_question(question)

        citations = self.format_citations(result["sources"])

        return {
            "answer": result["answer"],
            "citations": citations,
            "source_papers": result["sources"]
        }

    def format_citations(self, papers: List[Dict]) -> List[str]:
        """Format paper citations."""
        citations = []

        for paper in papers:
            authors = paper['authors'][:3] if paper['authors'] else ["Unknown"]
            citation = f"{', '.join(authors)} et al. ({paper['published'].year}). "
            citation += f"{paper['title']}. ArXiv:{paper['arxiv_id']}"
            citations.append(citation)

        return citations
