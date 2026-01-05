"""
Agentic RAG Implementation

Agent that makes intelligent decisions about what to retrieve and when.
Supports iterative refinement for complex questions.
"""

from typing import List, Tuple, Optional, Dict
from enum import Enum


class RetrievalDecision(Enum):
    """Decisions the agent can make about retrieval."""

    RETRIEVE = "retrieve"
    REFINE_QUERY = "refine_query"
    GENERATE = "generate"
    SEARCH_EXPANSION = "search_expansion"


class AgenticRAG:
    """RAG system where agent controls retrieval strategy."""

    def __init__(self, llm_client, vector_db, max_iterations: int = 3):
        """
        Initialize agentic RAG.

        Args:
            llm_client: LLM client for reasoning
            vector_db: Vector database for retrieval
            max_iterations: Maximum refinement iterations
        """
        self.llm = llm_client
        self.vector_db = vector_db
        self.max_iterations = max_iterations
        self.conversation_history = []

    def decide_retrieval(self, query: str, context: Optional[str] = None) -> Dict:
        """
        Use LLM to decide on retrieval strategy.

        Args:
            query: User query
            context: Previous context if iterating

        Returns:
            Decision dict with action and parameters
        """
        decision_prompt = f"""Based on this query, decide what retrieval strategy to use:

Query: {query}
{"Previous context: " + context if context else ""}

Decide one of:
1. RETRIEVE - directly retrieve documents
2. REFINE_QUERY - reformulate query first
3. SEARCH_EXPANSION - expand query with related terms
4. GENERATE - answer without retrieval (general knowledge)

Respond with decision and reasoning."""

        response = self.llm.generate(decision_prompt)
        return self._parse_decision(response, query)

    def _parse_decision(self, response: str, original_query: str) -> Dict:
        """Parse LLM decision response."""
        # Placeholder logic
        if "refine" in response.lower():
            return {"action": RetrievalDecision.REFINE_QUERY, "query": original_query}
        elif "expand" in response.lower():
            return {
                "action": RetrievalDecision.SEARCH_EXPANSION,
                "query": original_query,
            }
        elif "generate" in response.lower():
            return {"action": RetrievalDecision.GENERATE, "query": original_query}
        else:
            return {"action": RetrievalDecision.RETRIEVE, "query": original_query}

    def refine_query(self, original_query: str) -> str:
        """
        Refine query for better retrieval.

        Args:
            original_query: Original user query

        Returns:
            Refined query
        """
        refine_prompt = f"""Reformulate this query to be more specific and retrieval-friendly:

Original: {original_query}

Refined query:"""

        refined = self.llm.generate(refine_prompt)
        return refined.strip()

    def expand_query(self, query: str, num_expansions: int = 2) -> List[str]:
        """
        Generate related search queries.

        Args:
            query: Original query
            num_expansions: Number of related queries

        Returns:
            List of expanded queries
        """
        expand_prompt = f"""Generate {num_expansions} related search queries for:

Query: {query}

Related queries (one per line):"""

        response = self.llm.generate(expand_prompt)
        queries = response.strip().split("\n")
        return [q.strip() for q in queries if q.strip()]

    def retrieve_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve documents for query.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of (document, score) tuples
        """
        # Convert query to embedding and retrieve
        results = self.vector_db.query(query, k=k)
        return results

    def evaluate_retrieved_docs(
        self, documents: List[str], query: str
    ) -> Tuple[bool, float]:
        """
        Evaluate if retrieved documents are sufficient.

        Args:
            documents: Retrieved documents
            query: Original query

        Returns:
            Tuple of (sufficient, confidence_score)
        """
        eval_prompt = f"""Are these documents sufficient to answer the query?

Query: {query}

Documents:
{"".join(f"{i+1}. {doc[:200]}..." for i, doc in enumerate(documents))}

Respond with: YES or NO, and confidence (0-1)"""

        response = self.llm.generate(eval_prompt)

        # Parse response
        sufficient = "yes" in response.lower()
        confidence = 0.8 if sufficient else 0.4  # Placeholder

        return sufficient, confidence

    def execute(self, query: str) -> Dict:
        """
        Execute agentic RAG pipeline.

        Args:
            query: User query

        Returns:
            Dict with answer and metadata
        """
        current_query = query
        iteration = 0
        all_retrieved_docs = []

        while iteration < self.max_iterations:
            # Decide retrieval strategy
            decision = self.decide_retrieval(current_query)

            if decision["action"] == RetrievalDecision.REFINE_QUERY:
                current_query = self.refine_query(current_query)
                iteration += 1
                continue

            elif decision["action"] == RetrievalDecision.SEARCH_EXPANSION:
                expanded_queries = self.expand_query(current_query)
                # Retrieve for all queries
                for expanded_q in expanded_queries:
                    docs = self.retrieve_documents(expanded_q, k=3)
                    all_retrieved_docs.extend(docs)
                iteration += 1
                continue

            elif decision["action"] == RetrievalDecision.RETRIEVE:
                docs = self.retrieve_documents(current_query, k=5)
                all_retrieved_docs.extend(docs)

                # Evaluate retrieved documents
                doc_texts = [doc[0] for doc in docs]
                sufficient, confidence = self.evaluate_retrieved_docs(
                    doc_texts, query
                )

                if sufficient or iteration >= self.max_iterations - 1:
                    break

                # Refine for next iteration
                current_query = self.refine_query(query)
                iteration += 1
                continue

            else:  # GENERATE
                break

        # Generate answer from all retrieved documents
        context = "\n\n".join([doc[0] for doc in all_retrieved_docs])
        answer = self.generate_answer(query, context)

        return {
            "answer": answer,
            "query": query,
            "refined_query": current_query,
            "iterations": iteration,
            "retrieved_docs_count": len(all_retrieved_docs),
            "documents": [doc[0] for doc in all_retrieved_docs],
        }

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate final answer with context.

        Args:
            query: Original query
            context: Retrieved context

        Returns:
            Generated answer
        """
        answer_prompt = f"""Based on the provided context, answer this question:

Context:
{context}

Question: {query}

Answer:"""

        return self.llm.generate(answer_prompt)
