"""
ArXiv Chatbot Module

Conversational interface for exploring ArXiv papers.
"""

from datetime import datetime
from typing import Dict, List


class ArXivChatbot:
    """Chatbot for interacting with ArXiv papers."""

    def __init__(self):
        """Initialize chatbot."""
        self.answerer = None  # Will be initialized with PaperQuestionAnswerer
        self.synthesizer = None  # Will be initialized with ResearchSynthesizer
        self.conversation_history = []
        self.context_manager = None  # Will be initialized with ConversationContextManager

    def chat(self, user_message: str) -> str:
        """Handle user message in conversation."""
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now()
        })

        # Determine query type
        query_type = self.classify_query(user_message)

        # Generate response
        if query_type == "single_paper_qa":
            response = self.handle_single_paper_query(user_message)
        elif query_type == "multi_paper_synthesis":
            response = self.handle_synthesis_query(user_message)
        elif query_type == "research_trend":
            response = self.handle_trend_query(user_message)
        else:
            response = self.handle_general_query(user_message)

        # Store response
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })

        return response

    def classify_query(self, message: str) -> str:
        """Classify type of user query."""
        message_lower = message.lower()

        if any(word in message_lower for word in ["trend", "recent", "latest"]):
            return "research_trend"
        elif any(word in message_lower for word in ["compare", "difference", "similar", "across"]):
            return "multi_paper_synthesis"
        elif any(word in message_lower for word in ["summarize", "explain", "about"]):
            return "single_paper_qa"
        else:
            return "general"

    def handle_single_paper_query(self, message: str) -> str:
        """Handle question about specific papers."""
        if not self.answerer:
            return "Paper question answerer not configured."

        result = self.answerer.answer_with_citations(message)

        response = f"{result['answer']}\n\n**Sources:**\n"
        response += "\n".join([f"- {citation}" for citation in result['citations']])

        return response

    def handle_synthesis_query(self, message: str) -> str:
        """Handle synthesis query across papers."""
        if not self.synthesizer:
            return "Synthesis capabilities not configured."

        topic = self.extract_topic_from_query(message)
        synthesis = self.synthesizer.synthesize_research_area(topic, num_papers=15)

        response = f"## Research Synthesis: {topic}\n\n"
        response += f"Analyzed {synthesis['papers_analyzed']} papers\n\n"
        response += f"**Key Themes:** {', '.join(synthesis.get('key_themes', [])[:5])}\n\n"
        response += f"**Summary:** {synthesis.get('synthesis', 'Summary not available')}\n"

        return response

    def handle_trend_query(self, message: str) -> str:
        """Handle research trend query."""
        if not self.answerer or not self.answerer.retriever:
            return "Trend analysis not configured."

        try:
            result = self.answerer.retriever.get_trending_papers("cs.AI", days=7)

            response = "## Recent Research Trends\n\n"
            for i, paper in enumerate(result[:5], 1):
                response += f"{i}. **{paper['title']}**\n"
                response += f"   Authors: {', '.join(paper['authors'][:2])}\n"
                response += f"   {paper['summary'][:200]}...\n\n"

            return response
        except Exception as e:
            return f"Error retrieving trends: {str(e)}"

    def handle_general_query(self, message: str) -> str:
        """Handle general knowledge query."""
        if not self.answerer:
            return "Question answerer not configured."

        return self.answerer.answer_question(message)

    def extract_topic_from_query(self, message: str) -> str:
        """Extract main topic from query."""
        return message.replace("?", "").replace(".", "")

    def get_conversation_context(self) -> List[Dict]:
        """Get conversation history for context."""
        return self.conversation_history

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
