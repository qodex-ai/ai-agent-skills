"""
Conversation Management Module

Manages chat conversations and context.
"""

from datetime import datetime
from typing import List, Dict


class ConversationManager:
    """Manages conversation history and context."""

    def __init__(self, max_history: int = 10):
        """
        Initialize conversation manager.

        Args:
            max_history: Maximum messages to keep in history
        """
        self.messages: List[Dict] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """Add message to history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })

        # Maintain size limit
        if len(self.messages) > self.max_history:
            self.messages.pop(0)

    def get_context(self) -> str:
        """Get conversation context for LLM."""
        return "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.messages
        ])

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []

    def get_messages(self) -> List[Dict]:
        """Get all messages."""
        return self.messages


def refine_question(current_question: str, conversation: List[str]) -> str:
    """Expand implicit references in question."""
    if len(current_question.split()) < 5:
        if current_question.startswith(("it ", "that ", "this ")):
            # Reference previous context
            if conversation:
                previous_context = conversation[-1][:100]
                refined = f"{previous_context} {current_question}"
                return refined

    return current_question


def generate_follow_up_questions(context: str, response: str, llm=None) -> List[str]:
    """Generate follow-up questions."""
    if not llm:
        return [
            "Can you provide more details?",
            "How does this relate to other topics?",
            "What are the next steps?"
        ]

    prompt = f"""
    Based on this Q&A, generate 3 relevant follow-up questions:
    Context: {context[:500]}
    Response: {response[:500]}
    """

    follow_ups = llm.generate(prompt)
    return follow_ups
