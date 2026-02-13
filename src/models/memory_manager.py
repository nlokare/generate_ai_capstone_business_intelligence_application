"""
Memory management for contextual conversations.
Handles conversation history and context retention.
"""

from typing import List, Dict, Any, Optional
from collections import deque
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ConversationMemoryManager:
    """Manages conversation memory and context."""

    def __init__(self, k: int = 5):
        """
        Initialize memory manager.

        Args:
            k: Number of recent message exchanges to keep
        """
        self.k = k
        # Use deque with maxlen to automatically maintain window size
        self.messages: deque = deque(maxlen=k * 2)  # k exchanges = 2k messages

    def add_user_message(self, message: str):
        """
        Add a user message to memory.

        Args:
            message: User message text
        """
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        """
        Add an AI message to memory.

        Args:
            message: AI message text
        """
        self.messages.append(AIMessage(content=message))

    def add_exchange(self, user_message: str, ai_message: str):
        """
        Add a complete user-AI exchange to memory.

        Args:
            user_message: User message
            ai_message: AI response
        """
        self.add_user_message(user_message)
        self.add_ai_message(ai_message)

    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get the chat history.

        Returns:
            List of messages in history
        """
        return list(self.messages)

    def get_formatted_history(self) -> str:
        """
        Get formatted chat history as a string.

        Returns:
            Formatted conversation history
        """
        formatted = []

        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                formatted.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted.append(f"Assistant: {msg.content}")

        return "\n".join(formatted)

    def clear(self):
        """Clear the conversation memory."""
        self.messages.clear()

    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get memory variables for use in chains.

        Returns:
            Dictionary of memory variables
        """
        return {"chat_history": list(self.messages)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        """
        Save context from a conversation turn.

        Args:
            inputs: Input dictionary
            outputs: Output dictionary
        """
        if "input" in inputs:
            self.add_user_message(inputs["input"])
        if "output" in outputs:
            self.add_ai_message(outputs["output"])


class SessionMemoryManager:
    """Manages multiple conversation sessions."""

    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, ConversationMemoryManager] = {}

    def get_session(self, session_id: str, k: int = 5) -> ConversationMemoryManager:
        """
        Get or create a session.

        Args:
            session_id: Unique session identifier
            k: Number of exchanges to remember

        Returns:
            ConversationMemoryManager for the session
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemoryManager(k=k)

        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        """
        Clear a specific session.

        Args:
            session_id: Session to clear
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def delete_session(self, session_id: str):
        """
        Delete a session.

        Args:
            session_id: Session to delete
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def list_sessions(self) -> List[str]:
        """
        List all active sessions.

        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
