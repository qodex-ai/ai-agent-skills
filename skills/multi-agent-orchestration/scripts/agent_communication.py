"""
Agent Communication Management

Handle agent-to-agent communication, message passing, and shared state.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class MessageType(Enum):
    """Types of messages between agents."""

    DIRECT = "direct"
    BROADCAST = "broadcast"
    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ERROR = "error"


@dataclass
class Message:
    """Message between agents."""

    sender: str
    recipient: str
    content: str
    message_type: MessageType
    timestamp: float = field(default_factory=lambda: 0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "type": self.message_type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class MessageBroker:
    """Central message broker for agent communication."""

    def __init__(self):
        """Initialize message broker."""
        self.message_queue: List[Message] = []
        self.agent_inboxes: Dict[str, List[Message]] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}

    def send_message(self, message: Message) -> bool:
        """
        Send message from one agent to another.

        Args:
            message: Message object

        Returns:
            Whether message was delivered
        """
        self.message_queue.append(message)

        # Add to recipient's inbox
        if message.recipient not in self.agent_inboxes:
            self.agent_inboxes[message.recipient] = []
        self.agent_inboxes[message.recipient].append(message)

        # Trigger handlers
        self._trigger_handlers(message)

        return True

    def broadcast_message(self, sender: str, content: str, recipients: List[str]):
        """
        Broadcast message to multiple agents.

        Args:
            sender: Sending agent
            content: Message content
            recipients: List of recipient agents
        """
        for recipient in recipients:
            message = Message(
                sender=sender,
                recipient=recipient,
                content=content,
                message_type=MessageType.BROADCAST,
            )
            self.send_message(message)

    def request_response(
        self, sender: str, recipient: str, content: str, timeout: float = 5.0
    ) -> Optional[Message]:
        """
        Send request and wait for response.

        Args:
            sender: Requesting agent
            recipient: Agent to respond
            content: Request content
            timeout: Response timeout in seconds

        Returns:
            Response message or None
        """
        message = Message(
            sender=sender,
            recipient=recipient,
            content=content,
            message_type=MessageType.REQUEST,
            metadata={"timeout": timeout},
        )
        self.send_message(message)

        # Placeholder - wait for response
        # In production, implement actual timeout/wait mechanism
        return None

    def get_inbox(self, agent: str) -> List[Message]:
        """Get messages for specific agent."""
        return self.agent_inboxes.get(agent, [])

    def clear_inbox(self, agent: str) -> None:
        """Clear agent's inbox."""
        self.agent_inboxes[agent] = []

    def register_handler(
        self, message_type: MessageType, handler: Callable
    ) -> None:
        """Register handler for message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    def _trigger_handlers(self, message: Message) -> None:
        """Trigger registered handlers."""
        handlers = self.message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception:
                pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        type_counts = {}
        for message in self.message_queue:
            msg_type = message.message_type.value
            type_counts[msg_type] = type_counts.get(msg_type, 0) + 1

        return {
            "total_messages": len(self.message_queue),
            "by_type": type_counts,
            "total_agents": len(self.agent_inboxes),
            "agents": list(self.agent_inboxes.keys()),
        }


class SharedMemory:
    """Shared memory for tool-mediated agent communication."""

    def __init__(self):
        """Initialize shared memory."""
        self.memory: Dict[str, Any] = {}
        self.access_log: List[Dict] = []

    def write(self, key: str, value: Any, agent: str = "system") -> None:
        """
        Write to shared memory.

        Args:
            key: Memory key
            value: Value to store
            agent: Agent writing
        """
        self.memory[key] = {
            "value": value,
            "writer": agent,
            "access_count": 0,
        }
        self.access_log.append({
            "action": "write",
            "key": key,
            "agent": agent,
        })

    def read(self, key: str, agent: str = "system") -> Optional[Any]:
        """
        Read from shared memory.

        Args:
            key: Memory key
            agent: Agent reading

        Returns:
            Stored value or None
        """
        if key in self.memory:
            entry = self.memory[key]
            entry["access_count"] += 1
            self.access_log.append({
                "action": "read",
                "key": key,
                "agent": agent,
            })
            return entry["value"]
        return None

    def append(self, key: str, value: Any, agent: str = "system") -> None:
        """Append to list in shared memory."""
        if key not in self.memory:
            self.memory[key] = {
                "value": [],
                "writer": agent,
                "access_count": 0,
            }
        if isinstance(self.memory[key]["value"], list):
            self.memory[key]["value"].append(value)

    def get_all(self) -> Dict[str, Any]:
        """Get all shared memory contents."""
        return {
            key: entry["value"] for key, entry in self.memory.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory access statistics."""
        total_accesses = sum(
            entry["access_count"] for entry in self.memory.values()
        )

        return {
            "total_keys": len(self.memory),
            "total_accesses": total_accesses,
            "access_log_size": len(self.access_log),
        }


class ContextManager:
    """Manage context sharing between agents."""

    def __init__(self):
        """Initialize context manager."""
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.global_context: Dict[str, Any] = {}

    def create_context(self, context_id: str, initial_data: Optional[Dict] = None) -> None:
        """Create new context."""
        self.contexts[context_id] = initial_data or {}

    def update_context(self, context_id: str, data: Dict) -> None:
        """Update context data."""
        if context_id in self.contexts:
            self.contexts[context_id].update(data)

    def get_context(self, context_id: str) -> Dict[str, Any]:
        """Get context data."""
        return self.contexts.get(context_id, {})

    def set_global_context(self, key: str, value: Any) -> None:
        """Set global context variable."""
        self.global_context[key] = value

    def get_global_context(self, key: str) -> Optional[Any]:
        """Get global context variable."""
        return self.global_context.get(key)

    def context_to_string(self, context_id: str) -> str:
        """Convert context to formatted string."""
        context = self.get_context(context_id)
        return json.dumps(context, indent=2)


class CommunicationProtocol:
    """Define communication protocol between agents."""

    def __init__(self, broker: MessageBroker, shared_memory: SharedMemory):
        """Initialize protocol."""
        self.broker = broker
        self.shared_memory = shared_memory

    def request_analysis(
        self, requester: str, analyzer: str, subject: str
    ) -> None:
        """Request analysis from another agent."""
        message = Message(
            sender=requester,
            recipient=analyzer,
            content=f"Analyze: {subject}",
            message_type=MessageType.REQUEST,
            metadata={"action": "analyze"},
        )
        self.broker.send_message(message)

    def share_findings(self, source_agent: str, key: str, findings: Dict) -> None:
        """Share findings through shared memory."""
        self.shared_memory.write(key, findings, source_agent)

    def aggregate_results(
        self, agents: List[str], result_key: str
    ) -> Dict[str, Any]:
        """Aggregate results from multiple agents."""
        results = {}
        for agent in agents:
            agent_results = self.shared_memory.read(f"{agent}_{result_key}")
            if agent_results:
                results[agent] = agent_results

        return results

    def notify_status(self, agent: str, status: str) -> None:
        """Notify status update."""
        message = Message(
            sender=agent,
            recipient="orchestrator",
            content=f"Status: {status}",
            message_type=MessageType.FEEDBACK,
        )
        self.broker.send_message(message)

    def error_propagation(self, source_agent: str, error: str) -> None:
        """Propagate error to relevant agents."""
        message = Message(
            sender=source_agent,
            recipient="orchestrator",
            content=f"Error: {error}",
            message_type=MessageType.ERROR,
        )
        self.broker.send_message(message)
