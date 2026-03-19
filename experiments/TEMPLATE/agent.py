"""Experiment template — copy this folder to experiments/h<id>/ and modify.

Public API (required by evaluator):
    create_session() -> str
    send_message(session_id, message) -> SendMessageResult
"""

from __future__ import annotations

# Re-export session helpers and types so the evaluator finds them here.
from agent_base import (
    RetrievedContext,
    ChatMessage,
    TokenUsage,
    SendMessageResult,
    Session,
    RetrievalTraceStep,
    create_session,
    get_session,
    _add_message,
    _sessions,
)

# Import the champion's business logic as a starting point.
# Replace or override functions below to test your hypothesis.
from agent import (
    retrieve,
    send_message,
)
