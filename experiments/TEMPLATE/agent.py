"""Experiment template — documents the required file structure.

To create an experiment, copy the full champion agent into your folder:

    mkdir -p experiments/h<id>
    touch experiments/h<id>/__init__.py
    cp agent.py experiments/h<id>/agent.py

Then edit experiments/h<id>/agent.py with your hypothesis changes.

Do NOT import from the root agent or patch it — each experiment must be
a self-contained copy so it stays reproducible if the champion changes.

Required public API (called by the evaluator):
    create_session() -> str
    send_message(session_id, message) -> SendMessageResult

Shared types must be imported from agent_base, not duplicated:
    from agent_base import (
        RetrievedContext, ChatMessage, TokenUsage, SendMessageResult,
        Session, RetrievalTraceStep, create_session, get_session,
        _add_message, _sessions,
    )
"""
