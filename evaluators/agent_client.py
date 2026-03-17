"""Polymorphic agent client — local (agent.py) or HTTP (oragent API).

Set AGENT_MODE=http to use the HTTP API, otherwise defaults to local.
Set ORAGENT_URL to configure the HTTP endpoint.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is importable for the local agent
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
LOCAL_AGENT_TIMEOUT_S = float(os.environ.get("LOCAL_AGENT_TIMEOUT_S", "900"))


# ---------------------------------------------------------------------------
# Shared response types
# ---------------------------------------------------------------------------


@dataclass
class Context:
    document_id: str
    text: str
    title: str
    score: float


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentResponse:
    session_id: str
    response: str
    contexts: list[Context]
    model: str
    usage: Usage


# ---------------------------------------------------------------------------
# Abstract client
# ---------------------------------------------------------------------------


class AgentClient(ABC):
    @abstractmethod
    def create_session(self) -> str: ...

    @abstractmethod
    def send_message(self, session_id: str, message: str) -> AgentResponse: ...

    @abstractmethod
    def delete_session(self, session_id: str) -> None: ...


# ---------------------------------------------------------------------------
# HTTP client (calls oragent REST API)
# ---------------------------------------------------------------------------


class HttpAgentClient(AgentClient):
    def __init__(self, base_url: str | None = None):
        self._base_url = base_url or os.environ.get("ORAGENT_URL", "http://localhost:32522")

    def create_session(self) -> str:
        import requests

        r = requests.post(f"{self._base_url}/api/chat/session", timeout=10)
        r.raise_for_status()
        return r.json()["session_id"]

    def send_message(self, session_id: str, message: str) -> AgentResponse:
        import requests

        r = requests.post(
            f"{self._base_url}/api/chat/send-message",
            json={"session_id": session_id, "message": message},
            timeout=180,
        )
        r.raise_for_status()
        data = r.json()
        usage_raw = data.get("usage", {})
        contexts_raw = data.get("contexts", [])
        return AgentResponse(
            session_id=session_id,
            response=data.get("response", ""),
            contexts=[
                Context(
                    document_id=c.get("document_id", ""),
                    text=c.get("text", ""),
                    title=c.get("title", ""),
                    score=c.get("score", 0),
                )
                for c in contexts_raw
            ],
            model=data.get("model", "unknown"),
            usage=Usage(
                input_tokens=usage_raw.get("input_tokens", 0),
                cached_tokens=usage_raw.get("cached_tokens", 0),
                output_tokens=usage_raw.get("output_tokens", 0),
            ),
        )

    def delete_session(self, session_id: str) -> None:
        import requests

        try:
            requests.delete(f"{self._base_url}/api/chat/session/{session_id}", timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Local client (imports agent.py directly, runs in-process)
# ---------------------------------------------------------------------------


class LocalAgentClient(AgentClient):
    def __init__(self) -> None:
        import importlib

        module_name = os.environ.get("AGENT_MODULE", "agent")
        self._agent = importlib.import_module(module_name)

    def create_session(self) -> str:
        return self._agent.create_session()

    def send_message(self, session_id: str, message: str) -> AgentResponse:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    asyncio.run,
                    asyncio.wait_for(
                        self._agent.send_message(session_id, message),
                        timeout=LOCAL_AGENT_TIMEOUT_S,
                    ),
                ).result(timeout=LOCAL_AGENT_TIMEOUT_S + 5)
        else:
            result = asyncio.run(
                asyncio.wait_for(
                    self._agent.send_message(session_id, message),
                    timeout=LOCAL_AGENT_TIMEOUT_S,
                )
            )

        return AgentResponse(
            session_id=result.session_id,
            response=result.response,
            contexts=[
                Context(
                    document_id=c.document_id,
                    text=c.text,
                    title=c.title,
                    score=c.score,
                )
                for c in result.contexts
            ],
            model=result.model,
            usage=Usage(
                input_tokens=result.usage.input_tokens,
                cached_tokens=result.usage.cached_tokens,
                output_tokens=result.usage.output_tokens,
            ),
        )

    def delete_session(self, session_id: str) -> None:
        pass  # local sessions are in-memory, no cleanup needed


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_client() -> AgentClient:
    mode = os.environ.get("AGENT_MODE", "local").lower()
    if mode == "http":
        return HttpAgentClient()
    return LocalAgentClient()
