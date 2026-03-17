"""Agentic file-search RAG agent (bash-only tool).

Instead of Elasticsearch retrieval, this agent gives the LLM a single
bash tool to explore the ``data_as_files/`` folder and find answers
using standard shell commands (ls, cat, grep, head, find, etc.).

Public API (same shape as agent.py):
    result = await send_message(session_id, message)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODEL: str = os.environ.get("ORAGENT_MODEL", "gpt-5-mini")
OPENAI_TIMEOUT_S: float = float(os.environ.get("OPENAI_TIMEOUT_S", "300"))
DATA_ROOT: Path = Path(__file__).resolve().parent / "data_as_files"
BASH_TIMEOUT_S: int = 30  # max seconds per bash invocation

_openai = AsyncOpenAI(timeout=OPENAI_TIMEOUT_S, max_retries=2)

# ---------------------------------------------------------------------------
# Types (same as agent.py for evaluator compatibility)
# ---------------------------------------------------------------------------


@dataclass
class RetrievedContext:
    document_id: str
    text: str
    title: str
    score: float


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = ""
    contexts: list[RetrievedContext] = field(default_factory=list)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SendMessageResult:
    session_id: str
    response: str
    contexts: list[RetrievedContext]
    model: str
    usage: TokenUsage


@dataclass
class Session:
    id: str
    messages: list[ChatMessage]
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions: dict[str, Session] = {}


def create_session() -> str:
    from datetime import datetime, timezone

    sid = os.urandom(16).hex()
    now = datetime.now(timezone.utc).isoformat()
    _sessions[sid] = Session(id=sid, messages=[], created_at=now, updated_at=now)
    return sid


def get_session(session_id: str) -> Session | None:
    return _sessions.get(session_id)


def _add_message(session_id: str, msg: ChatMessage) -> None:
    from datetime import datetime, timezone

    session = _sessions[session_id]
    session.messages.append(msg)
    session.updated_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Bash tool implementation
# ---------------------------------------------------------------------------

_MAX_OUTPUT_CHARS = 12_000


def _run_bash(command: str) -> str:
    """Execute a bash command inside DATA_ROOT with a timeout and output cap."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(DATA_ROOT),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_S,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + f"\n... [truncated at {_MAX_OUTPUT_CHARS} chars]"
        if not output.strip():
            output = "(no output)"
        return output
    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {BASH_TIMEOUT_S}s"
    except Exception as exc:
        return f"[error] {exc}"


# ---------------------------------------------------------------------------
# Tool definition for the LLM
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash command. The working directory is the data folder "
                "containing all document files. Use standard commands like ls, cat, rg, "
                "grep, head, find, wc, etc. to explore and search files. Always quote file names!"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are a helpful AI assistant that answers questions by searching through a collection of documents stored as files.
You have bash access.

Answer the question directly and concisely based on the documents you find. If the question involves calculations, show your work. If you cannot find relevant documents, say so.\
"""


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def _run_agent(
    user_message: str,
    history: list[ChatMessage],
) -> tuple[str, list[RetrievedContext], TokenUsage]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_message})

    usage = TokenUsage()
    contexts: list[RetrievedContext] = []
    max_iterations = 15

    for _ in range(max_iterations):
        resp = await _openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=_TOOLS,
        )
        choice = resp.choices[0]

        if resp.usage:
            usage.input_tokens += resp.usage.prompt_tokens
            usage.output_tokens += resp.usage.completion_tokens
            if hasattr(resp.usage, "prompt_tokens_details") and resp.usage.prompt_tokens_details:
                usage.cached_tokens += getattr(
                    resp.usage.prompt_tokens_details, "cached_tokens", 0
                ) or 0

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message.model_dump())  # type: ignore[arg-type]
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                command = args.get("command", "")
                print(f"    [bash] {command}")
                tool_output = await asyncio.to_thread(_run_bash, command)
                preview = tool_output[:200].replace("\n", "\\n")
                print(f"    [out]  {preview}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                })
                # Track cat/read commands as retrieved contexts
                if any(cmd in command for cmd in ("cat ", "head ", "tail ")):
                    # Extract a snippet for context tracking
                    snippet = tool_output[:500] if tool_output else ""
                    if snippet and not snippet.startswith("[error]"):
                        contexts.append(RetrievedContext(
                            document_id=command,
                            text=snippet,
                            title=command,
                            score=1.0,
                        ))
            continue

        content = choice.message.content or ""
        print(f"    [answer] {content[:200]}")
        return content, contexts, usage

    print("    [answer] (max iterations reached)")
    return "", contexts, usage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def send_message(
    session_id: str,
    message: str,
) -> SendMessageResult:
    from datetime import datetime, timezone

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    _add_message(session_id, ChatMessage(role="user", content=message, timestamp=now))

    history = session.messages[:-1]
    content, contexts, usage = await _run_agent(message, history)

    now = datetime.now(timezone.utc).isoformat()
    _add_message(
        session_id,
        ChatMessage(role="assistant", content=content, timestamp=now, contexts=contexts),
    )

    return SendMessageResult(
        session_id=session_id,
        response=content,
        contexts=contexts,
        model=MODEL,
        usage=usage,
    )
