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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from openai import AsyncOpenAI

from agent_base import (
    RetrievedContext,
    ChatMessage,
    TokenUsage,
    SendMessageResult,
    Session,
    create_session,
    get_session,
    _add_message,
    _sessions,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODEL: str = os.environ.get("ORAGENT_MODEL", "gpt-5-mini")
OPENAI_TIMEOUT_S: float = float(os.environ.get("OPENAI_TIMEOUT_S", "300"))
_MNT_DATA = Path("/mnt/data")
DATA_ROOT: Path = _MNT_DATA if _MNT_DATA.is_dir() else _PROJECT_ROOT / "data_as_files"
BASH_TIMEOUT_S: int = 30  # max seconds per bash invocation

_openai = AsyncOpenAI(timeout=OPENAI_TIMEOUT_S, max_retries=2)


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
You have bash access. The working directory is {DATA_ROOT}.
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

    for iteration in range(max_iterations):
        # On the last iteration, force the model to answer instead of calling tools
        if iteration == max_iterations - 1:
            messages.append({
                "role": "system",
                "content": "Tool call limit reached. You MUST now provide your final answer based on what you have found so far. Do NOT call any more tools.",
            })

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
                if any(cmd in command for cmd in ("cat ", "head ", "tail ", "sed")):
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
