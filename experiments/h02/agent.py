"""Single-file RAG agent — Python port of oragent.

Public API:
    result = await send_message(session_id, message)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import requests as _requests
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Settings (mirrors oragent/src/settings.ts)
# ---------------------------------------------------------------------------

MODEL: str = os.environ.get("ORAGENT_MODEL", "gpt-5-mini")
SYSTEM_PROMPT: str = os.environ.get(
    "ORAGENT_SYSTEM_PROMPT", "You are a helpful AI assistant."
)
ELASTIC_URL: str = os.environ.get("ELASTIC_URL", "http://localhost:9200")
ELASTIC_API_KEY: str = os.environ.get("ELASTIC_API_KEY", "")
ELASTIC_INDEX: str = os.environ.get("ES_INDEX", "mtrag")
RETRIEVAL_K: int = int(os.environ.get("RETRIEVAL_K", "5"))
EMBED_MODEL: str = "text-embedding-3-small"

_openai = AsyncOpenAI()

# ---------------------------------------------------------------------------
# Types
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
# Elasticsearch retrieval
# ---------------------------------------------------------------------------


def _es_headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    if ELASTIC_API_KEY:
        h["Authorization"] = f"ApiKey {ELASTIC_API_KEY}"
    return h


def _es_vector_search(
    index: str, vector: list[float], k: int
) -> list[dict[str, Any]]:
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": vector},
                },
            },
        },
        "_source": {"excludes": ["embedding"]},
    }
    r = _requests.post(
        f"{ELASTIC_URL}/{index}/_search", headers=_es_headers(), json=body
    )
    if not r.ok:
        return _es_text_search(index, "", k)
    return r.json().get("hits", {}).get("hits", [])


def _es_text_search(
    index: str, query: str, k: int
) -> list[dict[str, Any]]:
    q: dict[str, Any] = (
        {"multi_match": {"query": query, "fields": ["text", "title", "content"]}}
        if query
        else {"match_all": {}}
    )
    r = _requests.post(
        f"{ELASTIC_URL}/{index}/_search",
        headers=_es_headers(),
        json={"size": k, "query": q},
    )
    if not r.ok:
        return []
    return r.json().get("hits", {}).get("hits", [])


def _hits_to_contexts(hits: list[dict[str, Any]]) -> list[RetrievedContext]:
    out: list[RetrievedContext] = []
    for hit in hits:
        src = hit.get("_source", {})
        out.append(
            RetrievedContext(
                document_id=src.get("document_id") or src.get("id") or hit.get("_id", ""),
                text=src.get("text") or src.get("content") or src.get("pageContent", ""),
                title=src.get("title", ""),
                score=hit.get("_score", 0),
            )
        )
    return out


def _diversify_hits(
    hits: list[dict[str, Any]], k: int, max_per_document: int = 2
) -> list[dict[str, Any]]:
    """Limit over-concentration from the same source document."""
    selected: list[dict[str, Any]] = []
    per_document: dict[str, int] = {}
    overflow: list[dict[str, Any]] = []

    for hit in hits:
        src = hit.get("_source", {})
        doc_id = src.get("document_id") or hit.get("_id", "")
        if per_document.get(doc_id, 0) < max_per_document:
            selected.append(hit)
            per_document[doc_id] = per_document.get(doc_id, 0) + 1
        else:
            overflow.append(hit)
        if len(selected) == k:
            return selected

    for hit in overflow:
        selected.append(hit)
        if len(selected) == k:
            break

    return selected


async def retrieve(
    query: str,
    index: str | None = None,
    k: int | None = None,
) -> list[RetrievedContext]:
    idx = index or ELASTIC_INDEX
    top_k = k or RETRIEVAL_K
    candidate_k = max(top_k * 3, top_k)
    try:
        resp = await _openai.embeddings.create(input=query, model=EMBED_MODEL)
        vector = resp.data[0].embedding
        hits = _es_vector_search(idx, vector, candidate_k)
        hits = _diversify_hits(hits, top_k)
    except Exception:
        try:
            hits = _es_text_search(idx, query, top_k)
        except Exception:
            return []
    return _hits_to_contexts(hits)


# ---------------------------------------------------------------------------
# Tool definition (mirrors createRetrievalTool)
# ---------------------------------------------------------------------------

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the knowledge base for relevant passages. Input should be a search query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}


async def _handle_tool_call(name: str, arguments: str) -> tuple[str, list[RetrievedContext]]:
    if name == "search_documents":
        args = json.loads(arguments)
        query = args.get("query", "")
        results = await retrieve(query)
        payload = [
            {
                "rank": i + 1,
                "document_id": r.document_id,
                "title": r.title,
                "text": r.text,
                "score": r.score,
            }
            for i, r in enumerate(results)
        ]
        return json.dumps(payload), results
    return "[]", []


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def _run_agent(
    user_message: str,
    history: list[ChatMessage],
) -> tuple[str, list[RetrievedContext], TokenUsage]:
    """Run the agent: retrieve, augment, call LLM with tool loop."""

    # 1. Pre-retrieval (always retrieve before agent runs)
    contexts = await retrieve(user_message)

    # 2. Augment user message with retrieved context
    augmented = user_message
    if contexts:
        block = "\n\n".join(
            f"[{i+1}] {(c.title + ': ') if c.title else ''}{c.text}"
            for i, c in enumerate(contexts)
        )
        augmented = f"Retrieved documents:\n{block}\n\nUser question: {user_message}"

    # 3. Build message list
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": augmented})

    # 4. Agent loop (tool calling)
    usage = TokenUsage()
    max_iterations = 10

    for _ in range(max_iterations):
        resp = await _openai.chat.completions.create(
            model=MODEL,
            messages=messages,  # type: ignore[arg-type]
            tools=[_SEARCH_TOOL],  # type: ignore[list-item]
        )
        choice = resp.choices[0]

        # Accumulate tokens
        if resp.usage:
            usage.input_tokens += resp.usage.prompt_tokens
            usage.output_tokens += resp.usage.completion_tokens
            if hasattr(resp.usage, "prompt_tokens_details") and resp.usage.prompt_tokens_details:
                usage.cached_tokens += getattr(
                    resp.usage.prompt_tokens_details, "cached_tokens", 0
                ) or 0

        # If the model wants to call tools, execute them and continue
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            messages.append(choice.message.model_dump())  # type: ignore[arg-type]
            for tc in choice.message.tool_calls:
                tool_output, tool_contexts = await _handle_tool_call(
                    tc.function.name, tc.function.arguments
                )
                contexts.extend(tool_contexts)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )
            continue

        # Done — extract content
        content = choice.message.content or ""
        return content, contexts, usage

    # Fell through max iterations — return last content
    return "", contexts, usage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def send_message(
    session_id: str,
    message: str,
) -> SendMessageResult:
    """Send a message and get an agent response.

    Args:
        session_id: Session identifier (must exist via create_session()).
        message: The user's new message.

    Returns:
        SendMessageResult with response text, retrieved contexts, model name,
        and token usage — same shape as oragent's POST /api/chat/send-message.
    """
    from datetime import datetime, timezone

    session = _sessions.get(session_id)
    if session is None:
        raise ValueError(f"Session {session_id} not found")

    now = datetime.now(timezone.utc).isoformat()
    _add_message(session_id, ChatMessage(role="user", content=message, timestamp=now))

    # Pass history *before* the new user message to the agent
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
