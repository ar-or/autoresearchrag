"""Single-file RAG agent — Python port of oragent.

Public API:
    result = await send_message(session_id, message)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import requests as _requests
from openai import AsyncOpenAI
from scripts.embedder import embed_batch

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
ELASTIC_TIMEOUT_S: float = float(os.environ.get("ELASTIC_TIMEOUT_S", "30"))
OPENAI_TIMEOUT_S: float = float(os.environ.get("OPENAI_TIMEOUT_S", "300"))
EMBED_MODEL: str = "text-embedding-3-small"

_openai = AsyncOpenAI(timeout=OPENAI_TIMEOUT_S, max_retries=2)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class RetrievedContext:
    lookup_id: str
    document_id: str
    text: str
    title: str
    score: float
    chunk_index: int = 0


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


@dataclass
class RetrievalState:
    last_query: str = ""
    search_results: dict[str, RetrievedContext] = field(default_factory=dict)
    read_results: list[RetrievedContext] = field(default_factory=list)
    read_lookup_ids: set[str] = field(default_factory=set)


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
        f"{ELASTIC_URL}/{index}/_search",
        headers=_es_headers(),
        json=body,
        timeout=ELASTIC_TIMEOUT_S,
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
        timeout=ELASTIC_TIMEOUT_S,
    )
    if not r.ok:
        return []
    return r.json().get("hits", {}).get("hits", [])


def _hit_identity(hit: dict[str, Any]) -> str:
    src = hit.get("_source", {})
    return (
        src.get("hash_id")
        or src.get("chunk_id")
        or hit.get("_id")
        or f"{src.get('document_id', '')}:{src.get('chunk_index', '')}:{src.get('text', '')[:64]}"
    )


def _fuse_hits_rrf(*ranked_lists: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    rrf_k = 60

    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            identity = _hit_identity(hit)
            if identity not in fused:
                fused[identity] = {"hit": hit, "score": 0.0}
            fused[identity]["score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        fused.values(),
        key=lambda item: (item["score"], item["hit"].get("_score", 0.0)),
        reverse=True,
    )
    return [item["hit"] for item in ranked[:k]]


def _hits_to_contexts(hits: list[dict[str, Any]]) -> list[RetrievedContext]:
    out: list[RetrievedContext] = []
    for hit in hits:
        src = hit.get("_source", {})
        out.append(
            RetrievedContext(
                lookup_id=_hit_identity(hit),
                document_id=src.get("document_id") or src.get("id") or hit.get("_id", ""),
                text=src.get("text") or src.get("content") or src.get("pageContent", ""),
                title=src.get("title", ""),
                score=float(hit.get("_score") or 0.0),
                chunk_index=int(src.get("chunk_index", 0) or 0),
            )
        )
    return out


def _dedupe_contexts(
    contexts: list[RetrievedContext],
    limit: int,
) -> list[RetrievedContext]:
    deduped: list[RetrievedContext] = []
    seen: set[tuple[str, int, str]] = set()
    for context in contexts:
        key = (context.document_id, context.chunk_index, context.lookup_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
        if len(deduped) >= limit:
            break
    return deduped


async def retrieve(
    query: str,
    index: str | None = None,
    k: int | None = None,
) -> list[RetrievedContext]:
    idx = index or ELASTIC_INDEX
    top_k = k or RETRIEVAL_K
    try:
        vector = (await asyncio.to_thread(embed_batch, [query]))[0]
        candidate_k = max(top_k * 2, top_k)
        vector_hits = _es_vector_search(idx, vector, candidate_k)
        text_hits = _es_text_search(idx, query, candidate_k)
        hits = _fuse_hits_rrf(vector_hits, text_hits, k=top_k)
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
        "description": "Search the knowledge base and return compact snippets plus lookup_ids. Use read_documents to open full chunks you want to inspect.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
}

_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_documents",
        "description": "Open one or more search results by lookup_id and optionally include adjacent chunks for local continuity.",
        "parameters": {
            "type": "object",
            "properties": {
                "lookup_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lookup IDs returned by search_documents.",
                },
                "include_neighbors": {
                    "type": "boolean",
                    "description": "Whether to include adjacent chunks for each selected result.",
                },
            },
            "required": ["lookup_ids"],
        },
    },
}


def _snippet(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _search_payload(contexts: list[RetrievedContext]) -> str:
    return json.dumps(
        [
            {
                "rank": i + 1,
                "lookup_id": context.lookup_id,
                "document_id": context.document_id,
                "title": context.title,
                "snippet": _snippet(context.text),
                "score": context.score,
            }
            for i, context in enumerate(contexts)
        ]
    )


def _read_payload(contexts: list[RetrievedContext]) -> str:
    return json.dumps(
        [
            {
                "rank": i + 1,
                "lookup_id": context.lookup_id,
                "document_id": context.document_id,
                "title": context.title,
                "chunk_index": context.chunk_index,
                "text": context.text,
                "score": context.score,
            }
            for i, context in enumerate(contexts)
        ]
    )


def _sort_read_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    return sorted(
        contexts,
        key=lambda context: (context.document_id, context.chunk_index, -context.score),
    )


def _es_chunk_neighbors(index: str, document_id: str, chunk_index: int) -> list[dict[str, Any]]:
    target_indices = [chunk_index - 1, chunk_index + 1]
    r = _requests.post(
        f"{ELASTIC_URL}/{index}/_search",
        headers=_es_headers(),
        json={
            "size": 2,
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"document_id": document_id}},
                        {"terms": {"chunk_index": target_indices}},
                    ]
                }
            },
            "_source": {"excludes": ["embedding"]},
            "sort": [{"chunk_index": {"order": "asc"}}],
        },
        timeout=ELASTIC_TIMEOUT_S,
    )
    if not r.ok:
        return []
    return r.json().get("hits", {}).get("hits", [])


async def _handle_tool_call(
    name: str,
    arguments: str,
    state: RetrievalState,
    index: str,
) -> tuple[str, list[RetrievedContext]]:
    args = json.loads(arguments) if arguments else {}
    if name == "search_documents":
        query = args.get("query", "")
        results = await retrieve(query, index=index)
        state.last_query = query
        state.search_results = {context.lookup_id: context for context in results}
        return _search_payload(results), results
    if name == "read_documents":
        lookup_ids = args.get("lookup_ids", []) or []
        include_neighbors = bool(args.get("include_neighbors", False))
        opened: list[RetrievedContext] = []
        already_read: list[str] = []
        for lookup_id in lookup_ids:
            if lookup_id in state.read_lookup_ids:
                already_read.append(lookup_id)
                continue
            context = state.search_results.get(lookup_id)
            if context is None:
                continue
            opened.append(context)
            state.read_lookup_ids.add(lookup_id)
            if include_neighbors:
                neighbor_hits = _es_chunk_neighbors(index, context.document_id, context.chunk_index)
                neighbors = _hits_to_contexts(neighbor_hits)
                opened.extend(neighbors)
                for neighbor in neighbors:
                    state.read_lookup_ids.add(neighbor.lookup_id)
        if not opened and already_read:
            return json.dumps(
                {
                    "status": "already_read",
                    "lookup_ids": already_read,
                    "message": "Requested chunks were already opened earlier in the conversation.",
                }
            ), state.read_results
        deduped = _sort_read_contexts(_dedupe_contexts(opened, RETRIEVAL_K * 2))
        state.read_results = _sort_read_contexts(
            _dedupe_contexts(state.read_results + deduped, RETRIEVAL_K * 2)
        )
        payload = {
            "status": "ok",
            "already_read": already_read,
            "documents": json.loads(_read_payload(deduped)),
        }
        return json.dumps(payload), state.read_results
    return "[]", []


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def _run_agent(
    user_message: str,
    history: list[ChatMessage],
) -> tuple[str, list[RetrievedContext], TokenUsage]:
    """Run the agent: retrieve, augment, call LLM with tool loop."""

    contexts = await retrieve(user_message)
    retrieval_state = RetrievalState(
        last_query=user_message,
        search_results={context.lookup_id: context for context in contexts},
        read_results=[],
    )

    # 1. Start with snippet-level discovery results instead of full chunks.
    augmented = user_message
    if contexts:
        block = "\n\n".join(
            f"[{i+1}] id={c.lookup_id} {(c.title + ': ') if c.title else ''}{_snippet(c.text)}"
            for i, c in enumerate(contexts)
        )
        augmented = (
            "Search results:\n"
            f"{block}\n\n"
            "Use exactly one tool call per turn. "
            "Use read_documents on promising lookup_ids before relying on a result. "
            "Set include_neighbors=true when you need adjacent context.\n\n"
            f"User question: {user_message}"
        )

    # 2. Build message list
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": augmented})

    # 3. Agent loop (tool calling)
    usage = TokenUsage()
    max_iterations = 10

    for _ in range(max_iterations):
        resp = await _openai.chat.completions.create(
            model=MODEL,
            messages=messages,  # type: ignore[arg-type]
            tools=[_SEARCH_TOOL, _READ_TOOL],  # type: ignore[list-item]
            parallel_tool_calls=False,
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
            tc = choice.message.tool_calls[0]
            tool_output, tool_contexts = await _handle_tool_call(
                tc.function.name,
                tc.function.arguments,
                retrieval_state,
                ELASTIC_INDEX,
            )
            if tc.function.name == "read_documents":
                contexts = list(retrieval_state.read_results)
            elif tc.function.name == "search_documents":
                contexts = list(retrieval_state.search_results.values())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output,
                }
            )
            if len(choice.message.tool_calls) > 1:
                messages.append(
                    {
                        "role": "system",
                        "content": "Only one tool call is allowed per turn in this ReAct loop. Continue with a single next action.",
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
