"""Single-file RAG agent — Python port of oragent.

Public API:
    result = await send_message(session_id, message)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
                document_id=src.get("document_id") or src.get("id") or hit.get("_id", ""),
                text=src.get("text") or src.get("content") or src.get("pageContent", ""),
                title=src.get("title", ""),
                score=hit.get("_score", 0),
            )
        )
    return out


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
# Planner / Reasoner / Executor helpers
# ---------------------------------------------------------------------------

def _context_block(contexts: list[RetrievedContext], limit: int | None = None) -> str:
    selected = contexts if limit is None else contexts[:limit]
    return "\n\n".join(
        f"[{i+1}] {(context.title + ': ') if context.title else ''}{context.text}"
        for i, context in enumerate(selected)
    )


def _history_block(history: list[ChatMessage], limit: int = 4) -> str:
    recent = history[-limit:]
    if not recent:
        return "(none)"
    return "\n".join(f"{message.role}: {message.content}" for message in recent)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _dedupe_contexts(
    contexts: list[RetrievedContext], limit: int | None = None
) -> list[RetrievedContext]:
    deduped: list[RetrievedContext] = []
    seen: set[tuple[str, str]] = set()
    for context in contexts:
        key = (context.document_id, context.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
        if limit is not None and len(deduped) >= limit:
            break
    return deduped


async def _complete_json(messages: list[dict[str, Any]], usage: TokenUsage) -> dict[str, Any]:
    resp = await _openai.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
        response_format={"type": "json_object"},
    )
    if resp.usage:
        usage.input_tokens += resp.usage.prompt_tokens
        usage.output_tokens += resp.usage.completion_tokens
        if hasattr(resp.usage, "prompt_tokens_details") and resp.usage.prompt_tokens_details:
            usage.cached_tokens += getattr(
                resp.usage.prompt_tokens_details, "cached_tokens", 0
            ) or 0
    return _parse_json_object(resp.choices[0].message.content or "{}")


async def _complete_text(messages: list[dict[str, Any]], usage: TokenUsage) -> str:
    resp = await _openai.chat.completions.create(
        model=MODEL,
        messages=messages,  # type: ignore[arg-type]
    )
    if resp.usage:
        usage.input_tokens += resp.usage.prompt_tokens
        usage.output_tokens += resp.usage.completion_tokens
        if hasattr(resp.usage, "prompt_tokens_details") and resp.usage.prompt_tokens_details:
            usage.cached_tokens += getattr(
                resp.usage.prompt_tokens_details, "cached_tokens", 0
            ) or 0
    return resp.choices[0].message.content or ""


def _evidence_block(evidence_pool: list[RetrievedContext]) -> str:
    if not evidence_pool:
        return "(none)"
    lines = []
    for idx, context in enumerate(evidence_pool, start=1):
        title = f"{context.title}: " if context.title else ""
        lines.append(f"[{idx}] {title}{context.text}")
    return "\n\n".join(lines)


async def _execute_search_action(query: str) -> list[RetrievedContext]:
    return await retrieve(query)


async def _run_planner_reasoner_executor(
    user_message: str,
    history: list[ChatMessage],
) -> tuple[str, list[RetrievedContext], TokenUsage]:
    usage = TokenUsage()
    planner = await _complete_json(
        [
            {
                "role": "system",
                "content": (
                    "You are the planner in a planner-reasoner-executor retrieval QA system. "
                    "Decompose the question into a compact plan before any action is executed. "
                    "Return JSON with keys: goal, subquestions, initial_queries, answer_requirements. "
                    "Use 1-3 focused initial_queries."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{_history_block(history)}\n\n"
                    f"Question:\n{user_message}"
                ),
            },
        ],
        usage,
    )

    evidence_pool: list[RetrievedContext] = []
    executed_queries: list[str] = []
    action_trace: list[dict[str, Any]] = []

    initial_queries = [
        query.strip()
        for query in planner.get("initial_queries", [])
        if isinstance(query, str) and query.strip()
    ]
    for query in initial_queries[:2]:
        results = await _execute_search_action(query)
        evidence_pool.extend(results)
        executed_queries.append(query)
        action_trace.append(
            {
                "step": len(action_trace) + 1,
                "action": "search",
                "query": query,
                "retrieved": min(len(results), RETRIEVAL_K),
            }
        )
    evidence_pool = _dedupe_contexts(evidence_pool, limit=max(RETRIEVAL_K * 2, RETRIEVAL_K))

    for step in range(3):
        reasoner = await _complete_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the reasoner in a planner-reasoner-executor retrieval QA system. "
                        "Inspect the plan, prior actions, and current evidence. "
                        "Choose the next executable action. "
                        "Return JSON with keys: action, query, evidence_ids, stop_reason. "
                        "action must be one of search or finalize. "
                        "Use search only when a focused missing fact remains."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{user_message}\n\n"
                        f"Plan:\n{json.dumps(planner)}\n\n"
                        f"Executed queries:\n{json.dumps(executed_queries)}\n\n"
                        f"Action trace:\n{json.dumps(action_trace)}\n\n"
                        f"Evidence pool:\n{_evidence_block(evidence_pool)}"
                    ),
                },
            ],
            usage,
        )

        action = str(reasoner.get("action", "finalize")).strip().lower()
        if action != "search":
            break

        query = str(reasoner.get("query", "")).strip()
        if not query or query in executed_queries:
            break

        results = await _execute_search_action(query)
        evidence_pool.extend(results)
        evidence_pool = _dedupe_contexts(
            evidence_pool,
            limit=max(RETRIEVAL_K * 2 + 2, RETRIEVAL_K),
        )
        executed_queries.append(query)
        action_trace.append(
            {
                "step": len(action_trace) + 1,
                "action": "search",
                "query": query,
                "retrieved": min(len(results), RETRIEVAL_K),
                "reasoner": reasoner,
            }
        )

    final_selection = await _complete_json(
        [
            {
                "role": "system",
                "content": (
                    "You are the final reasoner in a planner-reasoner-executor retrieval QA system. "
                    "Select the evidence that should be handed to the executor for answering. "
                    "Return JSON with keys: evidence_ids and answer_brief."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{user_message}\n\n"
                    f"Plan:\n{json.dumps(planner)}\n\n"
                    f"Action trace:\n{json.dumps(action_trace)}\n\n"
                    f"Evidence pool:\n{_evidence_block(evidence_pool)}"
                ),
            },
        ],
        usage,
    )
    selected_ids = [
        item for item in final_selection.get("evidence_ids", []) if isinstance(item, int)
    ]
    selected_contexts = [
        evidence_pool[item - 1]
        for item in selected_ids
        if 1 <= item <= len(evidence_pool)
    ]
    if not selected_contexts:
        selected_contexts = evidence_pool[:RETRIEVAL_K]

    final_answer = await _complete_text(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[
                {"role": message.role, "content": message.content}
                for message in history
            ],
            {
                "role": "user",
                "content": (
                    "Planner output:\n"
                    f"{json.dumps(planner)}\n\n"
                    "Executor action trace:\n"
                    f"{json.dumps(action_trace)}\n\n"
                    "Selected evidence:\n"
                    f"{_context_block(selected_contexts)}\n\n"
                    f"User question: {user_message}"
                ),
            },
        ],
        usage,
    )
    return final_answer, selected_contexts, usage


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


async def _run_agent(
    user_message: str,
    history: list[ChatMessage],
) -> tuple[str, list[RetrievedContext], TokenUsage]:
    """Run the H31 planner-reasoner-executor workflow."""
    return await _run_planner_reasoner_executor(user_message, history)


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
