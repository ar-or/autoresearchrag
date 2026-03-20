"""Single-file RAG agent — Python port of oragent.

Public API:
    result = await send_message(session_id, message)
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import requests as _requests
from openai import AsyncOpenAI
from scripts.embedder import embed_batch

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

# ---------------------------------------------------------------------------
# Settings (mirrors oragent/src/settings.ts)
# ---------------------------------------------------------------------------

MODEL: str = os.environ.get("ORAGENT_MODEL", "gpt-5-mini")
SYSTEM_PROMPT: str = os.environ.get(
    "ORAGENT_SYSTEM_PROMPT", "You are a helpful AI assistant."
)
ELASTIC_URL: str = os.environ.get("ES_HOST", "http://localhost:9200")
ELASTIC_API_KEY: str = os.environ.get("ELASTIC_API_KEY", "")
ELASTIC_INDEX: str = os.environ.get("ES_INDEX", "mtrag")
RETRIEVAL_K: int = int(os.environ.get("RETRIEVAL_K", "5"))
ELASTIC_TIMEOUT_S: float = float(os.environ.get("ELASTIC_TIMEOUT_S", "30"))
OPENAI_TIMEOUT_S: float = float(os.environ.get("OPENAI_TIMEOUT_S", "300"))
EMBED_MODEL: str = "text-embedding-3-small"

_openai = AsyncOpenAI(timeout=OPENAI_TIMEOUT_S, max_retries=2)


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


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
}


def _query_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }


def _context_signature(context: RetrievedContext) -> tuple[str, str, str]:
    return (context.document_id, context.title, context.text[:160])


def _context_overlap(query_terms: set[str], context: RetrievedContext) -> int:
    haystack = f"{context.title} {context.text}".lower()
    return sum(1 for term in query_terms if term in haystack)


def _query_complexity_bonus(query: str) -> float:
    lowered = query.lower()
    cues = (
        "both",
        "before",
        "after",
        "which other",
        "same",
        "different",
        "compare",
    )
    return 0.15 if any(cue in lowered for cue in cues) else 0.0


def _trace_step(
    query: str,
    contexts: list[RetrievedContext],
    latest_batch: list[RetrievedContext],
    previous_coverage: float,
) -> RetrievalTraceStep:
    query_terms = _query_terms(query)
    if not query_terms or not contexts:
        return RetrievalTraceStep(
            coverage=0.0,
            supportive_contexts=0,
            novelty_ratio=0.0,
            gain=0.0,
        )

    matched_terms: set[str] = set()
    supportive_contexts = 0
    for context in contexts:
        overlap = _context_overlap(query_terms, context)
        if overlap >= 2:
            supportive_contexts += 1
        haystack = f"{context.title} {context.text}".lower()
        matched_terms.update(term for term in query_terms if term in haystack)

    coverage = len(matched_terms) / max(len(query_terms), 1)
    previous_signatures = {
        _context_signature(context)
        for context in contexts[:-len(latest_batch)]
    } if latest_batch else set()
    novel = sum(
        1 for context in latest_batch if _context_signature(context) not in previous_signatures
    )
    novelty_ratio = novel / len(latest_batch) if latest_batch else 0.0
    return RetrievalTraceStep(
        coverage=coverage,
        supportive_contexts=supportive_contexts,
        novelty_ratio=novelty_ratio,
        gain=max(coverage - previous_coverage, 0.0),
    )


def _should_continue_with_q_lambda_proxy(
    query: str,
    trace: list[RetrievalTraceStep],
) -> bool:
    if not trace:
        return True
    search_round = len(trace) - 1
    if search_round >= 2:
        return False

    current = trace[-1]
    complexity_bonus = _query_complexity_bonus(query)

    continue_value = (
        0.55 * (1.0 - current.coverage)
        + 0.25 * current.gain
        + 0.20 * current.novelty_ratio
        + complexity_bonus
        - 0.15 * search_round
    )
    stop_value = (
        0.60 * current.coverage
        + 0.20 * min(current.supportive_contexts / 2.0, 1.0)
        + 0.20 * (1.0 - current.novelty_ratio if search_round > 0 else 0.0)
    )
    return continue_value > stop_value


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if not a_norm or not b_norm:
        return 0.0
    return dot / (a_norm * b_norm)


async def _sentence_rerank_contexts(
    query: str,
    contexts: list[RetrievedContext],
    top_k: int,
) -> list[RetrievedContext]:
    if not contexts:
        return contexts

    query_vector = (await asyncio.to_thread(embed_batch, [query]))[0]
    sentence_texts: list[str] = []
    sentence_to_context: list[int] = []
    for index, context in enumerate(contexts):
        for sentence in _split_sentences(context.text)[:6]:
            sentence_texts.append(sentence)
            sentence_to_context.append(index)
    if not sentence_texts:
        return contexts[:top_k]

    sentence_vectors = await asyncio.to_thread(embed_batch, sentence_texts)
    best_scores = [-1.0] * len(contexts)
    for idx, vector in enumerate(sentence_vectors):
        context_index = sentence_to_context[idx]
        best_scores[context_index] = max(
            best_scores[context_index],
            _cosine_similarity(query_vector, vector),
        )

    ranked = sorted(
        enumerate(contexts),
        key=lambda item: (best_scores[item[0]], item[1].score),
        reverse=True,
    )
    return [context for _, context in ranked[:top_k]]


def _use_rich_retrieval(query: str) -> bool:
    lowered = query.lower()
    cues = (
        "both",
        "compare",
        "which other",
        "same",
        "different",
        "before",
        "after",
        "ratio",
    )
    return len(query.split()) >= 12 or any(cue in lowered for cue in cues)


async def _rewrite_retrieval_query(query: str) -> str:
    prompt = (
        "Rewrite the question into a short retrieval query that preserves the original meaning, "
        "keeps key entities, dates, and numbers, and removes filler words. "
        "Return only the rewritten query.\n\n"
        f"Question: {query}"
    )
    try:
        resp = await _openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You rewrite questions for document retrieval."},
                {"role": "user", "content": prompt},
            ],
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or query
    except Exception:
        return query


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


async def _reflection_action(
    question: str,
    contexts: list[RetrievedContext],
) -> dict[str, str]:
    block = "\n\n".join(
        f"[{index+1}] {(context.title + ': ') if context.title else ''}{context.text[:280]}"
        for index, context in enumerate(contexts[:5])
    )
    prompt = (
        "Decide the next retrieval action for this question.\n"
        "Return JSON with keys action and query.\n"
        "Allowed actions: retrieve_more, use_current_evidence, answer_now.\n"
        "Use retrieve_more only if the evidence looks incomplete and provide one improved retrieval query.\n\n"
        f"Question: {question}\n\nEvidence:\n{block}"
    )
    try:
        resp = await _openai.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a retrieval controller."},
                {"role": "user", "content": prompt},
            ],
        )
        return _parse_json_object(resp.choices[0].message.content or "")
    except Exception:
        return {"action": "use_current_evidence", "query": question}


async def retrieve(
    query: str,
    index: str | None = None,
    k: int | None = None,
) -> list[RetrievedContext]:
    idx = index or ELASTIC_INDEX
    top_k = k or RETRIEVAL_K
    retrieval_query = query
    candidate_k = max(top_k * 2, top_k)
    if _use_rich_retrieval(query):
        retrieval_query = await _rewrite_retrieval_query(query)
        candidate_k = max(top_k * 3, top_k)
    try:
        vector = (await asyncio.to_thread(embed_batch, [retrieval_query]))[0]
        vector_hits = _es_vector_search(idx, vector, candidate_k)
        text_hits = _es_text_search(idx, retrieval_query, candidate_k)
        hits = _fuse_hits_rrf(vector_hits, text_hits, k=top_k)
    except Exception:
        try:
            hits = _es_text_search(idx, retrieval_query, top_k)
        except Exception:
            return []
    contexts = _hits_to_contexts(hits)
    try:
        return await _sentence_rerank_contexts(retrieval_query, contexts, top_k)
    except Exception:
        return contexts[:top_k]


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
    print(f"    [search] {user_message[:200]}")
    contexts = await retrieve(user_message)
    decision = await _reflection_action(user_message, contexts)
    if decision.get("action") == "retrieve_more":
        extra_query = (decision.get("query") or user_message).strip()
        extra_contexts = await retrieve(extra_query)
        seen = {(context.document_id, context.text) for context in contexts}
        for context in extra_contexts:
            key = (context.document_id, context.text)
            if key not in seen:
                seen.add(key)
                contexts.append(context)
    print(f"    [results] {len(contexts)} hits: {[c.title for c in contexts]}")

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
    trace = [_trace_step(user_message, contexts, contexts, 0.0)]
    allow_search = _should_continue_with_q_lambda_proxy(user_message, trace)

    for _ in range(max_iterations):
        request_kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
        }
        if allow_search:
            request_kwargs["tools"] = [_SEARCH_TOOL]
        resp = await _openai.chat.completions.create(
            **request_kwargs,  # type: ignore[arg-type]
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
            latest_batch: list[RetrievedContext] = []
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"    [tool_call] {tc.function.name}({tc.function.arguments})")
                tool_output, tool_contexts = await _handle_tool_call(
                    tc.function.name, tc.function.arguments
                )
                print(f"    [results] {len(tool_contexts)} hits: {[c.title for c in tool_contexts]}")
                contexts.extend(tool_contexts)
                latest_batch.extend(tool_contexts)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )
            trace.append(
                _trace_step(
                    user_message,
                    contexts,
                    latest_batch,
                    trace[-1].coverage,
                )
            )
            allow_search = _should_continue_with_q_lambda_proxy(user_message, trace)
            continue

        # Done — extract content
        content = choice.message.content or ""
        print(f"    [answer] {content[:200]}")
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
