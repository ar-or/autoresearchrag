"""Shared graph-retrieval agent factory for H42/H44/H45/H46."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import networkx as nx
import requests as _requests
from openai import AsyncOpenAI

from agent_base import (
    ChatMessage,
    RetrievedContext,
    SendMessageResult,
    TokenUsage,
    _add_message,
    _sessions,
    create_session,
)
from experiments.graph_layer import (
    GraphSnapshot,
    graph_query_terms,
    load_graph_snapshot,
    match_query_entities,
    pagerank_scores,
)
from scripts.embedder import embed_batch

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

_openai = AsyncOpenAI(timeout=OPENAI_TIMEOUT_S, max_retries=2)


def _es_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if ELASTIC_API_KEY:
        headers["Authorization"] = f"ApiKey {ELASTIC_API_KEY}"
    return headers


def _es_text_search(index: str, query: str, k: int) -> list[dict[str, Any]]:
    response = _requests.post(
        f"{ELASTIC_URL}/{index}/_search",
        headers=_es_headers(),
        json={
            "size": k,
            "query": {"multi_match": {"query": query, "fields": ["text", "title", "content"]}},
            "_source": {"excludes": ["embedding"]},
        },
        timeout=ELASTIC_TIMEOUT_S,
    )
    if not response.ok:
        return []
    return response.json().get("hits", {}).get("hits", [])


def _es_vector_search(index: str, vector: list[float], k: int) -> list[dict[str, Any]]:
    response = _requests.post(
        f"{ELASTIC_URL}/{index}/_search",
        headers=_es_headers(),
        json={
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": vector},
                    },
                }
            },
            "_source": {"excludes": ["embedding"]},
        },
        timeout=ELASTIC_TIMEOUT_S,
    )
    if not response.ok:
        return []
    return response.json().get("hits", {}).get("hits", [])


def _hit_identity(hit: dict[str, Any]) -> str:
    source = hit.get("_source", {})
    return (
        source.get("hash_id")
        or source.get("chunk_id")
        or hit.get("_id")
        or f"{source.get('document_id', '')}:{source.get('title', '')}"
    )


def _fuse_hits_rrf(*ranked_lists: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            identity = _hit_identity(hit)
            fused.setdefault(identity, {"score": 0.0, "hit": hit})
            fused[identity]["score"] += 1.0 / (60 + rank)
    ranked = sorted(
        fused.values(),
        key=lambda item: (item["score"], item["hit"].get("_score", 0.0)),
        reverse=True,
    )
    return [item["hit"] for item in ranked[:k]]


async def _retrieve_es_contexts(query: str, top_k: int) -> list[RetrievedContext]:
    try:
        vector = (await asyncio.to_thread(embed_batch, [query]))[0]
        vector_hits = _es_vector_search(ELASTIC_INDEX, vector, max(top_k * 2, top_k))
        text_hits = _es_text_search(ELASTIC_INDEX, query, max(top_k * 2, top_k))
        hits = _fuse_hits_rrf(vector_hits, text_hits, k=top_k)
    except Exception:
        hits = _es_text_search(ELASTIC_INDEX, query, top_k)

    contexts: list[RetrievedContext] = []
    for hit in hits:
        source = hit.get("_source", {})
        contexts.append(
            RetrievedContext(
                document_id=source.get("document_id") or source.get("id") or hit.get("_id", ""),
                title=source.get("title", ""),
                text=source.get("text") or source.get("content") or source.get("pageContent", ""),
                score=float(hit.get("_score", 0.0)),
            )
        )
    return contexts


def _dedupe_contexts(contexts: list[RetrievedContext], limit: int) -> list[RetrievedContext]:
    deduped: list[RetrievedContext] = []
    seen: set[tuple[str, str, str]] = set()
    for context in contexts:
        key = (context.document_id, context.title, context.text[:120])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(context)
        if len(deduped) >= limit:
            break
    return deduped


def _context_from_graph_doc(
    document_id: str,
    title: str,
    text: str,
    score: float,
    label: str,
) -> RetrievedContext:
    compact = " ".join(text.split())
    if len(compact) > 900:
        compact = compact[:897] + "..."
    return RetrievedContext(
        document_id=document_id,
        title=f"{label}: {title}",
        text=compact,
        score=score,
    )


def _community_contexts(
    snapshot: GraphSnapshot,
    matched_entities: list[str],
    limit: int,
    boost: float = 1.0,
) -> list[RetrievedContext]:
    scored: list[tuple[float, str]] = []
    matched_set = set(matched_entities)
    for community_id, community in snapshot.communities.items():
        overlap = len(snapshot.community_entities.get(community_id, set()) & matched_set)
        if not overlap:
            continue
        score = boost * (overlap + 0.1 * float(community.get("size", 0)))
        scored.append((score, community_id))
    scored.sort(reverse=True)
    contexts: list[RetrievedContext] = []
    for score, community_id in scored[:limit]:
        contexts.append(
            RetrievedContext(
                document_id=community_id,
                title="Graph community summary",
                text=str(snapshot.communities[community_id]["summary"]),
                score=score,
            )
        )
    return contexts


def _graph_doc_contexts(
    snapshot: GraphSnapshot,
    doc_scores: dict[str, float],
    limit: int,
    label: str,
) -> list[RetrievedContext]:
    ranked_ids = [
        document_id
        for document_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]
    contexts: list[RetrievedContext] = []
    for document_id in ranked_ids:
        row = snapshot.documents.get(document_id)
        if not row:
            continue
        contexts.append(
            _context_from_graph_doc(
                document_id=document_id,
                title=str(row.get("title", "")),
                text=str(row.get("text", "")),
                score=doc_scores[document_id],
                label=label,
            )
        )
    return contexts


def _seeded_doc_scores(snapshot: GraphSnapshot, matched_entities: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for entity_norm in matched_entities:
        for document_id in snapshot.entity_docs.get(entity_norm, set()):
            pagerank = float(snapshot.documents.get(document_id, {}).get("pagerank", 0.0))
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 + pagerank
    return scores


def _retrieve_graphrag(query: str, snapshot: GraphSnapshot) -> list[RetrievedContext]:
    matched_entities = match_query_entities(snapshot, query)
    if not matched_entities:
        return []
    doc_scores = _seeded_doc_scores(snapshot, matched_entities)
    contexts = _graph_doc_contexts(snapshot, doc_scores, limit=max(2, RETRIEVAL_K // 2), label="Graph local")
    contexts.extend(_community_contexts(snapshot, matched_entities, limit=2))
    return contexts


def _retrieve_lightrag(query: str, snapshot: GraphSnapshot) -> list[RetrievedContext]:
    matched_entities = match_query_entities(snapshot, query)
    if not matched_entities:
        return []
    lowered = query.lower()
    summary_cues = ("overall", "why", "how did", "relationship", "connection", "between")
    global_boost = 1.6 if any(cue in lowered for cue in summary_cues) else 1.0
    local_boost = 1.6 if global_boost == 1.0 else 1.0
    doc_scores = {doc_id: score * local_boost for doc_id, score in _seeded_doc_scores(snapshot, matched_entities).items()}
    contexts = _graph_doc_contexts(snapshot, doc_scores, limit=max(2, RETRIEVAL_K // 2), label="LightRAG local")
    contexts.extend(_community_contexts(snapshot, matched_entities, limit=2, boost=global_boost))
    return contexts


def _retrieve_hipporag(query: str, snapshot: GraphSnapshot) -> list[RetrievedContext]:
    matched_entities = match_query_entities(snapshot, query)
    if not matched_entities:
        return []

    personalization: dict[str, float] = {}
    for entity_norm in matched_entities:
        personalization[entity_norm] = personalization.get(entity_norm, 0.0) + 1.0
        for document_id in snapshot.entity_docs.get(entity_norm, set()):
            personalization[document_id] = personalization.get(document_id, 0.0) + 0.25

    norm = sum(personalization.values()) or 1.0
    personalization = {key: value / norm for key, value in personalization.items()}
    walk_scores = pagerank_scores(snapshot.graph, personalization=personalization, alpha=0.85)

    doc_scores: dict[str, float] = {}
    for document_id in snapshot.documents:
        score = float(walk_scores.get(document_id, 0.0))
        if score > 0:
            doc_scores[document_id] = score

    contexts = _graph_doc_contexts(snapshot, doc_scores, limit=max(2, RETRIEVAL_K // 2), label="HippoRAG walk")
    contexts.extend(_community_contexts(snapshot, matched_entities, limit=1))
    return contexts


def _retrieve_linearrag(query: str, snapshot: GraphSnapshot) -> list[RetrievedContext]:
    terms = graph_query_terms(query)
    if not terms:
        return []

    matched_groups = [
        group_key
        for group_key in snapshot.groups
        if group_key in terms
    ]
    matched_entities = match_query_entities(snapshot, query)
    for entity_norm in matched_entities:
        group_key = str(snapshot.entities.get(entity_norm, {}).get("group_key", ""))
        if group_key:
            matched_groups.append(group_key)

    doc_scores: dict[str, float] = {}
    for group_key in dict.fromkeys(matched_groups):
        for entity_norm in snapshot.group_entities.get(group_key, set()):
            for document_id in snapshot.entity_docs.get(entity_norm, set()):
                pagerank = float(snapshot.documents.get(document_id, {}).get("pagerank", 0.0))
                doc_scores[document_id] = doc_scores.get(document_id, 0.0) + 0.75 + pagerank
    for entity_norm in matched_entities:
        for document_id in snapshot.entity_docs.get(entity_norm, set()):
            doc_scores[document_id] = doc_scores.get(document_id, 0.0) + 1.5

    contexts = [
        RetrievedContext(
            document_id=f"{snapshot.graph_name}:linearrag-route",
            title="LinearRAG hierarchy route",
            text=(
                f"Matched hierarchy groups: {', '.join(list(dict.fromkeys(matched_groups))[:4]) or 'none'}. "
                f"Matched entities: {', '.join(matched_entities[:6]) or 'none'}."
            ),
            score=2.0,
        )
    ]
    contexts.extend(_graph_doc_contexts(snapshot, doc_scores, limit=max(2, RETRIEVAL_K // 2), label="LinearRAG entity route"))
    return contexts


GRAPH_RETRIEVERS: dict[str, Callable[[str, GraphSnapshot], list[RetrievedContext]]] = {
    "graphrag": _retrieve_graphrag,
    "lightrag": _retrieve_lightrag,
    "hipporag": _retrieve_hipporag,
    "linearrag": _retrieve_linearrag,
}


async def _retrieve(query: str, graph_mode: str, graph_name: str) -> list[RetrievedContext]:
    baseline = await _retrieve_es_contexts(query, RETRIEVAL_K)
    try:
        snapshot = await asyncio.to_thread(load_graph_snapshot, graph_name)
        graph_contexts = GRAPH_RETRIEVERS[graph_mode](query, snapshot)
    except Exception:
        graph_contexts = []
    return _dedupe_contexts(graph_contexts + baseline, RETRIEVAL_K)


async def _run_agent(
    user_message: str,
    history: list[ChatMessage],
    graph_mode: str,
    graph_name: str,
) -> tuple[str, list[RetrievedContext], TokenUsage]:
    contexts = await _retrieve(user_message, graph_mode=graph_mode, graph_name=graph_name)
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in history:
        messages.append({"role": message.role, "content": message.content})

    if contexts:
        block = "\n\n".join(
            f"[{index + 1}] {(context.title + ': ') if context.title else ''}{context.text}"
            for index, context in enumerate(contexts)
        )
        prompt = f"Retrieved documents:\n{block}\n\nUser question: {user_message}"
    else:
        prompt = user_message
    messages.append({"role": "user", "content": prompt})

    response = await _openai.chat.completions.create(model=MODEL, messages=messages)
    usage = TokenUsage()
    if response.usage:
        usage.input_tokens = response.usage.prompt_tokens
        usage.output_tokens = response.usage.completion_tokens
        if getattr(response.usage, "prompt_tokens_details", None):
            usage.cached_tokens = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
    return response.choices[0].message.content or "", contexts, usage


def make_graph_agent(graph_mode: str, graph_name: str):
    async def send_message(session_id: str, message: str) -> SendMessageResult:
        from datetime import datetime, timezone

        session = _sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        now = datetime.now(timezone.utc).isoformat()
        _add_message(session_id, ChatMessage(role="user", content=message, timestamp=now))
        history = session.messages[:-1]
        content, contexts, usage = await _run_agent(
            message,
            history,
            graph_mode=graph_mode,
            graph_name=graph_name,
        )
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

    return create_session, send_message
