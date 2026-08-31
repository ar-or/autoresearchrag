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


HYPOTHESIS_ID: str = os.environ.get("HYPOTHESIS_ID", "h72")
HYPOTHESIS_LABEL: str = "Dual Reconstruction Answer Reranker"
HYPOTHESIS_MODE: str = "dual_reconstruction_reranker"

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




# ---------------------------------------------------------------------------
# Hypothesis-specific helpers
# ---------------------------------------------------------------------------


def _usage_from_response(resp: Any) -> TokenUsage:
    usage = TokenUsage()
    if getattr(resp, "usage", None):
        usage.input_tokens += getattr(resp.usage, "prompt_tokens", 0) or 0
        usage.output_tokens += getattr(resp.usage, "completion_tokens", 0) or 0
        details = getattr(resp.usage, "prompt_tokens_details", None)
        if details:
            usage.cached_tokens += getattr(details, "cached_tokens", 0) or 0
    return usage


def _merge_usage(total: TokenUsage, delta: TokenUsage) -> None:
    total.input_tokens += delta.input_tokens
    total.cached_tokens += delta.cached_tokens
    total.output_tokens += delta.output_tokens


def _history_transcript(history: list[ChatMessage]) -> str:
    lines = []
    for msg in history[-8:]:
        role = "User" if msg.role == "user" else "Assistant"
        content = msg.content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _safe_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_json_array(text: str) -> list[Any]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _normalize_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = re.split(r"[,;\n]", raw)
    else:
        values = []
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text.lower() not in {"none", "null", "n/a"} and text not in out:
            out.append(text)
    return out


def _context_key(context: RetrievedContext) -> tuple[str, str, str]:
    return (context.document_id, context.title, context.text[:200])


def _dedupe_contexts(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen: set[tuple[str, str, str]] = set()
    out: list[RetrievedContext] = []
    for context in contexts:
        key = _context_key(context)
        if key in seen:
            continue
        seen.add(key)
        out.append(context)
    return out


def _format_context_block(contexts: list[RetrievedContext]) -> str:
    return "\n\n".join(
        f"[{i+1}] {(c.title + ': ') if c.title else ''}{c.text}"
        for i, c in enumerate(contexts)
    )


def _split_clauses(text: str) -> list[str]:
    pieces = re.split(r"(?<=[.;!?])\s+|\n+", text)
    return [piece.strip(" ;") for piece in pieces if piece.strip(" ;")]


def _extract_years(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", text)))


def _extract_question_terms(text: str) -> list[str]:
    stop = _STOPWORDS | {"did", "does", "many", "much", "than", "into", "from"}
    return [
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in stop
    ]


def _lexical_overlap_score(needles: list[str], haystack: str) -> float:
    if not needles:
        return 0.0
    lowered = haystack.lower()
    hits = sum(1 for needle in needles if needle.lower() in lowered)
    return hits / max(len(needles), 1)


async def _chat_json(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
) -> tuple[dict[str, Any], TokenUsage]:
    try:
        resp = await _openai.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        return _safe_json_object(content), _usage_from_response(resp)
    except Exception:
        return {}, TokenUsage()


async def _chat_text(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
) -> tuple[str, TokenUsage]:
    try:
        resp = await _openai.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        return content.strip(), _usage_from_response(resp)
    except Exception:
        return "", TokenUsage()


async def _build_semantic_profile(
    query: str,
    history: list[ChatMessage],
    *,
    stateful: bool = False,
) -> tuple[dict[str, Any], TokenUsage]:
    transcript = _history_transcript(history)
    label = "state contract" if stateful else "retrieval profile"
    prompt = (
        f"Build a compact {label} for the final user question. Return JSON with keys: "
        "entities, time_constraints, relation, intent, answer_type, evidence_expectations, retrieval_query, target_slots.\n\n"
        f"Conversation history:\n{transcript or '(none)'}\n\nFinal user question: {query}"
    )
    profile, usage = await _chat_json(
        "You extract concise retrieval constraints for grounded QA.",
        prompt,
    )
    profile.setdefault("entities", [])
    profile.setdefault("time_constraints", [])
    profile.setdefault("target_slots", [])
    profile["entities"] = _normalize_list(profile.get("entities"))
    profile["time_constraints"] = _normalize_list(profile.get("time_constraints"))
    profile["target_slots"] = _normalize_list(profile.get("target_slots"))
    profile["evidence_expectations"] = _normalize_list(profile.get("evidence_expectations"))
    if not profile.get("retrieval_query"):
        anchor_text = " ".join(profile["entities"] + profile["time_constraints"])
        profile["retrieval_query"] = " ".join(part for part in [query, anchor_text] if part).strip()
    return profile, usage


def _score_profile_alignment(context: RetrievedContext, profile: dict[str, Any]) -> float:
    haystack = f"{context.title}\n{context.text}"
    entities = _normalize_list(profile.get("entities"))
    time_constraints = _normalize_list(profile.get("time_constraints")) or _extract_years(" ".join(entities))
    expectations = _normalize_list(profile.get("evidence_expectations"))
    relation = str(profile.get("relation", "")).strip()
    intent = str(profile.get("intent", "")).strip()
    score = context.score
    score += 2.0 * _lexical_overlap_score(entities, haystack)
    score += 1.2 * _lexical_overlap_score(time_constraints, haystack)
    score += 0.8 * _lexical_overlap_score(expectations, haystack)
    score += 0.6 * _lexical_overlap_score([relation, intent], haystack)
    return score


def _filter_contexts_by_constraints(
    contexts: list[RetrievedContext],
    profile: dict[str, Any],
    *,
    strict: bool,
) -> list[RetrievedContext]:
    entities = _normalize_list(profile.get("entities"))
    time_constraints = _normalize_list(profile.get("time_constraints"))
    filtered: list[RetrievedContext] = []
    for context in contexts:
        haystack = f"{context.title}\n{context.text}".lower()
        entity_hits = sum(1 for entity in entities if entity.lower() in haystack)
        time_hits = sum(1 for item in time_constraints if item.lower() in haystack)
        if strict:
            if entities and entity_hits == 0:
                continue
            if time_constraints and time_hits == 0:
                continue
        else:
            if entities and entity_hits == 0 and time_constraints and time_hits == 0:
                continue
        filtered.append(context)
    return filtered or contexts


def _cluster_contexts_by_source(contexts: list[RetrievedContext]) -> list[list[RetrievedContext]]:
    grouped: dict[str, list[RetrievedContext]] = {}
    for context in contexts:
        key = context.title or context.document_id
        grouped.setdefault(key, []).append(context)
    return list(grouped.values())


def _detect_conflicts(contexts: list[RetrievedContext], profile: dict[str, Any]) -> list[RetrievedContext]:
    dominant_years = set(_normalize_list(profile.get("time_constraints")))
    if not dominant_years:
        return contexts
    filtered: list[RetrievedContext] = []
    for context in contexts:
        years = set(_extract_years(context.text))
        if years and not (years & dominant_years):
            continue
        filtered.append(context)
    return filtered or contexts


async def _draft_answer_once(
    user_message: str,
    history: list[ChatMessage],
    contexts: list[RetrievedContext],
    *,
    extra_instruction: str = "",
) -> tuple[str, TokenUsage]:
    transcript = _history_transcript(history)
    prompt = (
        f"Conversation history:\n{transcript or '(none)'}\n\n"
        f"Retrieved evidence:\n{_format_context_block(contexts)}\n\n"
        f"Question: {user_message}\n\n"
        f"{extra_instruction}\n"
        "Answer briefly and only from the evidence."
    )
    return await _chat_text(
        "You answer questions using only the provided evidence.",
        prompt,
    )


async def _critique_draft(
    user_message: str,
    history: list[ChatMessage],
    draft: str,
    contexts: list[RetrievedContext],
    *,
    require_verdict: bool = False,
    route_failure: bool = False,
) -> tuple[dict[str, Any], TokenUsage]:
    transcript = _history_transcript(history)
    prompt = (
        "Review the draft answer against the evidence and return JSON with keys: "
        "verdict, grounded, missing_bridge_facts, unsupported_claims, failure_type, bridge_query, revision_plan.\n\n"
        f"Conversation history:\n{transcript or '(none)'}\n\n"
        f"Question: {user_message}\n\nDraft: {draft}\n\nEvidence:\n{_format_context_block(contexts)}"
    )
    critique, usage = await _chat_json(
        "You are a strict evidence-grounded RAG critic.",
        prompt,
    )
    if require_verdict and critique.get("verdict") not in {"GOOD", "BAD"}:
        grounded = bool(critique.get("grounded"))
        critique["verdict"] = "GOOD" if grounded else "BAD"
    if route_failure and critique.get("failure_type") not in {"relevance", "mapping", "synthesis"}:
        unsupported = _normalize_list(critique.get("unsupported_claims"))
        critique["failure_type"] = "mapping" if critique.get("bridge_query") else ("synthesis" if unsupported else "relevance")
    return critique, usage


async def _revise_draft(
    user_message: str,
    history: list[ChatMessage],
    draft: str,
    contexts: list[RetrievedContext],
    plan: Any,
) -> tuple[str, TokenUsage]:
    transcript = _history_transcript(history)
    prompt = (
        f"Conversation history:\n{transcript or '(none)'}\n\n"
        f"Question: {user_message}\n\n"
        f"Current draft: {draft}\n\n"
        f"Revision plan: {json.dumps(plan, ensure_ascii=True)}\n\n"
        f"Evidence:\n{_format_context_block(contexts)}\n\n"
        "Revise the answer so every claim is supported by the evidence. Keep it short."
    )
    return await _chat_text(
        "You rewrite answers to be fully supported by the given evidence.",
        prompt,
    )


async def _generate_answer_candidates(
    user_message: str,
    history: list[ChatMessage],
    contexts: list[RetrievedContext],
    count: int,
) -> tuple[list[str], TokenUsage]:
    transcript = _history_transcript(history)
    prompt = (
        f"Conversation history:\n{transcript or '(none)'}\n\n"
        f"Question: {user_message}\n\n"
        f"Evidence:\n{_format_context_block(contexts)}\n\n"
        f"Return a JSON object with key candidates holding exactly {count} short answer candidates."
    )
    data, usage = await _chat_json(
        "You propose multiple concise evidence-grounded answer candidates.",
        prompt,
        temperature=0.4,
    )
    candidates = _normalize_list(data.get("candidates"))
    return candidates[:count], usage


async def _select_candidate_by_reconstruction(
    user_message: str,
    candidates: list[str],
    profile: dict[str, Any],
) -> tuple[str, TokenUsage]:
    prompt = (
        "Score each candidate by how well it reconstructs the missing target slots of the question. "
        "Return JSON with keys best_candidate and reasoning.\n\n"
        f"Question: {user_message}\n"
        f"Known anchors / constraints: {json.dumps(profile, ensure_ascii=True)}\n"
        f"Candidates: {json.dumps(candidates, ensure_ascii=True)}"
    )
    data, usage = await _chat_json(
        "You rerank answer candidates by reconstruction fidelity.",
        prompt,
    )
    best = str(data.get("best_candidate", "")).strip()
    if best in candidates:
        return best, usage
    return (candidates[0] if candidates else ""), usage


async def _verify_slots(
    user_message: str,
    draft: str,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], TokenUsage]:
    prompt = (
        "Check whether the draft answer resolves each target slot from the question. Return JSON with keys "
        "resolved_slots, missing_slots, followup_query.\n\n"
        f"Question: {user_message}\nDraft: {draft}\nProfile: {json.dumps(profile, ensure_ascii=True)}"
    )
    data, usage = await _chat_json(
        "You verify whether an answer resolves the intended unknowns.",
        prompt,
    )
    data["missing_slots"] = _normalize_list(data.get("missing_slots"))
    return data, usage


async def _license_answer_clauses(
    user_message: str,
    answer: str,
    contexts: list[RetrievedContext],
) -> tuple[list[dict[str, Any]], TokenUsage]:
    clauses = _split_clauses(answer) or ([answer.strip()] if answer.strip() else [])
    prompt = (
        "For each answer clause, classify it as SUPPORTED, CONTRADICTED, or INSUFFICIENT against the evidence. "
        "Return JSON with key clauses, where each item has keys text, verdict, justification.\n\n"
        f"Question: {user_message}\n"
        f"Answer clauses: {json.dumps(clauses, ensure_ascii=True)}\n\n"
        f"Evidence:\n{_format_context_block(contexts)}"
    )
    data, usage = await _chat_json(
        "You act as a licensing oracle over evidence-grounded answer clauses.",
        prompt,
    )
    clauses_out = data.get("clauses", [])
    return clauses_out if isinstance(clauses_out, list) else [], usage


def _aggregate_clause_verdicts(clauses: list[dict[str, Any]]) -> tuple[str, list[str]]:
    supported: list[str] = []
    contradicted = 0
    insufficient = 0
    for item in clauses:
        text = str(item.get("text", "")).strip()
        verdict = str(item.get("verdict", "")).upper().strip()
        if verdict == "SUPPORTED" and text:
            supported.append(text)
        elif verdict == "CONTRADICTED":
            contradicted += 1
        else:
            insufficient += 1
    if contradicted and not supported:
        return "CONTRADICTED", supported
    if insufficient and not supported:
        return "INSUFFICIENT", supported
    if contradicted and supported:
        return "MIXED", supported
    return "SUPPORTED", supported


async def retrieve(
    query: str,
    index: str | None = None,
    k: int | None = None,
    history: list[ChatMessage] | None = None,
) -> list[RetrievedContext]:
    idx = index or ELASTIC_INDEX
    top_k = k or RETRIEVAL_K
    retrieval_query = query
    candidate_k = max(top_k * 2, top_k)
    usage = TokenUsage()
    profile: dict[str, Any] = {}

    if HYPOTHESIS_ID in {"h73", "h74", "h75", "h76", "h81", "h82", "h83", "h84"}:
        profile, profile_usage = await _build_semantic_profile(
            query,
            history or [],
            stateful=HYPOTHESIS_ID in {"h81", "h82", "h83", "h84"},
        )
        _merge_usage(usage, profile_usage)
        retrieval_query = str(profile.get("retrieval_query") or query).strip() or query
    if HYPOTHESIS_ID in {"h75", "h83"}:
        candidate_k = max(top_k * 4, top_k)
    elif _use_rich_retrieval(query):
        rewritten = await _rewrite_retrieval_query(retrieval_query)
        retrieval_query = rewritten or retrieval_query
        candidate_k = max(top_k * 3, top_k)

    try:
        vector = (await asyncio.to_thread(embed_batch, [retrieval_query]))[0]
        vector_hits = _es_vector_search(idx, vector, candidate_k)
        text_hits = _es_text_search(idx, retrieval_query, candidate_k)
        hits = _fuse_hits_rrf(vector_hits, text_hits, k=candidate_k)
    except Exception:
        try:
            hits = _es_text_search(idx, retrieval_query, candidate_k)
        except Exception:
            return []
    contexts = _hits_to_contexts(hits)

    if HYPOTHESIS_ID in {"h74", "h75", "h76", "h81", "h82"} and profile:
        if HYPOTHESIS_ID == "h75":
            clusters = _cluster_contexts_by_source(contexts)
            ranked_clusters = sorted(
                clusters,
                key=lambda cluster: max(_score_profile_alignment(item, profile) for item in cluster),
                reverse=True,
            )
            trimmed: list[RetrievedContext] = []
            for cluster in ranked_clusters[:max(2, top_k)]:
                trimmed.extend(sorted(cluster, key=lambda item: _score_profile_alignment(item, profile), reverse=True)[:2])
            contexts = trimmed
        contexts = sorted(contexts, key=lambda item: _score_profile_alignment(item, profile), reverse=True)
        strict = HYPOTHESIS_ID in {"h76", "h82"}
        contexts = _filter_contexts_by_constraints(contexts, profile, strict=strict)
        if HYPOTHESIS_ID == "h76":
            contexts = _detect_conflicts(contexts, profile)

    contexts = _dedupe_contexts(contexts)
    try:
        reranked = await _sentence_rerank_contexts(retrieval_query, contexts, max(top_k, len(contexts)))
    except Exception:
        reranked = contexts
    return reranked[:top_k]


async def _postprocess_answer(
    user_message: str,
    history: list[ChatMessage],
    draft: str,
    contexts: list[RetrievedContext],
    usage: TokenUsage,
) -> str:
    if not draft:
        return draft

    if HYPOTHESIS_ID == "h69":
        critique, extra = await _critique_draft(user_message, history, draft, contexts)
        _merge_usage(usage, extra)
        if critique.get("unsupported_claims") or critique.get("missing_bridge_facts"):
            revised, extra = await _revise_draft(user_message, history, draft, contexts, critique)
            _merge_usage(usage, extra)
            return revised or draft
        return draft

    if HYPOTHESIS_ID == "h70":
        critique, extra = await _critique_draft(user_message, history, draft, contexts, route_failure=True)
        _merge_usage(usage, extra)
        failure_type = critique.get("failure_type")
        revised_contexts = contexts
        if failure_type in {"relevance", "mapping"}:
            bridge_query = str(critique.get("bridge_query") or user_message).strip() or user_message
            extra_contexts = await retrieve(bridge_query, history=history)
            revised_contexts = _dedupe_contexts(contexts + extra_contexts)
        revised, extra = await _revise_draft(user_message, history, draft, revised_contexts, critique)
        _merge_usage(usage, extra)
        return revised or draft

    if HYPOTHESIS_ID == "h71":
        critique, extra = await _critique_draft(user_message, history, draft, contexts, require_verdict=True)
        _merge_usage(usage, extra)
        if critique.get("verdict") == "GOOD":
            return draft
        revised, extra = await _revise_draft(user_message, history, draft, contexts, critique)
        _merge_usage(usage, extra)
        return revised or draft

    if HYPOTHESIS_ID == "h72":
        profile, extra = await _build_semantic_profile(user_message, history)
        _merge_usage(usage, extra)
        candidates, extra = await _generate_answer_candidates(user_message, history, contexts, 3)
        _merge_usage(usage, extra)
        if draft not in candidates and draft:
            candidates = [draft] + candidates
        best, extra = await _select_candidate_by_reconstruction(user_message, candidates[:4], profile)
        _merge_usage(usage, extra)
        return best or draft

    if HYPOTHESIS_ID == "h73":
        profile, extra = await _build_semantic_profile(user_message, history)
        _merge_usage(usage, extra)
        verdict, extra = await _verify_slots(user_message, draft, profile)
        _merge_usage(usage, extra)
        if verdict.get("missing_slots"):
            followup_query = str(verdict.get("followup_query") or profile.get("retrieval_query") or user_message).strip() or user_message
            extra_contexts = await retrieve(followup_query, history=history)
            revised_contexts = _dedupe_contexts(contexts + extra_contexts)
            revised, extra = await _revise_draft(user_message, history, draft, revised_contexts, verdict)
            _merge_usage(usage, extra)
            return revised or draft
        return draft

    if HYPOTHESIS_ID in {"h77", "h78", "h84"}:
        clauses, extra = await _license_answer_clauses(user_message, draft, contexts)
        _merge_usage(usage, extra)
        aggregate, supported = _aggregate_clause_verdicts(clauses)
        if HYPOTHESIS_ID == "h77":
            return "; ".join(supported) if supported else draft
        if HYPOTHESIS_ID == "h78":
            coverage = len(supported) / max(len(clauses), 1)
            return "; ".join(supported) if supported and coverage >= 0.6 and aggregate == "SUPPORTED" else "INSUFFICIENT_EVIDENCE"
        if aggregate == "SUPPORTED":
            return "; ".join(supported) if supported else draft
        if aggregate == "MIXED" and supported:
            return "; ".join(supported)
        return "INSUFFICIENT_EVIDENCE"

    if HYPOTHESIS_ID == "h79":
        clauses, extra = await _license_answer_clauses(user_message, draft, contexts)
        _merge_usage(usage, extra)
        repaired: list[str] = []
        for item in clauses:
            verdict = str(item.get("verdict", "")).upper()
            text = str(item.get("text", "")).strip()
            if verdict == "SUPPORTED" and text:
                repaired.append(text)
                continue
            if verdict == "INSUFFICIENT" and text:
                revised, extra = await _revise_draft(user_message, history, text, contexts, item)
                _merge_usage(usage, extra)
                if revised:
                    repaired.append(revised)
        return "; ".join(repaired) if repaired else draft

    if HYPOTHESIS_ID == "h80":
        candidates, extra = await _generate_answer_candidates(user_message, history, contexts, 3)
        _merge_usage(usage, extra)
        if draft and draft not in candidates:
            candidates = [draft] + candidates
        best_answer = draft
        best_score = -1.0
        for candidate in candidates[:3]:
            clauses, extra = await _license_answer_clauses(user_message, candidate, contexts)
            _merge_usage(usage, extra)
            aggregate, supported = _aggregate_clause_verdicts(clauses)
            score = len(supported) - (2 if aggregate == "CONTRADICTED" else 0)
            if score > best_score:
                best_score = score
                best_answer = "; ".join(supported) if supported else candidate
        return best_answer or draft

    if HYPOTHESIS_ID == "h83":
        counter_query = f"counter evidence contradiction {user_message}"
        counter_contexts = await retrieve(counter_query, history=history)
        revised_contexts = _dedupe_contexts(contexts + counter_contexts)
        critique, extra = await _critique_draft(user_message, history, draft, revised_contexts, route_failure=True)
        _merge_usage(usage, extra)
        revised, extra = await _revise_draft(user_message, history, draft, revised_contexts, critique)
        _merge_usage(usage, extra)
        return revised or draft

    return draft


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
    contexts = await retrieve(user_message, history=history)
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
        content = await _postprocess_answer(user_message, history, content, contexts, usage)
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
