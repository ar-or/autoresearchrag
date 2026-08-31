#!/usr/bin/env python3
"""MT-RAG benchmark evaluator.

Environment variables:
  AGENT_MODE                  - "local" (default) or "http"
  ORAGENT_URL                 - oragent base URL when AGENT_MODE=http
  N                           - max conversations, 0 = all (default: 0)
  PARALLELISM                 - concurrency degree (default: 32)
  MTRAG_COLLECTION            - optional domain or collection filter
  MTRAG_GENERATION_MODE       - "gold_history" (default) or "rollout"
  MTRAG_RETRIEVAL_QUERY_MODES - comma list of: lastturn,rewrite,questions
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_client import make_client
from cost import CostCalculator
from metrics import (
    compute_retrieval_metrics,
    exact_match,
    extractiveness_rouge,
    is_abstention_response,
    rouge_l,
    token_f1,
    token_precision,
    token_recall,
)


def split_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_CONVERSATIONS = int(os.environ.get("N", "0"))
PARALLELISM = int(os.environ.get("PARALLELISM", "32"))
MTRAG_SCOPE = os.environ.get("MTRAG_COLLECTION", "")
MTRAG_INDEX_COLLECTION_NAME = os.environ.get("MTRAG_INDEX_COLLECTION_NAME", "mycollection3")
GENERATION_MODE = os.environ.get("MTRAG_GENERATION_MODE", "gold_history").lower()
RETRIEVAL_QUERY_MODES = split_csv(
    os.environ.get("MTRAG_RETRIEVAL_QUERY_MODES", "lastturn,rewrite,questions")
)

VALID_GENERATION_MODES = {"gold_history", "rollout"}
VALID_RETRIEVAL_QUERY_MODES = {"lastturn", "rewrite", "questions"}
if GENERATION_MODE not in VALID_GENERATION_MODES:
    print(
        f"WARNING: Unknown MTRAG_GENERATION_MODE={GENERATION_MODE!r}; "
        "falling back to 'gold_history'"
    )
    GENERATION_MODE = "gold_history"
if not RETRIEVAL_QUERY_MODES:
    RETRIEVAL_QUERY_MODES = ["lastturn", "rewrite", "questions"]
RETRIEVAL_QUERY_MODES = [
    mode for mode in RETRIEVAL_QUERY_MODES if mode in VALID_RETRIEVAL_QUERY_MODES
]
if not RETRIEVAL_QUERY_MODES:
    RETRIEVAL_QUERY_MODES = ["lastturn", "rewrite", "questions"]

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "mt-rag-benchmark"
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


RETRIEVAL_DIR = DATA_DIR / "human" / "retrieval_tasks"
ALL_DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]
RETRIEVAL_QUERY_FILES = {
    "lastturn": "{domain}_lastturn.jsonl",
    "rewrite": "{domain}_rewrite.jsonl",
    "questions": "{domain}_questions.jsonl",
}
DOMAIN_TO_COLLECTION = {
    "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
    "cloud": "mt-rag-ibmcloud-elser-512-100-20240502",
    "fiqa": "mt-rag-fiqa-beir-elser-512-100-20240501",
    "govt": "mt-rag-govt-elser-512-100-20240611",
}
DOMAIN_ALIASES = {
    "clapnq": "clapnq",
    "cloud": "cloud",
    "ibmcloud": "cloud",
    "fiqa": "fiqa",
    "govt": "govt",
}
REQUESTED_SCOPE_VALUES = split_csv(MTRAG_SCOPE)
REQUESTED_DOMAINS = {
    DOMAIN_ALIASES[value.lower()]
    for value in REQUESTED_SCOPE_VALUES
    if value.lower() in DOMAIN_ALIASES
}
REQUESTED_COLLECTIONS = {
    value
    for value in REQUESTED_SCOPE_VALUES
    if value.lower() not in DOMAIN_ALIASES and value not in {"*", "all", "ALL"}
}


def find_data_files() -> list[Path]:
    rag_file = DATA_DIR / "human" / "generation_tasks" / "RAG.jsonl"
    if rag_file.exists():
        return [rag_file]
    files: list[Path] = []
    files.extend(DATA_DIR.glob("**/*.jsonl"))
    return sorted(
        f for f in files if f.stat().st_size > 100 and "generation_tasks" in str(f)
    )


def load_tasks(files: list[Path]) -> list[dict]:
    tasks: list[dict] = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return tasks


def group_conversations(tasks: list[dict]) -> dict[str, list[dict]]:
    convos: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        cid = task.get("conversation_id", task.get("task_id", "unknown"))
        convos[cid].append(task)
    for cid in convos:
        convos[cid].sort(key=lambda task: int(task.get("turn", 0)))
    return dict(convos)


def collection_to_domain(collection: str) -> str:
    lowered = collection.lower()
    if "clapnq" in lowered:
        return "clapnq"
    if "ibmcloud" in lowered or lowered == "cloud":
        return "cloud"
    if "fiqa" in lowered:
        return "fiqa"
    if "govt" in lowered:
        return "govt"
    return "unknown"


def task_matches_scope(task: dict) -> bool:
    if not REQUESTED_SCOPE_VALUES:
        return True
    collection = task.get("Collection", task.get("collection", ""))
    domain = collection_to_domain(collection)
    if collection in REQUESTED_COLLECTIONS:
        return True
    return domain in REQUESTED_DOMAINS


def retrieval_domain_selected(domain: str) -> bool:
    if not REQUESTED_SCOPE_VALUES:
        return True
    if domain in REQUESTED_DOMAINS:
        return True
    return DOMAIN_TO_COLLECTION.get(domain, "") in REQUESTED_COLLECTIONS


def active_domains_from_tasks(tasks: list[dict]) -> list[str]:
    domains = sorted(
        {
            collection_to_domain(task.get("Collection", task.get("collection", "")))
            for task in tasks
        }
        - {"unknown"}
    )
    if domains:
        return domains
    if REQUESTED_DOMAINS:
        return sorted(REQUESTED_DOMAINS)
    domains_from_collections = sorted(
        {
            collection_to_domain(collection)
            for collection in REQUESTED_COLLECTIONS
            if collection_to_domain(collection) != "unknown"
        }
    )
    return domains_from_collections or ALL_DOMAINS[:]


def configure_agent_scope_filters(domains: list[str]) -> None:
    source_prefixes = [f"mtrag:{domain}:" for domain in domains]
    os.environ["MTRAG_ALLOWED_SOURCE_PREFIXES"] = ",".join(source_prefixes)
    os.environ["MTRAG_ALLOWED_COLLECTION_NAMES"] = MTRAG_INDEX_COLLECTION_NAME


def _es_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("ELASTIC_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    return headers


def count_indexed_mtrag_docs(domain: str) -> int:
    es_url = os.environ.get("ES_HOST", "http://localhost:9200")
    es_index = os.environ.get("ES_INDEX", "mtrag")
    response = requests.post(
        f"{es_url}/{es_index}/_count",
        headers=_es_headers(),
        json={
            "query": {
                "bool": {
                    "filter": [
                        {"prefix": {"source": f"mtrag:{domain}:"}},
                        {"term": {"collection_name": MTRAG_INDEX_COLLECTION_NAME}},
                    ]
                }
            }
        },
        timeout=30,
    )
    response.raise_for_status()
    return int(response.json().get("count", 0))


def ensure_mtrag_index_ready(domains: list[str]) -> dict[str, int]:
    counts = {domain: count_indexed_mtrag_docs(domain) for domain in domains}
    missing = [domain for domain, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(
            "No indexed MT-RAG passages found for "
            f"{', '.join(missing)} in the configured Elasticsearch index. "
            "Run `evaluators/mtrag/ingest_mtrag.py` for the missing domains "
            "and rerun the evaluator."
        )
    return counts


# ---------------------------------------------------------------------------
# Prompt/history helpers
# ---------------------------------------------------------------------------


def normalize_task_label(raw_value, fallback: str) -> str:
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else fallback
    value = str(raw_value or fallback).strip()
    if not value:
        return fallback
    return value


def normalize_multi_turn_label(raw_value) -> str:
    lowered = normalize_task_label(raw_value, "unknown").lower()
    if lowered in {"follow up", "follow-up"}:
        return "Follow-up"
    if lowered == "clarification":
        return "Clarification"
    if lowered in {"n/a", "na"}:
        return "N/A"
    return lowered or "unknown"


def extract_chat_messages(input_data) -> list[dict[str, str]]:
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data.strip()}] if input_data.strip() else []
    if not isinstance(input_data, list):
        text = str(input_data).strip()
        return [{"role": "user", "content": text}] if text else []

    messages: list[dict[str, str]] = []
    for item in input_data:
        if isinstance(item, dict):
            speaker = str(item.get("speaker") or item.get("role") or "").strip().lower()
            if speaker == "user":
                role = "user"
            elif speaker in {"agent", "assistant"}:
                role = "assistant"
            else:
                continue
            content = str(item.get("text") or item.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
            continue
        if isinstance(item, str):
            text = item.strip()
            if text:
                messages.append({"role": "user", "content": text})
    return messages


def get_latest_user_message(input_data) -> str:
    messages = extract_chat_messages(input_data)
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return str(input_data)


def split_history_and_current_user(
    input_data,
) -> tuple[list[dict[str, str]], str, list[dict[str, str]]]:
    messages = extract_chat_messages(input_data)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            return messages[:index], messages[index]["content"], messages
    if messages:
        return messages[:-1], messages[-1]["content"], messages
    return [], "", []


def render_conversation_transcript(messages: list[dict[str, str]]) -> str:
    lines = [
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content']}"
        for message in messages
        if message.get("content")
    ]
    if not lines:
        return ""
    return (
        "Use the conversation below as context and answer the final user message.\n\n"
        + "\n".join(lines)
    )


def normalize_query_text(text: str) -> str:
    lines: list[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("|user|:"):
            stripped = stripped[len("|user|:") :].strip()
        elif stripped.startswith("|assistant|:"):
            stripped = stripped[len("|assistant|:") :].strip()
        elif stripped.startswith("|agent|:"):
            stripped = stripped[len("|agent|:") :].strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Gold relevant document IDs
# ---------------------------------------------------------------------------


def _hash_passage_id(raw_id: str) -> str:
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def get_gold_relevant_ids(task: dict) -> set[str]:
    relevant: set[str] = set()
    for ctx in task.get("contexts", []):
        if not ctx.get("reference"):
            continue
        doc_id = ctx.get("document_id", ctx.get("id", ""))
        if doc_id:
            relevant.add(_hash_passage_id(doc_id))
    return relevant


# ---------------------------------------------------------------------------
# Prediction/result builders
# ---------------------------------------------------------------------------


def build_prediction(
    task: dict,
    conversation_id: str,
    resp,
    evaluation_mode: str,
    history_seeded: bool,
) -> dict:
    collection = task.get("Collection", task.get("collection", ""))
    return {
        "task_id": task.get("task_id", ""),
        "conversation_id": conversation_id,
        "collection": collection,
        "domain": collection_to_domain(collection),
        "turn": int(task.get("turn", 0)),
        "input": task.get("input"),
        "contexts": [
            {
                "document_id": context.document_id,
                "text": context.text,
                "title": context.title,
                "score": context.score,
            }
            for context in resp.contexts
        ],
        "predictions": [{"text": resp.response}],
        "targets": task.get("targets", []),
        "gold_contexts": task.get("contexts", []),
        "tokens": {
            "input": resp.usage.input_tokens,
            "cached": resp.usage.cached_tokens,
            "output": resp.usage.output_tokens,
        },
        "model": resp.model,
        "answerability": normalize_task_label(task.get("Answerability"), "unknown"),
        "multi_turn": normalize_multi_turn_label(task.get("Multi-Turn")),
        "question_types": [
            str(value).strip()
            for value in task.get("Question Type", [])
            if str(value).strip()
        ]
        or ["unknown"],
        "rewritten_query": task.get("rewritten_query", ""),
        "gold_reference_count": len(get_gold_relevant_ids({"contexts": task.get("contexts", [])})),
        "evaluation_mode": evaluation_mode,
        "history_seeded": history_seeded,
    }


def build_retrieval_result(task: dict, resp) -> dict:
    return {
        "query_id": task["query_id"],
        "domain": task["domain"],
        "query_variant": task["query_variant"],
        "query_text": task["text"],
        "retrieved_ids": [context.document_id for context in resp.contexts],
        "qrels": task["qrels"],
        "tokens": {
            "input": resp.usage.input_tokens,
            "cached": resp.usage.cached_tokens,
            "output": resp.usage.output_tokens,
        },
        "model": resp.model,
    }


# ---------------------------------------------------------------------------
# Generation task runners
# ---------------------------------------------------------------------------


def run_conversation_rollout(client, conversation_id: str, tasks: list[dict]) -> list[dict]:
    session_id = client.create_session()
    predictions: list[dict] = []
    try:
        for task in tasks:
            user_text = get_latest_user_message(task.get("input", ""))
            resp = client.send_message(session_id, user_text)
            predictions.append(
                build_prediction(
                    task,
                    conversation_id,
                    resp,
                    evaluation_mode="rollout",
                    history_seeded=False,
                )
            )
    finally:
        client.delete_session(session_id)
    return predictions


def run_conversation_gold_history(
    client,
    conversation_id: str,
    tasks: list[dict],
) -> list[dict]:
    predictions: list[dict] = []
    for task in tasks:
        session_id = client.create_session()
        try:
            history, current_user, messages = split_history_and_current_user(task.get("input", ""))
            request_text = current_user or get_latest_user_message(task.get("input", ""))
            history_seeded = False
            if history:
                history_seeded = client.seed_session_messages(session_id, history)
            if history and not history_seeded:
                request_text = render_conversation_transcript(messages)
            resp = client.send_message(session_id, request_text)
            predictions.append(
                build_prediction(
                    task,
                    conversation_id,
                    resp,
                    evaluation_mode="gold_history",
                    history_seeded=history_seeded,
                )
            )
        finally:
            client.delete_session(session_id)
    return predictions


def run_conversation(client, conversation_id: str, tasks: list[dict]) -> list[dict]:
    if GENERATION_MODE == "rollout":
        return run_conversation_rollout(client, conversation_id, tasks)
    return run_conversation_gold_history(client, conversation_id, tasks)


async def run_conversation_async(
    sem: asyncio.Semaphore,
    client,
    idx: int,
    total: int,
    cid: str,
    conv_tasks: list[dict],
) -> list[dict]:
    async with sem:
        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
            predictions = await loop.run_in_executor(None, run_conversation, client, cid, conv_tasks)
        except Exception as exc:
            print(f"    [{idx}/{total}] ERROR conversation {cid}: {exc}")
            return []
        elapsed = time.time() - t0
        print(
            f"    [{idx}/{total}] conversation {cid} ({len(conv_tasks)} turns) "
            f"done in {elapsed:.1f}s"
        )
        return predictions


# ---------------------------------------------------------------------------
# Standalone retrieval tasks
# ---------------------------------------------------------------------------


def load_qrels(domain: str) -> dict[str, dict[str, int]]:
    qrel_path = RETRIEVAL_DIR / domain / "qrels" / "dev.tsv"
    if not qrel_path.exists():
        return {}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(qrel_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("query-id"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], int(parts[2])
            qrels[qid][cid] = score
    return qrels


def load_retrieval_tasks(query_modes: list[str]) -> list[dict]:
    tasks: list[dict] = []
    for query_variant in query_modes:
        pattern = RETRIEVAL_QUERY_FILES[query_variant]
        for domain in ALL_DOMAINS:
            if not retrieval_domain_selected(domain):
                continue
            qrels = load_qrels(domain)
            if not qrels:
                continue
            query_file = RETRIEVAL_DIR / domain / pattern.format(domain=domain)
            if not query_file.exists():
                continue
            with open(query_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    qid = obj.get("_id", obj.get("id", ""))
                    text = normalize_query_text(obj.get("text", ""))
                    if qid and text and qid in qrels:
                        tasks.append(
                            {
                                "query_id": qid,
                                "text": text,
                                "domain": domain,
                                "query_variant": query_variant,
                                "qrels": dict(qrels[qid]),
                            }
                        )
    return tasks


def limit_retrieval_tasks(tasks: list[dict], per_variant_limit: int) -> list[dict]:
    if per_variant_limit <= 0:
        return tasks
    limited: list[dict] = []
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        by_variant[task["query_variant"]].append(task)
    for query_variant in RETRIEVAL_QUERY_MODES:
        limited.extend(by_variant.get(query_variant, [])[:per_variant_limit])
    return limited


def run_retrieval_task(client, task: dict) -> dict:
    session_id = client.create_session()
    try:
        resp = client.send_message(session_id, task["text"])
        return build_retrieval_result(task, resp)
    finally:
        client.delete_session(session_id)


async def run_retrieval_task_async(
    sem: asyncio.Semaphore,
    client,
    idx: int,
    total: int,
    task: dict,
) -> dict | None:
    async with sem:
        t0 = time.time()
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_retrieval_task, client, task)
        except Exception as exc:
            print(f"    [{idx}/{total}] ERROR retrieval {task['query_id']}: {exc}")
            return None
        elapsed = time.time() - t0
        print(
            f"    [{idx}/{total}] retrieval {task['query_variant']}:{task['domain']}:"
            f"{task['query_id'][:20]} done in {elapsed:.1f}s"
        )
        return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def append_metric_values(
    metrics_all: dict[str, list[float]],
    metrics_by_group: dict[str, dict[str, list[float]]],
    metric_values: dict[str, float],
    groups: list[str],
) -> None:
    for name, value in metric_values.items():
        metrics_all[name].append(float(value))
        for group in groups:
            metrics_by_group[group][name].append(float(value))


def generation_groups(pred: dict) -> list[str]:
    groups = {
        f"evaluation_mode={pred.get('evaluation_mode', 'unknown')}",
        f"collection={pred.get('collection', 'unknown')}",
        f"domain={pred.get('domain', 'unknown')}",
        f"answerability={pred.get('answerability', 'unknown')}",
        f"multi_turn={pred.get('multi_turn', 'unknown')}",
    }
    for question_type in pred.get("question_types", []) or ["unknown"]:
        groups.add(f"question_type={question_type}")
    return sorted(groups)


def evaluate_predictions(all_predictions: list[dict]):
    gen_metrics_all: dict[str, list[float]] = defaultdict(list)
    gen_metrics_by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    ret_metrics_all: dict[str, list[float]] = defaultdict(list)
    ret_metrics_by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    summary = {
        "turns": len(all_predictions),
        "retrieval_scorable_turns": 0,
        "turns_without_gold_support_labels": 0,
        "zero_context_turns": 0,
    }

    for pred in all_predictions:
        groups = generation_groups(pred)
        pred_text = pred["predictions"][0]["text"] if pred["predictions"] else ""
        support_text = "\n\n".join(
            context.get("text", "") for context in pred.get("contexts", []) if context.get("text")
        )
        if not pred.get("contexts"):
            summary["zero_context_turns"] += 1

        targets = pred.get("targets", [])
        if targets:
            target = targets[0]
            ref_text = target.get("text", str(target)) if isinstance(target, dict) else str(target)
            precision = token_precision(pred_text, ref_text)
            recall = token_recall(pred_text, ref_text)
            f1 = token_f1(pred_text, ref_text)
            rl = rouge_l(pred_text, ref_text)
            em = exact_match(pred_text, ref_text)
            answerability = pred.get("answerability", "unknown")
            abstains = is_abstention_response(pred_text)
            expects_abstention = answerability in {"UNANSWERABLE", "CONVERSATIONAL"}
            idk_conditioned_f1 = (
                1.0
                if expects_abstention and abstains
                else 0.0
                if expects_abstention or abstains
                else f1
            )
            idk_conditioned_rouge = (
                1.0
                if expects_abstention and abstains
                else 0.0
                if expects_abstention or abstains
                else rl
            )
            append_metric_values(
                gen_metrics_all,
                gen_metrics_by_group,
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "rouge_l": rl,
                    "exact_match": em,
                    "support_precision": token_precision(pred_text, support_text) if support_text else 0.0,
                    "support_extractiveness_rouge_l": (
                        extractiveness_rouge(support_text, pred_text) if support_text else 0.0
                    ),
                    "abstention_rate": float(abstains),
                    "abstention_accuracy": float(abstains == expects_abstention),
                    "idk_conditioned_f1": idk_conditioned_f1,
                    "idk_conditioned_rouge_l": idk_conditioned_rouge,
                    "response_length_chars": float(len(pred_text)),
                },
                groups,
            )

        retrieved_ids = [
            context.get("document_id", "")
            for context in pred.get("contexts", [])
            if context.get("document_id")
        ]
        gold_relevant = get_gold_relevant_ids({"contexts": pred.get("gold_contexts", [])})
        if gold_relevant:
            summary["retrieval_scorable_turns"] += 1
            append_metric_values(
                ret_metrics_all,
                ret_metrics_by_group,
                compute_retrieval_metrics(retrieved_ids, gold_relevant),
                groups,
            )
        else:
            summary["turns_without_gold_support_labels"] += 1

    return (
        gen_metrics_all,
        gen_metrics_by_group,
        ret_metrics_all,
        ret_metrics_by_group,
        summary,
    )


def evaluate_retrieval_tasks(results: list[dict]):
    ret_metrics_all: dict[str, list[float]] = defaultdict(list)
    ret_metrics_by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    summary = {
        "queries": len(results),
        "queries_with_gold_qrels": 0,
        "zero_hit_queries": 0,
    }

    for result in results:
        gold_relevant = {
            _hash_passage_id(corpus_id)
            for corpus_id, score in result["qrels"].items()
            if score > 0
        }
        if not gold_relevant:
            continue
        summary["queries_with_gold_qrels"] += 1
        if not result["retrieved_ids"]:
            summary["zero_hit_queries"] += 1
        groups = [
            f"variant={result['query_variant']}",
            f"domain={result['domain']}",
            f"variant={result['query_variant']}|domain={result['domain']}",
        ]
        append_metric_values(
            ret_metrics_all,
            ret_metrics_by_group,
            compute_retrieval_metrics(result["retrieved_ids"], gold_relevant),
            groups,
        )

    return ret_metrics_all, ret_metrics_by_group, summary


def _print_summary(summary: dict[str, int | str] | None) -> None:
    if not summary:
        return
    print("  Summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")


def _print_metric_block(
    title: str,
    metrics_all: dict[str, list[float]],
    metrics_by_group: dict[str, dict[str, list[float]]],
    summary: dict[str, int | str] | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    _print_summary(summary)
    if not metrics_all:
        print("  No results to evaluate.")
        return
    print("  Overall:")
    for name in sorted(metrics_all.keys()):
        print(f"    {name:>28s}: {avg(metrics_all[name]):.4f}  (n={len(metrics_all[name])})")
    for group in sorted(metrics_by_group.keys()):
        print(f"\n  [{group}]")
        for name in sorted(metrics_by_group[group].keys()):
            print(
                f"    {name:>28s}: {avg(metrics_by_group[group][name]):.4f}  "
                f"(n={len(metrics_by_group[group][name])})"
            )


def print_metrics(
    gen_all,
    gen_by_group,
    ret_all,
    ret_by_group,
    generation_summary,
    standalone_ret_all=None,
    standalone_ret_by_group=None,
    standalone_summary=None,
) -> None:
    _print_metric_block(
        "RETRIEVAL METRICS (generation tasks)",
        ret_all,
        ret_by_group,
        generation_summary,
    )
    if standalone_ret_all is not None:
        _print_metric_block(
            "RETRIEVAL METRICS (standalone retrieval tasks)",
            standalone_ret_all,
            standalone_ret_by_group or {},
            standalone_summary,
        )
    _print_metric_block("GENERATION METRICS", gen_all, gen_by_group)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def aggregate_token_usage(records: list[dict]) -> tuple[dict[str, int], str]:
    total_tokens = {"input": 0, "cached": 0, "output": 0}
    model_name = "unknown"
    for record in records:
        tokens = record.get("tokens", {})
        total_tokens["input"] += tokens.get("input", 0)
        total_tokens["cached"] += tokens.get("cached", 0)
        total_tokens["output"] += tokens.get("output", 0)
        model = record.get("model", "unknown")
        if model != "unknown":
            model_name = model
    return total_tokens, model_name


def print_usage_summary(title: str, records: list[dict], wall_elapsed: float | None = None) -> None:
    total_tokens, model_name = aggregate_token_usage(records)
    n_records = len(records) or 1
    cost_calc = CostCalculator(model_name)
    total_cost = cost_calc.cost(
        total_tokens["input"], total_tokens["cached"], total_tokens["output"]
    )

    print(f"\n{title}")
    print(f"  Model:              {model_name}")
    if wall_elapsed is not None:
        print(f"  Wall clock:         {wall_elapsed:.1f}s  (parallelism={PARALLELISM})")
    print(f"  Records:            {len(records)}")
    print()
    print("  Token usage:        input      cached     output")
    print(
        f"    Total:            {total_tokens['input']:>9}  "
        f"{total_tokens['cached']:>9}  {total_tokens['output']:>9}"
    )
    print(
        f"    Avg/record:       {total_tokens['input']//n_records:>9}  "
        f"{total_tokens['cached']//n_records:>9}  {total_tokens['output']//n_records:>9}"
    )
    print()
    print(f"  Pricing (per 1M tokens):  {cost_calc.format_pricing_line()}")
    print(f"  Total cost:         ${total_cost:.6f}")
    print(f"  Avg cost/record:    ${total_cost/n_records:.6f}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main():
    if not DATA_DIR.exists():
        print(f"ERROR: Data not found at {DATA_DIR}")
        print("Run download_data.sh first.")
        sys.exit(1)

    data_files = find_data_files()
    if not data_files:
        print(f"ERROR: No JSONL files found in {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(data_files)} data file(s)")

    tasks = load_tasks(data_files)
    print(f"Loaded {len(tasks)} generation tasks")

    if REQUESTED_SCOPE_VALUES:
        tasks = [task for task in tasks if task_matches_scope(task)]
        print(f"Filtered to {len(tasks)} generation tasks for scope {REQUESTED_SCOPE_VALUES}")

    conversations = group_conversations(tasks)
    conv_ids = sorted(conversations.keys())
    if NUM_CONVERSATIONS > 0:
        conv_ids = conv_ids[:NUM_CONVERSATIONS]

    selected_generation_tasks = [
        task
        for cid in conv_ids
        for task in conversations.get(cid, [])
    ]
    retrieval_tasks = load_retrieval_tasks(RETRIEVAL_QUERY_MODES)
    retrieval_tasks = limit_retrieval_tasks(retrieval_tasks, NUM_CONVERSATIONS)

    active_domains = sorted(
        set(active_domains_from_tasks(selected_generation_tasks))
        | {task["domain"] for task in retrieval_tasks}
    )
    if not active_domains:
        active_domains = active_domains_from_tasks(tasks)

    configure_agent_scope_filters(active_domains)
    try:
        indexed_counts = ensure_mtrag_index_ready(active_domains)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    mode = os.environ.get("AGENT_MODE", "local").lower()
    print(
        "MT-RAG Evaluator | "
        f"mode={mode} generation_mode={GENERATION_MODE} "
        f"retrieval_query_modes={','.join(RETRIEVAL_QUERY_MODES)} "
        f"(parallelism={PARALLELISM})"
    )
    print(f"Active MT-RAG domains: {', '.join(active_domains)}")
    print(
        "Indexed MT-RAG passages: "
        + ", ".join(f"{domain}={indexed_counts[domain]}" for domain in active_domains)
    )
    if mode == "http":
        print(
            "WARNING: gold-history seeding and in-process source filtering are exact only "
            "for local mode. HTTP mode falls back to transcript prompts and depends on the "
            "server using the same code/env."
        )

    client = make_client()

    print(f"Running {len(conv_ids)} conversation(s)...\n")

    sem = asyncio.Semaphore(PARALLELISM)
    generation_wall_start = time.time()
    generation_coroutines = [
        run_conversation_async(sem, client, index + 1, len(conv_ids), cid, conversations[cid])
        for index, cid in enumerate(conv_ids)
    ]
    generation_results = await asyncio.gather(*generation_coroutines)
    generation_wall_elapsed = time.time() - generation_wall_start

    all_predictions: list[dict] = []
    for predictions in generation_results:
        all_predictions.extend(predictions)

    print_usage_summary("Generation Run", all_predictions, generation_wall_elapsed)

    timestamp = int(time.time())
    predictions_path = PREDICTIONS_DIR / f"predictions_{GENERATION_MODE}_{timestamp}.jsonl"
    write_jsonl(predictions_path, all_predictions)
    print(f"\nGeneration predictions saved to {predictions_path}")

    (
        gen_all,
        gen_by_group,
        ret_all,
        ret_by_group,
        generation_summary,
    ) = evaluate_predictions(all_predictions)

    standalone_ret_all = None
    standalone_ret_by_group = None
    standalone_summary = None

    if retrieval_tasks:
        print(f"\nRunning {len(retrieval_tasks)} standalone retrieval task(s)...\n")
        retrieval_wall_start = time.time()
        retrieval_coroutines = [
            run_retrieval_task_async(sem, client, index + 1, len(retrieval_tasks), task)
            for index, task in enumerate(retrieval_tasks)
        ]
        retrieval_results = await asyncio.gather(*retrieval_coroutines)
        retrieval_results = [result for result in retrieval_results if result is not None]
        retrieval_wall_elapsed = time.time() - retrieval_wall_start
        print_usage_summary("Standalone Retrieval Run", retrieval_results, retrieval_wall_elapsed)

        retrieval_path = PREDICTIONS_DIR / f"retrieval_predictions_{timestamp}.jsonl"
        write_jsonl(retrieval_path, retrieval_results)
        print(f"\nStandalone retrieval results saved to {retrieval_path}")

        (
            standalone_ret_all,
            standalone_ret_by_group,
            standalone_summary,
        ) = evaluate_retrieval_tasks(retrieval_results)
    else:
        print("\nNo standalone retrieval tasks found.")

    print_metrics(
        gen_all,
        gen_by_group,
        ret_all,
        ret_by_group,
        generation_summary,
        standalone_ret_all,
        standalone_ret_by_group,
        standalone_summary,
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
