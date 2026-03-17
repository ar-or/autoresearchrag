#!/usr/bin/env python3
"""MT-RAG Benchmark Evaluator.

Environment variables:
  AGENT_MODE       - "local" (default) or "http"
  ORAGENT_URL      - oragent base URL when AGENT_MODE=http
  N                - max conversations, 0 = all (default: 0)
  PARALLELISM      - concurrency degree (default: 32)
  MTRAG_COLLECTION - filter by domain/collection (default: all)
"""

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_client import make_client
from cost import CostCalculator

from metrics import (
    token_f1,
    rouge_l,
    exact_match,
    compute_retrieval_metrics,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_CONVERSATIONS = int(os.environ.get("N", "0"))
PARALLELISM = int(os.environ.get("PARALLELISM", "32"))
MTRAG_COLLECTION = os.environ.get("MTRAG_COLLECTION", "")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "mt-rag-benchmark"
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


RETRIEVAL_DIR = DATA_DIR / "human" / "retrieval_tasks"
ALL_DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]


def find_data_files() -> list[Path]:
    rag_file = DATA_DIR / "human" / "generation_tasks" / "RAG.jsonl"
    if rag_file.exists():
        return [rag_file]
    patterns = ["**/*.jsonl"]
    files = []
    for p in patterns:
        files.extend(DATA_DIR.glob(p))
    return sorted(f for f in files if f.stat().st_size > 100 and "generation_tasks" in str(f))


def load_tasks(files: list[Path]) -> list[dict]:
    tasks = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                    tasks.append(task)
                except json.JSONDecodeError:
                    continue
    return tasks


def group_conversations(tasks: list[dict]) -> dict[str, list[dict]]:
    convos: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        cid = task.get("conversation_id", task.get("task_id", "unknown"))
        convos[cid].append(task)
    for cid in convos:
        convos[cid].sort(key=lambda t: int(t.get("turn", 0)))
    return dict(convos)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def get_latest_user_message(input_data) -> str:
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        for msg in reversed(input_data):
            if isinstance(msg, dict):
                if msg.get("speaker") == "user":
                    return msg.get("text", "")
                if msg.get("role") == "user":
                    return msg.get("content", msg.get("text", ""))
            if isinstance(msg, str):
                return msg
        return str(input_data[-1]) if input_data else ""
    return str(input_data)


# ---------------------------------------------------------------------------
# Gold relevant document IDs
# ---------------------------------------------------------------------------


def _hash_passage_id(raw_id: str) -> str:
    """Apply the same hash used by ingest_mtrag.py so IDs are comparable."""
    return hashlib.sha256(raw_id.encode()).hexdigest()[:16]


def get_gold_relevant_ids(task: dict) -> set[str]:
    """Return hashed IDs of gold-relevant contexts.

    Relevance is determined by the ``reference`` field (True means the context
    was used to produce the gold answer).  Contexts without ``reference: true``
    are distractors.  The returned IDs are hashed with the same scheme the
    ingester uses so they can be compared directly against retrieved IDs.
    """
    relevant: set[str] = set()
    for ctx in task.get("contexts", []):
        if not ctx.get("reference"):
            continue
        doc_id = ctx.get("document_id", ctx.get("id", ""))
        if doc_id:
            relevant.add(_hash_passage_id(doc_id))
    return relevant


# ---------------------------------------------------------------------------
# Run one conversation (sequential turns, but conversations run in parallel)
# ---------------------------------------------------------------------------


def run_conversation(client, conversation_id: str, tasks: list[dict]) -> list[dict]:
    session_id = client.create_session()
    predictions = []
    try:
        for task in tasks:
            user_text = get_latest_user_message(task.get("input", ""))
            resp = client.send_message(session_id, user_text)

            contexts_raw = [
                {
                    "document_id": c.document_id,
                    "text": c.text,
                    "title": c.title,
                    "score": c.score,
                }
                for c in resp.contexts
            ]

            prediction = {
                "task_id": task.get("task_id", ""),
                "conversation_id": conversation_id,
                "collection": task.get("Collection", task.get("collection", "")),
                "turn": task.get("turn", 0),
                "input": task.get("input"),
                "contexts": contexts_raw,
                "predictions": [{"text": resp.response}],
                "targets": task.get("targets", []),
                "gold_contexts": task.get("contexts", []),
                "tokens": {
                    "input": resp.usage.input_tokens,
                    "cached": resp.usage.cached_tokens,
                    "output": resp.usage.output_tokens,
                },
                "model": resp.model,
            }
            predictions.append(prediction)
    finally:
        client.delete_session(session_id)
    return predictions


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
            loop = asyncio.get_event_loop()
            preds = await loop.run_in_executor(
                None, run_conversation, client, cid, conv_tasks
            )
        except Exception as e:
            print(f"    [{idx}/{total}] ERROR conversation {cid}: {e}")
            return []
        elapsed = time.time() - t0
        print(f"    [{idx}/{total}] conversation {cid} ({len(conv_tasks)} turns) done in {elapsed:.1f}s")
        return preds


# ---------------------------------------------------------------------------
# Standalone retrieval tasks (BEIR format)
# ---------------------------------------------------------------------------


def load_retrieval_tasks() -> list[dict]:
    """Load BEIR-format retrieval tasks from human/retrieval_tasks/.

    Returns a list of dicts with keys: query_id, text, domain, qrels (dict of
    corpus_id -> relevance score).  Only 'lastturn' queries are loaded by
    default (closest to the agent's single-turn retrieval setting).
    """
    tasks: list[dict] = []
    for domain in ALL_DOMAINS:
        domain_dir = RETRIEVAL_DIR / domain
        if not domain_dir.exists():
            continue

        # Load qrels
        qrel_path = domain_dir / "qrels" / "dev.tsv"
        if not qrel_path.exists():
            continue
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        with open(qrel_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("query-id"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    qid, cid, score = parts[0], parts[1], int(parts[2])
                    qrels[qid][cid] = score

        # Load queries (prefer lastturn)
        query_file = domain_dir / f"{domain}_lastturn.jsonl"
        if not query_file.exists():
            continue
        with open(query_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = obj.get("_id", obj.get("id", ""))
                text = obj.get("text", "")
                # Strip speaker prefix like "|user|: "
                if text.startswith("|user|:"):
                    text = text[len("|user|:"):].strip()
                if qid and text and qid in qrels:
                    tasks.append({
                        "query_id": qid,
                        "text": text,
                        "domain": domain,
                        "qrels": dict(qrels[qid]),
                    })
    return tasks


def run_retrieval_task(client, task: dict) -> dict:
    """Run a single retrieval task and return results."""
    session_id = client.create_session()
    try:
        resp = client.send_message(session_id, task["text"])
        retrieved_ids = [c.document_id for c in resp.contexts]
        return {
            "query_id": task["query_id"],
            "domain": task["domain"],
            "retrieved_ids": retrieved_ids,
            "qrels": task["qrels"],
            "tokens": {
                "input": resp.usage.input_tokens,
                "cached": resp.usage.cached_tokens,
                "output": resp.usage.output_tokens,
            },
            "model": resp.model,
        }
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
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, run_retrieval_task, client, task
            )
        except Exception as e:
            print(f"    [{idx}/{total}] ERROR retrieval {task['query_id']}: {e}")
            return None
        elapsed = time.time() - t0
        print(f"    [{idx}/{total}] retrieval {task['domain']}:{task['query_id'][:20]} done in {elapsed:.1f}s")
        return result


def evaluate_retrieval_tasks(results: list[dict]):
    """Evaluate standalone retrieval results against qrels.

    The qrels corpus-IDs are raw benchmark IDs; retrieved IDs from the agent
    are hashed.  We hash the qrels IDs to make them comparable.
    """
    ret_metrics_all: dict[str, list[float]] = defaultdict(list)
    ret_metrics_by_domain: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for result in results:
        # Hash qrel corpus IDs to match indexed IDs
        gold_relevant = {_hash_passage_id(cid) for cid, score in result["qrels"].items() if score > 0}
        retrieved_ids = result["retrieved_ids"]
        if gold_relevant and retrieved_ids:
            ret = compute_retrieval_metrics(retrieved_ids, gold_relevant)
            domain = result["domain"]
            for k, v in ret.items():
                ret_metrics_all[k].append(v)
                ret_metrics_by_domain[domain][k].append(v)

    return ret_metrics_all, ret_metrics_by_domain


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_predictions(all_predictions: list[dict]):
    gen_metrics_by_collection = defaultdict(lambda: defaultdict(list))
    ret_metrics_by_collection = defaultdict(lambda: defaultdict(list))
    gen_metrics_all = defaultdict(list)
    ret_metrics_all = defaultdict(list)

    for pred in all_predictions:
        collection = pred.get("collection", "unknown")
        pred_text = pred["predictions"][0]["text"] if pred["predictions"] else ""
        targets = pred.get("targets", [])
        if targets:
            t0_target = targets[0]
            ref_text = t0_target.get("text", str(t0_target)) if isinstance(t0_target, dict) else str(t0_target)
            f1 = token_f1(pred_text, ref_text)
            rl = rouge_l(pred_text, ref_text)
            em = exact_match(pred_text, ref_text)
            for store in (gen_metrics_all, gen_metrics_by_collection[collection]):
                store["f1"].append(f1)
                store["rouge_l"].append(rl)
                store["exact_match"].append(em)

        retrieved_ids = [c.get("document_id", "") for c in pred.get("contexts", []) if c.get("document_id")]
        gold_pred = {"contexts": pred.get("gold_contexts", [])}
        gold_relevant = get_gold_relevant_ids(gold_pred)
        if gold_relevant and retrieved_ids:
            ret = compute_retrieval_metrics(retrieved_ids, gold_relevant)
            for k, v in ret.items():
                ret_metrics_all[k].append(v)
                ret_metrics_by_collection[collection][k].append(v)

    return gen_metrics_all, gen_metrics_by_collection, ret_metrics_all, ret_metrics_by_collection


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_metric_block(title: str, metrics_all: dict, metrics_by_group: dict):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if metrics_all:
        for k in sorted(metrics_all.keys()):
            print(f"  {k:>12s}: {avg(metrics_all[k]):.4f}  (n={len(metrics_all[k])})")
        for group in sorted(metrics_by_group.keys()):
            print(f"\n  [{group}]")
            for k in sorted(metrics_by_group[group].keys()):
                print(f"    {k:>12s}: {avg(metrics_by_group[group][k]):.4f}  (n={len(metrics_by_group[group][k])})")
    else:
        print("  No results to evaluate.")


def print_metrics(gen_all, gen_by_col, ret_all, ret_by_col, standalone_ret_all=None, standalone_ret_by_domain=None):
    _print_metric_block("RETRIEVAL METRICS (generation tasks)", ret_all, ret_by_col)
    if standalone_ret_all is not None:
        _print_metric_block("RETRIEVAL METRICS (standalone retrieval tasks)", standalone_ret_all, standalone_ret_by_domain or {})
    _print_metric_block("GENERATION METRICS", gen_all, gen_by_col)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main():
    client = make_client()
    mode = os.environ.get("AGENT_MODE", "local")

    print(f"MT-RAG Evaluator | mode={mode}  (parallelism={PARALLELISM})")

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
    print(f"Loaded {len(tasks)} tasks")

    if MTRAG_COLLECTION:
        tasks = [t for t in tasks if t.get("Collection", t.get("collection", "")) == MTRAG_COLLECTION]
        print(f"Filtered to {len(tasks)} tasks for collection '{MTRAG_COLLECTION}'")

    conversations = group_conversations(tasks)
    conv_ids = sorted(conversations.keys())
    if NUM_CONVERSATIONS > 0:
        conv_ids = conv_ids[:NUM_CONVERSATIONS]
    print(f"Running {len(conv_ids)} conversation(s)...\n")

    sem = asyncio.Semaphore(PARALLELISM)
    wall_start = time.time()

    coros = [
        run_conversation_async(sem, client, i + 1, len(conv_ids), cid, conversations[cid])
        for i, cid in enumerate(conv_ids)
    ]
    results = await asyncio.gather(*coros)

    wall_elapsed = time.time() - wall_start

    all_predictions = []
    for preds in results:
        all_predictions.extend(preds)

    # Aggregate token usage
    total_tokens = {"input": 0, "cached": 0, "output": 0}
    model_name = "unknown"
    for pred in all_predictions:
        tokens = pred.get("tokens", {})
        total_tokens["input"] += tokens.get("input", 0)
        total_tokens["cached"] += tokens.get("cached", 0)
        total_tokens["output"] += tokens.get("output", 0)
        m = pred.get("model", "unknown")
        if m != "unknown":
            model_name = m

    n_turns = len(all_predictions) or 1
    cost_calc = CostCalculator(model_name)
    total_cost = cost_calc.cost(total_tokens["input"], total_tokens["cached"], total_tokens["output"])

    print(f"\n  Model:              {model_name}")
    print(f"  Wall clock:         {wall_elapsed:.1f}s  (parallelism={PARALLELISM})")
    print(f"  Turns evaluated:    {len(all_predictions)}")
    print()
    print(f"  Token usage:        input      cached     output")
    print(f"    Total:            {total_tokens['input']:>9}  {total_tokens['cached']:>9}  {total_tokens['output']:>9}")
    print(f"    Avg/turn:         {total_tokens['input']//n_turns:>9}  {total_tokens['cached']//n_turns:>9}  {total_tokens['output']//n_turns:>9}")
    print()
    print(f"  Pricing (per 1M tokens):  {cost_calc.format_pricing_line()}")
    print(f"  Total cost:         ${total_cost:.6f}")
    print(f"  Avg cost/turn:      ${total_cost/n_turns:.6f}")

    out_path = PREDICTIONS_DIR / f"predictions_{int(time.time())}.jsonl"
    with open(out_path, "w") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\nPredictions saved to {out_path}")

    gen_all, gen_by_col, ret_all, ret_by_col = evaluate_predictions(all_predictions)

    # Run standalone retrieval tasks
    standalone_ret_all = None
    standalone_ret_by_domain = None
    retrieval_tasks = load_retrieval_tasks()
    if retrieval_tasks:
        if MTRAG_COLLECTION:
            retrieval_tasks = [t for t in retrieval_tasks if t["domain"] == MTRAG_COLLECTION]
        if NUM_CONVERSATIONS > 0:
            retrieval_tasks = retrieval_tasks[:NUM_CONVERSATIONS]
        print(f"\nRunning {len(retrieval_tasks)} standalone retrieval task(s)...\n")
        ret_coros = [
            run_retrieval_task_async(sem, client, i + 1, len(retrieval_tasks), t)
            for i, t in enumerate(retrieval_tasks)
        ]
        ret_results = await asyncio.gather(*ret_coros)
        ret_results = [r for r in ret_results if r is not None]
        standalone_ret_all, standalone_ret_by_domain = evaluate_retrieval_tasks(ret_results)
    else:
        print("\nNo standalone retrieval tasks found.")

    print_metrics(gen_all, gen_by_col, ret_all, ret_by_col, standalone_ret_all, standalone_ret_by_domain)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
