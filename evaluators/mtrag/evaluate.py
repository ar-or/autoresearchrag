#!/usr/bin/env python3
"""MT-RAG Benchmark Evaluator for oragent.

Environment variables:
  ORAGENT_URL      - oragent base URL (default: http://localhost:32522)
  MTRAG_N          - max conversations, 0 = all (default: 0)
  MTRAG_COLLECTION - filter by domain/collection (default: all)
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from metrics import (
    token_f1,
    rouge_l,
    exact_match,
    compute_retrieval_metrics,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ORAGENT_URL = os.environ.get("ORAGENT_URL", "http://localhost:32522")
MTRAG_N = int(os.environ.get("MTRAG_N", "0"))
MTRAG_COLLECTION = os.environ.get("MTRAG_COLLECTION", "")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "mt-rag-benchmark"
PREDICTIONS_DIR = SCRIPT_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def find_data_files() -> list[Path]:
    """Find the RAG generation task JSONL file."""
    rag_file = DATA_DIR / "human" / "generation_tasks" / "RAG.jsonl"
    if rag_file.exists():
        return [rag_file]
    patterns = ["**/*.jsonl"]
    files = []
    for p in patterns:
        files.extend(DATA_DIR.glob(p))
    return sorted(f for f in files if f.stat().st_size > 100 and "generation_tasks" in str(f))


def load_tasks(files: list[Path]) -> list[dict]:
    """Load all tasks from JSONL files."""
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
    """Group tasks by conversation_id, sorted by turn number."""
    convos = defaultdict(list)
    for task in tasks:
        cid = task.get("conversation_id", task.get("task_id", "unknown"))
        convos[cid].append(task)
    # Sort each conversation by turn
    for cid in convos:
        convos[cid].sort(key=lambda t: t.get("turn", 0))
    return dict(convos)


# ---------------------------------------------------------------------------
# oragent API helpers
# ---------------------------------------------------------------------------


def create_session() -> str:
    r = requests.post(f"{ORAGENT_URL}/api/chat/session")
    r.raise_for_status()
    return r.json()["session_id"]


def send_message(session_id: str, message: str) -> dict:
    r = requests.post(
        f"{ORAGENT_URL}/api/chat/send-message",
        json={"session_id": session_id, "message": message},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def delete_session(session_id: str):
    try:
        requests.delete(f"{ORAGENT_URL}/api/chat/session/{session_id}", timeout=10)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def get_latest_user_message(input_data) -> str:
    """Extract the latest user message from the task input."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        # Multi-turn: list of messages with speaker/text format
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


def build_prompt(user_text: str) -> str:
    return user_text


# ---------------------------------------------------------------------------
# Gold relevant document IDs
# ---------------------------------------------------------------------------


def get_gold_relevant_ids(task: dict) -> set[str]:
    """Extract IDs of relevant documents from task gold contexts."""
    relevant = set()
    for ctx in task.get("contexts", []):
        feedback = ctx.get("feedback", {})
        if isinstance(feedback, dict) and feedback.get("relevant", "").lower() == "yes":
            doc_id = ctx.get("document_id", ctx.get("id", ""))
            if doc_id:
                relevant.add(doc_id)
    # If no feedback info, treat all contexts as relevant
    if not relevant:
        for ctx in task.get("contexts", []):
            doc_id = ctx.get("document_id", ctx.get("id", ""))
            if doc_id:
                relevant.add(doc_id)
    return relevant


# ---------------------------------------------------------------------------
# Run one conversation
# ---------------------------------------------------------------------------


def run_conversation(conversation_id: str, tasks: list[dict]) -> list[dict]:
    session_id = create_session()
    predictions = []
    try:
        for task in tasks:
            user_text = get_latest_user_message(task.get("input", ""))
            prompt = build_prompt(user_text)

            response = send_message(session_id, prompt)
            response_text = response.get("response", "")
            contexts = response.get("contexts", [])

            prediction = {
                "task_id": task.get("task_id", ""),
                "conversation_id": conversation_id,
                "collection": task.get("Collection", task.get("collection", "")),
                "turn": task.get("turn", 0),
                "input": task.get("input"),
                "contexts": contexts,
                "predictions": [{"text": response_text}],
                "targets": task.get("targets", []),
                "gold_contexts": task.get("contexts", []),
            }
            predictions.append(prediction)
    finally:
        delete_session(session_id)
    return predictions


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
        # Generation metrics
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

        # Retrieval metrics
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


def print_metrics(gen_all, gen_by_col, ret_all, ret_by_col):
    print("\n" + "=" * 60)
    print("RETRIEVAL METRICS")
    print("=" * 60)
    if ret_all:
        for k in sorted(ret_all.keys()):
            print(f"  {k:>12s}: {avg(ret_all[k]):.4f}  (n={len(ret_all[k])})")
        for col in sorted(ret_by_col.keys()):
            print(f"\n  [{col}]")
            for k in sorted(ret_by_col[col].keys()):
                print(f"    {k:>12s}: {avg(ret_by_col[col][k]):.4f}  (n={len(ret_by_col[col][k])})")
    else:
        print("  No retrieval results to evaluate.")

    print("\n" + "=" * 60)
    print("GENERATION METRICS")
    print("=" * 60)
    if gen_all:
        for k in sorted(gen_all.keys()):
            print(f"  {k:>12s}: {avg(gen_all[k]):.4f}  (n={len(gen_all[k])})")
        for col in sorted(gen_by_col.keys()):
            print(f"\n  [{col}]")
            for k in sorted(gen_by_col[col].keys()):
                print(f"    {k:>12s}: {avg(gen_by_col[col][k]):.4f}  (n={len(gen_by_col[col][k])})")
    else:
        print("  No generation results to evaluate.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"MT-RAG Evaluator | url={ORAGENT_URL}")

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
    if MTRAG_N > 0:
        conv_ids = conv_ids[:MTRAG_N]
    print(f"Running {len(conv_ids)} conversation(s)...")

    all_predictions = []
    for i, cid in enumerate(conv_ids, 1):
        conv_tasks = conversations[cid]
        print(f"  [{i}/{len(conv_ids)}] conversation {cid} ({len(conv_tasks)} turns)")
        t0 = time.time()
        try:
            preds = run_conversation(cid, conv_tasks)
            all_predictions.extend(preds)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s")

    # Save predictions
    out_path = PREDICTIONS_DIR / f"predictions_{int(time.time())}.jsonl"
    with open(out_path, "w") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\nPredictions saved to {out_path}")

    # Evaluate
    gen_all, gen_by_col, ret_all, ret_by_col = evaluate_predictions(all_predictions)
    print_metrics(gen_all, gen_by_col, ret_all, ret_by_col)


if __name__ == "__main__":
    main()
