#!/usr/bin/env python3
"""Evaluate agent on HotpotQA dev distractor (first N examples).

Environment variables:
  AGENT_MODE  - "local" (default) or "http"
  ORAGENT_URL - oragent base URL when AGENT_MODE=http
  N           - max examples to evaluate (default: 20)
  PARALLELISM - concurrency degree (default: 32)
"""

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_client import make_client
from cost import CostCalculator
from hotpot_evaluate_v1 import update_answer, update_sp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent / "data" / "hotpot_dev_distractor_v1.json"
NUM_EXAMPLES = int(os.environ.get("N", "20"))
PARALLELISM = int(os.environ.get("PARALLELISM", "32"))


# ---------------------------------------------------------------------------
# Prompt building & parsing
# ---------------------------------------------------------------------------


def build_prompt(example: dict) -> str:
    """Send only the question — the agent must retrieve context itself."""
    question = example["question"]
    return (
        f"{question}\n\n"
        "Respond with ONLY a JSON object (no markdown, no explanation):\n"
        '{"answer": "<your answer>", "supporting_facts": [["<title>", <sentence_index>], ...]}'
    )


def parse_response(text: str) -> tuple[str, list[list]]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
        answer = str(data.get("answer", ""))
        sp = data.get("supporting_facts", [])
        cleaned_sp = []
        for item in sp:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cleaned_sp.append([str(item[0]), int(item[1])])
        return answer, cleaned_sp
    except (json.JSONDecodeError, ValueError, TypeError):
        return text.strip(), []


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def query_agent(client, prompt: str) -> dict:
    session_id = client.create_session()
    try:
        resp = client.send_message(session_id, prompt)
        return {
            "response": resp.response,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "cached_tokens": resp.usage.cached_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            "model": resp.model,
        }
    finally:
        client.delete_session(session_id)


async def evaluate_one(
    sem: asyncio.Semaphore,
    client,
    idx: int,
    total: int,
    ex: dict,
) -> dict:
    qid = ex["_id"]
    prompt = build_prompt(ex)

    async with sem:
        t0 = time.time()
        usage: dict = {}
        model = "unknown"
        try:
            loop = asyncio.get_event_loop()
            resp_json = await loop.run_in_executor(None, query_agent, client, prompt)
            raw_response = resp_json.get("response", "")
            usage = resp_json.get("usage", {})
            model = resp_json.get("model", "unknown")
            answer, sp = parse_response(raw_response)
        except Exception as e:
            print(f"  [{idx+1}/{total}] ERROR: {e}")
            answer, sp = "", []
        elapsed = time.time() - t0

    q_input = usage.get("input_tokens", 0)
    q_cached = usage.get("cached_tokens", 0)
    q_output = usage.get("output_tokens", 0)
    cost_calc = CostCalculator(model)
    q_cost = cost_calc.cost(q_input, q_cached, q_output)

    print(
        f"  [{idx+1}/{total}] {elapsed:.1f}s  answer={answer[:80]}  sp={len(sp)}  "
        f"tokens={q_input}/{q_cached}/{q_output}  cost=${q_cost:.6f}  {ex['question'][:60]}"
    )

    return {
        "qid": qid,
        "answer": answer,
        "sp": sp,
        "gold": ex,
        "duration_s": round(elapsed, 2),
        "tokens": {"input": q_input, "cached": q_cached, "output": q_output},
        "cost_usd": round(q_cost, 8),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main():
    if not DATA_PATH.exists():
        print(f"ERROR: Data not found at {DATA_PATH}")
        sys.exit(1)

    client = make_client()
    mode = os.environ.get("AGENT_MODE", "local")

    print(f"Loading data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    examples = data[:NUM_EXAMPLES]
    n = len(examples)
    print(f"Evaluating {n} examples  mode={mode}  (parallelism={PARALLELISM})\n")

    sem = asyncio.Semaphore(PARALLELISM)
    wall_start = time.time()

    tasks = [evaluate_one(sem, client, i, n, ex) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks)

    wall_elapsed = time.time() - wall_start

    # Collect predictions & token stats
    predictions: dict = {"answer": {}, "sp": {}}
    gold_list: list[dict] = []
    total_tokens = {"input": 0, "cached": 0, "output": 0}
    model_name = "unknown"
    for r in results:
        predictions["answer"][r["qid"]] = r["answer"]
        predictions["sp"][r["qid"]] = r["sp"]
        gold_list.append(r["gold"])
        total_tokens["input"] += r["tokens"]["input"]
        total_tokens["cached"] += r["tokens"]["cached"]
        total_tokens["output"] += r["tokens"]["output"]
        if r["model"] != "unknown":
            model_name = r["model"]

    cost_calc = CostCalculator(model_name)
    total_cost = cost_calc.cost(total_tokens["input"], total_tokens["cached"], total_tokens["output"])

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Model:              {model_name}")
    print(f"  Wall clock:         {wall_elapsed:.1f}s  (parallelism={PARALLELISM})")

    metrics: dict[str, float] = {
        "em": 0, "f1": 0, "prec": 0, "recall": 0,
        "sp_em": 0, "sp_f1": 0, "sp_prec": 0, "sp_recall": 0,
        "joint_em": 0, "joint_f1": 0, "joint_prec": 0, "joint_recall": 0,
    }

    for dp in gold_list:
        cur_id = dp["_id"]
        can_eval_joint = True

        if cur_id not in predictions["answer"]:
            print(f"missing answer {cur_id}")
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, predictions["answer"][cur_id], dp["answer"]
            )

        if cur_id not in predictions["sp"]:
            print(f"missing sp fact {cur_id}")
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, predictions["sp"][cur_id], dp["supporting_facts"]
            )

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.0
            joint_em = em * sp_em
            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    for k in metrics:
        metrics[k] /= n

    print(f"\n{'Metric':<20} {'Score':>8}")
    print("-" * 30)
    print(f"{'Answer EM':<20} {metrics['em']:>8.4f}")
    print(f"{'Answer F1':<20} {metrics['f1']:>8.4f}")
    print(f"{'SP EM':<20} {metrics['sp_em']:>8.4f}")
    print(f"{'SP F1':<20} {metrics['sp_f1']:>8.4f}")
    print(f"{'Joint EM':<20} {metrics['joint_em']:>8.4f}")
    print(f"{'Joint F1':<20} {metrics['joint_f1']:>8.4f}")

    print()
    print(f"  Token usage:        input      cached     output")
    print(f"    Total:            {total_tokens['input']:>9}  {total_tokens['cached']:>9}  {total_tokens['output']:>9}")
    print(f"    Avg/question:     {total_tokens['input']//n:>9}  {total_tokens['cached']//n:>9}  {total_tokens['output']//n:>9}")
    print()
    print(f"  Pricing (per 1M tokens):  {cost_calc.format_pricing_line()}")
    print(f"  Total cost:         ${total_cost:.6f}")
    print(f"  Avg cost/question:  ${total_cost/n:.6f}")
    print("=" * 70)

    pred_path = Path(__file__).resolve().parent / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {pred_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
