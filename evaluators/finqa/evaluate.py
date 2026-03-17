#!/usr/bin/env python3
"""Evaluate agent on FinQA – financial numerical reasoning.

Environment variables:
  AGENT_MODE     - "local" (default) or "http"
  ORAGENT_URL    - oragent base URL when AGENT_MODE=http
  N              - max examples to evaluate (default: 20)
  FINQA_PARALLEL - concurrency degree (default: 32)
"""

import asyncio
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_client import make_client
from cost import CostCalculator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent / "data" / "dev.json"
NUM_EXAMPLES = int(os.environ.get("N", "20"))
PARALLELISM = int(os.environ.get("FINQA_PARALLEL", "32"))
EXAMPLE_TIMEOUT_S = float(os.environ.get("FINQA_EXAMPLE_TIMEOUT_S", "900"))
MAX_RETRIES = int(os.environ.get("FINQA_MAX_RETRIES", "2"))
PREDICTIONS_DIR = Path(__file__).resolve().parent / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

REL_TOL = 1e-3
ABS_TOL = 1e-6


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_prompt(example: dict) -> str:
    """Send only the question — the agent must retrieve context itself."""
    question = example["qa"]["question"]
    return (
        f"{question}\n\n"
        "Respond with ONLY the answer. "
        "If the answer is numeric, give a single number (no units, no symbols, no words). "
        "If the answer is a percentage, express it as a decimal (e.g. 0.25 for 25%). "
        "If the answer is yes or no, respond with only 'yes' or 'no'."
    )


# ---------------------------------------------------------------------------
# Answer parsing & comparison
# ---------------------------------------------------------------------------


def parse_boolean(text: str) -> str | None:
    """Return 'yes' or 'no' if the text is a boolean answer, else None."""
    cleaned = text.strip().lower()
    cleaned = re.sub(r"```[^`]*```", "", cleaned, flags=re.DOTALL).strip()
    # Check first meaningful word
    first_word = cleaned.split()[0] if cleaned.split() else ""
    first_word = first_word.rstrip(".,!;:")
    if first_word in ("yes", "no"):
        return first_word
    return None


def parse_number(text: str) -> float | None:
    text = text.strip()
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = text.replace(",", "").replace("$", "").replace("%", "")
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def answers_match(predicted: float, gold: float) -> bool:
    if math.isclose(predicted, gold, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True
    if abs(gold) < 10 and math.isclose(predicted, gold * 100, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True
    if abs(predicted) < 10 and math.isclose(predicted * 100, gold, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True
    return False


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
    total_count: int,
    ex: dict,
) -> dict:
    qid = ex["id"]
    gold_ans = ex["qa"]["exe_ans"]
    gold_is_bool = isinstance(gold_ans, str) and gold_ans.lower() in ("yes", "no")
    gold_bool = gold_ans.lower() if gold_is_bool else None
    try:
        gold_float = float(gold_ans) if gold_ans is not None and not gold_is_bool else None
    except (ValueError, TypeError):
        gold_float = None
    prompt = build_prompt(ex)

    async with sem:
        t0 = time.time()
        resp_json: dict = {}
        raw_response = ""
        usage: dict = {}
        model = "unknown"
        loop = asyncio.get_event_loop()
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                resp_json = await loop.run_in_executor(None, query_agent, client, prompt)
                raw_response = resp_json.get("response", "")
                usage = resp_json.get("usage", {})
                model = resp_json.get("model", "unknown")
                break
            except TimeoutError as e:
                raw_response = ""
                print(
                    f"  [{idx+1}/{total_count}] TIMEOUT attempt={attempt}/{MAX_RETRIES + 1}: {e}"
                )
                if attempt == MAX_RETRIES + 1:
                    break
            except Exception as e:
                raw_response = ""
                print(
                    f"  [{idx+1}/{total_count}] ERROR attempt={attempt}/{MAX_RETRIES + 1}: {e}"
                )
                if attempt == MAX_RETRIES + 1:
                    break
        elapsed = time.time() - t0

    q_input = usage.get("input_tokens", 0)
    q_cached = usage.get("cached_tokens", 0)
    q_output = usage.get("output_tokens", 0)

    match = False
    predicted = None
    if gold_is_bool:
        pred_bool = parse_boolean(raw_response)
        predicted = pred_bool
        match = pred_bool == gold_bool
    else:
        predicted = parse_number(raw_response)
        if predicted is not None and gold_float is not None:
            match = answers_match(predicted, gold_float)

    status = "PASS" if match else "FAIL"
    cost_calc = CostCalculator(model)
    q_cost = cost_calc.cost(q_input, q_cached, q_output)
    gold_display = gold_bool if gold_is_bool else gold_float
    print(
        f"  [{idx+1}/{total_count}] {status}  predicted={predicted}  gold={gold_display}  "
        f"duration={elapsed:.1f}s  tokens={q_input}/{q_cached}/{q_output}  "
        f"cost=${q_cost:.6f}  {ex['qa']['question'][:60]}"
    )

    return {
        "id": qid,
        "question": ex["qa"]["question"],
        "gold_answer": gold_ans,
        "predicted_raw": raw_response[:200],
        "predicted_parsed": predicted,
        "correct": match,
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
        print("Run download_data.sh first.")
        sys.exit(1)

    os.environ.setdefault("LOCAL_AGENT_TIMEOUT_S", str(EXAMPLE_TIMEOUT_S))
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

    correct = sum(1 for r in results if r["correct"])
    total_tokens = {"input": 0, "cached": 0, "output": 0}
    sum_duration = 0.0
    model_name = "unknown"
    for r in results:
        total_tokens["input"] += r["tokens"]["input"]
        total_tokens["cached"] += r["tokens"]["cached"]
        total_tokens["output"] += r["tokens"]["output"]
        sum_duration += r["duration_s"]
        if r["model"] != "unknown":
            model_name = r["model"]

    accuracy = correct / n if n > 0 else 0.0
    cost_calc = CostCalculator(model_name)
    total_cost = cost_calc.cost(total_tokens["input"], total_tokens["cached"], total_tokens["output"])

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Model:              {model_name}")
    print(f"  Examples:           {n}")
    print(f"  Correct:            {correct}")
    print(f"  Execution Accuracy: {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print()
    print(f"  Wall clock:         {wall_elapsed:.1f}s")
    print(f"  Sum of durations:   {sum_duration:.1f}s  (avg {sum_duration/n:.1f}s per question)")
    print(f"  Parallelism:        {PARALLELISM}")
    print()
    print(f"  Token usage:        input      cached     output")
    print(f"    Total:            {total_tokens['input']:>9}  {total_tokens['cached']:>9}  {total_tokens['output']:>9}")
    print(f"    Avg/question:     {total_tokens['input']//n:>9}  {total_tokens['cached']//n:>9}  {total_tokens['output']//n:>9}")
    print()
    print(f"  Pricing (per 1M tokens):  {cost_calc.format_pricing_line()}")
    print(f"  Total cost:         ${total_cost:.6f}")
    print(f"  Avg cost/question:  ${total_cost/n:.6f}")
    print("=" * 70)

    out_path = PREDICTIONS_DIR / f"predictions_{int(time.time())}.json"
    summary = {
        "model": model_name,
        "accuracy": accuracy,
        "wall_clock_s": round(wall_elapsed, 2),
        "total_cost_usd": round(total_cost, 8),
        "total_tokens": total_tokens,
        "predictions": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPredictions saved to {out_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
