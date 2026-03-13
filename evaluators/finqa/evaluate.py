#!/usr/bin/env python3
"""Evaluate oragent on FinQA – financial numerical reasoning.

Each FinQA example contains a financial report (pre_text + table + post_text)
and a question requiring numerical reasoning.  The gold answer is a number
stored in qa.exe_ans.  We ask the agent to answer and compare numerically.

Environment variables:
  ORAGENT_URL    - oragent base URL (default: http://localhost:32522)
  FINQA_N        - max examples to evaluate (default: 20)
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

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("ORAGENT_URL", "http://localhost:32522")
DATA_PATH = Path(__file__).resolve().parent / "data" / "dev.json"
NUM_EXAMPLES = int(os.environ.get("FINQA_N", "20"))
PARALLELISM = int(os.environ.get("FINQA_PARALLEL", "32"))
PREDICTIONS_DIR = Path(__file__).resolve().parent / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Relative tolerance for numeric comparison (1e-3 = 0.1 %)
REL_TOL = 1e-3
ABS_TOL = 1e-6

# Pricing per million tokens (gpt-5-mini defaults, overridable)
PRICE_INPUT = float(os.environ.get("PRICE_INPUT_PER_M", "0.25"))
PRICE_CACHED = float(os.environ.get("PRICE_CACHED_PER_M", "0.025"))
PRICE_OUTPUT = float(os.environ.get("PRICE_OUTPUT_PER_M", "2.00"))


def calc_cost(inp: int, cached: int, out: int) -> float:
    uncached = inp - cached
    return (uncached * PRICE_INPUT + cached * PRICE_CACHED + out * PRICE_OUTPUT) / 1_000_000


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def format_table(table: list[list[str]]) -> str:
    """Render the table as a markdown-style table."""
    if not table:
        return ""
    header = table[0]
    rows = table[1:]
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    lines = [fmt.format(*[str(c) for c in header])]
    lines.append(" | ".join("-" * w for w in col_widths))
    for row in rows:
        lines.append(fmt.format(*[str(c) for c in row]))
    return "\n".join(lines)


def build_prompt(example: dict) -> str:
    """Build a prompt from a FinQA example."""
    parts: list[str] = []
    parts.append(
        "You are given a financial report excerpt with text and a table. "
        "Answer the question with a single number (no units, no symbols, no words). "
        "If the answer is a percentage, express it as a decimal (e.g. 0.25 for 25%).\n"
    )

    pre = example.get("pre_text", [])
    if pre:
        parts.append("### Report text (before table)")
        parts.append(" ".join(pre))
        parts.append("")

    table = example.get("table", [])
    if table:
        parts.append("### Table")
        parts.append(format_table(table))
        parts.append("")

    post = example.get("post_text", [])
    if post:
        parts.append("### Report text (after table)")
        parts.append(" ".join(post))
        parts.append("")

    question = example["qa"]["question"]
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append("Respond with ONLY the numeric answer (a single number, nothing else).")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer parsing & comparison
# ---------------------------------------------------------------------------


def parse_number(text: str) -> float | None:
    """Extract a number from the agent response."""
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
    """Check if predicted answer matches gold within tolerance."""
    if math.isclose(predicted, gold, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True
    if abs(gold) < 10 and math.isclose(predicted, gold * 100, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True
    if abs(predicted) < 10 and math.isclose(predicted * 100, gold, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True
    return False


# ---------------------------------------------------------------------------
# Async oragent API
# ---------------------------------------------------------------------------


async def query_agent(prompt: str, retries: int = 3) -> dict:
    """Send a question to the agent and return the full response JSON."""
    timeout = aiohttp.ClientTimeout(total=180)
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(f"{BASE_URL}/api/chat/session") as r:
                    r.raise_for_status()
                    data = await r.json()
                    session_id = data["session_id"]
                try:
                    async with http.post(
                        f"{BASE_URL}/api/chat/send-message",
                        json={"session_id": session_id, "message": prompt},
                    ) as r:
                        r.raise_for_status()
                        return await r.json()
                finally:
                    try:
                        async with http.delete(f"{BASE_URL}/api/chat/session/{session_id}"):
                            pass
                    except Exception:
                        pass
        except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError, asyncio.TimeoutError):
            if attempt < retries - 1:
                await asyncio.sleep(3)
            else:
                raise


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def evaluate_one(
    sem: asyncio.Semaphore,
    idx: int,
    total_count: int,
    ex: dict,
) -> dict:
    """Evaluate a single example, respecting the semaphore for concurrency."""
    qid = ex["id"]
    gold_ans = ex["qa"]["exe_ans"]
    try:
        gold_float = float(gold_ans) if gold_ans is not None else None
    except (ValueError, TypeError):
        gold_float = None  # non-numeric gold answer (e.g. "yes"/"no")
    prompt = build_prompt(ex)

    async with sem:
        t0 = time.time()
        resp_json = {}
        raw_response = ""
        usage = {}
        model = "unknown"
        try:
            resp_json = await query_agent(prompt)
            raw_response = resp_json.get("response", "")
            usage = resp_json.get("usage", {})
            model = resp_json.get("model", "unknown")
        except Exception as e:
            raw_response = ""
            print(f"  [{idx+1}/{total_count}] ERROR: {e}")
        elapsed = time.time() - t0

    q_input = usage.get("input_tokens", 0)
    q_cached = usage.get("cached_tokens", 0)
    q_output = usage.get("output_tokens", 0)

    predicted = parse_number(raw_response)
    match = False
    if predicted is not None and gold_float is not None:
        match = answers_match(predicted, gold_float)

    status = "PASS" if match else "FAIL"
    q_cost = calc_cost(q_input, q_cached, q_output)
    print(
        f"  [{idx+1}/{total_count}] {status}  predicted={predicted}  gold={gold_float}  "
        f"duration={elapsed:.1f}s  tokens={q_input}/{q_cached}/{q_output}  "
        f"cost=${q_cost:.6f}  {ex['qa']['question'][:60]}"
    )

    return {
        "id": qid,
        "question": ex["qa"]["question"],
        "gold_answer": gold_ans,
        "predicted_raw": raw_response[:200],
        "predicted_number": predicted,
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

    print(f"Loading data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    examples = data[:NUM_EXAMPLES]
    n = len(examples)
    print(f"Evaluating {n} examples against {BASE_URL}  (parallelism={PARALLELISM})\n")

    sem = asyncio.Semaphore(PARALLELISM)
    wall_start = time.time()

    tasks = [
        evaluate_one(sem, i, n, ex)
        for i, ex in enumerate(examples)
    ]
    results = await asyncio.gather(*tasks)

    wall_elapsed = time.time() - wall_start

    # Aggregate
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
    total_cost = calc_cost(total_tokens["input"], total_tokens["cached"], total_tokens["output"])

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
    print(f"  Pricing (per 1M tokens):  input=${PRICE_INPUT}  cached=${PRICE_CACHED}  output=${PRICE_OUTPUT}")
    print(f"  Total cost:         ${total_cost:.6f}")
    print(f"  Avg cost/question:  ${total_cost/n:.6f}")
    print("=" * 70)

    # Save predictions
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
