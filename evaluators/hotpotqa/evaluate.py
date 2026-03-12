#!/usr/bin/env python3
"""Evaluate oragent on HotpotQA dev distractor (first N examples)."""

import json
import os
import re
import sys
import time

import requests

from hotpot_evaluate_v1 import update_answer, update_sp

BASE_URL = os.environ.get("ORAGENT_URL", "http://localhost:32522")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "hotpot_dev_distractor_v1.json")
NUM_EXAMPLES = int(os.environ.get("HOTPOTQA_N", "20"))


def build_prompt(example):
    """Build prompt with context paragraphs and question."""
    lines = []
    lines.append("Given the following paragraphs, answer the question and identify the supporting facts.\n")
    for title, sentences in example["context"]:
        lines.append(f"### {title}")
        for i, sent in enumerate(sentences):
            lines.append(f"[{i}] {sent}")
        lines.append("")
    lines.append(f"Question: {example['question']}\n")
    lines.append('Respond with ONLY a JSON object (no markdown, no explanation):')
    lines.append('{"answer": "<your answer>", "supporting_facts": [["<title>", <sentence_index>], ...]}')
    return "\n".join(lines)


def parse_response(text):
    """Extract JSON from agent response, handling markdown code blocks."""
    # Try to find JSON in code blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        # Try to find a JSON object directly
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
        answer = str(data.get("answer", ""))
        sp = data.get("supporting_facts", [])
        # Ensure sp is list of [str, int] pairs
        cleaned_sp = []
        for item in sp:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                cleaned_sp.append([str(item[0]), int(item[1])])
        return answer, cleaned_sp
    except (json.JSONDecodeError, ValueError, TypeError):
        return text.strip(), []


def query_agent(prompt, retries=3):
    """Send a question to the agent and get the response."""
    for attempt in range(retries):
        try:
            # Create session
            resp = requests.post(f"{BASE_URL}/api/chat/session", timeout=10)
            resp.raise_for_status()
            session_id = resp.json()["session_id"]

            try:
                # Send message
                resp = requests.post(
                    f"{BASE_URL}/api/chat/send-message",
                    json={"session_id": session_id, "message": prompt},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json().get("response", resp.text)
            finally:
                # Clean up session
                requests.delete(f"{BASE_URL}/api/chat/session/{session_id}", timeout=10)
        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                print(f"  Connection failed, retrying in 5s (attempt {attempt+1}/{retries})...")
                time.sleep(5)
            else:
                raise


def main():
    # Load dataset
    print(f"Loading data from {DATA_PATH}")
    with open(DATA_PATH) as f:
        data = json.load(f)

    examples = data[:NUM_EXAMPLES]
    print(f"Evaluating {len(examples)} examples against {BASE_URL}\n")

    predictions = {"answer": {}, "sp": {}}
    gold_list = []

    for i, ex in enumerate(examples):
        qid = ex["_id"]
        prompt = build_prompt(ex)
        print(f"[{i+1}/{len(examples)}] {ex['question'][:80]}...")

        try:
            raw_response = query_agent(prompt)
            answer, sp = parse_response(raw_response)
        except Exception as e:
            print(f"  ERROR: {e}")
            answer, sp = "", []

        predictions["answer"][qid] = answer
        predictions["sp"][qid] = sp
        gold_list.append(ex)

        print(f"  Answer: {answer[:100]}")
        print(f"  SP facts: {len(sp)}")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    metrics = {
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

    N = len(gold_list)
    for k in metrics:
        metrics[k] /= N

    print(f"\n{'Metric':<20} {'Score':>8}")
    print("-" * 30)
    print(f"{'Answer EM':<20} {metrics['em']:>8.4f}")
    print(f"{'Answer F1':<20} {metrics['f1']:>8.4f}")
    print(f"{'SP EM':<20} {metrics['sp_em']:>8.4f}")
    print(f"{'SP F1':<20} {metrics['sp_f1']:>8.4f}")
    print(f"{'Joint EM':<20} {metrics['joint_em']:>8.4f}")
    print(f"{'Joint F1':<20} {metrics['joint_f1']:>8.4f}")

    # Save predictions for later analysis
    pred_path = os.path.join(os.path.dirname(__file__), "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nPredictions saved to {pred_path}")


if __name__ == "__main__":
    main()
