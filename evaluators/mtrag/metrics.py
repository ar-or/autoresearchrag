"""Local evaluation metrics for MT-RAG benchmark."""

import re
import string
import math
from collections import Counter


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles, punctuation, and extra whitespace."""
    s = s.lower()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    s = " ".join(s.split())
    return s


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    hits = sum(1 for rid in retrieved_top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = retrieved_ids[:k]
    # DCG
    dcg = 0.0
    for i, rid in enumerate(retrieved_top_k):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because rank starts at 1
    # Ideal DCG
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    cutoffs: list[int] | None = None,
) -> dict[str, float]:
    if cutoffs is None:
        cutoffs = [1, 3, 5, 10]
    metrics = {}
    for k in cutoffs:
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    return metrics
