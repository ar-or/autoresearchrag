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


def token_precision(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    return num_common / len(pred_tokens)


def token_recall(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    return num_common / len(ref_tokens)


def token_f1(prediction: str, reference: str) -> float:
    precision = token_precision(prediction, reference)
    recall = token_recall(prediction, reference)
    if precision == 0.0 or recall == 0.0:
        return 0.0
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


def extractiveness_rouge(source_text: str, prediction: str) -> float:
    normalized_source = normalize_answer(source_text)
    normalized_prediction = normalize_answer(prediction)
    if not normalized_source or not normalized_prediction:
        return float(normalized_source == normalized_prediction)
    if normalized_prediction in normalized_source:
        return 1.0
    return rouge_l(source_text, prediction)


def is_abstention_response(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return True
    abstention_markers = (
        "i dont know",
        "i do not know",
        "i dont have",
        "i do not have",
        "dont have the answer",
        "do not have the answer",
        "dont have enough information",
        "do not have enough information",
        "not enough information",
        "cannot answer",
        "cant answer",
        "cannot determine",
        "unable to answer",
        "document does not provide",
        "documents do not provide",
        "context does not provide",
        "i am sorry but",
        "im sorry but",
        "sorry but i",
    )
    return any(marker in normalized for marker in abstention_markers)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def _unique_preserve_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_ids: list[str] = []
    for rid in ids:
        if rid in seen:
            continue
        seen.add(rid)
        unique_ids.append(rid)
    return unique_ids


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = _unique_preserve_order(retrieved_ids)[:k]
    hits = sum(1 for rid in retrieved_top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_top_k = _unique_preserve_order(retrieved_ids)[:k]
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
