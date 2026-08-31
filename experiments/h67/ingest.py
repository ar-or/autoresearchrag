#!/usr/bin/env python3
"""Benchmark-scoped BioRAG ingest for HotpotQA N=30 and MT-RAG N=3."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest import ensure_ingest_ready, ensure_index, ingest_prepared_items, prepare_document

ES_HOST = os.environ.get("ES_HOST", "http://localhost:9200")
ES_INDEX = os.environ.get("ES_INDEX", "h67_taxonomy_subset")
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY", "")

HOTPOT_DATA = PROJECT_ROOT / "evaluators" / "hotpotqa" / "data" / "hotpot_dev_distractor_v1.json"
MTRAG_GEN = PROJECT_ROOT / "evaluators" / "mtrag" / "data" / "mt-rag-benchmark" / "human" / "generation_tasks" / "RAG.jsonl"
MTRAG_RET = PROJECT_ROOT / "evaluators" / "mtrag" / "data" / "mt-rag-benchmark" / "human" / "retrieval_tasks"
MTRAG_CORPORA = PROJECT_ROOT / "evaluators" / "mtrag" / "data" / "mt-rag-benchmark" / "corpora" / "passage_level"


def _es_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if ELASTIC_API_KEY:
        headers["Authorization"] = f"ApiKey {ELASTIC_API_KEY}"
    return headers


def _reset_index() -> None:
    requests.delete(f"{ES_HOST}/{ES_INDEX}", headers=_es_headers(), timeout=60)
    ensure_index()
    ensure_ingest_ready()


def _predict_taxonomy(title: str, text: str) -> tuple[list[str], str]:
    lowered = f"{title} {text}".lower()
    labels: list[str] = []
    if re.search(r"\b\d{4}\b", lowered):
        labels.append("temporal/date")
    if any(token in lowered for token in ("born", "died", "married", "actor", "writer", "president")):
        labels.append("entity/person/biography")
    if any(token in lowered for token in ("city", "country", "river", "province", "located")):
        labels.append("entity/location/geography")
    if any(token in lowered for token in ("film", "album", "song", "novel", "band")):
        labels.append("entity/work/media")
    if any(token in lowered for token in ("battle", "war", "election", "government", "minister")):
        labels.append("event/politics/history")
    if any(token in lowered for token in ("company", "bank", "stock", "market", "revenue")):
        labels.append("domain/finance/organization")
    if not labels:
        labels.append("general/fact/entity")
    return labels, " > ".join(labels)


def _prepare_raw(item: tuple[str, str, str, str]):
    collection_name, title, text, doc_id = item
    return prepare_document(
        f"RAW_SUPPORT\nTITLE: {title}\nCONTENT: {text}",
        hash_key=f"{doc_id}:raw_support",
        fields={
            "title": title,
            "source": f"{collection_name}:{doc_id}:raw_support",
            "document_id": doc_id,
            "doc_name": title,
            "collection_name": collection_name,
            "view_type": "raw_support",
        },
    )


def _prepare_taxonomy_doc(item: tuple[str, str, str, str]):
    collection_name, title, text, doc_id = item
    labels, hierarchy_path = _predict_taxonomy(title, text)
    enriched_text = (
        "TAXONOMY_ENRICHED\n"
        f"TITLE: {title}\n"
        f"TAXONOMY_LABELS: {', '.join(labels)}\n"
        f"HIERARCHY_PATH: {hierarchy_path}\n"
        f"CONTENT: {text}"
    )
    return prepare_document(
        enriched_text,
        hash_key=f"{doc_id}:taxonomy",
        fields={
            "title": title,
            "source": f"{collection_name}:{doc_id}:taxonomy",
            "document_id": f"{doc_id}:taxonomy",
            "parent_document_id": doc_id,
            "doc_name": title,
            "collection_name": collection_name,
            "view_type": "taxonomy_enriched",
            "taxonomy_labels": ", ".join(labels),
            "hierarchy_path": hierarchy_path,
        },
    )


def _iter_hotpot_documents():
    with open(HOTPOT_DATA) as f:
        data = json.load(f)[:30]
    paragraphs: dict[str, str] = {}
    for example in data:
        for title, sentences in example["context"]:
            paragraphs.setdefault(title, " ".join(sentences))
    for title, text in paragraphs.items():
        doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
        yield ("mycollection2", title, text, doc_id)


def _needed_mtrag_ids() -> dict[str, set[str]]:
    needed: dict[str, set[str]] = defaultdict(set)
    convos: dict[str, list[dict]] = defaultdict(list)
    with open(MTRAG_GEN) as f:
        for line in f:
            obj = json.loads(line)
            convos[obj.get("conversation_id", obj.get("task_id", ""))].append(obj)
    for cid in sorted(convos)[:3]:
        for task in convos[cid]:
            collection = task.get("Collection", task.get("collection", ""))
            for ctx in task.get("contexts", []):
                if ctx.get("reference"):
                    needed[collection].add(ctx.get("document_id", ctx.get("id", "")))
    for domain in ["clapnq", "cloud", "fiqa", "govt"]:
        qrel = MTRAG_RET / domain / "qrels" / "dev.tsv"
        qf = MTRAG_RET / domain / f"{domain}_lastturn.jsonl"
        if not qrel.exists() or not qf.exists():
            continue
        qrels: dict[str, list[str]] = defaultdict(list)
        with open(qrel) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("query-id"):
                    continue
                qid, cid, score = line.split("\t")[:3]
                if int(score) > 0:
                    qrels[qid].append(cid)
        count = 0
        with open(qf) as f:
            for line in f:
                obj = json.loads(line)
                qid = obj.get("_id", obj.get("id", ""))
                if qid not in qrels:
                    continue
                collection = "mt-rag-clapnq-elser-512-100-20240503" if domain == "clapnq" else ("mt-rag-govt-elser-512-100-20240611" if domain == "govt" else f"mt-rag-{domain}-elser-512-100-20240503")
                for cid in qrels[qid]:
                    needed[collection].add(cid)
                count += 1
                if count >= 3 and domain == "clapnq":
                    break
    return needed


def _iter_mtrag_documents():
    needed = _needed_mtrag_ids()
    domain_to_collection = {
        "clapnq": "mt-rag-clapnq-elser-512-100-20240503",
        "cloud": "mt-rag-cloud-elser-512-100-20240503",
        "fiqa": "mt-rag-fiqa-elser-512-100-20240503",
        "govt": "mt-rag-govt-elser-512-100-20240611",
    }
    for domain, collection in domain_to_collection.items():
        wanted = needed.get(collection, set())
        if not wanted:
            continue
        zip_path = MTRAG_CORPORA / f"{domain}.jsonl.zip"
        with zipfile.ZipFile(zip_path) as zf:
            jsonl_name = zf.namelist()[0]
            with zf.open(jsonl_name) as handle:
                for raw_line in handle:
                    passage = json.loads(raw_line)
                    passage_id = passage.get("_id", passage.get("id", ""))
                    if passage_id not in wanted:
                        continue
                    text = passage.get("text", "").strip()
                    if not text:
                        continue
                    doc_id = hashlib.sha256(passage_id.encode()).hexdigest()[:16]
                    title = passage.get("title", "") or passage_id
                    yield ("mycollection3", title, text, doc_id)


def main() -> None:
    _reset_index()
    items = list(_iter_hotpot_documents()) + list(_iter_mtrag_documents())
    raw_ok, raw_err = ingest_prepared_items(items, _prepare_raw, total_items=len(items), progress_every=500)
    tax_ok, tax_err = ingest_prepared_items(items, _prepare_taxonomy_doc, total_items=len(items), progress_every=500)
    print(f"Indexed raw chunks: {raw_ok} ({raw_err} errors)")
    print(f"Indexed taxonomy chunks: {tax_ok} ({tax_err} errors)")


if __name__ == "__main__":
    main()
