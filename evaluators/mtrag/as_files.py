#!/usr/bin/env python3
"""Export MT-RAG passage-level corpora as plain text files."""

import json
import os
import zipfile
from pathlib import Path

CORPORA_DIR = Path(__file__).resolve().parent / "data" / "mt-rag-benchmark" / "corpora" / "passage_level"
ALL_DOMAINS = ["clapnq", "cloud", "fiqa", "govt"]


def as_files(output_dir: Path):
    for domain in ALL_DOMAINS:
        zip_path = CORPORA_DIR / f"{domain}.jsonl.zip"
        if not zip_path.exists():
            print(f"  {domain}: data not found at {zip_path}, skipping")
            continue

        domain_dir = output_dir / domain
        os.makedirs(domain_dir, exist_ok=True)

        written = 0
        skipped = 0

        with zipfile.ZipFile(zip_path) as zf:
            jsonl_name = zf.namelist()[0]
            with zf.open(jsonl_name) as f:
                for line in f:
                    passage = json.loads(line)
                    text = passage.get("text", "").strip()
                    if not text:
                        continue
                    passage_id = passage.get("_id", passage.get("id", ""))
                    filepath = domain_dir / f"{passage_id}.txt"
                    if filepath.exists():
                        skipped += 1
                        continue
                    filepath.write_text(text)
                    written += 1

        print(f"  {domain}: Written: {written}, Skipped: {skipped}")
