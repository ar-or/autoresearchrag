#!/usr/bin/env python3
"""Export HotpotQA context paragraphs as plain text files."""

import hashlib
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "hotpot_dev_distractor_v1.json"


def as_files(output_dir: Path):
    if not DATA_PATH.exists():
        print(f"  Data not found at {DATA_PATH}, skipping")
        return

    with open(DATA_PATH) as f:
        data = json.load(f)

    # Deduplicate paragraphs by title
    paragraphs: dict[str, str] = {}
    for ex in data:
        for title, sentences in ex["context"]:
            if title not in paragraphs:
                paragraphs[title] = " ".join(sentences)

    written = 0
    skipped = 0
    for title, text in paragraphs.items():
        doc_id = hashlib.sha256(title.encode()).hexdigest()[:16]
        filepath = output_dir / f"{doc_id}.txt"
        if filepath.exists():
            skipped += 1
            continue
        filepath.write_text(text)
        written += 1

    print(f"  Written: {written}, Skipped: {skipped}")
