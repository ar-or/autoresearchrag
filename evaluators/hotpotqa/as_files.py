#!/usr/bin/env python3
"""Export HotpotQA paragraphs as plain text files (fullwiki setting).

Each unique Wikipedia paragraph becomes its own file, named by article
title. The agent must retrieve the right paragraphs from ~66k files.
"""

import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "hotpot_dev_distractor_v1.json"


def _safe_filename(title: str) -> str:
    """Convert a title to a safe filename."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title)
    name = name.strip(". ")
    if not name:
        name = "_"
    return name + ".txt"


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
                paragraphs[title] = "\n".join(sentences)

    written = 0
    skipped = 0
    for title, text in paragraphs.items():
        filepath = output_dir / _safe_filename(title)
        if filepath.exists():
            skipped += 1
            continue
        filepath.write_text(text)
        written += 1

    print(f"  Written: {written}, Skipped: {skipped}")
