#!/usr/bin/env python3
"""Build the Memgraph graph layer for H44."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.graph_ingest_common import run_graph_ingest


def main() -> None:
    run_graph_ingest(
        graph_name=os.environ.get("GRAPH_LAYER_NAME", "h44"),
        variant="lightrag",
    )


if __name__ == "__main__":
    main()
