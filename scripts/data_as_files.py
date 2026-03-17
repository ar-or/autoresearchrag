#!/usr/bin/env python3
"""Export evaluator source data as plain text files on disk.

Auto-discovers evaluators by globbing evaluators/*/as_files.py, imports each
module, and calls module.as_files(output_dir).

Usage:
  uv run python scripts/data_as_files.py              # all evaluators
  uv run python scripts/data_as_files.py finqa mtrag   # specific evaluators only
"""

import importlib.util
import os
import sys
from glob import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "data_as_files"


def discover_evaluators() -> dict[str, Path]:
    """Return {name: path} for each evaluator that has an as_files.py."""
    result = {}
    for p in sorted(glob(str(PROJECT_ROOT / "evaluators" / "*" / "as_files.py"))):
        path = Path(p)
        name = path.parent.name
        result[name] = path
    return result


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"as_files_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    evaluators = discover_evaluators()
    if not evaluators:
        print("No evaluators found with as_files.py")
        sys.exit(1)

    # Filter to requested evaluators if specified
    requested = sys.argv[1:]
    if requested:
        for name in requested:
            if name not in evaluators:
                print(f"ERROR: No as_files.py found for evaluator '{name}'")
                print(f"Available: {', '.join(evaluators)}")
                sys.exit(1)
        evaluators = {k: v for k, v in evaluators.items() if k in requested}

    for name, path in evaluators.items():
        print(f"\n=== {name} ===")
        output_dir = OUTPUT_ROOT / name
        os.makedirs(output_dir, exist_ok=True)

        mod = load_module(name, path)
        mod.as_files(output_dir)


if __name__ == "__main__":
    main()
