#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

if [ -d "$DATA_DIR/mt-rag-benchmark" ]; then
  echo "Data already downloaded at $DATA_DIR/mt-rag-benchmark"
  echo "To re-download, remove the directory and run again."
  exit 0
fi

mkdir -p "$DATA_DIR"
echo "Cloning IBM MT-RAG benchmark..."
git clone --depth 1 https://github.com/IBM/mt-rag-benchmark.git "$DATA_DIR/mt-rag-benchmark"
echo "Done. Data available at $DATA_DIR/mt-rag-benchmark"
