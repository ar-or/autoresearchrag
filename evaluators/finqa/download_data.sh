#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p data
echo "Downloading FinQA dev dataset..."
curl -L -o data/dev.json \
  "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json"
echo "Done. File saved to data/dev.json ($(wc -c < data/dev.json) bytes)"
