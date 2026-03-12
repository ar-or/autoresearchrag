#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p data
echo "Downloading HotpotQA dev distractor dataset..."
curl -L -o data/hotpot_dev_distractor_v1.json \
  "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
echo "Done. File saved to data/hotpot_dev_distractor_v1.json"
