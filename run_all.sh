#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
python -u evaluate.py --data "${CUB_ROOT:-data/CUB_200_2011}" --episodes 200 2>&1 | tee results/run.log
