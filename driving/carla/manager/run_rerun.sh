#!/bin/bash
cd "$(dirname "$0")"

echo "[rerun] CLM4/ETrA3 faster c2 nonpc — 6 trials"
python3 auto_run.py --trials-file trials_rerun.txt

echo "=== 완료 ==="
