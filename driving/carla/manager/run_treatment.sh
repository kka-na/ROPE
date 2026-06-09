#!/bin/bash
cd "$(dirname "$0")"

echo "[treatment] no npc — 50, 90, 130 km/h  (DR enabled, data/treatment/)"
python3 auto_run.py --dr --woc --nonpc --speeds 50 90 130 --resume

echo "=== treatment 완료 ==="
