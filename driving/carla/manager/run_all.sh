#!/bin/bash
cd "$(dirname "$0")"

echo "[1/2] no npc — 50, 90, 130 km/h"
python3 auto_run.py --woc --nonpc --speeds 50 90 130 --resume

echo "[2/2] npc — 30, 70, 110 km/h"
python3 auto_run.py --woc --npc --speeds 30 70 110 --resume

echo "=== 전체 완료 ==="
