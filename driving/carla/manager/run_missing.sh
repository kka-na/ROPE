#!/bin/bash
cd "$(dirname "$0")"
echo "[MISSING] 누락 시나리오 재실행 (0km/h 제외, --resume 적용)"
python3 auto_run.py --trials-file ../missing_scenarios.txt --resume
echo "=== 완료 ==="
