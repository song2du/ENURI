#!/usr/bin/env bash
# 가상환경 python을 자동으로 잡아서 run_negotiation.py를 실행하는 래퍼.
# 사용법: bash benchmark/run.sh [--seed N] [--model MODEL] [--family FAMILY]
# 2026-07-23: .venv 위치를 레포 루트에서 benchmark/.venv로 변경 (이전 루트 .venv는 이 환경에
# 없었음 -- 새로 benchmark/ 안에 만듦, requirements.txt도 benchmark/requirements.txt로 분리).
set -e
cd "$(dirname "$0")"
.venv/Scripts/python.exe run_negotiation.py "$@"
