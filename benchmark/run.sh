#!/usr/bin/env bash
# 가상환경 python을 자동으로 잡아서 run_negotiation.py를 실행하는 래퍼.
# 사용법: bash benchmark/run.sh [--seed N] [--model MODEL] [--family FAMILY]
set -e
cd "$(dirname "$0")"
../.venv/Scripts/python.exe run_negotiation.py "$@"
