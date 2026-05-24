#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-/home/ubuntu/data/asearcher_local_rag_full}"
LOG_FILE="${LOG_FILE:-/tmp/asearcher_full_rag_download.log}"
PID_FILE="${PID_FILE:-/tmp/asearcher_full_rag_download.pid}"
PYTHON="${PYTHON:-/home/ubuntu/.venv/bin/python}"

mkdir -p "$OUT_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Full ASearcher RAG download already running with pid $(cat "$PID_FILE")"
  echo "Log: $LOG_FILE"
  exit 0
fi

setsid "$PYTHON" -u /home/ubuntu/AReaL/scripts/download_asearcher_full_rag.py \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "Started full ASearcher RAG download pid $(cat "$PID_FILE")"
echo "Target: $OUT_DIR"
echo "Log: $LOG_FILE"
