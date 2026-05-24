#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/AReaL}"
ASEARCHER_DIR="${ASEARCHER_DIR:-$REPO_ROOT/ASearcher}"
PYTHON="${PYTHON:-/home/ubuntu/.venv/bin/python}"
PORT="${1:-5001}"
RAG_DIR="${RAG_DIR:-/home/ubuntu/data/asearcher_local_rag_subset}"
MODEL_DIR="${MODEL_DIR:-/home/ubuntu/models/e5-base-v2}"
ADDR_DIR="${ADDR_DIR:-/tmp/areal/rag_server_addr}"
LOG_FILE="${LOG_FILE:-/tmp/asearcher_rag_${PORT}.log}"
PID_FILE="${PID_FILE:-/tmp/asearcher_rag_${PORT}.pid}"

mkdir -p "$ADDR_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "ASearcher RAG server already running with pid $(cat "$PID_FILE")"
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ASEARCHER_DIR:${PYTHONPATH:-}"
export ASEARCHER_RAG_HOST="${ASEARCHER_RAG_HOST:-127.0.0.1}"

setsid env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHONPATH="$PYTHONPATH" \
  ASEARCHER_RAG_HOST="$ASEARCHER_RAG_HOST" \
  "$PYTHON" -u "$ASEARCHER_DIR/tools/local_retrieval_server.py" \
    --index_path "$RAG_DIR/e5.index/e5_Flat.index" \
    --corpus_path "$RAG_DIR/wiki_corpus.jsonl" \
    --pages_path "$RAG_DIR/wiki_webpages.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "$MODEL_DIR" \
    --port "$PORT" \
    --save-address-to "$ADDR_DIR" \
    > "$LOG_FILE" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "Started ASearcher RAG server pid $(cat "$PID_FILE") on ${ASEARCHER_RAG_HOST}:${PORT}"
echo "Log: $LOG_FILE"
