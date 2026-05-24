#!/usr/bin/env bash
set -u

REPO="${1:-/home/ubuntu/AReaL}"
SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:11451}"
RAG_URL="${RAG_URL:-http://127.0.0.1:5001}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-asearcher-0.5b-local-practice}"
TRIAL_NAME="${TRIAL_NAME:-trial0}"

echo "== repo/files =="
for f in \
  "$REPO/notebook/search_agent_zh_practice.ipynb" \
  "$REPO/examples/search_agent/local_0.5b_single_gpu.yaml" \
  "$REPO/examples/search_agent/practice_tiny.jsonl" \
  "$REPO/scripts/launch_asearcher_rag_subset.sh" \
  "$REPO/ASearcher/tools/local_retrieval_server.py"; do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

echo "== python/imports =="
if test -x /home/ubuntu/.venv/bin/python; then
  /home/ubuntu/.venv/bin/python - <<'PY' || true
import importlib
for name in ["torch", "areal", "sglang", "aiohttp", "faiss", "wandb"]:
    try:
        mod = importlib.import_module(name)
        print(f"OK {name} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"MISSING {name}: {exc}")
PY
else
  echo "MISSING /home/ubuntu/.venv/bin/python"
fi

echo "== ports/processes =="
ss -ltnp | rg ':11451|:5001|:14514' || true
ps -ef | rg 'sglang|launch_server|local_retrieval_server|ipykernel' | rg -v rg || true

echo "== sglang health/generate =="
curl --max-time 5 -s -o /tmp/search_practice_sglang_health.out -w 'health http=%{http_code} time=%{time_total}\n' \
  "$SGLANG_URL/health" || true
/home/ubuntu/.venv/bin/python - <<PY || true
import requests, time
try:
    t = time.time()
    r = requests.post(
        "$SGLANG_URL/generate",
        json={"text": "The capital city of China is", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}},
        timeout=20,
    )
    print("generate", r.status_code, round(time.time() - t, 3), r.text[:300])
except Exception as exc:
    print("generate FAILED", type(exc).__name__, exc)
PY

echo "== rag retrieve =="
curl --max-time 10 -s -o /tmp/search_practice_rag.out -w 'rag http=%{http_code} time=%{time_total}\n' \
  -H 'Content-Type: application/json' \
  -d '{"queries":["capital city of China"],"topk":1,"return_scores":false}' \
  "$RAG_URL/retrieve" || true
test -s /tmp/search_practice_rag.out && head -c 500 /tmp/search_practice_rag.out && echo || true

echo "== stale disk entries =="
STALE_ROOT="/tmp/areal/name_resolve/ubuntu/${EXPERIMENT_NAME}/${TRIAL_NAME}/update_weights_from_disk"
if test -d "$STALE_ROOT"; then
  find "$STALE_ROOT" -name ENTRY -print
else
  echo "OK no stale update_weights_from_disk directory"
fi

echo "== gpu =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
