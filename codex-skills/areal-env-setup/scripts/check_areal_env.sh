#!/usr/bin/env bash
set -u

REPO="${1:-/home/ubuntu/AReaL}"
echo "== repo =="
test -d "$REPO" && echo "OK repo: $REPO" || echo "MISSING repo: $REPO"

echo "== python =="
if test -x /home/ubuntu/.venv/bin/python; then
  /home/ubuntu/.venv/bin/python --version
else
  echo "MISSING /home/ubuntu/.venv/bin/python"
fi

echo "== cuda =="
command -v nvcc || true
test -f /usr/local/cuda/include/cuda.h && echo "OK cuda.h" || echo "MISSING cuda.h"
test -f /usr/local/cuda/include/curand.h && echo "OK curand.h" || echo "MISSING curand.h"
test -f /usr/local/cuda/lib64/libcudart.so && echo "OK libcudart" || echo "MISSING libcudart"
test -f /usr/local/cuda/lib64/libcurand.so && echo "OK libcurand" || echo "MISSING libcurand"

echo "== compilers =="
command -v gcc-12 || true
command -v g++-12 || true
command -v ninja || true

echo "== key files =="
for f in \
  "$REPO/notebook/math_reflection_zh_practice.ipynb" \
  "$REPO/notebook/search_agent_zh_practice.ipynb" \
  "$REPO/examples/math/gsm8k_grpo_single_gpu.yaml" \
  "$REPO/examples/search_agent/local_0.5b_single_gpu.yaml" \
  "$REPO/examples/search_agent/practice_tiny.jsonl" \
  "$REPO/scripts/launch_asearcher_rag_subset.sh" \
  "$REPO/docs/practice-notes/math-reflection-practice-learning-log.md" \
  "$REPO/docs/practice-notes/math-reflection-practice-knowledge-review.md"; do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

echo "== search rag files =="
for f in \
  "/home/ubuntu/AReaL/ASearcher/tools/local_retrieval_server.py" \
  "/home/ubuntu/models/e5-base-v2/config.json" \
  "/home/ubuntu/data/asearcher_local_rag_subset/wiki_corpus.jsonl" \
  "/home/ubuntu/data/asearcher_local_rag_subset/wiki_webpages.jsonl" \
  "/home/ubuntu/data/asearcher_local_rag_subset/e5.index/e5_Flat.index"; do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

echo "== sglang process =="
ps -ef | rg 'sglang|launch_server|11451' | rg -v rg || true
ss -ltnp | rg ':11451|:5001|:14514' || true

echo "== sglang health =="
if command -v curl >/dev/null; then
  curl --max-time 3 -s -o /tmp/areal_sglang_health_check.out -w 'http=%{http_code} time=%{time_total}\n' http://127.0.0.1:11451/health || true
fi

echo "== search rag health =="
if command -v curl >/dev/null; then
  curl --max-time 5 -s -o /tmp/areal_rag_check.out -w 'http=%{http_code} time=%{time_total}\n' \
    -H 'Content-Type: application/json' \
    -d '{"queries":["capital city of China"],"topk":1,"return_scores":false}' \
    http://127.0.0.1:5001/retrieve || true
  test -s /tmp/areal_rag_check.out && head -c 300 /tmp/areal_rag_check.out && echo || true
fi

echo "== stale disk weight entries =="
find /tmp/areal/name_resolve/ubuntu/asearcher-0.5b-local-practice/trial0/update_weights_from_disk \
  -name ENTRY -print 2>/dev/null || true

echo "== gpu =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
