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
  "$REPO/examples/math/gsm8k_grpo_single_gpu.yaml" \
  "$REPO/docs/practice-notes/math-reflection-practice-learning-log.md" \
  "$REPO/docs/practice-notes/math-reflection-practice-knowledge-review.md"; do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done

echo "== sglang process =="
ps -ef | rg 'sglang|launch_server|11451' | rg -v rg || true
ss -ltnp | rg ':11451' || true

echo "== gpu =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
