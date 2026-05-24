---
name: areal-env-setup
description: Set up or repair the local AReaL math-reflection notebook environment with Qwen 0.5B, SGLang, CUDA/JIT dependencies, single-GPU YAML settings, notebook environment variables, and validation checks. Use when preparing a fresh machine, restarting the practice notebook, or fixing environment drift before rollout/training.
---

# AReaL Environment Setup

Use this skill before running `/home/ubuntu/AReaL/notebook/math_reflection_zh_practice.ipynb` or when moving the practice work to a new machine/repo.

## Setup Checklist

1. Work from the repo:

```bash
cd /home/ubuntu/AReaL
```

2. Ensure CUDA compile dependencies exist:

```bash
sudo apt-get update
sudo apt-get install -y cuda-compiler-12-9 cuda-cudart-dev-12-9 libcurand-dev-12-9 g++-12 ninja-build
```

3. Export compile/runtime environment before launching SGLang or running notebook cells:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export AREAL_LLM_SERVER_ADDRS=127.0.0.1:11451
```

4. In the notebook, set torch distributed variables for single-GPU FSDP:

```python
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(MASTER_PORT)
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
```

5. Use the single-GPU YAML:

```text
examples/math/gsm8k_grpo_single_gpu.yaml
```

The important settings are:

```yaml
actor:
  path: Qwen/Qwen2.5-0.5B-Instruct
  backend: fsdp:d1p1t1
  weight_update_mode: disk
  attn_impl: sdpa
rollout:
  backend: sglang:d1p1t1
sglang:
  context_length: 8192
  mem_fraction_static: 0.45
  disable_cuda_graph: true
  disable_overlap_schedule: true
gconfig:
  n_samples: 2
  max_new_tokens: 512
```

Use `disk` weight update for single-GPU practice because actor and SGLang share one GPU; XCCL/NCCL can fail with `Duplicate GPU detected`.

## Notebook-Specific Rules

- Start SGLang before initializing `RemoteSGLangEngine`.
- Use `/home/ubuntu/.venv/bin/python` or `sys.executable` for launch commands, not an unrelated Python.
- In tokenizer calls that feed SGLang JSON payloads, use `return_dict=False`.
- Initialize actor with `create_process_group(...)` before `initialize(...)`.
- In notebook cells, use `await asyncio.to_thread(actor.update_weights, weight_update_meta)` for blocking weight updates.
- Prefer `WeightUpdateMeta.from_disk(...)` for the single-GPU practice notebook.

## Validation

Run:

```bash
~/.codex/skills/sglang-notebook-debug/scripts/check_sglang_server.py --url http://127.0.0.1:11451
```

The environment is ready when `/health` returns 200 and `/generate` decodes a plausible answer.

You can also run:

```bash
~/.codex/skills/areal-env-setup/scripts/check_areal_env.sh
```

to check CUDA, compiler, ninja, repo files, and the current SGLang process state.

## Common Repairs

- Missing `cuda.h`, `curand.h`, or `nvcc`: install CUDA compiler/dev packages and set `CUDA_HOME`.
- Missing `cc1plus`: install `g++-12` and export `CXX=/usr/bin/g++-12`.
- Missing `ninja`: install `ninja-build`.
- Server port occupied: kill old SGLang parent, scheduler, and detokenizer processes before relaunch.
- Server health fails after prior training crash: restart SGLang and restart the notebook kernel if torch distributed/NCCL state was corrupted.
