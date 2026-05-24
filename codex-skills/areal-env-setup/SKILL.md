---
name: areal-env-setup
description: Set up or repair the local AReaL math-reflection and search-agent notebook environment with Qwen 0.5B, SGLang, ASearcher local RAG, CUDA/JIT dependencies, single-GPU YAML settings, notebook environment variables, wandB logging, and validation checks. Use when preparing a fresh machine, restarting practice notebooks, or fixing environment drift before rollout/training.
---

# AReaL Environment Setup

Use this skill before running either practice notebook:

```text
/home/ubuntu/AReaL/notebook/math_reflection_zh_practice.ipynb
/home/ubuntu/AReaL/notebook/search_agent_zh_practice.ipynb
```

or when moving the practice work to a new machine/repo.

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

5. Use the single-GPU YAML for math:

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

6. Use the single-GPU YAML for search:

```text
examples/search_agent/local_0.5b_single_gpu.yaml
```

The important settings are:

```yaml
actor:
  path: Qwen/Qwen2.5-0.5B-Instruct
  backend: fsdp:d1p1t1
  weight_update_mode: disk
  attn_impl: sdpa
  log_agent_stats: false
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
train_dataset:
  path: examples/search_agent/practice_tiny.jsonl
```

Keep `actor.log_agent_stats: false` for the practice SearchAgentWorkflow unless the workflow explicitly returns AReaL agent-stat fields such as `begin_of_trajectory`.

7. For search practice, start the local ASearcher RAG subset server:

```bash
/home/ubuntu/AReaL/scripts/launch_asearcher_rag_subset.sh 5001
```

Expected local resources:

```text
/home/ubuntu/AReaL/ASearcher
/home/ubuntu/models/e5-base-v2
/home/ubuntu/data/asearcher_local_rag_subset/wiki_corpus.jsonl
/home/ubuntu/data/asearcher_local_rag_subset/wiki_webpages.jsonl
/home/ubuntu/data/asearcher_local_rag_subset/e5.index/e5_Flat.index
```

This is a real local RAG server built from official ASearcher code, e5, FAISS, and an official-corpus subset. It is not the full 45M-row ASearcher knowledge base.

## Notebook-Specific Rules

- Start SGLang before initializing `RemoteSGLangEngine`.
- Use `/home/ubuntu/.venv/bin/python` or `sys.executable` for launch commands, not an unrelated Python.
- If `SGLangConfig.build_cmd(...)` returns `python3`, replace it with `/home/ubuntu/.venv/bin/python` before `subprocess.Popen(...)`. Starting with system Python can fail with missing packages such as `aiohttp`.
- In tokenizer calls that feed SGLang JSON payloads, use `return_dict=False`.
- Initialize actor with `create_process_group(...)` before `initialize(...)`.
- In notebook cells, use `await asyncio.to_thread(actor.update_weights, weight_update_meta)` for blocking weight updates.
- Prefer `WeightUpdateMeta.from_disk(...)` for the single-GPU practice notebook.
- For single-GPU training, initialize with the actor allocation's parallel strategy:

```python
actor_alloc = ModelAllocation.from_str("fsdp:d1p1t1")
parallel_strategy = actor_alloc.parallel

actor = FSDPPPOActor(config=config.actor)
actor.create_process_group(parallel_strategy=parallel_strategy)
actor.initialize(None, ft_spec)

rollout = RemoteSGLangEngine(config.rollout)
rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
```

- In training loops, use `actor.rollout_batch(...)`, move the batch to `actor.device`, and pass the returned advantage batch to PPO:

```python
batch = actor.rollout_batch(next(data_generator), workflow=workflow)
batch = tensor_container_to(batch, actor.device)

logps = actor.compute_logp(batch)
for traj, logp in zip(batch, logps):
    traj["prox_logp"] = logp

adv_batch = actor.compute_advantages(batch)
actor.ppo_update(adv_batch)
```

- For search workflows, parse only the current completion, not the whole context:

```python
completion_str = tokenizer.decode(resp.output_tokens)
search_query = parse_search_query(completion_str)
answer = parse_answer(completion_str)
```

- Only call RAG if `search_query` is truthy. `call_search_tool(...)` returns one result list per query, so use `(await call_search_tool(...))[0]`.
- Search results are tool-provided context, so append them with `loss_mask=0`.
- Search answers can be lists; compute reward with `max(f1_score(answer, gt) for gt in data["answer"])`.
- Assert trajectory field lengths before returning:

```python
assert len(input_ids) == len(logprobs)
assert len(input_ids) == len(loss_mask)
```

- For long search practice runs with wandB, use project `asearcher`, set `verbose=False`, and log at least reward mean, success rate, sequence length, step time, and learning rate.

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

For search practice, run:

```bash
~/.codex/skills/areal-env-setup/scripts/check_search_practice_env.sh
```

The search environment is ready when:

- SGLang on `127.0.0.1:11451` has `/health` 200 and `/generate` succeeds;
- ASearcher RAG on `127.0.0.1:5001` returns `/retrieve` results;
- no stale `update_weights_from_disk/*/ENTRY` files remain for the current search experiment/trial.

## Common Repairs

- Missing `cuda.h`, `curand.h`, or `nvcc`: install CUDA compiler/dev packages and set `CUDA_HOME`.
- Missing `cc1plus`: install `g++-12` and export `CXX=/usr/bin/g++-12`.
- Missing `ninja`: install `ninja-build`.
- Server port occupied: kill old SGLang parent, scheduler, and detokenizer processes before relaunch.
- Server health fails after prior training crash: restart SGLang and restart the notebook kernel if torch distributed/NCCL state was corrupted.
- SGLang port listens but `/health` or `/generate` times out: the server is wedged. Kill the parent `launch_server`, `sglang::scheduler`, and `sglang::detokenizer`, then relaunch with `/home/ubuntu/.venv/bin/python`.
- RAG `localhost:5001` refuses connections: run `/home/ubuntu/AReaL/scripts/launch_asearcher_rag_subset.sh 5001`.
- `NameEntryExistsError` under `update_weights_from_disk`: remove stale disk weight-update entries for the current experiment/trial:

```bash
rm -rf /tmp/areal/name_resolve/ubuntu/asearcher-0.5b-local-practice/trial0/update_weights_from_disk
```

- `DistNetworkError ... EADDRINUSE ... port: 14514`: an old notebook kernel/process group owns `MASTER_PORT`. Restart the notebook kernel, or choose a fresh `MASTER_PORT` before `create_process_group(...)`. Diagnose with `ss -ltnp | rg ':14514'`.
- PPO update shape assertions after failed attempts can be stale stats-tracker state. Restart the kernel, or clear trackers:

```python
from collections import defaultdict
from areal.utils import stats_tracker

for tracker in stats_tracker.TRACKERS.values():
    tracker.denominators = {}
    tracker.reduce_types = {}
    tracker.stats = defaultdict(list)
```

- `RuntimeError: 'begin_of_trajectory' is expected to log agent statistics`: set `config.actor.log_agent_stats = False` or `actor.log_agent_stats: false` in the practice YAML.
- `No backend type associated with device type cpu`: call `batch = tensor_container_to(batch, actor.device)` before `compute_logp`, `compute_advantages`, and `ppo_update`.
- `KeyError: 'advantages'`: use `adv_batch = actor.compute_advantages(batch)` and pass `adv_batch` to `actor.ppo_update(...)`.
- `list indices must be integers or slices, not str` in `format_search_results`: unwrap search batch results with `(await call_search_tool(...))[0]`.
- `'coroutine' object is not subscriptable`: write `(await call_search_tool(...))[0]`, not `await call_search_tool(...)[0]`.
