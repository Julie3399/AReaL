---
name: sglang-notebook-debug
description: Debug local SGLang servers used from notebooks or AReaL workflows, especially when rollout hangs, /health fails, /generate fails, ports are occupied, detokenizer heartbeat is stale, CUDA/JIT dependencies are missing, or weight update mode breaks single-GPU practice runs.
---

# SGLang Notebook Debug

Use this skill to diagnose and recover a local SGLang server backing an AReaL notebook. Favor direct evidence: processes, ports, server logs, `/health`, and a minimal `/generate` request.

## Fast Workflow

1. Inspect live state:

```bash
ps -ef | rg 'sglang|launch_server|11451' | rg -v rg
ss -ltnp | rg ':11451|:14514' || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
tail -120 /tmp/sglang_11451.log
```

2. Check whether the server is merely listening or actually healthy:

```bash
curl --max-time 3 -s -o /tmp/health.out -w '%{http_code} %{time_total}\n' http://127.0.0.1:11451/health
```

3. Verify true generation, not just health. Use `scripts/check_sglang_server.py`:

```bash
python ~/.codex/skills/sglang-notebook-debug/scripts/check_sglang_server.py --url http://127.0.0.1:11451
```

4. If unhealthy, clean duplicate or wedged SGLang processes before restarting:

```bash
pkill -TERM -f 'areal.experimental.inference_service.sglang.launch_server' || true
pkill -TERM -f 'sglang::scheduler' || true
pkill -TERM -f 'sglang::detokenizer' || true
sleep 2
pkill -KILL -f 'areal.experimental.inference_service.sglang.launch_server|sglang::scheduler|sglang::detokenizer' || true
```

5. Restart with the notebook's intended config. For the AReaL math notebook, preserve:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
export AREAL_LLM_SERVER_ADDRS=127.0.0.1:11451
```

Start in a detached session and log to `/tmp/sglang_11451.log`. Avoid launching a second server on the same port while the old one is alive.

## Failure Signatures And Fixes

- **Port is listening but rollout hangs**: Inspect `/tmp/sglang_11451.log`. If it says `Health check failed. Server couldn't get a response from detokenizer`, the server is wedged. Kill all SGLang parent, scheduler, and detokenizer processes, then restart.
- **`address already in use`**: A previous server is still bound to `127.0.0.1:11451`. Kill it or choose another port and update `AREAL_LLM_SERVER_ADDRS`.
- **`FileNotFoundError: No such file or directory: 'ninja'`** during rope/kvcache JIT: install `ninja-build`, then restart SGLang.
- **CUDA/JIT header errors such as missing `cuda.h` or `curand.h`**: ensure CUDA toolkit/dev packages are installed and `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`, `CC`, `CXX`, and `CUDAHOSTCXX` are set before launch.
- **Manual `/generate` fails with `BatchEncoding is not JSON serializable`**: tokenizer returned a `BatchEncoding`. Use `return_dict=False` in `tokenizer.apply_chat_template(...)` before JSON posting.
- **AReaL single-GPU weight sync fails with `Duplicate GPU detected`**: actor and SGLang are both using the same GPU. Use disk weight update for single-GPU notebook practice:

```python
weight_update_meta = WeightUpdateMeta.from_disk(
    experiment_name=config.experiment_name,
    trial_name=config.trial_name,
    file_root=config.cluster.fileroot,
    name="default",
    clear_checkpoint_after_load=True,
)
actor.connect_engine(rollout, weight_update_meta)
```

Set the YAML to `actor.weight_update_mode: disk`. Reserve `WeightUpdateMeta.from_fsdp_xccl(...)` for multi-GPU or otherwise valid NCCL topologies.
- **`The specified group name has already been created`**: repeated XCCL initialization reused an NCCL group name. Prefer disk mode on single GPU. If XCCL is required, use a unique `nccl_group_name` and ensure the engine code respects it.
- **Notebook event-loop error during weight update**: if `asyncio.run() cannot be called from a running event loop`, run the blocking update in a thread from notebook code:

```python
await asyncio.to_thread(actor.update_weights, weight_update_meta)
```

## Validation Standard

Consider SGLang usable only after all are true:

- exactly one intended server owns the target port;
- `/health` returns HTTP 200;
- a minimal `/generate` request returns HTTP 200 and decodes to a plausible answer;
- no fresh fatal stack trace appears in `/tmp/sglang_11451.log`;
- GPU memory/process state matches the intended server, with no duplicate orphan scheduler process.

For AReaL notebook work, also verify the notebook uses `return_dict=False` when passing token ids to `ModelRequest`, and use disk weight update on single GPU.
