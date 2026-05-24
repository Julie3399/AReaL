# AReaL Math Reflection Practice 知识点复习稿

## 1. 整个 notebook 在做什么

这个 practice notebook 的目标是用 AReaL 搭一个数学题 RL 训练闭环。核心流程是：从 GSM8k 取一道题，把题目转成 chat prompt，用 SGLang 生成答案，用可验证 reward 判断答案对错，再把 prompt、生成 token、logprob、reward、mask 组成 trajectory，最后交给 FSDP actor 做 PPO/GRPO 更新。

它不是一个单纯的推理 notebook，而是同时包含了 rollout、reward、trajectory construction、reflection、多 sample grouping、actor training、weight update 几个部分。理解这个 notebook 的关键是看清楚数据怎么流动：

```text
GSM8k data
-> workflow
-> SGLang rollout
-> reward
-> trajectory
-> actor PPO update
-> weight update back to rollout
```

## 2. AReaL 里的几个核心角色

`workflow` 定义“一个样本如何被 rollout 成训练数据”。它负责把输入数据变成 prompt，调用 inference engine 生成，计算 reward，并返回 actor 能训练的 trajectory。

`rollout engine` 是推理端。这里使用的是 `RemoteSGLangEngine`，它不直接在 notebook 里跑模型 forward，而是通过 HTTP 和本地 SGLang server 通信。

`actor` 是训练端。这里使用 `FSDPPPOActor`，负责重新计算 logprob、计算 advantage、做 PPO update。actor 和 rollout 一开始通常加载同一个 base model，但训练后 actor 权重会变化，所以需要同步回 rollout。

`WeightUpdateMeta` 描述 actor 权重如何同步给 rollout。多 GPU 合法拓扑可以用 XCCL/NCCL；单 GPU notebook 练习应该用 disk 模式。

## 3. 如何构造一个最小 RLVR workflow

一个最小 RLVR workflow 需要完成四件事。

第一，把输入样本里的 `messages` 转成 token ids。GSM8k 样本通常是 OpenAI chat 格式，所以需要用：

```python
prompt_ids = tokenizer.apply_chat_template(
    data["messages"],
    tokenize=True,
    add_generation_prompt=True,
    return_dict=False,
)
```

这里 `return_dict=False` 很重要，否则可能返回 `BatchEncoding`，不能 JSON serialize。

第二，构造 `ModelRequest`：

```python
req = ModelRequest(
    rid=rid,
    input_ids=input_ids,
    gconfig=self.gconfig,
)
```

`rid` 是 request id，一般用 `uuid.uuid4().hex` 保证唯一。`input_ids` 是 prompt tokens。`gconfig` 是 generation config，比如 max_new_tokens、temperature、top_p 等。

第三，调用推理引擎生成：

```python
resp = await engine.agenerate(req)
```

这里的 `engine` 是 AReaL 的 inference engine 抽象，在这个 notebook 中背后是 SGLang server。`await` 表示这是异步请求。

第四，计算 reward：

```python
gen_str = tokenizer.decode(resp.output_tokens)
loop = asyncio.get_event_loop()
reward = await loop.run_in_executor(
    rw_executor,
    functools.partial(reward_fn, gen_str, answer),
)
```

这里不要直接同步调用 `reward_fn(gen_str, answer)`。在 workflow 里，`gen` 是 async function，event loop 还要继续调度其他 rollout/generation 任务；如果 reward 计算较慢或里面有阻塞逻辑，直接调用会卡住 event loop。所以更合理的写法是先拿到当前 event loop，再通过 `loop.run_in_executor(...)` 把 reward function 丢给 `rw_executor` 执行，并用 `await` 等它返回。概念上 reward 仍然是“模型输出字符串和标准答案比对，得到 0/1”，但执行位置是在 executor 里，而不是 event loop 主线程里。

## 4. trajectory 应该包含什么

workflow 最终返回的不是字符串，而是训练用的 trajectory dict。最小字段包括：

```python
{
    "input_ids": ...,
    "logprobs": ...,
    "loss_mask": ...,
    "rewards": ...,
    "attention_mask": ...,
}
```

`input_ids` 是 prompt tokens + generated tokens：

```python
input_ids = resp.input_tokens + resp.output_tokens
```

`logprobs` 必须和 `input_ids` 一样长。prompt 部分不是这次 rollout 里模型要优化的 action，所以用 0 占位：

```python
logprobs = [0.0] * resp.input_len + resp.output_logprobs
```

`loss_mask` 决定哪些 token 参与 policy loss。prompt 不训练，generation 训练：

```python
loss_mask = [0] * resp.input_len + [1] * resp.output_len
```

`attention_mask` 通常是真实 token 为 1，padding 为 0。在单条 trajectory 里可以先全 1，后面 concat/pad 时再处理 padding。

这里最重要的知识点是：`input_ids`、`logprobs`、`loss_mask` 必须逐 token 对齐。否则 PPO 会在错误位置上更新模型。

## 5. 为什么 prompt 部分 logprob 是 0

训练时我们优化的是模型“生成出的 answer tokens”，不是题目本身。prompt 是环境给定的条件，不是 policy action。所以虽然完整序列里包含 prompt，但 prompt 部分的 logprob 只是为了 shape 对齐，用 `0.0` 占位。

真正参与 policy loss 的是 generated tokens，对应 `loss_mask=1` 的部分。这样 actor 只会因为自己的回答被 reward 强化或惩罚，而不会因为题目文本本身被训练。

## 6. event loop / async / executor 的位置

`engine.agenerate(req)` 是异步生成请求。它会把请求发给 SGLang server，然后等待返回。Python 的 event loop 负责调度这种异步任务。

如果后面有 CPU-bound 或 blocking 的 reward 计算，可以用：

```python
loop = asyncio.get_event_loop()
reward = await loop.run_in_executor(
    rw_executor,
    functools.partial(reward_fn, gen_str, answer),
)
```

或者在 notebook 中用：

```python
await asyncio.to_thread(...)
```

今天先记住一个实用规则：在 Jupyter notebook 中已经存在 running event loop，所以不要随便在 cell 里调用 `asyncio.run(...)`。如果需要运行 async workflow，优先直接 `await ...`，或者用 AReaL 提供的同步封装。

## 7. Reflection workflow 的核心思想

单轮 RLVR 是一次 prompt 一次 answer。Reflection workflow 是多轮：如果模型第一轮答错，就把之前的回答和反馈放回上下文，让模型再尝试。

它的核心循环是：

```text
generate answer
-> compute reward
-> if correct: stop
-> else: append reflection prompt
-> next turn
```

`max_turns` 控制最多反思几轮。`turn_discount` 控制晚答对的 reward 折扣。比如第一轮答对 reward 是 1.0，第二轮答对可能是 0.9，第三轮是 0.81。这会鼓励模型尽早答对，而不是无限依赖反思。

Reflection workflow 返回的仍然是 trajectory，只是这个 trajectory 可能包含多轮对话内容和最后的 reward。

## 8. GroupedReflectionWorkflow / 多答案采样

GRPO 需要同一个问题生成多个答案，然后在组内比较 reward。因此 `GroupedReflectionWorkflow` 的作用是：对同一个 data sample 跑多次 reflection workflow，再把结果合并。

注意当前 inference engine 不支持一次请求里 `n_samples > 1`。所以不能直接让 `engine.agenerate` 生成多个 samples，而是要多次调用单样本 generation。

正确思路是：

```python
single_gconfig = self.gconfig.new(n_samples=1)
workflow = ReflectionWorkflow(single_gconfig, ...)
results = await asyncio.gather(...)
return concat_padded_tensors(results)
```

这里 `concat_padded_tensors` 很关键。因为不同 sample 生成长度不同，需要 pad 到同一长度，再合成一个 batch dict。

最后返回的结构是一个 dict，其中 batch dimension 对应同一个 prompt 的多个 sampled answers。例如：

```python
"rewards": tensor([1., 0.])
```

表示同一道题采样了两个答案，一个对，一个错。

## 9. rollout_batch 返回值怎么理解

`rollout.rollout_batch(data, workflow=workflow)` 的返回值通常是 `list[dict]`。

外层 list 对应不同输入问题。每个 dict 是这个问题经过 workflow 后得到的 trajectory batch。如果使用 grouped workflow，那么这个 dict 内部的第一个维度可能是 group size，也就是同一道题的多个答案。

所以后面 actor 处理 batch 时，要记住：

```python
batch = actor.rollout_batch(...)
```

这里的 `batch` 是 list，不是单个 dict。因此不能写：

```python
batch["prox_logp"] = logps
```

而应该逐条写：

```python
logps = actor.compute_logp(batch)
for traj, logp in zip(batch, logps):
    traj["prox_logp"] = logp
```

## 10. actor 训练一步包含什么

一个同步训练 step 的核心流程是：

```python
batch = actor.rollout_batch(next(data_generator), workflow=workflow)

logps = actor.compute_logp(batch)
for traj, logp in zip(batch, logps):
    traj["prox_logp"] = logp

adv_batch = actor.compute_advantages(batch)

actor.ppo_update(adv_batch)
actor.step_lr_scheduler()

await asyncio.to_thread(actor.update_weights, weight_update_meta)

actor.set_version(global_step + 1)
rollout.set_version(global_step + 1)
```

这里每一步的意义是：

`rollout_batch`：用当前 rollout policy 生成训练数据。

`compute_logp`：actor 重新计算当前策略下这些 token 的 logprob，作为 proximal policy logprob。

`compute_advantages`：根据 reward、group norm、mask 等计算 advantage。

`ppo_update`：真正更新 actor 参数。

`step_lr_scheduler`：更新学习率。

`update_weights`：把 actor 新权重同步给 rollout，使下一轮生成使用新策略。

`set_version`：标记 actor 和 rollout 的权重版本。

## 11. actor 初始化为什么要 create_process_group

当前 AReaL 版本里，`FSDPPPOActor` 需要先：

```python
actor.create_process_group(parallel_strategy=actor_alloc.parallel)
```

再：

```python
actor.initialize(None, ft_spec)
```

因为 `create_process_group` 会初始化 torch distributed、device mesh、parallel helper、logger 等状态。即使单卡，也要有一个 world size = 1 的 process group。否则会出现类似：

```text
AttributeError: 'FSDPPPOActor' object has no attribute 'logger'
```

这说明 actor 的分布式训练环境还没初始化好。

## 12. ModelAllocation 怎么理解

```python
rollout_alloc = ModelAllocation.from_str("sglang:d1p1t1")
actor_alloc = ModelAllocation.from_str("fsdp:d1p1t1")
```

`sglang` 表示 rollout backend 是 SGLang。

`fsdp` 表示 actor backend 是 FSDP。

`d1p1t1` 表示：

```text
d1 = data parallel size 1
p1 = pipeline parallel size 1
t1 = tensor parallel size 1
```

在我们的单卡练习里，这些都是 1。虽然没有真正多卡并行，但 AReaL 仍然用这个 allocation 描述模型和 engine 的分布式形状。

## 13. 权重同步：XCCL vs Disk

XCCL/NCCL 权重同步：

```python
weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
    gen_allocation=rollout_alloc
)
```

这种方式让 actor 直接通过 NCCL 把权重 broadcast 给 SGLang。优点是快，不需要写盘。缺点是要求硬件拓扑合法。我们在单卡 notebook 中 actor 和 SGLang 都在同一张 GPU 上，NCCL 会报：

```text
Duplicate GPU detected
```

所以单卡不适合这种方式。

Disk 权重同步：

```python
weight_update_meta = WeightUpdateMeta.from_disk(
    experiment_name=config.experiment_name,
    trial_name=config.trial_name,
    file_root=config.cluster.fileroot,
    name="default",
    clear_checkpoint_after_load=True,
)
```

这种方式让 actor 先把模型权重保存到磁盘，SGLang 再从磁盘加载。优点是单卡稳定，缺点是慢一些。对于 notebook 练习，应该优先使用 disk 模式。

对应 YAML 里也要设置：

```yaml
actor:
  weight_update_mode: disk
```

## 14. SGLang server 如何判断是否真的可用

不能只看 server 进程存在，也不能只看端口 listen。真正可用至少要满足：

```text
进程存在
端口 11451 listening
/health 返回 200
/generate 能返回合理输出
日志没有新的 fatal error
```

我们最后用这个请求验证：

```text
2+2=? Answer briefly.
```

返回：

```text
4<|im_end|>
```

这说明 SGLang server 真的能生成。

常见 server 问题包括：

`address already in use`：旧 server 没清掉。

`detokenizer heartbeat failed`：server 已经卡死，需要 kill parent/scheduler/detokenizer。

`No such file or directory: ninja`：SGLang JIT 编译 kernel 缺 `ninja-build`。

`BatchEncoding is not JSON serializable`：手动发请求时 tokenizer 输出不是普通 list。

## 15. 之后如何快速 catch up

如果之后重新打开这个 notebook，可以按下面顺序恢复理解：

先看实验准备，确认配置是单卡、Qwen 0.5B、SGLang、FSDP、disk weight update。

再看 RLVRWorkflow，确认自己理解 `prompt_ids -> ModelRequest -> agenerate -> reward -> trajectory`。

然后看 ReflectionWorkflow，理解“错了就追加反思 prompt 再生成”。

再看 GroupedReflectionWorkflow，理解“同一道题采样多个答案，然后 concat/pad 成 group batch”。

最后看同步训练，理解 `rollout_batch -> compute_logp -> compute_advantages -> ppo_update -> update_weights`。

如果跑的时候卡住，优先检查 SGLang：用 `sglang-notebook-debug` skill 的脚本确认 `/health` 和 `/generate`。如果训练到权重同步失败，先确认是不是还在用 XCCL；单卡练习应该用 disk。
