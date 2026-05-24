# AReaL Math Reflection Practice Notebook 学习记录 Draft

## 实验准备

这一部分主要是在把 notebook 从“能读”变成“能跑”。我加载了单卡练习用的 YAML 配置，并把模型改成 `Qwen/Qwen2.5-0.5B-Instruct`。这里我学到，AReaL 的配置不是只决定模型路径，还会影响 rollout backend、actor backend、batch size、generation 参数、SGLang server 的显存占用、context length、CUDA graph 等运行细节。对于 notebook 练习来说，配置要保守，目标是先跑通最小闭环，而不是追求最大吞吐。

在环境变量部分，我理解了 notebook 里需要显式设置 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`、`LOCAL_RANK`，因为后面 FSDP actor 会初始化 torch distributed process group。即使是单卡练习，这些分布式环境变量也不能完全省略。我们还补了 CUDA 相关环境变量，比如 `CUDA_HOME`、`PATH`、`LD_LIBRARY_PATH`、`CC`、`CXX`、`CUDAHOSTCXX`，因为 SGLang/FlashInfer/TVM JIT 会在运行时编译 kernel。

启动 SGLang server 是今天 debug 最多的部分。我们学到，server 不是“进程在就算好”，而是要同时确认端口监听、`/health` 返回 200、`/generate` 能真实生成。过程中遇到过 CUDA toolkit 缺失、`curand.h` 缺失、`g++-12` 缺失、`ninja` 缺失、detokenizer heartbeat 失效、端口被旧 server 占用等问题。最后的验证方式是直接发一个最小生成请求，让模型回答 `2+2=?`，返回 `4<|im_end|>` 才说明 SGLang 真的可用。

GSM8k 数据集部分的重点是理解每个样本包含 `messages` 和 `answer`。`messages` 是 OpenAI chat 格式的 prompt，后面要通过 tokenizer 的 chat template 转成 token ids；`answer` 是 reward function 用来验证模型输出是否正确的标准答案。这里的练习目标不是训练一个强模型，而是构造出可以被 RL 使用的数据流。

导入模块这一节串起了整个 notebook 的角色分工：`RemoteSGLangEngine` 负责 rollout，`FSDPPPOActor` 负责训练，`ModelRequest` 表示一次生成请求，`GenerationHyperparameters` 控制采样，`WeightUpdateMeta` 描述训练后如何把 actor 权重同步给 rollout engine。这里我第一次比较清楚地看到，AReaL 不是一个单模型脚本，而是 inference engine、training engine、workflow、data generator 共同组成的系统。

## 定义简单的单轮工作流

奖励函数部分是 RLVR 的核心：不需要 reward model，而是通过可验证答案判断生成是否正确。数学题场景下，reward function 会从模型输出里抽取答案，并和标准答案比较，返回 0 或 1。这个设计让训练信号非常明确，也解释了为什么这个流程叫 Reinforcement Learning from Verifiable Rewards。

单轮工作流 `RLVRWorkflow` 是第一个重要练习点。它要完成 `prompt -> generation -> reward -> trajectory` 的闭环。`gen` 里需要构造 `ModelRequest`，传入 `rid`、`input_ids` 和 `gconfig`，然后用 `await engine.agenerate(req)` 调用 SGLang 生成。这里的 `engine` 不是本地模型对象，而是 AReaL 抽象出来的 inference engine，背后连接远程 SGLang server。

我在这里还记录了一个尚未完全理解的问题：`async`、`event loop` 和 `executor`。现在的理解是，`await engine.agenerate(req)` 是异步等待远程生成；reward 计算如果可能阻塞 event loop，可以通过 executor 放到线程或进程池里运行。这个部分我还需要之后回顾，因为它关系到为什么 notebook 里有些地方不能直接 `asyncio.run()`，而要用 `await` 或 `asyncio.to_thread(...)`。

构造 trajectory 时，我学到 `input_ids`、`logprobs`、`loss_mask` 必须对齐。完整训练序列是 prompt tokens 加 generated tokens。prompt 部分不是模型这次要优化的 action，所以 `logprobs` 用 `0.0` 占位，`loss_mask` 是 0；generation 部分使用 `resp.output_logprobs`，`loss_mask` 是 1。这个对齐非常关键，否则 PPO loss 会在错误的位置上训练。

测试单轮工作流时，我们踩到了 tokenizer 的实际坑：`tokenizer.apply_chat_template(...)` 必须保证返回普通 list，而不是 `BatchEncoding`。所以要用 `return_dict=False`。否则 HTTP 请求发给 SGLang 时会报 `Object of type BatchEncoding is not JSON serializable`。这个错误让我意识到 workflow 里传给 engine 的数据必须是可 JSON 序列化的简单结构。

## 将单轮工作流拓展成多轮带反思的工作流

反思提示部分引入了多轮 workflow 的思想：如果第一轮答案错了，不是直接结束，而是把之前的回答和反馈放回上下文，让模型反思并再次尝试。这里我理解到 reflection 不是一个神秘模块，本质上是手动构造新的 prompt，让模型看到自己的前一次输出和“答案不对”的信号。

带反思的 workflow 和单轮 workflow 的主要区别在于，它会维护 turns。每一轮生成后都会计算 reward；如果已经正确，就可以提前结束；如果不正确，就把反思提示追加到 messages 里继续下一轮。这里的 `max_turns` 控制最多反思几次，`turn_discount` 控制越晚答对奖励越低。比如第一轮答对 reward 是 1.0，第二轮才答对可能是 0.9。这个设计鼓励模型尽早给出正确答案。

测试 reflection workflow 时，我学到判断是否正确不能只看模型输出文本，还要看返回的 trajectory 是否结构正确：`input_ids`、`logprobs`、`loss_mask`、`rewards`、`attention_mask` 都要存在，shape 要合理，prompt 部分 mask 为 0，生成部分 mask 为 1。如果第一轮答对，最终 reward 应该保持 1.0，而不是被错误地乘上 discount。

## 让 reflection 工作流对每个问题生成多个答案

这一节把单条 trajectory 扩展成 group。GRPO/Grouped rollout 的直觉是：同一个问题采样多个答案，然后在组内比较 reward，用相对优势来训练。这里我理解到 `n_samples > 1` 不是让一次 `engine.agenerate` 直接返回多个答案，因为当前 inference engine 不支持 `n_samples > 1`。正确方式是对同一个问题多次调用生成，每次 `n_samples=1`，再把结果合并。

我们之前遇到过一个错误：workflow 返回了 `list[dict]`，但 AReaL 的 workflow executor 期望单次 `arun_episode` 返回一个 `dict`。所以 grouped workflow 里需要用 `concat_padded_tensors(results)` 把多个 sample pad/concat 成一个 batch dict。最后 rollout_batch 返回的外层 list 对应不同问题，内层 dict 的 batch dimension 对应同一问题的多个 samples。

这一节也让我更清楚地区分了两个“batch”：一个是 data generator 给出的多个问题；另一个是同一个问题生成多个答案形成的 group。比如 `rewards: tensor([1., 0.])` 表示同一个 prompt 的两个 sampled answers，一个对一个错。这是 GRPO 可以工作的基础。

## 将工作流接入强化学习训练流程

同步训练部分把前面的 workflow 接入 actor。这里我理解了 rollout 和 actor 的关系：rollout engine 负责用当前策略生成 trajectories，actor engine 负责训练当前策略。一次同步训练 step 大致是：先 `actor.rollout_batch(...)` 得到 batch，再 `actor.compute_logp(batch)` 重新计算当前 actor 对这些 token 的 logprob，写入 `prox_logp`，然后 `actor.compute_advantages(batch)`，再 `actor.ppo_update(adv_batch)` 做 PPO 更新，最后同步权重给 rollout。

这里有一个重要细节：`batch` 是 `list[dict]`，所以不能写 `batch["prox_logp"] = logps`。应该逐条写回：

```python
for traj, logp in zip(batch, logps):
    traj["prox_logp"] = logp
```

这和 AReaL trainer 里的真实写法一致。之后 `compute_advantages` 的返回值也应该接住，传给 `ppo_update`。这让我理解到 notebook 练习虽然简化，但数据结构仍然要尊重真实 trainer 的接口。

actor 初始化部分也有一个版本适配点：当前 AReaL 里 `FSDPPPOActor` 需要先 `create_process_group(...)`，再 `initialize(...)`。因为 logger、device mesh、parallel helper 等状态是在 process group 创建时初始化的。如果直接 initialize，会报 `FSDPPPOActor object has no attribute logger`。

权重同步是同步训练里最容易误解的一部分。原本 notebook 用的是 `WeightUpdateMeta.from_fsdp_xccl(...)`，表示用 NCCL/XCCL 从 actor 直接把权重广播给 SGLang。这在多 GPU 合法拓扑下速度快，但在我们单 GPU notebook 中 actor 和 SGLang 都在 GPU 0，NCCL 会报 `Duplicate GPU detected`。所以单卡练习应该改成 disk weight update，也就是 actor 把权重保存到磁盘，SGLang 再从磁盘加载。对应 YAML 里也要把 `actor.weight_update_mode` 改成 `disk`。

Jupyter 里还会遇到 event loop 问题。某些 AReaL/SGLang 内部函数会调用 `uvloop.run` 或类似逻辑，但 notebook 本身已经有一个 running event loop。因此在 notebook 中执行权重同步时，用：

```python
await asyncio.to_thread(actor.update_weights, weight_update_meta)
```

会更稳。它的意思是把这个阻塞同步过程放到普通线程里执行，同时 notebook 仍然等待它完成。

异步训练部分我目前还没有完全展开，但已经能理解它和同步训练的区别：同步训练是 rollout 完再 train，train 完再 update weights；异步训练会让数据准备和训练之间有更多重叠。这个部分之后 review 时可以重点看 `prepare_batch`、rollout queue、weight version 这些概念。

## 今天整体收获

今天这份 practice notebook 最重要的学习，不是某一个 TODO 的答案，而是把 AReaL 里几个组件串起来了：GSM8k 数据提供 prompt 和 answer；workflow 把 prompt 变成 rollout trajectory；SGLang server 提供生成能力；reward function 提供可验证训练信号；actor 负责 PPO 更新；weight update 把训练后的 actor 权重同步回 rollout。只要其中任何一环的数据结构、异步调用、进程状态或设备拓扑不对，整个 notebook 就会卡住或失败。

我也学到，做这类 RL system notebook 练习时，debug 不能只看 Python traceback。很多问题发生在 notebook 外部：SGLang 子进程、CUDA JIT、端口占用、GPU 显存、HTTP `/generate`、detokenizer heartbeat、NCCL group。以后遇到 rollout 卡住，要先判断是 workflow 逻辑错、server 不健康、tokenizer 输出类型错，还是权重同步方式不适合当前硬件。

最后，我们把 SGLang server 的排障流程固化成了 `sglang-notebook-debug` skill。以后如果 server 又卡住，可以按 skill 里的流程检查：进程、端口、GPU、日志、`/health`、真实 `/generate`。这个比只看 notebook cell 是否还在运行可靠得多。
