# AutoModel SC streaming：PR3 设计与实现

状态：**设计已确认，实现与分级验收进行中**。本文覆盖
[issue #5090 的 PR3](https://github.com/NVIDIA-NeMo/RL/issues/5090)，初期仅支持 `in_order`。

## 开发基线和已核实事实

2026-11-05 查询：[PR1 #4963](https://github.com/NVIDIA-NeMo/RL/pull/4963) 仍 OPEN。
本分支直接基于其最新 HEAD `047d31a7f75dda062e82e29561a7e8460dc24b7c`；
已 fetch 的 main 为 `6ef563814b7e98756892316452afbea63b151d7d`，不包含该 HEAD。
PR3 不从之前用于 LA1 实验的个人补丁分支开发，也不假设 main 已支持 AutoModel SC。

AutoModel pin 为 `b916107a5ce8e77a3f18682f8b0eaa46a4b54389`，Bridge pin 为
`7e9dcd609edf2bb9478d3fa4d9b9f8ee3875c16e`。
已核对 sibling checkout 的实际 SHA 和所读文件无本地修改。
从 `nemo_rl.1030.sqsh` 直接读取到 PyTorch **2.13.0+cu130** 源码；主机 uv 缓存
2.10.0 不作为实现依据。[依赖证据](../../validation/pr3/source-evidence.json)记录源码哈希。

调用链及各层职责：

| 层 | 当前行为 | PR3 最小修改 |
| --- | --- | --- |
| `SingleControllerActor._train_pump` | 选完整 prompt 组，逐 chunk LP/advantage/backward，整步 finish | 给 AutoModel streaming 传现有 `group_multiple`；补取消时的 train-step 收尾 |
| `TQPolicy.begin/train_microbatches_from_meta/finish/abort` | 已支持 N 次 dispatch；一次 dispatch 等待全部 rank 完成 | 复用 API、结果聚合及 replica-leader 去重 |
| `TQWorkerMixin.*_presharded` | fetch/materialize 后调用 backend | 保持传输职责，不存模型梯度或重复实现归一化 |
| `AutomodelPolicyWorkerImpl` | 一步只接受一个完整 batch；梯度保留到 finish | 扩展 `_TrainStepState`、跨 dispatch 累积、finish 统一归一化与失败状态 |
| Megatron 对照 | placeholder=1，finish 计数/缩放，`no_sync` 后集中 finalize | 复用计数/指标思想；不照搬其 DDP 同步实现 |

## 一步内的状态和提交契约

扩展现有内部 dataclass，显式保存：loss/normalizer、GBS/MBS、normalization mode、
dispatch/内部 microbatch 数、已提交到 worker 的物理行数、FP64 本地有效 sequence/token
计数、detach 后的标量指标、buffer/router snapshots，以及 execution-failed 状态。
梯度留在模型参数的 FSDP 分片 `.grad`，不放进 Ray/TQ，也不跨 dispatch 保留激活图。

- **begin**：只在 idle 且未被 commit failure 封禁时执行；snapshot、zero_grad、train mode。
- **dispatch**：每个 rank 行数相等、正数且为 MBS 倍数，累计物理行数不超过 GBS/DP；
  一次 dispatch 完整结束才接受下一次。执行失败后禁止继续 dispatch/finish，必须 abort。
- **finish**：要求至少一次成功 dispatch、全局有效 token/sequence 非零，再归一化、clip、
  optimizer、gate-bias、scheduler，各一次。只有成功返回才允许 controller 推进权重版本/refit。
- **abort**：不更新参数/optimizer/scheduler，恢复可变 buffers/router loads，清空梯度、
  指标及 dispatcher 临时引用；重复调用无副作用。不承诺恢复 RNG 或重放随机 rollout。
- **commit failure**：从第一次不可逆操作前升起原有 fence；即使 cleanup 成功也不能撤销。
  optimizer/gate-bias/scheduler/cleanup 任一提交阶段失败，要求 checkpoint 恢复。

完整 prompt-group 消费、target_step 和 sample ID 唯一性仍由共享 SC 的 claims/ledger 保证。
`sample_mask=0` 不等于缺失物理行；有效 sequence 数不能拿来检查 GBS。
现有 controller 会跳过全局无有效 token 的 chunk，因此 worker 的累计物理行数可能小于
GBS：完成性由 controller 的已选完整组计数验证，worker 不强行要求有效行数等于 GBS。
全局零有效 token 的整步报错/abort，不产生空 optimizer update。

## 损失与指标归一化

使用现有 `ClippedPGLossFn` 和 `automodel_forward_backward`，不另写一套 PPO/GRPO loss。
首 dispatch 若已是完整 GBS，沿用 PR1 的实际分母路径，保留 full-batch/V1 回归基线；
首 dispatch 为部分 batch 时进入 deferred mode，后续 dispatch 均保持该模式。

设 `m_i` 是 sample mask，`t_ij` 是已经移位到 next-token 的 token mask：

```text
S = Σ_ranks,chunks,i m_i
T = Σ_ranks,chunks,i,j>0 m_i · t_ij
token objective    = Σ m_i · t_ij · loss_ij / T
sequence objective = Σ m_i · mean_valid_tokens(loss_i) / S
```

Deferred mode 给 loss 传 `global_valid_seqs=global_valid_toks=1`，保留 sequence 内部的
token mean；各 chunk 累积整步 numerator。保留现有 backward 的 `dp_size*cp_size`
补偿（本阶段 CP=1），finish 将有效计数在**逻辑 DP group**求和，并在 clip 前按 loss type
对每个梯度分片乘 `1/T` 或 `1/S`，不能再除 dispatch 数、MBS 数或 GBS。
这在实数运算上与 full-batch 等价；浮点求和/缩放次序变化须由 parity 测量，不能宣称 bitwise。

指标按 loss 自带 `metric_normalizations` 分别除 T/S；`NONE` 的计数/局部统计和 min/max
不缩放。保留每个内部 microbatch 的贡献，经现有 `aggregate_training_statistics` 和
`aggregate_step_metrics` 一次聚合；`global_loss` 仍是一整步的全局标量形状。
LR 在 scheduler 前采集，global_valid_* 表示整步总数；原始 count、extrema 不能当 mean。
只接受已审计的 ClippedPG 指标映射，避免未知 loss 默默套用错误分母。

**不能统一后缩放的附加目标首版必须显式拒绝 streaming**：非零
`positive_example_nll_weight` 使用 microbatch 内 correct-token 分母；AutoModel 的非零
MoE aux loss 通过独立 autograd scaler 注入梯度，见下文。Full-batch 原行为不因此改变。

## FSDP 同步与显存

首版保持 PR1 每个内部 backward 的同步和正常 reshard；不使用跨 dispatch `no_sync`，
finish 不额外 all-reduce 已规约过的梯度。实际 PyTorch 2.13
`_fsdp_collectives.py:744` 对已有 `sharded_param.grad` **累加新规约分片**；旧分片不会再次规约。
线性规约与逐 chunk 累加可交换，配合原有 DP 补偿和最后一次归一化得到正确整步梯度。
EP 的 dense/expert mesh、归约因子和裁剪保持 AutoModel 现有实现，须用 EP 实测确认。
特别保留 `scale_grads_and_clip_grad_norm` 内 `DP/ep_shard_size` 的 expert 补偿，不能把
dense 参数与 expert 参数默认看作同一个规约 group，也不能在 worker 外再补偿一次。

额外持久显存以梯度分片、buffer snapshots、小型 counters 为主；仅当前 microbatch
保留完整梯度/激活，不随 dispatch 数线性增长。规约 dtype 仍为现有 FP32，实际持久
grad dtype 和峰值在实验记录，不把 reduce dtype 当作 master-grad dtype。
禁同步路径会保留未分片 accumulated gradients，且仅设回 True 不会凭空发起一次 backward；
本 PR 不引入手工调用 FSDP 私有 post-backward 的结束同步方案。
通信次数不因 PR3 减少，加速预期来自当前 step 剩余 rollout 与前部训练重叠，不能承诺收益。

## MoE 和取消生命周期

Gate `_cumulative_expert_load` 是 optimizer-step 状态：跨 dispatch 保存，LP 在 eval mode
不得增加它；finish 只调用一次现有 `_update_moe_gate_bias_if_supported`。
保留 AutoModel checkpoint-recompute 的 load-preservation context，避免重算重复累计。
abort 恢复 snapshots，包括 gate 的普通 Python tensor 属性；`_last_expert_load` /
`_last_aux_loss` 等诊断缓存也恢复/清空，防止 LP 或取消污染下一步。
每个 dispatch 完整 backward 返回后可释放 `release_moe_dispatcher_state` 的临时图引用；
不能在 backward/recompute 未结束时释放。finish/abort 同样执行 cleanup。

实际 `MoEAuxLossAutoScaler.backward` 返回独立 `ones_like(aux_loss)*main_loss_backward_scale`，
并不继承主 loss 的上游缩放。因此把混合梯度整体除 T/S 会改变现有 aux 梯度权重。
setup 可按配置提前拒绝已知非零 aux；worker 在首个 partial dispatch、任何 forward 前再检查
**实际模块**的 aux coefficient，非零即拒绝 deferred mode，不能仅检查 HF config。
这样不需要把 async 调度配置塞进 worker API，也不会禁止既有 full-batch aux 行为。
Qwen3.5 计划 recipe 已将 aux 置零，但这不验证非零 aux 的正确性。
将来支持非零 aux 必须隔离其梯度/分母或改变经过验证的 aux API，单独审查。
Qwen3.5 自身 gate-bias factor=0；可更新 bias 的生命周期要另用小型原生 Gate/MoE 测试覆盖。

共享 SC 当前 teardown 未调用 trainer abort。PR3 加一处最小生命周期修复：跟踪已发出的
train RPC，在取消/失败时先等待其结束，再对尚未 commit 的 open step abort。
ownership 必须在发送 begin 前建立，commit-in-progress 必须在发送 finish 前建立；
否则取消恰好发生在 begin/finish RPC 返回前时，单靠 await 后设置的 `step_open` 会漏清理。
`asyncio.to_thread` 被取消不会停止底层 Ray/GPU 工作，不能并发发送 abort 清梯度。
若 RPC 无法在有界期限内结束、rank collective 出错或 commit 结果不明，终止/恢复 worker；
禁止当作“abort 成功”继续下一步。已 commit 的步骤不回滚；保留原异常及 cleanup 失败原因。
Checkpoint 仍在已提交步边界保存；不序列化半步 GPU 梯度。恢复从最后已提交 checkpoint
和共享 rollout ledger 开始，半步组不可在旧梯度上重复训练。

## 范围、配置和实施顺序

只放开 full-batch threshold 限制：`min_groups_for_streaming_train < num_prompts_per_step`。
保留 text GRPO、完整 prompt 组、full-parameter FSDP2、TP=CP=1、固定 microbatch、
无 packing/dynamic batching/HSDP/CPU offload/LoRA/draft/MTP/VLM/量化/PPO/distillation/colocation，
disaggregated async vLLM、`in_order`、lookahead=0、零 prompt-drop budgets。
首版 advantage 只资格化逐 prompt 的 `grpo`，不把整批 whitening 的算法直接用于 chunk。
Lookahead>0 和其他 sampler 不借本 PR 自动放开。

每个 dispatch 必须满足 DP 及 train/logprob MBS 对齐。复用现有 sampler 参数：

```text
sample_multiple = DP × lcm(train_micro_batch_size, logprob_batch_size)
group_multiple = sample_multiple / gcd(sample_multiple, num_generations_per_prompt)
```

在 claim 前按 group_multiple 取整，未对齐尾组留在 buffer；验证完整 step 可整除且阈值存在
可行对齐大小。不能只检查配置阈值，因为 greedy selection 可能返回 5 个而不是 4 个组。
Qwen3.5 的 DP16、MBS1、4 responses 得到 multiple=4：8 组一步可分 4+4；min=1 也不能
让单个 prompt 组直接送进 16 个 training ranks。沿用共享 capacity 验证，含正在训练的 claims。

1. **先扩展固定输入 harness**：同一初始模型/optimizer/scheduler/RNG，V1/full-batch SC/
   2 次和 3 次 streaming；先 shard 再切各 rank 的输入，固定 padding、microbatch 顺序。
2. **实现 worker 数值/状态契约**：counts、normalizers、metrics、失败/abort；V1 不改，
   full-batch 严格回归先保持，再接受有明确误差界的 streaming。
3. **接入共享 alignment/cancellation**：扩展已有 sampler/pump tests；不增加 AutoModel sampler。
4. **GPU 分级验收**：dense → native MoE EP2/bias/recompute → EP16 fixed-input replay →
   真 TQ/vLLM 的取消恢复与 GPU 时间轴 overlap；全部门禁后才启动长跑。
5. **长跑**：Qwen3.5-35B-A3B-Base，16 train GPU EP16 + 8 rollout GPU TP8，batch=8×4、
   MBS1、总长4096/生成2048、同 optimizer/seed，streaming threshold4，lookahead0，
   目标300步。按用户最新要求复用同事已有 baseline，不再提交 full-batch 300步对照。
   先检查原任务首段，成功后继续同一任务，按实际 checkpoint/时间限制续段。

[验收矩阵与命令](../../validation/pr3/ACCEPTANCE.md)给出每一门禁的操作、证据与判据。
本阶段已核实源码，并在 debug allocation `21326763` 成功导入实际依赖、采集
[runtime manifest](../../validation/pr3/dependency-runtime.json)；节点随后释放。
实施阶段已通过 dense FP32、多 dispatch native MoE EP2、activation checkpointing、
真实 Gate bias、取消及带 ready/in-flight 任务的恢复检查。BF16 整体梯度门槛及独立
optimizer 执行对照通过，原有逐参数严格门槛未通过，不能混为一项。
具体结果、证据及仍未验证的配置以
[执行记录](../../validation/pr3/RESULTS.md)和
[BF16 数值说明](../../validation/pr3/BF16_NUMERICS.md)为准。
