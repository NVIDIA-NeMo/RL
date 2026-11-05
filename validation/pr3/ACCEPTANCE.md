# PR3 acceptance plan

**设计已确认，实施与验收进行中。** 下文保留验收规格，实际通过、失败及未运行状态见
[RESULTS.md](RESULTS.md)。既有 PR1 的通过记录不作为本分支 streaming 通过的证据。

## 固定环境和输入

- 所有 pytest/torchrun/训练在 Slurm allocation 内执行；jobname 统一 `yuekaiz-` 前缀。
- 基线/依赖 SHA 见 [source-evidence.json](source-evidence.json)。进入实施时再次核实 upstream
  PR1 HEAD；若更新，先记录是否 rebase，不能混用不同版本的比较结果。
- 初始用 `nemo_rl.1030.sqsh` 和其已有 AutoModel worker venv，`uv --no-sync`，避免隐式更新。
  若实际依赖不兼容，明确记为环境受限并保存错误，不能把跳过当通过。
- harness 复用 `tests/functional/automodel_sc_full_batch_parity.py`，增加
  `--local-batch-size`、`--dispatch-sizes`（每 rank 行数，用逗号分隔，和等于 local batch）。
  这两个参数已实现；下列命令在已分配容器内执行。
- tiny dense/tiny native MoE 沿用 `test_automodel_split_parity.py` fixtures；每 rank 6 行，
  比较 dispatch `6`、`3,3`、`1,2,3`，每种 token/sequence loss 至少连续 3 个 optimizer steps。
  保留全局 sample IDs；固定每 rank microbatch 顺序、padding、mask、LP/advantages 和所有 RNG。
  先对全量 metadata shard 一次再切各 rank；chunk-first 的生产 sharding 另做独立验收。

```bash
# 在分配的容器内设置；实施前初始化本 worktree 的 pinned submodules。
export PR3_REPO=/lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming
export PR3_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker/bin/python
export PYTHONPATH="$PR3_REPO:$PR3_REPO/3rdparty/Automodel-workspace/Automodel"
export UV_NO_SYNC=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
cd "$PR3_REPO"

# 命令 A：CPU 状态和两 GPU parity；显式选择 automodel，避免默认过滤。
uv run --no-project --python "$PR3_PYTHON" python -m pytest -q \
  --automodel-only \
  tests/unit/models/policy/test_automodel_split_state.py \
  tests/unit/models/policy/test_automodel_split_parity.py

# 命令 B：扩展后可对已准备的模型/recipe 定点重放；PR3_MODEL、PR3_RECIPE 必须是真实路径。
for dispatch in 6 3,3 1,2,3; do
  uv run --no-project --python "$PR3_PYTHON" python -m torch.distributed.run \
    --standalone --nproc-per-node=2 tests/functional/automodel_sc_full_batch_parity.py \
    --model "$PR3_MODEL" --recipe "$PR3_RECIPE" --steps 3 \
    --local-batch-size 6 --dispatch-sizes "$dispatch" --strict-determinism \
    --rtol 1e-5 --atol 1e-7 --source-sha "$(git rev-parse HEAD)" \
    --output "validation/pr3/results/parity-$dispatch"
done

# 命令 C：共享逻辑测试，新增 AutoModel alignment / cancellation cases。
uv run --no-project --python "$PR3_PYTHON" python -m pytest -q \
  -m 'not mcore and not hf_gated' \
  tests/unit/single_controller/test_automodel_config.py \
  tests/unit/single_controller/test_train_step_lifecycle.py \
  tests/unit/single_controller/test_sampler_interface.py \
  tests/unit/single_controller/test_train_pump_e2e.py \
  tests/unit/single_controller/test_checkpoint_recovery_matrix.py
```

Dense/HF 命令 B 加 `--force-hf`；native MoE recipe 指定 EP2/HybridEP，不能 force HF。
2-GPU tests 和大模型 replay 复用同一比较逻辑。大模型另加 TE FusedAdam/raw state
采集，不用小模型 AdamW 的结果替代真实 optimizer 资格化。

## 分阶段门禁（原设计规格，执行状态见 RESULTS.md）

| 门禁 | 可执行操作/配置 | 必需判据与保存证据 | 设计阶段状态 |
| --- | --- | --- | --- |
| G0 baseline | 记录 HEAD/submodule/image/包版本；命令 A 既有 case | PR1 full-batch/V1 严格回归保持；测试 collection/skip 明细完整 | 源码已核实，测试未执行 |
| G1 dense parity | A/B：2 GPUs，EP1，固定输入，`6` / `3,3` / `1,2,3` | 每 rank **finish 内完成归一化后、clip 前**梯度；所有参数/buffers、完整 optimizer/scheduler 状态，连续3步 | NOT RUN |
| G2 normalizers/metrics | 同 G1，长短序列混合、跨 rank mask 不同、局部全 masked、整步全 masked、token/sequence 两种 loss，reference KL/TIS 开启 | counts 精确；loss/entropy/genKL/token diagnostics 按各自分母；零步不 commit；超过 GBS/不对齐/失败后 finish 拒绝 | NOT RUN |
| G3 native MoE | A/B：native EP2，activation checkpointing on/off；aux=0；另加真实 Gate bias factor>0 fixture | dense/expert/router 梯度及状态通过；LP 插入 chunk 间不改 grad/load；重算不重复 load；bias 每 commit 一次 | NOT RUN |
| G4 lifecycle failure | dispatch1/2 后 abort；inject pre-backward、post-backward、clip、optimizer、gate-bias、scheduler、cleanup failure；abort 后再开始 | 未 commit 的可变状态恢复、无残余梯度/dispatcher 图；重复 abort 安全；提交失败永久 fence；整数 counters 精确 | NOT RUN |
| G5 shared alignment/cancel | C：Qwen shape DP16,G=4；ready 4/5/7/8 组；取消 begin、等待下一 chunk、正在 RPC、commit 阶段 | 4/5/7 ready 最多 claim4，8 可 claim8；未对齐尾组不丢不重；abort 不和工作并发；未结 RPC/不明 commit 走 checkpoint recovery | NOT RUN |
| G6 EP16 replay | 16 training GPUs，Qwen3.5-35B-A3B-Base，full32 rows 对两次16 rows，至少3个真实固定 GRPO updates，TE FusedAdam | 每 rank preclip gradient、参数、raw optimizer/master/moments/remainders、scheduler、gate/buffer；沿用 G1/G3 判据并记录 dtype | NOT RUN |
| G7 real overlap | debug allocation，dense Qwen2.5-1.5B；2 train+2 rollout GPU；8×4，threshold2；人为 gate 后半 rollout | 前部 backward 开始时后部 rollout **尚未完成**；同一 step ≥2 dispatch；GPU generation execute 与 training F/B 时间区间相交，不能只看 RPC/CPU 时间 | NOT RUN |
| G8 cancellation/resume | G7 产生至少2个 dispatch 的 open step 后取消；从最后 committed checkpoint 重启，带 ready+in-flight 组 | 无半步 optimizer/scheduler/version 前进；sample claims/IDs 不丢不重；恢复后至少3步；无 stale gate/grad；不宣称随机轨迹等同 uninterrupted | NOT RUN |
| G9 long run | Qwen3.5 EP16，streaming4、lookahead0、300步；复用同事 baseline | 保存所有 native/W&B scalars、dispatch分布/有效样本/版本、checkpoint及峰值显存；失败主动监控，不删异常点 | 见 RESULTS.md |

G1 必须移动/扩展原 harness 的比较位置：原来在 `train_microbatch` 后直接比较 `.grad`，
streaming 此时尚未归一化；不能拿 raw numerator 梯度与 full-batch mean 梯度比较。
统一在 `scale_grads_and_clip_grad_norm` 入口采集，既防漏缩放，也防错把裁剪后相近当 parity。

初始数值门槛：full-batch 既有 strict case 仍 `rtol=atol=0`；FP32 streaming 采用
`rtol=1e-5, atol=1e-7`。BF16 资格化单独记录并在执行前冻结阈值：梯度 relative-L2≤1%，
非零整体梯度 cosine≥0.9999，norm 相对误差≤1%；optimizer moments 初始
`rtol=1e-2, atol=1e-6`，FP32 master 参数绝对误差≤1e-7，BF16 参数误差≤1 ULP。
整数/shape/keys/sample IDs/version/scheduler step 必须精确。保存逐 tensor 最大误差及
梯度方向，不只报一个全局均值；超门槛先定位求和/归约/dtype/随机性，不能事后放宽后宣称通过。
若这些拟定 BF16 门槛无法解释或实际配置不适用，停止该资格化并将修订依据提交审查。

## 真实训练命令和配置

实施阶段新增薄 recipe，继承现有 Qwen3.5 full-batch recipe；除了 threshold 与 output identity，
两边配置经规范化比较应相同。不能在这里顺带引入 LA1 或改变 loss。示例启动命令可直接使用
既有 recipe 和命令行 override（需先完成 PR3 实现）：

```bash
# 在各自的3-node allocation内；输出/W&B身份由 sbatch launcher 分别设置。
uv run --no-project --python /opt/nemo_rl_venv/bin/python python \
  examples/run_grpo_single_controller.py \
  --config examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-base-3n8g-automodel-ep16-single-controller-full-batch.yaml \
  grpo.max_num_steps=300 async_rl.min_groups_for_streaming_train=4
# 最终只提交 streaming；full-batch 数值对照保留在 fixed-input harness。
```

配置：8 prompts×4 responses=32，MBS1，16 train/EP16/TP1/CP1，8 rollout/TP8，
总长4096/生成2048，BF16，aux=0/MTP=0，LR1e-6/warmup10/WD0.1，原有 TIS2。
检查首段后原 job 继续，资源时间到再从 committed checkpoint 续段，不为“短跑成功”重置实验。
若实际 dispatcher 大量一步只调用一次，必须报告，不能用这种曲线证明 streaming 加速。

收敛/性能对比：reward、length、loss、entropy、gen/policy KL error、grad_norm、LR、
valid/masked counts、训练吞吐、generation throughput、step time、first-backward 延迟、
dispatch 数与 GPU peak memory。图中保留 step time；在线 validation/acc 当前 SC 不支持，
明确标记 unavailable，不用训练 reward 冒充 validation。
同一 seed 的独立 rollout 曲线不要求逐步相等，也不能单独证明收敛质量等价。

所有结果记录 source SHA、镜像/依赖版本、命令、job ID、native log/W&B ID、checkpoint 和
passed/failed/not-run/environment-limited；每10分钟查真实 Slurm 状态及进度，异常主动处理。
旧的900步实验保持 hold。长跑只有 EP16 前置验证成功后才启动。
