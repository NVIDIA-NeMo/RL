# HybridEP Dispatcher A/B Measurement Design

## Goal

Measure the performance and numerical trade-offs of HybridEP against each
recipe's non-HybridEP `alltoall` dispatcher for three MoE GRPO workloads on
OCI-HSG:

- Qwen3-30B-A3B, 4 nodes × 4 GPUs, synchronous colocated rollout.
- Qwen3-235B-A22B, 16 nodes × 4 GPUs, synchronous colocated rollout.
- Nemotron3 Super 120B-A12B, 32 nodes × 4 GPUs, asynchronous one-off-policy
  rollout with 16 policy and 16 generation nodes.

The primary comparison is policy-training and LogProb step time and
tokens/second/GPU. Padding diagnostics and short-run numerical stability are
reported separately.

## Fixed Sources and Runtime

- NeMo-RL branch:
  `sna/qwen30-pr2964-f725-pin-oci-20260727`.
- DeepEP:
  `f725d29699f5bda9ba789456bb9579af69844685`.
- Megatron-LM:
  `4d04e7625c5e84f984a9f01aef58cb006b0aa7ac`.
- Megatron-Bridge for Super:
  config-reuse branch based on
  `45e4e4be2591186ac795eea4205c44089b45fcfd`.
- Container SHA256:
  `5e9f6066897057d8701e0722a5023c08a997f10f4eff61340c249ed73f7c33fc`.
- All arms run 20 GRPO steps with checkpointing and W&B disabled, TensorBoard
  enabled, and the recipe seed unchanged.

## Model Contracts

### Qwen3-30B-A3B

- Recipe:
  `examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n4g.yaml`.
- Shape: 4 nodes × 4 GPUs, segment size 4.
- Existing recipe dispatcher: `alltoall`.
- Existing paired arms:
  - clean f725 HybridEP job `5639548`;
  - recipe-default `alltoall` job `5645086`;
  - HybridEP padding diagnostic job `5644880`.

### Qwen3-235B-A22B

- Recipe:
  `examples/configs/recipes/llm/performance/grpo-qwen3-235b-16n4g.yaml`.
- Shape: 16 nodes × 4 GPUs, segment size 16.
- Policy: TP2, PP4, EP16, CP2 with sequence parallelism.
- Generation: colocated vLLM TP8.
- Sequence packing: enabled with fused loss.
- The recipe itself selects the flex/HybridEP dispatcher. A recipe-inheritance
  mode is therefore not a valid non-HybridEP control.

### Nemotron3 Super 120B-A12B

- Recipe:
  `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-async-1off.yaml`.
- Shape: 32 nodes × 4 GPUs, segment size 8.
- Policy: 16 nodes, TP2, EP16.
- Generation: 16 dedicated nodes, vLLM TP4.
- Async GRPO: enabled with one-step trajectory age and in-flight refits.
- Sequence packing: enabled.
- `NCCL_NVLS_ENABLE=0` remains mandatory.

## Launcher Interface

The existing launcher gains one explicit dispatcher mode:

- `DISPATCHER_MODE=hybridep`: force the flex dispatcher, HybridEP backend, and
  32 SMs.
- `DISPATCHER_MODE=recipe`: do not add dispatcher overrides.
- `DISPATCHER_MODE=alltoall`: force
  `policy.megatron_cfg.moe_token_dispatcher_type=alltoall`.

`alltoall` is required for Qwen3-235B because its recipe already selects
HybridEP. The launcher must reject unknown modes, record the selected mode in
`submission.env`, and keep padding logging disabled unless the mode is
`hybridep`.

A reusable model profile is added for Qwen3-235B with the exact recipe, node,
GPU, segment, time-limit, 20-step, and f725 values above. Existing Qwen3-30B
and Super profiles remain unchanged.

## Experiment Matrix

Each new model uses three distinct arms:

1. **HybridEP diagnostic**
   - `DISPATCHER_MODE=hybridep`.
   - Enable bounded padding logging on rank 0 for at most 4,096 calls.
   - Enable expert-group reduction so each record contains group raw, padded,
     and added-token totals.
   - Use only for padding and numerical diagnostics, because the extra
     reduction and log I/O perturb performance.

2. **Clean HybridEP performance**
   - `DISPATCHER_MODE=hybridep`.
   - Disable padding logging.
   - Use for policy, LogProb, generation, and end-to-end performance.

3. **Clean non-HybridEP control**
   - `DISPATCHER_MODE=alltoall`.
   - Disable padding logging.
   - Keep every other recipe, source, image, seed, topology, batch, and
     checkpoint setting identical.

The Super diagnostic arm also serves as the first full validation of the
AutoBridge config-reuse fix. Clean Super performance arms are submitted only
after that job passes the 20-step gate.

## Metrics

For clean A/B arms, report steps 5–20:

- mean and median total step time;
- ratio-of-sums end-to-end tokens/second/GPU;
- policy-training time and tokens/second/GPU;
- policy/reference LogProb time and tokens/second/GPU;
- generation time and tokens/second/GPU as a secondary metric.

The headline speedup uses:

```text
speedup_percent = (HybridEP throughput / alltoall throughput - 1) × 100
time_reduction_percent = (1 - HybridEP time / alltoall time) × 100
```

For diagnostic arms, report:

- sampled call range and call count;
- `sum(group_added_tokens) / sum(group_raw_tokens)`;
- median, p95, and maximum per-call `group_overhead_pct`;
- the exact token shape and training phase for material outliers.

For numerical stability, compare steps 5–20 reward, KL penalty, entropy, and
gradient norm; validation accuracy and average length at steps 10 and 20; and
median/p95 token/probability consistency errors. Large error outliers are not
summarized with a simple mean.

## Interpretation Rules

- Padding is transient activation/compute/communication overhead, not
  persistent disk storage.
- A per-call maximum is not presented as overall overhead. Weighted totals are
  primary.
- A 20-step numerical comparison can establish smoke-level consistency only;
  it cannot prove equal long-horizon convergence.
- Same-segment runs are preferred. If Slurm assigns different nodes, report
  the node lists and treat differences near run-to-run noise cautiously.
- A performance arm with diagnostic logging is never substituted for a clean
  arm.

## Submission and Failure Handling

- Pull the pushed branch and initialize recursive submodules before each
  submission.
- Select the highest current user-level FairShare account.
- Require `sbatch --test-only` before every real submission.
- Record one canonical job ID and complete provenance in the run directory.
- Monitor every job for at least five minutes and scan for actor loss, timeout,
  NCCL, CUDA, OOM, NaN, Inf, and model-config failures.
- Do not fall back to older DeepEP commits because f725 already passed the
  Qwen3-30B 20-step gate. Investigate model- or topology-specific failures
  before changing the dependency.

## Reporting

The existing secret-free HTML report remains the live dashboard. It records
the exact commits, jobs, node lists, terminal states, sampled padding window,
performance deltas, numerical metrics, caveats, and failure evidence for all
three models.
