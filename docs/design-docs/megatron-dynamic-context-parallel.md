# Megatron dynamic context parallelism

Dynamic CP lets the Ray driver assign short sequences to independent model
replicas and longer sequences to larger context-parallel groups within the same
optimizer step. It uses the Megatron-Core version pinned through Megatron Bridge.
No dependency source edits or submodule changes are required.

## Configuration

```yaml
policy:
  megatron_cfg:
    context_parallel_size: 1
    pipeline_model_parallel_size: 1
    dynamic_context_parallel:
      enabled: true
      min_size: 1
      max_size: 8
      tokens_per_rank: 4096
  sequence_packing:
    enabled: true
  dynamic_batching:
    enabled: false
```

The static CP size defines the initialized topology. Active CP sizes are powers
of two within the combined DP × CP domain, including sizes larger or smaller
than static CP. `max_size: null` uses the complete domain. The domain and CP
bounds must be powers of two. Budgets count padded tokens per CP rank, before
tensor sequence parallelism. A sequence that cannot fit at the maximum size is
rejected before dispatch.

There is one `tokens_per_rank` budget for scoring and training. Training usually
has the tighter memory limit because it retains activations, so this is the safe
schedule to share. The first policy or reference-logprob call builds the
immutable schedule. Later score calls with the same ordered lengths reuse it,
and training consumes it while recomputing its own valid-token and
valid-sequence denominators. The payload is still rebuilt at each stage because
score and train carry different fields, but `plan_cp_phases` runs only once.

Set `tokens_per_rank` to the largest packed token count that one rank can safely
train. Setting it lower than the ordinary sequence-packing budget forces extra
CP groups and smaller model calls even when memory does not require them. The
scheduler can still increase CP for work in a partially occupied phase so idle
lanes contribute, matching the balanced hybrid-CP behavior.

This path currently supports PP=1 and the standard Ray policy data path.
TransferQueue, split execution, model-owned multimodal packing, atomic preference
pairs, MTP, draft training, fused linear logprobs, and training CUDA graphs are
not supported. Dynamic batching and HybridEP input prepadding are also disabled
because the driver plan must remain the sole owner of packing and microbatch
boundaries. Omit or disable the configuration for existing static behavior.

For MoE, the minimum active size is raised until `active_CP * TP >= EP`. Workers
also check that each actual expert communication group is contained within its
task's ranks. This keeps every EP collective inside one active-CP task and prevents
experts from communicating across independently scheduled CP blocks. Router
auxiliary losses and per-layer MoE metrics stay attached to the real microbatch
that produced them; placeholder tasks carry zero valid tokens and do not affect
loss normalization. `global_aux_loss`, expert tensor parallelism, quantile
balancing, and overlapped MoE microbatch execution are rejected because their
collective or scheduling domains cannot follow the active task safely.

## Dispatch and execution

One scheduling lane contains a complete TP replica. The driver visits sequences
in descending length order. Before opening a task at a sample's minimum CP size,
it tries spare capacity in already-required larger-CP tasks, then existing tasks
of the same size. Each admission uses the destination task's padding and token
budget. This avoids leaving holes beside long samples while making extra model
calls for short samples. It assigns each packed task to an aligned,
contiguous group of lanes. The worker verifies those lane IDs against MCore's
initialized DP × CP group and resolves the active attention group using MCore's
hybrid-group API. TP ranks receive the same task payload. EP is a constraint on
group placement, not another data-sharding axis.

Before finalizing an initial placement, the driver repeatedly expands the smallest real task
to the next CP power of two while unused lanes remain. Its padding factor and
padded-token count are recalculated at the larger CP size. This mirrors MCore's
`fill_empty_gpus` policy and turns idle lanes into useful attention work. If an
explicit `max_size` prevents another expansion, the remaining lanes execute a
small zero-mask placeholder.

Adjacent placements with the same lane partition are merged into one
synchronization group. Packed tasks are redistributed between equal-size CP
subgroups using longest-processing-time placement with
`sum(sequence_length²) / active_CP` as the estimated attention cost. A subgroup
may consequently execute more packed tasks than another subgroup. Every packed
task still has its own checked token budget; merging never combines their
activations or padding allocations.

Every lane executes at least one task per synchronization group. A zero-mask
placeholder is used only when a lane has no real task, which lets all DDP ranks
participate in the final gradient synchronization. The standard MCore PP=1
executor runs every lane's local tasks except its last under `no_sync`; the last
local backward starts gradient synchronization. A domain-wide barrier occurs
only before the first task of each synchronization group. Faster subgroups wait
there after completing their shorter task lists, before any rank changes its CP
topology. The boundary marker is part of `ProcessedMicrobatch`, so an MCore
rerun repeats the barrier.

MCore creates hybrid groups during Bridge initialization. NeMo-RL then selects
the standard no-pipeline executor for these driver-planned groups. Optimizer
and DDP groups remain fixed. `PackedSeqParams.local_cp_size` and `cp_group`
carry the active attention topology; size one explicitly uses `cp_group=None`.
The NeMo-RL runtime also initializes the TE CP stream when a model built with
static CP=1 first needs larger groups.

The pinned attention implementation still reads its constructor process-group
collection for RoPE. NeMo-RL isolates that collection and binds its CP entry to
the active group through forward and backward, then restores the original
collection. The model receives a shallow copy of packed metadata with a real
singleton group for CP=1, because RoPE interprets `None` as a static-group
fallback. Loss and logprob code retain the explicit size-one/None convention.

During worker setup, only dynamic-CP Megatron actors register a Ray serializer
for tensor results. Importing MCore in the worker replaces
`torch.storage._load_from_bytes` with MCore's safe loader. A tensor does not
store a Megatron object, but PyTorch's normal storage pickle records that loader
function by module name; deserializing such a result would therefore make the
lightweight Ray driver import `megatron`.

The serializer runs when Ray materializes an actor method's return value, after
the worker has copied result tensors to CPU and before the driver's
`get_all_worker_results` completes. It applies once to every tensor nested in
that return value. In the GRPO recipe this includes the full policy-logprob and
reference-logprob result rounds and the small loss/gradient metric tensors from
each training step. It encodes contiguous bytes, dtype, and shape in a NumPy
payload, including BF16 and empty tensors. Any unexpected CUDA result is staged
to host memory instead of failing serialization, and the driver reconstructs an
independent, writable CPU tensor. It does not modify MCore's checkpoint loader
or serialize model parameters and GPU activations.

## Packing, outputs, and normalization

Padding is computed before scheduling and checked again after packing on each
worker. Each sequence is aligned to the least common multiple of the user
padding factor and the active CP, sequence-parallel, and precision requirements.
CP>1 uses the two-chunk balanced layout. Padding tokens and placeholder samples
are excluded from the objective and returned outputs.

Static `REPLICATED_AXES` remains valid for static CP. Dynamic dispatch removes CP
from both input replication and output replication filters. Outputs are gathered
from every DP × CP lane; the first lane of each active task owns its reconstructed
rows. The driver keeps those rows once and restores original sample order,
checking for missing or duplicate sample IDs. A static CP-rank-zero filter would
discard valid results.

Score workers execute every step in a multi-global-batch plan and concatenate
their task outputs before owner-row selection. They do not assume
`plan.steps[0]`; this keeps policy, reference, and top-k scoring aligned with the
training-sized schedule batches cached by the driver.

The driver computes valid sequence and token denominators from the unique
global batch before replication, separately for every optimizer step. The
differentiable CP logprob gather replicates the loss over the active CP group.
For this gathered loss, each lane's task is multiplied by

```text
number_of_local_tasks / (static_CP * active_CP)
```

This cancels the pinned no-pipeline executor's
`static_CP / number_of_local_tasks`
scaling and compensates the active-CP gather's backward SUM. DDP sums gradients
over the fixed DP × CP domain. Metrics are retained only on task owners before
global aggregation. `tests/unit/models/megatron/test_dynamic_cp_scaling.py`
checks this multiplier against the installed MCore loss callback. A future
MCore pin must pass that contract test and distributed parity before adoption.

For MoE load balancing, MCore's per-token path assumes
`local_valid_tokens * TP_CP_size` equals the active task's token count. Dynamic
packing can leave different valid-token counts on participating shards, so the
worker sums the exact valid count over the active TP×CP group and applies a
per-shard correction through `moe_grad_scale_func`. MCore's z-loss coefficient
and attachment factors already cancel to form each rank's valid-token sum; its
temporary coefficient is inversely adjusted so the shared autograd scaler does
not change that gradient. For reporting, ordinary aux
metrics are divided by unique real tasks, while z-loss reproduces MCore's
average over every real TP×CP rank participation.

## GB200 five-step smoke test

From the RL checkout on the Slurm login node:

```bash
DRY_RUN=0 bash perf_runs/run_gb200_dynamic_cp.sh
```

The launcher requests four nodes with four GPUs each, partition `batch`, QoS
`short`, and a two-hour limit. It uses the container and HF cache defaults in the
launcher, which can be overridden through its environment variables. Omitting
`DRY_RUN=0` prints the submission command without submitting.

The container preflight checks the Bridge pin against the RL checkout and runs
CPU dispatch tests. A four-GPU test then compares the actual packing, CP
logprob collectives, loss, gradients, and an SGD update against an unsharded
reference for active sizes 4, 2, and 1 with base CP=1 and base CP=2. Another
four-GPU test exercises fused RoPE and TE attention in a small transformer,
including TP=1/2 and base CP=1/2. It also checks actual worker score serialization
and reassembled sample logprobs against a CP=1 reference. Then
`grpo-qwen3-32b-4n4g-megatron-dynamiccp-quick.yaml` runs five GRPO steps with
TP=2, PP=1, base CP=1, and active CP up to 8. Both policy and reference logprob
passes are enabled. Checkpoints and validation are disabled for the smoke test.

Driver output is in `<job-id>-logs/ray-driver.log`; TensorBoard metrics are in
`logs/dynamic-cp-<UTC timestamp>`. The final metrics check requires steps 1–5,
finite loss and gradient norm, positive valid-token counts, training/scoring
importance ratios within 0.01 of one, and generation KL below 0.1, and writes
`smoke_result.json` in that log directory. A completed smoke run verifies
execution and finite training metrics, not convergence or a speedup over static CP.

### Dynamic-CP MoE quick smoke tests

`perf_runs/run_gb200_dynamic_cp_moe.sh` runs the distributed loss and attention
preflights followed by a two-to-five-step real-model GRPO smoke test. It defaults
to Qwen3-30B-A3B, five steps, four nodes total (two generation nodes and two
policy nodes inherited from the async 1-off recipe), and a one-hour QoS limit:

```bash
DYNAMIC_CP_MOE_STEPS=5 DRY_RUN=0 \
  bash perf_runs/run_gb200_dynamic_cp_moe.sh
```

Select the other model cases with `DYNAMIC_CP_MOE_MODEL=qwen235b` or
`DYNAMIC_CP_MOE_MODEL=nemotron3-nano`. Qwen3-30B-A3B uses TP1/EP8 and therefore
runs its policy task at CP8. Qwen3-235B-A22B uses TP8/EP16, so its minimum active
CP is two. Nemotron-3-Nano-30B-A3B uses TP2/EP8, so its minimum active CP is four.
Larger active sizes remain available for longer generated sequences.

The Qwen3-30B-A3B recipe exercises `aux_loss`; Qwen3-235B-A22B exercises
`seq_aux_loss`; and the Nemotron recipe exercises its inherited router setup.
The post-run check requires the expected active CP size, finite training metrics,
the requested number of steps, and the configured MoE metric when applicable.

The recipes log to both TensorBoard and W&B. The launcher defaults
`WANDB_MODE=online`, uses `/home/humairafirdo/hf_home`, and prints both settings
before submission. Set `WANDB_MODE=offline` explicitly when online logging is not
wanted.

### Ten-step Nsight profile

`perf_runs/run_gb200_dynamic_cp_profile.sh` runs the same dense Qwen3-32B setup
for ten steps with TensorBoard and W&B enabled. It profiles only Megatron policy
workers because that is where dynamic CP executes. By default, Nsight captures
all ten steps with `PROFILE_STEP_RANGE=1:11`. Use
`PROFILE_STEP_RANGE=3:6` for a smaller steady-state-only report. The launcher
is a dry run unless `DRY_RUN=0` is explicitly supplied.

Each runtime packed task has an NVTX label such as
`dynamic_cp/group_2/task_1/cp_4/lane_2/data`. Within a group, different lanes
may have different maximum task indices. The post-run check requires ten
dynamic training plans, at least two active CP sizes, at least one group with
multiple sequential packed tasks, valid training metrics, and at least one
completed policy `.nsys-rep` file on the head node. It reports how many groups
had uneven per-lane task counts and also requires every training
dispatch to report `schedule=reused`. `ray.sub` copies reports from all nodes
into `<job-id>-logs/ray/**/nsight/`.

### Ten-step dynamic/static comparison

`perf_runs/run_gb200_cp_comparison.sh` runs matched ten-step jobs with the same
Qwen3-30B-A3B model, batch, TP4/EP4/PP1 policy topology, generation setup,
container, and W&B project. EP4 is only a sharding change; it does not remove
experts or change model weights. On the two policy nodes, TP4 creates two lanes
and `CP1 * TP4 = EP4`, so a complete expert group fits inside CP1. The dynamic
run can execute two CP1 tasks or one CP2 task, while the capacity-matched static
run stays at CP2.

The default workload has an 8192-token ceiling, 4096 tokens per rank, and a
global batch of 512 formed from 16 prompts times 32 generations. The launcher
uses partition `batch`, inherits the account's default QoS, and requests four
hours by default (`TIME_LIMIT` overrides it). Use different `CP_RUN_NAME` values so
the W&B runs and local logs remain distinct.

The launcher also accepts `CP_NUM_STEPS`, `CP_TRAIN_GLOBAL_BATCH_SIZE`,
`CP_NUM_PROMPTS_PER_STEP`, `CP_NUM_GENERATIONS_PER_PROMPT`,
`CP_MAX_TOTAL_SEQUENCE_LENGTH`, `CP_TOKENS_PER_RANK`, `CP_MAX_SIZE`, and
`STATIC_CP_SIZE`. Prompt count times generations must equal the global batch.
A static CP1 run remains useful as an unconstrained throughput and CP1 sanity
reference when it fits in memory, but it is not the capacity-matched baseline
for 8192 tokens at the 4096-token budget.

This is a match to the configured memory budget, not proof that CP is necessary
on GB200. Measure peak memory and test static CP1 before concluding that 8192
tokens require CP2. If CP1 fits and performs better, use that as the practical
baseline; increase `CP_TOKENS_PER_RANK` to the measured training-safe budget.
Do not reduce the budget just to make the scheduler report more CP sizes.

The correctness smoke keeps the original TP1/EP8 topology and is therefore
forced to CP8. The performance pair uses TP4/EP4 specifically to expose an
adaptive CP1/CP2 choice on the same eight policy GPUs. Both sides of the pair
use TP4/EP4, so the measured difference is dynamic versus fixed CP rather than
a model or expert-layout difference between the two runs.

After each run, `perf_runs/analyze_cp_sequence_lengths.py` writes
`sequence_length_distribution.json` beside the driver log. It reports length
percentiles, how many samples stopped exactly at the configured ceiling, and
the CP size each sample required before optional idle-lane expansion.

Dynamic schedule logs also report `tasks_by_cp` and `packing_utilization`.
Required CP for an individual sample can differ from its scheduled CP when it
fills an existing larger task or when spare lanes help process a task. Performance
runs do not require a fixed mixture of sizes. For an explicit coverage test, set
`CP_REQUIRE_SIZES="1 2"` (or `"1 2 4"`). The correctness checks still require all
steps, finite metrics, score/train agreement, and schedule reuse.

Both comparison modes run four-GPU loss and attention parity checks before
training. These tests explicitly retain CP1 coverage after cross-size packing,
including a model initialized at static CP2. Set `CP_GPU_PREFLIGHT=0` only when
reusing validation of the same code/container. Preflight time is outside the
reported training-step timings.

### Qwen30B packing regression: jobs 7202770 and 7203284

Both ten-step jobs passed their smoke checks and processed approximately 26.6M
tokens. Excluding step one, the dynamic job averaged 525.65 seconds per step;
static CP2 averaged 483.72 seconds, so dynamic took 8.67% longer. Training took
390.62 versus 357.51 seconds and policy/reference scoring took 130.03 versus
121.26 seconds. These are independent async rollouts, not identical token batches.

The old scheduler packed each required CP size separately. At 4096 tokens/rank,
`[6000, 2000]` became two calls even though both samples fit in one 8192-token CP2
pack. The cross-size packing fix admits the 2000-token sequence into that
already-required call, checks CP2 alignment, and still leaves independent CP1
tasks when there is remaining short work. EP containment and loss scaling are
unchanged; the worker derives its microbatch count from the resulting plan.

CPU replay of the ten saved dynamic batches changes the sum of local calls per
lane from 3630 to 3330; static MFFD packing of those same lengths needs 3322.
That is an 8.26% reduction in model calls, not a measured GPU speedup. Rerun the
pair to measure actual performance. A near-zero gain remains possible because
most work in this workload still executes at CP2 after efficient packing.

Reproduce the recorded timings and replay the current planner:

```bash
uv run --no-sync python perf_runs/analyze_cp_comparison.py \
  logs/qwen30-gbs512-seq8192-dyncp-20260916T220850Z \
  logs/qwen30-gbs512-seq8192-staticcp2-20260916T220850Z \
  --lanes 2 --tp 4 --tokens-per-rank 4096
```

### Dense Qwen3-32B comparison

Set `CP_MODEL=qwen32b` to select the new dense recipes. They use four nodes total:
two policy nodes (TP2/PP1/EP1, four CP scheduling lanes) and two generation nodes
(vLLM TP2). Defaults are 16384 total tokens, 4096 tokens/rank, GBS512, ten steps,
activation checkpointing, W&B online, dynamic CP1/2/4 versus static CP4. Both
modes inherit the same workload, optimizer, precision, and generation settings.

```bash
PAIR_TAG=$(date -u +%Y%m%dT%H%M%SZ)
CP_MODEL=qwen32b CP_MODE=dynamic TIME_LIMIT=06:00:00 \
  CP_RUN_NAME="qwen32-gbs512-seq16384-dynamic-${PAIR_TAG}" DRY_RUN=0 \
  bash perf_runs/run_gb200_cp_comparison.sh
CP_MODEL=qwen32b CP_MODE=static STATIC_CP_SIZE=4 TIME_LIMIT=06:00:00 \
  CP_RUN_NAME="qwen32-gbs512-seq16384-static4-${PAIR_TAG}" DRY_RUN=0 \
  bash perf_runs/run_gb200_cp_comparison.sh
```

For a smaller first validation set `CP_NUM_STEPS=2 CP_NUM_PROMPTS_PER_STEP=2
CP_TRAIN_GLOBAL_BATCH_SIZE=64 QOS=short TIME_LIMIT=01:00:00` and use a distinct
run name. Ten steps are an initial performance sample, not
convergence validation. Report training/scoring separately from total step time
and repeat close results before claiming a speedup.
