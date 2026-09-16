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
not supported. Omit or disable the configuration for existing static behavior.

For MoE, the minimum active size is raised until `active_CP * TP >= EP`. Workers
also check that each actual expert communication group is contained within its
task's ranks. This constraint prevents expert collectives from crossing task
boundaries; it is not an end-to-end validation of MoE auxiliary losses or router
replay. The smoke recipe uses a dense model.

## Dispatch and execution

One scheduling lane contains a complete TP replica. The driver packs sequences
of the same required CP size and assigns each packed task to an aligned,
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

During worker setup, dynamic-CP Megatron actors register a Ray serializer for
CPU tensor results. Importing MCore in the worker replaces
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
payload, including BF16 and empty tensors, then reconstructs an independent,
writable CPU tensor in the driver. It does not modify MCore's checkpoint loader
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
model, batch, TP=2, PP=1, base CP=1, generation setup, container, and W&B
project. Set `CP_MODE=dynamic` to allow active CP sizes 1–8, or
`CP_MODE=static` to keep CP=1. Use different `CP_RUN_NAME` values so the W&B
runs and local logs remain distinct.

The launcher accepts `CP_MAX_TOTAL_SEQUENCE_LENGTH`, `CP_TOKENS_PER_RANK`,
`CP_MAX_SIZE`, and `STATIC_CP_SIZE`. A fair capacity-matched comparison uses the
smallest fixed CP that can accommodate the configured maximum at the same
per-rank token budget. For example, compare dynamic CP1–2 against static CP2
with an 8192-token maximum and a 4096-token per-rank budget. Static CP1 remains
useful as an unconstrained throughput reference when it fits in memory; static
CP8 is a capacity-matched baseline only when the workload actually requires
CP8.

After each run, `perf_runs/analyze_cp_sequence_lengths.py` writes
`sequence_length_distribution.json` beside the driver log. It reports length
percentiles, how many samples stopped exactly at the configured ceiling, and
the CP size each sample required before optional idle-lane expansion.
