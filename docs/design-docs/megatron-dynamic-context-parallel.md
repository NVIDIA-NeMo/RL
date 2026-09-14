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
      train_tokens_per_rank: 1024
      logprob_tokens_per_rank: 1024
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

Each phase covers the entire DP × CP domain. Unoccupied lanes execute a small
zero-mask placeholder, so every rank performs the same number of
forward/backward calls and reaches gradient synchronization together. A barrier
between phases keeps changes of active group coordinated, including MCore
reruns. This conservative scheduler introduces more synchronization and padding
than Megatron's balanced scheduler; it is not a throughput-equivalent port of
that scheduler.

MCore creates hybrid groups during Bridge initialization. NeMo-RL then selects
the standard no-pipeline executor for these driver-planned phases. Optimizer
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

Dynamic-CP workers register a Ray serializer for CPU tensor results. It uses
NumPy byte arrays and preserves tensor dtype and shape, including BF16. This
keeps the driver independent of MCore even when MCore replaces PyTorch's tensor
storage loader with a backend-specific function. MCore's checkpoint loader is
left intact.

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
For this gathered loss, NeMo-RL multiplies by

```text
number_of_phases / (static_CP * active_CP)
```

This cancels the pinned no-pipeline executor's `static_CP / number_of_phases`
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

Each runtime phase has an NVTX label such as
`dynamic_cp/phase_7/cp_4/lane_2/data`. The post-run check requires ten dynamic
training plans, transitions between CP=1 and CP>1, valid training metrics, and
at least one completed policy `.nsys-rep` file on the head node. `ray.sub` also
copies reports from all nodes into `<job-id>-logs/ray/**/nsight/`.
