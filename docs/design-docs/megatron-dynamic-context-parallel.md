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

This path currently supports PP=1 and the standard Ray policy data path. Atomic
preference groups, MTP, HybridEP input prepadding, `global_aux_loss`, and expert
tensor parallelism are supported. Hybrid MTP requires matching attention and
linear CP layouts so the same packed layout can be used safely. HybridEP keeps
logical sequence boundaries for MTP while marking its enlarged physical buffer
as padded for Transformer Engine attention. Non-colocated distillation teachers
keep their own static CP topology; Dynamic CP remains enabled only on the student.

TransferQueue/split execution, model-owned multimodal packing, draft training,
fused linear logprobs, dynamic batching, training CUDA graphs, MLA, and chunkwise
linear CP are not supported. Omit or disable the configuration for existing
static behavior.

For MoE, the minimum active size is raised until
`active_CP * TP >= ETP * EP`. Workers
also check that each actual expert communication group is contained within its
task's ranks. This keeps every EP collective inside one active-CP task and prevents
experts from communicating across independently scheduled CP blocks. Router
auxiliary losses and per-layer MoE metrics stay attached to the real microbatch
that produced them; placeholder tasks carry zero valid tokens and do not affect
loss normalization. `global_aux_loss` uses aligned placeholder rounds so every
rank enters its full-domain collective in the same order. Quantile balancing and
overlapped MoE microbatch execution remain rejected because their mask or
scheduling requirements cannot follow the active task safely.

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
`fill_empty_gpus` policy and turns idle lanes into useful attention work. If the
larger alignment would exceed the task's token budget, or an explicit `max_size`
prevents expansion, the task keeps its valid size and remaining lanes execute a
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
packing can leave different valid-token counts on participating shards. A router
pre-hook sums the exact valid count over the active TP×CP group, and the temporary
router attachment uses that task count directly. The worker's ordinary
`1/global_valid_tokens` gradient scale therefore remains sufficient and does not
need a second per-shard correction. For reporting, ordinary aux
metrics are divided by unique real tasks, while z-loss reproduces MCore's
average over every real TP×CP rank participation.

## Validation

CPU planner and dispatch behavior is covered by the unit tests under
`tests/unit/distributed/`. Four-GPU correctness coverage is available through:

```bash
bash tests/functional/dynamic_cp.sh
```

The functional test compares loss, gradients, an optimizer update, output
reassembly, fused RoPE, and Transformer Engine attention across active CP sizes
1, 2, and 4, including TP and initialized-static-CP variants.

End-to-end performance should be measured with matched local recipes. Keep the
model, generated token batch, parallel topology, optimizer, and precision fixed;
compare dynamic CP with the smallest static CP that fits the same workload.
Record score, reference-score, refit, training, and total step time separately,
plus peak memory, MFU, task counts by CP size, packing utilization, schedule
reuse, and the sequence-length distribution. Ten or twenty steps are useful for
a smoke/performance sample, but are not enough to claim training convergence.
