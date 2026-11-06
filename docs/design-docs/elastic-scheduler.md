# Elastic Scheduler

This document proposes an elastic scheduler for the SingleController (SC): GPUs are
grouped into *cells* that switch between serving rollouts and hosting a training
replica as the rollout workload drains, instead of keeping a fixed rollout/training
split (async) or switching the whole pool at once (colocate). It follows section 4 of
the QwenGyre paper (arXiv 2609.33848).

Two pieces exist today:

- The scheduling policy,
  `nemo_rl/algorithms/single_controller_utils/elastic_scheduler.py`. It is a pure
  state machine with no Ray, GPU or asyncio dependency.
- Its first consumer, `tools/elastic_scheduler_replay.py`. It replays the rollout
  durations a real SC run recorded through the policy and projects step time and idle
  GPU time for elastic cells, a fixed split and colocate. This is the go/no-go
  measurement for the rest of the plan (see
  [Measuring before building](#measuring-before-building-offline-replay)).

**No training path calls the policy yet.** There is no config key and no change in
training behavior. The rest of this document is the plan for wiring it in.

## Problem

Agentic rollouts have heavy-tailed durations. In a step of `B` groups the last few
executions run long after most GPUs have nothing to do:

- **Async (disaggregated)** keeps a fixed training pool. Rollout GPUs go idle while the
  tail drains, and training GPUs go idle while they wait for a batch.
- **Colocate** switches the whole pool. Every GPU waits for the slowest execution, then
  every GPU trains.

The paper's elastic scheduler lets cells move to training one at a time as in-flight
work drops, and lets later cells join the batch that is already running. Reported
results:

- Up to **1.78x** over Async and **1.85x** over Colocate end to end.
- Qwen 3.6 122B on NL2Repo, 48 steps: 1.42-1.53x over Async and 1.36-1.47x over
  Colocate.
- Qwen 3.8 2.4T: 75.42 h, compared with 134.55 h for Async (1.78x) and 91.47 h for
  Colocate (1.21x). The gain over Colocate is smaller because many durations cluster
  near the timeout, so there is little tail to fill.
- DeepSWE and TerminalBench (Table 2): 1.38-1.57x over Async and 1.61-1.85x over
  Colocate.
- Ablation (Table 1, 12 steps, 32 nodes as 8 cells x 4 nodes, 144 concurrent
  executions per cell, mean rollout 1.93 h, GSPO with groups of 16):

  | Config | Hours |
  |---|---|
  | A0: Async | 25.08 |
  | A3: Async with streaming | 22.18 |
  | C0: Colocate | 23.23 |
  | C2: Colocate with standalone pool, streaming, coarse switch | 19.23 |
  | E0: Elastic | 16.12 (1.19x over C2) |
  | E1: Elastic with `mu = 1` | 17.87 (+10.9% over E0; still 1.24x over A3, 1.08x over C2) |
  | E2: Elastic with 4 cells of 8 nodes | 16.47 (+2.2%) |

- Measured switch times: rollout to training 8.52 s, training to rollout 3.46 s.

E1 matters most for NeMo RL. It is the variant that fits the SC's
one-refit-per-step model, and it still beats the strongest colocated baseline.

## Model

Notation follows the paper. All counters are in **execution** units, where one
execution is one rollout of one prompt. A group holds `N` executions.

| Symbol | Meaning |
|---|---|
| `K` | elastic cells; cell 0 is the *core*, cells 1..K-1 are *satellites* |
| `C_e`, `C_s` | concurrent executions per cell, and in the standalone rollout pool |
| `k` | cells not currently in rollout (always the prefix `0..k-1`) |
| `B`, `N`, `mu` | groups per step, executions per group, steps per weight publication |
| `d`, `p`, `f` | executions dispatched, launched, finished |
| `w = d - f` | waterlevel: queued plus running |
| `rho_min`, `tau` | supply fraction of `B`, and how long supply must hold |

Rules:

- **eq. 3:** `C_R = C_s + (K - k) C_e` is the rollout capacity.
- **eq. 4:** launch `min(d - p, C_R - (p - f))` executions.
- **eq. 5:** the next cell may begin moving to training only if `w <= C_R - C_e`,
  meaning the remaining work fits without it. In addition, ready, unclaimed groups,
  capped at the groups the batch still needs, must stay at or above `rho_min * B`
  continuously for `tau`. Transitions are serial and in cell order with the core
  first, so the batch's optimizer owner is always the first to train. The supply
  timer resets at every transition and at batch start.
- **eq. 6:** at the end of a burst the core stays in training only if
  `w + mu N B <= C_s + (K - 1) C_e` and supply already holds. That is, the next
  burst's dispatch can be served without the core, and the core has work to start.
  Satellites always return, in descending order.

### What the policy module provides

`ElasticScheduler(policy=ElasticSchedulerPolicy(...), clock=...)`:

- Counters, in two unit systems with one conversion between them (see
  [Units](#units-groups-at-the-boundary-executions-inside)):
  - Group units, for the SC: `record_dispatch_groups`, `record_launch_groups`,
    `record_finish_groups` and `launchable_groups()`.
  - Execution units, for a caller that sees single executions:
    `record_dispatch`, `record_launch` (refuses more than `launchable()`) and
    `record_finish` (refuses more than are running).
  - Derived values are `queued`, `running`, `waterlevel` and `rollout_capacity`.
- Decisions:
  - `launchable()` implements eq. 4, and `launchable_groups()` rounds it down to
    whole groups.
  - `next_training_cell(ready_groups=, groups_needed=)` implements eq. 5 with the
    supply hold. It returns a cell index or `None`.
  - `plan_burst_end(ready_groups=)` implements eq. 6 and returns a `BurstEndPlan`.
- Transitions:
  - `begin_training_transition` / `finish_training_transition` move a cell to
    training. Beginning re-checks order, the serial rule and eq. 5, so the caller
    cannot force a bad move.
  - `begin_rollout_transition` / `finish_rollout_transition` move it back. The caller
    reports the finish only once the engine serves the published weights. Until then
    the cell does not count toward `C_R`.
  - `open_batch` / `close_batch` mark the joining window.

A cell stops counting toward `C_R` as soon as its move to training *begins*. Rerouted
requests must not land back on it.

The policy is a frozen dataclass, the same shape as `FleetHealthPolicy`. The
user-facing BaseModel is deferred to the phase that consumes it, because a config key
with no reader is dead configuration.

### Units: groups at the boundary, executions inside

The paper's capacities and counters are in executions: `C_e` is how many concurrent
executions an engine serves. The SC admits, launches and commits whole prompt groups
of `N` executions. Mixing the two silently would be off by a factor of `N`, so the
boundary is explicit:

- `ElasticSchedulerPolicy` carries `group_size` (`N`) and `steps_per_burst` (`mu`).
  `ElasticSchedulerPolicy.executions(groups)` is the only place groups become
  executions, and `burst_dispatch` (`mu N B`) is derived from it rather than passed
  in.
- The SC calls only the `*_groups` methods. `launchable_groups()` is
  `launchable() // N`, so a launch decision never splits a group. If `C_e` is not a
  multiple of `N`, the remainder slots stay unused rather than starting part of a
  group.
- Supply (`ready_groups`, `groups_needed`) and `B` stay in groups, as in the paper's
  eq. 5.

Group-level accounting holds a group's `N` slots until its last execution ends, which
is conservative: cells move to training later than per-execution accounting would
allow. Per-execution counting needs a per-execution completion signal from NeMo-Gym,
which the SC does not receive today. The execution-unit methods are kept for that
caller.

## Measuring before building: offline replay

`tools/elastic_scheduler_replay.py` answers "would elastic cells help this workload?"
from a run that already happened, before any controller or worker change.

**Input.** An SC run records one `rl.sc.generate_and_push` span per dispatch attempt
when telemetry includes the `per_prompt` span group. The span covers one prompt
group's generation and data-plane commit. The tool reads these spans from either
source the repo already supports:

- OTLP/JSON lines, for example an OpenTelemetry Collector `file` exporter placed in
  front of the usual backend.
- The stdout of `telemetry.exporter: console`. Interleaved log lines are skipped.

```yaml
telemetry:
  enabled: true
  span_groups: per_step,per_prompt
```

The aggregated `rollout/throughput/group_completion_seconds_{mean,p50,p95}` metrics are not
enough, because the policy reacts to the shape of the tail, not to its quantiles.

**Run.**

```bash
uv run python tools/elastic_scheduler_replay.py --trace spans.jsonl \
    --num-cells 8 --cell-capacity 144 --group-size 16 --batch-groups 64 \
    --train-step-s 3600 --num-steps 12
```

`--train-step-s` is the time *one cell* needs for one optimizer step. Derive it from
the recorded `timing/train/policy_and_reference_logprobs` plus
`timing/train/policy_training`, multiplied by the number of cells' worth of GPUs that
trained in the recorded run. `--to-training-s` and `--to-rollout-s` default to the
paper's 8.52 s and 3.46 s, and `--refit-s` is the fixed split's weight-sync time.

**Output.** For each strategy on the same `K` cells: total and mean step time, idle
cell-seconds as a fraction, and switching overhead. A rollout cell is idle in
proportion to its unused concurrency, and a training cell is idle while it has no
micro-step. The strategies are:

- `elastic`: every role decision comes from `ElasticScheduler`, with `mu = 1`.
- `fixed_split`: a fixed number of training cells (every split is tried unless
  `--fixed-split-train-cells` is given), with `--max-lookahead-steps` of off-policy
  lookahead.
- `colocate`: the full batch, then all cells train.

**What it assumes.**

- Group durations do not depend on which engine runs them or how loaded it is.
  Durations recorded under different contention replay as-is.
- Groups hold their slots until the whole group ends (see Units).
- The fixed split's refit pauses only the trainers. In the SC, refit also pauses
  generation, so the fixed split is projected optimistically.
- Training time scales linearly with the number of training cells.

**Example (synthetic).** 400 Pareto(1.2) group durations (scale 60 s, capped at
1800 s; p50 109 s, p95 897 s), `K = 4`, `C_e = 24`, `N = 4`, `B = 8`, 960 s of
one-cell training per step, 6 steps:

| Strategy | Hours | Idle |
|---|---|---|
| elastic | 0.77 | 28.8% |
| fixed split, 3 training cells, 1 step of lookahead | 0.72 | 25.3% |
| fixed split, 3 training cells, no lookahead | 1.69 | 69.0% |
| colocate | 1.56 | 65.1% |

On this trace, elastic halves colocate's time and roughly matches a well-tuned
off-policy fixed split. It does so while staying within half a step of on-policy,
which is the regime where a fixed split degrades to 1.69 h. That is the kind of
answer Phase 2 must produce on real traces before Phase 3 starts. A workload where
the best fixed split already wins clearly is a no-go.

## Mapping onto the SingleController

### Ownership

The `ElasticScheduler` instance lives in `SingleControllerActor`, next to the buffer,
sampler and fleet health state. All mutations happen on the actor's single asyncio
loop, so no locking is needed. Counter sources:

| Counter | SC event |
|---|---|
| `d` | `sampler.admit` admits a prompt, so `N` executions are recorded |
| `p` | the rollout pump launches a group. This replaces the `max_inflight_prompts` semaphore with a capacity gate that launches whole groups while `launchable() >= N` |
| `f` | the dispatch task's `finally` path, covering success, failure, timeout and cancellation |
| supply | ready, unclaimed groups in `TQReplayBuffer` |
| groups needed | `_target_groups_for_step` minus the groups already handed to training |

A model request retried on another engine stays inside its execution and changes no
counter. This is what makes rerouting invisible to the policy.

### Cells, placement groups and worker groups

- **Cells** are `RayVirtualCluster` segments. `_build_clusters` already places
  segment-aligned bundles through `segment_size` / `prepare_segment_topology`. A cell
  is one segment-aligned slice of the colocated cluster.
- **The core** (cell 0) is the existing `TQPolicy` Megatron worker group, together
  with its colocated Megatron-inference (MInf) engine. It owns the optimizer,
  checkpoints and the authoritative weights.
- **Satellites** are new Megatron worker groups on their own cells. Each has the
  same model-parallel layout as the core, plus parameters and gradient buffers but
  **no optimizer state**, and each has a colocated MInf engine.
- **The standalone pool** (`C_s`) is the existing non-colocated inference cluster,
  vLLM or MInf.

Two constraints in today's code drive this choice. The SC rejects colocated vLLM, and
colocated SC exists only with Megatron generation. Cells are therefore MInf-backed.

### Reused versus new

Reused:

- Streaming training (`min_groups_for_streaming_train`, the split
  `begin_train_step` / `train_microbatches_from_meta` / `finish_train_step` API, and the
  chunk cap from #5036) serves as the micro-step queue. The controller assigns chunks
  to cells. "Atomic claim" is free because the asyncio loop is single-threaded.
- The generation router's `set_serving_backends` serves as the rerouting switch.
  Aborting a backend's in-flight calls returns 500, and NeMo-Gym's bounded retry
  re-sends the call to a remaining backend. See
  [Router and colocation constraints](#router-and-colocation-constraints) for what
  works with MInf cells today.
- Fleet health (`fleet_health.py`): returned cells take the existing STALE-then-refit
  path.
- Refit membership (`weight_sync/membership.py`) handles reconciliation when the
  serving set changes.
- The MInf stand-down/wake path (`blocks_training`, `suspend_request_deadlines`,
  `finish_generation` / `prepare_for_generation`, `wake_carries_weight_updates`) is
  used for the core's switch.
- The NIXL checkpoint engine: its pair-rank-i-with-rank-i transfer and its
  `shutdown` then `init_communicator` re-init after membership changes are the model
  for the satellite weight pull.
- The checkpoint format is unchanged. Only the core checkpoints.

New:

- Satellite worker groups (gradient-only replicas).
- Prefix communicator sets (described below).
- A `LENT` fleet-health state. It means "removed from serving on purpose; not
  failed". It must not trip the DEAD-then-STALE rule or count as a failure.
- Router destination priority (standalone first, then rollout cells in descending
  order) as a tiebreak inside load-normalized least-outstanding.
- The controller pushing the router's serving set directly, without fleet health,
  plus an engine-side abort of a cell's in-flight calls on MInf.
- Per-request retry on the native (non-Gym) rollout path.
- The capacity gate in `_rollout_pump`, and global token normalization across cells.

## Transition protocol

### Rollout to training (cell `c`)

1. `next_training_cell` returns `c`, and the controller calls
   `begin_training_transition(c)`. `C_R` drops by `C_e` immediately, so the capacity
   gate stops launching onto `c`.
2. **Reroute.** Remove `c`'s backends from the router's serving set, then abort the
   in-flight calls on `c`. Gym retries each one on another backend, preferring the
   standalone pool and then the highest-numbered rollout cell. KV migration is
   optional and out of scope. A retried call re-prefills, and the cost is the
   prefix length.
3. **Offload/onload.** Release the MInf engine's KV cache and weights, then onload the
   Megatron training state. For a satellite, that state is parameters and gradient
   buffers only.
4. **Weight pull (satellites only).** Copy the core's training weights for the
   current policy version. The core is the source rank-for-rank, like the NIXL
   engine's rank pairing, but policy to policy. The paper uses Mooncake RDMA. A NIXL
   or NCCL broadcast over the prefix group is the NeMo RL equivalent.
5. `finish_training_transition(c)`. The cell starts taking micro-steps from the
   current batch if its joining window is still open. If the window has closed, the
   cell idles until the burst-end plan returns it. `plan_burst_end` refuses to run
   while any cell is mid-transition, so the controller waits for late joiners first. Phase 6 keeps this race small by
   closing the window only when the remaining micro-steps are already claimed.

**Joining an in-progress step.** Micro-steps are the streaming chunks of one
optimizer step. Every active cell, the core included, claims the next unclaimed
chunk, runs forward and backward with 1F1B, and accumulates gradients locally. A
satellite that joins late simply claims fewer chunks. Gradients are summed, and loss
normalization uses global token and sequence counts, which the controller knows
before any chunk is trained. The result therefore matches the single-replica result
no matter how many cells joined.

### Gradient reduction with dynamic membership

The paper prebuilds `K-1` prefix communicator sets. For `m` in `1..K-1` and each core
DP rank `r`, `G_{m,r}` contains rank `r` of cells `0..m`. When the batch closes with
cells `0..m` in training:

1. Each satellite reduces its accumulated gradients into the matching core rank over
   `G_{m,r}`.
2. The core runs its normal DP all-reduce and the optimizer step.
3. If this is not the last update of the burst (`mu > 1`), the core broadcasts the
   new weights back over `G_{m,r}`.

Training cells are always a prefix, so these `K-1` sets cover every membership with
no communicator creation on the hot path. Their cost is NCCL buffer memory,
estimated next.

#### Memory per communicator

NCCL allocates buffers per *channel* per *connected peer*. The defaults per
connection are:

- Simple: `NCCL_BUFFSIZE` = 4 MiB.
- LL: 512 KiB.
- LL128: about 4.7 MiB (120 elements x 640 threads x 8 steps x 8 bytes).

That is about **9.2 MiB per channel per peer**. A ring connects one peer per channel,
and a tree up to three. Since NCCL 2.22, connections are made lazily on first use
(`NCCL_RUNTIME_CONNECT=1`), so an algorithm that is never used allocates nothing.

`G_{m,r}` spans nodes, because cells are node-aligned, so NCCL picks network channels.
On an 8-NIC H100 node that is typically 16 channels. For ring-only traffic:

```
per communicator ~= 16 channels x 1 peer x 9.2 MiB ~= 150 MiB per GPU
(ring and tree both used: up to ~4 peers ~= 600 MiB)
```

Core rank `r` belongs to all `K-1` sets, and satellite `c` to `K-c` of them, so the
core GPUs are the worst case. With `K = 8`, that is 7 x 150 MiB ~= 1.0 GiB per core GPU
(up to ~4 GiB with trees). This is on top of the core's training state, at the
moment the core needs the most memory.

These are estimates from NCCL's default buffer sizes, not measurements. Phase 6
measures them with `torch.cuda.mem_get_info()` before and after each communicator's
first reduction.

#### Alternatives

| Option | Extra memory on a core GPU | Cost |
|---|---|---|
| A. `K-1` prefix communicators (paper) | `(K-1) x M` (~1-4 GiB at `K = 8`) | None at runtime |
| B. One parent communicator over rank `r` of all `K` cells, split into prefix children with `ncclCommSplit(config.splitShare = 1)` | ~`1 x M` | Children share the parent's buffers, so their collectives serialize; only one prefix is used per step, so nothing is lost. The split is collective over the parent, so it runs once at startup, while every cell is idle. With `splitShare` set, children share the parent's abort flag; the abort path in `stateless_process_group.py` assumes it is unset and must be revisited |
| C. One communicator over all `K` cells; non-training cells contribute zeros | `1 x M` | Every reduction involves the cells serving rollouts: they launch a kernel, stream a gradient-sized buffer of zeros, and gate the core on their inference schedule. Rejected except as a debugging fallback |
| D. Build `G_{m,r}` lazily with `new_group(use_local_synchronization=True)`, keep an LRU of `L` | `L x M` | One `ncclCommInitRank` over `m+1` ranks per new membership. Hidden by starting the build when cell `m` *begins* its 8.5 s move to training, so it is off the critical path |
| E. Point-to-point `send`/`recv` on one communicator, with a ring reduce-scatter written over the prefix only | per connected peer, only for peers used | Non-members make no call. Hand-written reduction instead of NCCL's tuned one |

Cap channels per communicator with `ncclConfig_t.maxCTAs` (via
`ProcessGroupNCCL.Options.config`). Satellite-to-core reduction runs once per step and
is not latency-critical, so 4 channels instead of 16 cuts each estimate by 4x.

**Recommendation.** Start Phase 6 with B, since it costs one communicator's memory and
keeps NCCL's tuned collectives. Use D if `splitShare` conflicts with the existing
abort and rebuild code. Use A only where the measured `(K-1) x M` fits. All options
share one interface (`reduce_to_core(prefix_len)`), so switching between them does
not touch the scheduler.

### Router and colocation constraints

The constraints as they stand in the code:

- **The router is URL-based, not vLLM-specific.** `_maybe_start_generation_router`
  already fronts the per-model-parallel-group OpenAI URLs that MInf reserves
  (`MegatronGeneration.reserve_http_server_addresses`). What is vLLM-only is fleet
  health: only `VllmGeneration` implements `attach_fleet_health`, and fleet health is
  today the only caller of `set_serving_backends`. Elastic cells therefore need no
  router port to MInf. They need:
  1. The controller calling `set_serving_backends` with the URLs of the cells in
     rollout plus the standalone pool. This is a full-set push, so it is idempotent.
     The controller is the membership authority for lent cells, and fleet health,
     when enabled, intersects its own view with the controller's.
  2. A map from cell to URLs, which is static: a cell is a fixed set of
     model-parallel groups.
  3. An engine-side "abort all in-flight calls" on the removed cell's MInf servers.
     Removing a URL from the serving set only stops *new* calls; in-flight calls must
     fail so the router returns 500 and Gym retries elsewhere.
  4. Priority as a tiebreak: pick by `(inflight / capacity, tier, url)` instead of
     `(inflight, url)`, with the standalone pool in tier 0 and higher-numbered cells
     in lower tiers. Strict priority would pile every call onto the standalone pool,
     and the capacity gate already bounds the total.
- **Native (non-Gym) rollouts** have no router. Elastic cells require NeMo-Gym until
  per-request retry exists on the native path.
- **Colocation.** SC accepts `colocated.enabled` only for `backend: megatron`, and
  then requires `min_groups_for_streaming_train == num_prompts_per_step`. The reason
  for the second rule is that a colocated engine must stand down before its GPUs
  train, and with one engine that means after the last rollout. With cells, the rule
  becomes per cell: *a cell trains only after its own engine has stood down*. Other
  cells keep generating, so streaming is legal. Phase 5 replaces the global check
  with this per-cell one when `elastic_scheduler.enabled`, and keeps the global check
  otherwise.
- **vLLM.** SC rejects colocated vLLM because `RolloutManager.generate_and_push`
  needs the disaggregated async engine. Elastic cells do not need colocated vLLM:
  cells are MInf, and the standalone pool is a normal non-colocated engine. It should
  also be MInf, to avoid a vLLM/MInf numerics mismatch between pools (see Staleness).

### Training to rollout

1. After the final update of the burst, `close_batch()`, then
   `plan_burst_end(ready_groups=...)`.
2. For each returned cell, in plan order, call `begin_rollout_transition(c)`, offload
   the training state, and onload the MInf engine.
3. The returned cell receives the newly published weights through the normal refit.
   For satellites this goes through the existing STALE-then-refit path, so a cell
   cannot serve old weights.
4. `finish_rollout_transition(c)`, then re-add the cell to the router's serving set.
   Only now does it count toward `C_R` again.
5. If the core is retained, it keeps its training state and starts the next batch.
   `open_batch()` resets the supply timer.

## Staleness and importance-sampling correction

- Every engine serving at a given time serves the same published version, so
  rerouting a call does not change which policy version produced it. A retried call
  may mix tokens from two engines of the same version. Engines differ only in
  numerics, so cells should run the same backend: mixing vLLM in the standalone pool
  with MInf cells adds a train/inference mismatch that IS correction has to absorb.
- Satellites compute log-probs from the same weights as the core, so they introduce
  no additional staleness.
- `mu = 1` maps directly onto the SC's per-step refit and `max_lookahead_versions`.
  `mu > 1` publishes every `mu` steps, so off-policyness grows with `mu`. The SC's
  existing version-lag accounting and IS correction apply unchanged. The boundary
  dispatch offset `phi` (dispatch `(phi + mu) N B` up front) has no SC equivalent and
  needs a new knob.
- An open question is how token capture (`token-capture-ledger.md`) treats aborted
  and retried calls. The retried call must be what is recorded.

## Checkpointing

- The scheduler state is soft. Counters and roles can be rebuilt on restart: every
  cell starts in rollout, `d` comes from the restored rollout snapshot, and
  `p = f = 0`.
- Only the core checkpoints. Satellites hold no optimizer state, so a checkpoint is
  identical to today's.
- A transition must not overlap the rollout-snapshot cut. The checkpoint path waits
  for in-progress transitions to finish, the same way it already waits for refit.

## Failures

- **A rollout cell dies.** The existing fleet-health path applies. The scheduler also
  needs `C_R` to shrink (a follow-up `retire_cell`), otherwise eq. 4 over-launches.
- **A satellite dies while training.** Its claimed micro-steps are lost. The initial
  policy is to fail the step and rely on checkpoint resume. Later, its chunks can be
  re-queued and the batch reduced with the largest prefix group still intact, but any
  `G_{m,r}` with `m >=` the failed cell is broken until the communicators are rebuilt
  at a step boundary.
- **The core dies.** This is fatal, as it is today.
- **A transition hangs** (offload, pull or abort never completes). A per-transition
  timeout fails the step. The scheduler refuses new transitions while one is in
  progress, so a hung transition stalls further moves but never corrupts the role
  state.

## Where it does not help

- **Environment-bound rollouts.** When the time is spent in sandboxes or tool calls
  rather than on GPUs, moving cells to training early saves GPU-hours. It does not
  save wall-clock time if async training already hides behind the rollout tail.
- **Short or uniform rollouts.** With no long tail, `w` never falls far enough below
  `C_R - C_e` to free a cell early.
- **Durations clustered at the timeout.** The 2.4T result (1.21x over Colocate)
  shows the gain shrinking as termination times bunch up.
- **Small budgets.** One or two cells (`K <= 2`) collapse to colocate. The paper
  notes the method needs enough GPUs for several cells, each a complete training
  replica.
- **Training-bound workloads.** If training dominates, there is little rollout idle
  time to reclaim.
- **`mu > 1` and data order.** Batches inside a burst come from one dispatch, so there
  is no global shuffle across the burst.
- **`C_e` is concurrency, not throughput.** Eq. 5 assumes a cell's slots are
  interchangeable. Long-context executions with large KV can make a cell
  memory-bound before it is slot-bound.

## Phased plan

| Phase | Scope | Exit criterion |
|---|---|---|
| 1 (this PR) | Pure policy (eqs. 3-6, supply hold, serial core-first order, group-unit boundary), unit tests, a discrete-event simulation (DES) test, this doc. No wiring, no config | Unit tests pass |
| 2 (tool in this PR) | `tools/elastic_scheduler_replay.py`: replay recorded `rl.sc.generate_and_push` durations through the policy against a fixed split and colocate. Remaining work is running it on real traces. The #4914 rollout-only simulator is a cheap way to record a trace for a new recipe without a trainer | Projected win over the best fixed split on a real NeMo RL workload; go/no-go |
| 3 | Counters and the capacity gate in `_rollout_pump`, plus metrics (`w`, `C_R`, cell roles, supply). `async_rl.elastic_scheduler` BaseModel, opt-in and off by default, with the gate as its first consumer | No behavior change when disabled; parity when `K = 1` |
| 4 | `LENT` fleet-health state, router destination priority, per-cell drain/abort rerouting, router and fleet-health support for MInf | Cells removed from and restored to serving with no lost executions |
| 5 | Core-only switching (the paper's C2): the core moves to training on eq. 5 instead of waiting for the full batch. This lifts the colocated full-batch requirement | Matches or beats today's colocated SC |
| 6 | Satellites: gradient-only worker groups, prefix communicators, weight pull, central reduction. `mu = 1` (E1) | Loss and grad-norm parity with a single replica; wall-clock win on a tail-heavy task |
| 7 | Multi-step bursts (`mu > 1`) with eq. 6 core retention, and the `phi` dispatch offset. KV migration optional | E0-style gain over E1 |
| 8 | Failure hardening: `retire_cell`, satellite loss with chunk re-queue, communicator rebuild, transition timeouts | Fault-injection tests |

Phases 3-5 are useful on their own. Phase 5 alone corresponds to C2, which the paper
measures at 1.21x over plain Colocate (C0).

## Open questions and risks

- **Cell engine.** MInf is the only colocated engine the SC supports, so cells are
  MInf (see [Router and colocation constraints](#router-and-colocation-constraints)).
  vLLM cells would need SC support for colocated vLLM with sleep/wake, which is not
  planned.
- **Routing versus prefix cache.** The router is least-outstanding and not sticky, so
  multi-turn calls already lose prefix-cache locality. Priority routing does not fix
  that.
- **Units.** Resolved at the boundary (see
  [Units](#units-groups-at-the-boundary-executions-inside)). The remaining cost is
  that group-level accounting is conservative until Gym reports per-execution
  completion.
- **Communicator memory.** Estimated at ~150 MiB per prefix communicator per GPU
  (ring, 16 channels). The paper's `K-1` sets cost ~1 GiB on each core GPU at
  `K = 8`. Splitting with shared resources (option B) brings this to one
  communicator's worth. Unmeasured until Phase 6.
- **Replay fidelity.** The replay ignores load-dependent durations and generation
  pauses during refit. It is a go/no-go filter, not a speedup promise.
- **Scope.** GRPO/GSPO-style group algorithms with the Megatron backend only. PPO with
  a critic, DTensor and per-role layouts from #4801 (mixed widths, `worker_offset`)
  are out of scope until a satellite layout story exists.
- **Value without phases 3-6.** In this PR the policy's only consumer is the replay
  tool. Training benefits only once Phase 2 shows a win on our workloads and
  Phases 3-6 land.
