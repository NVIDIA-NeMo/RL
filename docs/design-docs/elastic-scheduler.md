# Elastic Scheduler

This document proposes an elastic scheduler for the SingleController (SC): GPUs are
grouped into *cells* that switch between serving rollouts and hosting a training
replica as the rollout workload drains, instead of keeping a fixed rollout/training
split (async) or switching the whole pool at once (colocate). It follows section 4 of
the QwenGyre paper (arXiv 2609.33848).

Only the scheduling policy exists today:
`nemo_rl/algorithms/single_controller_utils/elastic_scheduler.py`. It is a pure state
machine with no Ray, GPU or asyncio dependency, and **nothing in the SC calls it
yet**. There is no config key and no change in behavior. The rest of this document is
the plan for wiring it in.

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

### What the Phase-1 module provides

`ElasticScheduler(policy=ElasticSchedulerPolicy(...), clock=...)`:

- Counters: `record_dispatch`, `record_launch` (refuses more than `launchable()`), and
  `record_finish` (refuses more than are running). Derived values are `queued`,
  `running`, `waterlevel` and `rollout_capacity`.
- Decisions:
  - `launchable()` implements eq. 4.
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
  re-sends the call to a remaining backend.
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
  order) in place of pure least-outstanding.
- Router and fleet-health support for MInf. Today only vLLM implements
  `attach_fleet_health`, and the router is vLLM-only.
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
no communicator creation on the hot path. The cost is NCCL memory for `(K-1) x
DP x TP x PP` extra communicators, which is listed under risks.

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
| 1 (this PR) | Pure policy (eqs. 3-6, supply hold, serial core-first order), unit tests, a discrete-event simulation (DES) test, this doc. No wiring, no config | Unit tests pass |
| 2 | Offline replay: feed recorded per-execution durations and train times (or the #4914 rollout-only simulator) through the policy, and compare against async and colocate | Projected win on a real NeMo RL workload; go/no-go |
| 3 | Counters and the capacity gate in `_rollout_pump`, plus metrics (`w`, `C_R`, cell roles, supply). `async_rl.elastic_scheduler` BaseModel, opt-in and off by default, with the gate as its first consumer | No behavior change when disabled; parity when `K = 1` |
| 4 | `LENT` fleet-health state, router destination priority, per-cell drain/abort rerouting, router and fleet-health support for MInf | Cells removed from and restored to serving with no lost executions |
| 5 | Core-only switching (the paper's C2): the core moves to training on eq. 5 instead of waiting for the full batch. This lifts the colocated full-batch requirement | Matches or beats today's colocated SC |
| 6 | Satellites: gradient-only worker groups, prefix communicators, weight pull, central reduction. `mu = 1` (E1) | Loss and grad-norm parity with a single replica; wall-clock win on a tail-heavy task |
| 7 | Multi-step bursts (`mu > 1`) with eq. 6 core retention, and the `phi` dispatch offset. KV migration optional | E0-style gain over E1 |
| 8 | Failure hardening: `retire_cell`, satellite loss with chunk re-queue, communicator rebuild, transition timeouts | Fault-injection tests |

Phases 3-5 are useful on their own. Phase 5 alone corresponds to C2, which the paper
measures at 1.21x over plain Colocate (C0).

## Open questions and risks

- **Cell engine.** MInf is the only colocated engine the SC supports. Elastic cells
  with vLLM would need SC support for colocated vLLM with sleep/wake.
- **Routing versus prefix cache.** The router is least-outstanding and not sticky, so
  multi-turn calls already lose prefix-cache locality. Priority routing does not fix
  that.
- **Units.** The SC admits and trains in groups, while the paper counts executions.
  The gate launches whole groups and keeps capacity in executions. Partial groups
  never launch.
- **Communicator memory.** `K-1` prefix sets per core rank may not fit at large `K`
  and DP. A fallback is to build `G_{m,r}` lazily at the step boundary when membership
  changes.
- **Scope.** GRPO/GSPO-style group algorithms with the Megatron backend only. PPO with
  a critic, DTensor and per-role layouts from #4801 (mixed widths, `worker_offset`)
  are out of scope until a satellite layout story exists.
- **Value without phases 2-6.** The Phase-1 module has no production consumer. Its
  value depends on Phase 2 confirming a win on our workloads.
