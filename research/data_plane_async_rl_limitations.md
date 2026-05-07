# NeMo-RL Data Plane — Async RL Limitations & Risks

**Owner:** zhiyul
**Date:** 2026-05-04
**Status:** v2 — was a scoping note; now includes a concrete recommended path (§5). The risk register (§2) is the analytical basis; §5 is the proposed implementation.
**Companion documents:** [`data_plane_integration_plan.md`](./data_plane_integration_plan.md), [`data_plane_test_plan.md`](./data_plane_test_plan.md)

---

## 0. TL;DR

- TQ today is a **KV-by-uid store with per-task barrier semantics**, sized for *intra-step* tensor transport between fixed barriers (driver → DP, generation → ref/old-logp → train).
- Async GRPO is an **inter-step producer/consumer queue** with weight-version-tagged trajectories, bounded buffering, age-based filtering, and pause/resume around refit.
- These are different abstractions. Forcing async onto today's TQ surface either (a) loses safety properties async relies on, or (b) reintroduces a parallel control plane on the consumer side, defeating the point of routing through TQ.
- **verl arrived at the same boundary independently** — `verl/experimental/fully_async_policy/` ships its own `MessageQueue` rather than extend TransferQueue, and `verl/experimental/one_step_off_policy/` doesn't reference TQ at all. Treat that as evidence, not coincidence.
- **Recommendation (§5):** TQ as data plane, existing `ReplayBuffer` as control plane. Extend `ReplayBuffer` to hold `KVBatchMeta` instead of tensor batches; tensors live in TQ. Zero TQ controller changes, zero new abstractions. ~70 lines of new code, sync TQ path untouched.

---

## 1. Where the gap is, in one diagram

```
                 SYNC (today, on TQ)                 ASYNC (today, in-memory)
                 ───────────────────                 ────────────────────────
   driver        shard step N → kv_batch_put         (trainer, single actor)
                 dp_metas[i] → DP rank i                    │
                                                            ▼
   workers       per-rank kv_batch_get                AsyncTrajectoryCollector
                 train(meta_i) ─────────►            (Ray actor, long-running)
                                                            │
   barrier       partition per step;                        ▼  per-trajectory
                 kv_clear at boundary                  ReplayBuffer (Ray actor)
                                                       max_size, version-tagged
                                                            │
                                                            ▼  sample(cwv, age)
                                                          trainer step
```

The sync path's "barrier" — every key has a single producer, a single set of consumers known in advance, and a clean step boundary at which the partition is drained — is exactly what TQ is designed for. The async path replaces the barrier with a **streaming, multi-version, bounded-with-eviction** pipe. None of those words appear in TQ's controller today.

---

## 2. Risk register

Each entry is `R-<n>: <one-line risk>` followed by *Why it matters*, *What's missing in TQ*, and *Workaround cost*. Risks are ordered by how load-bearing they are for async correctness.

### R-1. No first-class weight-version axis on keys

**Why it matters.** Async GRPO's correctness depends on `min_valid_version ≤ traj_version ≤ current_weight_version` and `target_weight_version == cwv` filtering (`nemo_rl/algorithms/async_utils.py:135-172`). The version is *the* discriminator that prevents off-policy drift past the importance-sampling regime.

**What's missing.** TQ keys are opaque strings. `KVBatchMeta.tags` (per-key dict) can carry a version number, but there is no controller-side index that lets a consumer say "give me up to N keys whose tag.version ∈ [a, b] and that no consumer has read yet." The fetch primitive is `kv_batch_get(keys=[...])` — exact uid list known to the caller.

**Workaround cost.** Either (a) encode version in key (`gen{wv}_traj{n}`) and rebuild the version index in a Ray actor that subscribes to `kv_batch_put`, or (b) keep the existing `ReplayBuffer` actor and demote TQ to "tensor blob storage, key tagged with version." Option (b) is what the next risk (R-2) drives toward, but at that point most of the *value* of TQ for the async path is gone — you've just moved the tensor bytes off the object store.

### R-2. No bounded queue / eviction primitive

**Why it matters.** `ReplayBuffer` enforces `max_size = num_prompts_per_step * max_age * 2` with FIFO eviction on overflow (`async_utils.py:43-71`). Without that bound, a fast generator (e.g. weight version pinned for a slow refit) blows out controller memory.

**What's missing.** TQ has `kv_clear(keys, partition_id)` but it is *push-based GC by the writer*, not a queue cap. There is no "keep N most recent, drop the rest" or "evict when partition exceeds N samples" mode.

**Workaround cost.** Add a separate evictor actor that watches the partition and calls `kv_clear` — this is essentially `ReplayBuffer` with extra hops. Or accept unbounded growth and rely on max_age TTL eviction (R-7), which is monotonic in time but not in count, so a long generation step still spikes memory.

### R-3. No filtered fetch / consumer-side query

**Why it matters.** `ReplayBuffer.sample(current_weight_version, max_age, n)` does **conditional selection**: filter by version range, prefer trajectories whose `target_weight_version == cwv`, stall if not enough are ready (`async_utils.py:102-217`).

**What's missing.** The two TQ fetch modes are:
- `get_meta(partition_id, task_name, required_fields, batch_size, ...)` — task-mediated, advances per-task counter, returns *up to* `batch_size` produced samples. No filter expression.
- `kv_batch_get(keys=[...])` — exact uid list, caller already knows the keys.

Neither expresses "any N keys where tag.version is in this range." The closest emulation is "subscribe to all puts, mirror state in an external actor, do the filter there, then call `kv_batch_get`" — i.e. the `ReplayBuffer` actor in another shape.

**Workaround cost.** Same as R-1 (b): you keep the consumer-side selection logic, TQ just stores tensors. Reasonable compromise but the TQ surface is doing very little work.

### R-4. `production_status` is single-shot, not multi-consumer-aware

**Why it matters.** Per `nemo_rl/data_plane/interfaces.py:111-115`, the controller flips `production_status[sample, field] = 1` once on `kv_batch_put`. That flip is the "ready" signal downstream consumers wait on. It's a *level*, not an *edge*.

In sync GRPO, every sample has a known consumer set (`consumer_tasks=["ref_logp","old_logp","train"]`) and the partition is wiped at the step boundary, so the level model is fine.

In async, the trainer at version V consumes a *subset* of buffered trajectories (those targeting V); the rest stay live and may be selected at V+1. There is no way to express "this sample has been consumed by trainer@V but is still available for trainer@V+1" using `production_status` alone — multiple-consumer reuse turns into a manual key-lifecycle problem.

**What's missing.** A reference-counted or per-consumer-token consumption model. Not in TQ today.

**Workaround cost.** Don't use `get_meta` for trainer fetches at all on the async path; only use `kv_batch_get` driven by the `ReplayBuffer` selection. That works, but it means the per-task counter in `check_consumption_status` (`interfaces.py:172-179`) becomes meaningless for the trainer task on the async path, and any monitoring built on top of it breaks.

### R-5. `check_consumption_status` is a partition-level barrier

**Why it matters.** It returns true iff *every* consumer task has consumed *every* sample (`interfaces.py:172-179`). That's the wait condition for a clean step boundary on the sync path.

**What's missing.** Async never reaches "every consumer has consumed every sample" — there is always more in flight. The primitive doesn't apply, and any code that uses it as a quiescence check (e.g. before `kv_clear`-ing a step) silently misbehaves.

**Workaround cost.** Don't call it on the async path. Document that `check_consumption_status` is sync-only. Cheap, but it means the TQ surface has a method that's a no-op in half its use cases — small but real maintenance load.

### R-6. No producer pause / version gating on the controller

**Why it matters.** Async GRPO pauses the collector around refit (`AsyncTrajectoryCollector.pause / prepare_for_refit / resume_after_refit`, `async_utils.py:344-426`) so trajectories landing on disk during the refit window aren't tagged with a stale weight version. The collector's `_refit_pause_cleared` event is what enforces this — the *producer* gates itself.

If the producer is correct, TQ doesn't need to know. **But** any in-flight async generator that completes during refit will still call `kv_batch_put` with the old weight version and TQ will happily flip `production_status`. Today the collector serializes `pause → wait_in_flight → refit → resume`; if a future async path lets the generator continue across refit (via `in_flight_weight_updates`), TQ has no controller-side way to reject puts tagged with weight version < current.

**What's missing.** No put-time predicate ("reject if tag.version < N") on the controller.

**Workaround cost.** Keep the discipline at the producer (current model). Acceptable, but it means the async-via-TQ design inherits a correctness invariant that lives outside TQ — exactly the kind of out-of-band coordination R-1 was supposed to remove.

### R-7. GC / TTL is not version-aware

**Why it matters.** Sync `kv_clear`s at every step boundary — old keys are guaranteed dead. Async needs "drop keys whose `weight_version < current - max_age`," and that lower bound moves at trainer speed, not generator speed.

**What's missing.** No TTL primitive at all; `kv_clear` is by explicit key list or whole-partition wipe.

**Workaround cost.** External evictor actor (same actor as R-2's bounded-queue workaround). Doable but it's another moving part to checkpoint and recover.

### R-8. Driver-side balanced packing assumes a fixed step batch

**Why it matters.** The "driver-side balanced packing" trick in `grpo_sync.py:640-704` (the headline of commit `a085559c`) calls `shard_by_batch_size(dp_world, ..., bin_count_multiple=DP_world)` *once* on the full step batch, then ships per-rank pre-balanced metas to workers. This avoids the per-rank packing skew that deadlocked the 30B run at step 4.

The good news: async also has a single point at step time where the trainer pulls a chosen batch from the buffer. The driver still sees the full GBS pre-fan-out. So **this part ports cleanly**.

**Caveat.** The buffer composition is mixed-version; if any per-rank packing decision depends on metadata that varies with weight version (e.g. some encoder heuristic), the balanced-packing invariant `n_microbatches uniform across DP` could regress in non-obvious ways. Add this to the test plan if/when porting. Today there is no such version-dependent packing, so this is forward-looking, not active.

### R-9. Codec is tensor-only — async rollout outputs are richer

**Why it matters.** `kv_batch_put` rejects non-tensor leaves (`interfaces.py:199-200`, codec at `nemo_rl/data_plane/codec.py`). Sync rollouts already pre-tensorize.

Async multi-turn / agent-loop rollouts emit per-turn metadata (tool call traces, env states, partial reward signals). On the in-memory path these can ride along as Python objects in the BatchedDataDict. On TQ, they have to be serialized to tensors (or to bytes-in-tensor) up front, which expands the codec surface and makes debugging harder ("why is this tool trace coming back as a uint8 blob?").

**What's missing.** Nothing in TQ — this is a constraint that exists already on the sync path and just has a wider blast radius on the async path.

**Workaround cost.** Either tighten the codec (define a "blob field" with explicit serializer per field name) or keep non-tensor metadata in a side channel (back to a parallel control plane, defeats the point).

### R-10. Checkpoint surface expands to include TQ controller state

**Why it matters.** Sync GRPO checkpointing: trainer state + dataloader state. Period. The TQ partition for step N is ephemeral — it's wiped before checkpointing.

Async GRPO checkpointing today: trainer state + dataloader state + `ReplayBuffer.state_dict()` (versions, targets, trajectories). On a TQ-mediated async path, the `ReplayBuffer` equivalent is partially or wholly *inside* the TQ controller. Recovery becomes "restore TQ + trainer + collector to a coherent point in time."

**What's missing.** A TQ partition snapshot/restore primitive coordinated with trainer checkpoints. TQ doesn't ship one.

**Workaround cost.** Drain TQ at every checkpoint boundary (defeats async's whole point — you've added an artificial barrier) or build coordinated snapshot/restore. Real engineering effort, easy to get wrong, expensive to test.

### R-11. Failure-mode taxonomy doubles

**Why it matters.** Sync TQ failure modes: controller died, storage full, schema mismatch, key not found. Bounded list, all "fail loud" via the existing tests in §4.3 of the test plan.

Async adds: producer/consumer version skew (consumer reads V+1 keys before producer publishes V), eviction races (consumer fetches just-evicted key), backpressure deadlocks (buffer full, generator blocked, trainer waiting on a target version that will never be produced), pause/resume torn states. Each needs a targeted test.

**Workaround cost.** Roughly doubles the chaos-test surface in `data_plane_test_plan.md` §5.3 / §8. Not a blocker — just a real cost the schedule has to absorb.

### R-12. verl precedent: TQ was *not* extended for async

**Why it matters.** `grep -rln "transfer_queue" verl/verl/experimental/` returns empty. The two async paths (`one_step_off_policy/`, `fully_async_policy/`) bypass TQ. `fully_async_policy/message_queue.py` is a custom MessageQueue. This is the same team that wrote the TQ integration on the sync path. They had every incentive to extend TQ; they didn't.

This is not a hard constraint on us. But it's a strong signal that R-1..R-7 are not minor — at minimum it suggests *they* concluded that fixing them was more work than building a sibling abstraction. Worth replicating their reasoning before assuming we'll find a shortcut.

---

## 3. What you'd actually have to build

If we decide to support async on TQ, here is the minimum surface change, ordered by depth-of-change:

| # | Change | Where | Why |
|---|--------|-------|-----|
| 1 | Version-tagged keys + version index | TQ controller (or an indexer actor in `nemo_rl/data_plane/`) | R-1, R-3 |
| 2 | Filtered fetch: `(version_range, target_version)` predicate | New `kv_query` on `DataPlaneClient` | R-3 |
| 3 | Bounded partition with FIFO eviction | TQ controller config or external evictor | R-2 |
| 4 | Reference-counted / per-consumer consumption | TQ controller — non-trivial schema change | R-4 |
| 5 | Version-aware TTL on `kv_clear` | New `kv_clear_below_version(v)` | R-7 |
| 6 | Put-time predicate (reject `tag.version < N`) | TQ controller hook | R-6 |
| 7 | Coordinated snapshot/restore for partition + trainer | New checkpoint hook on `DataPlaneClient` | R-10 |
| 8 | Codec extension for non-tensor rollout metadata | `nemo_rl/data_plane/codec.py` | R-9 |
| 9 | Async-specific chaos tests | `tests/test_data_plane*` + nightly | R-11 |

That's a non-trivial program. Items 1–4 are the load-bearing ones; without them, the async-on-TQ path is just "ReplayBuffer with extra hops."

**But:** the recommendation in §5 leans into exactly that — *intentionally* "ReplayBuffer with extra hops" — because the existing `ReplayBuffer` already implements items 1–4 correctly and is already tested. The recommended path needs only **items 8 (codec extension, optional) and 9 (chaos tests)** from this table, plus small additions to existing files. Items 1–7 stay unfixed in TQ; ReplayBuffer covers them on the consumer side. See §5.

---

## 4. Four options, with honest costs

### Option A — Do the full extension (items 1–9 of §3)

**Pros.** One data plane, one mental model, TQ becomes load-bearing for both sync and async. Best long-term story.

**Cons.** Items 1, 4, and 7 are TQ-controller-side changes; we don't own that codebase. Even with upstream cooperation, easily a quarter of work before async parity. Item 4 is a schema change with backwards-compat implications.

**When to pick.** If TQ's roadmap already includes a producer/consumer-queue mode for other reasons. Don't drive that conversation from NeMo-RL alone.

### Option B — Sibling abstraction (verl pattern)

**Pros.** Don't touch TQ. Build a `nemo_rl/data_plane/queue/` MessageQueue (or wrap an existing one) for the trajectory pipe. TQ stays sync-only and stable. Well-trodden path — verl proves it works.

**Cons.** Two abstractions to learn, configure, and document. Easy to misroute (a sample lands in the wrong store). And — crucially for our codebase — `ReplayBuffer` is already a strict superset of MessageQueue (bounded eviction, version-aware sample, multi-target reuse), so adding MQ between them is dead weight.

**When to pick.** If we wanted to *replace* `ReplayBuffer` with something simpler and accept verl-level off-policy drift. We don't.

### Option B′ — TQ + extended `ReplayBuffer` (RECOMMENDED, see §5)

**Pros.** Reuses everything that already works. `ReplayBuffer` keeps its version filter, age gate, target-version selection, FIFO eviction, and `state_dict / load_state_dict` — none of which exist in TQ and none of which need to. The sync TQ path (`grpo_sync.py`) is completely untouched. ~70 lines of new code total.

**Cons.** Tensors travel TQ → `kv_batch_get` → driver materialize → repacked into per-DP-rank metas → TQ → workers `kv_batch_get`. Two TQ round-trips per step (vs. one for sync). At GBS scales we already run, this is dominated by NCCL collectives, but it's a real overhead.

**When to pick.** Now. Lowest schedule risk, smallest new test surface, preserves all existing async correctness guarantees.

### Option C — Keep async on the in-memory path (status quo)

**Pros.** Zero new code. Async already works.

**Cons.** No data-plane benefits for async (no observability hooks, no pluggable backend, no codec discipline). Two-tier story persists indefinitely.

**When to pick.** If async usage is small and not growing. This is where we are today.

**Recommendation:** Option B′. The remainder of the document (§5) details the implementation.

---

## 5. Recommended path: TQ as data plane, `ReplayBuffer` as control plane

### 5.1 The answer in one sentence

**Make `ReplayBuffer` hold `KVBatchMeta` references instead of tensor batches. Tensors live in TQ. Everything else — version filtering, age gating, target-version selection, FIFO eviction, checkpointing — stays exactly where it is in `nemo_rl/algorithms/async_utils.py`.**

### 5.2 Why this is the right answer

The five things async correctness depends on are *already implemented* and *already tested* in `ReplayBuffer`. None of them are in TQ. None of them need to be:

| Invariant | Where it lives | Code |
|---|---|---|
| Version filtering `min_valid ≤ v ≤ cwv` | `ReplayBuffer.sample` | `async_utils.py:135-151` |
| Target-version selection `target == cwv` | `ReplayBuffer.sample` | `async_utils.py:166-192` |
| Multi-consumer reuse (one traj, multiple targets) | `ReplayBuffer.add(target_weight_versions: list[int])` | `async_utils.py:74-77` |
| Bounded buffer + FIFO eviction | `ReplayBuffer.add` | `async_utils.py:69-71` |
| Checkpoint state | `ReplayBuffer.state_dict / load_state_dict` | exists |

`ReplayBuffer` doesn't care what `trajectory` *is*. It currently holds tensor batches; it could equally hold `KVBatchMeta`. The list-and-version bookkeeping never inspects the payload.

This means:

- ✅ Zero TQ controller changes.
- ✅ Zero new abstraction (no MessageQueue — see §5.6).
- ✅ Sync TQ path (`grpo_sync.py`) untouched — no regression risk.
- ✅ The driver-side balanced-packing trick from `grpo_sync.py:640-722` is reused as-is — the only thing that changes is what *produces* the BatchedDataDict at step time.

### 5.3 Data flow

```
   producer side                                          trainer side
   ─────────────                                          ────────────
   AsyncTrajectoryCollector
   ├─ rollout → batch_tensors
   ├─ kv_batch_put(traj_keys, partition="rollouts",
   │      fields=batch_tensors,
   │      tags=[{"version": v}, ...])  ← TQ holds bytes
   └─ replay_buffer.add(
          KVBatchMeta(keys=traj_keys, ...),
          weight_version=v,
          target_weight_versions=[v+1, ..., v+max_age])
                                                          replay_buffer.sample(
                                                              current_weight_version,
                                                              max_age_steps,
                                                              num_prompt_groups)
                                                          ↓
                                                          metas: list[KVBatchMeta]
                                                          ↓ (driver)
                                                          [kv_batch_get(m.keys) for m in metas]
                                                          ↓
                                                          train_data: BatchedDataDict
                                                          ↓ (FROM HERE = grpo_sync.py:640-722)
                                                          shard_by_batch_size(dp_world, …)
                                                          ↓
                                                          dp_metas (per-rank)
                                                          ↓
                                                          policy.train(dp_metas)
                                                            └─ @dp_dispatch list[KVBatchMeta]
                                                               (already exists, dispatch.py:88)
```

The trainer step is **identical to today's sync TQ path** from `shard_by_batch_size` onward. The only new logic is the four-line preamble that turns "sample N metas → materialize" into a `BatchedDataDict`.

### 5.4 The four touchpoints

**(1) `AsyncTrajectoryCollector` — TQ producer hook.**

`async_utils.py` (~10 lines added inside the existing collector loop). The actual buffer method is `push_with_wait_signal` (`async_utils.py:55-82`), not `add`; it returns `"full"` / `"success"`. Use the loop's running event loop to `await kv_batch_put` rather than `asyncio.run` (avoids the running-loop conflict — see §5.9 Race 3):

```python
# was:
#     status = replay_buffer.push_with_wait_signal(
#         batch_tensors, weight_version, target_weight_version)
# becomes (when data_plane.enabled):
keys = [f"v{weight_version}_p{prompt_id}_g{i}" for i in range(n_samples)]
await dp_client.kv_batch_put(                # await directly — collector loop is async
    keys=keys, partition_id="rollouts",
    fields=batch_tensors,
    tags=[{"version": weight_version}] * len(keys),
)
meta = KVBatchMeta(partition_id="rollouts", keys=keys, ...)
status = replay_buffer.push_with_wait_signal(
    meta, weight_version, target_weight_version
)
if status == "full":
    # buffer rejected — bytes already in TQ; clear them or they leak
    await dp_client.kv_clear(keys, partition_id="rollouts")
```

The trailing `kv_clear` on `"full"` is the reverse of §5.9 Race 1: if the buffer rejects after we wrote to TQ, we own the cleanup.

**(2) `ReplayBuffer` — TQ-aware GC at both eviction *and* consume.**

The §2 R-7 sketch said "eviction calls `kv_clear`." That's necessary but **not sufficient** — every consumed trajectory also needs its TQ keys cleared, otherwise TQ leaks at the rate of training throughput (~`num_prompts` keys per step). See §5.9 Race 1 for the full analysis.

> **Async uses targeted-key clears, never partition-wide wipes.** The sync trainer can do `dp_client.kv_clear(keys=None, partition_id="train")` (`grpo_sync.py:1072`) at the end of each step because (a) all workers have returned before the driver reaches that line — Ray fan-out barrier, and (b) keys are step-namespaced (`f"step{N}_dp{r}_s{i}"`) so step-N keys are dead at step N+1. Async has *no* step barrier; a partition-wide wipe would destroy in-flight rollout data the trainer hasn't consumed yet. All async clears must be per-meta: `dp_client.kv_clear(m.keys, m.partition_id)`. The sketches below follow this rule.

Two additions to `ReplayBuffer`, both inside the buffer's lock so push/sample stay serialized (§5.9 Race 5):

```python
class ReplayBuffer:
    def __init__(self, max_size: int, dp_client: DataPlaneClient | None = None):
        ...
        self._dp_client = dp_client       # None for the in-memory path

    def push_with_wait_signal(self, trajectory, weight_version, target_weight_version):
        with self._lock:
            if len(self.trajectories) >= self.max_size:
                return "full"
            # (no eviction-on-overflow today — push returns "full" and
            # the producer is expected to retry. If/when eviction lands,
            # add: kv_clear(evicted.keys) under this same lock.)
            ...

    def sample(self, num_prompt_groups, current_weight_version, max_age_steps):
        with self._lock:
            ...
            consumed_metas = [self.trajectories[i] for i in selected_indices]
            for idx in sorted(selected_indices, reverse=True):
                self.trajectories.pop(idx)
                self.trajectory_versions.pop(idx)
                self.target_weight_versions.pop(idx)
            # Free TQ payload BEFORE returning so the trainer can't observe
            # an inconsistent (meta-popped, key-still-live) state.
            if self._dp_client is not None:
                for m in consumed_metas:
                    self._dp_client.kv_clear(m.keys, m.partition_id)
            return {"trajectories": consumed_metas, "avg_trajectory_age": ...}
```

The `dp_client.kv_clear` call goes to a Ray actor (sub-ms latency) and is held under the buffer lock. Trade-off: push/sample see ~`O(num_consumed_per_step)` extra under-lock time. At realistic batch sizes this is negligible vs. the actual training step. Releasing the lock around `kv_clear` is *not safe* — see §5.9 Race 5.

The `dp_client=None` default preserves the in-memory path when `data_plane.enabled=False`. ~10 lines net.

**Periodic stale-version GC** (defends against §5.9 Race 2): when the trainer calls `set_weight_version(v)` (`async_utils.py:344`), scan and `kv_clear` any meta with `traj_version < v − max_age` that the version filter would otherwise leave stranded:

```python
def set_weight_version(self, version: int):
    with self._lock:
        ...
        cutoff = version - self._max_age_steps
        stale_idx = [i for i, v in enumerate(self.trajectory_versions) if v < cutoff]
        if stale_idx and self._dp_client is not None:
            for i in stale_idx:
                self._dp_client.kv_clear(
                    self.trajectories[i].keys, self.trajectories[i].partition_id
                )
        for i in sorted(stale_idx, reverse=True):
            self.trajectories.pop(i); self.trajectory_versions.pop(i); self.target_weight_versions.pop(i)
```

~10 lines, runs O(buffer_size) at refit time only. Trivial cost; closes Race 2.

**(3a) Extract driver-side balanced packing as a shared helper.**

Today `grpo_sync.py:605-704` inlines ~100 lines of "compute pre-shards with `bin_count_multiple=DP_world`, then for each pre-shard `kv_batch_put` seed fields and build a `KVBatchMeta`." That block is going to be **identical** in `grpo_async_dp.py` — refactor it before duplicating.

Two distinct concerns, two helpers, in a new module **`nemo_rl/data_plane/preshard.py`** (separate from `nemo_rl/data_plane/sharding.py`, which is metadata-only sort-by-seqlen for `@dp_dispatch`):

```python
# nemo_rl/data_plane/preshard.py

def driver_balanced_preshards(
    train_data: BatchedDataDict,
    *,
    dp_world: int,
    policy_cfg: dict,
) -> list[BatchedDataDict]:
    """Shard with bin_count_multiple=dp_world — keeps per-rank n_microbatches
    uniform across DP. Without this, sequence-packing / dynamic-batching produce
    variable per-rank bin counts and Megatron deadlocks at the first cross-DP
    collective. See commit a085559c. Pure transform; no I/O, no TQ."""
    seqpack_cfg = policy_cfg.get("sequence_packing", {}) or {}
    dynbatch_cfg = policy_cfg.get("dynamic_batching", {}) or {}
    gbs = policy_cfg["train_global_batch_size"]
    if dynbatch_cfg.get("enabled", False):
        dba = {...}  # current grpo_sync.py:615-625 body
        return train_data.shard_by_batch_size(dp_world, batch_size=gbs, dynamic_batching_args=dba)[0]
    if seqpack_cfg.get("enabled", False):
        spa = {...}  # current grpo_sync.py:626-637 body
        return train_data.shard_by_batch_size(dp_world, batch_size=gbs, sequence_packing_args=spa)[0]
    return train_data.shard_by_batch_size(dp_world, batch_size=gbs)


def fan_out_per_rank_metas(
    pre_shards: list[BatchedDataDict],
    *,
    dp_client: DataPlaneClient,
    partition_id: str,
    key_prefix: str,                 # e.g. f"step{total_steps}" or f"v{wv}_step{step}"
    seed_fields: list[str],
) -> list[KVBatchMeta]:
    """For each pre-shard: kv_batch_put seed fields, build KVBatchMeta with
    micro_batch_indices/lengths/elem_counts_per_gb in extra_info so
    train_presharded can reattach packing metadata post-fetch."""
    # current grpo_sync.py:657-704 body, with key namespace parameterized
```

Both helpers are pure functions of their args — easy to unit test, easy to mock `dp_client` for the second.

**(3b) `grpo_sync.py` shrinks to use the helpers.**

The ~100-line block at `grpo_sync.py:605-704` collapses to:

```python
pre_shards = driver_balanced_preshards(
    train_data, dp_world=dp_world, policy_cfg=master_config["policy"],
)
dp_metas = fan_out_per_rank_metas(
    pre_shards,
    dp_client=dp_client,
    partition_id="train",
    key_prefix=f"step{total_steps}",
    seed_fields=_DP_SEED_FIELDS,
)
```

This refactor lands as **PR 0** (see §5.8) so it's covered by the existing sync parity tests (`data_plane_test_plan.md` §4.5) before the async path consumes it.

**(3c) New trainer entrypoint — `nemo_rl/algorithms/grpo_async_dp.py`.**

Mirrors the sibling pattern (`grpo_sync.py` is to `grpo.py` as `grpo_async_dp.py` is to `async_grpo_train`). The inner step body uses the same helpers:

```python
# 1. Sample metas from ReplayBuffer (filter/version/age handled internally)
sampled = ray.get(replay_buffer.sample.remote(
    current_weight_version=weight_version,
    max_age_steps=max_trajectory_age_steps,
    num_prompt_groups=num_prompts_per_step,
))
if sampled is None:
    continue  # buffer not ready yet; collector will catch up
rollout_metas: list[KVBatchMeta] = sampled["trajectories"]

# 2. Materialize on the driver — one round-trip per meta, by-key
materialized = [dp_client.kv_batch_get(m.keys, m.partition_id) for m in rollout_metas]
train_data = concat_batched_dicts(materialized)

# 3. Driver-side balanced packing + per-rank fan-out (SHARED HELPERS)
pre_shards = driver_balanced_preshards(
    train_data, dp_world=dp_world, policy_cfg=master_config["policy"],
)
dp_metas = fan_out_per_rank_metas(
    pre_shards,
    dp_client=dp_client,
    partition_id="train",
    key_prefix=f"v{weight_version}_step{total_steps}",   # versioned namespace
    seed_fields=_DP_SEED_FIELDS,
)

# 4. Existing @dp_dispatch list[KVBatchMeta] path (dispatch.py:88-92)
train_results = policy.train(dp_metas, loss_fn=loss_fn, timer=timer)
```

The `policy.train(dp_metas)` call uses the *existing* `@dp_dispatch list[KVBatchMeta]` path. No new dispatch logic. The async-specific code in this file is the outer loop / refit / validation / checkpointing — all copyable from `async_grpo_train` — plus the 4 lines for sample-and-materialize. **The packing logic itself is not duplicated.**

**(4) `examples/run_grpo.py` — extend dispatcher.**

```python
if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
    if master_config.get("data_plane", {}).get("enabled", False):
        from nemo_rl.algorithms.grpo_async_dp import async_grpo_train_dp
        trainer = async_grpo_train_dp
    else:
        from nemo_rl.algorithms.grpo import async_grpo_train
        trainer = async_grpo_train
else:
    # existing sync dispatch unchanged
    ...
```

~5 lines.

### 5.5 Total new code

| Component | New / Net | Reuses |
|---|---|---|
| `nemo_rl/data_plane/preshard.py` helpers | ~80 new (extracted from `grpo_sync.py:605-704`) | existing `BatchedDataDict.shard_by_batch_size` |
| `grpo_sync.py` refactor to call helpers | **−95 / +5 net** | helpers above |
| `AsyncTrajectoryCollector` TQ producer hook | ~12 new (incl. "full"-rejection rollback) | existing collector loop |
| `ReplayBuffer` TQ-aware GC (consume + stale-version) | ~25 new | existing `_lock`, `sample`, `set_weight_version` |
| `ReplayBuffer.load_state_dict` orphan-key reconciliation | ~15 new | existing state_dict path |
| `grpo_async_dp.py` step body | ~25 new (4-line preamble + outer loop boilerplate) | preshard helpers, `async_grpo_train` outer loop |
| `run_grpo.py` dispatch | ~5 new | existing pattern |
| **Net new lines** | **~62** | — |

The refactor PR (extracting `preshard.py`) is net-zero on production code count — it just moves ~80 lines from inline to a helper. The async-specific work is the GC plumbing (Race 1 / Race 2 / R-10 fixes) — bigger than the ~30 originally listed because §5.9 surfaced two real GC gaps that have to land for correctness, not just polish.

No new ABC. No new actor. No new partition schema. No TQ controller changes.

### 5.6 Why not MessageQueue (Option B)

For our codebase, `ReplayBuffer` is a strict superset of MessageQueue:

- MQ has bounded eviction → `ReplayBuffer.max_size` already does it.
- MQ has version-blind FIFO → `ReplayBuffer.sample` does *better* (version-aware).
- MQ has `asyncio.Condition` blocking pull → `ReplayBuffer.sample` returns `None` and the caller polls. Blocking ergonomics is ~10 extra lines if we want it later, **and is not load-bearing for correctness**.
- MQ has `None` termination sentinel → `ReplayBuffer.clear()` exists; shutdown coordination already lives in `AsyncTrajectoryCollector`.

Adding MessageQueue would mean three abstractions (TQ + MQ + ReplayBuffer) where two suffice. Don't.

### 5.7 What this *doesn't* solve from §2

Honest about what's still open under Option B′:

- **R-6 (no producer-side version gating).** Discipline stays at the producer (collector pauses around refit). Same as today's in-memory async path. Acceptable, but compounds with §5.9 Race 2 — see fix there.
- **R-9 (codec tensor-only).** `KVBatchMeta.extra_info` already carries non-tensor metadata for sync packing — same channel works for async rollout traces. Codec extension only needed if a richer schema lands.
- **R-10 (checkpoint surface).** `ReplayBuffer.state_dict()` now contains key strings instead of tensors. Restore needs **bidirectional** orphan reconciliation — keys-in-TQ-not-in-buffer (clear them) AND keys-in-buffer-not-in-TQ (drop the meta from the buffer with a warning). See §5.9 Race 4 for the full sketch (~15 lines).
- **R-11 (chaos surface).** Real, but additive to the existing async test harness — independent of which option we pick. The §5.9 races each need a targeted test in PR 5.

None of these are blockers; all have known workarounds inherited from the in-memory async path or are spelled out in §5.9.

### 5.8 Suggested PR order

Each PR is independently revertable. The first three land before any user-facing change.

1. **PR 0** *(refactor only, no behavior change)*: extract `nemo_rl/data_plane/preshard.py` (`driver_balanced_preshards`, `fan_out_per_rank_metas`). Replace `grpo_sync.py:605-704` with the two helper calls. Covered by existing sync parity tests (`data_plane_test_plan.md` §4.5) — if those pass, the refactor is correct. **This is the prerequisite that prevents the packing block from being duplicated in step 4.**
2. **PR 1** *(test-only)*: confirm `KVBatchMeta` round-trips through `ReplayBuffer.state_dict / load_state_dict` cleanly. No production code change. Validates the unverified assumption in §5.2.
3. **PR 2**: `ReplayBuffer.push_with_wait_signal` accepts `KVBatchMeta`; `sample` and `set_weight_version` call `kv_clear` for consumed and stale-version metas; `load_state_dict` does bidirectional orphan reconciliation. All gated on `dp_client is not None` so the in-memory path is unaffected. Closes §5.9 Races 1, 2, and 4. Sync TQ path completely untouched.
4. **PR 3**: `AsyncTrajectoryCollector` TQ producer hook, behind `data_plane.enabled`. Uses `await` (not `asyncio.run`) — see §5.9 Race 3. Includes `kv_clear` rollback on `"full"` rejection. Producer is runnable end-to-end but has no consumer.
5. **PR 4**: `grpo_async_dp.py` + `run_grpo.py` dispatch. Reuses `preshard.py` helpers from PR 0. End-to-end async-on-TQ path callable.
6. **PR 5**: Async-specific chaos tests (mirror `data_plane_test_plan.md` §5.3 / §8 for the async path — eviction races, version skew at restart, refit-window key handling).

By PR 4 the path is functional behind a config flag. By PR 5 it has the same chaos-test coverage the sync path got. PR 0 is the one ordering constraint: it must precede PR 4, so the async trainer never copy-pastes the packing block.

### 5.9 Concurrency & races

There are no file locks anywhere in `nemo_rl/` (`grep -rn "FileLock|fcntl|filelock"` is empty). Synchronization is one of three things:

| Mechanism | Where | Notes |
|---|---|---|
| `threading.Lock` | `ReplayBuffer._lock` (`async_utils.py:53`); `AsyncTrajectoryCollector._pg_lock`, `_threads_lock`, `_generation_check_lock` (`:258-290`) | Cross-thread serialization within a single Ray actor. |
| `threading.Event` | `_manual_pause_cleared`, `_refit_pause_cleared`, `_generation_limit_cleared` (`:261-274`) | Pause/resume signaling for the producer thread. |
| Ray actor model | `ReplayBuffer`, `AsyncTrajectoryCollector`, TQ controller (named global actor) | Ray serializes method calls per actor. The `threading.Lock`s above are technically redundant under Ray, but defensive — they catch the bug if anyone non-actorizes a class later. |

The "one Ray cluster per experiment" assumption (`nemo_rl/data_plane/README.md:99`) removes the need for inter-process file locks: the TQ controller is a globally named Ray actor, so two concurrent experiments collide at actor-name conflict, which fails loud at startup.

#### Atomicity & ordering guarantees the controller already gives us

Three properties are load-bearing for everything below; spelling them out so the races aren't re-derived each time:

1. **`kv_batch_put` is atomic from the consumer's point of view.** The adapter wraps the underlying call in `await asyncio.to_thread(self._tq.kv_batch_put, ...)` (`transfer_queue.py:461-467`) and the call is one Ray actor method on the named global controller. The producer `await` blocks until the controller flips `production_status` for *every* `(sample, field)` pair in the batch — this is the ACK flow (NOTIFY_DATA_UPDATE → bit-flip under `data_status_lock` → NOTIFY_DATA_UPDATE_ACK → producer unblocks). Consumers see the entire batch or none of it; **never partial.**
2. **Single-threaded controller request loop.** The TQ controller's `_process_request` is a single ZMQ loop; client RPCs are serialized by it without any app-level locking on our side. `data_status_lock` exists only to interlock that loop against `_update_data_status` (the storage-NOTIFY handler). Application code that targets the controller — `kv_batch_put`, `kv_batch_get`, `kv_clear`, `get_meta`, `check_consumption_status` — is already serialized end-to-end. We add no controller-side locks.
3. **`kv_clear` is unconditional.** `clear_partition` does not consult `production_status` or per-task consumption counters; it pops keys regardless (controller.py:1482). Sync GRPO can rely on a structural step barrier (Ray fan-out + step-namespaced keys) to make whole-partition wipes safe; async cannot. See §5.4(2)'s "targeted-key clears" callout.

So none of the five races below are about "consumers see partial bytes" or "can the put race itself" — those are precluded. They are about **cleanup, semantic version-tag skew, async-loop nesting, cross-store coherence, and compound-operation atomicity** respectively.

Five concrete races to plan around. Race 1 was a bug in earlier drafts of §5; Races 2–5 are mitigations called out explicitly so they don't get forgotten:

#### Race 1 — `kv_clear`-on-consume (memory leak in TQ)

**The bug.** Earlier drafts of §5.4(2) said "eviction calls `kv_clear`" but never cleared keys for *consumed* (sampled) trajectories. `ReplayBuffer.sample` pops them at `:214`, but their TQ payload lives on indefinitely.

**Math of the leak.** Buffer holds `O(num_prompts × max_age)` metas at steady state; trainer consumes `num_prompts` per step. Without consume-time GC, **every step leaks `num_prompts` worth of TQ keys** — leak rate equals training throughput. After `N` steps, `N × num_prompts × n_gens × per_traj_bytes` orphans in TQ. Linear leak; not survivable.

**Fix.** §5.4(2) — `ReplayBuffer.sample` calls `kv_clear` on consumed metas under the buffer lock, before returning.

#### Race 2 — Refit-window semantic version-tag skew

**This race is *not* about partial visibility** (atomicity guarantee 1 above precludes that). It's about the version label captured at generation start versus the trainer's weight version when the meta finally lands in the buffer.

**Sequence (with atomicity made explicit).**

```
T0  Collector reads `current_weight_version = V_old`. Generation begins at V_old.
T1  Generator (long; vLLM async_engine) runs while trainer continues stepping.
T2  Generator returns n samples. Collector calls `await dp_client.kv_batch_put(...)`.
T3        ┌─ TQ atomic write begins. During this window: refit may complete →
          │  trainer bumps weight version to V_new.
T4        └─ kv_batch_put returns. ACK confirms production_status bit is set.
T5  Collector calls `replay_buffer.push_with_wait_signal(meta, weight_version=V_old, ...)`.
T6  Trainer at V_new calls sample(cwv=V_new, max_age=K).
```

The race lives between T0 and T5: the collector tags the meta with `V_old` (correct — that *is* when generation happened), but the meta only becomes visible to the trainer at T5, by which point `current_weight_version` may already be `V_new`. The TQ write itself (T3→T4) is atomic and post-ACK fully visible — that's not the issue.

**What happens at T6.** Filter checks `V_new − K ≤ V_old ≤ V_new`:
- `V_old ≥ V_new − K` → meta is in-window, used. Fine.
- `V_old < V_new − K` → meta is stale. Filter rejects it, but `sample` only pops *consumed* metas — the stale meta sits in the buffer un-sampled. Since today's `push_with_wait_signal` *rejects* when full rather than evicting (`async_utils.py:69-71`), an unsampled stale meta might never be removed at all. Combined with Race 1's pre-fix state, those TQ keys leaked too.

**Fix.** §5.4(2) — `ReplayBuffer.set_weight_version` does a stale-version GC pass: scan for `traj_version < cwv − max_age`, `kv_clear` them, drop from the buffer. O(buffer_size) per refit; closes the gap.

#### Race 3 — `asyncio.run` from a running event loop

**Where it would bite.** `grpo_sync.py:676` does `asyncio.run(dp_client.kv_batch_put(...))` from synchronous trainer code — no enclosing loop, fine. But `AsyncTrajectoryCollector` has internal threads and the proposed producer hook would call `asyncio.run` from inside one. If the collector runs inside an asyncio loop (vLLM `async_engine` integration may require this — needs verification at PR 3), `asyncio.run` raises `RuntimeError: asyncio.run() cannot be called from a running event loop`.

**Fix.** §5.4(1) — `await dp_client.kv_batch_put(...)` directly from the collector's async context. The TQ adapter's `kv_batch_put` is already `async def` (`transfer_queue.py:438`), so this is the natural form. Falls back to `asyncio.new_event_loop() + run_until_complete` if a sync call site is ever needed.

#### Race 4 — Checkpoint cross-store coherence

**The window.** No atomic "snapshot trainer + ReplayBuffer + TQ partition." Buffer's `state_dict` may record keys whose TQ payload was written but not flushed (or vice versa) — depends on whether trainer-checkpoint or TQ-controller-state was saved first.

**On restore — two directions.** The §5.7 line about "one-time `kv_clear` of orphaned keys at startup" only handles **keys-in-TQ-but-dead-in-buffer**. The reverse (**keys-in-buffer-but-dead-in-TQ**) also needs handling: a `kv_batch_get` on a missing key would fail at first sample.

**Fix.** `ReplayBuffer.load_state_dict` does bidirectional reconciliation:

```python
def load_state_dict(self, state, dp_client=None):
    ...  # restore metadata as before
    if dp_client is None:
        return
    # 1. Drop metas whose TQ payload didn't survive.
    alive = []
    for meta in self.trajectories:
        try:
            dp_client.kv_batch_get(meta.keys[:1], meta.partition_id)  # probe
            alive.append(meta)
        except KeyNotFoundError:
            pass
    dropped = len(self.trajectories) - len(alive)
    if dropped:
        warnings.warn(f"ReplayBuffer: dropped {dropped} metas with missing TQ payload on restore")
    self.trajectories = alive
    # 2. Clear TQ keys not referenced by any meta (orphans).
    live_keys = {k for m in alive for k in m.keys}
    for partition_id in dp_client.list_partitions():
        all_keys = dp_client.list_keys(partition_id)   # if the adapter exposes one
        orphans = [k for k in all_keys if k not in live_keys]
        if orphans:
            dp_client.kv_clear(orphans, partition_id)
```

Step 2 needs a `list_keys` method on `DataPlaneClient` that doesn't currently exist — adding it is one ABC method + NoOp + TQ implementations. ~15 lines if the TQ adapter can reflect partition state; otherwise step 2 degrades to "warn-and-leak" with manual recovery via `kv_clear(partition=)`.

#### Race 5 — Eviction-vs-sample under the buffer lock *(safe, but easy to break)*

**Why it's safe today.** `ReplayBuffer._lock` serializes `push_with_wait_signal` and `sample`. Both run under the same lock; consumer never sees a half-popped state.

**The footgun.** If the proposed `kv_clear` calls were made *outside* the lock (to reduce critical-section duration), this sequence becomes possible:

1. `sample` pops meta `M`, releases lock, schedules `kv_clear(M.keys)` in the background.
2. New `push_with_wait_signal` lands at the same key (extremely unlikely with current namespace `f"v{wv}_p{pid}_g{i}"`, but not impossible if key generation ever loosens).
3. Background `kv_clear` destroys the *new* key.

**Fix.** §5.4(2) — keep `kv_clear` *inside* the lock, accept the sub-ms overhead. The Ray-actor `kv_clear` call is sub-millisecond in practice, dominated by RPC. If profiling later shows the lock holding up push throughput, the right fix is a separate per-meta deletion queue with a monotonic deletion epoch — not "release the lock." Don't release the lock.

---

## 6. What this document is *not*

- **Not a verdict on TQ.** TQ is the right abstraction for the sync path; the limitations in §2 are scope mismatches, not bugs.
- **Not exhaustive.** Second-order issues (metrics fan-out, observability middleware behavior under streaming workloads, cross-language client compatibility) are skipped — they're downstream of the R-1..R-7 decisions and orthogonal to the §5 recommendation.
- **Not a freeze.** §5 is a recommendation, not a contract. PR 1 is intentionally test-only so we can validate the round-trip assumption before committing to the full path.

---

## 7. References

- `nemo_rl/algorithms/grpo.py:2365-3197` — `async_grpo_train`, `AsyncTrajectoryCollector` lifecycle, refit pause/resume.
- `nemo_rl/algorithms/async_utils.py:36-235` — `ReplayBuffer.{add, sample}` (version filtering, age gating, eviction). **The control plane in §5.**
- `nemo_rl/algorithms/async_utils.py:260-426` — `AsyncTrajectoryCollector` pause/resume around refit, generation-limit backpressure. **The producer hook lands here in §5.4 (1).**
- `nemo_rl/algorithms/grpo_sync.py:605-704` — driver-side balanced packing + per-rank fan-out. **§5.4 (3a) extracts this into `nemo_rl/data_plane/preshard.py`; PR 0 in §5.8.**
- `nemo_rl/algorithms/grpo_sync.py:712-722` — `policy.train(dp_metas)` `@dp_dispatch` call site. **§5.4 (3c) reuses verbatim.**
- `nemo_rl/data_plane/sharding.py` — control-plane-only metadata sharder (sort-by-seqlen on `list[str] + list[int]`). **Distinct from the new `preshard.py` in §5.4 (3a):** `sharding.py` is for `@dp_dispatch` default fan-out from a single meta; `preshard.py` is for driver-side balanced packing of full `BatchedDataDict`s.
- `nemo_rl/data_plane/interfaces.py:94-229` — `DataPlaneClient` ABC. R-1..R-5 grounded here; §5 uses only the existing `kv_batch_put / kv_batch_get / kv_clear` surface.
- `nemo_rl/data_plane/dispatch.py:84-153` — `@dp_dispatch`, `list[KVBatchMeta]` handling. **§5.4 (3) reuses this without changes.**
- `nemo_rl/distributed/worker_groups.py:824-953` — `run_all_workers_sharded_data` positional dispatch by `worker_coords[axis]`.
- `verl/experimental/fully_async_policy/message_queue.py` — verl's sibling abstraction. Compared in §5.6.
- `verl/experimental/one_step_off_policy/` — verl's one-step-off path; no TQ references.
- Commit `a085559c` — TQ integration for sync GRPO. Sync-only by design.
