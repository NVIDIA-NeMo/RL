<!-- Moved out of nemo_rl/data_plane/README.md to keep the README focused
     on the implemented sync trainer. This is a design-proposal doc — none
     of the patterns below are wired into production. -->

# Async path (proposed)

The data-plane interface covers both sync and async, but the **sync
trainer (`grpo_train_sync`) uses only half of it**. The other half is
reserved for the async trainer (not yet landed). Everything below
documents the design proposal and open questions for that path. None
of it is wired into production today.

## Sync vs Async at a glance

| Concern | Sync (implemented) | Async (TODO) |
|---|---|---|
| **Who knows the keys?** | Driver — `SyncRolloutActor` returns `KVBatchMeta` with `meta.keys` populated | TQ — trainer doesn't know which samples are ready until it asks |
| **Data fetch API** | `kv_batch_get(meta.keys, ..., select_fields=[...])` — direct by key | `claim_meta(...)` → `get_data(meta)` — discover-then-fetch |
| **Consumer cursor?** | Not needed — driver controls who reads what | `claim_meta` advances a per-task cursor; `check_consumption_status` confirms drain |
| **Step boundary** | `kv_clear(meta.keys)` at end of step | Same |

In sync mode the driver always knows exactly which keys are in TQ
because it triggered every write. The task-mediated API
(`claim_meta` / `get_data` / `check_consumption_status`) is implemented
and tested but **not yet wired into any production codepath** — it's
the future async-trainer's entry point.

### Why two API surfaces?

The deciding question is **"does the caller already know the keys?"**

- **Yes** → use direct-by-key (`kv_batch_get`). The sync trainer is
  always in this case: the rollout actor's return value carries
  `meta.keys`. Cheapest path, no coordination.
- **No** → use task-mediated (`claim_meta` → `get_data`). The async
  trainer is in this case: rollouts and training run concurrently, so
  the trainer must ask TQ "what's ready for me to consume?" The
  consumer cursor (`task_name`) prevents the same sample from being
  claimed twice.

verl follows the same split — its `ReplayBuffer.sample()` returns a
`KVBatchMeta` from keys it tracks via `global_steps` tags, then fetches
via `kv_batch_get`. No `claim_meta` is used in verl's sync trainer
either.

## Proposed E2E flow — async GRPO

In the async path, rollout and training run concurrently on separate
Ray actors. The trainer doesn't know which samples are ready ahead of
time, so it uses the task-mediated half of the API
(`claim_meta` / `get_data` / `check_consumption_status`) instead of
direct-by-key reads.

```
[PRODUCER — continuous, never waits for trainer]
┌─ AsyncTrajectoryCollector (Ray @remote)                              ┐
│   async_utils/trajectory_collector.py                                │
│   Loop:                                                              │
│     rollout → flatten → mask → prompt extract                        │
│     kv_first_write(bulk, keys=[v<ver>_p<i>_g<j>, …])                 │
│       → dp_client.kv_batch_put                                       │
│   Pushes only KVBatchMeta onto an in-memory replay buffer            │
│   (bulk lives in TQ, never on the driver).                           │
└──────────────────────────────────────────────────────────────────────┘

[CONSUMER — async trainer]
┌─ DRIVER · async grpo trainer (proposed)                              ┐
│ ① policy.prepare_step(num_samples, group_size)                       │
│      → register_partition("train", DP_TRAIN_FIELDS,                  │
│                            consumer_tasks=["prev_lp","ref_lp","train"])│
│                                                                      │
│ ② meta = dp_client.claim_meta(                                       │
│       partition_id="train",                                          │
│       task_name="train",                                             │
│       required_fields=DP_TRAIN_FIELDS,                               │
│       batch_size=GBS,                                                │
│   )                                                                  │
│   ↑ BLOCKS until GBS samples have all required fields produced.      │
│     This is the *only* point where the per-task cursor advances —    │
│     TQ's underlying ``get_meta(mode="fetch")`` marks those samples   │
│     as consumed by ``task_name``, so they won't be returned again    │
│     to the same task.                                                │
│                                                                      │
│ ③ data = dp_client.get_data(meta, select_fields=…)                   │
│   ↑ Pure key-list fetch (no cursor advancement here — that already   │
│     happened at claim_meta). Or call ``policy.train_from_meta(meta)``│
│     and let the workers fetch per-rank.                              │
│                                                                      │
│ ④ training: same shard_meta_for_dp + fan-out as sync.                │
│   Workers fetch per-rank via dp_client.kv_batch_get and materialize. │
│                                                                      │
│ ⑤ Sync barrier before clearing:                                      │
│   dp_client.check_consumption_status(                                │
│       "train", task_names=["prev_lp","ref_lp","train"])              │
│   ↑ True iff every consumer task has drained — safe to drop the data.│
│                                                                      │
│ ⑥ dp_client.kv_clear(keys=meta.keys, partition_id="train")           │
└──────────────────────────────────────────────────────────────────────┘
```

**Why these methods are needed in async (but not sync):**

| Method | Async role | Sync equivalent |
|---|---|---|
| `claim_meta` | discover + claim ready samples; per-task cursor prevents double-claim | not needed — actor returns `meta.keys` directly |
| `get_data` | resolve meta → TensorDict (pure key-list fetch — no cursor advancement) | not needed — workers call `kv_batch_get` directly |
| `check_consumption_status` | safe-clear barrier when multiple consumers must drain before kv_clear | not needed — single-thread Python ordering guarantees drain order |

## Filtering without fetching bulk

**Design constraint:** rollout writes samples continuously; many will
be discarded (off-policy beyond tolerance, DAPO `std == 0`,
format-check failures, length thresholds, …). The filter decision
**must not require reading bulk tensor data**.

The filter state has to live somewhere small. Three alternative
options — pick one based on what TQ/dataplane features are available
and how decoupled you want the cleanup to be.

### Option 1 — In TQ as a gating field (works today)

The producer (or an intermediate stage) writes a small marker column
ONLY for samples that should be visible to downstream tasks. The
consumer `claim_meta(required_fields=["marker"])` only matches
samples where that field exists.

```python
# Producer writes a small bool per survivor:
dp_client.kv_batch_put(
    keys=survivor_keys, partition_id="train",
    fields=TensorDict({"_train_ready": torch.ones(K)}, batch_size=[K]),
)
# Trainer never sees the non-survivors:
meta = dp_client.claim_meta(task_name="train",
                            required_fields=["input_ids", "_train_ready"],
                            batch_size=GBS)
```

- ✅ Server-side enforcement; consumer needs no special exclusion logic.
- ✅ Works with TQ as-is.
- ✗ Decision must be made at write time; no good story for filters
  that become true *after* the write (e.g. weight-version drift).

### Option 2 — In TQ as tags (recommended; works today)

The producer stamps primitive metadata (`weight_version`, `std`,
`total_reward`, `produced_at`) as **tags** on each key. Tags live on
the TQ controller alongside production status; reading them needs no
data RPC. The consumer inspects them in-memory:

```python
# Producer:
tags = [{"weight_version": v, "std": s.item(), "produced_at": t}
        for s, t in zip(stds, timestamps)]
dp_client.put_samples(sample_ids=keys, partition_id="train", fields=..., tags=tags)

# Consumer (post-claim, no data fetch):
meta = dp_client.claim_meta(task_name="train", required_fields=[...], batch_size=K)
survivors = [i for i, tag in enumerate(meta.tags)
             if current_version - tag["weight_version"] <= MAX_AGE]
meta = meta.subset(survivors)
```

- ✅ Zero data fetch — tags travel with the meta.
- ✅ Works for *time-varying* filters (compare tag vs. current state).
- ✅ Implemented: `KVBatchMeta.tags` is populated by the TQ adapter
  from TQ's `custom_meta`; canonical example is
  `tests/unit/data_plane/test_async_rl_filter_flow.py`.

### Option 3 — Outside TQ entirely, in `AsyncTrajectoryCollector`

The collector keeps a small driver-side ledger: `dict[key,
SampleMetadata]` tracking `weight_version`, `produced_at`, `status`,
etc. Sampling for training first consults the ledger, applies the
filter, and only then issues direct-by-key reads against TQ. TQ never
sees the filter — it's just a KV store.

```python
# inside AsyncTrajectoryCollector (Ray @remote)
def sample(self, batch_size: int, max_age: int) -> KVBatchMeta:
    current_v = self._current_weight_version
    survivor_keys = [
        k for k, m in self._ledger.items()
        if (current_v - m.weight_version) <= max_age and m.status == "ready"
    ][:batch_size]
    return KVBatchMeta(
        partition_id="train", task_name=None,
        keys=survivor_keys,
        fields=DP_TRAIN_FIELDS,
        sequence_lengths=[self._ledger[k].seq_len for k in survivor_keys],
    )
```

- ✅ Zero TQ-side changes.
- ✅ Maximum flexibility — any predicate, any state.
- ✗ Two sources of truth (collector ledger vs. TQ controller). On a
  collector crash the ledger evaporates; needs reconciliation (e.g.
  walk TQ partition on restart and reseed).

## Timestamping / staleness specifically

A common case worth singling out: rollouts produced under weight
version `v` may be too stale by version `v + N`. Four ways to handle
it, no bulk fetch needed in any of them:

| Approach | Where state lives | Filter cost | Needs new feature? |
|---|---|---|---|
| Tag-stamp `weight_version`; consumer post-filters | TQ tags | zero | none (works today; see Option 2) |
| Small `weight_version` field; `get_data(select_fields=["weight_version"])` | TQ field | one tiny RPC per claim | none |
| **Versioned partitions** (`train_v17`, `train_v18`, …) | TQ partition naming | zero | partition lifecycle helpers |
| `AsyncTrajectoryCollector` ledger with TTL | driver-side dict | zero | new collector method |

**Versioned partitions** is interesting because it makes wholesale
staleness handling free: producers write into `train_v<current>`,
trainer claims from `[train_v<current-N> .. train_v<current>]`, and
`kv_clear(partition_id="train_v<old>")` retires an entire generation
of samples in one call.

## Mark-as-stale, defer the kv_clear

Filtered keys' bulk still sits in TQ. Two cleanup patterns:

**Pattern A — driver-side stale set + batched clear (recommended for
single-collector deployments):**

```python
stale_keys: set[str] = set()
stale_keys.update(filter_meta.keys[i] for i in non_survivors)

# Periodically (every K steps or size threshold):
if len(stale_keys) > 4096:
    dp_client.kv_clear(keys=list(stale_keys), partition_id="train")
    stale_keys.clear()
```

No TQ-side coordination. Bulk lingers briefly, bounded by the threshold.

**Pattern B — TQ-side stale-marker field + cleanup task (decoupled):**

`claim_meta` filters on field production, not tag values — so marking
via tags alone doesn't gate cleanup. Write a dedicated marker field:

```python
dp_client.kv_batch_put(
    keys=stale_keys, partition_id="train",
    fields=TensorDict({"_stale": torch.ones(len(stale_keys), dtype=torch.bool)},
                       batch_size=[len(stale_keys)]),
)
# A separate cleanup task:
cleanup_meta = dp_client.claim_meta(
    partition_id="train", task_name="cleanup",
    required_fields=["_stale"], batch_size=K,
)
dp_client.kv_clear(keys=cleanup_meta.keys, partition_id="train")
```

Pattern A is simpler. Pattern B decouples the cleanup cadence from
the filter site (useful if multiple producers can mark stale).

## Proposed enhancements

**TQ / data-plane side (in priority order):**

1. **Server-side tag filtering in `claim_meta`**: e.g.
   `claim_meta(..., tag_filter=lambda t: t["weight_version"] >= cutoff)`.
   Today the consumer must claim everything ready and then filter
   in-memory; a tag predicate would push this server-side. Requires
   upstream TQ change.
2. **Versioned-partition helpers**: convenience methods
   `register_versioned_partition(prefix, version)` + `claim_meta`
   variant that takes a partition range. Cheap because TQ already
   supports per-partition lifecycle.

**`AsyncTrajectoryCollector` side (no TQ changes needed):**

1. **Per-key ledger**: `dict[str, SampleMetadata]` on the collector
   actor, populated at write time with `weight_version`,
   `produced_at`, `seq_len`, `status`.
2. **`sample(batch_size, predicate)`**: returns a `KVBatchMeta` of
   survivors after applying `predicate` to ledger entries. Trainer
   never touches TQ for filtering.
3. **Mark-stale set + periodic batched `kv_clear`**: collector also
   owns a background coroutine that drains stale keys on a cadence
   (every K steps or by buffer pressure).
4. **Backpressure hook**: when ledger size approaches
   `storage_capacity`, evict by oldest weight version. Decouples
   producer from training rate.

The collector-side path is the cheapest to land (zero TQ changes) and
gives the most flexibility; the TQ-side path scales better when
filtering needs to live close to the data (e.g. multiple trainers
filtering differently on the same partition).

## Open questions

- **`required_fields` granularity**: gate trainer on the full
  `DP_TRAIN_FIELDS` set, or pipeline — start training as soon as
  `input_ids` + `generation_logprobs` are ready and gate on
  `advantages` per microbatch?
- **Stale-data policy**: if the producer is multiple weight-versions
  ahead of the trainer, drop those samples or use them with
  importance-sampling correction?
- **Polling cadence**: `claim_meta_poll_interval_s` controls how often
  `claim_meta` retries. Too aggressive = wasted CPU; too lazy =
  trainer-rollout coupling.
- **Backpressure**: if rollout outpaces training, when does the
  producer start blocking on TQ capacity?
  (`storage_capacity` × `num_storage_units` is the hard cap.)
- **Cleanup cadence**: stale-key batch size for `kv_clear` —
  per-step, per-N-steps, or size-threshold?
