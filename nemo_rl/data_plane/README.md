# nemo_rl.data_plane

Stable boundary between NeMo-RL and the underlying data-plane backend
(currently `transfer_queue`; future: `nv-dataplane`). Every call site in
`nemo_rl/algorithms`, `nemo_rl/experience`, `nemo_rl/models` goes through
`DataPlaneClient`. No code imports `transfer_queue` directly outside the
adapter.

---

## Vocabulary

- **partition** — a named bucket of samples in TQ (e.g. `"train"`,
  `"val"`). One per training step.
- **sample** — one row in a partition, identified by a per-sample **key**
  (e.g. `"<uid>_g0"`). Lives in TQ until `kv_clear`.
- **field** — a named column (e.g. `input_ids`, `advantages`). Producers
  write fields; consumers select them on read. Each `(sample, field)`
  pair has an independent "produced?" bit on the TQ controller.
- **task** — a *consumer* name (e.g. `"prev_lp"`, `"train"`). Each task
  has its own consumption cursor, used by the task-mediated API only.
- **`KVBatchMeta`** — the receipt returned by writes. Carries the keys,
  partition id, sequence lengths, and the **fields written in this
  put**. NOT a partition-wide schema view — see the cheat-sheet below.

---

## Mental model

**TQ is a bus, not a database.** Bulk tensors (input_ids, logprobs,
masks) live in TQ for the duration of one GRPO step. The driver never
holds bulk between rollout and training — it only handles small
per-sample slices (rewards, advantages) and metadata (`KVBatchMeta`).
At the end of the step, `kv_clear` drops everything.

**Three layers, one-way dependency:**

```
algorithms/grpo_sync.py            ← orchestration (sync trainer)
        │
        ▼
data_plane/{column_io, preshard}   ← producer/consumer helpers
        │
        ▼
data_plane/interfaces.py           ← stable boundary (DataPlaneClient)
        │
        ▼
data_plane/adapters/               ← TransferQueue / NoOp / future nv-dataplane
```

---

## Scope of this README

This README documents the **sync** trainer (`grpo_train_sync`) — what
is actually implemented and tested. The data-plane interface also has
hooks for a future async trainer, but those methods are not yet wired
into any production codepath.

For the async design proposal, filtering / staleness strategies, and
open questions, see **[Async path (proposed)](#async-path-proposed)**
at the bottom.

---

## E2E flow — one sync GRPO step

Each step shows the user-facing call and (where useful) the file +
function that implements it. Sites in parentheses are internal — you
typically don't call them directly.

```
┌─ DRIVER · grpo_sync.py: grpo_train_sync ─────────────────────────────┐
│ ① policy.prepare_step(num_samples, group_size)                       │
│      → TQPolicy.prepare_step (tq_policy.py)                          │
│         → dp_client.register_partition("train", DP_TRAIN_FIELDS, …)  │
│                                                                      │
│ ② rollout_actor.rollout_to_tq.remote(repeated_batch, uids=…)         │
│   ← single Ray RPC into SyncRolloutActor (sync_rollout_actor.py).    │
│   Steps ③–⑥ all run inside the actor; driver only sees the result.   │
└────────────┬─────────────────────────────────────────────────────────┘
             │ Ray call
             ▼
┌─ ACTOR · SyncRolloutActor.rollout_to_tq (sync_rollout_actor.py) ─────┐
│ ③ self.policy_generation.clear_logger_metrics()                      │
│   rollout → run_multi_turn_rollout (or async / nemo_gym variant)     │
│ ④ flatten + mask + prompt extract                                    │
│     → batched_message_log_to_flat_message (data/llm_message_utils.py)│
│ ⑤ kv_first_write(bulk, keys=[uid_g0,…], dp_client=…)                 │
│     (column_io.py)                                                   │
│     → codec.pack_jagged_fields  (rectangular → jagged on the wire)   │
│     → dp_client.kv_batch_put                                         │
│ ⑥ self.policy_generation.finish_generation()                         │
│   self.policy_generation.get_logger_metrics()                        │
│   return (meta, slice, rollout_metrics, gen_metrics)                 │
└────────────┬─────────────────────────────────────────────────────────┘
             │ result tuple
             ▼
┌─ DRIVER · grpo_sync.py (logprob phase) ──────────────────────────────┐
│ ⑦ prev_lp = policy.get_logprobs_from_meta(meta)                      │
│   ref_lp  = policy.get_reference_policy_logprobs_from_meta(meta)     │
│   ↓ inside TQPolicy.get_logprobs_from_meta (tq_policy.py):           │
│     shard_meta_for_dp(meta, dp_world=N, sequence_packing_args=…)     │
│       (preshard.py — pure metadata, no I/O)                          │
│     fan-out: worker.get_logprobs_presharded.remote(shard) × N        │
└────────────┬─────────────────────────────────────────────────────────┘
             │ Ray fan-out, one call per DP rank
             ▼
┌─ WORKER · {Megatron,DTensor}PolicyWorker (× N DP ranks) ─────────────┐
│ ⑧ data = self._fetch(shard)                                          │
│     (worker_mixin.py · TQWorkerMixin._fetch)                         │
│     → dp_client.kv_batch_get(shard.keys, select_fields=…)            │
│     → codec.materialize  (jagged → padded; pad value from tokenizer) │
│ ⑨ forward → logprobs                                                 │
│ ⑩ leader-only:                                                       │
│   self._write_back_result_field(shard, result, "prev_logprobs", …)   │
│     (worker_mixin.py)                                                │
│     → codec.pack_per_token_field (rectangular → jagged)              │
│     → dp_client.kv_batch_put                                         │
│   (same pattern repeats for reference_policy_logprobs)               │
└────────────┬─────────────────────────────────────────────────────────┘
             │ aggregated results to driver
             ▼
┌─ DRIVER (small slice only — no bulk) · grpo_sync.py ─────────────────┐
│ ⑪ extras_bdd = read_columns(                                         │
│        policy.dp_client, meta,                                       │
│        select_fields=["token_logprobs", "rewards"])                  │
│      (column_io.py → kv_batch_get → codec.materialize)               │
│   compute advantages (tiny driver compute)                           │
│ ⑫ write_columns(policy.dp_client, meta,                              │
│        {"advantages": adv, "sample_mask": sample_mask})              │
│      (column_io.py → codec.pack_jagged_fields → kv_batch_put)        │
│                                                                      │
│   [optional] dynamic_sampling: meta.subset(survivors) +              │
│              policy.dp_client.kv_clear(dropped_keys, …)              │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             ▼
┌─ DRIVER → WORKER (train phase) · grpo_sync.py ───────────────────────┐
│ ⑬ train_results = policy.train_from_meta(meta, loss_fn=…)            │
│   ↓ inside TQPolicy.train_from_meta (tq_policy.py):                  │
│     shard_meta_for_dp again                                          │
│     fan-out: worker.train_presharded.remote(shard) × N               │
│       data = self._fetch(shard) → codec.materialize                  │
│       forward + loss → optimizer.step()                              │
│   (training is terminal — no write-back)                             │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             ▼
┌─ DRIVER (step-end) · grpo_sync.py ───────────────────────────────────┐
│ ⑭ policy.dp_client.kv_clear(keys=meta.keys, partition_id="train")    │
└──────────────────────────────────────────────────────────────────────┘
                                                  → next step → ①
```

### Where jagged pack/unpack happens

The on-wire layout is jagged (variable-length-aware via
`torch.nested`). The transitions are:

| Direction | Where | Helper |
|---|---|---|
| Rectangular → jagged (producer side) | every `kv_batch_put` | `codec.pack_jagged_fields` |
| Jagged → padded (consumer side) | every `kv_batch_get` reader | `codec.materialize` (called inside `read_columns` and `TQWorkerMixin._fetch`) |
| Per-token write-back (worker leader) | `_write_back_result_field` | `codec.pack_per_token_field` (tolerates SP padding) |

Jagged-on-wire saves wire bytes proportional to length skew; padding
tax is paid only when a consumer needs a rectangular tensor.

---

## Concrete: sequence-length flow (seqpack / dynbatch)

The trickiest piece is how `meta.sequence_lengths` flows from the
rollout actor through `shard_meta_for_dp` and ends up routing samples
to DP ranks. Worked example with 2 prompts × 2 generations = 4 samples:

**Step 1 — Rollout produces flat sequences.** The rollout actor calls
`batched_message_log_to_flat_message`, which concatenates ALL turns
(user + assistant) per sample. `input_lengths[i] = prompt_len_i + response_len_i`:

```
sample 0 (uid=u0, gen=0):  prompt=3 tok, response=4 tok → input_lengths=7
sample 1 (uid=u0, gen=1):  prompt=3 tok, response=2 tok → input_lengths=5
sample 2 (uid=u1, gen=0):  prompt=2 tok, response=6 tok → input_lengths=8
sample 3 (uid=u1, gen=1):  prompt=2 tok, response=3 tok → input_lengths=5
```

**Step 2 — `kv_first_write` writes the column and returns meta:**

```python
# inside SyncRolloutActor.rollout_to_tq
keys = [f"{uid}_g{i}" for uid in uids for i in range(n_gen)]
# keys = ["u0_g0", "u0_g1", "u1_g0", "u1_g1"]
meta = kv_first_write(bulk_batch, keys=keys, dp_client=self._dp_client, …)

# meta.keys             = ["u0_g0", "u0_g1", "u1_g0", "u1_g1"]
# meta.sequence_lengths = [   7,        5,       8,        5  ]
#                          ↑ row-aligned: meta.keys[i] ↔ meta.sequence_lengths[i]
```

**Step 3 — `shard_meta_for_dp` shards by length-balanced packing
(driver-side, no TQ I/O):**

```python
# With 2 DP ranks + seqpack:
shards, _ = shard_meta_for_dp(meta, dp_world=2,
                              sequence_packing_args={…})

# rank 0: idx=[2, 1]  (lens 8+5=13, packed together)
#   shard.keys             = ["u1_g0", "u0_g1"]
#   shard.sequence_lengths = [8, 5]
# rank 1: idx=[0, 3]  (lens 7+5=12)
#   shard.keys             = ["u0_g0", "u1_g1"]
#   shard.sequence_lengths = [7, 5]
```

**Step 4 — Each worker fetches its own slice from TQ:**

```python
# inside MegatronPolicyWorker.train_presharded (via TQWorkerMixin._fetch)
data = self._fetch(shard)
# → kv_batch_get(keys=shard.keys, partition_id, select_fields=DP_TRAIN_FIELDS)
```

**Why no mismatch is possible:** `shard_meta_for_dp` slices both
`meta.keys` and `meta.sequence_lengths` with the *same* `idx_list`.
They're coupled scalars indexed together. A row index `j` in any
shard always points to the same original sample in TQ.

**Subtle gotcha — `make_sequence_length_divisible_by`:** `input_ids`
gets padded to a multiple of TP×CP for Megatron, but `input_lengths`
reflects the **actual content length** before that alignment. Seqpack
balances on actual lengths; padding is reapplied per shard inside the
worker.

```
input_ids:     [1,2,3,4,5,6,7, 0,0]  ← padded to 9 (divisible by 4)
input_lengths: 7                      ← actual content length
meta.sequence_lengths: 7              ← what seqpack uses ✓
```

---

## API surface — DataPlaneClient

`nemo_rl/data_plane/interfaces.py`. Eight methods grouped by intent.

### Lifecycle
- `register_partition(partition_id, fields, num_samples, consumer_tasks, …)`
- `close()`

### Direct-by-key (used by sync trainer)
- `kv_batch_put(keys, partition_id, fields, tags?) → KVBatchMeta`
- `kv_batch_get(keys, partition_id, select_fields) → TensorDict`
- `kv_clear(keys, partition_id)`

### Task-mediated (TODO — reserved for the future async trainer)
- `claim_meta(partition_id, task_name, required_fields, batch_size) → KVBatchMeta`
- `get_data(meta, select_fields) → TensorDict`
- `check_consumption_status(partition_id, task_names) → bool`

### `KVBatchMeta` cheat-sheet

`KVBatchMeta` is the receipt for a put — **not a partition-wide schema
view**. A common confusion: `meta.fields` only contains the fields
written in *this specific put*, not every field that has ever been
written to the partition.

| Attribute | Meaning | Typical use |
|---|---|---|
| `partition_id` | Which TQ partition these keys live in | Pass back to `kv_batch_get(... partition_id=...)` |
| `keys` | Per-sample row identifiers | Pass to `kv_batch_get`; permuted by `shard_meta_for_dp` |
| `fields` | Fields written **by the put that minted this meta** | Used to derive `select_fields` when the caller wants "everything available at first put"; ignored if the caller already knows what to fetch |
| `sequence_lengths` | Per-row valid lengths (NOT padded) | Used by `shard_meta_for_dp` for length-balanced sharding |
| `extra_info` | Free-form bag for `rollout_metrics`, `pad_to_multiple`, packing metadata | Read by consumers that need it |
| `task_name` | Optional consumer tag | Carried through; not used by direct-by-key reads |

The same `meta` can be read N times with different `select_fields` —
that's how the logprob/ref-logprob/train phases each pull a different
column subset out of the same first-write.

### Hard rules

- **No Python leaves on the bus.** `kv_batch_put(fields=...)` must be a
  `TensorDict` of tensors (or `np.ndarray(dtype=object)` for non-tensor
  columns, which the codec packs). Primitives → `tags=`. Arbitrary
  Python objects → Ray object store.
- **`select_fields` is required on `kv_batch_get`.** No fallback
  to "fetch all fields" — that's the most expensive shape the wire can
  take and the most common foot-gun. Callers must name what they read.
  `get_data` is consistent: requires either `select_fields` or
  `meta.fields`; raises on both missing.

---

## Helpers above the client (`nemo_rl/data_plane/`)

| Helper | What it does |
|---|---|
| `column_io.kv_first_write(batch, *, keys, dp_client, …) → KVBatchMeta` | One flat `kv_batch_put` of every tensor field in the rollout output. Caller mints `keys`. Used by `SyncRolloutActor`. |
| `column_io.read_columns(client, meta, select_fields) → BatchedDataDict` | `kv_batch_get` + `materialize` (decode jagged + object-array fields). |
| `column_io.write_columns(client, meta, fields)` | Typed `kv_batch_put` for driver/worker deltas under existing meta. |
| `preshard.shard_meta_for_dp(meta, dp_world, …) → list[KVBatchMeta]` | Pure metadata split. Length-balanced when `sequence_packing_args` / `dynamic_batching_args` is passed. |
| `KVBatchMeta.subset(idxs)` / `.slice(start, stop)` / `.concat(other)` | Pure metadata transforms used by dynamic sampling. |
| `codec.pack_jagged_fields(fields, *, lengths) → TensorDict` | Single source of truth for jagged-pack + `np.ndarray(dtype=object)` passthrough — called by both `kv_first_write` and `write_columns`. |

---

## Per-sample key invariant

Mint **once** at rollout, reuse forever:

```
uid   = "step17_prompt_42"      # opaque, from driver dataset iter
key_i = f"{uid}_g{i}"           # i ∈ [0, n_gen)
```

Every `kv_batch_put` / `kv_batch_get` for that sample uses the same key.
Worker write-backs append columns under the same keys; nothing remints.
Callers (e.g. `SyncRolloutActor`) build the key list inline before
calling `kv_first_write(batch, keys=…)`.

---

## Concrete examples

**Rollout produces (one Ray RPC, bundles 6 steps — see `rollout_to_tq` docstring):**

```python
# In grpo_sync.py
uids = [str(uuid.uuid4()) for _ in range(n_prompts)]
(meta, slice_extras, rollout_metrics, gen_metrics) = ray.get(
    rollout_actor.rollout_to_tq.remote(
        repeated_batch,
        uids=uids,
        partition_id=policy.tq_partition_id,
        first_iter=(dynamic_sampling_num_gen_batches == 1),
    )
)
# meta.keys             = ["<uid>_g0", "<uid>_g1", …]
# meta.sequence_lengths = [<actual content lengths>]
# meta.fields           = ["input_ids", "input_lengths", "generation_logprobs",
#                          "token_mask", "sample_mask", …multimodal extras…]
```

**Driver appends a column (small delta, no bulk crosses):**

```python
adv_inputs = read_columns(policy.dp_client, meta,
                          select_fields=["token_logprobs", "rewards"])
advantages = compute_advantages(adv_inputs)
write_columns(policy.dp_client, meta, {"advantages": advantages})
```

**Worker fan-out (driver — user-facing call):**

```python
# In grpo_sync.py the driver calls a single TQPolicy method;
# shard_meta_for_dp + Ray fan-out happens inside it.
train_results = policy.train_from_meta(meta, loss_fn=loss_fn, timer=timer)
```

Internally (`tq_policy.py: TQPolicy.train_from_meta`):

```python
dp_metas, _ = shard_meta_for_dp(
    meta, dp_world=N, batch_size=GBS,
    sequence_packing_args=cfg.seqpack,
)
results = ray.get([
    worker[i].train_presharded.remote(dp_metas[i], loss_fn=loss_fn)
    for i in range(N)
])
return _aggregate_train_results(results)
```

**Worker fetch + leader write-back (inside `train_presharded` /
`get_logprobs_presharded`):**

```python
# {Megatron,DTensor}PolicyWorker mixes in TQWorkerMixin.
# Inside get_logprobs_presharded(meta):
data = self._fetch(meta)                   # kv_batch_get → materialize
logprobs = self._run_one_logprob_step(data)
# Leader-only write-back so jagged row-lengths match the initial put:
self._write_back_result_field(
    meta, logprobs,
    result_key="logprobs",
    tq_field="prev_logprobs",
)
```

**Step-end teardown:**

```python
client.kv_clear(keys=meta.keys, partition_id=meta.partition_id)
```

---

## Call counts per sync step

Steady state on the validation run (32 samples, 8 GPUs, no PP/TP):

| TQ call | Site | Count / step | Payload |
|---|---|---:|---|
| `register_partition` | driver | 1 | metadata only |
| `kv_batch_put` (rollout) | SyncRolloutActor | 1 | full bulk (~600 KB; GBs at scale) |
| `shard_meta_for_dp` | driver | 3 | no I/O |
| `kv_batch_get` (lp inputs) | workers | 8 (per DP) | input slice |
| `kv_batch_put` (lp out) | workers (leader) | 1 | prev_logprobs delta |
| `kv_batch_get` (ref input) | workers | 8 | input slice |
| `kv_batch_put` (ref out) | workers (leader) | 1 | ref_logprobs delta |
| `kv_batch_get` (adv slice) | driver | 1 | small (rewards + token_lp) |
| `kv_batch_put` (advantages) | driver | 1 | small delta |
| `kv_batch_get` (train) | workers | 8 | full slice |
| `kv_batch_get` (log_data) | driver | 1 | input_ids only |
| `kv_clear` | driver | 1 | drop |

Total: ~32 TQ RPCs / step. 24 of those are per-DP fetch fan-out
(3 phases × 8 ranks).

---

## How callers reach the client

Training-loop code (`grpo_sync.py`) doesn't call `DataPlaneClient`
methods directly for lifecycle. Instead it goes through `TQPolicy`,
which is a `Policy` subclass that owns the client and exposes
training-loop-friendly methods:

| Training-loop method | What it calls underneath |
|---|---|
| `policy.prepare_step(num_samples, group_size)` | `client.register_partition("train", DP_TRAIN_FIELDS, num_samples, ["prev_lp", "ref_lp", "train"], …)` |
| `policy.train_from_meta(meta)` | per-rank `_fetch` → `client.kv_batch_get` |
| `policy.get_logprobs_from_meta(meta)` | per-rank `_fetch` + leader `_write_back` |
| `policy.dp_client` | direct handle when the driver needs `read_columns` / `write_columns` / `kv_clear` |

So when terryk asked "does `register_partition` need a more
training-loop-y name?" — the answer is that `prepare_step` already is
that name; `register_partition` is one level lower (TQ's own term for
declaring a partition's schema + consumer set).

---

## Configuration

The data plane is configured via a `data_plane:` block in the master
YAML (`examples/configs/...`). Defaults should live in the YAML — the
exemplar YAML is the single source of truth.

Expected shape:

```yaml
data_plane:
  enabled: true                  # required; false skips the TQ trainer entirely
  impl: transfer_queue           # only one impl today
  backend: simple                # "simple" or "mooncake_cpu"

  # simple-backend tuning:
  storage_capacity: 1000000      # max samples held across partitions
  num_storage_units: 2           # parallel storage actors

  # mooncake_cpu-backend tuning:
  global_segment_size: 4294967296   # bytes per storage segment (default 4 GiB)
  local_buffer_size: 1073741824     # bytes per local buffer    (default 1 GiB)

  # poll cadence (both backends):
  get_meta_poll_interval_s: 0.01    # claim_meta polling-mode tick (async path)
```

Backend choice:
- **`simple`** — ZMQ-backed; lowest setup overhead. Default for tests
  and small runs.
- **`mooncake_cpu`** — Mooncake transfer engine; higher throughput at
  scale. Required for multi-node clusters with large bulk volume.

Capacity rule of thumb (any backend):

```
storage_capacity ≥ 2 × num_prompts × n_gens × max_seq_len
                   × bytes_per_token × num_active_fields
```

The `2 ×` headroom covers dynamic sampling overflow and one step of
pipelining between rollout and training.

---

## Install

`tensordict` and `TransferQueue` are base nemo-rl dependencies — `uv sync`
is enough. Worker venvs built per-backend (FSDP2, DTensor, mcore,
automodel) pick them up automatically; no `[data-plane]` extra.

---

## When `data_plane.enabled=False`

`build_data_plane_client` raises — there is no NoOp prod fallback.
For the no-data-plane path use the legacy
`nemo_rl.algorithms.grpo.grpo_train`; the sync trainer
`grpo_train_sync` requires `enabled=True` and a `TQPolicy`.

`NoOpDataPlaneClient` (`adapters/noop.py`) exists only as a unit-test
fixture for the ABC contract tests.

---

## Performance characterization

End-to-end parity vs the legacy driver-bulk path on the toy validation
run:

- Steps 1–7 are bit-exact (loss + reward); divergence afterward is the
  expected stochastic drift from accumulated policy updates.
- Steady-state step time: **+0.21 s** (1-hop 7.86 s vs legacy 7.65 s,
  ~3 %).

Per-phase breakdown (steady state, steps 2–19):

| Phase | v4 (1-hop) | Legacy | Δ |
|---|---:|---:|---:|
| Total step time | 7.606 s | 7.393 s | **+0.213 s** |
| policy_training | 0.596 s | 0.567 s | +0.028 s |
| generation | 1.502 s | 1.528 s | −0.027 s |
| policy_and_ref_logprob | 1.588 s | 1.448 s | **+0.141 s** |
| residual (driver bookkeeping) | 3.920 s | 3.850 s | +0.070 s |

**The +0.21 s overhead is entirely TQ RPC roundtrip cost in the
logprob phase** (two worker calls × one fetch + one write each).
Generation and training are unchanged.

### Crossover scale (where TQ wins)

TQ overhead is mostly latency-bound (~constant per step), while legacy
driver fan-out is bandwidth-bound (scales with batch tensor volume ×
DP fan-out).

| Scale | Batch / step | DP ranks | Legacy cost | Winner |
|---|---:|---:|---:|---|
| Toy (1B, 512 tok, BS 32) | 0.6 MB | 8 | ~50 ms | **legacy +0.21 s** |
| Small prod (8B, 1k tok, BS 256) | ~10 MB | 8 | ~300 ms | **roughly tied** |
| Mid prod (70B, 4k tok, BS 1024) | ~250 MB | 32 | ~5–10 s | **TQ wins** |
| Long-context (8k–32k seq, 16 gens) | 1–5 GB | 64+ | tens of s | **TQ wins** |

Crossover: **~10 MB / step / DP-rank** of effective batch volume. Long
sequences, more generations per prompt, and more DP ranks all push
toward TQ.

### Cheapest optimizations (deferred)

1. Fuse `get_logprobs` + `get_reference_policy_logprobs` into one
   worker call — saves ~70 ms (one TQ input-fetch).
2. Overlap TQ write-back with next-phase fetch — saves another
   ~30–50 ms.

Both are clean refactors inside `tq_policy.py` / `worker_mixin.py`;
not on the critical path.

---

## Where to look

| Concern | File |
|---|---|
| Stable boundary (ABC) | `nemo_rl/data_plane/interfaces.py` |
| Adapter (TransferQueue impl) | `nemo_rl/data_plane/adapters/transfer_queue.py` |
| Adapter (NoOp, test only) | `nemo_rl/data_plane/adapters/noop.py` |
| Codec (jagged pack / unpack) | `nemo_rl/data_plane/codec.py` |
| Column-level helpers | `nemo_rl/data_plane/column_io.py` (`read_columns`, `write_columns`, `kv_first_write`) |
| DP-rank meta sharding | `nemo_rl/data_plane/preshard.py` |
| Worker fetch + leader write-back | `nemo_rl/data_plane/worker_mixin.py` |
| Schema constants | `nemo_rl/data_plane/schema.py` |
| Rollout actor (first put) | `nemo_rl/experience/sync_rollout_actor.py` |
| TQ-mediated Policy subclass | `nemo_rl/models/policy/tq_policy.py` |
| End-to-end orchestration | `nemo_rl/algorithms/grpo_sync.py` |
| Unit tests | `tests/data_plane/unit/` |
| Functional tests (real backends) | `tests/data_plane/functional/` |

---

## Operational assumptions

- One Ray cluster per experiment. The TQ controller is a globally
  named Ray actor — running two trainers in the same cluster collides.
- Storage capacity sizing — see the formula in the
  "Configuration" section above.

---

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

### Option 2 — In TQ as tags (needs tag propagation in `KVBatchMeta`)

The producer stamps primitive metadata (`weight_version`, `std`,
`total_reward`, `produced_at`) as **tags** on each key. Tags live on
the TQ controller alongside production status; reading them needs no
data RPC. The consumer inspects them in-memory:

```python
# Producer:
tags = [{"weight_version": v, "std": s.item(), "produced_at": t}
        for s, t in zip(stds, timestamps)]
dp_client.kv_batch_put(keys=keys, partition_id="train", fields=..., tags=tags)

# Consumer (post-claim, no data fetch):
meta = dp_client.claim_meta(task_name="train", required_fields=[...], batch_size=K)
survivors = [i for i, tag in enumerate(meta.tags)
             if current_version - tag["weight_version"] <= MAX_AGE]
meta = meta.subset(survivors)
```

- ✅ Zero data fetch — tags travel with the meta.
- ✅ Works for *time-varying* filters (compare tag vs. current state).
- ✗ **Requires our `KVBatchMeta` to expose `tags`** (todo — see
  feature proposal below).

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
| Tag-stamp `weight_version`; consumer post-filters | TQ tags | zero | nemo-rl `KVBatchMeta.tags` propagation |
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

1. **Propagate `tags` through nemo-rl `KVBatchMeta`** (small change,
   high leverage). TQ's `KVBatchMeta` already carries `tags:
   list[dict]`; our `interfaces.py:KVBatchMeta` only lifts
   `input_lengths`. Add `tags: list[dict] | None` and have the
   adapter pass them through. Unlocks Option 2 entirely.
2. **Server-side tag filtering in `claim_meta`**: e.g.
   `claim_meta(..., tag_filter=lambda t: t["weight_version"] >= cutoff)`.
   Today the consumer must claim everything ready and then filter
   in-memory; a tag predicate would push this server-side. Requires
   upstream TQ change.
3. **Versioned-partition helpers**: convenience methods
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
- **Polling cadence**: `get_meta_poll_interval_s` controls how often
  `claim_meta` retries. Too aggressive = wasted CPU; too lazy =
  trainer-rollout coupling.
- **Backpressure**: if rollout outpaces training, when does the
  producer start blocking on TQ capacity?
  (`storage_capacity` × `num_storage_units` is the hard cap.)
- **Cleanup cadence**: stale-key batch size for `kv_clear` —
  per-step, per-N-steps, or size-threshold?
