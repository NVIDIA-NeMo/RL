# nemo_rl.data_plane

Stable boundary between NeMo-RL and the underlying data-plane backend
(currently `transfer_queue`; future: `nv-dataplane`). Every call site in
`nemo_rl/algorithms`, `nemo_rl/experience`, `nemo_rl/models` goes through
`DataPlaneClient`. No code imports `transfer_queue` directly outside the
adapter.

---

## Vocabulary

- **partition** — a named data-flow scope in TQ (e.g. `"train"`,
  `"val"`). Each partition owns its own field schema, consumer task
  set, and per-sample production-status matrix. Sync GRPO uses one
  stable partition (`"train"`) that is cleared and reused across
  steps — different partitions are for different data flows
  (training vs validation vs replay buffer), not for different steps.
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

**TQ is a distributed storage and transfer engine.** It holds bulk
tensors (input_ids, logprobs, masks) addressed by per-sample keys,
moves them between producer and consumer Ray actors over the wire,
and tracks per-`(sample, field)` production status so consumers know
when their inputs are ready. Storage is transient: data lives in TQ
for the duration of one GRPO step and `kv_clear` drops it at step
end. The driver never holds bulk between rollout and training — only
small per-sample slices (rewards, advantages) and metadata
(`KVBatchMeta`) cross the driver.

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

## Legacy vs TQ-mediated — same algorithm, encapsulated I/O

The TQ-mediated trainer (`grpo_train_sync`) is meant to read like the
legacy in-memory trainer (`grpo_train`). The algorithm is identical;
only the data-fetch and lifecycle calls move behind `TQPolicy` / `meta`
methods. Per-step side-by-side:

| Step | Legacy (`grpo.py: grpo_train`) | TQ-mediated (`grpo_sync.py: grpo_train_sync`) |
|---|---|---|
| Step start | (implicit) | `policy.prepare_step(N, group_size)` |
| Rollout | `run_multi_turn_rollout(...)` driver-side | `ray.get(rollout_actor.rollout_to_tq.remote(...))` — bulk written to TQ inside the actor |
| Carry per-row data | `repeated_batch[k]` | `driver_carry[k]` (returned alongside `meta`) |
| Reward scale / shape / baseline / std | unchanged | unchanged |
| Mirror std for filter | `std` tensor in scope | `meta.stamp_tags({"std": …, "baseline": …})` |
| Dynamic sampling filter | `repeated_batch.select_indices(keep_idx)` | `meta.subset(keep_idx)` + `driver_carry.select_indices(keep_idx)` (inside `_apply_dynamic_sampling`, which also `kv_clear`s dropped uids) |
| Overlong filter / mask | unchanged | unchanged |
| Read columns for masking | `repeated_batch["generation_logprobs"]`, `repeated_batch["token_mask"]` | `policy.read_from_dataplane(meta, select_fields=["generation_logprobs", "token_mask"])` |
| Compute advantage | unchanged | unchanged |
| Write back advantage | mutate `repeated_batch["advantages"]` | `policy.write_to_dataplane(meta, {"advantages": …})` |
| Train | `policy.train(repeated_batch, loss_fn)` | `policy.train_from_meta(meta, loss_fn)` |
| Step end | (Python GC) | `policy.finish_step(meta)` |

**The shape of the algorithm is unchanged.** Each TQ-mediated step has
a one-to-one counterpart in legacy; the only difference is where data
lives (Python memory vs TQ) and which method moves it.

Per-stage audit grade after the encapsulation refactor: **A**. The
trainer body never references `policy.dp_client` directly — only meta
and policy methods. `_apply_dynamic_sampling` still takes a raw
`dp_client` argument by design so unit tests can inject
`NoOpDataPlaneClient`.

---

## E2E flow — one sync GRPO step

```
┌─ DRIVER · grpo_train_sync ───────────────────────────────────────────┐
│ ① policy.prepare_step(num_samples, group_size)                       │
│      → register "train" partition with DP_TRAIN_FIELDS schema        │
│ ② meta, driver_carry, *_ = ray.get(                                  │
│       rollout_actor.rollout_to_tq.remote(repeated_batch, uids=…))    │
│      ← single Ray RPC; actor runs rollout + flatten + mask +         │
│        kv_first_write of bulk under uid-derived keys.                │
└────────────┬─────────────────────────────────────────────────────────┘
             │ bulk now in TQ; driver has meta + driver_carry slice
             ▼
┌─ DRIVER (reward + advantage, on driver_carry only) ──────────────────┐
│ ③ scale_rewards / apply_reward_shaping (legacy parity)               │
│ ④ baseline, std = calculate_baseline_and_std_per_prompt(...)         │
│   meta.stamp_tags({"std": …, "baseline": …})                         │
│      → filter-without-fetch primitive on meta                        │
│ ⑤ [optional] _apply_dynamic_sampling(meta, driver_carry, …)          │
│      → meta.subset(keep) + driver_carry.select_indices(keep)         │
│      → dp_client.kv_clear(dropped_keys)                              │
│ ⑥ overlong filter (loss_multiplier = 0 on truncated rows)            │
└────────────┬─────────────────────────────────────────────────────────┘
             ▼
┌─ DRIVER → WORKERS (logprob phase) ───────────────────────────────────┐
│ ⑦ prev_lp = policy.get_logprobs_from_meta(meta)                      │
│   ref_lp  = policy.get_reference_policy_logprobs_from_meta(meta)     │
│      ↓ inside the policy method:                                     │
│         shard_meta_for_dp(meta) — length-balanced split, pure meta   │
│         fan-out: worker.get_logprobs_presharded.remote(shard) × N    │
│           → _fetch(shard) → kv_batch_get → materialize               │
│           → forward → logprobs                                       │
│           → leader writes back as new TQ column on meta.keys         │
│ ⑧ extras  = policy.read_from_dataplane(meta, select_fields=[…])      │
│   advantages = compute_advantages(...)                               │
│ ⑨ policy.write_to_dataplane(meta, {"advantages": …, "sample_mask":…})│
└────────────┬─────────────────────────────────────────────────────────┘
             ▼
┌─ DRIVER → WORKERS (train + cleanup) ─────────────────────────────────┐
│ ⑩ policy.train_from_meta(meta, loss_fn=…)                            │
│      ↓ same shard_meta_for_dp + fan-out shape; no write-back         │
│        (training is terminal).                                       │
│ ⑪ policy.finish_step(meta) → drop step's bulk from TQ                │
└──────────────────────────────────────────────────────────────────────┘
                                                  → next step → ①
```

Bulk tensors live in TQ; the driver only holds `meta` + the small
`driver_carry` slice. On-wire layout is jagged
(`codec.pack_jagged_fields` ↔ `codec.materialize` at every put / get).

---

## `KVBatchMeta`

The receipt for a put. `meta.fields` is only what was written by *this*
put, not the partition-wide schema. See `interfaces.py` for the ABC.

| Attribute | Meaning |
|---|---|
| `partition_id` | TQ partition these keys live in |
| `keys` | Per-sample row identifiers |
| `fields` | Fields written by the put that minted this meta |
| `sequence_lengths` | Per-row valid (unpadded) lengths — drives length-balanced sharding |
| `tags` | `list[dict]` 1:1 with `keys` — per-row primitive sidecar for filter-without-fetch |
| `extra_info` | Batch-level bag (`rollout_metrics`, `pad_to_multiple`, packing metadata) |
| `task_name` | Optional consumer tag, carried through |

**Hard rules** — `kv_batch_put` fields must be `TensorDict` of tensors
(or `np.ndarray(dtype=object)`); primitives go on `tags`. `select_fields`
is required on every `kv_batch_get` — no implicit "fetch all".

---

## Helpers above the client

| Helper | What it does |
|---|---|
| `column_io.kv_first_write` | Rollout actor's flat first put. Caller mints `keys`. |
| `column_io.read_columns` / `write_columns` | `kv_batch_get` / `kv_batch_put` + jagged ↔ padded materialize. |
| `preshard.shard_meta_for_dp` | Pure metadata split, length-balanced when packing args are passed. |
| `KVBatchMeta.subset` / `.slice` / `.concat` | Pure meta transforms used by dynamic sampling; thread `tags` 1:1 with `keys`. |
| `KVBatchMeta.stamp_tags` | Mirror per-row scalars onto `meta.tags`. Init-if-None + length check. |
| `codec.pack_jagged_fields` | Jagged-pack at every put boundary. |

---

## Per-sample key invariant

Keys are minted **once** at rollout (`key_i = f"{uid}_g{i}"`) and reused
for every subsequent `kv_batch_put` / `kv_batch_get` on that sample.
Worker write-backs append new columns under the same keys.

---

## Concrete examples

### Call shapes

A real step at production scale —
`num_prompts_per_step=128, num_generations_per_prompt=4`, DP world = 8,
prompt ≈ 512 tok, response ≤ 1024 tok. Final batch is `128 × 4 = 512`
rows.

**1. Step prepare + rollout** (driver — `grpo_train_sync` body):

```python
# Open the per-step TQ partition. Cleared and reused across steps.
policy.prepare_step(num_samples=512, group_size=4)

# One Ray RPC bundles: clear gen metrics → rollout → flatten + mask →
# kv_first_write of bulk to TQ → finish_generation → metrics snapshot.
# The actor handles 6 stages internally; the driver gets back the
# meta handle + a small per-row tensor slice.
n_prompts = repeated_batch.size                # 512 (= 128 prompts × 4 gens)
uids = [str(uuid.uuid4()) for _ in range(n_prompts // 4)]   # 128 uids
meta, driver_carry, rollout_metrics, gen_metrics = ray.get(
    rollout_actor.rollout_to_tq.remote(
        repeated_batch,
        uids=uids,
        partition_id=policy.tq_partition_id,         # "train"
        first_iter=(dynamic_sampling_num_gen_batches == 1),
    )
)
# meta.keys             ≈ ["a3f9_g0", "a3f9_g1", "a3f9_g2", "a3f9_g3",
#                          "b7c1_g0", …]                       (512 keys)
# meta.sequence_lengths ≈ [847, 612, 1503, 989, 711, …]        (actual lens)
# meta.fields           = ["input_ids", "input_lengths",
#                          "generation_logprobs", "token_mask",
#                          "sample_mask", …multimodal extras…]
# driver_carry          : BatchedDataDict of per-row tensors
#                         (total_reward, loss_multiplier, truncated,
#                          length, input_lengths, prompt_ids_for_adv,
#                          response_token_lengths, GDPO components)
```

**2. Reward + dynamic sampling** (driver, on `driver_carry` only):

```python
driver_carry = scale_rewards(driver_carry, cfg["grpo"]["reward_scaling"])
if cfg["grpo"]["reward_shaping"]["enabled"]:
    driver_carry = apply_reward_shaping(driver_carry, cfg["grpo"]["reward_shaping"])
driver_carry["baseline"], driver_carry["std"] = (
    calculate_baseline_and_std_per_prompt(
        driver_carry["prompt_ids_for_adv"],
        driver_carry["total_reward"],
        torch.ones_like(driver_carry["total_reward"]),
        leave_one_out_baseline=cfg["grpo"]["use_leave_one_out_baseline"],
    )
)
# Mirror std/baseline onto meta so dynamic sampling can filter on
# meta alone (no tensor fetch).
meta.stamp_tags(
    {
        "std": driver_carry["std"].tolist(),
        "baseline": driver_carry["baseline"].tolist(),
    }
)

# DAPO non-zero-std filter — drops rows where the prompt's reward
# variance is zero, kv_clears their bulk, accumulates survivors
# across iterations until train_prompts_size (512) is reached.
if cfg["grpo"]["use_dynamic_sampling"]:
    pending_meta, pending_carry, *_ = _apply_dynamic_sampling(
        meta=meta, driver_carry=driver_carry,
        pending_meta=pending_meta, pending_carry=pending_carry,
        train_prompts_size=512,
        num_gen_batches=dynamic_sampling_num_gen_batches,
        max_gen_batches=cfg["grpo"]["dynamic_sampling_max_gen_batches"],
        dp_client=policy.dp_client,
    )
```

**3. Logprob + advantage + write-back**:

```python
# Worker fan-out happens inside these. Per-DP-rank shard via
# shard_meta_for_dp(meta, dp_world=8, …); each worker fetches its
# ~64 keys via kv_batch_get and writes back the result column under
# the same keys on the leader.
prev_lp = policy.get_logprobs_from_meta(meta, timer=timer)["logprobs"]
ref_lp  = policy.get_reference_policy_logprobs_from_meta(meta, timer=timer)
ref_lp  = ref_lp["reference_logprobs"]

# Driver-side per-token columns for masking. Tiny delta — just two
# fields × 512 rows.
extras = policy.read_from_dataplane(
    meta,
    select_fields=["generation_logprobs", "token_mask"],
    pad_value_dict=_pad_dict,
)
advantages = adv_estimator.compute_advantage(
    prompt_ids=driver_carry["prompt_ids_for_adv"],
    rewards=rewards, mask=mask,
    repeated_batch=adv_inputs,
    logprobs_policy=prev_lp,
    logprobs_reference=ref_lp,
)

# Write the per-token advantage + post-masking sample_mask back to TQ
# under meta.keys so workers fetch the unified view in train.
policy.write_to_dataplane(
    meta,
    fields={"advantages": advantages, "sample_mask": sample_mask},
)
```

**4. Train + cleanup**:

```python
train_results = policy.train_from_meta(meta, loss_fn=loss_fn, timer=timer)
policy.finish_step(meta)                              # drop step's bulk from TQ
```

**5. Validation path** — slim `driver_carry` to skip ~1 MB/batch:

```python
# inside validate_sync; val_batch_size ≈ 64
policy.prepare_val_partition(n_prompts, partition_id="val")
meta, driver_carry, rollout_metrics, _ = ray.get(
    rollout_actor.rollout_to_tq.remote(
        val_batch, uids=uids, partition_id="val",
        finish_generation=False,                       # keep inference state warm
        task_to_env_override=val_task_to_env,
        carry_keys=["total_reward"],                   # only field val consumes
    )
)
total_rewards.extend(driver_carry["total_reward"].tolist())
mlog_cols = policy.read_from_dataplane(
    meta, select_fields=["turn_roles", "turn_contents"],
)
policy.finish_step(meta)
```

### Sequence-length flow (seqpack / dynbatch)

How `meta.sequence_lengths` routes samples to DP ranks. Worked example
sized to one production microbatch — 4 prompts × 2 generations = 8
samples, DP world = 4, lengths typical of math/code rollouts.

```
# Rollout actor flattens prompt + response per sample.
# input_lengths[i] = prompt_len_i + response_len_i (actual content,
# unpadded).
sample 0 (a3f9_g0):  prompt=312, response=  892 → input_lengths=1204
sample 1 (a3f9_g1):  prompt=312, response=  187 → input_lengths= 499
sample 2 (b7c1_g0):  prompt=421, response= 1024 → input_lengths=1445   ← long
sample 3 (b7c1_g1):  prompt=421, response=  455 → input_lengths= 876
sample 4 (c0d8_g0):  prompt=148, response=  213 → input_lengths= 361   ← short
sample 5 (c0d8_g1):  prompt=148, response=  339 → input_lengths= 487
sample 6 (d2e1_g0):  prompt=276, response=  651 → input_lengths= 927
sample 7 (d2e1_g1):  prompt=276, response=  402 → input_lengths= 678

# kv_first_write returns meta row-aligned with keys:
meta.keys             = ["a3f9_g0", "a3f9_g1", "b7c1_g0", "b7c1_g1",
                         "c0d8_g0", "c0d8_g1", "d2e1_g0", "d2e1_g1"]
meta.sequence_lengths = [    1204,       499,      1445,       876,
                              361,       487,       927,       678 ]

# shard_meta_for_dp slices keys + sequence_lengths with the SAME
# idx_list — driver-side, no TQ I/O. Length-balanced via seqpack:
rank 0:  idx=[2, 4]      → keys=["b7c1_g0","c0d8_g0"]   lens=[1445, 361]   = 1806
rank 1:  idx=[0, 5]      → keys=["a3f9_g0","c0d8_g1"]   lens=[1204, 487]   = 1691
rank 2:  idx=[6, 1]      → keys=["d2e1_g0","a3f9_g1"]   lens=[ 927, 499]   = 1426
rank 3:  idx=[3, 7]      → keys=["b7c1_g1","d2e1_g1"]   lens=[ 876, 678]   = 1554
# Σ packed lengths per rank within ~25% — well-balanced.

# Each worker fetches its own ~64 keys per step from TQ:
data = self._fetch(shard)  # kv_batch_get(shard.keys, select_fields=…)
```

**Gotcha — `make_sequence_length_divisible_by` (TP×CP alignment)**:
`input_ids` is padded to a multiple of TP×CP at write time (e.g. 8 for
TP=4, CP=2), but `input_lengths` is the actual content length. Seqpack
balances on actual lengths; padding is reapplied per shard.

```
# row with input_lengths=1204, TP×CP=8 → input_ids padded to 1208:
input_ids:             [t0, t1, …, t1203,  0, 0, 0, 0]   # 1208 elems
input_lengths:                                   1204     # actual
meta.sequence_lengths:                           1204     # what seqpack uses ✓
```

---

## Configuration

The data plane is configured via a `data_plane:` block in the master
YAML (`examples/configs/...`). **YAML is the single source of truth
for defaults** — the adapter has no hidden `cfg.get(key, default)`
fallbacks. The canonical exemplar is
`examples/configs/grpo_math_1B.yaml`.

All eight keys below are **required** when `enabled=true`. Recipes
under `examples/configs/recipes/**/*.yaml` inherit them via
`defaults:` from the exemplar.

```yaml
data_plane:
  enabled: false                       # flip to true to engage grpo_train_sync
  impl: transfer_queue                 # only one impl today
  backend: "simple"                    # "simple" or "mooncake_cpu"
  storage_capacity: 1000000            # max samples retained per partition
  num_storage_units: 2                 # storage shards
  claim_meta_poll_interval_s: 0.5      # blocking-claim poll cadence
  global_segment_size: 549755813888    # 512 GiB — used when backend == "mooncake_cpu"
  local_buffer_size:   68719476736     # 64 GiB  — used when backend == "mooncake_cpu"
  # observability:                     # NotRequired
  #   enabled: false
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

## When `data_plane.enabled=False`

`build_data_plane_client` raises — there is no NoOp prod fallback.
For the no-data-plane path use the legacy
`nemo_rl.algorithms.grpo.grpo_train`; the sync trainer
`grpo_train_sync` requires `enabled=True` and a `TQPolicy`.

`NoOpDataPlaneClient` (`adapters/noop.py`) exists only as a unit-test
fixture for the ABC contract tests.

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

## Async path (proposed)

The data-plane interface covers both sync and async, but the **sync
trainer uses only half of it**. The task-mediated half
(`claim_meta` / `get_data` / `check_consumption_status`) is reserved
for the async trainer, which is not yet wired into production.

Design proposal, filtering / staleness strategies, and open questions:
see [`docs/data-plane-async-proposal.md`](docs/data-plane-async-proposal.md).
