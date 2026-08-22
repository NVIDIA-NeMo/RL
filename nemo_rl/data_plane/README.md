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
| `extra_info` | Batch-level bag (`rollout_metrics`, `pad_to_multiple`, `global_forward_pad_seqlen`, packing metadata) |
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

**Gotcha — DP-rank seq-dim alignment (`global_forward_pad_seqlen`)**:
Each DP rank's `_fetch` would otherwise pad to its slice's local max,
so two ranks in the same step could forward at different seq dims.
That breaks any collective that assumes cross-rank shape uniformity
(mcore MoE all-to-all, CP, etc.). The data plane handles this with a
single per-batch cap minted on the driver:

* `TQPolicy._stamp_pad_seqlen(meta)` runs before every fan-out
  (`train_from_meta`, `_logprob_dispatch`, `read_from_dataplane`).
  Idempotent — sets `meta.extra_info["global_forward_pad_seqlen"]`
  to `round_up(max(meta.sequence_lengths), max(pad_to_multiple,
  sequence_length_round))` on first call, no-op on subsequent calls.
* `shard_meta_for_dp` propagates `extra_info` to every per-rank meta
  via `dict(meta.extra_info)` — so all ranks see the same target.
* Worker `_fetch` and driver `read_columns` both pass
  `pad_to_seqlen = meta.extra_info["global_forward_pad_seqlen"]`
  into `codec.materialize`, which right-pads the seq dim to that
  absolute target. All DP ranks within a step therefore return
  columns at one identical seq dim.

Opt out in tests with `_fetch(..., dp_aligned_seq_len=False)` to
observe per-rank local-pad behavior.

```
# 4 DP ranks, slice maxes: [1208, 1320, 944, 1080]; sequence_length_round=64
global_forward_pad_seqlen = round_up(1320, 64) = 1344
# All 4 ranks pad their materialized tensors to seq_dim=1344.
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
  observability:                       # NotRequired
    enabled: true                      # per-op timing / latency percentiles / volume
    verify_tensor_hash: false          # debug: wire-in vs wire-out tensor check
```

### Observability

`enabled: true` wraps the adapter in `MetricsDataPlaneClient`, which records
per-op wall time, latency percentiles (fixed-bucket histogram, so per-rank
counts sum into one cluster-wide distribution) and byte volume. `snapshot()`
returns the cumulative view; `get_step_metrics(step_time_s)` returns the
per-step delta already flattened for the logger.

**Scope: one process, not the cluster.** Every process builds its own
client with its own counters — the driver, each policy worker, the rollout
actor. `grpo_train_sync` logs the *driver's*, under `data_plane/driver/`.
The driver issues about one op of each kind per step, so `calls` is small
by construction; the bulk traffic is the rollout actor's `kv_first_write`
and the workers' per-DP-rank `get_samples`, and neither appears in these
series. Do not read `comm_volume_mb` as cluster-wide volume.

`OpStats` is additive on purpose, and `merge_snapshots()` uses it: the
histogram buckets and the regression sufficient statistics from every rank
*sum* into one cluster-wide view. Everything derived — percentiles, the
affine fit, throughput — is recomputed from the merged totals, never
averaged across ranks (averaging per-rank percentiles does not give a
cluster percentile).

**A per-op breakdown table** is logged alongside the series, under
`data_plane/{cluster,driver}/breakdown` — one row per op, ordered by wall
time so the expensive one reads first:

| op | calls | mean_ms | max_ms | wall_ms | overhead_ms | transfer_ms | p50_ms | p99_ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| get | 200 | 11.17 | 16.0 | 2232 | 6.03 | 5.15 | 12.9 | 16.0 |
| put | 48 | 23.66 | 31.9 | 1135 | 12.06 | 11.60 | — | — |

Everything in ms on that row is **per call**, and the two split terms add
to `mean_ms`: that `get` row reads "each call cost 11.17 ms, of which 6.03
was fixed per-request overhead and 5.15 was bandwidth at this step's mean
request size". Per call the overhead term *is* the fitted constant, so it
is comparable against a hardware number. `calls` and `wall_ms` are the only
extensive columns.

**Per-call figures describe the wire; sums describe the run.** `wall_ms` on the cluster
path is summed over processes that ran concurrently, so it is process-time
and scales with DP degree — 200 gets of 11 ms across 8 ranks reads 2232
while the wall clock was 279. Dividing by the process count only trades one
arbitrary denominator for another. Per call is invariant to both DP degree
and batch size: the same workload at 8 and at 32 ranks reports 11.16 and
11.06 ms while `wall_ms` quadruples. Use `mean_ms` to compare runs and
cluster sizes, `wall_ms` to attribute cost across ops within one step.

`overhead_ms` and `transfer_ms` stack to the measured `wall_ms` — that row
reads "2233 ms of get was 1207 ms of fixed per-call cost and 1026 ms of
bandwidth", which is a decision you can act on.

A stack of line charts answers "how did put's wall time trend"; this
answers "where did the step go", which is a table. Cells are empty rather
than zero where a series was withheld (a percentile below the sample gate,
a fit that is not trustworthy) — a zero would read as a measurement. It is
built from the same metrics dict that is logged, so the table and the
series cannot disagree. Only wandb renders it; other backends skip it.

**Every series says what kind of number it is.** A per-step delta and an
instantaneous level shared the `_mb` suffix and a chart, with nothing to
tell them apart:

| namespace | meaning | example |
|---|---|---|
| `step/` | what happened during this step; resets | `step/comm_volume_mb` |
| `now/` | what is true at this instant; persists | `now/bytes_outstanding_mb` |

A rising `now/bytes_outstanding_mb` is not an accumulation bug — it is the
leak signal the metric exists for: bytes put and never cleared.

Two more read differently in the cluster view and are named to say so:

- **`step/wall_ms_per_process`**, not a bare fraction. `wall_ms` sums
  processes that ran concurrently, so dividing it by one step's wall clock
  exceeded 1 whenever they overlapped (measured 1.054 across ten
  processes). Per process it is a duration in ms, like everything else.
- **Percentiles are per step and clamped to the exact `max_ms`**, and are
  withheld entirely below 50 calls in the window. Bucket interpolation
  spreads a bucket's samples uniformly across it, so calls clustered low in
  a wide bucket read high — 160 calls of 120 ms all land in `(100, 250]`
  and interpolate to a p50 of 175, above every call observed and above the
  max reported beside it. The max is measured exactly, so it is the tighter
  bound.

`grpo_train_sync` fans out to the driver and every policy worker, and logs
the combined result under `data_plane/cluster/` instead of the driver's
own. **It does not reach the rollout actor**, which builds its own client
and is not on the worker group — so `kv_first_write`, the write of the
whole rollout, is not in these totals. It falls back to `data_plane/driver/` when the fan-out finds only one
process. Measured: **~2.4 ms and ~1 kB per process per step** for 10
processes, against a 6x wider view of the traffic. The fan-out is
best-effort — a rank that cannot answer is dropped rather than failing the
step.

`observability_overhead_ms` reports what the measurement itself cost — the
whole bill, both halves:

- every process's wrapper time (its wall time minus the time its inner
  client was working), and
- the fan-out that gathered and merged the snapshots.

The second is the larger. In the cross-process e2e the wrapper cost 0.13 ms
and the fan-out 2.31 ms, so a figure covering only the first understated by
19x. Measured whole: **~2.4 ms, about 0.9% of data-plane time** for 10
processes.

It is deliberately not clamped to 100%. Against a fast backend the ratio
can exceed 1, meaning measuring cost more than the operation measured — a
signal worth seeing rather than hiding.

**Units:** every duration is `_ms`, every volume is `_mb`, no exceptions —
a chart mixing `wall_s` against `p99_ms` puts a 0.008 beside a 24.85 and
reads as a data-plane bug rather than an axis one.

**Per step you get, per op tag:** `calls`, `wall_ms`, `max_ms`, and — when
the affine fit is trustworthy — `overhead_ms` and `transfer_ms`. Those last
two are the split of the op's time into fixed per-request cost and
bandwidth, in ms, and they *stack*: together they are the model's estimate
of the step's `wall_ms`, so charting them against the measured `wall_ms`
shows the breakdown and the model error in one picture. The coefficients
come from the cumulative fit (a model should be stable); the attribution is
per step, applied to that step's calls and bytes. A ratio was tried first
and was the wrong shape — cumulative and therefore flat, and unitless on an
axis of milliseconds. Not percentiles — the
histogram is cumulative by design, so a per-step p50 off it goes flat, and
at the handful of calls an op makes in one step a p99 is bucket geometry
rather than data (one sample in the `(10, 25]` bucket always yields
`10 + 15*0.99 = 24.85`). `max_ms` is exact, scoped to the step, and says
the same thing at that sample size. The percentiles remain in `snapshot()`,
where the cumulative sample count justifies them.

Measured against a no-op inner client on the payload the wire actually
carries — 256 ragged rows, 12 MB, jagged per-token fields as
`pack_jagged_fields` leaves them: **~37 µs per put, ~15 µs per get**, under
0.1% of a 59 ms operation. What is left is dominated by the per-key
attribution `clear_samples` needs to undo.

This is **on in the exemplar config**, which is where a v1 `TypedDict`
default lives — so recipes inheriting `grpo_math_1B.yaml` get it, and a
config with no `observability:` block still falls back to `False` at the
factory. It only engages when `data_plane.enabled` is true either way, so
it costs nothing for runs that don't use the data plane. There is
no default per-op sink: `get_step_metrics()` is the surface, and
`grpo_train_sync` logs it once a step under the `data_plane/` prefix — so
the series reach whatever backends the run has enabled (wandb, TensorBoard,
MLflow). Roughly 5-8 series per distinct op tag. Set
`observability.callback` if you additionally want a hook on every transfer;
`log_event` is exported for that.

`verify_tensor_hash: true` additionally records a `torch.hash_tensor`
fingerprint on every put and re-checks it on every get, so a tensor that
changes between wire-in and wire-out is reported (`hash/mismatches`)
instead of being trained on silently.

Two granularities, because torch has no ragged hash kernel:

| leaf | digest | scope |
|---|---|---|
| rectangular (`rewards`, `input_lengths`) | one per row, `hash_tensor(..., dim=1)` | per sample id; survives shard reads |
| jagged (`input_ids`, `generation_logprobs`, `token_mask`, `advantages`) | one over the values buffer, XORed per row with that row's length | per batch; a shard read reports unverified |

Giving the jagged fields per-row digests would mean padding each one out to
a rectangle first. On a realistically ragged batch that rectangle is 3.5×
the real payload and cost 13× more, to answer a question the buffer digest
already answers.

**Detection is not attribution, and the difference is the whole point of
the split.** The same corruption, injected into an 8-row batch:

| corruption | jagged leaf | rectangular leaf |
|---|---|---|
| 1 element changed in `u3` | caught, flags **all 8 rows** | caught, names **`u3`** |
| `u5` zeroed | caught, flags all 8 rows | caught, names `u5` |
| `u3`↔`u4` swapped | caught only if their lengths differ | caught, names `u3`,`u4` |
| nothing | clean | clean |

A jagged digest covers the whole values buffer, so any change moves every
row's value: it says *this batch is wrong*, never *this sample is wrong*.
Since `pack_jagged_fields` leaves ~94% of the payload jagged (only
`rewards` and `input_lengths` stay rectangular), that is the normal
resolution — you learn a step's transfer diverged and have to bisect for
the row yourself.

Verified by injecting corruption into the round trip. Caught: a
single-element change in every dtype, a truncated row, a zeroed row, a
bf16→fp32 precision change, and a row served from the wrong sample — with
**zero false alarms** over a 500-row randomized soak, every shard grouping
from 1 to 256, reversed id order, field subsets and delta writes. Known
limits, measured rather than assumed:

- A **mis-shard** (two rows swapped) is caught on a jagged field only when
  the two rows differ in length — 58/60 on ragged rollout data, never on a
  uniform-length batch. Rectangular fields catch it unconditionally.
- It reads every tensor byte again on both sides — ~2.4 ms for a 12 MB
  jagged batch, on put and again on get. Keep it to debugging runs.
- A rectangular field that comes back **jagged** (one row truncated makes
  the batch ragged) has no comparable digest and is dropped — counted in
  `hash/fields_skipped`, not reported as a mismatch.
- Only rows this process wrote can be checked. A consumer-side client
  reports them under `hash/rows_unverified` rather than counting them
  clean, and `hash/fields_skipped` reports any leaf it could not compare
  — watch that one, since a guard that quietly stops covering a field still
  reports zero mismatches.

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
