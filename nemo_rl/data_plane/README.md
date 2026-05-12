# nemo_rl.data_plane

Stable boundary between NeMo-RL and any data-plane implementation
(currently `transfer_queue`; future: `nv-dataplane`). All call sites in
`nemo_rl/algorithms`, `nemo_rl/experience` and `nemo_rl/models` go
through `DataPlaneClient` — never `import transfer_queue` directly.
That's the swappable boundary.

This README is the canonical reference: quickstart for users, runtime
view for anyone touching `nemo_rl/algorithms/grpo_sync.py`,
`nemo_rl/experience/sync_rollout_actor.py`, or `nemo_rl/data_plane/`.

## Install

`tensordict` and `TransferQueue==0.1.6` are base dependencies of
nemo-rl — `uv sync` (or `pip install -e .`) is enough; there is no
`[data-plane]` extra to remember. Worker venvs (built per-backend by
`nemo_rl.utils.venvs.create_local_venv` via bare `uv sync`) pick them up
automatically too, so the TQ adapter works on every worker class
(FSDP2, DTensor, mcore, automodel) without per-extra plumbing.

## Quickstart

```python
from tensordict import TensorDict
import torch

from nemo_rl.data_plane import build_data_plane_client

client = build_data_plane_client({
    "enabled": True,
    "impl": "transfer_queue",
    "backend": "simple",          # or "mooncake_cpu"
    "storage_capacity": 1_000_000,
    "num_storage_units": 2,
})

client.register_partition(
    partition_id="train",
    fields=["input_ids", "advantages"],
    num_samples=1024,
    consumer_tasks=["prev_lp", "ref_lp", "train"],
)

# Producer (rollout, ref policy, …) — sync put. Use ``async_kv_batch_put``
# only when composing with an existing event loop (e.g. async rollout
# actor).
client.kv_batch_put(
    keys=["uid-0", "uid-1"],
    partition_id="train",
    fields=TensorDict({"input_ids": torch.zeros(2, 128, dtype=torch.long)},
                      batch_size=[2]),
)

# Consumer — task-mediated discovery + claim (advances per-task cursor).
meta = client.claim_meta(
    partition_id="train",
    task_name="train",
    required_fields=["input_ids", "advantages"],
    batch_size=64,
)
batch = client.get_data(meta)        # TensorDict
```

## When `enabled=False`

The factory raises — there is intentionally no NoOp prod fallback.
Use the legacy `nemo_rl.algorithms.grpo.grpo_train` trainer for that
case (it never engages the data plane). The TQ-mediated trainer lives
at `nemo_rl.algorithms.grpo_sync.grpo_train_sync` and assumes
`enabled=True`.

`NoOpDataPlaneClient` exists in `adapters/noop.py` purely as a test
fixture for the ABC contract tests — production callers must not import
it.

## Hard rules

These are checked at the adapter; violating them is a `TypeError`, not
a warning.

* **No Python leaves on the bus.** `kv_batch_put(fields=...)` must be a
  `TensorDict` of tensors. Use `tags=` for primitives, the Ray object
  store for arbitrary Python objects.
* **`select_fields` is required on read.** `get_data` raises if neither
  `select_fields` nor `meta.fields` is set — silently fetching the full
  sample record is not allowed.

---

## The API surface

Everything goes through `DataPlaneClient`
(`nemo_rl/data_plane/interfaces.py`). Eight methods, three groups.

### Lifecycle

- `register_partition(partition_id, fields, num_samples, consumer_tasks, ...)`
  declares the partition schema and which consumer tasks read from it.
- `close()` releases controller / storage handles.

### Task-mediated (consumer-counter aware)

- `claim_meta(partition_id, task_name, required_fields, batch_size) → KVBatchMeta`
  discovers and claims samples ready for `task_name`; advances TQ's
  per-task consumption cursor as a side effect.
- `get_data(meta, select_fields) → TensorDict` resolves a meta to data.
- `check_consumption_status(...) → bool`.

### Direct-by-key (the hot path in sync 1-hop)

- `kv_batch_put(keys, partition_id, fields)` — producer entrypoint;
  flips `production_status[sample, field] = 1` as a side effect.
- `kv_batch_get(keys, partition_id, select_fields) → TensorDict` — direct fetch.
- `kv_clear(keys, partition_id)` — drop.

### Helpers built on top (`nemo_rl/data_plane/`)

- `kv_first_write(batch, uids, ...) → KVBatchMeta` — single flat
  `kv_batch_put` of all rollout fields.
- `read_columns(client, meta, select)` — `kv_batch_get → materialize`.
- `write_columns(client, meta, fields)` — typed `kv_batch_put` for deltas.
- `shard_meta_for_dp(meta, dp_world)` — pure metadata split, no I/O,
  no key remint.
- `meta.subset(idxs)` / `meta.slice(start, stop)` / `meta.concat(other)` —
  pure metadata transforms (methods on `KVBatchMeta`; used by
  dynamic_sampling).

## Per-sample key invariant

Mint **once** at rollout, reuse forever:

```
  uid   = "step17_prompt_42"          # opaque, from driver dataset iter
  key_i = f"{uid}_g{i}"               # one per generation, i ∈ [0, n_gen)
```

Every `kv_batch_put` / `kv_batch_get` for that sample uses the same key.
Worker write-backs append columns; nothing remints.

## E2E lifecycle for one GRPO step

```
┌──────────────────────────── DRIVER (grpo_sync.py) ─────────────────────────────┐
│                                                                                │
│ ① register_partition(pid="step17", fields=[input_ids, ..., advantages, ...],   │
│                       num_samples=N*G, consumer_tasks=["lp","ref","train"])    │
│                                                                                │
└─────────────┬──────────────────────────────────────────────────────────────────┘
              │  spawns
              ▼
┌──────────── SyncRolloutActor (Ray @remote) ───────────────────────────────────┐
│   vllm.generate → flatten → mask → prompt extract                              │
│ ② kv_batch_put( keys=[uid_g0..uid_gN-1],                                       │
│                 fields=TensorDict({input_ids, gen_logprobs, token_mask, ...})) │
│   returns meta → driver                                                        │
└──────────────────────────────────────────────────────────────────────────────┬─┘
                                                                               │
              ┌─ DRIVER ─────────────────────────────────────────────────┐    │
              │ ③ shard_meta_for_dp(meta, dp_world=8)  → [m₀..m₇]        │◄───┘
              │   (pure metadata, no I/O, no key remint)                 │
              └────┬─────────────────────────────────────────────────────┘
                   │  Ray-call per DP rank with mᵢ
                   ▼
┌──────────── MegatronPolicyWorker[rank=i] (×8) ─────────────────────────────────┐
│ ④ kv_batch_get(keys=mᵢ.keys, select=[input_ids, token_mask, ...])              │
│   forward → prev_logprobs                                                      │
│ ⑤ leader-only: kv_batch_put(keys=mᵢ.keys, fields={prev_logprobs:T})  ── PHASE 1│
│                                                                                │
│ ⑥ kv_batch_get(...)  → ref_logprobs                                            │
│ ⑦ leader-only: kv_batch_put({reference_policy_logprobs:T})           ── PHASE 2│
└──────────────────────────────────────────────────────────────────────────────┬─┘
                                                                               │
              ┌─ DRIVER (small slice work, never bulk) ──────────────────┐    │
              │ ⑧ read_columns(meta, select=[token_logprobs, rewards])   │◄───┘
              │   compute advantages (vectorized, on driver, tiny)       │
              │ ⑨ write_columns(meta, {advantages: T})                   │
              │                                                          │
              │   [optional] dynamic_sampling: meta.subset(...)          │
              │   [optional] kv_clear(dropped_keys)                      │
              └────┬─────────────────────────────────────────────────────┘
                   │  shard_meta_for_dp again, Ray-call per rank
                   ▼
┌──────────── MegatronPolicyWorker[rank=i] (×8) ─────────────────────────────────┐
│ ⑩ kv_batch_get(select=[input_ids, prev_logprobs, ref_lp, advantages, masks])   │
│   loss → grad → optimizer.step()                                               │
│   (no write-back: training is terminal for this partition)                     │
└──────────────────────────────────────────────────────────────────────────────┬─┘
                                                                               │
              ┌─ DRIVER (step-end housekeeping) ─────────────────────────┐    │
              │ ⑪ kv_batch_get(select=[input_ids])  ← stash for log_data │◄───┘
              │ ⑫ kv_clear(keys=meta.keys, partition_id=pid)             │
              └──────────────────────────────────────────────────────────┘

       (next step → ① again with a fresh partition_id)
```

Mental model: **TQ is the bus, not a database.** It holds bulk between
stages of one step, then `kv_clear` drops it. Driver only handles small
per-sample slices; workers handle bulk via TQ.

## Call counts per step

Steady state on the validation run (32 samples, 8 GPUs, no PP/TP):

| TQ call                    | Site                | Count / step | Payload                           |
|----------------------------|---------------------|-------------:|-----------------------------------|
| `register_partition`       | driver              | 1            | metadata only                     |
| `kv_batch_put` (rollout)   | SyncRolloutActor    | 1            | full bulk (~600 KB; GBs at scale) |
| `shard_meta_for_dp`        | driver              | 3            | no I/O                            |
| `kv_batch_get` (lp inputs) | workers             | 8 (per DP)   | input slice                       |
| `kv_batch_put` (lp out)    | workers (leader)    | 1            | prev_logprobs delta               |
| `kv_batch_get` (ref input) | workers             | 8            | input slice                       |
| `kv_batch_put` (ref out)   | workers (leader)    | 1            | ref_logprobs delta                |
| `kv_batch_get` (adv slice) | driver              | 1            | small (rewards + token_lp)        |
| `kv_batch_put` (advantages)| driver              | 1            | small delta                       |
| `kv_batch_get` (train)     | workers             | 8            | full slice                        |
| `kv_batch_get` (log_data)  | driver              | 1            | input_ids only                    |
| `kv_clear`                 | driver              | 1            | drop                              |

Total: ~32 TQ RPCs / step (excluding `shard_meta_for_dp`, which is
no-I/O). 24 of those are the per-DP fetch fan-out (3 phases × 8 ranks).

## Concrete examples

**Rollout produces (only first-write):**
```python
meta = kv_first_write(
    final_batch_cpu=batch,
    uids=[f"step{step}_p{i}" for i in range(num_prompts)],
    dp_client=policy._dp_client,
    partition_id=f"grpo_step_{step}",
)
# meta.keys = ["step17_p0_g0", "step17_p0_g1", ..., "step17_p7_g3"]
# meta.fields = ["input_ids", "input_lengths", "generation_logprobs",
#                "token_mask", "sample_mask", ...]
```

**Driver appends a column (small delta, no bulk):**
```python
slice_ = read_columns(client, meta, select_fields=["token_logprobs", "rewards"])
advantages = compute_advantages(slice_)         # tiny driver compute
write_columns(client, meta, {"advantages": advantages})
```

**Worker fan-out (driver):**
```python
shards, _ = shard_meta_for_dp(meta, dp_world=8)
ray.get([
    worker[i].train_from_meta.remote(shards[i])
    for i in range(8)
])
```

**Worker fetch + leader write-back (in `worker_mixin._write_back`):**
```python
inputs = read_columns(self._dp_client, meta, select_fields=LP_SEED_FIELDS)
prev_lp = self.forward(inputs)
if self._is_replica_leader():
    write_columns(self._dp_client, meta, {"prev_logprobs": prev_lp})
```

**Step-end teardown:**
```python
log_input_ids = read_columns(client, meta, select_fields=["input_ids"])
client.kv_clear(keys=meta.keys, partition_id=meta.partition_id)
```

## Performance characterization

End-to-end parity vs the legacy driver-bulk path on the toy validation
run:

- Steps 1–7 are bit-exact (loss + reward); divergence afterward is the
  expected stochastic drift from accumulated policy updates.
- Steady-state step time: **+0.21 s** (1-hop 7.86 s vs legacy 7.65 s,
  ~3 %).

Per-phase breakdown (steady state, steps 2–19):

| Phase                         | v4 (1-hop) | Legacy   | Δ          |
|-------------------------------|-----------:|---------:|-----------:|
| Total step time               | 7.606 s    | 7.393 s  | **+0.213 s** |
| policy_training               | 0.596 s    | 0.567 s  | +0.028 s   |
| generation                    | 1.502 s    | 1.528 s  | −0.027 s   |
| policy_and_ref_logprob        | 1.588 s    | 1.448 s  | **+0.141 s** |
| residual (driver bookkeeping) | 3.920 s    | 3.850 s  | +0.070 s   |

**The +0.21 s overhead is entirely TQ RPC roundtrip cost in the
logprob phase** (two worker calls × one fetch + one write each).
Generation and training are unchanged.

### Crossover scale (where TQ wins)

TQ overhead is mostly latency-bound (~constant per step), while legacy
driver fan-out is bandwidth-bound (scales with batch tensor volume ×
DP fan-out). Mental model:

- Legacy driver overhead ≈ ~5 ms/MB × (4 full-batch transfers per step)
  × DP-fan-out
- TQ overhead ≈ ~200 ms fixed (after fuse-and-overlap optimization:
  ~100 ms)

| Scale                                    | Batch / step | DP ranks | Legacy cost | Winner                  |
|------------------------------------------|-------------:|---------:|------------:|-------------------------|
| Toy (this run, 1B, 512 tok, BS 32)       | 0.6 MB       | 8        | ~50 ms      | **legacy +0.21 s**      |
| Small prod (8B, 1k tok, BS 256)          | ~10 MB       | 8        | ~300 ms     | **roughly tied**        |
| Mid prod (70B, 4k tok, BS 1024)          | ~250 MB      | 32       | ~5–10 s     | **TQ wins decisively**  |
| Long-context (8k–32k seq, GRPO 16 gens)  | 1–5 GB       | 64+      | tens of s   | **TQ wins decisively**  |

Rough crossover: **~10 MB / step / DP-rank of effective batch volume**.
Long sequences, more generations per prompt, and more DP ranks all
push the needle hard toward TQ.

### Cheapest optimizations (deferred)

1. **Fuse `get_logprobs` + `get_reference_policy_logprobs` into one
   worker call** — saves ~70 ms (one TQ input-fetch). Brings overhead
   from +0.21 s → ~+0.14 s.
2. **Overlap TQ write-back with next-phase fetch** — saves another
   ~30–50 ms. Combined: ~+0.10 s overhead, effectively at parity.

Both are clean refactors inside `tq_policy.py` /
`worker_mixin.py` and don't touch `grpo_sync.py`. Not on the
critical path; flag for the next data-plane optimization round.

## Where to look in the code

| Concern                          | File                                                                 |
|----------------------------------|----------------------------------------------------------------------|
| Stable boundary                  | `nemo_rl/data_plane/interfaces.py`                                   |
| Adapter (TransferQueue impl)     | `nemo_rl/data_plane/adapters/transfer_queue.py`                      |
| Column helpers above DP client   | `nemo_rl/data_plane/column_io.py` (`read_columns`, `write_columns`)  |
| First-write helper + rollout actor | `nemo_rl/experience/sync_rollout_actor.py`                         |
| DP-rank meta sharding            | `nemo_rl/data_plane/preshard.py`                                     |
| Worker fetch + write-back        | `nemo_rl/data_plane/worker_mixin.py`                                 |
| TQ-aware policy facade           | `nemo_rl/models/policy/tq_policy.py`                                 |
| End-to-end orchestration         | `nemo_rl/algorithms/grpo_sync.py`                                    |
| Unit tests                       | `tests/data_plane/unit/`                                             |

## Operational assumptions

* One Ray cluster per experiment. The TQ controller is a globally
  named Ray actor; running two trainers in the same cluster will
  collide.
* Storage capacity sizing rule of thumb:
  `storage_capacity ≥ 2 × num_prompts × n_gens × max_seq_len ×
  bytes_per_token × num_active_fields`.
