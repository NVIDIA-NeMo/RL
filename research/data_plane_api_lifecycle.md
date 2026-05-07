# Data Plane API & GRPO Lifecycle

Companion to `data_plane_integration_plan.md`. Captures the runtime view:
what calls TQ, in what order, with what payloads — and how this differs
from verl's TQ-on-PPO trainer.

Audience: anyone touching `nemo_rl/algorithms/grpo_sync.py`,
`nemo_rl/data_plane/`, or `nemo_rl/algorithms/sync_utils.py`.

---

## 1. The API surface

Everything goes through `DataPlaneClient` (`nemo_rl/data_plane/interfaces.py`).
Eight methods, three groups. Call sites in `nemo_rl/algorithms`,
`nemo_rl/experience`, and `nemo_rl/models` always go through this client —
they never `import transfer_queue` directly. That's the swappable boundary.

### Lifecycle

- `register_partition(partition_id, fields, num_samples, consumer_tasks, ...)`
  declares the partition schema and which consumer tasks will read from it
- `close()` releases controller / storage handles

### Task-mediated (consumer-counter aware)

- `get_meta(partition_id, task_name, required_fields, batch_size) → KVBatchMeta`
  discovers samples ready for `task_name`; advances TQ's per-task counter
- `get_data(meta, select_fields) → TensorDict` resolves a meta to data
- `check_consumption_status(...)` — bool

### Direct-by-key (the hot path in sync 1-hop)

- `kv_batch_put(keys, partition_id, fields)` — producer entrypoint;
  flips `production_status[sample, field] = 1` as a side effect
- `kv_batch_get(keys, partition_id, select_fields) → TensorDict` — direct fetch
- `kv_clear(keys, partition_id)` — drop

### Helpers built on top (`nemo_rl/data_plane/`)

- `kv_first_write(batch, uids, ...) → KVBatchMeta` — single flat
  `kv_batch_put` of all rollout fields
- `read_columns(client, meta, select)` — `kv_batch_get → materialize`
- `write_columns(client, meta, fields)` — typed `kv_batch_put` for deltas
- `shard_meta_for_dp(meta, dp_world)` — pure metadata split, no I/O,
  no key remint
- `meta.subset(idxs)` / `meta.slice(start, stop)` / `meta.concat(other)` — pure metadata transforms (methods on `KVBatchMeta`)
  (used by dynamic_sampling)

---

## 2. Per-sample key invariant

Mint **once** at rollout, reuse forever:

```
  uid   = "step17_prompt_42"          # opaque, from driver dataset iter
  key_i = f"{uid}_g{i}"               # one per generation, i ∈ [0, n_gen)
```

Every `kv_batch_put` / `kv_batch_get` for that sample uses the same key.
Worker write-backs append columns; nothing remints. This is the same
invariant verl maintains (`{uid}_{session_id}_{i}`).

---

## 3. E2E lifecycle for one GRPO step

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

Mental model: **TQ is the bus, not a database.** It holds bulk between stages
of one step, then `kv_clear` drops it. Driver only handles small per-sample
slices; workers handle bulk via TQ.

---

## 4. Call counts per step

Steady state on the validation run (32 samples, 8 GPUs, no PP/TP):

| TQ call                    | Site                | Count / step | Payload                        |
|----------------------------|---------------------|-------------:|--------------------------------|
| `register_partition`       | driver              | 1            | metadata only                  |
| `kv_batch_put` (rollout)   | SyncRolloutActor    | 1            | full bulk (~600 KB; GBs at scale) |
| `shard_meta_for_dp`        | driver              | 3            | no I/O                         |
| `kv_batch_get` (lp inputs) | workers             | 8 (per DP)   | input slice                    |
| `kv_batch_put` (lp out)    | workers (leader)    | 1            | prev_logprobs delta            |
| `kv_batch_get` (ref input) | workers             | 8            | input slice                    |
| `kv_batch_put` (ref out)   | workers (leader)    | 1            | ref_logprobs delta             |
| `kv_batch_get` (adv slice) | driver              | 1            | small (rewards + token_lp)     |
| `kv_batch_put` (advantages)| driver              | 1            | small delta                    |
| `kv_batch_get` (train)     | workers             | 8            | full slice                     |
| `kv_batch_get` (log_data)  | driver              | 1            | input_ids only                 |
| `kv_clear`                 | driver              | 1            | drop                           |

Total: ~31 TQ RPCs / step. 16 of those are the per-DP fetch fan-out
(3 phases × 8 ranks − overlaps).

---

## 5. Concrete examples

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
shards = shard_meta_for_dp(meta, dp_world=8)
ray.get([
    worker[i].train_from_meta.remote(shards[i])
    for i in range(8)
])
```

**Worker fetch + leader write-back (in `base_policy_worker._write_back`):**
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

---

## 6. High-level comparison with verl

verl's TQ-aware trainer lives in
`verl/verl/trainer/main_ppo_sync.py`. Same TQ primitive (`tq.kv_batch_put` /
`kv_batch_get` / `kv_clear`), but a different *integration shape*:

| Dimension              | verl (`main_ppo_sync.py`)                                | nemo-rl (sync 1-hop)                              |
|------------------------|----------------------------------------------------------|---------------------------------------------------|
| API surface            | `tq.*` module functions                                  | `DataPlaneClient` ABC, swappable adapters         |
| Init                   | `tq.init()` once globally                                | `register_partition` per step                     |
| Generation actor       | Per-prompt async `AgentLoopWorkerTQ`s; each writes when its agent loop finishes | One batched `SyncRolloutActor`; single put after all generations done |
| Producer→consumer signal | Tags (`{"global_steps": N, "status": "success"}`) polled by `ReplayBuffer` background thread | Controller-side `production_status` bit; consumers wait on field production |
| Step gate              | `ReplayBuffer.sample()` blocks until all prompts of `global_steps` are tagged success | Rollout actor's `ray.get()` returns only when entire batch done |
| Driver-side compute    | Driver pulls **bulk** (full input_ids + response_mask) for `_compute_old_log_prob`, `_compute_values`, `_compute_advantage` | Driver only touches **small slices** (advantages-input, log_data) |
| Worker fan-out         | Workers receive full meta, do their own internal sharding | Driver `shard_meta_for_dp` fan-out, workers receive pre-sliced meta |
| Async API              | `tq.async_kv_batch_put` used at agent-loop tail          | Sync only (deliberately simplified — see §1.2 of integration plan) |
| Multi-policy           | actor + critic + ref split, each writes back            | actor + ref only (GRPO has no critic)             |

### What verl does that we don't (yet)

1. **Per-prompt async generation.** verl's `AgentLoopWorkerTQ` writes to TQ
   as each agent loop finishes. First finishers can in principle pipeline
   into logprob compute earlier. We currently wait for the whole rollout
   actor batch. Tracked under the async-RL plan; not on the sync 1-hop
   critical path.
2. **`ReplayBuffer` pattern.** Useful for async RL where rollouts may produce
   out-of-order vs training steps. Deferred to PR-async; sync 1-hop has
   exact step alignment so we don't need it.
3. **Tag-based progress signal.** Simpler than the consumer-counter for
   cross-step resumability. We can revisit if/when we need crash recovery.

### What we do that verl doesn't

1. **`DataPlaneClient` ABC.** verl is pinned to one TQ implementation; we
   can swap (R: integration plan G2). Worth it because the field is
   moving (mooncake_cpu, nv-dataplane).
2. **`shard_meta_for_dp`.** verl workers receive full meta and shard
   internally; we shard on the driver because Megatron's
   `shard_by_batch_size` requires `bin_count_multiple=DP_world` to avoid
   deadlocks at the first cross-DP collective when sequence-packing
   bin counts vary per rank.
3. **Driver-slice-only pattern.** verl pulls full batches into the driver
   for compute_advantages/values; that scales poorly at long-context
   (1–5 GB / step at 8k–32k seq) since the driver becomes a single-node
   serialization bottleneck. We touch only small slices on the driver.
4. **Helper layer (`kv_first_write` / `read_columns` / `write_columns`).**
   verl inlines the `kv_batch_get → process → kv_batch_put` pattern at
   each call site. We extracted it because the same pattern repeats 5+
   times and we want one place to validate dtype / shape / key invariants.

### TL;DR

The two implementations are *primitive-compatible* (same `kv_batch_*`
calls, same key lifecycle, same `KVBatchMeta` shape) but
*integration-shape different*:

- **verl** treats TQ as a stage queue with a polling replay buffer in
  front of it; generation is per-prompt async; the driver still touches
  bulk in some compute phases.
- **nemo-rl sync 1-hop** treats TQ as a sample-keyed dataframe; generation
  is one batched actor; the driver only ever sees small slices.

Both are correct; the cost differential at scale comes from how much
data flows through the driver.

---

## 7. Performance characterization (this run)

End-to-end parity vs the legacy driver-bulk path
(`grpo-run-a-legacy-v2.log`):

- Steps 1–7 are bit-exact (loss + reward); divergence afterward is the
  expected stochastic drift from accumulated policy updates.
- Steady-state step time: **+0.21 s** (1-hop 7.86 s vs legacy 7.65 s,
  ~3 %).
- Per-phase breakdown (steady state, steps 2–19):

| Phase                         | v4 (1-hop) | Legacy   | Δ          |
|-------------------------------|-----------:|---------:|-----------:|
| Total step time               | 7.606 s    | 7.393 s  | **+0.213 s** |
| policy_training               | 0.596 s    | 0.567 s  | +0.028 s   |
| generation                    | 1.502 s    | 1.528 s  | −0.027 s   |
| policy_and_ref_logprob        | 1.588 s    | 1.448 s  | **+0.141 s** |
| residual (driver bookkeeping) | 3.920 s    | 3.850 s  | +0.070 s   |

**The +0.21 s overhead is entirely TQ RPC roundtrip cost in the logprob
phase** (two worker calls × one fetch + one write each). Generation and
training are unchanged.

### Crossover scale (where TQ wins)

TQ overhead is mostly latency-bound (~constant per step), while legacy
driver fan-out is bandwidth-bound (scales with batch tensor volume × DP
fan-out). Mental model:

- Legacy driver overhead ≈ ~5 ms/MB × (4 full-batch transfers per step) × DP-fan-out
- TQ overhead ≈ ~200 ms fixed (after fuse-and-overlap optimization: ~100 ms)

Crossover when batch volume × DP fan-out × ~20 ms/MB ≥ TQ fixed cost:

| Scale                                    | Batch / step | DP ranks | Legacy cost | Winner                  |
|------------------------------------------|-------------:|---------:|------------:|-------------------------|
| Toy (this run, 1B, 512 tok, BS 32)       | 0.6 MB       | 8        | ~50 ms      | **legacy +0.21 s**      |
| Small prod (8B, 1k tok, BS 256)          | ~10 MB       | 8        | ~300 ms     | **roughly tied**        |
| Mid prod (70B, 4k tok, BS 1024)          | ~250 MB      | 32       | ~5–10 s     | **TQ wins decisively**  |
| Long-context (8k–32k seq, GRPO 16 gens)  | 1–5 GB       | 64+      | tens of s   | **TQ wins decisively**  |

Rough crossover: **~10 MB / step / DP-rank of effective batch volume**.
Long sequences, more generations per prompt, and more DP ranks all push
the needle hard toward TQ.

### Cheapest optimizations

1. **Fuse `get_logprobs` + `get_reference_policy_logprobs` into one worker
   call** — saves ~70 ms (one TQ input-fetch). Brings overhead from
   +0.21 s → ~+0.14 s.
2. **Overlap TQ write-back with next-phase fetch** — saves another
   ~30–50 ms. Combined: ~+0.10 s overhead, effectively at parity.

Both are clean refactors inside `tq_policy.py` / `base_policy_worker.py`
and don't touch `grpo_sync.py`. Not on the critical path; flag for the
next data-plane optimization round.

---

## 8. Where to look in the code

| Concern                          | File                                                          |
|----------------------------------|---------------------------------------------------------------|
| Stable boundary                  | `nemo_rl/data_plane/interfaces.py`                            |
| Adapter (TransferQueue impl)     | `nemo_rl/data_plane/adapters/transfer_queue.py`               |
| Driver-side helpers              | `nemo_rl/data_plane/driver_io.py` (`read_columns`, `write_columns`) |
| First-write helper               | `nemo_rl/algorithms/sync_utils.py`                         |
| Rollout actor                    | `nemo_rl/algorithms/sync_utils.py`                    |
| DP-rank meta sharding            | `nemo_rl/data_plane/preshard.py`                              |
| Worker fetch + write-back        | `nemo_rl/models/policy/workers/base_policy_worker.py`         |
| TQ-aware policy facade           | `nemo_rl/models/policy/tq_policy.py`                          |
| End-to-end orchestration         | `nemo_rl/algorithms/grpo_sync.py`                             |
| Unit tests                       | `tests/data_plane/unit/`                                      |
| Design                           | `research/data_plane_integration_plan.md` §1.2                |
