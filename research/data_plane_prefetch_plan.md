# Data-plane prefetch plan

**Status**: Exploration / parking lot. Not slated for current sync 1-hop landing.
**Owner**: zhiyul
**Date**: 2026-05-06

## TL;DR

The sync 1-hop trainer's per-step timeline has two TQ fetches that occur
**after** the heavy `policy.train_from_meta` call but read data unrelated
to the train result (`input_ids` for log_data jsonl, calibration slice
for `sync_kv_scales`). Both could be prefetched during the train window,
saving ~30-60 ms per step. Whether this is worth the API surface is
unclear; this doc captures the analysis so we can revisit after baseline
parity is established.

The right primitive is `concurrent.futures.ThreadPoolExecutor` at the
call site, **not** async on the `DataPlaneClient` ABC. Async was
explicitly dropped from the ABC after this analysis (see
`data_plane_integration_plan.md` §1.2 commit history).

## 1. Per-step timeline (grpo_sync.py, post-1-hop)

```
[rollout actor]                                  ~seconds (vLLM-bound)
  ↓ Ray return: meta + slice (small)
[driver: scale_rewards / shaping / overlong]     <1 ms
[driver: baseline/std]                           <1 ms
[driver: dynamic_sampling on slice]              <1 ms
[policy.get_logprobs_from_meta]                  ~hundreds of ms (worker)
[policy.get_reference_policy_logprobs_from_meta] ~hundreds of ms (worker)
[read_columns: generation_logprobs, token_mask]  ~10 ms TQ fetch
[masking + advantage compute]                    ~10-50 ms
[write_columns: advantages + sample_mask delta]  ~10 ms TQ put
[policy.train_from_meta]                         ~SECONDS  ◄─── long stretch
[read_columns: input_ids (for log_data jsonl)]   ~10 ms      ◄─── prefetchable
[read_columns: calib fields (sync_kv_scales)]    ~50 ms      ◄─── prefetchable
[policy.calibrate_qkv_fp8_scales]                ~100 ms
[kv_clear(meta.keys)]                            ~5 ms
```

The two boxed reads are independent of `train_from_meta`'s result.
They could begin before `train_from_meta` is called and complete during
its execution.

## 2. Why this isn't a load-bearing optimization

- Train step is the dominant cost (~95% of step wall time).
- The two prefetchable reads sum to ~60 ms.
- At ~5-second step times, that's ~1.2% wall-time saving.
- At ~30-second step times (large models), it's ~0.2%.

Worth doing if it's clean. Not worth API contortions.

## 3. Three design options

### A) `concurrent.futures.ThreadPoolExecutor` at the call site (RECOMMENDED)

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    log_fut = ex.submit(
        client.kv_batch_get,
        keys=meta.keys, partition_id=meta.partition_id,
        select_fields=["input_ids"],
    )
    calib_fut = ex.submit(
        client.kv_batch_get,
        keys=meta.keys, partition_id=meta.partition_id,
        select_fields=calib_fields,
    ) if sync_kv_scales else None

    train_results = policy.train_from_meta(meta, loss_fn=loss_fn, timer=timer)

    log_input_ids_td = log_fut.result()
    calib_td = calib_fut.result() if calib_fut else None
```

Pros:
- Trainer body stays sync. No `asyncio.run`, no `async def`.
- Zero new ABC surface — `kv_batch_get` is already sync.
- ThreadPoolExecutor is the idiomatic Python primitive for this pattern.
- Underlying `_tq.kv_batch_get` releases the GIL during the network wait,
  so the train thread is free to do its CPU work in parallel.

Cons:
- One ThreadPoolExecutor created per step (small but real overhead).
  Could keep a class-level pool to amortize.
- The trainer body grows by ~10 lines.

### B) Sync wrapper helper in `data_plane/driver_io.py`

```python
@contextmanager
def prefetch(dp_client, meta, *field_groups):
    """Submit one read_columns per field_group on a thread pool; yield
    a list of futures. Caller calls .result() when ready."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(field_groups)) as ex:
        futures = [
            ex.submit(read_columns, dp_client, meta, fields)
            for fields in field_groups
        ]
        yield futures

# Usage:
with prefetch(client, meta, ["input_ids"], calib_fields) as (log_fut, calib_fut):
    train_results = policy.train_from_meta(meta, ...)
    log_input_ids = log_fut.result()
    calib_data = calib_fut.result() if sync_kv_scales else None
```

Pros over A:
- Hides the threadpool plumbing.
- Caller sees declarative `with prefetch(...) as ...:`.

Cons:
- One more helper to maintain.
- Slightly less obvious than A — readers have to look up `prefetch`.

### C) async API on the ABC + `asyncio.gather` in trainer

```python
async def step_with_prefetch():
    log_fut = client.async_kv_batch_get(...)
    calib_fut = client.async_kv_batch_get(...)
    train_results = await asyncio.to_thread(policy.train_from_meta, meta, ...)
    log_input_ids, calib_td = await asyncio.gather(log_fut, calib_fut)
```

Pros:
- Composes with future async I/O (HTTP, async Ray, vLLM async engine).

Cons:
- Trainer body must become `async def`.
- `policy.train_from_meta` must be wrapped in `asyncio.to_thread` (it's
  CPU + Ray, not async-native).
- Adds `async_kv_batch_get` to the ABC — exactly the speculative API
  surface we just dropped.
- No actual benefit over A unless the caller already has other async
  I/O to gather with.

**Rejected** for the same reason `async_kv_batch_put` was dropped: YAGNI.

## 4. Open questions

### 4.1 Pool lifetime — per-step vs per-trainer

Per-step: `with ThreadPoolExecutor(max_workers=2) as ex:` creates and
shuts down a pool every step. Pool creation is ~ms; shutdown waits for
in-flight tasks. Probably fine.

Per-trainer: a single pool stored on the trainer scope, reused across
steps. Avoids creation cost. Need to manage cleanup at trainer exit.

Verdict: start per-step, measure, upgrade to per-trainer only if the pool
overhead shows up in profiling.

### 4.2 What else could be prefetched?

Currently only the two post-train reads are obvious prefetch candidates.
Other windows:

- **`get_logprobs` / `get_reference_policy_logprobs` in parallel**: both
  consume `meta.keys` + LP_SEED_FIELDS, both write back distinct columns
  (`prev_logprobs`, `reference_policy_logprobs`). Today they run
  sequentially. Could run concurrently if Ray dispatch supports it.
  Bigger change — touches `TQPolicy.get_*_from_meta`.
- **Driver delta-write + train**: `write_columns(advantages, sample_mask)`
  could fire-and-forget; train_from_meta doesn't read those columns
  itself (workers do, post-fetch). But workers fetch right at the start
  of `train_presharded`, so the put MUST complete before workers start.
  No room to overlap unless we add explicit ordering signaling.
- **Cross-step `kv_clear`**: at end of step N, clear is fire-and-forget;
  step N+1's rollout doesn't depend on the clear (uids are uuid4, no
  collisions). Saves ~5 ms/step. Trivial.

### 4.3 Pool size

For the two prefetch reads after train: `max_workers=2`. If we extend
to 3-4 prefetches per step (cross-step clear, parallel logprobs),
`max_workers=4` is enough. The default thread-pool ceiling (~32) is
plenty.

### 4.4 Error handling

A prefetch that errors: `future.result()` re-raises on the main thread.
Same semantics as the sync call. Good — no special handling needed.

A prefetch whose result is never claimed (caller takes a different
branch): the thread completes, the future GC'd. No leak.

### 4.5 Interaction with `kv_clear` at step end

If we prefetch `input_ids` for log_data and the kv_clear runs in
parallel (cross-step optimization), there's a race: the clear could
remove keys before the prefetch reads them. Today both happen serially
after train, so no risk. If we ever parallelize them, need explicit
ordering — but that's a bigger refactor.

## 5. When to land this

**Don't land yet.** Order of priorities:

1. Sync 1-hop parity tests pass. (PR-D — the only remaining piece.)
2. Profile a real GRPO run and see whether the post-train TQ reads
   actually show up in step-time breakdown.
3. **Only if** they do, land Option A (the inline ThreadPoolExecutor
   pattern). ~10 LoC in `grpo_sync.py`.
4. If multiple call sites end up using the same pattern, extract Option B
   (`prefetch` context manager helper).

The whole thing is a 1-2% wall-time optimization. Not worth touching
until baseline numbers are settled.

## 6. What to NOT do

- **Don't add `async_kv_batch_get` / `async_kv_batch_put` to the ABC.**
  This was explicitly dropped after the sync 1-hop refactor for YAGNI
  reasons. Re-adding speculatively for prefetch would re-introduce the
  async-without-await footgun the dual-API split was meant to eliminate.
- **Don't make `grpo_train_sync` `async def`.** The trainer is a sync
  pipeline; mixing in async would force every caller boundary to either
  `await` or `asyncio.run`, defeating the readability win.
- **Don't put the threadpool in `DataPlaneClient`.** The pool lives where
  the concurrency lives — which is the trainer's call site. Adapters
  stay synchronous and stateless w.r.t. concurrency.

## 7. References

- `nemo_rl/algorithms/grpo_sync.py` — main consumer, has the
  prefetchable read sites.
- `nemo_rl/data_plane/driver_io.py` — would host Option B's `prefetch`
  helper.
- `data_plane_integration_plan.md` §1.2 — sync vs async API decision
  history.
- `concurrent.futures.ThreadPoolExecutor` — stdlib primitive of choice.
