# TP=2 cascade-to-zero fix

Production failure (FT 30B, TP=2, cycle 2 burst kill of 2 shards):

```
↻ ensure_collective_synced [attempt 1/6]: world_size 7 → 1 (gen=0)   # cascade
↻ ensure_collective_synced [attempt 2-6/6]: world_size 1 → 1
RuntimeError: ensure_collective_synced failed to rendezvous after 6 attempts
```

Router went from N ready leaders to 0: `_evict_failed_workers_inline`
mass-evicted ALL surviving leaders.

## (a) Repro

Built a mock-based unit-test reproducer, NOT a 4-GPU GB300 pod. The bug
is purely a router-side **policy decision**; underlying NCCL mechanics
are already understood (TP=1 path runs 150+ steps cleanly with prior
fixes). The unit test
`test_evict_failed_workers_inline_skips_alive_survivors` reproduces the
cascade: 4 shards, 2 dead + 2 alive survivors, all 4 init_collective
futures fail. Without the fix → mass-eviction (world 8→0). With the
fix → only the 2 dead shards evicted (world 8→4).

## (b) Failure modes hit

At TP=2, surviving leaders fail with a MIX of:

- `DistNetworkError`/`DistStoreError` — TCPStore times out at 30s waiting for the dead shards' TP partners' rank-keys.
- `RayActorError` — dead shard's TP partner's `cudaErrorLaunchFailure` propagates back through vLLM's `collective_rpc`, killing the leader's actor process.
- `RuntimeError` — NCCL bootstrap timeout from the non-blocking `_poll_raw_async` path.

Mixed types → `_is_rendezvous_master_failure` returns False (requires
ALL failures in `{DistStoreError, DistNetworkError}`) → eviction runs →
positionally maps `failed_idxs → shard_ids` and removes everything,
including alive survivors.

## (c) Per-fix outcome

| Fix                                                                            | Outcome                                                                                                            |
| :----------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **A. Per-shard `is_alive` ping before eviction** (applied)                     | Robust at any TP/PP — fresh authoritative liveness signal, not a guess from exception types.                        |
| B. Treat mixed `RayActorError`+`DistNetworkError` as rendezvous-master failure | Heuristic-on-heuristic — fragile against future failure-type variations.                                            |
| C. Structural cascade detection (skip if all survivors fail simultaneously)    | Mis-fires when one survivor really IS poisoned alongside truly-dead shards: would leave the poisoned one in service. |
| D. Reduce rendezvous timeout                                                   | Orthogonal — speeds up detection but doesn't change policy.                                                         |

## (d) Applied fix

`nemo_rl/models/generation/generation_router.py` — added
`_filter_targets_by_liveness(candidate_sids)` and called it from
`_evict_failed_workers_inline` before eviction:

```python
async def _filter_targets_by_liveness(self, candidate_sids, probe_timeout_s=5.0):
    """Returns (dead_shard_ids, alive_shard_ids) by pinging each
    candidate's leader actor with is_alive (5s timeout)."""
    # asyncio.gather over per-shard probes:
    #   ray.get(leader.is_alive.remote(), timeout=5) → True  → alive
    #   ray.get raises (RayActorError / GetTimeoutError)     → dead

# In _evict_failed_workers_inline, after computing `targets`:
if targets:
    dead_targets, alive_targets = await self._filter_targets_by_liveness(targets)
    if alive_targets:
        print(f"[router] skipping {len(alive_targets)} alive survivors ...")
    targets = dead_targets
```

Behavior: TP=1 single kill unchanged (dead → evicted). TP=N burst:
dead → evicted, alive → preserved, train retries at smaller world.
True rendezvous-master timeout: `_is_rendezvous_master_failure`
short-circuits BEFORE probe, fast path preserved.

Unit tests added — all 3 PASS on the dev pod:
- `test_filter_targets_by_liveness_splits_dead_and_alive`
- `test_evict_failed_workers_inline_skips_alive_survivors` (the cascade)
- `test_evict_failed_workers_inline_dead_shard_evicts_normally` (TP=1 regression)

Files modified:
- `nemo_rl/models/generation/generation_router.py` (+109 lines)
- `tests/unit/models/generation/test_generation_router.py` (+~190 lines)

## (e) What's still uncovered

1. **No live GB300 end-to-end validation.** Fix is unit-test-validated against the policy logic. The user drives the live FT run separately.
2. **Genuinely poisoned alive survivor.** If a leader's CUDA context is wedged but its actor process is alive, `is_alive=True` keeps it in service and `ensure_collective_synced` fails all 6 attempts. Trade-off vs cascade-to-zero: run dies cleanly, gen cluster stays up, `--replace` recovers. Health-poll's `/openapi.json` probe will eventually cordon the wedged shard.
3. **Pre-existing test failures on this branch** (`test_5xx_cordons_and_replays`, `test_remove_shard_kills_actors_frees_pg_and_resets`) — unrelated to this fix.
4. **TP partner asymmetry.** Probe pings leader only; if TP partner is dead but leader is alive, shard not evicted. Mitigated by kai-scheduler atomic pod teardown + FI's `actor-kill` hitting `/admin/remove_shard` (kills all shard actors).
