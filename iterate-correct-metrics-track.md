# iterate: ensure the correct metrics

**Mode:** fallback (no `--done` given). **Task type:** debug/investigation →
simplification check not selected.

**Question:** are `step/wall_s` and `step/self/overhead_ms` an accurate account
of what the data plane costs, or do they under-report? Under-reporting is the
unacceptable outcome — every number the PR publishes rests on them.

## Checks selected

- **Verification** — do the counters reconcile against independent measurement
  (`timing/train/*` phase timers, end-to-end `total_step_time`)?
- **Defect review** — is the accounting itself correct: what is billed where,
  and is any cost unbilled?

Not selected: simplification (investigation, not a code change); security
(no auth/input/network surface).

## Round 1

**Trial:** code audit of the billing split, while a same-node interleaved A/B
(`off/on/off/on`, 20 steps each, job 711125) runs.

**Finding — I have been quoting the wrong total.** The two counters are
*disjoint*, not nested:

- `_emit` → `stats.total_wall_ms += wall_ms` (`observability.py:1667`) — the
  inner RPC only.
- `_bill_self` → `self_ms += elapsed - _last_inner_ms` — wrapper time with the
  RPC subtracted out.

So the data plane's accounted cost is `wall + self`, and the guard's accounted
delta from the earlier (different-node) A/B is:

| | on | off | delta |
|---|---|---|---|
| `step/wall_s` | 1345 ms | 1048 ms | +297 |
| `step/self/overhead_ms` | 123 ms | 4.4 ms | +119 |
| **accounted total** | **1468 ms** | **1052 ms** | **+416** |

I had been reporting 119 ms as "the guard's cost". It is **~416 ms** by our own
counters. The README and the PR body both need this.

**Still unexplained:** step time moved 1670 ms; 416 ms is accounted. 1.25 s
outstanding — either variance or under-reporting. The same-node interleaved run
is what separates those.

**Dead-end (round 0, recorded so it is not retried):** attributing the gap to
mirror-column payload. `comm_volume_mb` unchanged (24.42 vs 24.58 MB) and the
arithmetic is 98 kB against 24 MB = 0.4%. Retracted in `b3fc916db`.

**Verdict:** iterate — verification open, waiting on job 711125.
