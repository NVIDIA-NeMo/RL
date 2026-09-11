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

## Round 2

**Trial:** same-node interleaved A/B, `off/on/off/on`, 20 steps each, job 711125
on `nvl72d054-T18`, at PR head `b3fc916db` (so `wall_s` is the max reduction).

| run | step_s | sd | wall_s | ovh_ms | accounted | rows_checked |
|---|---|---|---|---|---|---|
| off | 14.27 | 4.00 | 0.182 | 3.3 | 186 ms | — |
| off2 | 13.62 | 3.97 | 0.177 | 3.4 | 181 ms | — |
| on | 14.26 | 2.84 | 0.249 | 117.2 | 366 ms | 2560 |
| on2 | 13.34 | 2.83 | 0.245 | 114.3 | 359 ms | 2560 |

- noise floor: off vs off2 = 0.645 s; on vs on2 = 0.926 s
- guard effect on step time: **-0.147 s** (guard-on *faster*, inside the floor)
- guard accounted cost: **+0.179 s** (wall +67 ms, self +112 ms)

**Verification: passes.** Hypothesis (1) step-time variance is confirmed — the
earlier 1.67 s was two different nodes against a 0.6-0.9 s floor and a 3-4 s
sd. Hypothesis (2) counters under-reporting is **not supported**: the accounted
180 ms is the whole measurable effect, and nothing appears in step time that
the instrument fails to bill.

**Defect review: passes.** The one finding from round 1 was my own misreading
(wall and self are disjoint, so the guard's cost is their sum ~180 ms, not
self alone); no code defect. Fixed in the docs, `db60af637`.

**Decisions:** no new checks surfaced. Simplification stays out of scope
(investigation). Superseded: every earlier cost figure in this doc and in the
README — 10.2 ms, 119 ms, 416 ms, 1.7 s. The number is ~180 ms/step accounted,
unmeasurable end to end.

**Verdict:** done — both selected checks clean against the latest change, the
set stabilised, and the conclusion rests on a run performed this round.
