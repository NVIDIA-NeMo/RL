# Checkpoint210 real-rollout findings

Job21373254 completed0:0 on2026-11-06, elapsed53m32s. Image1030,
Qwen3.5-35B-A3B,16training GPUs/EP16 plus8rollout GPUs. Tested source commit
`c7ddf1035af145c6533a786991730890f8c773f3`; final source hashes matched.
The later curve commit5dd1a2735 changes documentation/artifacts only.
The original BF16 1% qualification remains **FAILED**. Diagnostic completion
is not a numerical qualification or a release gate.

## Inputs and independent checks

- Replay uses exactly32 unique responses/eight complete prompt groups from
  checkpoint210, one rollout weight version210,38,413valid response tokens.
  Captured stream210 has15,087nonzero advantage tensor entries and a nonzero
  update. Weights, optimizer, scheduler and RNG are restored before each case.
- All112per-rank reports (seven cases times16ranks) exist; global exact flags
  were recomputed from every report. See ep16-rank-manifest.json.
- Independent token-weighted K3 is0.0011291237654963173. Logged full K3 is
  0.0011291275454823335; split K3 is0.0011291275460311723. Absolute aggregation
  error versus the independent calculation is about3.8e-9; full/split difference
  is about5.5e-13. This batch supplies no evidence of a gen-KL aggregation defect.
- Native independent full/stream resumes use the same32IDs/prompt tokens, but
  only8/32complete responses match at version210 and0/32at211. Their native
  timing/KL differences are not paired same-token numerical evidence.

## Numerical boundary

| Control | Global preclip gradient relative L2 versus full | Other evidence |
| --- | --- | --- |
| Full repeat | 0 | Gradients, parameters, public/raw optimizer and scheduler exactly repeat on all16ranks. |
| Independent full-input numerator reference | 1.20855% | Moving only normalization reproduces the fixed-shard streaming difference. |
| Split while preserving DP microbatches | 1.20855% | Exactly matches numerator reference in gradients, parameters and public/raw optimizer on all16ranks; scheduler matches full. |
| Split then use normal production DP sharding | 30.6616% | Valid-token eval LP maximum absolute difference0.676001 versus full; model/public/raw optimizer differ. |
| Full with recomputed policy LP | 21.8206% | Compared with stored-LP full; the captured previous LP was computed in the native layout. |
| Production split with recomputed policy LP | 31.1277% | Versus recomputed full:20.2634%, cosine0.97951; scheduler still exact. |

The fixed-shard state path matches an independent ordinary-worker reference.
The initial production-placement discrepancy cannot be attributed only to
denominator rounding. Changing the microbatch/rank layout changes forward LPs
even before an update. The completed follow-up below narrows this to execution
layout rather than a30%intrinsic accumulation discrepancy; the responsible
operator/kernel has not been identified.

With recomputed LPs, full/split genKL is0.00111522995644/0.00111765878622.
This one-step difference does not explain the roughly1.86x final20-step KL
gap in the independent300-step trajectories. Post-update LP outliers reach
about4 in logprob units even in normalization-only controls; do not dismiss
model/update sensitivity as harmless because the average diagnostic is small.

## Completed production-layout follow-up

Job21391011 completed0:0,22m08s, on two interactive nodes/16H100 with image1030.
Executed source756c5eb77e5120d181f1a387bbd83d63ddcb4387; all16source hashes matched
at exit and were checked against that commit. Product worker, original parity
harness and replay.py were unchanged. Full-input calls now use the concatenation
of the two production chunk shards on each rank, preserving microbatch order.

The same32saved responses, input hash, weight version210 and38,413valid tokens
are reused. All16optimizer and scheduler counters were210 before the first
update; LR1e-6, BF16 raw moment buffers and compressed-state scales were present.
Every case/rank's actual fetched IDs and seven valid-token columns/dtypes were
checked; all use forward padding2365. All112new per-rank reports were verified.

| Matched-layout control | Observed result |
| --- | --- |
| Full repeat | Exact preclip gradients/parameters/public+raw optimizer/scheduler on all16ranks. |
| Fixed split versus independent numerator reference | Exact gradients/parameters/public+raw optimizer on all16ranks; scheduler agrees with full. |
| Production split versus full, stored previous LP | Gradient relative L2 **1.18555%**, cosine0.999929775; pre-update eval LP maximum difference0. |
| Production split versus full, recomputed previous LP in both | Gradient relative L2 **1.13169%**, cosine0.999935974; recomputed genKL0.00111765879842/0.00111765878622. |
| Fixed-workload worker medians,3alternating pairs | Full5.630319s, split8.732814s; about3.102494s extra backend time. |

The former30.66%stored-LP and20.26%recomputed-LP mismatches reduce to about1.19%
and1.13%when the full reference shares production microbatches. This supports an
execution-layout sensitivity in this model/backend, in addition to the independently
reproduced normalization-placement rounding. It is not evidence that the1%gate
passes: both matched-layout gradient comparisons still exceed it, and parameters
and raw optimizer state differ versus ordinary normalized full-input execution.

Direct exact raw-state comparisons against the numerator reference are recorded
for fixed split. Production split reports compare with ordinary full; exact
raw-state equality between the two split variants was not directly recorded.
Their input/order, aggregate gradient diagnostics and post-update LPs can be
checked separately without claiming that missing tensor comparison was run.

The roughly22.5%gradient difference of full_recompute versus stored-LP full changes
the previous-LP input. The matched-input pipeline comparison is
split_production_recompute versus full_recompute, whose difference is1.13169%.
The300-step trajectory gap has not been causally assigned to a specific operator,
normalization choice or their cumulative effect. This one checkpoint/batch also
does not qualify every model, seed, padding layout or recovery scenario.

Evidence: production-layout-summary.json, production-layout-rank-manifest.json
and production-layout-source-hashes.txt. CPU-only job21392643 completed0:0,
3m07s, using image1030. Its saved-tensor comparison found:

- Old production-split eval LP and new unified full eval LP agree exactly on
  all38,413valid tokens, across separate GPU allocations. In contrast, old full
  versus unified full differs on16,448tokens, max0.676001,p99absolute0.155548.
- Within the unified layout, full/split pre-update LP and recomputed full/split
  LP agree exactly on every valid token.
- Post-update numerator-reference versus fixed split, and fixed versus production
  split, also have exactly equal saved LPs. This is functional LP evidence, not
  a substitute for the unrecorded production-versus-numerator raw-state comparison.
- Changing normalization placement gives post-update LP max4.206896,p99absolute
  0.216075;1,945tokens exceed0.1 and3exceed1. These compare updated policies on
  the same **old** rollout and are not live next-step LA0 gen-KL measurements.

See production-layout-lp-comparison.json for complete statistics, source tensor
hashes and the comparison-script hash. No new inference/training was performed.

## Performance and overlap

- After numerical warmup, three alternating fixed-input timing pairs gave
  median5.452882s full and8.692046s split. The extra3.239164s includes repeated
  logprob/preparation/local-fetch/training/commit work; no per-component ablation
  was performed. Rollout and network Ray/TQ costs are outside this timer.
- Native warm stream/full measured131.964/129.464s on different response tokens;
  these are observations, not a fixed-workload speedup comparison.
- Warm ungated traces contain5,621training and4,405,259generation kernels.
  Cross-node timestamp intersection reports8,108intervals/26.77363ms. First
  training starts84.314s before the final generation kernel. An independent
  bound on host clock offset at capture time was not recorded, so exact small
  intersections remain provisional. The profiler synchronizes rank0 and trace
  export adds about220s to final weight-sync timing; do not use profiled step
  time as throughput evidence. No artificial admission gate/delay was used.
- The300-step curves are already refreshed through300 in commit5dd1a2735.
  Steps11..300 mean seconds/update are133.30streaming versus133.59full-batch LA0,
  while generated lengths differ. No end-to-end streaming speedup is established.

Evidence: ep16-summary.json, ep16-native-capture-check.json,
ep16-rank-manifest.json and ep16-source-hashes.txt. Raw captures, rank files,
logs and14GBgeneration trace remain under results/ep16-21373254. Keep protected
checkpoint210 until the placement follow-up no longer needs it. No old900,
MCore or sibling RL-pr4963 investigation was changed.
