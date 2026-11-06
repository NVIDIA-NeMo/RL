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
The larger production-placement discrepancy is a new unresolved numerical
finding: it cannot be attributed only to denominator rounding. Changing the
microbatch/rank layout changes forward LPs even before an update. This points
to a layout-dependent execution difference; it does not yet identify a faulty
MoE kernel, routing decision, input contract or cross-dispatch state path.

With recomputed LPs, full/split genKL is0.00111522995644/0.00111765878622.
This one-step difference does not explain the roughly1.86x final20-step KL
gap in the independent300-step trajectories. Post-update LP outliers reach
about4 in logprob units even in normalization-only controls; do not dismiss
model/update sensitivity as harmless because the average diagnostic is small.

Next control uses full-input ordinary/numerator references with the **same
production DP microbatches** as streaming. It reuses the saved capture and
checkpoint, validates actual fetched valid-token fields/dtypes, records initial
optimizer/scheduler counters, and changes no product code. If the production
split matches that numerator reference, the earlier placement difference also
exists without streaming. A mismatch needs a worker-state investigation.

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
