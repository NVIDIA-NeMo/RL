# Gemma4 E2B IT streaming LA0: 900-step results

The same experiment completed 900 optimizer updates on 2026-11-09. Checkpoint
900, the finished W&B run, all 900 committed receipts, and node releases were
verified. This is exploratory training evidence, not numerical qualification.

- [W&B run](https://wandb.ai/yuekaiz/nemo-rl-automodel-sc/runs/yuekaiz5090gemmala0s42).
- [Eight-panel PNG](results/curve-comparison/comparison.png),
  [CSV](results/curve-comparison/comparison.csv), and
  [computed summary](results/curve-comparison/comparison-summary.json).
- [Final status](results/final900-status.json),
  [checkpoint scalar audit](results/checkpoint900-state.json),
  [online-history verification](results/final900-online-verification.json), and
  [Slurm/driver verification](results/final900-slurm-verification.json).

The generation KL diagnostic showed no sustained runaway through step 900. Its
first 50-step mean was 0.000390821, last 50-step mean 0.000554309, and last 20-step
mean 0.000540240. The global maximum remained 0.000959170 at step 392; the next
highest value was 0.000680204 at step 840. It settled around a higher level after
the early segment. This shape does not explain the backend discrepancy or prove
algorithm parity.

Training reward increased descriptively from a first 50-step mean of 0.365703 to
a last 50-step mean of 0.517282. These are sampled training rewards, not held-out
accuracy. Policy KL still had isolated spikes (maximum 2.414998 at step 93), and
preclip gradient norm reached 47.360508 at step 843, followed by 1.063994 at step
844. The AutoModel clipping threshold remained 1.0. The causes of these spikes
remain unresolved.

## Own-run windows

| Metric | First 50 (1–50) | Last 50 (851–900) | Last 20 (881–900) |
| --- | ---: | ---: | ---: |
| Reward | 0.365703 | 0.517282 | 0.523242 |
| Generation KL | 0.000390821 | 0.000554309 | 0.000540240 |
| Policy loss | 0.029499 | 0.021759 | 0.024637 |
| Generation tokens/sample | 1657.58 | 1318.67 | 1329.76 |
| Approximate entropy | 0.186954 | 0.170428 | 0.160490 |
| Preclip gradient norm | 0.468913 | 1.756438 | 2.380600 |
| Measured step time (s) | 73.89 | 69.62 | 70.58 |

Policy KL and gradient-norm axes are logarithmic. Raw points and trailing
10-step arithmetic means are shown. Timing includes measured checkpoint/cold-step
overhead, but excludes Slurm wait and model initialization. No controlled warmed
GPU overlap or fixed-workload speedup was measured for this Gemma long run.

## Existing baselines and shared coverage

The baselines are the PR author's existing batch-512 runs
[V1 `74860be5`](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/74860be5)
(actual 157 steps) and
[full-batch SC LA0 `96779513`](https://wandb.ai/ys_fishcool-nvidia/nemo-rl-automodel-sc/runs/96779513)
(actual 164 steps). No new baseline was started. Histories were refreshed
read-only for the final plot. Shared statistics stop at 157; the full-SC tail to
164 is retained, and absent coverage is shaded. Nothing is extrapolated to 900.

| Common steps 1–157 | V1 synchronous | Full-batch SC LA0 | Streaming LA0 |
| --- | ---: | ---: | ---: |
| Mean reward | 0.41849 | 0.41843 | 0.41544 |
| Generation tokens/sample | 1429.90 | 1424.12 | 1384.05 |
| Mean gen_kl_error | 0.000392377 | 0.000386466 | 0.000405584 |
| Mean step time, steps 11–157 (s) | 72.83 | 71.86 | 69.78 |

The time difference accompanies shorter streaming outputs, independent rollouts,
and historical runs with one seed. It does not isolate a streaming speedup or
establish convergence equivalence, especially after the baselines end.

## Evidence and limits

| Check | Result | Scope |
| --- | --- | --- |
| Native updates 1–900 | VERIFIED | Unique, contiguous receipts; step equals trainer version; all lag 0; all eight plotted metrics finite. |
| Streaming dispatches | VERIFIED | 899 updates used two dispatches; step 12 used one full-ready dispatch. Readiness 16 is not a fixed chunk cap. |
| Nonzero training signal | VERIFIED | Every update had nonzero reported gradients and advantages. Valid samples were 512 on 887 updates and 511 on 13. |
| Saved checkpoint 900 | VERIFIED SCALARS | Four optimizer shards; 505 optimizer counters at 900; scheduler/trainer/sampler at 900; LR 2e-6; 28,800 consumed prompts; epoch 0. |
| Same online run/history | VERIFIED | Finished at 900; unsampled steps 1–900 and ten training metrics exactly match native records. Rounded timing is excluded from exact comparison. |
| Resume receipt/LR/log continuity | VERIFIED | Boundaries 150→151, 340→341, 524→525, 709→710, and 893→894; later unsampled history boundaries and saved counters retained. |
| Recovery with unfinished work | OBSERVED | 64 unfinished groups redispatched before new data, alongside zero canonical groups. Independent all-group consumption/uniqueness and completed-canonical replay are not established here. |
| Gemma same-input gradient/parameter/raw optimizer parity | NOT ASSESSED | This run does not replace the fixed-input harness. |
| Earlier Qwen BF16 original 1% gate | FAILED, UNCHANGED | No threshold was relaxed. |
| Physical refit/off-policy tensors and all-rank raw-gradient oracle | NOT ASSESSED HERE | Version/logging consistency is not a same-token/same-weight tensor oracle. |
| Warmed GPU overlap/controlled throughput | NOT ASSESSED | Multiple dispatches and logged step times are insufficient. |
| Online validation/accuracy | NOT COLLECTED | No validation/acc panel or held-out score was fabricated. |

## Execution and provenance

The recipe stayed local `google/gemma-4-E2B-it`, one 8-H100 debug node at a time,
four training/FSDP2/EP1 plus four async-vLLM rollout GPUs, TP=CP=1, batch 512
(32 prompts × 16 responses), seed 42, training/logprob microbatch 1,
sequence/generation limits 4096/2048, LA0, in_order, readiness 16, buffer 32,
LR 2e-6, FusedAdam BF16 states, warmup 10 then ConstantLR, and image
`nemo_rl.1030.sqsh`. Checkpoint interval was 10, retention 2, and each segment's
training deadline 3h45m. The unchanged 1,791,700-row loader stayed at max epochs
1. Continuations changed only target steps/display name and restored optimizer,
scheduler, data cursor, rollout recovery state, and the same W&B identity.

| Allocation | Completed interval | Node | Driver exit | Slurm exit |
| --- | --- | --- | ---: | --- |
| 21546953 | 0→150 | pool0-00343 | 0 | 0:0 |
| 21557906 | 150→340 | pool0-00941 | 0 | 0:0 |
| 21568255 | 340→524 | pool0-00671 | 0 | 0:0 |
| 21579059 | 524→709 | pool0-01905 | 0 | 0:0 |
| 21594351 | 709→893 | pool0-00826 | 0 | 0:0 |
| 21607787 | 893→900 | pool0-01223 | 0 | 0:0 |

All six allocations were sequential and are absent from squeue. Slurm CANCELLED
records the expected release of idle interactive Ray allocations after successful
driver exits and durable progress. Direct attach was used; no batch-training
chain was submitted. A bounded release gate retained the final allocation for
read-only checks before resuming the supervisor. The final node and monitor
have exited. No unrelated jobs were changed.

Several actor shutdowns emitted ignored W&B service-close atexit BrokenPipeError
messages. Driver exits, checkpoints, and all 900 online rows were independently
verified. Warnings remain in original logs; no model, numerical code, dependency,
or image change was made to silence them.

Frozen numerical source and final analysis hashes:

- AutoModel worker: `7f59824258aafb13f5540810776dfa18c86a623856c9c89e4c6e26881259f74d`.
- Fixed-input harness: `ccb26bda2ad5a1408cbbd125cf2aec5a119d232ecaa1777bebbce0071c0dc399`.
- Final analysis: `9096768b5a22515c4831a3f8124648859a087aefe2b3d303d244c406b537334d`.

The PR3 development base remains PR1 revision `047d31a7`; newer PR1 revision
`097524718` was inspected but not rebased into this experiment. JSON records
include per-stage preflight commits, model-config and scalar metadata hashes,
and native log/history hashes. These checks do not establish checkpoint tensor
equivalence. See the broader [PR3 results](../RESULTS.md) for earlier
qualification evidence and restrictions.
