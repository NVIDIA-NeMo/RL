# Gemma4 E2B IT streaming LA0: 900 updates completed

The same experiment completed 900 updates on 2026-11-09. Checkpoint 900, the
finished W&B run, all 900 native/online training records, and node releases were
verified. All drivers exited successfully; the serial debug allocations and
monitor have ended. This is exploratory evidence: Gemma fixed-input streaming
parity is **NOT ASSESSED**, and the earlier Qwen BF16 original 1% gate remains
**FAILED**.

[900-step report](RESULTS900.md) | [Eight-panel PNG](results/curve-comparison/comparison.png) |
[CSV](results/curve-comparison/comparison.csv) | [Computed summary](results/curve-comparison/comparison-summary.json)

`gen_kl_error` showed no sustained runaway through step 900. Its first 50-step
mean was 0.000390821, last 50-step mean 0.000554309, and maximum 0.000959170 at
step 392. Policy KL still spiked (maximum 2.414998 at step 93), and preclip
gradient norm reached 47.360508 at step 843 before returning lower. The clipping
threshold remained 1.0; the causes of these spikes remain unresolved. Training
reward increased from a first 50-step mean of 0.365703 to a last 50-step mean of
0.517282. This is sampled training reward, not validation accuracy.

The eight panels show reward, generation length, generation/policy KL, policy
loss, approximate entropy, preclip gradient norm, and step time. Raw points and
trailing 10-step means are retained. Policy KL and gradient norm use logarithmic
axes. No validation/acc was collected.

The existing PR-author baselines end at steps 157 (V1, `74860be5`) and 164
(full-batch SC, `96779513`). No new baseline was trained. Shared statistics stop
at 157; actual tails and shaded missing coverage are shown without extrapolation.

| Common steps 1–157 | V1 synchronous | Full-batch SC LA0 | Streaming LA0 |
| --- | ---: | ---: | ---: |
| Mean reward | 0.41849 | 0.41843 | 0.41544 |
| Generation tokens/sample | 1429.90 | 1424.12 | 1384.05 |
| Mean gen_kl_error | 0.000392377 | 0.000386466 | 0.000405584 |
| Mean step time, steps 11–157 (s) | 72.83 | 71.86 | 69.78 |

Shorter outputs, independent rollouts, and historical runs prevent a controlled
speedup claim from these times. Multiple dispatches alone do not prove GPU overlap.

Verified runtime evidence:

- 899 updates used two dispatches; step 12 used one. All 900 updates had nonzero
  reported gradients/advantages, lag 0, and finite plotted metrics. Valid sample
  counts were 512 on 887 updates and 511 on 13 updates.
- [Checkpoint audit](results/checkpoint900-state.json): four optimizer shards,
  505 optimizer counters at 900, scheduler/trainer/sampler at 900, LR 2e-6, and
  28,800 consumed prompts. These are saved scalar checks.
- [Online verification](results/final900-online-verification.json): all 900
  unsampled steps and ten training metrics exactly match native records.
- [Slurm verification](results/final900-slurm-verification.json): six sequential
  allocations, successful driver/Slurm exit codes, and no owned allocation left.

Recoveries 150→151, 340→341, 524→525, 709→710, and 893→894 preserved observed
version, LR, and same-run logging continuity. Later runs explicitly redispatched
64 unfinished groups before new data, alongside zero canonical groups.
Independent all-group consumption/uniqueness, completed-canonical recovery,
physical refit/off-policy tensor identity, and all-rank raw gradient/optimizer
parity remain unverified here.

The recipe stayed Gemma4 E2B IT, one 8-H100 node at a time, four training/FSDP2/EP1
plus four async-vLLM rollout GPUs, TP=CP=1, batch 512 (32×16), seed 42, microbatch
1, sequence/generation limits 4096/2048, LA0, readiness 16, buffer 32, LR 2e-6,
FusedAdam BF16 states, warmup 10 then ConstantLR, and image `nemo_rl.1030.sqsh`.
See [RESUME900.md](RESUME900.md), per-stage provenance, and
[final status](results/final900-status.json).

W&B service-close atexit warnings were retained; checkpoints and all online rows
were independently confirmed. No numerical code or dependency changed for them.
The experiment is complete: stale monitor notices must not start more work.
