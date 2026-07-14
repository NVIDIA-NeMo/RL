# Refit communicator shrink scaling results

Date: 2026-07-13

## Outcome

The communicator shrink demo completed successfully at 32, 64, 128, 256, 512,
and 1,024 total GPUs. Each run used an even training/inference split and removed
the two ranks belonging to generation instance 3 before the step 6 refit. All
runs completed ten GRPO steps without a post-shrink NCCL hang.

## Metric definitions

- **Pre-shrink `weight_sync`**: mean of steps 1-5.
- **Transition `weight_sync`**: step 6, the first refit after shrinking the
  communicator.
- **Steady post-shrink `weight_sync`**: mean of steps 7-10.
- **Fault Handling Time**: the step 6 timer around
  `adjust_refit_comm_group(...)`.

## Results

| Total GPUs | Train / inference | New communicator size | Excluded ranks | Fault Handling Time (s) | Pre-shrink `weight_sync` mean (s) | Transition `weight_sync` (s) | Steady post-shrink mean (s) | Steady delta |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 / 16 | 30 | 22, 23 | 1.53 | 2.12 | 3.65 | 2.17 | +0.05 (+2.1%) |
| 64 | 32 / 32 | 62 | 38, 39 | 1.38 | 2.06 | 3.99 | 2.08 | +0.01 (+0.7%) |
| 128 | 64 / 64 | 126 | 70, 71 | 3.48 | 2.07 | 3.80 | 2.15 | +0.08 (+3.6%) |
| 256 | 128 / 128 | 254 | 134, 135 | 1.67 | 2.18 | 3.89 | 2.20 | +0.02 (+0.8%) |
| 512 | 256 / 256 | 510 | 262, 263 | 2.03 | 2.29 | 4.19 | 2.41 | +0.11 (+4.9%) |
| 1,024 | 512 / 512 | 1,022 | 518, 519 | 3.69 | 2.45 | 4.61 | 2.64 | +0.19 (+7.9%) |

## Raw `weight_sync` values

The shrink occurs before the step 6 refit.

| Total GPUs | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 | Step 7 | Step 8 | Step 9 | Step 10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2.17 | 2.02 | 2.08 | 2.33 | 2.01 | 3.65 | 2.12 | 2.10 | 2.16 | 2.29 |
| 64 | 2.00 | 2.13 | 2.03 | 2.11 | 2.05 | 3.99 | 2.04 | 2.21 | 2.06 | 2.00 |
| 128 | 2.02 | 2.04 | 2.28 | 1.99 | 2.03 | 3.80 | 2.05 | 2.03 | 2.32 | 2.19 |
| 256 | 2.27 | 2.06 | 2.30 | 2.07 | 2.22 | 3.89 | 2.31 | 2.16 | 2.19 | 2.15 |
| 512 | 2.60 | 2.13 | 2.00 | 2.40 | 2.34 | 4.19 | 2.29 | 2.59 | 2.24 | 2.51 |
| 1,024 | 2.72 | 2.46 | 2.29 | 2.44 | 2.34 | 4.61 | 2.58 | 2.91 | 2.26 | 2.82 |

## SLURM jobs

| Total GPUs | Job ID | Nodes | State | Exit code | Elapsed |
| ---: | ---: | ---: | --- | ---: | ---: |
| 32 | 13926534 | 4 | COMPLETED | 0:0 | 00:09:07 |
| 64 | 13927148 | 8 | COMPLETED | 0:0 | 00:11:36 |
| 128 | 13927167 | 16 | COMPLETED | 0:0 | 00:13:30 |
| 256 | 13927169 | 32 | COMPLETED | 0:0 | 00:13:38 |
| 512 | 13927173 | 64 | COMPLETED | 0:0 | 00:18:10 |
| 1,024 | 13927182 | 128 | COMPLETED | 0:0 | 00:23:54 |

## Run configuration and interpretation

- Eight GPUs per node and an even training/inference GPU split.
- vLLM tensor parallel size 2, so excluding one generation instance removes two
  communicator ranks.
- Global batch size equals the total GPU count, with four generations per
  prompt. This keeps the batch four times larger than training data parallel
  size across scales.
- Ten steps, fault injected after step 5, checkpointing disabled, and W&B
  disabled.
- The 64-1,024 GPU jobs were submitted without dependencies and overlapped in
  time. Shared-fabric contention may therefore contribute to variance.

These are single-run demo measurements, so the non-monotonic fault handling
times should not be treated as a scaling curve. The observed shrink duration
was 1.38-3.69 seconds. The first post-shrink refit paid a one-time transition
cost, while later refits returned close to their pre-shrink timing at every
tested scale.
