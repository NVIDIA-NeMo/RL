# TAR Drift A/B Report (Baseline vs Aggressive)

Date: February 25, 2026

## Goal

Measure whether policy updates change speculative decode Token Acceptance Rate (TAR) when the draft model is fixed.

The experiment design is A/B within each setup:

1. `train` arm: policy updates enabled.
2. `frozen` arm: policy learning rate set to `0.0` (control).

If `train` diverges from `frozen`, the effect is from policy updates, not just prompt/sampling noise.

## Setup 1: Baseline A/B (low-pressure updates)

Run group:

- `exp_root=/home/scratch.shaunakj_other/logs/tar-drift-ab-temp1-s2-steps30-2026-02-24-221251`

Arms:

- `train`: `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-train-steps30-2026-02-24-221251`
- `frozen`: `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-frozen-steps30-2026-02-24-221251`

Key behavior:

- Default GRPO recipe behavior (warmup scheduler, default clipping, default reference KL penalty).
- Control arm uses `++policy.optimizer.kwargs.lr=0.0`.

## Setup 2: Aggressive A/B (higher update pressure)

Run group:

- `exp_root=/home/scratch.shaunakj_other/logs/tar-drift-ab-aggr2-temp1-s2-steps30-2026-02-25-015509`

Arms:

- `train`: `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-train-aggr2-steps30-2026-02-25-015509`
- `frozen`: `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-frozen-aggr2-steps30-2026-02-25-015509`

Aggressive overrides (both arms unless noted):

- `++grpo.max_num_epochs=4`
- `++loss_fn.ratio_clip_min=0.5`
- `++loss_fn.ratio_clip_max=0.5`
- `++loss_fn.reference_policy_kl_penalty=0.0`
- `++policy.max_grad_norm=5.0`
- `train` only: `++policy.optimizer.kwargs.lr=1e-4`
- `frozen` only: `++policy.optimizer.kwargs.lr=0.0`

## Quantitative Summary

All values are from TensorBoard event summaries for 30 steps.

| Run | TAR first | TAR last | TAR slope | TAR std | KL mean | KL last | LR first | LR last |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline train | 0.6537 | 0.6259 | 0.000002 | 0.0461 | 0.000988 | 0.001001 | 0.00000050 | 0.00000311 |
| baseline frozen | 0.6536 | 0.6338 | -0.000042 | 0.0430 | 0.000989 | 0.001121 | 0.00000000 | 0.00000000 |
| aggressive train | 0.6537 | 0.9484 | 0.013026 | 0.1325 | 0.000544 | 0.000128 | 0.00001000 | 0.00006220 |
| aggressive frozen | 0.6537 | 0.6325 | -0.000044 | 0.0430 | 0.000989 | 0.001121 | 0.00000000 | 0.00000000 |

## Intuition

1. Why the baseline setup looked "flat":
   - The train and frozen arms are almost identical in TAR trend.
   - This indicates noise-dominated behavior (prompt mix, temperature=1 sampling, small effective updates), not strong policy-movement effects.

2. Why aggressive setup separates clearly:
   - Train LR and epoch pressure are much higher, while frozen remains at LR=0.
   - Under this condition, train TAR departs strongly from frozen TAR, so the update mechanism is now measurably affecting acceptance behavior.

3. How to interpret the TAR increase in aggressive train:
   - With fixed draft, higher TAR means target next-token choices became more aligned with what the draft proposes on this rollout distribution.
   - This is not automatically "better model quality"; it is specifically stronger target-draft agreement under the sampled trajectories.

4. On "draft drift":
   - Draft weights are fixed in both setups.
   - The thing that changes is target policy behavior relative to a fixed draft, observed indirectly via TAR.

## Reproduction Steps

### A. Baseline A/B

1. Launch baseline script:

```bash
bash /home/scratch.shaunakj_other/tmp/run-tar-ab-exp-2026-02-24-220952.sh
```

2. Track status:

```bash
tail -f /home/scratch.shaunakj_other/logs/tar-drift-ab-temp1-s2-steps30-2026-02-24-221251/status.log
```

3. Inspect per-arm logs:

```bash
# train
tail -f /home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-train-steps30-2026-02-24-221251/run-30steps-train.log

# frozen
tail -f /home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-frozen-steps30-2026-02-24-221251/run-30steps-frozen.log
```

### B. Aggressive A/B

1. Launch aggressive script:

```bash
bash /home/scratch.shaunakj_other/tmp/run-tar-ab-exp-aggr2-2026-02-25-015420.sh
```

2. Track status:

```bash
tail -f /home/scratch.shaunakj_other/logs/tar-drift-ab-aggr2-temp1-s2-steps30-2026-02-25-015509/status.log
```

3. Inspect per-arm logs:

```bash
# train
tail -f /home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-train-aggr2-steps30-2026-02-25-015509/run-30steps-train-aggr2.log

# frozen
tail -f /home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-frozen-aggr2-steps30-2026-02-25-015509/run-30steps-frozen-aggr2.log
```

### C. Quick Summary Extraction (optional)

```bash
.venv/bin/python - <<'PY'
from tensorboard.backend.event_processing import event_accumulator
import glob, os, numpy as np

runs = {
    'baseline_train': '/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-train-steps30-2026-02-24-221251',
    'baseline_frozen': '/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-frozen-steps30-2026-02-24-221251',
    'aggr2_train': '/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-train-aggr2-steps30-2026-02-25-015509',
    'aggr2_frozen': '/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp1-s2-frozen-aggr2-steps30-2026-02-25-015509',
}

for name, run_dir in runs.items():
    ev = sorted(glob.glob(os.path.join(run_dir, 'exp_001', 'tensorboard', 'events.out.tfevents.*')))
    ea = event_accumulator.EventAccumulator(ev[-1]); ea.Reload()
    tar = ea.Scalars('performance/token_acceptance_rate')
    kl = ea.Scalars('train/policy_kl_error')
    lr = ea.Scalars('train/lr')

    steps = np.array([x.step for x in tar], float)
    vals = np.array([x.value for x in tar], float)
    slope = np.polyfit(steps, vals, 1)[0]

    print(
        name,
        f"n={len(vals)}",
        f"tar_first={vals[0]:.4f}",
        f"tar_last={vals[-1]:.4f}",
        f"tar_slope={slope:.6f}",
        f"kl_mean={np.mean([x.value for x in kl]):.6f}",
        f"kl_last={kl[-1].value:.6f}",
        f"lr_first={lr[0].value:.8f}",
        f"lr_last={lr[-1].value:.8f}",
    )
PY
```

## Why Spec TAR Goes Up When It Does

`Spec Decode Token Acceptance Rate (TAR)` is a compatibility metric between:

- draft model proposals, and
- target model verification.

When TAR rises, it means the target is agreeing with draft proposals more often on that step's sampled rollouts.

Key intuition from these runs:

1. TAR is not the same as reward.
   - In aggressive train, correlation between step-level `Avg Reward` and TAR is weak (`corr ~= 0.072`), so TAR can rise even when reward is low.
   - Example: aggressive train step 23 has `Avg Reward=0.2500` but `Spec TAR=0.9184`.

2. TAR is sensitive to rollout mix and response shape.
   - Across all four runs, TAR has a negative correlation with proposed token count (`corr ~= -0.42`), i.e., steps with fewer/easier continuation points tend to have higher acceptance.

3. Aggressive updates changed target-draft agreement dynamics.
   - Baseline train and baseline frozen remain close in TAR behavior.
   - Aggressive train separates sharply from aggressive frozen, showing this is update-driven behavior (not only sampling noise).

4. Why the late sharp rise in aggressive train (around steps 16-30).
   - The aggressive profile uses higher update pressure (`lr=1e-4` schedule, `max_num_epochs=4`, looser clip bounds, zero ref-KL penalty).
   - As updates accumulate, the target policy moves into a regime where draft proposals are accepted more often on sampled prompts, producing sustained high TAR values.

Takeaway:

- TAR increases when the target becomes more draft-compatible on the sampled trajectory distribution.
- This does not automatically imply better absolute quality; it specifically indicates stronger draft-target alignment under those rollouts.

## Per-Step Avg Reward and Spec TAR

### Baseline / Train

| Step | Avg Reward | Spec TAR |
|---:|---:|---:|
| 1 | 0.7500 | 0.6537 |
| 2 | 0.7500 | 0.6971 |
| 3 | 0.5000 | 0.6776 |
| 4 | 0.5000 | 0.6527 |
| 5 | 0.8750 | 0.7191 |
| 6 | 0.5000 | 0.7266 |
| 7 | 0.5000 | 0.6273 |
| 8 | 1.0000 | 0.7674 |
| 9 | 0.7500 | 0.7716 |
| 10 | 0.0000 | 0.7054 |
| 11 | 1.0000 | 0.6666 |
| 12 | 0.5000 | 0.6237 |
| 13 | 0.6250 | 0.7259 |
| 14 | 0.5000 | 0.7539 |
| 15 | 1.0000 | 0.7468 |
| 16 | 0.7500 | 0.7043 |
| 17 | 1.0000 | 0.7030 |
| 18 | 0.8750 | 0.6877 |
| 19 | 0.0000 | 0.6690 |
| 20 | 1.0000 | 0.7306 |
| 21 | 0.8750 | 0.7776 |
| 22 | 1.0000 | 0.7679 |
| 23 | 1.0000 | 0.6920 |
| 24 | 1.0000 | 0.7339 |
| 25 | 1.0000 | 0.7635 |
| 26 | 0.5000 | 0.6558 |
| 27 | 1.0000 | 0.6998 |
| 28 | 0.2500 | 0.6851 |
| 29 | 1.0000 | 0.6319 |
| 30 | 0.7500 | 0.6259 |

### Baseline / Frozen

| Step | Avg Reward | Spec TAR |
|---:|---:|---:|
| 1 | 0.7500 | 0.6536 |
| 2 | 0.6250 | 0.7043 |
| 3 | 0.6250 | 0.6853 |
| 4 | 0.5000 | 0.6495 |
| 5 | 0.8750 | 0.7347 |
| 6 | 0.5000 | 0.7082 |
| 7 | 0.5000 | 0.6500 |
| 8 | 1.0000 | 0.7690 |
| 9 | 0.8750 | 0.7415 |
| 10 | 0.0000 | 0.6948 |
| 11 | 0.8750 | 0.6619 |
| 12 | 0.6250 | 0.6348 |
| 13 | 0.6250 | 0.7223 |
| 14 | 0.5000 | 0.7626 |
| 15 | 1.0000 | 0.7348 |
| 16 | 0.7500 | 0.7230 |
| 17 | 1.0000 | 0.7078 |
| 18 | 0.8750 | 0.6817 |
| 19 | 0.0000 | 0.6558 |
| 20 | 1.0000 | 0.7232 |
| 21 | 1.0000 | 0.7887 |
| 22 | 1.0000 | 0.7565 |
| 23 | 1.0000 | 0.6864 |
| 24 | 1.0000 | 0.7370 |
| 25 | 1.0000 | 0.7487 |
| 26 | 0.5000 | 0.6499 |
| 27 | 1.0000 | 0.7092 |
| 28 | 0.0000 | 0.6800 |
| 29 | 0.8750 | 0.6438 |
| 30 | 0.6250 | 0.6338 |

### Aggressive / Train

| Step | Avg Reward | Spec TAR |
|---:|---:|---:|
| 1 | 0.7500 | 0.6537 |
| 2 | 0.7500 | 0.7070 |
| 3 | 0.6250 | 0.6744 |
| 4 | 0.6250 | 0.6209 |
| 5 | 0.5000 | 0.6981 |
| 6 | 0.5000 | 0.6994 |
| 7 | 0.5000 | 0.6016 |
| 8 | 1.0000 | 0.6977 |
| 9 | 0.3750 | 0.7216 |
| 10 | 0.0000 | 0.7228 |
| 11 | 0.0000 | 0.5428 |
| 12 | 0.0000 | 0.5600 |
| 13 | 0.1250 | 0.5883 |
| 14 | 0.5000 | 0.7106 |
| 15 | 1.0000 | 0.7630 |
| 16 | 0.5000 | 0.8547 |
| 17 | 0.7500 | 0.8543 |
| 18 | 0.7500 | 0.8564 |
| 19 | 0.1250 | 0.8879 |
| 20 | 0.6250 | 0.7966 |
| 21 | 0.8750 | 0.8340 |
| 22 | 1.0000 | 0.9147 |
| 23 | 0.2500 | 0.9184 |
| 24 | 0.1250 | 0.9167 |
| 25 | 0.6250 | 0.9691 |
| 26 | 0.2500 | 0.9512 |
| 27 | 0.5000 | 0.9610 |
| 28 | 0.0000 | 0.9353 |
| 29 | 0.6250 | 0.9432 |
| 30 | 0.2500 | 0.9484 |

### Aggressive / Frozen

| Step | Avg Reward | Spec TAR |
|---:|---:|---:|
| 1 | 0.7500 | 0.6537 |
| 2 | 0.6250 | 0.7039 |
| 3 | 0.6250 | 0.6848 |
| 4 | 0.5000 | 0.6485 |
| 5 | 0.8750 | 0.7335 |
| 6 | 0.5000 | 0.7091 |
| 7 | 0.5000 | 0.6493 |
| 8 | 1.0000 | 0.7681 |
| 9 | 0.8750 | 0.7407 |
| 10 | 0.0000 | 0.6956 |
| 11 | 0.8750 | 0.6619 |
| 12 | 0.6250 | 0.6355 |
| 13 | 0.6250 | 0.7227 |
| 14 | 0.5000 | 0.7629 |
| 15 | 1.0000 | 0.7348 |
| 16 | 0.7500 | 0.7232 |
| 17 | 1.0000 | 0.7087 |
| 18 | 0.8750 | 0.6814 |
| 19 | 0.0000 | 0.6555 |
| 20 | 1.0000 | 0.7225 |
| 21 | 1.0000 | 0.7878 |
| 22 | 1.0000 | 0.7573 |
| 23 | 1.0000 | 0.6854 |
| 24 | 1.0000 | 0.7376 |
| 25 | 1.0000 | 0.7474 |
| 26 | 0.5000 | 0.6501 |
| 27 | 1.0000 | 0.7085 |
| 28 | 0.0000 | 0.6797 |
| 29 | 0.8750 | 0.6443 |
| 30 | 0.6250 | 0.6325 |
