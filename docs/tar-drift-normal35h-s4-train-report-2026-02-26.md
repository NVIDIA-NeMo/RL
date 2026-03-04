# Train Arm Report (Normal 3.5h, Spec-4, Seed 124)

## Run identity

- Date: February 26, 2026 (PST)
- Mode: `train`
- Script: `/home/scratch.shaunakj_other/tmp/run-tar-normal35h-s4.sh`
- Status: completed successfully (`rc=0`)

## Timing

- Start: `2026-02-26 18:09:06 PST`
- End: `2026-02-26 21:23:15 PST`
- Wall-clock duration: `3h 14m 09s`

## Core config used

- `RUN_STEPS=135`
- `ARM_TIMEOUT=230m`
- `SPEC_TOKENS=4`
- `TRAIN_LR=1e-5`
- `MAX_EPOCHS=2`
- `PROMPTS_PER_STEP=2`
- `GENERATIONS_PER_PROMPT=4`
- `TEMP=0.6`
- Conservative constraints retained:
  - `loss_fn.ratio_clip_min=0.2`
  - `loss_fn.ratio_clip_max=0.2`
  - `loss_fn.reference_policy_kl_penalty=0.01`
  - `policy.max_grad_norm=1.0`

## Completion and outputs

- `status.log` contains:
  - `seed=124 arm=train rc=0`
  - `DONE mode=train`
- Per-step outputs written: `135` files (`train_data_step1.jsonl` ... `train_data_step135.jsonl`)

## Metrics summary (135 logged steps)

- TAR (`Spec Decode Token Acceptance Rate`)
  - Mean: `0.5951`
  - Early-20 mean: `0.5982`
  - Recent-20 mean: `0.6227`
  - Delta (recent - early): `+0.0245`
  - Latest: `0.4708`

- Reward
  - Mean: `0.7148`
  - Early-20 mean: `0.7000`
  - Recent-20 mean: `0.7312`
  - Delta (recent - early): `+0.0312`
  - Latest: `1.0000`

- Generation KL Error
  - Mean: `0.0012`
  - Early-20 mean: `0.0017`
  - Recent-20 mean: `0.0006`
  - Delta (recent - early): `-0.0011`
  - Latest: `0.0009`

- Step time (seconds)
  - Mean: `80.38s`
  - Early-20 mean: `83.11s`
  - Recent-20 mean: `75.48s`
  - Delta (recent - early): `-7.63s`
  - Latest: `76.35s`

## TAR trend note

- 5-step moving-average TAR was noisy but not persistently down.
- Full-run MA(5) linear slope was approximately flat (`-0.000045` per point).
- Recent-window MA(5) trend was up.

## End-of-run notes

- Run stopped at configured step budget (`Max number of steps has been reached`).
- Non-fatal shutdown warnings appeared (`.nfs` cleanup `OSError: [Errno 16]`, NCCL process-group warnings).
- These did not affect completion status (`rc=0`).

## Artifact paths

- Sweep status: `/home/scratch.shaunakj_other/logs/tar-drift-normal35h-seed124-s4-steps135-2026-02-26-180906/status.log`
- Train run log: `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0p6-s4-seed124-train-normal35h-steps135-2026-02-26-180906/run-135steps-seed124-train-normal35h.log`
- Train step data dir: `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0p6-s4-seed124-train-normal35h-steps135-2026-02-26-180906/exp_001`
