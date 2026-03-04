# Session Memory (2026-02-27): Restartable Normal 3.5h Spec-4 Run Script

## What was decided

- Goal: observe spec-drift with moderate update pressure (not aggressive).
- Budget: approximately 3.5 hours per arm.
- Speculative window: `num_speculative_tokens=4`.
- Run order must be `train` first, then `frozen`.

## Script saved

- Script path:
  `/home/scratch.shaunakj_other/tmp/run-tar-normal35h-s4.sh`
- Default mode: `train`
- Supported modes: `train | frozen | both`
- In `both` mode, execution order is:
  1. `train`
  2. `frozen`

## Default run profile in script

- `RUN_STEPS=135`
- `ARM_TIMEOUT=3h25m`
- `SPEC_TOKENS=4`
- `SEED=124`
- `TRAIN_LR=1e-5`
- `FROZEN_LR=0.0`
- `MAX_EPOCHS=2`
- `PROMPTS_PER_STEP=2`
- `GENERATIONS_PER_PROMPT=4`
- `TEMP=0.6`
- Conservative regularization retained:
  - `loss_fn.ratio_clip_min=0.2`
  - `loss_fn.ratio_clip_max=0.2`
  - `loss_fn.reference_policy_kl_penalty=0.01`
  - `policy.max_grad_norm=1.0`

## How to run after cluster restart

Run train arm first:

```bash
bash /home/scratch.shaunakj_other/tmp/run-tar-normal35h-s4.sh train
```

Then run frozen arm:

```bash
bash /home/scratch.shaunakj_other/tmp/run-tar-normal35h-s4.sh frozen
```

Or run both sequentially (train then frozen):

```bash
bash /home/scratch.shaunakj_other/tmp/run-tar-normal35h-s4.sh both
```

## Logging/output conventions

- Sweep status log:
  `/home/scratch.shaunakj_other/logs/tar-drift-normal35h-seed<SEED>-s4-steps<RUN_STEPS>-<timestamp>/status.log`
- Per-arm run logs (under each arm logroot):
  `run-<RUN_STEPS>steps-seed<SEED>-<arm>-normal35h.log`
- Each arm writes an `rc=<code>` line to `status.log` on completion/timeout.
