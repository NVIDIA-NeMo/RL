# EAGLE3 TAR Drift A/B (5 Seeds) Runbook

This runbook launches GRPO TAR-drift A/B experiments on EAGLE3 with:
- 5 seeds
- `train` arm (verifier updates enabled)
- `frozen` arm (same setup, `lr=0.0`)

Generator script:
- `/home/scratch.shaunakj_other/Development/RL/tools/generate_eagle_tar_drift_slurm_submits.py`

Generated submit scripts are written under:
- `/home/scratch.shaunakj_other/Development/RL/slurm_submits/generated-<timestamp>/`

## 1) Generate Slurm submit scripts

```bash
cd /home/scratch.shaunakj_other/Development/RL

python3 tools/generate_eagle_tar_drift_slurm_submits.py \
  --dataset-split train \
  --partition <PARTITION> \
  --account <ACCOUNT> \
  --qos <QOS>
```

If your cluster does not require one of `partition/account/qos`, omit it.
`--dataset-split train` uses the full OpenMathInstruct-2 training split.

Get the newest generated directory:

```bash
GEN_DIR=$(ls -td /home/scratch.shaunakj_other/Development/RL/slurm_submits/generated-* | head -n1)
echo "$GEN_DIR"
```

## 2) Submit jobs

Submit all 5 seeds (`train` then `frozen` per seed by default):

```bash
cd "$GEN_DIR"
./submit_all.sh
```

Submit one seed manually:

```bash
cd "$GEN_DIR"
sbatch --parsable submit_eagle_tardrift_seed123.sbatch
```

Run only one arm for a seed:

```bash
cd "$GEN_DIR"
sbatch --parsable submit_eagle_tardrift_seed123.sbatch train
sbatch --parsable submit_eagle_tardrift_seed123.sbatch frozen
```

## 3) Optional smoke test (cheap)

```bash
cd "$GEN_DIR"
sbatch --parsable \
  --export=ALL,RUN_STEPS=1,MAX_EPOCHS=1,PROMPTS_PER_STEP=1,GENERATIONS_PER_PROMPT=1,SAVE_PERIOD=1,ARM_TIMEOUT=60m \
  submit_eagle_tardrift_seed123.sbatch frozen
```

## 4) If you hit vLLM ABI/runtime mismatch

If logs show vLLM import/ABI issues in frozen env, resubmit with:

```bash
cd "$GEN_DIR"
sbatch --parsable \
  --export=ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=0,RAY_ENABLE_UV_RUN_RUNTIME_ENV=1,NRL_FORCE_REBUILD_VENVS=true \
  submit_eagle_tardrift_seed123.sbatch frozen
```

## 5) Monitor progress

Check queue:

```bash
squeue --me
```

Check Slurm stdout for a job:

```bash
tail -f /home/scratch.shaunakj_other/logs/slurm-eagle-tardrift-s123-<JOBID>.out
```

Confirm GRPO actually started:

```bash
rg -n "Setting up data|Training dataset loaded|Ray cluster for policy initialized|Setting up model and training" \
  /home/scratch.shaunakj_other/logs/slurm-eagle-tardrift-s123-<JOBID>.out
```

## 6) Output locations

Per-arm run logs:
- `/home/scratch.shaunakj_other/logs/tar-drift-reasonable-eagle3-s4-seed<SEED>-<ARM>-steps<STEPS>-<TS>/run-<...>.log`

Per-arm checkpoints/results:
- `/home/scratch.shaunakj_other/results/tar-drift-reasonable-eagle3-s4-seed<SEED>-<ARM>-steps<STEPS>-<TS>/`

Per-seed status log:
- `/home/scratch.shaunakj_other/logs/tar-drift-reasonable-seed<SEED>-s<SPEC>-steps<STEPS>-<TS>/status.log`

## 7) Cancel jobs

Cancel one job:

```bash
scancel <JOBID>
```

Cancel all your active jobs:

```bash
scancel -u "$USER"
```

## Default profile in current generator

- `train_lr=5e-6`
- `frozen_lr=0.0`
- `max_epochs=1`
- `ratio_clip=0.2`
- `ref_kl=0.01`
- `max_grad_norm=1.0`
- `num_prompts_per_step=2`
- `num_generations_per_prompt=4`
- `temperature=0.6`
- `num_speculative_tokens=4`

## Path policy

Generated scripts are patched to avoid `/tmp` and `$SCR/tmp` for run artifacts:
- Ray temp root uses `$SCR/t/ray`
- local copied config uses `$SCR/t/...`
