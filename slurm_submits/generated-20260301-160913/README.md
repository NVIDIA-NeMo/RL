# Tar-Drift Reasonable EAGLE3 Study Runbook

This directory contains Slurm submit scripts for the study.

## 1) Move to this directory

```bash
cd /home/scratch.shaunakj_other/Development/RL/slurm_submits/generated-20260301-160913
```

## 2) Set cluster + runtime mode (recommended for b200)

Use b200 partition and frozen/system Python env:

```bash
export SBATCH_EXTRA_ARGS="--partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb --gpus-per-node=8 --cpus-per-task=224"
export SBATCH_EXPORT_VARS="ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=1,NRL_FORCE_REBUILD_VENVS=false,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0"
```

## 3) Launch fresh train runs (all seeds)

```bash
./submit_train_all.sh
```

Seeds covered: `123 124 125 126 127`.

## 4) Resume training from checkpoints (seed123 + seed127)

Use this exact command to resume from `step_135` checkpoints and continue to `RUN_STEPS=270`:

```bash
RESUME_CKPT_DIR_123=/home/scratch.shaunakj_other/results/tar-drift-reasonable-eagle3-s4-seed123-train-steps135-2026-03-02-045851 \
RESUME_CKPT_DIR_127=/home/scratch.shaunakj_other/results/tar-drift-reasonable-eagle3-s4-seed127-train-steps135-2026-03-02-060549 \
RUN_STEPS=270 \
SBATCH_EXTRA_ARGS="--partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb --gpus-per-node=8 --cpus-per-task=224" \
SBATCH_EXPORT_VARS="ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=1,NRL_FORCE_REBUILD_VENVS=false,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0" \
./submit_resume_train_123_127.sh
```

Resume submissions now default `NRL_SKIP_TRAIN_DATALOADER_RESUME=1` when `RESUME_CHECKPOINT_DIR` is set.
This avoids torchdata `StopIteration` crashes on some terminal sampler states while still loading:

1. policy weights from `step_*/policy/weights`
2. optimizer state from `step_*/policy/optimizer`

## 5) Monitor jobs

```bash
squeue -u "$USER"
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed,Partition,ReqTRES
```

Check live logs:

```bash
tail -f /home/scratch.shaunakj_other/logs/slurm-eagle-tardrift-s123-<JOBID>.out
tail -f /home/scratch.shaunakj_other/logs/slurm-eagle-tardrift-s127-<JOBID>.out
```

## 6) Control checkpoint saving

Checkpoint saving is enabled for `train` mode and disabled for `frozen` mode.

Default knobs used by the sbatch scripts:

```bash
SAVE_PERIOD=50
CKPT_KEEP_TOP_K=6
CKPT_SIZE_GIB=368
SPACE_RESERVE_GIB=300
CONCURRENT_TRAIN_JOBS=5
```

You can override them per submission, for example:

```bash
SAVE_PERIOD=50 \
CKPT_KEEP_TOP_K=3 \
RUN_STEPS=270 \
SBATCH_EXTRA_ARGS="--partition=b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb --gpus-per-node=8 --cpus-per-task=224" \
SBATCH_EXPORT_VARS="ALL,NEMO_RL_PY_EXECUTABLES_SYSTEM=1,NRL_FORCE_REBUILD_VENVS=false,RAY_ENABLE_UV_RUN_RUNTIME_ENV=0" \
./submit_train_all.sh
```

The script may increase effective `save_period` and reduce `keep_top_k` automatically when free-space safety limits are hit.

## 7) Confirm checkpoint resume actually happened

In the run log, confirm all of:

1. `Using checkpoint directory: ...seedXXX...`
2. `resume_step=step_135`
3. `Loading weights from .../step_135/policy/weights`

Example grep:

```bash
rg -n "Using checkpoint directory|resume_step=|Loading weights from .*step_135/policy/weights" \
  /home/scratch.shaunakj_other/logs/tar-drift-reasonable-eagle3-s4-seed123-train-steps270-*/run-270steps-seed123-train-eagle3.log \
  /home/scratch.shaunakj_other/logs/tar-drift-reasonable-eagle3-s4-seed127-train-steps270-*/run-270steps-seed127-train-eagle3.log
```

## 8) Optional: frozen phase

Run frozen only:

```bash
./submit_frozen_all.sh
```

Or chain train then frozen in one command:

```bash
DEP_TYPE=afterany ./submit_trains_then_frozen.sh
```

(`afterok` is stricter and runs frozen only if all train jobs succeed.)
