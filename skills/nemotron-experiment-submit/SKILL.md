---
name: nemotron-experiment-submit
description: Submit and monitor NeMo-RL NemotronLabsDiffusion/GRPO experiments on dfw using tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh, including required CONFIG passing, run naming, env tags, rebuild flags, logs, and Slurm status checks.
---

# Nemotron Experiment Submission

Use this skill when submitting or debugging NeMo-RL training experiments from the current RL repo on dfw.

## Working Directory

Run from:

```bash
cd /home/snorouzi/diffusion_RL/RL
```

Submit through:

```bash
tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh
```

Always pass the intended config file explicitly with `CONFIG=...`; do not rely on the script default when launching a named experiment.

## Standard Submit Command

```bash
RUN_NAME=<run_name> \
WANDB_RUN_NAME=<run_name> \
JOB_NAME=<short_slurm_name> \
PARTITION=batch \
TIME=04:00:00 \
NODES=1 \
ENV_TAG=mb_3rdparty_sglagn_local_fork \
CONFIG=examples/configs/<config_file>.yaml \
NRL_FORCE_REBUILD_VENVS=false \
FORCE_REINSTALL_PACKAGES=false \
FORCE_REINSTALL_SGLANG=false \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --sbatch
```

Use PARTITION=batch for all sbatch submissions. The dfw batch partition has a 4-hour limit, so keep TIME at or below 04:00:00. Adjust NODES as needed, but do not submit these experiments to backfill, batch_long, batch_large, or batch_large_long.

## Common Configs

Default NemotronLabsDiffusion AR GRPO:

```bash
CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml
```

JustGRPO leftmost reveal toy config:

```bash
CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_justgrpo_leftmost_megatron_toy_p8_g8.yaml
```

Default AR toy config:

```bash
CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron_toy_p8_g8.yaml
```

Long-context AR config:

```bash
CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron_seq16384_gen8192.yaml
```

## Rebuild and Reinstall Flags

Use these deliberately:

- `NRL_FORCE_REBUILD_VENVS=false`: reuse worker venvs. Use this for normal reruns after envs exist.
- `NRL_FORCE_REBUILD_VENVS=true`: rebuild worker venvs. Use after dependency changes or when stale worker packages are suspected.
- `FORCE_REINSTALL_NEMO_RL=false`: keep the editable NeMo-RL install. Use this for normal Python-only changes under `nemo_rl/...`; new driver/worker processes import the active worktree directly.
- `FORCE_REINSTALL_SGLANG=true`: force reinstall SGLang into the driver env. Use after changing the SGLang dependency source in `pyproject.toml` or when validating a new SGLang checkout.
- `FORCE_REINSTALL_PACKAGES=true`: broad reinstall switch. Avoid unless needed; it can trigger long package builds.

NeMo-RL is installed editable in the usual driver and Ray worker environments, so source-only edits to the active RL worktree do not require `NRL_FORCE_REBUILD_VENVS=true` or `FORCE_REINSTALL_NEMO_RL=true`. Rebuilding the venv can unnecessarily trigger slow dependency builds such as TransformerEngine.

If `sglang` is an editable path dependency in `pyproject.toml`, normal Python edits in that SGLang checkout are picked up by new worker processes after restart. Dependency, metadata, kernel, or compiled-extension changes still require reinstall/rebuild.

## Env Tag

Use this shared env tag for the current dependency layout:

```bash
ENV_TAG=mb_3rdparty_sglagn_local_fork
```

Do not change `ENV_TAG` for normal Python edits in the existing editable dependency trees. Reuse the tag so follow-up runs reuse the same driver and worker envs.

Change `ENV_TAG` only when one of the dependency paths changes, for example a new `pyproject.toml` path for SGLang or a different Megatron-Bridge/Megatron-LM workspace path. After changing dependency paths, run once with the new tag and the needed reinstall/rebuild flags. The first run with a fresh tag may spend significant time building packages such as TransformerEngine; later runs with the same tag should be faster.

## Long-Context Notes

The long-context config sets `policy.max_total_sequence_length: 16384` and currently caps SGLang generation at `policy.generation.max_new_tokens: 4096`. This still increases policy/logprob memory pressure because the training tensors and masks are sized for the full `max_total_sequence_length`, not only generated tokens.

For long-context experiments, start with smaller `policy.logprob_batch_size`, `policy.train_micro_batch_size`, and SGLang request concurrency if memory fails. Avoid changing model `seq_length` casually; for these runs, the intent was longer training/evaluation sequence budget, not changing the base checkpoint architecture.

## Logs and Checkpoints

The default run directory is:

```text
/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/<RUN_NAME>
```

Key files:

```text
run log:     /lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/<RUN_NAME>/run.log
slurm log:   /lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/<RUN_NAME>/slurm-<jobid>.out
checkpoints: /lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/<RUN_NAME>/checkpoints
```

## Monitoring

Check Slurm status:

```bash
squeue -j <jobid> -o "%.18i %.12P %.35j %.8u %.2t %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
```

Tail logs:

```bash
RUN_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/runs/diffusion_rl/<RUN_NAME>
tail -200 "$RUN_DIR/run.log"
tail -200 "$RUN_DIR/slurm-<jobid>.out"
```

Useful grep checks:

```bash
grep -n "SGLANG_SOURCE\|runtime_versions\|FastDiffuser: block_size\|selection_policy\|Started a local Ray\|Step [0-9]/\|Loss:\|RuntimeError\|Traceback" "$RUN_DIR"/run.log "$RUN_DIR"/slurm-<jobid>.out
```

For JustGRPO leftmost runs, confirm SGLang logs include `selection_policy=leftmost`. If the log only prints `FastDiffuser: block_size=... max_steps=... temperature=... threshold=...`, the worker may be importing an older SGLang build.

## Local or Interactive Run

Only run locally inside an allocated interactive GPU node/container:

```bash
RUN_NAME=<run_name> \
WANDB_RUN_NAME=<run_name> \
ENV_TAG=mb_3rdparty_sglagn_local_fork \
CONFIG=examples/configs/<config_file>.yaml \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --local
```
