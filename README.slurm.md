# `user_run_slurm.sh` — Slurm submission wrapper for NeMo-RL

Thin wrapper around `ray.sub` that submits a NeMo-RL Ray training job to Slurm via `sbatch`. All knobs are environment variables; no CLI flags.

The script:
1. Sources `/lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh` (HuggingFace + W&B tokens).
2. Builds container `MOUNTS` (`$PWD:$PWD` and `/lustre:/lustre` — hard-coded).
3. Picks a Slurm partition + walltime from `LONG_RUN` / `MIDDLE_RUN`.
4. Calls `sbatch ... ray.sub` (must live in the same directory).

## Quick start

```bash
# Interactive mode: COMMAND unset → cluster comes up idle on 1 node,
# then attach into the head node via the generated <jobid>-attach.sh
bash user_run_slurm.sh
# ... wait for the job to start, then:
bash <jobid>-attach.sh

# Non-interactive: pass a training COMMAND
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml" \
  bash user_run_slurm.sh

# 4-node middle-length run
NNODES=4 MIDDLE_RUN=true COMMAND="..." bash user_run_slurm.sh

# 8-node long batch run with a custom container
NNODES=8 LONG_RUN=true \
  CONTAINER=/lustre/fsw/portfolios/coreai/users/me/enroot-images/foo.squashfs \
  COMMAND="..." \
  bash user_run_slurm.sh
```

## Two modes: `COMMAND` set vs. unset

`ray.sub` keys its behavior off `COMMAND`:

- **`COMMAND` unset / empty → interactive mode.** The Ray cluster is brought up across `NNODES` and left idle. Use the generated `<jobid>-attach.sh` in this directory to drop into the head-node container (`bash <jobid>-attach.sh`), or to dispatch a one-off command into it (`COMMAND='echo hi' bash <jobid>-attach.sh`). Ideal for debugging, ad-hoc REPL work, or staging multiple runs against the same cluster.
- **`COMMAND` set → non-interactive mode.** `ray.sub` `srun`s the command on the head node and the job ends when it exits. Logs land in `<jobid>-logs/ray-driver.log`.

## Environment variables

### Overridable (read by the script with a default)

| Var | Default | Purpose |
|---|---|---|
| `NNODES` | `1` | Number of nodes (`sbatch -N`). Per-node GPU count is hard-coded to `--gres=gpu:8`. |
| `COMMAND` | empty | Training command run on the head node by `ray.sub`. **Empty = interactive mode** (cluster idle, attach via `<jobid>-attach.sh`); set it to launch a non-interactive run. |
| `LONG_RUN` | `false` | If `true` → `batch` partition, 4h walltime. |
| `MIDDLE_RUN` | unset | If `true` (and `LONG_RUN` is not) → `batch_short` partition, 2h walltime. Otherwise → `interactive` partition, 2h walltime. |
| `ACCOUNT` | `coreai_dlalgo_nemorl` | Slurm `--account`. |
| `JOBNAME` | `8b_vllm_reinforcer_${PARTITION}` | Slurm `--job-name`. Submissions use `--dependency=singleton`, so same-name jobs serialize. |
| `CONTAINER` | yukih's `nemo-rl:c8d167a-48250558.squashfs` | Enroot squashfs to run inside. Override to test new images. |

### From `secrets.sh` (sourced at the top)

| Var | Purpose |
|---|---|
| `HUGGINGFACE_TOKEN` | Forwarded to `ray.sub` for HF auth. |
| `WANDB_API_KEY` | Forwarded to `ray.sub` for W&B logging. |

### Hard-coded — edit the script, env has no effect

| Var | Value | Notes |
|---|---|---|
| `MOUNTS` | `$PWD:$PWD,/lustre:/lustre` | Reassigned unconditionally on line 4. Setting `MOUNTS` in your shell is silently overwritten. Commented-out examples in the script show how to add `/opt/checkpoints` etc. |
| `HF_HOME` | `/lustre/fsw/portfolios/coreai/users/zhiyul/hf` | Inline export on line 62; no `${HF_HOME:-...}` fallback. Edit the script if you need a different HF cache. |
| `--gres` | `gpu:8` | Hard-coded on the `sbatch` line. |
| `secrets.sh` path | `/lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh` | Sourced unconditionally on line 3. |

## Partition / time matrix

| `LONG_RUN` | `MIDDLE_RUN` | Partition | Time |
|---|---|---|---|
| `true` | — | `batch` | `4:00:00` |
| anything | `true` | `batch_short` | `2:00:00` |
| `false`/unset | unset | `interactive` | `2:00:00` |

## Common gotchas

- **`secrets.sh` is required.** The script sources `/lustre/fsw/portfolios/coreai/users/zhiyul/secrets.sh`. If you're a different user, edit the path or create your own.
- **`COMMAND` defaults to empty (interactive mode).** That is intentional — the cluster stays up idle so you can attach via `<jobid>-attach.sh`. If you wanted a non-interactive run and forgot to export `COMMAND`, the job will just sit there burning allocation; cancel with `scancel <jobid>` and resubmit.
- **`--dependency=singleton`.** Two submissions with the same `JOBNAME` will queue, not run in parallel. Set a unique `JOBNAME` for concurrent experiments.
- **`ray.sub` must sit next to this script.** `sbatch` is called with a bare `ray.sub` filename — runs from the current working directory.
- **Per-run mounts.** `MOUNTS` is hard-coded — setting it in your shell does nothing. To mount additional checkpoint dirs (e.g. `/opt/checkpoints`), edit the `MOUNTS=` block in the script. Commented-out examples show the format.
- **`HF_HOME` is hard-coded** to `/lustre/fsw/portfolios/coreai/users/zhiyul/hf`. Edit the script if you need a different HuggingFace cache (different user, scratch volume, etc.).
- **Extra GPUs per node are not configurable here.** `--gres=gpu:8` is hard-coded; change the script if you need a different per-node GPU count.

## Output artifacts

`ray.sub` produces logs and an `<jobid>-attach.sh` script in this directory. The latter lets you `srun --jobid=... --overlap` into a running job. Existing `*-attach.sh` files in this directory are leftovers from past submissions — they're safe to delete.
