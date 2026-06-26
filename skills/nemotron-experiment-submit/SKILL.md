---
name: nemotron-experiment-submit
description: Submit and monitor NeMo-RL NemotronLabsDiffusion GRPO-family experiments on dfw, including AR GRPO, JustGRPO, and DiffuGRPO configs, run naming, env tags, rebuild flags, logs, and Slurm status checks.
---

# RL on NemotronLabsDiffusion Experiment Submission

Use this skill when submitting or debugging NeMo-RL training experiments from the current RL repo on dfw.

## Working Directory

Run from the active RL checkout:

```bash
cd ~/diffusion_RL/RL
```

Submit through the repo's Nemotron diffusion sbatch wrapper:

```bash
tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh
```

The wrapper name contains `ar`, but it is also used for JustGRPO and DiffuGRPO configs unless the repo introduces a newer algorithm-specific wrapper. The selected algorithm comes from `CONFIG=...` and any `EXTRA_CONFIG_OVERRIDES`, not from the script name.

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

After submission, explicitly share the exact submission command with the user together with the Slurm job id.

Use PARTITION=batch for all sbatch submissions. The dfw batch partition has a 4-hour limit, so keep TIME at or below 04:00:00. Adjust NODES as needed, but do not submit these experiments to backfill, batch_long, batch_large, or batch_large_long.

## Account Selection (Fair Share)

Pick `ACCOUNT` based on Slurm fair-share: submit under the eligible account with the **highest current FairShare**, since that account gets the best scheduling priority (shortest queue wait).

Check fair-share before submitting:

```bash
sshare -U -u $USER -o Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare -p
```

The `FairShare` column is what matters (higher = better priority, range 0-1). Choose the account with the largest value, with two caveats:

- Restrict the choice to accounts compatible with this work: the run dirs and HF cache live under the `coreai` portfolio (`/lustre/fsw/portfolios/coreai/...`), so use a `coreai_dlalgo_*` account. Do not use a non-coreai account such as `nvr_lpr_llm` even if it shows a higher FairShare, since it does not have access to these paths/partition.
- An account's FairShare drops as its recent usage rises, so the best choice changes over time. Re-check `sshare` for each new submission rather than hardcoding one account.

In practice `coreai_dlalgo_genai` has had a higher FairShare than the wrapper default `coreai_dlalgo_llm` (whose FairShare is depressed by heavy recent usage), so prefer `ACCOUNT=coreai_dlalgo_genai` unless `sshare` says otherwise at submit time.

## Common Configs

AR GRPO:

- main: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml`
- toy: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron_toy_p8_g8.yaml`

JustGRPO leftmost reveal:

- main: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_justgrpo_leftmost_megatron.yaml`
- toy: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_justgrpo_leftmost_megatron_toy_p8_g8.yaml`

DiffuGRPO with FastDiffuser:

- main: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_diffugrpo_megatron.yaml`
- toy: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_diffugrpo_megatron_toy_p8_g8.yaml`

For a smoke test, use the toy config for the relevant mode and cap steps through `EXTRA_CONFIG_OVERRIDES`, for example `grpo.max_num_steps=3 grpo.val_at_end=false checkpointing.enabled=false logger.wandb_enabled=false`.


## DeepScaleR dataset

DeepScaleR variants train on `agentica-org/DeepScaleR-Preview-Dataset` (`dataset_name: DeepScaler`, ~40K math problems) and validate on AIME 2024 (`HuggingFaceH4/aime_2024`, `dataset_name: AIME2024`, `repeat: 16`). Both route through the `math` env / `hf_math_verify` reward. The datasets are prefetched into the offline HF cache at `/lustre/fsw/portfolios/coreai/users/$USER/hf_home/datasets` (the training container is HF-offline; prefetch new datasets from a login node into that path before submitting).

Configs (DeepScaleR train + AIME2024 validation; 4096 sequence budget = 350 prompt + 3744 generation; KL off; `val_period`/`save_period` = 10):

- AR GRPO:   `CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml`
- DiffuGRPO: `CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_diffugrpo_megatron.yaml`

The diffugrpo config inherits the data block + 4096 sequence budget from the AR DeepScaleR config; KL-off (`loss_fn.reference_policy_kl_penalty: 0.0` + `grpo.skip_reference_policy_logprobs_calculation: true`) and the val/save intervals live in the AR parent.

### max_new_tokens must be divisible by FastDiffuser block_size (32)

For DiffuGRPO, `policy.generation.max_new_tokens` MUST be a multiple of the FastDiffuser `block_size` (32). A non-multiple (e.g. 3746) crashes generation with a RoPE view error (`view size is not compatible with input tensor's size and stride ...` in `rotary_embedding`). Use 3744 (= 117*32). AR mode has no such constraint but is kept at 3744 for consistency. For ar grpo use EXPANDABLE_SEGMENTS flag to avoid OOM(fragmentation OOM). DiffuGRPO did not need it (it does not OOM).

### Block JustGRPO: set EXPANDABLE_SEGMENTS=false (avoids a severe refit slowdown)

For Block JustGRPO runs, submit with `EXPANDABLE_SEGMENTS=false`. With `expandable_segments:True` enabled (which `EXPANDABLE_SEGMENTS=true` injects as `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the Megatron policy workers), the per-step weight refit (`prepare_for_generation/transfer_and_update_weights`) becomes pathologically slow and dominates the step. Measured on an otherwise-identical 16-node Block JustGRPO comparison: refit was ~6.9s/step with the flag OFF (4k context) vs ~661s/step with the flag ON (8k context) — roughly a 100x inflation that pushed the step from ~5 min to ~25 min. All other stages (generation, logprob, training) scaled only ~2-3x as expected for the larger context, so the refit was the sole regression. The expandable-segments allocator's virtual-memory map/unmap overhead is hit hard by the refit's large-buffer weight streaming plus optimizer offload. Block JustGRPO does not rely on expandable segments for OOM avoidance (like DiffuGRPO, it does not fragmentation-OOM), so keep the flag off.

Note: AR GRPO still uses `EXPANDABLE_SEGMENTS=true` to avoid fragmentation OOM at long sequence length — this guidance is specific to Block JustGRPO (and the DiffuGRPO family, which also does not need it).

### Multi-node: set cluster.num_nodes to match NODES

`NODES` only sizes the Slurm allocation; the wrapper does NOT inject `cluster.num_nodes`, and the config defaults to `cluster.num_nodes: 1`. For multi-node runs you MUST override it via `EXTRA_CONFIG_OVERRIDES`, or NeMo-RL runs on a single node:

```bash
NODES=8 ... EXTRA_CONFIG_OVERRIDES='cluster.num_nodes=8' ...
```

### Example 8-node AR DeepScaleR submit

```bash
RUN_NAME=deepscaler_ar_3b_8n WANDB_RUN_NAME=deepscaler_ar_3b_8n JOB_NAME=ds_ar_8n \
ACCOUNT=coreai_dlalgo_genai PARTITION=batch TIME=04:00:00 NODES=8 \
ENV_TAG=mb_3rdparty_sglagn_local_fork EXPANDABLE_SEGMENTS=true \
CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml \
NRL_FORCE_REBUILD_VENVS=false FORCE_REINSTALL_PACKAGES=false FORCE_REINSTALL_SGLANG=false \
EXTRA_CONFIG_OVERRIDES='cluster.num_nodes=8 policy.logprob_batch_size=4' \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --sbatch
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

Use this same env tag for DiffuGRPO megatron runs as well:

```bash
ENV_TAG=mb_3rdparty_sglagn_local_fork
```

Do not change `ENV_TAG` for normal Python edits in the existing editable dependency trees. Reuse the tag so follow-up runs reuse the same driver and worker envs.

Change `ENV_TAG` only when one of the dependency paths changes, for example a new `pyproject.toml` path for SGLang or a different Megatron-Bridge/Megatron-LM workspace path. After changing dependency paths, run once with the new tag and the needed reinstall/rebuild flags. The first run with a fresh tag may spend significant time building packages such as TransformerEngine; later runs with the same tag should be faster.

## Logs and Checkpoints

The default run directory is:

```text
/lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>
```

Key files:

```text
run log:     /lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>/run.log
slurm log:   /lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>/slurm-<jobid>.out
checkpoints: /lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>/checkpoints
```

## Monitoring

Check Slurm status:

```bash
squeue -j <jobid> -o "%.18i %.12P %.35j %.8u %.2t %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
```

Tail logs:

```bash
RUN_DIR=/lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>
tail -200 "$RUN_DIR/run.log"
tail -200 "$RUN_DIR/slurm-<jobid>.out"
```

Useful grep checks:

```bash
grep -n "SGLANG_SOURCE\|runtime_versions\|FastDiffuser: block_size\|selection_policy\|Started a local Ray\|Step [0-9]/\|Loss:\|RuntimeError\|Traceback" "$RUN_DIR"/run.log "$RUN_DIR"/slurm-<jobid>.out
```

For JustGRPO leftmost runs, confirm SGLang logs include `selection_policy=leftmost`. If the log only prints `FastDiffuser: block_size=... max_steps=... temperature=... threshold=...`, the worker may be importing an older SGLang build.

For DiffuGRPO confidence runs, confirm SGLang logs include `FastDiffuser: block_size=32`, `threshold=0.9`, and `selection_policy=confidence`, and confirm the policy worker is `DiffuGRPOMegatronPolicyWorker`.

## Local or Interactive Run

Only run locally inside an allocated interactive GPU node/container:

```bash
RUN_NAME=<run_name> \
WANDB_RUN_NAME=<run_name> \
ENV_TAG=mb_3rdparty_sglagn_local_fork \
CONFIG=examples/configs/<config_file>.yaml \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --local
```
