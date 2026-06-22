# Qwen 3.5 runtime overlay

This directory contains the Qwen 3.5-specific recipe and runtime overlay for
running `Qwen/Qwen3.5-397B-A17B` on top of the Carlos/grpo-studies
`mlperf-training` branch.

The goal is to keep Qwen 3.5 support isolated. Non-Qwen/Nemotron runs should use
the baked container code and normal recipes unless this directory is explicitly
selected.

## Why this exists

Qwen 3.5 support currently needs more than YAML values:

- Megatron Bridge must recognize the HF architecture/model type
  `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`; the required
  container already carries that support.
- The vLLM async worker needs Qwen 3.5 prompt/prefix repair and parser handling
  for multi-turn tool-use trajectories.
- NeMo-Gym needs Qwen-specific tolerance for token-contiguity issues caused by
  reasoning/tool-call retokenization.
- The SWE reward path should not apply Nemotron/Nano-specific scalar response
  penalties that can zero valid Qwen trajectories.

Hydra config can select model-family values, but it cannot create Pyxis/container
file mounts. Since these compatibility files must override files inside the baked
container at `/opt/nemo-rl`, a small launcher hook is required.

## How selection works

Use the normal NeMo-Gym launcher and select the Qwen 3.5 recipe:

```bash
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
examples/nemo_gym/launch_nemo_gym_multinode_training.sh <optional Hydra overrides>
```

When `RECIPE` points under `qwen_35/`,
`examples/nemo_gym/launch_nemo_gym_multinode_training.sh` automatically stages
Qwen-specific files under `${OUT_DIR}/qwen_35_mounts/` and mounts the staged
copies:

- `${OUT_DIR}/qwen_35_mounts/configs` to `/opt/nemo-rl/qwen_35/configs`
- every file under `${OUT_DIR}/qwen_35_mounts/overrides` to its matching path under `/opt/nemo-rl`

The staging step is intentional: the running container should not bind directly
to mutable files in the live Git checkout. Otherwise a `git pull` or local edit
can invalidate the mounted inode and produce errors such as `OSError: [Errno 116]
Stale file handle` while the job is starting.

For example:

```text
qwen_35/overrides/nemo_rl/models/megatron/setup.py
  -> /opt/nemo-rl/nemo_rl/models/megatron/setup.py
```

This lets the selected config and only that config bring the Qwen 3.5 runtime
patches into the container.

## What the config changes

`qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml` inherits:

```text
examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml
```

It carries the Qwen 3.5-specific runtime settings and the standard R2E
study defaults that should not depend on the launch site:

- Qwen 3.5 vLLM tool/reasoning parser defaults.
- Qwen 3.5 special token IDs used by token-aware checks.
- Qwen-safe response-penalty defaults.
- Megatron/vLLM compatibility values that should travel with Qwen 3.5.
- Standard R2E training HPs: GBS 512, 32 prompts x 16 generations, LR 5e-6,
  64k context, async GRPO age 1, PP=2 policy defaults, and SWE 15-turn
  timeouts.
- `policy.sequence_packing.enabled=false`, because Qwen 3.5 GDN currently does
  not support packed sequences.

It does **not** bake in cluster size, data paths, checkpoint paths, experiment
names, walltime, or node allocation. Those remain normal launcher environment
variables and minimal Hydra overrides. Deliberate shmoo/smoke changes, such as
PP=1 or smaller GBS, should be explicit overrides.

## What the overlay changes

The current overlay files are:

- `nemo_rl/models/megatron/setup.py`
  - Carries the Megatron setup compatibility used by the Qwen 3.5 runs.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
  - Carries the policy-worker compatibility used by the Qwen 3.5 runs.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`
  - Handles Qwen 3.5 parser, prefix repair, prompt truncation, and related
    multi-turn generation behavior.
- `nemo_rl/environments/nemo_gym.py`
  - Handles Qwen-specific token-contiguity behavior for NeMo-Gym rollouts.

These are intentionally under `qwen_35/overrides` instead of directly modifying
the base tree. That makes it clear which files are Qwen 3.5-specific and keeps
base grpo-studies behavior visible.

- `3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py`
  - Preserves the Gym Responses API behavior used by the known-good smoke path,
    including chat-template metadata pass-through. This remains an overlay
    because some remote checkouts do not have the `3rdparty/Gym-workspace` tree
    populated.


## Ptyche training HPs

Carlos's Ptyche Qwen3 run used ordinary training hyperparameters on top of a
different model/topology:

```text
/lustre/fsw/coreai_mlperf_training/users/cgomes/mlperf-rl/results/coreai_mlperf_training-grpo.grpo-235b-swe-async_gbs512_lr5.0e-6_mt30_r1
```

Important detail: despite the run-name `gbs512`, the final resolved Ptyche config
and MLLOG reported `train_global_batch_size=256`, `num_prompts_per_step=16`, and
`num_generations_per_prompt=16`. We do not keep a separate YAML for this because
the useful parts are simple overrides:

```bash
"policy.train_global_batch_size=256" \
"grpo.num_prompts_per_step=16" \
"grpo.num_generations_per_prompt=16" \
"grpo.max_num_steps=16" \
"grpo.val_batch_size=52" \
"grpo.max_val_samples=52" \
"grpo.val_at_start=true" \
"grpo.val_period=5" \
"grpo.val_at_end=true" \
"policy.megatron_cfg.optimizer.lr=5.0e-6" \
"policy.megatron_cfg.optimizer.min_lr=5.0e-6" \
"policy.megatron_cfg.scheduler.lr_warmup_iters=3" \
"policy.megatron_cfg.scheduler.lr_decay_iters=100000" \
"policy.megatron_cfg.scheduler.lr_decay_style=constant" \
"env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=720" \
"env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_tests_timeout=300" \
"env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=720" \
"env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_tests_timeout=180"
```

## Prompt truncation

`NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=<N|none>` controls the prompt
truncation fallback used by the Qwen vLLM async worker patch. The launcher sets
the default used by the known-good R2E runs; override it only when intentionally
testing a different context budget.

## OCI-HSG R2E study example

The following is the 64-node R2E/Qwen 3.5 study shape used for the
`swe-qwen35-r2e-wlogging-64k-30turn-pp2real-32train32gen-4h-*` runs on
OCI-HSG. It is intentionally verbose so the run can be reproduced without
depending on local shell wrappers.

Run from the grpo-studies checkout on branch `arigazz/qwen35-clobber`:

```bash
cd /lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/grpo-studies-qwen35
git checkout arigazz/qwen35-clobber
git pull --ff-only origin arigazz/qwen35-clobber

stamp=$(date +%Y%m%d-%H%M%S)
export EXP_NAME="swe-qwen35-r2e-wlogging-64k-30turn-pp2real-32train32gen-4h-qwen35branch-${stamp}"

export REPO_LOCATION="$PWD"
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
export CONTAINER_IMAGE_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/containers/nemorl_v0.6_prebaked_arm_clean_w_logging.sqsh

export SLURM_ACCOUNT=coreai_mlperf_training
export SLURM_PARTITION=batch
export SLURM_QOS=short
export SBATCH_QOS=short
export SBATCH_GRES=gpu:4
export SLURM_TIME=4:00:00

export GPUS_PER_NODE=4
export TRAIN_NODES=32
export GEN_NODES=32
export NODES=64

export HF_CKPT_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/nemotron3_ultra_550b/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7
export CONTAINER_HF_CKPT_PATH="$HF_CKPT_PATH"
export NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/nemotron3_ultra_550b/mcore_ckpt_cache
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="$NRL_MEGATRON_CHECKPOINT_DIR"

export NEMO_GYM_SWE_TRAIN_DATA_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/grpo-studies/data_swe/r2e_easy_l20_train.with_sifs.jsonl
export NEMO_GYM_SWE_VALIDATION_DATA_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/arigazzi/grpo-studies/data_swe/r2e_easy_l20_val.with_sifs.jsonl
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH=/inputs/nemo_gym/data/train.jsonl
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH=/inputs/nemo_gym/data/validation.jsonl

export NEMO_GYM_SWE_SIF_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/swe-gym
export CONTAINER_NEMO_GYM_SWE_SIF_DIR="$NEMO_GYM_SWE_SIF_DIR"
export NEMO_GYM_SWE_FALLBACK_SIF_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/nemotron-ultra-swe

# Some systems, such as Ptyche, need `/dev/fuse:/dev/fuse` appended to
# EXTRA_MOUNTS for nested SIF execution. OCI-HSG has not required it in this
# recipe; add it only if the target cluster reports FUSE/apptainer mount errors.

# Default is 65535 for a 64k context run, leaving one token of headroom so
# vLLM does not reject prompts at exactly/just above max_model_len.
export QWEN35_TRUNCATE_PROMPT_TOKENS=65535
export NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=65535
```

On Lyris, use the cluster-local paths instead. The primary R2E root covers the
current R2E train/validation JSONL there, so leave
`NEMO_GYM_SWE_FALLBACK_SIF_DIR` unset unless a future dataset needs it. Lyris
does require `/dev/fuse` for nested R2E SIF execution:

```bash
export CONTAINER_IMAGE_PATH=/lustre/fsw/coreai_mlperf_training/users/cgomes/containers/optimized+nemorl_v0.6_prebaked_arm_clean_w_logging
export HF_CKPT_PATH=/lustre/fsw/coreai_mlperf_training/users/arigazzi/nemotron3_ultra_550b/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/8472618112abcbd45acbcdc58436aff4233c23f7
export CONTAINER_HF_CKPT_PATH="$HF_CKPT_PATH"
export NRL_MEGATRON_CHECKPOINT_DIR=/lustre/fsw/coreai_mlperf_training/users/arigazzi/nemotron3_ultra_550b/mcore_ckpt_cache
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="$NRL_MEGATRON_CHECKPOINT_DIR"
export NEMO_GYM_SWE_TRAIN_DATA_PATH=/lustre/fsw/coreai_mlperf_training/users/arigazzi/grpo-studies/data_swe/r2e_easy_l20_train.with_sifs.jsonl
export NEMO_GYM_SWE_VALIDATION_DATA_PATH=/lustre/fsw/coreai_mlperf_training/users/arigazzi/grpo-studies/data_swe/r2e_easy_l20_val.with_sifs.jsonl
export NEMO_GYM_SWE_SIF_DIR=/lustre/fsw/coreai_mlperf_training/users/hfilaretov/data
unset NEMO_GYM_SWE_FALLBACK_SIF_DIR
export EXTRA_MOUNTS=/dev/fuse:/dev/fuse
```

Then launch. The recipe owns the R2E SIF formatter list; do not pass it as a
Hydra CLI override because the `{instance_id}` placeholders are parsed as
override grammar unless heavily escaped.

```bash
bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
  "policy.model_name=${HF_CKPT_PATH}" \
  "data.train.data_path=${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH}" \
  "data.validation.data_path=${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH}" \
  "sif_dir=${NEMO_GYM_SWE_SIF_DIR}" \
  "logger.wandb_enabled=False"
```

Notes:

- `SBATCH_GRES=gpu:4` is required on OCI-HSG `batch`; without it Slurm rejects
  the job because no GPU TRES is requested.
- The Qwen wrapper owns the stable training HPs and R2E SIF formatter list.
  Do not pass correctness-critical knobs such as sequence packing, parser
  settings, token IDs, or formatter paths as CLI overrides.
- The result directory is `${REPO_LOCATION}/results/${EXP_NAME}`.
- The Ray driver log appears under
  `${REPO_LOCATION}/results/${EXP_NAME}/logs/<jobid>-logs/ray-driver.log`.

## Safety boundary

The default behavior is:

- recipes under `qwen_35/`: Qwen config and Qwen overlay are mounted.
- all other recipes: no Qwen overlay is mounted.

This is the cleanest current compromise: selecting the Qwen 3.5 config is enough
for users, while the only base-tree code change is the small launcher hook needed
to make container mounts possible.
