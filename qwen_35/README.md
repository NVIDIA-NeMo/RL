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
- The Megatron setup path needs Qwen 3.5 compatibility for the GatedDeltaNet
  code path used by the current container stack.
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
qwen_35/overrides/nemo_rl/models/megatron/community_import.py
  -> /opt/nemo-rl/nemo_rl/models/megatron/community_import.py
```

This lets the selected config and only that config bring the Qwen 3.5 runtime
patches into the container.

## What the config changes

`qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml` inherits:

```text
examples/nemo_gym/grpo_qwen3_235b_swe_openhands_async.yaml
```

It intentionally carries only model-family-specific settings:

- Qwen 3.5 vLLM tool/reasoning parser defaults.
- Qwen 3.5 special token IDs used by token-aware checks.
- Qwen-safe response-penalty defaults.
- Narrow Megatron/vLLM compatibility values that should travel with Qwen 3.5.

It does **not** bake in cluster size, data paths, checkpoint paths, experiment
names, walltime, or node allocation. Those remain normal launcher environment
variables and Hydra overrides.

## What the overlay changes

The current overlay files are:

- `nemo_rl/models/megatron/community_import.py`
  - Carries the random-init and Megatron import compatibility used by the
    known-good Qwen 3.5 smoke path.
- `nemo_rl/models/megatron/setup.py`
  - Adds Qwen 3.5 Megatron setup compatibility, including GatedDeltaNet handling.
- `nemo_rl/models/policy/workers/megatron_policy_worker.py`
  - Carries the policy-worker compatibility used by the Qwen 3.5 runs.
- `nemo_rl/models/generation/vllm/__init__.py`
  - Ensures the patched vLLM generation path is selected.
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

## Escape hatches

- `QWEN35_OVERLAY=0`: disable automatic Qwen overlay mounting.
- `QWEN35_OVERLAY=1`: force overlay mounting even if `RECIPE` is not under
  `qwen_35/`.
- `QWEN35_OVERLAY_DIR=/path/to/overrides`: use a different overlay directory.
- `QWEN35_CONFIG_DIR=/path/to/configs`: use a different config directory.
- `NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=<N|none>`: controls the prompt
  truncation fallback used by the Qwen vLLM async worker patch.
- `NEMO_RL_QWEN35_FORCE_TORCH_GDN=1`: forces the torch GatedDeltaNet fallback.
- `NEMO_RL_QWEN35_ALLOW_NON_MONOTONIC_PREFIX=1`: re-enables the defensive
  monotonic-prefix fallback. The default is strict because the known-good
  diagnostic run did not need this path.
- `NEMO_RL_ALLOW_NONCONTIGUOUS_MESSAGE_TOKENS=true`: re-enables collapsing
  non-contiguous trajectory history to the current turn. The default is strict
  because the known-good diagnostic run did not hit this path.

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
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="$NEMO_GYM_SWE_TRAIN_DATA_PATH"
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="$NEMO_GYM_SWE_VALIDATION_DATA_PATH"

export NEMO_GYM_SWE_SIF_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/swe-gym
export CONTAINER_NEMO_GYM_SWE_SIF_DIR="$NEMO_GYM_SWE_SIF_DIR"
export EXTRA_MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/nemotron-ultra-swe/r2e_gym:/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/nemotron-ultra-swe/r2e_gym"

# Some systems, such as Ptyche, need `/dev/fuse:/dev/fuse` appended to
# EXTRA_MOUNTS for nested SIF execution. OCI-HSG has not required it in this
# recipe; add it only if the target cluster reports FUSE/apptainer mount errors.

# Default is 65535 for a 64k context run, leaving one token of headroom so
# vLLM does not reject prompts at exactly/just above max_model_len.
export QWEN35_TRUNCATE_PROMPT_TOKENS=65535
export NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=65535
```

Then launch:

```bash
R2E_FORMATTERS='[\"/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/swe-gym/r2egym/{instance_id}.sif\",\"/lustre/fs1/portfolios/coreai/projects/coreai_mlperf_training/users/hfilaretov/data/nemotron-ultra-swe/r2e_gym/{instance_id}.sif\"]'

bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
  "policy.model_name=${HF_CKPT_PATH}" \
  "data.train.data_path=${NEMO_GYM_SWE_TRAIN_DATA_PATH}" \
  "data.validation.data_path=${NEMO_GYM_SWE_VALIDATION_DATA_PATH}" \
  "sif_dir=${NEMO_GYM_SWE_SIF_DIR}" \
  "logger.wandb_enabled=False" \
  "logger.wandb.project=nemotron-3-ultra" \
  "policy.train_global_batch_size=128" \
  "policy.train_micro_batch_size=1" \
  "policy.logprob_batch_size=1" \
  "policy.generation_batch_size=64" \
  "grpo.num_prompts_per_step=16" \
  "grpo.num_generations_per_prompt=8" \
  "grpo.val_batch_size=null" \
  "grpo.max_num_steps=50" \
  "grpo.async_grpo.enabled=true" \
  "grpo.async_grpo.max_trajectory_age_steps=1" \
  "grpo.val_period=5" \
  "grpo.val_at_start=true" \
  "grpo.val_at_end=false" \
  "policy.megatron_cfg.tensor_model_parallel_size=4" \
  "policy.megatron_cfg.expert_model_parallel_size=64" \
  "policy.megatron_cfg.expert_tensor_parallel_size=1" \
  "policy.megatron_cfg.context_parallel_size=1" \
  "policy.megatron_cfg.pipeline_model_parallel_size=2" \
  "policy.megatron_cfg.sequence_parallel=true" \
  "policy.megatron_cfg.moe_token_dispatcher_type=alltoall" \
  "policy.megatron_cfg.apply_rope_fusion=false" \
  "++policy.megatron_cfg.gradient_accumulation_fusion=false" \
  "++policy.megatron_cfg.checkpoint.async_strategy=mcore" \
  "policy.generation.vllm_cfg.tensor_parallel_size=8" \
  "policy.generation.vllm_cfg.pipeline_parallel_size=1" \
  "policy.generation.vllm_cfg.expert_parallel_size=8" \
  "policy.generation.vllm_cfg.gpu_memory_utilization=0.55" \
  "policy.generation.vllm_cfg.max_model_len=65536" \
  "policy.max_total_sequence_length=65536" \
  "policy.generation.max_new_tokens=65536" \
  "data.max_input_seq_length=null" \
  "++data.use_multiple_dataloader=false" \
  "checkpointing.enabled=false" \
  "++checkpointing.save_optimizer=false" \
  "checkpointing.save_period=1000000" \
  "checkpointing.checkpoint_must_save_by=null" \
  "env.nemo_gym.skip_venv_if_present=true" \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.agent_max_turns=30" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.agent_max_turns=30" \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.concurrency=128" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.concurrency=128" \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.swebench_agent_timeout=360" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.swebench_agent_timeout=180" \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=${R2E_FORMATTERS}" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.container_formatter=${R2E_FORMATTERS}" \
  "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser=qwen3_xml" \
  "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser=qwen3" \
  "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser_plugin=null" \
  "token_ids.eos=248046" \
  "token_ids.think_open=248068" \
  "token_ids.think_close=248069" \
  "penalize_duplicated_reasoning=false" \
  "penalize_empty_final_answer=false" \
  "penalize_eos_token=false" \
  "penalize_malformed_think_tag=false" \
  "policy.sequence_packing.enabled=false" \
  "policy.generation.temperature=1.0" \
  "policy.generation.top_p=1.0" \
  "policy.megatron_cfg.optimizer.lr=5.0e-6" \
  "policy.megatron_cfg.optimizer.min_lr=4.999e-6" \
  "policy.megatron_cfg.scheduler.lr_warmup_iters=3" \
  "policy.megatron_cfg.scheduler.lr_decay_iters=100000" \
  "grpo.overlong_filtering=true" \
  "++grpo.skip_reference_policy_logprobs_calculation=true" \
  "grpo.max_val_samples=null" \
  "+policy.generation.mcore_generation_config.buffer_size_gb=8" \
  "grpo.async_grpo.in_flight_weight_updates=false"
```

Notes:

- `SBATCH_GRES=gpu:4` is required on OCI-HSG `batch`; without it Slurm rejects
  the job because no GPU TRES is requested.
- Keep the escaped quotes in `R2E_FORMATTERS`. The launcher interpolates
  overrides into a shell command, and unescaped quotes are consumed before
  Hydra sees the list. Without escaping, Hydra rejects the `{instance_id}`
  placeholder in the formatter paths.
- The result directory is `${REPO_LOCATION}/results/${EXP_NAME}`.
- The Ray driver log appears under
  `${REPO_LOCATION}/results/${EXP_NAME}/logs/<jobid>-logs/ray-driver.log`.

## Safety boundary

Non-Qwen runs are unaffected unless `QWEN35_OVERLAY=1` is set explicitly. The
default behavior is:

- recipes under `qwen_35/`: Qwen config and Qwen overlay are mounted.
- all other recipes: no Qwen overlay is mounted.

This is the cleanest current compromise: selecting the Qwen 3.5 config is enough
for users, while the only base-tree code change is the small launcher hook needed
to make container mounts possible.
