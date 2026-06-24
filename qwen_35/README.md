# Qwen 3.5 runtime overlay

This directory contains the Qwen 3.5-specific recipe and runtime overlay for
running `Qwen/Qwen3.5-397B-A17B` on top of the GitHub RL `mlperf-training`
branch.

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

## Container image

The Qwen overlay is mounted at launch time, but the job still needs a Gym-capable
NeMo RL container. Use a site-provided equivalent image, or build and publish the
bleeding-edge Gym image from this checkout:

```bash
export IMAGE_PREFIX=registry.example.com/my-team/nemo-rl

docker buildx build --platform linux/amd64 \
  -t "${IMAGE_PREFIX}:v0.6.0-gym" \
  --push \
  -f docker/Dockerfile.gym_v0.6.0 .

docker buildx build --platform linux/amd64 \
  --build-arg BASE_IMAGE="${IMAGE_PREFIX}:v0.6.0-gym" \
  --build-arg NEMO_RL_REF=main \
  --build-arg MEGATRON_BRIDGE_REF=main \
  --build-arg MEGATRON_LM_REF=main \
  -t "${IMAGE_PREFIX}:gym-bleeding-edge" \
  --push \
  -f docker/Dockerfile.gym_bleeding_edge .
```

Set `CONTAINER_IMAGE_PATH` to whatever image reference the target Slurm/Pyxis
site accepts for that image: a registry reference, `.sqsh` path, or other
site-local image identifier. The Qwen files in this branch do not need to be
baked into the image; the launcher stages and mounts them under `/opt/nemo-rl`.

## How selection works

Use the normal NeMo-Gym launcher and select the Qwen 3.5 recipe:

```bash
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml
examples/nemo_gym/launch_nemo_gym_multinode_training.sh <optional Hydra overrides>
```

For the benchmark-shaped variant, use:

```bash
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_benchmark.yaml
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

## Required launch inputs

Before submitting a training job, set these environment variables:

| Variable | Purpose |
| --- | --- |
| `EXP_NAME` | Result directory name under `${REPO_LOCATION}/results`. |
| `REPO_LOCATION` | Host path to this RL checkout. Usually `"$PWD"`. |
| `RECIPE` | Qwen 3.5 recipe under `qwen_35/configs/`, usually the standard or benchmark config. |
| `CONTAINER_IMAGE_PATH` | Gym-capable container image built above or provided by the site. |
| `SLURM_ACCOUNT`, `SLURM_PARTITION` | Slurm allocation. |
| `GPUS_PER_NODE` | GPUs per node requested by the job. |
| `TRAIN_NODES`, `GEN_NODES` | Policy training and generation node counts. |
| `HF_CKPT_PATH` | Host path to the Qwen 3.5 HF checkpoint. |
| `NRL_MEGATRON_CHECKPOINT_DIR` | Host path for the Megatron checkpoint cache/output. |
| `NEMO_GYM_SWE_TRAIN_DATA_PATH` | Host path to the SWE train JSONL. |
| `NEMO_GYM_SWE_VALIDATION_DATA_PATH` | Host path to the SWE validation JSONL. |
| `NEMO_GYM_SWE_SIF_DIR` | Host directory containing R2E/SWE SIF images. |

Common optional inputs:

- `SLURM_TIME`, `SLURM_QOS`/`SBATCH_QOS`, and `SBATCH_GRES` or `SLURM_GRES`
  select cluster-specific queue settings.
- `NODES` can be set explicitly; if omitted, the launcher computes it from
  `TRAIN_NODES + GEN_NODES`.
- `NEMO_GYM_SWE_FALLBACK_SIF_DIR` adds a second host SIF root for datasets that
  reference images outside `NEMO_GYM_SWE_SIF_DIR`.
- `EXTRA_MOUNTS` appends site-specific container mounts, for example
  `/dev/fuse:/dev/fuse` on clusters that need FUSE for nested SIF execution.
- `CONTAINER_HF_CKPT_PATH`, `CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR`,
  `CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH`,
  `CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH`, and
  `CONTAINER_NEMO_GYM_SWE_SIF_DIR` override the default in-container paths.
  Leave them at the launcher defaults unless the site requires a different
  mount layout.

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

`qwen_35/configs/grpo_qwen35_397b_swe_openhands_async_benchmark.yaml` inherits
the standard config and overrides only benchmark-shape knobs: 16 prompts x 8
generations, 20 steps, validation every 2 steps with 256 validation samples,
GBS 128, LR/min-LR 2e-6, warmup 2, training EP 16, and validation agent timeout
360 seconds. The vLLM parallelism and other Qwen 3.5 defaults remain inherited
from the standard config.

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
base behavior visible.

- `3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py`
  - Preserves the Gym Responses API behavior used by the known-good smoke path,
    including chat-template metadata pass-through. This remains an overlay
    because some remote checkouts do not have the `3rdparty/Gym-workspace` tree
    populated.


## Optional shmoo overrides

Use explicit Hydra overrides for intentional smoke/shmoo runs instead of baking
site- or run-specific values into the Qwen wrapper. For example, a smaller
training shape can be launched with:

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

## Training launch template

The following template is site-neutral. Fill in the required host paths and
Slurm values in the environment before launching; the `:?` guards fail early if
a required value is missing.

Run from an RL checkout that contains this directory:

```bash
: "${RL_CHECKOUT_DIR:?set to the host RL checkout path}"
cd "${RL_CHECKOUT_DIR}"

stamp=$(date +%Y%m%d-%H%M%S)
export EXP_NAME="${EXP_NAME:-swe-qwen35-r2e-${stamp}}"

export REPO_LOCATION="$PWD"
export RECIPE=qwen_35/configs/grpo_qwen35_397b_swe_openhands_async.yaml

: "${CONTAINER_IMAGE_PATH:?set to a Gym-capable NeMo RL image}"
: "${SLURM_ACCOUNT:?set to the Slurm account}"
: "${SLURM_PARTITION:?set to the Slurm partition}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export TRAIN_NODES="${TRAIN_NODES:-32}"
export GEN_NODES="${GEN_NODES:-32}"
export NODES="${NODES:-$((TRAIN_NODES + GEN_NODES))}"
export SLURM_TIME="${SLURM_TIME:-4:00:00}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:${GPUS_PER_NODE}}"

: "${HF_CKPT_PATH:?set to the host Qwen 3.5 HF checkpoint directory}"
export CONTAINER_HF_CKPT_PATH="${CONTAINER_HF_CKPT_PATH:-${HF_CKPT_PATH}}"

: "${NRL_MEGATRON_CHECKPOINT_DIR:?set to the host Megatron checkpoint directory}"
export CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR="${CONTAINER_NRL_MEGATRON_CHECKPOINT_DIR:-/inputs/nemo_gym/mcore_ckpt}"

: "${NEMO_GYM_SWE_TRAIN_DATA_PATH:?set to the host SWE train JSONL}"
: "${NEMO_GYM_SWE_VALIDATION_DATA_PATH:?set to the host SWE validation JSONL}"
export CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_TRAIN_DATA_PATH:-/inputs/nemo_gym/data/train.jsonl}"
export CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH="${CONTAINER_NEMO_GYM_SWE_VALIDATION_DATA_PATH:-/inputs/nemo_gym/data/validation.jsonl}"

: "${NEMO_GYM_SWE_SIF_DIR:?set to the host directory containing SWE task SIF images}"
export CONTAINER_NEMO_GYM_SWE_SIF_DIR="${CONTAINER_NEMO_GYM_SWE_SIF_DIR:-/inputs/nemo_gym/sif}"

# Optional: set when the dataset references SIF images outside the primary SIF root.
# export NEMO_GYM_SWE_FALLBACK_SIF_DIR="${FALLBACK_SIF_ROOT:?set to the fallback SIF root}"

# Optional: some clusters need FUSE for nested SIF execution.
# export EXTRA_MOUNTS="${EXTRA_MOUNTS:+${EXTRA_MOUNTS},}/dev/fuse:/dev/fuse"

# Default is 65535 for a 64k context run, leaving one token of headroom so
# vLLM does not reject prompts at exactly/just above max_model_len.
export NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS=65535
```

Then launch. The recipe owns the R2E SIF formatter list; do not pass it as a
Hydra CLI override because the `{instance_id}` placeholders are parsed as
override grammar unless heavily escaped. The recipe also reads the model path,
data paths, and SIF directory from the `CONTAINER_*` environment variables set
above, so normal launches do not need path overrides.

```bash
bash examples/nemo_gym/launch_nemo_gym_multinode_training.sh \
  "logger.wandb_enabled=False"
```

Notes:

- Some Slurm partitions require `SBATCH_GRES` or `SLURM_GRES`; without a GPU
  TRES request, the scheduler may reject the job.
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
