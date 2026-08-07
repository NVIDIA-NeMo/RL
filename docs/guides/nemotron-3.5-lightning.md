# Nemotron 3.5 Lightning

This guide explains how to post-train Nemotron 3.5 Lightning with NeMo RL on
**GB200 NVL72** (ARM64 / aarch64) hardware.

## Overview

The reference recipe trains
[`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
with asynchronous GRPO and NeMo Gym rewards. Policy generation and training
run on separate node pools. GenRM and NL2Bash are deployed as external vLLM
pools in the same Slurm heterogeneous allocation, while the content-safety
judge runs in the Gym pool.

The recipe and shared launcher are under
`examples/nemo_gym/nemotron-3.5-lightning/`:

- `rlvr.yaml` contains the training, generation, and Gym configuration.
- `lightning35_launch.sh` handles Slurm submission, code snapshots, mounts,
  persistent caches, and deployment-specific overrides.

### Reference topology

| Training | Policy generation | Gym safety judge | External GenRM | External NL2Bash | Total |
|---:|---:|---:|---:|---:|---:|
| 32 nodes | 32 nodes | 2 nodes | 16 nodes | 4 nodes | 86 nodes |

The launcher reserves the 20 external-service nodes outside the NeMo RL Ray
cluster. It starts eight independent TP=8, DP=1 GenRM servers on 16 nodes and
four independent TP=4, DP=1 NL2Bash servers on four nodes. Each service pool
has a lightweight load balancer. Training starts only after every backend and
both load balancers are healthy. The two Gym nodes host a TP=4, DP=2 safety
judge.

The reference topology assumes four GPUs per node. Override node counts,
`GPUS_PER_NODE`, or segment sizes only after confirming that the model
parallelism still fits and every allocated GPU is used.

## Container

Nemotron 3.5 Lightning uses vLLM and requires an **aarch64 (arm64)** image for
GB200 NVL72 nodes. Prebake the Gym virtual environments used by `rlvr.yaml` to
avoid building them on every training launch. From the root of the NeMo RL
repository, build and push the image:

```bash
docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  -f docker/Dockerfile \
  --target release \
  -t <your-registry>/nemo-rl:main-lightning35-prefetched-venvs \
  --push \
  --build-context nemo-rl=. \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml" \
  .
```

Build arguments:

- `NEMO_GYM_PREFETCH_CONFIGS` builds the Gym virtual environments referenced
  by the RLVR config into the image.
- `SKIP_SGLANG_BUILD=1` skips SGLang because this recipe uses vLLM.
- `SKIP_TRTLLM_BUILD=1` skips TensorRT-LLM because this recipe does not use it.
- `MAX_JOBS` controls parallel build jobs; tune it for the build machine.
- `--build-context nemo-rl=.` builds from the current checkout. Without it,
  the Dockerfile pulls `NVIDIA-NeMo/RL.git#main`.

On a Slurm cluster using [enroot](https://github.com/NVIDIA/enroot), convert
the image to squashfs:

```bash
enroot import -o nemo-rl-lightning35.sqsh \
  docker://<your-registry>/nemo-rl:main-lightning35-prefetched-venvs
```

Pass the resulting `.sqsh` path as `CONTAINER`. A registry image URI can be
used instead on clusters that do not require a local squashfs image.

## Download and prepare the data

Download the
[`nvidia/Nemotron-RL-Lightning-Training-Blend`](https://huggingface.co/datasets/nvidia/Nemotron-RL-Lightning-Training-Blend)
dataset and restore its source-backed placeholders. This follows the same
general workflow as the
[Nemotron 3 Nano data preparation](nemotron-3-nano.md#download-and-prepare-the-data):

```bash
uvx --from huggingface-hub hf download \
  nvidia/Nemotron-RL-Lightning-Training-Blend \
  --repo-type dataset \
  --local-dir data

chmod +x data/fill_placeholders.py
./data/fill_placeholders.py --input-dir data --output-dir data/restored
```

The resulting `data/restored/rlvr.jsonl` is already in the format expected by
`NemoGymDataset`. The released blend has one training split, and validation is
disabled in the reference recipe, so pass this file as both `TRAIN_PATH` and
`VAL_PATH`:

```bash
TRAIN_PATH=$PWD/data/restored/rlvr.jsonl
VAL_PATH=$PWD/data/restored/rlvr.jsonl
```

Dataset rows select their Gym agent through `agent_ref`; therefore, the agents
named by the data must be included in `env.nemo_gym.config_paths` in
`rlvr.yaml`.

## Prepare the models

Set `MODEL_PATH` to a Transformers-compatible Nemotron 3.5 Lightning
checkpoint. To start from the released checkpoint, use:

```bash
MODEL_PATH=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
```

The reward stack also requires:

- `GENRM_MODEL`: GenRM checkpoint or Hugging Face model ID.
- `GENRM_REASONING_PARSER`: shared path to the vLLM reasoning-parser plugin
  used by the GenRM checkpoint.
- `NL2BASH_JUDGE_MODEL`: general-purpose judge checkpoint or model ID.
- `SAFETY_JUDGE_MODEL`: content-safety judge checkpoint or model ID.

The external services expose the OpenAI-compatible model name `model`. The
launcher injects the load-balancer URLs and served model names into the Gym
configuration after both pools are healthy. Model-specific vLLM arguments
remain in `lightning35_launch.sh`; the helpers under
`tools/external_gym_vllm/` implement the generic external-pool lifecycle.

Absolute service model, parser, tool, state, and log paths must be visible in
the external-service containers. Add their shared filesystem to
`EXTRA_MOUNTS`. Hugging Face model IDs do not require a filesystem mount.

## Prepare the code

Clone NeMo RL and its submodules:

```bash
git clone --recursive -b main https://github.com/NVIDIA-NeMo/RL.git
cd RL
```

## Build the sandbox container

Some Gym environments execute code or verification tools in a sandbox. Build
the sandbox image from the
[NeMo Skills Dockerfile](https://github.com/NVIDIA-NeMo/Skills/blob/main/dockerfiles/Dockerfile.sandbox):

```bash
git clone https://github.com/NVIDIA-NeMo/Skills.git
cd Skills
docker build -t nemo-skills-sandbox:latest -f dockerfiles/Dockerfile.sandbox .
```

For Slurm clusters using enroot, convert it to a `.sqsh`:

```bash
enroot import -o nemo-skills-sandbox.sqsh \
  dockerd://nemo-skills-sandbox:latest
```

Pass this image as `SANDBOX_CONTAINER` when launching training.

## Launch script

Run `examples/nemo_gym/nemotron-3.5-lightning/lightning35_launch.sh` from the
repository root. The launcher handles Slurm submission, source snapshots,
persistent caches, container mounts, external vLLM services, and
deployment-specific Hydra overrides. Training hyperparameters remain in
`rlvr.yaml`.

Required variables:

| Variable | Purpose |
|---|---|
| `EXP_NAME` | Slurm job name, W&B run name, and output-directory suffix. Reuse it with the same `RESULTS_DIR` to resume. |
| `MODEL_PATH` | Initial Nemotron 3.5 Lightning policy checkpoint. |
| `TRAIN_PATH`, `VAL_PATH` | Restored `rlvr.jsonl` from the Lightning training blend. |
| `GENRM_MODEL` | GenRM checkpoint or Hugging Face model ID. |
| `GENRM_REASONING_PARSER` | Shared path to the GenRM vLLM parser plugin. |
| `NL2BASH_JUDGE_MODEL` | General-purpose judge checkpoint or model ID. |
| `SAFETY_JUDGE_MODEL` | Content-safety judge checkpoint or model ID. |
| `CONTAINER` | NeMo RL image (`.sqsh` path or registry image URI). |
| `SANDBOX_CONTAINER` | Sandbox image from [Build the sandbox container](#build-the-sandbox-container). |
| `PERSISTENT_CACHE` | Shared vLLM, Triton, Inductor, and model cache directory. |
| `SLURM_PARTITION`, `SLURM_ACCOUNT` | Slurm allocation settings. |

Useful optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `RESULTS_DIR` | `results/$EXP_NAME` | Stable checkpoint root and per-submission logs. |
| `BASE_LOG_DIR` | `$RESULTS_DIR/ray_logs` | Ray and external-service logs. |
| `EXTRA_MOUNTS` | empty | Comma-separated `host:container` mount pairs. |
| `WALLTIME` | `4:00:00` | Slurm time limit. |
| `SLURM_QOS`, `SLURM_RESERVATION`, `EXCLUDE_NODES` | empty | Optional Slurm controls. |
| `NUM_TRAIN_NODES`, `NUM_GEN_NODES`, `NUM_GYM_NODES` | `32`, `32`, `2` | Nodes in the NeMo RL allocation component. |
| `GENRM_REPLICAS`, `NUM_GENRM_NODES` | `8`, `16` | External GenRM pool size. |
| `NL2BASH_REPLICAS`, `NUM_NL2BASH_NODES` | `4`, `4` | External general-judge pool size. |
| `WANDB_API_KEY` | unset | Enables W&B when set. |
| `WANDB_PROJ` | `nemotron-3.5-lightning` | W&B project. |
| `HF_HOME`, `HF_TOKEN` | unset | Hugging Face cache and gated-model token. |
| `USE_SNAPSHOT` | `1` | Snapshot tracked source before submission. |
| `USE_CUSTOM_VLLM` | `0` | Source the repository custom vLLM environment when set to `1`. |
| `DRY_RUN` | `0` | Print the resolved command without submitting. |

Additional positional arguments after `rlvr` are forwarded as Hydra
overrides.

## RLVR

The reference configuration uses TP=4, CP=4, EP=16, PP=1, a global batch size
of 8192, 512 prompts per step, and 16 generations per prompt.

```bash
EXP_NAME=lightning35-rlvr \
MODEL_PATH=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
TRAIN_PATH=$PWD/data/restored/rlvr.jsonl \
VAL_PATH=$PWD/data/restored/rlvr.jsonl \
GENRM_MODEL=/path/to/genrm-checkpoint \
GENRM_REASONING_PARSER=/path/to/ultra_v3_reasoning_parser.py \
NL2BASH_JUDGE_MODEL=/path/to/general-judge-checkpoint \
SAFETY_JUDGE_MODEL=/path/to/safety-judge-checkpoint \
CONTAINER=/path/to/nemo-rl-lightning35.sqsh \
SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
PERSISTENT_CACHE=/path/to/shared/cache \
RESULTS_DIR=/path/to/results/lightning35-rlvr \
BASE_LOG_DIR=/path/to/results/lightning35-rlvr/ray_logs \
EXTRA_MOUNTS=/shared:/shared \
SLURM_PARTITION=your-partition \
SLURM_ACCOUNT=your-account \
bash examples/nemo_gym/nemotron-3.5-lightning/lightning35_launch.sh rlvr
```
