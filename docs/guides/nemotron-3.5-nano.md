# Nemotron 3.5 Nano

This guide describes the reference NeMo RL recipes for post-training Nemotron
3.5 Nano on four-GPU GB200 nodes:

- SWE reinforcement learning with executable software-engineering environments.
- RLVR with external GenRM and general-purpose judge pools plus an in-cluster
  safety judge.

The recipes and shared launcher are under
`examples/nemo_gym/nemotron-3.5-nano/`.

## Reference topology

| Recipe | Training | Policy generation | Gym judges | External GenRM | External NL2Bash | Total |
|---|---:|---:|---:|---:|---:|---:|
| SWE | 16 nodes | 32 nodes | 0 nodes | 0 nodes | 0 nodes | 48 nodes |
| RLVR | 32 nodes | 32 nodes | 2 nodes | 16 nodes | 4 nodes | 86 nodes |

The RLVR launcher reserves 20 nodes outside the NeMo RL Ray cluster. It starts
eight independent TP=8, DP=1 GenRM servers on 16 nodes and four independent
TP=4, DP=1 NL2Bash judge servers on four nodes. Each pool has a lightweight
load balancer. Training starts only after both complete pools and load
balancers are healthy. The two Gym nodes host the TP=4, DP=2 safety judge.

Both reference profiles assume four GPUs per node. Override the node counts,
`GPUS_PER_NODE`, and `SEGMENT_SIZE` only after confirming that all model
parallelism and judge allocations still fit.

## Prepare the code and containers

Clone NeMo RL with its submodules:

```bash
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git
cd RL
```

Prepare:

- A NeMo RL training container containing the required vLLM version.
- A NeMo Skills sandbox container for executable Gym environments.
- A shared cache directory visible from every allocated node.
- Mounts that expose the model, data, cache, sandbox images, and repository
  checkout inside the training container.

On enroot-based clusters, `CONTAINER` and `SANDBOX_CONTAINER` may be `.sqsh`
paths. `EXTRA_MOUNTS` is a comma-separated list of `host:container` mappings.

The inline external-service launcher currently expects `BASE_LOG_DIR`,
`EXTERNAL_VLLM_TOOLS_DIR_HOST`, and absolute local GenRM, NL2Bash, or plugin
paths to be under `/lustre`, which is mounted into its service containers. A
Hugging Face model ID may be used instead of an absolute model path.

## Prepare the inputs

Set `MODEL_PATH` to a transformers-compatible Nemotron 3.5 Nano checkpoint.
Prepare separate JSONL files for training and validation in the format expected
by `NemoGymDataset`.

### SWE sandbox images

The SWE recipe expects `SIF_DIR` to contain these layouts:

```text
SIF_DIR/
├── swerebench/{instance_id}.sif
├── nv_internal/{instance_id}.sif
├── r2e_gym/{instance_id}.sif
├── swegym/sweb.eval.arm64.{instance_id}.sif
├── swebench/swe-bench.eval.arm64.{instance_id}.sif
└── mercor/swebenchpro_ots/{instance_id}.sif
```

Only directories represented in your dataset need to contain images. Update
`container_formatter` in `swe.yaml` if your image naming convention differs.

### RLVR judges

The RLVR profile requires:

- `GENRM_MODEL`: GenRM checkpoint or Hugging Face model ID.
- `GENRM_REASONING_PARSER`: shared path to the vLLM reasoning-parser plugin
  used by that checkpoint.
- `NL2BASH_JUDGE_MODEL`: general-purpose judge checkpoint or model ID.
- `SAFETY_JUDGE_MODEL`: safety judge checkpoint or model ID.

The reference external deployments use expert parallelism and serve the
OpenAI-compatible model name `model`. The launcher injects both load-balancer
URLs and model names into Gym after all backends become healthy.
The model-specific vLLM arguments are intentionally defined in
`nano35_launch.sh`; `tools/external_gym_vllm/run_in_allocation.sh` only implements
the generic lifecycle for the named external pools.

## Launch script

Both recipes use:

```text
examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh
```

Common required variables:

| Variable | Purpose |
|---|---|
| `EXP_NAME` | Slurm job name, W&B run name, and output-directory suffix. |
| `MODEL_PATH` | Starting Nemotron 3.5 Nano checkpoint. |
| `TRAIN_PATH`, `VAL_PATH` | Training and validation JSONL files. |
| `CONTAINER` | NeMo RL training container. |
| `SANDBOX_CONTAINER` | NeMo Skills sandbox container. |
| `PERSISTENT_CACHE` | Shared vLLM, Triton, Inductor, and model cache root. |
| `SLURM_PARTITION`, `SLURM_ACCOUNT` | Slurm allocation settings. |

Useful optional variables:

| Variable | Default | Purpose |
|---|---|---|
| `RESULTS_DIR` | `results/$EXP_NAME` | Checkpoints and per-submission logs. |
| `BASE_LOG_DIR` | `$RESULTS_DIR/ray_logs` | Ray and external-judge logs. |
| `EXTRA_MOUNTS` | empty | Additional container mounts. |
| `WALLTIME` | `4:00:00` | Slurm time limit. |
| `SLURM_QOS`, `SLURM_RESERVATION`, `EXCLUDE_NODES` | empty | Optional Slurm controls. |
| `WANDB_API_KEY` | unset | Enables W&B when set. |
| `WANDB_PROJ` | `nemotron-3.5-nano` | W&B project. |
| `HF_HOME`, `HF_TOKEN` | unset | Hugging Face cache and token. |
| `USE_SNAPSHOT` | `1` | Snapshot tracked source before submission. |
| `USE_CUSTOM_VLLM` | `0` | Set to `1` to source the repository's custom vLLM environment. |
| `DRY_RUN` | `0` | Print the resolved training command without submitting. |

Additional Hydra overrides may follow the recipe name.

## Run SWE

```bash
EXP_NAME=nano35-swe \
MODEL_PATH=/path/to/nemotron-3.5-nano-checkpoint \
TRAIN_PATH=/path/to/swe-train.jsonl \
VAL_PATH=/path/to/swe-validation.jsonl \
SIF_DIR=/path/to/swe-sif-root \
CONTAINER=/path/to/nemo-rl-container.sqsh \
SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
PERSISTENT_CACHE=/path/to/shared/cache \
EXTRA_MOUNTS=/shared:/shared,/lustre:/lustre \
SLURM_PARTITION=your-partition \
SLURM_ACCOUNT=your-account \
bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe
```

The reference SWE configuration uses TP=4, CP=16, EP=32, GBS=512, 32 prompts
per step, 16 generations per prompt, an agent concurrency of 1024, and a
1,200-second SWE test timeout.

## Run RLVR

```bash
EXP_NAME=nano35-rlvr \
MODEL_PATH=/path/to/nemotron-3.5-nano-checkpoint \
TRAIN_PATH=/path/to/rlvr-train.jsonl \
VAL_PATH=/path/to/rlvr-validation.jsonl \
GENRM_MODEL=/path/to/genrm-checkpoint \
GENRM_REASONING_PARSER=/path/to/ultra_v3_reasoning_parser.py \
NL2BASH_JUDGE_MODEL=/path/to/general-judge-checkpoint \
SAFETY_JUDGE_MODEL=/path/to/safety-judge-checkpoint \
CONTAINER=/path/to/nemo-rl-container.sqsh \
SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
PERSISTENT_CACHE=/path/to/shared/cache \
RESULTS_DIR=/lustre/path/to/results/nano35-rlvr \
BASE_LOG_DIR=/lustre/path/to/ray-logs/nano35-rlvr \
EXTRA_MOUNTS=/shared:/shared,/lustre:/lustre \
SLURM_PARTITION=your-partition \
SLURM_ACCOUNT=your-account \
bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh rlvr
```

The reference RLVR configuration uses TP=4, CP=4, EP=16, GBS=8192, 512
prompts per step, and 16 generations per prompt. The external defaults are
eight TP=8 GenRM replicas on 16 four-GPU nodes and four TP=4 NL2Bash replicas
on four more nodes.

## Inspect a launch

Set `DRY_RUN=1` to verify the node split, mounts, paths, and generated Hydra
overrides without submitting:

```bash
DRY_RUN=1 \
EXP_NAME=nano35-dry-run \
MODEL_PATH=/path/to/model \
TRAIN_PATH=/path/to/train.jsonl \
VAL_PATH=/path/to/validation.jsonl \
SIF_DIR=/path/to/swe-sif-root \
CONTAINER=/path/to/container.sqsh \
SANDBOX_CONTAINER=/path/to/sandbox.sqsh \
PERSISTENT_CACHE=/path/to/cache \
SLURM_PARTITION=your-partition \
SLURM_ACCOUNT=your-account \
bash examples/nemo_gym/nemotron-3.5-nano/nano35_launch.sh swe
```

After submission, the launcher prints the Slurm, Ray, checkpoint, and
per-submission log directories. Reusing the same `EXP_NAME` and `RESULTS_DIR`
allows the checkpoint manager to resume a later submission.
