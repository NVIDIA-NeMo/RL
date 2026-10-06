# Nemotron 3.5 Super

This guide describes post-training Nemotron 3.5 Super with NeMo RL and
[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) on **GB200 NVL72** hardware.
Training uses Megatron; policy generation uses vLLM.

## Overview

The workflow combines reinforcement learning with verifiable rewards (RLVR),
specialized teacher training, and multi-teacher on-policy distillation (MOPD):

1. **Student RLVR** trains an SFT checkpoint using GRPO and verifiable rewards.
2. **RLHF, reasoning, and vision teachers** each start from the Student RLVR
   checkpoint and specialize in their respective domains.
3. **MOPD** initializes the student from Student RLVR and distills the teacher
   panel. Student RLVR also serves as the general teacher.

```text
        ┌─────┐
        │ SFT │
        └──┬──┘
           v
    ┌──────────────┐
    │ Student RLVR │
    └──┬───────────┘
       │
       │   ┌────────────────────┐
       ├──>│  General Teacher   │──┐
       │   └────────────────────┘  │
       │   ┌────────────────────┐  │
       ├──>│    RLHF Teacher    │──┤
       │   └────────────────────┘  │
       │   ┌────────────────────┐  │
       ├──>│  Reasoning Teacher │──┤
       │   └────────────────────┘  │
       │   ┌────────────────────┐  │
       ├──>│   Vision Teacher   │──┤
       │   └────────────────────┘  │
       │                           │
       v                           v
   ┌─────────┐               ┌──────────┐
   │ Student │──────────────>│   MOPD   │
   └─────────┘               └────┬─────┘
                                  v
                         ┌──────────────────┐
                         │ Final checkpoint │
                         └──────────────────┘
```

All five stages use
[`super35_launch.sh`](../../../../examples/nemo_gym/nemotron-3.5-super/super35_launch.sh)
and standalone YAML files in `examples/nemo_gym/nemotron-3.5-super/`.
Teacher checkpoints are frozen during MOPD; the student is trained on its own
rollouts using teacher log probabilities.

## Code and container

Use the `super-v3.5-posttraining` branch and its pinned dependencies:

```bash
git clone --recursive --branch super-v3.5-posttraining https://github.com/NVIDIA-NeMo/RL.git
cd RL
```

Build an ARM64 image with the Gym environments for all five stages:

```bash
docker buildx build --platform linux/arm64 --progress=plain \
  -f docker/Dockerfile --target release \
  --build-context nemo-rl=. \
  --build-arg MAX_JOBS=8 \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg SKIP_TRTLLM_BUILD=1 \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS="\
--env-file examples/nemo_gym/nemotron-3.5-super/prefetch_env.json \
examples/nemo_gym/nemotron-3.5-super/student_rlvr.yaml \
examples/nemo_gym/nemotron-3.5-super/rlhf_teacher.yaml \
examples/nemo_gym/nemotron-3.5-super/reasoning_teacher.yaml \
examples/nemo_gym/nemotron-3.5-super/vision_teacher.yaml \
examples/nemo_gym/nemotron-3.5-super/mopd.yaml" \
  -t <your-registry>/nemo-rl:super35 --push .

enroot import -o nemo-rl-super35.sqsh docker://<your-registry>/nemo-rl:super35
```

Slurm submission requires Pyxis/enroot.

RLVR, reasoning, and MOPD also use a
[NeMo Skills sandbox](https://github.com/NVIDIA-NeMo/Skills/blob/main/dockerfiles/Dockerfile.sandbox)
for code/tool environments. Provide a compatible sandbox image through
`SANDBOX_CONTAINER`.

## Judge services

All judges and reward models run in **external vLLM pools** on dedicated
nodes in the same Slurm allocation. The launcher starts them and connects Gym
automatically.

Set `JUDGE_CONTAINER` to a vLLM/Ray image supporting the judges used by your stage:
[Nemotron Ultra GenRM](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM),
Qwen3-235B-A22B-Instruct-2507-FP8, the content-safety reasoning model, or
DeepSeek-V4-Flash. Pass their Hugging Face model IDs or local checkpoint paths
in the launch commands below.
Serving settings and per-model container overrides are in
[`judge_pools.sh`](../../../../examples/nemo_gym/nemotron-3.5-super/judge_pools.sh).
See [External Gym vLLM pools](../../../../tools/external_gym_vllm/README.md)
for serving-image requirements.

## Launch configuration

Default allocations assume **four GPUs per GB200 node** and include judge nodes.
Training and serving settings are specified in the linked configs.

| Stage / config | Training nodes | Inference nodes | Judge nodes | Total nodes |
|---|---:|---:|---:|---:|
| [Student RLVR](../../../../examples/nemo_gym/nemotron-3.5-super/student_rlvr.yaml) | 32 | 112 | 26 | 170 |
| [RLHF teacher](../../../../examples/nemo_gym/nemotron-3.5-super/rlhf_teacher.yaml) | 16 | 16 | 32 | 64 |
| [Reasoning teacher](../../../../examples/nemo_gym/nemotron-3.5-super/reasoning_teacher.yaml) | 16 | 44 | 4 | 64 |
| [Vision teacher](../../../../examples/nemo_gym/nemotron-3.5-super/vision_teacher.yaml) | 8 | 8 | 0 | 16 |
| [MOPD](../../../../examples/nemo_gym/nemotron-3.5-super/mopd.yaml) | 64 | 48 | 0 | 112 |

MOPD inference comprises 30 rollout nodes and 18 teacher nodes.

Run the commands below from the repository root, replacing the paths, Slurm
account, and partition for your cluster.

The checkout, checkpoints, datasets, and result directories must all be under
`SHARED_ROOT`, which is mounted at the same path on every node and in sandboxes.
Use `EXTRA_MOUNTS` for image stores outside it. Optionally set `SLURM_QOS` or
`SLURM_RESERVATION` for your cluster.

Prepend `DRY_RUN=1` to any launch command to inspect its allocation and command
without submitting a job. Set `NUM_TRAIN_NODES` and `NUM_GEN_NODES` to change the
allocation; append training config overrides after the stage name.

## Stage 1 — Student RLVR

Start from the SFT checkpoint and specify the three judge checkpoints:

```bash
SHARED_ROOT=/shared \
CONTAINER=/shared/images/nemo-rl-super35.sqsh \
JUDGE_CONTAINER=/shared/images/vllm-judges.sqsh \
SANDBOX_CONTAINER=/shared/images/nemo-skills-sandbox.sqsh \
DATA_ROOT=/shared/data/super35 \
SLURM_ACCOUNT=<your-account> \
SLURM_PARTITION=<your-partition> \
WALLTIME=4:00:00 \
GENRM_CHECKPOINT=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM \
NL2BASH_CHECKPOINT=/shared/checkpoints/qwen3-235b-a22b-instruct-2507-fp8 \
SAFETY_CHECKPOINT=/shared/checkpoints/content-safety-reasoning \
MODEL_PATH=/shared/checkpoints/super35-sft \
TRAIN_PATH=/shared/data/super35/rlvr.train.jsonl \
EXP_NAME=super35-student-rlvr \
bash examples/nemo_gym/nemotron-3.5-super/super35_launch.sh student_rlvr
```

## Stage 2 — Teacher training

All three teachers start from the Student RLVR checkpoint and can train
independently.

### RLHF teacher

```bash
SHARED_ROOT=/shared \
CONTAINER=/shared/images/nemo-rl-super35.sqsh \
JUDGE_CONTAINER=/shared/images/vllm-judges.sqsh \
SLURM_ACCOUNT=<your-account> \
SLURM_PARTITION=<your-partition> \
WALLTIME=4:00:00 \
RLVR_CHECKPOINT=/shared/checkpoints/super35-rlvr-hf \
GENRM_CHECKPOINT=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM \
TRAIN_PATH=/shared/data/super35/rlhf.train.jsonl \
EXP_NAME=super35-rlhf-teacher \
bash examples/nemo_gym/nemotron-3.5-super/super35_launch.sh rlhf_teacher
```

### Reasoning teacher

```bash
SHARED_ROOT=/shared \
CONTAINER=/shared/images/nemo-rl-super35.sqsh \
JUDGE_CONTAINER=/shared/images/vllm-judges.sqsh \
SANDBOX_CONTAINER=/shared/images/nemo-skills-sandbox.sqsh \
DATA_ROOT=/shared/data/super35 \
SLURM_ACCOUNT=<your-account> \
SLURM_PARTITION=<your-partition> \
WALLTIME=4:00:00 \
RLVR_CHECKPOINT=/shared/checkpoints/super35-rlvr-hf \
REASONING_JUDGE_CHECKPOINT=/shared/checkpoints/deepseek-v4-flash \
TRAIN_PATH=/shared/data/super35/reasoning.train.jsonl \
EXP_NAME=super35-reasoning-teacher \
bash examples/nemo_gym/nemotron-3.5-super/super35_launch.sh reasoning_teacher
```

### Vision teacher

```bash
SHARED_ROOT=/shared \
CONTAINER=/shared/images/nemo-rl-super35.sqsh \
SLURM_ACCOUNT=<your-account> \
SLURM_PARTITION=<your-partition> \
WALLTIME=4:00:00 \
RLVR_CHECKPOINT=/shared/checkpoints/super35-rlvr-hf \
TRAIN_PATH=/shared/data/super35/vision.train.jsonl \
EXP_NAME=super35-vision-teacher \
bash examples/nemo_gym/nemotron-3.5-super/super35_launch.sh vision_teacher
```

## Stage 3 — MOPD

Use Hugging Face exports of the three teacher outputs. Student RLVR is both the
initial student and the general teacher:

```bash
SHARED_ROOT=/shared \
CONTAINER=/shared/images/nemo-rl-super35.sqsh \
SANDBOX_CONTAINER=/shared/images/nemo-skills-sandbox.sqsh \
DATA_ROOT=/shared/data/super35 \
SLURM_ACCOUNT=<your-account> \
SLURM_PARTITION=<your-partition> \
WALLTIME=4:00:00 \
RLVR_CHECKPOINT=/shared/checkpoints/super35-rlvr-hf \
RLHF_TEACHER_PATH=/shared/checkpoints/super35-rlhf-hf \
REASONING_TEACHER_PATH=/shared/checkpoints/super35-reasoning-hf \
VISION_TEACHER_PATH=/shared/checkpoints/super35-vision-hf \
TRAIN_PATH=/shared/data/super35/mopd.train.jsonl \
EXP_NAME=super35-mopd \
bash examples/nemo_gym/nemotron-3.5-super/super35_launch.sh mopd
```

## Monitoring and resuming

Results default to `$SHARED_ROOT/results/$EXP_NAME`. The launcher prints the
`ray-driver.log` path after submission. TensorBoard is enabled;
for W&B, pass `WANDB_API_KEY` inline and append `logger.wandb_enabled=true`.

Resubmit the same command and experiment name to resume the latest checkpoint.
Use a new experiment name for a fresh run. If a run has reached its step limit,
append `grpo.max_num_steps=<new-limit>` when resubmitting.
