# Nemotron 3 Super

**Technical Report:** [NVIDIA Nemotron-3-Super Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)

This guide explains how to post-train the Nemotron 3 Super model using NeMo RL.

## Container

The released NeMo RL base container is:

```
nvcr.io/nvidia/nemo-rl:v0.7.0
```

To reduce launch-time setup, prebake the NeMo-Gym virtual environments into a derived image:

```text
docker buildx build \
  -t your-registry/nemo-rl:v0.7.0.prefetched_venvs \
  --push \
  -f- . <<'EOF'
FROM nvcr.io/nvidia/nemo-rl:v0.7.0

ARG NEMO_GYM_CUDA=cu130
ARG NEMO_GYM_VLLM_VERSION=0.17.0
ARG WHEEL_ARCH=x86_64
RUN <<'RUNEOF' bash -exu -o pipefail
cd /opt/nemo-rl
GYM_VLLM_OVERRIDE=/opt/nemo-rl/.gym-vllm-override.txt
printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s+%s-cp38-abi3-manylinux_2_35_%s.whl\n' \
    "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_CUDA}" "${WHEEL_ARCH}" > "${GYM_VLLM_OVERRIDE}"

UV_TORCH_BACKEND="${NEMO_GYM_CUDA}" \
UV_OVERRIDE="${GYM_VLLM_OVERRIDE}" \
UV_LINK_MODE=symlink uv run python examples/nemo_gym/prefetch_venvs.py \
    examples/nemo_gym/prefetch_super_all_envs.yaml
rm -f "${GYM_VLLM_OVERRIDE}"
RUNEOF
EOF
```

Push the derived image to a registry visible to the training cluster and use it as `CONTAINER` in the launch commands below.

## Download and prepare the data

```bash
# Download RL data blends (rlvr1, rlvr2, rlvr3, swe1, swe2, rlhf)
uvx --from huggingface-hub hf download nvidia/Nemotron-RL-Super-Training-Blends --repo-type dataset --local-dir=data_with_placeholders


# Fill in placeholders in data blends
chmod +x data_with_placeholders/fill_placeholders.py
./data_with_placeholders/fill_placeholders.py --input-dir data_with_placeholders --output-dir data_filled


# Create train/val splits for each data blend (last 100 rows held out for validation)
for f in data_filled/*.jsonl; do
  name=$(basename "$f" .jsonl)
  mkdir -p "data/$name"
  head -n -100 "$f" > "data/$name/train-split.jsonl"
  tail -n 100 "$f" > "data/$name/val-split.jsonl"
done
```

## Prepare the code
```bash
git clone --recursive https://github.com/NVIDIA-NeMo/RL.git
cd RL
```

## Training the model

RL training for Nemotron 3 Super consists of 3 main stages:

1. Reinforcement Learning with Verifiable Rewards (RLVR)
2. SWE RL
3. RLHF with length penalty to reduce verbosity

The RLVR stage consists of 3 sub-stages with different data blends and the SWE RL stage consists of 2 sub-stages, for 6 total stages.

### Build sandbox container

Several [Gym](https://github.com/NVIDIA-NeMo/Gym) environments used during training rely on a sandbox container for code execution, including [NeMo-Skills tools](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/ns_tools) (stateful Python execution with math verification) and [Lean4 formal proof verification](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/math_formal_lean). To build the sandbox container, use the [NeMo-Skills Dockerfile](https://github.com/NVIDIA-NeMo/Skills/blob/main/dockerfiles/Dockerfile.sandbox):

```bash
git clone https://github.com/NVIDIA-NeMo/Skills.git
cd Skills
git checkout a5da59797890284af4df1c2a9c10990b33623a9d
docker build -t nemo-skills-sandbox:latest -f dockerfiles/Dockerfile.sandbox .
```

For SLURM clusters using [enroot](https://github.com/NVIDIA/enroot), convert the image to a `.sqsh` file:

```bash
enroot import -o nemo-skills-sandbox.sqsh dockerd://nemo-skills-sandbox:latest
```

### Launch script

Each stage is launched with `super_launch.sh`. Set the following variables before running:

* `$DATA_DIR`: Path to the **final** `data` directory produced in [Download and prepare the data](#download-and-prepare-the-data).
* `$SANDBOX_CONTAINER`: The sandbox container image from [Build sandbox container](#build-sandbox-container) (`.sqsh` path or registry URI).
* `$PERSISTENT_CACHE`: Path to a directory used to store caches for vLLM and FlashInfer.
* `$EXTRA_MOUNTS`: Comma-separated `host:container` mount pairs for shared filesystems that your data, models, and checkpoints reside on (e.g. `EXTRA_MOUNTS=/scratch:/scratch,/lustre:/lustre`). The launch script only mounts the code snapshot directory by default.
* `$SIF_DIR`: *(Stage 2.2 only)* Path to the directory containing Apptainer `.sif` images for the SWE-bench environments. These are converted Docker images from R2E-Gym, SWE-Gym, and SWE-Bench Verified. See [Stage 2.2](#stage-22---swe-2-64-h100-gpu-nodes) for download instructions.
* `$SLURM_PARTITION`
* `$SLURM_ACCOUNT`

`MODEL_PATH` is the input checkpoint for each stage. Stage 1.1 starts from the SFT checkpoint; every subsequent stage takes the output of the previous one.

The launch examples below use the H100 prod configs. For stages with GPU judge servers, set `SBATCH_NUM_NODES` to the total node count so SLURM allocates both the NeMo-RL nodes and the extra NeMo-Gym nodes. Lower-DP H100 variants are available under `examples/nemo_gym/nemotron-3-super/small_scale/`.

### Stage 1 - RLVR

#### Stage 1.1 - RLVR 1 (183 H100 GPU nodes)
```bash
EXP_NAME=stage1.1-rlvr1 \
CONFIG_PATH=examples/nemo_gym/nemotron-3-super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/sft_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr1/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr1/val-split.jsonl \
CONTAINER=your-registry/nemo-rl:v0.7.0.prefetched_venvs \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SBATCH_NUM_NODES=183 \
bash super_launch.sh
```

#### Stage 1.2 - RLVR 2 (183 H100 GPU nodes)
```bash
EXP_NAME=stage1.2-rlvr2 \
CONFIG_PATH=examples/nemo_gym/nemotron-3-super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/rlvr1_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr2/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr2/val-split.jsonl \
CONTAINER=your-registry/nemo-rl:v0.7.0.prefetched_venvs \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SBATCH_NUM_NODES=183 \
bash super_launch.sh
```

#### Stage 1.3 - RLVR 3 (183 H100 GPU nodes)
```bash
EXP_NAME=stage1.3-rlvr3 \
CONFIG_PATH=examples/nemo_gym/nemotron-3-super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/rlvr2_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr3/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr3/val-split.jsonl \
CONTAINER=your-registry/nemo-rl:v0.7.0.prefetched_venvs \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SBATCH_NUM_NODES=183 \
bash super_launch.sh
```

### Stage 2 - SWE

#### Stage 2.1 - SWE 1 (64 H100 GPU nodes)
```bash
EXP_NAME=stage2.1-swe1 \
CONFIG_PATH=examples/nemo_gym/nemotron-3-super/stage2_swe1.yaml \
MODEL_PATH=/path/to/rlvr3_checkpoint \
TRAIN_PATH=$DATA_DIR/swe1/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe1/val-split.jsonl \
CONTAINER=your-registry/nemo-rl:v0.7.0.prefetched_venvs \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

#### Stage 2.2 - SWE 2 (64 H100 GPU nodes)

This stage requires Apptainer images for the SWE Gym environments.

First, install [Apptainer](https://apptainer.org/docs/admin/main/installation.html) if it is not already available:

```bash
# Ubuntu/Debian
wget https://github.com/apptainer/apptainer/releases/download/v1.3.1/apptainer_1.3.1_amd64.deb
sudo apt install -y ./apptainer_1.3.1_amd64.deb
```

Then run the download script, which pulls Docker images from the R2E-Gym, SWE-Gym, and SWE-Bench Verified datasets on HuggingFace and converts them to `.sif` files:

```bash
./examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif --concurrency 16
```

Then launch the training run with `SIF_DIR` pointing at the directory containing the downloaded `.sif` files:

```bash
EXP_NAME=stage2.2-swe2 \
CONFIG_PATH=examples/nemo_gym/nemotron-3-super/stage2_swe2.yaml \
MODEL_PATH=/path/to/swe1_checkpoint \
TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe2/val-split.jsonl \
CONTAINER=your-registry/nemo-rl:v0.7.0.prefetched_venvs \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SIF_DIR=/path/to/sif \
bash super_launch.sh
```

### Stage 3 - RLHF (100 H100 GPU nodes)
```bash
EXP_NAME=stage3-rlhf \
CONFIG_PATH=examples/nemo_gym/nemotron-3-super/stage3_rlhf.yaml \
MODEL_PATH=/path/to/swe2_checkpoint \
TRAIN_PATH=$DATA_DIR/rlhf/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlhf/val-split.jsonl \
CONTAINER=your-registry/nemo-rl:v0.7.0.prefetched_venvs \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
EXTRA_MOUNTS=$EXTRA_MOUNTS \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SBATCH_NUM_NODES=100 \
bash super_launch.sh
```
