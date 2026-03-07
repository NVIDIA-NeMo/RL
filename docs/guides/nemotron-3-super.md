# Nemotron 3 Super

This guide explains how to post-train the Nemotron 3 Super model using NeMo RL.

## Download and prepare the data

```bash
# Download RL data blends (rlvr1, rlvr2, rlvr3, swe1, swe2, rlhf)
uvx --from huggingface-hub hf download nvidia/Nemotron-3-Super-RL-Training-Blends --repo-type dataset --local-dir=data_with_placeholders


# Fill in placeholders in data blends
chmod +x data_with_placeholders/fill_placeholders.py
./data_with_placeholders/fill_placeholders.py --input-dir data_with_placeholders --output-dir data_filled


# Create train/val splits for each data blend (last 1000 rows held out for validation)
for f in data_filled/*.jsonl; do
  name=$(basename "$f" .jsonl)
  mkdir -p "data/$name"
  head -n -1000 "$f" > "data/$name/train-split.jsonl"
  tail -n 1000 "$f" > "data/$name/val-split.jsonl"
done
```

## Prepare the code
Training Nemotron 3 Super currently requires the `super-v3` branch.
```bash
# Checkout NeMo RL
git clone -b super-v3 https://github.com/NVIDIA-NeMo/RL.git
cd RL

# Initialize the submodules
git submodule update --init --recursive
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
docker build -t nemo-skills-sandbox:latest -f dockerfiles/Dockerfile.sandbox .
```

For SLURM clusters using [enroot](https://github.com/NVIDIA/enroot), convert the image to a `.sqsh` file:

```bash
enroot import -o nemo-skills-sandbox.sqsh dockerd://nemo-skills-sandbox:latest
```

### Launch script

Each stage of training uses the `super_launch.sh` script to launch the training job. In the instructions below, be sure to correctly set the following variables:

* `$DATA_DIR`: Path to the **final** `data` directory produced in [Download and prepare the data](#download-and-prepare-the-data).
* `$SANDBOX_CONTAINER`: The sandbox container image from [Build sandbox container](#build-sandbox-container) (`.sqsh` path or registry URI).
* `$PERSISTENT_CACHE`: Path to a directory used to store caches for vLLM and FlashInfer.
* `$SLURM_PARTITION`
* `$SLURM_ACCOUNT`
* Optional: `$EXTRA_MOUNTS` — additional comma-separated `host:container` mount pairs for your cluster. The launch script automatically mounts `MODEL_PATH`, data directories, `PERSISTENT_CACHE`, and `SIF_DIR` into the container, but you may still need this for other paths (e.g. `EXTRA_MOUNTS=/scratch:/scratch`).

`MODEL_PATH` is the input checkpoint for each stage. Stage 1.1 starts from the SFT checkpoint; every subsequent stage takes the output of the previous one.

Node counts listed below (from `cluster.num_nodes` in each config) assume B200 nodes with 8 GPUs each and may need adjustment for other GPU types.

### Stage 1 - RLVR

#### Stage 1.1 - RLVR 1 (109 nodes)
```bash
EXP_NAME=stage1.1-rlvr1 \
CONFIG_PATH=examples/configs/super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/sft_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr1/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr1/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

#### Stage 1.2 - RLVR 2 (109 nodes)
```bash
EXP_NAME=stage1.2-rlvr2 \
CONFIG_PATH=examples/configs/super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/rlvr1_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr2/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr2/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

#### Stage 1.3 - RLVR 3 (109 nodes)
```bash
EXP_NAME=stage1.3-rlvr3 \
CONFIG_PATH=examples/configs/super/stage1_rlvr.yaml \
MODEL_PATH=/path/to/rlvr2_checkpoint \
TRAIN_PATH=$DATA_DIR/rlvr3/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlvr3/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

### Stage 2 - SWE

#### Stage 2.1 - SWE 1 (64 nodes)
```bash
EXP_NAME=stage2.1-swe1 \
CONFIG_PATH=examples/configs/super/stage2_swe1.yaml \
MODEL_PATH=/path/to/rlvr3_checkpoint \
TRAIN_PATH=$DATA_DIR/swe1/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe1/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

#### Stage 2.2 - SWE 2 (64 nodes)

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
CONFIG_PATH=examples/configs/super/stage2_swe2.yaml \
MODEL_PATH=/path/to/swe1_checkpoint \
TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe2/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
SIF_DIR=/path/to/sif \
bash super_launch.sh
```

### Stage 3 - RLHF (72 nodes)
```bash
EXP_NAME=stage3-rlhf \
CONFIG_PATH=examples/configs/super/stage3_rlhf.yaml \
MODEL_PATH=/path/to/swe2_checkpoint \
TRAIN_PATH=$DATA_DIR/rlhf/train-split.jsonl \
VAL_PATH=$DATA_DIR/rlhf/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```
