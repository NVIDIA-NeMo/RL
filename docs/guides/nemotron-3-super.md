# Nemotron 3 Super

This guide explains how to post-train the Nemotron 3 Super model using NeMo RL.

## Download and prepare the data

```bash
# Download RL data blends (rlvr1, rlvr2, rlvr3, swe1, swe2, rlhf)
uvx --from huggingface-hub hf download nvidia/Nemotron-3-Super-RL-Training-Blend --repo-type dataset --local-dir=data_with_placeholders


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
Note that we currently require using the `super-v3` branch to train Nemotron 3 Super.
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

The RLVR stage consists of 3 sub-stages that use different data blends. The SWE RL stage consists of 2 sub-stages. This means there are 6 total stages in the full RL pipeline.

### Build sandbox container

TODO

### Launch script

Each stage of training uses the `super_launch.sh` script to launch the training job. In the instructions below, be sure to correctly set the following variables:

* `$DATA_DIR`: Path to the **final** `data` directory produced in [Download and prepare the data](#download-and-prepare-the-data).
* `$SANDBOX_CONTAINER`: The location of the sandbox container in the [Build sandbox container](#build-sandbox-container) section.
* `$PERSISTENT_CACHE_DIR`: The location to a folder that will be used to store caches for vllm and flashinfer.
* `$SLURM_PARTITION`
* `$SLURM_ACCOUNT`
* Optional: `$EXTRA_MOUNTS` — comma-separated host:container mount pairs for your cluster (e.g. `EXTRA_MOUNTS=/scratch:/scratch,/lustre:/lustre`). Omit if not needed.

In all the recipes, the `MODEL_PATH` also needs to be correctly set. The starting checkpoint for RL (Stage 1.1) is the SFT finetuned checkpoint. Each subsequent stage in the RL pipeline takes as an input checkpoint the output of the previous stage.

The number of nodes required for each stage is specified in the `cluster.num_nodes` config in the corresponding config file. This corresponds to the number of B200 nodes (8 GPUs each) required for training and may need to be adjusted when using different GPUs.

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
PERSISTENT_CACHE=$PERSISTENT_CACHE_DIR \
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
PERSISTENT_CACHE=$PERSISTENT_CACHE_DIR \
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
PERSISTENT_CACHE=$PERSISTENT_CACHE_DIR \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

### Stage 2 - SWE 1 (64 nodes)
```bash
EXP_NAME=stage2.1-swe1 \
CONFIG_PATH=examples/configs/super/stage2_swe1.yaml \
MODEL_PATH=/path/to/rlvr3_checkpoint \
TRAIN_PATH=$DATA_DIR/swe1/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe1/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE_DIR \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```

### Stage 2 - SWE 2 (64 nodes)
#### Download Apptainer images for SWE stage

```bash
uv run --with datasets examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif --concurrency 16

# Update container formatter in examples/configs/super/stage2_swe2.yaml
container_formatter:
  - "/path/to/sif/r2egym_{instance_id}.sif"
  - "/path/to/sif/swegym_sweb.eval.x86_64.{instance_id}.sif"
  - "/path/to/sif/swebench_sweb.eval.x86_64.{instance_id}.sif"
```
```bash
EXP_NAME=stage2.2-swe2 \
CONFIG_PATH=examples/configs/super/stage2_swe2.yaml \
MODEL_PATH=/path/to/swe1_checkpoint \
TRAIN_PATH=$DATA_DIR/swe2/train-split.jsonl \
VAL_PATH=$DATA_DIR/swe2/val-split.jsonl \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super \
SANDBOX_CONTAINER=$SANDBOX_CONTAINER \
PERSISTENT_CACHE=$PERSISTENT_CACHE_DIR \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
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
PERSISTENT_CACHE=$PERSISTENT_CACHE_DIR \
SLURM_PARTITION=$SLURM_PARTITION \
SLURM_ACCOUNT=$SLURM_ACCOUNT \
bash super_launch.sh
```
