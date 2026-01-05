# Nemotron 3 Nano

This guide explains how to post-train the [Nemotron 3 Nano model](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf) using NeMo RL.

## Download and prepare the data

```bash
# Download RL data blend
uvx --from huggingface-hub hf download nvidia/Nemotron-3-Nano-RL-Training-Blend --repo-type dataset --local-dir=data

# Fill in placeholders in dataset
chmod +x data/create_nanov3_jsonl.py
./data/create_nanov3_jsonl.py --input data/train.jsonl --output data/train-full.jsonl

# Use the last 1000 rows for validation
head -n -1000 data/train-full.jsonl > data/train-split.jsonl
tail -n 1000 data/train-full.jsonl > data/val-split.jsonl
```

## Prepare the code
Note that we currently require using the `nano-v3` branch to train Nemotron 3 Nano.
```bash
# Checkout NeMo RL
git clone -b nano-v3 https://github.com/NVIDIA-NeMo/RL.git
cd RL

# Initialize the submodules
git submodule update --init --recursive
```

## Create a launch script

Create a file named `launch.sh` with the following contents. Be sure to fill in the `DATA_DIR`, `MODEL_CHECKPOINT`, `WANDB_API_KEY`, `SLURM_ACCOUNT`, `SLURM_PARTITION`, `MOUNTS`. Note that the default recipe (`examples/nemo_gym/grpo_nanov3.yaml`) uses 32 nodes.

```bash
CODE_DIR=$PWD
SLURM_JOB_NAME=nano-v3-rl-training

# Fill these in
DATA_DIR=...
MODEL_CHECKPOINT=...
WANDB_API_KEY=...
SLURM_ACCOUNT=...
SLURM_PARTITION=...
MOUNTS=... # SRC:DST[,SRC:DST...] e.g., MOUNTS="/lustre:/lustre,/data:/data"

CONTAINER="nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano"
COMMAND="uv run examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/grpo_nanov3.yaml data.train_jsonl_fpath=$DATA_DIR/train-split.jsonl data.validation_jsonl_fpath=$DATA_DIR/val-split.jsonl policy.model_name=$MODEL_CHECKPOINT logger.wandb_enabled=True"

COMMAND="${COMMAND}" \
CONTAINER="${CONTAINER}" \
MOUNTS="${MOUNTS}" \
WANDB_API_KEY=${WANDB_API_KEY} \
sbatch \
    --nodes=32 \
    --account="${SLURM_ACCOUNT}" \
    --job-name="${SLURM_JOB_NAME}" \
    --partition="${SLURM_PARTITION}" \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```


## Launch training
```bash
bash launch.sh
```
