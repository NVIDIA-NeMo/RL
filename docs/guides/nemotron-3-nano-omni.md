# Nemotron 3 Nano Omni

This guide explains how to post-train the Nemotron 3 Nano Omni vision-language model with GRPO using NeMo RL on the AutoModel backend.

It covers two recipes:

- **CLEVR-CoGenT** — runs on a single 8-GPU node (interactive container).
- **MMPR-Tiny** — runs on 4 nodes via Slurm.

Both share the same checkpoint, model code, and reward pipeline; they differ only in the dataset, reward functions, and node count.

## Recipe 1 — CLEVR-CoGenT (single-node)

The CLEVR-CoGenT recipe uses [`examples/configs/vlm_grpo_nemotron_omni.yaml`](../../examples/configs/vlm_grpo_nemotron_omni.yaml). It expects 8 GPUs on a single node, EP=8 across the experts, and TP=8 in vLLM.

Key knobs in the config:

| Field | Value |
|---|---|
| `policy.model_name` | path to the Nemotron-Omni HF checkpoint |
| `policy.dtensor_cfg.expert_parallel_size` | 8 |
| `policy.generation.vllm_cfg.tensor_parallel_size` | 8 |
| `policy.max_total_sequence_length` | 8192 |
| `data.train.dataset_name` | `clevr-cogent` (split `train`) |
| `data.validation.dataset_name` | `clevr-cogent` (split `valA`) |
| `env.clevr-cogent.reward_functions` | `format` (0.2) + `exact_alnum` (0.8) |

CLEVR is loaded automatically from HuggingFace by the `clevr-cogent` response dataset on first run; no manual prep is required.

### Launch (interactive container)

From inside the container on an 8-GPU node:

```bash
export NRL_MAMBA_PREFILL_DECODE_SYNC="${NRL_MAMBA_PREFILL_DECODE_SYNC:-1}"

uv run examples/run_vlm_grpo.py --config examples/configs/vlm_grpo_nemotron_omni.yaml \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=1
```

To override the model path or any other YAML field, append Hydra-style overrides:

```bash
uv run examples/run_vlm_grpo.py --config examples/configs/vlm_grpo_nemotron_omni.yaml \
    policy.model_name=/path/to/your/checkpoint \
    cluster.gpus_per_node=8 cluster.num_nodes=1
```

## Recipe 2 — MMPR-Tiny (4-node Slurm)

The MMPR-Tiny recipe uses [`examples/configs/vlm_grpo_nemotron_omni_mmpr.yaml`](../../examples/configs/vlm_grpo_nemotron_omni_mmpr.yaml). Differences vs. the CLEVR recipe:

| Field | Value |
|---|---|
| `data.train.dataset_name` | `mmpr-tiny` |
| `data.train.download_dir` | local cache dir for MMPR-Tiny (loader auto-downloads from HF) |
| `data.train.split_validation_size` | `0.008` (val split carved out of train) |
| `policy.max_total_sequence_length` | 10000 |
| `env.mmpr-tiny.reward_functions` | `geo3k` (1.0, `format_score: 0.1`) |


### Launch (4-node Slurm)

Submit with `ray.sub`. From the repo root on a Slurm login node:

```bash
# --- Cluster config ---
export SBATCH_ACCOUNT=your_slurm_account
export SBATCH_PARTITION=batch
export SBATCH_TIME=4:00:00
export CONTAINER=/path/to/containers/nemo-rl-nano-v3-vl-<tag>.sqsh
export MOUNTS=/lustre:/lustre
export HF_HOME=/path/to/cache/huggingface
export TMPDIR=/tmp/nrl-${USER}
export NCCL_DEBUG=WARN
export NRL_IGNORE_VERSION_MISMATCH=1

# --- Run config ---
NUM_NODES=4
GPUS_PER_NODE=8
JOB_NAME=grpo-nemotron-omni-mmpr
RESULTS_DIR=$PWD/results/${JOB_NAME}
CONFIG_PATH=examples/configs/vlm_grpo_nemotron_omni_mmpr.yaml

# --- Build the training command (run inside the container on every node) ---
export COMMAND="\
export PYTHONPATH=\${PYTHONPATH:-}:/path/to/automodel-omni && \
export CUDA_LAUNCH_BLOCKING=0 && \
export TORCH_USE_CUDA_DSA=0 && \
export NRL_MAMBA_PREFILL_DECODE_SYNC=1 && \
mkdir -p ${HF_HOME} ${TMPDIR} ${RESULTS_DIR} && \
uv run examples/run_vlm_grpo.py --config ${CONFIG_PATH} \
    cluster.num_nodes=${NUM_NODES} \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    checkpointing.checkpoint_dir='${RESULTS_DIR}' \
    logger.wandb.name='${JOB_NAME}'"

# --- Submit ---
sbatch \
    --nodes=${NUM_NODES} \
    --account=${SBATCH_ACCOUNT} \
    --job-name=nemo-rl-${JOB_NAME} \
    --partition=${SBATCH_PARTITION} \
    --time=${SBATCH_TIME} \
    --dependency=singleton \
    --gres=gpu:${GPUS_PER_NODE} \
    ray.sub
```

To run on a different node count, change `NUM_NODES` and the `--nodes` flag.
