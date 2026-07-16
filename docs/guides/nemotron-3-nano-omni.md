# Nemotron 3 Nano Omni

This guide explains how to post-train the Nemotron 3 Nano Omni vision-language model with GRPO using NeMo RL. Both the AutoModel and Megatron backends are supported for image-and-text training.

## AutoModel backend

It covers two recipes:

- **CLEVR-CoGenT** — runs on a single 8-GPU node (interactive container).
- **MMPR-Tiny** — runs on 4 nodes via Slurm.

Both share the same checkpoint, model code, and reward pipeline; they differ only in the dataset, reward functions, and node count.

### Recipe 1 — CLEVR-CoGenT (single-node)

The CLEVR-CoGenT recipe uses [`examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v1.yaml). It expects 8 GPUs on a single node, EP=8 across the experts, and TP=8 in vLLM.

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

uv run examples/run_vlm_grpo.py --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v1.yaml \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=1
```

To override the model path or any other YAML field, append Hydra-style overrides:

```bash
uv run examples/run_vlm_grpo.py --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v1.yaml \
    policy.model_name=/path/to/your/checkpoint \
    cluster.gpus_per_node=8 cluster.num_nodes=1
```

### Recipe 2 — MMPR-Tiny (4-node Slurm)

The MMPR-Tiny recipe uses [`examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml). Differences vs. the CLEVR recipe:

| Field | Value |
|---|---|
| `data.train.dataset_name` | `mmpr-tiny` |
| `data.train.download_dir` | local cache dir for MMPR-Tiny (loader auto-downloads from HF) |
| `data.train.split_validation_size` | `0.008` (val split carved out of train) |
| `policy.max_total_sequence_length` | 8192 |
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
CONFIG_PATH=examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml

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

## Megatron backend

The Megatron backend uses a dedicated `NemotronOmniModel` supplied by Megatron Bridge. The Hugging Face processor expands each image placeholder into the complete media-token sequence before the batch reaches the model. NeMo RL passes that expanded sequence and the image tensors to the model; `NemotronOmniModel` replaces the media-token positions with RADIO encoder outputs and then performs sequence packing and context-parallel sharding.

This is the same model-owned packing boundary used by maintained Megatron VLM integrations. It differs from the historical Nemotron Omni `LLaVAModel` path, which collapsed the expanded media-token sequence before packing and expanded it again inside the model. The dedicated model removes that extra representation change and allows the integration to use Megatron Bridge and Megatron-LM from their maintained main branches.

The current Megatron recipes cover Nano image-and-text GRPO. Super, video, and audio training are follow-up work and are not enabled by these recipes.

### Checkpoint compatibility

Use the `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` Hugging Face checkpoint or a checkpoint converted with the dedicated `NemotronOmniModel` integration. Legacy Megatron checkpoints whose parameter names use an `llava_model` prefix are not compatible with this model definition. Reconvert those checkpoints from the original Hugging Face checkpoint instead of loading them directly.

### Maintained recipes

| Workload | Recipe | Topology |
|---|---|---|
| CLEVR-CoGenT | [`vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8.v1.yaml) | 1 node, 8 GPUs, TP=8, EP=8 |
| MMPR-Tiny | [`vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-megatron-tp8ep16.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-megatron-tp8ep16.v1.yaml) | 4 nodes, 8 GPUs per node, TP=8, EP=16, vLLM TP=2 |

Launch the single-node Megatron recipe from inside the container on an 8-GPU node:

```bash
uv run examples/run_vlm_grpo.py \
    --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8.v1.yaml
```

For a four-node Slurm run, use the `ray.sub` example above with the following configuration path and omit the AutoModel-specific `PYTHONPATH` addition:

```bash
CONFIG_PATH=examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-megatron-tp8ep16.v1.yaml
```

The recipes keep sequence packing enabled because the model owns the packing step after multimodal embedding insertion. They also request raw generation log probabilities so that vLLM and the Megatron policy compare the same pre-processor probability values when generation constraints such as `bad_words` are active. The generation context cap prevents the processor-expanded image prompt plus generated response from exceeding the configured 8192-token context length.
