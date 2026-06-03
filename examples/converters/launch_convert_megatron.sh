#!/bin/bash
# Usage:
#   # HF -> Megatron (import)
#   HF_PATH=meta-llama/Llama-3.2-1B MEGATRON_PATH=/path/to/output ./examples/converters/launch_convert_megatron.sh import
#
#   # Megatron -> HF (export)
#   MEGATRON_PATH=/path/to/megatron/ckpt HF_PATH=/path/to/hf/output ./examples/converters/launch_convert_megatron.sh export
#
#   # Override defaults:
#   HF_PATH=meta-llama/Llama-3.2-1B MEGATRON_PATH=/path/to/output NUM_NODES=4 TP=4 EP=8 WALLTIME=4:00:00 ./examples/converters/launch_convert_megatron.sh import
#
# Required env vars:
#   MEGATRON_PATH  - Megatron checkpoint path (output for import, input for export)
#   HF_PATH        - HF model ID or path (input for import, output for export)
#
# Optional env vars:
#   HF_REF_MODEL   - HF reference model for architecture detection during export
#                     (default: nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-RandomWeight)
#                     Only used for export; ignored for import.
#   TP / PP / EP / ETP - parallelism settings (default: 8/1/8/1)
#   NUM_NODES      - number of SLURM nodes (default: 2)
#   GPUS_PER_NODE  - GPUs per node (default: 4)
#   WALLTIME       - job walltime (default: 1:00:00)
#   SLURM_ACCOUNT  - SLURM account (default: llmservice_nemotron_ultra)
#   PARTITION      - SLURM partition (default: batch)
#   CONTAINER      - container sqsh path
#   SLURM_QOS      - SLURM QOS (e.g., "short", "interactive")
#   HF_TOKEN       - HuggingFace token (needed for gated models)
set -euo pipefail

COMMAND=${1:?Usage: $0 <import|export>}
if [[ "$COMMAND" != "import" && "$COMMAND" != "export" ]]; then
    echo "ERROR: First argument must be 'import' or 'export', got: $COMMAND"
    exit 1
fi

MEGATRON_PATH=${MEGATRON_PATH:?Set MEGATRON_PATH}
HF_PATH=${HF_PATH:?Set HF_PATH (HF model ID or path for import, output directory for export)}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

# Parallelism
TP=${TP:-8}
PP=${PP:-1}
EP=${EP:-8}
ETP=${ETP:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# Megatron-Core requires world_size to be divisible by (TP * PP * CP).
# Specify NUM_NODES directly; Megatron will error at runtime if the
# world_size is incompatible with the parallelism settings.
NUM_NODES=${NUM_NODES:-2}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))

echo "Parallelism: TP=$TP PP=$PP EP=$EP ETP=$ETP -> $NUM_NODES nodes x $GPUS_PER_NODE GPUs = $WORLD_SIZE ranks"

# Model
HF_REF_MODEL=${HF_REF_MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-RandomWeight}

# SLURM
WALLTIME=${WALLTIME:-1:00:00}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-llmservice_nemotron_ultra}
PARTITION=${PARTITION:-batch}
SLURM_QOS=${SLURM_QOS:-}

# Container
CONTAINER=${CONTAINER:?Set CONTAINER to the NeMo RL container (path to a .sqsh or an NGC image URI)}
MOUNTS="/lustre:/lustre,${PROJECT_ROOT}/examples/converters/convert_megatron_checkpoints.py:/opt/nemo-rl/examples/converters/convert_megatron_checkpoints.py"

CONVERT_SCRIPT=/opt/nemo-rl/examples/converters/convert_megatron_checkpoints.py
MASTER_PORT=${MASTER_PORT:-29500}

# Build the script args
if [[ "$COMMAND" == "import" ]]; then
    SCRIPT_ARGS="import --hf-model $HF_PATH --megatron-path $MEGATRON_PATH --tp $TP --pp $PP --ep $EP --etp $ETP"
else
    SCRIPT_ARGS="export --hf-model $HF_REF_MODEL --megatron-path $MEGATRON_PATH --hf-path $HF_PATH --tp $TP --pp $PP --ep $EP --etp $ETP"
fi

echo "Container: $CONTAINER"
echo "Command: $COMMAND"
echo "Script args: $SCRIPT_ARGS"
echo "Submitting $NUM_NODES-node job..."

# Write the batch script to a temp file to avoid nested quoting hell with --wrap.
BATCH_SCRIPT=$(mktemp /tmp/convert_batch_XXXXXX.sh)
cat > "$BATCH_SCRIPT" <<EOF
#!/bin/bash
set -eoux pipefail

MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n1)

srun \\
  --ntasks-per-node=1 \\
  --container-image=${CONTAINER} \\
  --container-mounts=${MOUNTS} \\
  --no-container-mount-home \\
  bash -c '
    ${HF_TOKEN:+export HF_TOKEN=${HF_TOKEN}}
    python-MegatronPolicyWorker -m torch.distributed.run \
      --nproc_per_node=${GPUS_PER_NODE} \
      --nnodes=${NUM_NODES} \
      --node_rank=\$SLURM_NODEID \
      --master_addr='"\$MASTER_ADDR"' \
      --master_port=${MASTER_PORT} \
      ${CONVERT_SCRIPT} ${SCRIPT_ARGS}
  '
EOF

echo "--- Batch script ($BATCH_SCRIPT) ---"
cat "$BATCH_SCRIPT"
echo "---"

sbatch \
    --nodes="$NUM_NODES" \
    --exclusive \
    --mem=0 \
    --gres=gpu:"$GPUS_PER_NODE" \
    --account="$SLURM_ACCOUNT" \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --job-name="convert-${COMMAND}" \
    ${SLURM_QOS:+--qos="$SLURM_QOS"} \
    "$BATCH_SCRIPT"
