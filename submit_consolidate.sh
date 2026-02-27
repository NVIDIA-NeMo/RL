#!/bin/bash
# Consolidate NeMo-RL sharded checkpoints into HuggingFace format.
# Runs inside the container (uses nemo_automodel for consolidation).
# Does NOT need Ray or GPUs — just CPU and filesystem access.
#
# Usage (from RL/ directory):
#   bash submit_consolidate.sh

REPO_ROOT="/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL"
MY_CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh"
MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha,/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl"

LOG_DIR="${REPO_ROOT}/consolidate-logs"
mkdir -p "${LOG_DIR}"

cat > /tmp/consolidate_job.sh << 'JOBEOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --job-name=consolidate-ckpt
#SBATCH --partition=batch
#SBATCH --time=0:30:0
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL"
MY_CONTAINER="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/nemo_rl/nemo-rl.sqsh"
MOUNTS="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha:/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha,/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl:/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl"

srun --no-container-mount-home \
  --container-mounts="${MOUNTS}" \
  --container-image="${MY_CONTAINER}" \
  --container-workdir="${REPO_ROOT}" \
  bash -c '
set -euo pipefail
export HF_TOKEN=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
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/avenkateshha/hf

REPO_ROOT="/lustre/fsw/portfolios/coreai/users/avenkateshha/nemo_rl/RL"

echo "=== Consolidating SFT checkpoint (step_50) ==="
uv run ${REPO_ROOT}/examples/converters/consolidate_checkpoint.py \
  --input ${REPO_ROOT}/results/sft/step_50/policy/weights \
  --output ${REPO_ROOT}/results/sft/step_50_hf \
  --model-name meta-llama/Llama-3.2-1B \
  --overwrite

echo ""
echo "=== Consolidating off-policy distillation checkpoint (step_50) ==="
uv run ${REPO_ROOT}/examples/converters/consolidate_checkpoint.py \
  --input ${REPO_ROOT}/checkpoints/distillation-meta-llama/Llama-3.2-1B/step_50/policy/weights \
  --output ${REPO_ROOT}/checkpoints/distillation-meta-llama/Llama-3.2-1B/step_50_hf \
  --model-name meta-llama/Llama-3.2-1B \
  --overwrite

echo ""
echo "=== Done! Consolidated checkpoints: ==="
echo "  SFT:        ${REPO_ROOT}/results/sft/step_50_hf"
echo "  Off-policy: ${REPO_ROOT}/checkpoints/distillation-meta-llama/Llama-3.2-1B/step_50_hf"
'
JOBEOF

echo "Submitting checkpoint consolidation job..."
sbatch --output="${LOG_DIR}/consolidate-%j.log" /tmp/consolidate_job.sh
