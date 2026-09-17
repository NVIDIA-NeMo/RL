#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/precision-matrix-submit-test.XXXXXX")
trap 'rm -rf "${TMP_ROOT}"' EXIT

mkdir -p \
  "${TMP_ROOT}/bin" \
  "${TMP_ROOT}/hf/hub/models--Qwen--Qwen3-30B-A3B" \
  "${TMP_ROOT}/home"
touch "${TMP_ROOT}/container.sqsh" "${TMP_ROOT}/home/.netrc"

cat > "${TMP_ROOT}/bin/sbatch" <<'EOF'
#!/usr/bin/env bash
printf 'COMMAND=%s\n' "${COMMAND:-}"
printf 'SETUP_COMMAND=%s\n' "${SETUP_COMMAND:-}"
printf '%s\n' "$@"
EOF
chmod +x "${TMP_ROOT}/bin/sbatch"

output=$(
  PATH="${TMP_ROOT}/bin:${PATH}" \
  ACTION=test-only \
  CLUSTER=oci \
  PARTITION=batch \
  MODEL=qwen30 \
  MODE=sync \
  ARM=bf16-mxfp8 \
  MAX_STEPS=20 \
  AFTEROK_JOB_ID=12345 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  CONTAINER="${TMP_ROOT}/container.sqsh" \
  HF_HOME_SOURCE="${TMP_ROOT}/hf" \
  WANDB_HOME="${TMP_ROOT}/home" \
  RESULT_ROOT="${TMP_ROOT}/results" \
  LOCAL_ROOT="${TMP_ROOT}/local" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -Fx -- '--dependency=afterok:12345' <<<"${output}" >/dev/null
grep -F -- "${TMP_ROOT}/results/source-archives/nemo-rl-" <<<"${output}" >/dev/null
grep -F -- 'resolve_slurm_cli_path()' "${REPO}/ray.sub" >/dev/null
grep -F -- '/cm/local/apps/slurm/*/bin' "${REPO}/ray.sub" >/dev/null
grep -F -- 'Unable to find srun, scontrol, and sinfo' "${REPO}/ray.sub" >/dev/null

qwen35_output=$(
  ACTION=render \
  CLUSTER=oci \
  MODEL=qwen35 \
  MODE=sync \
  ARM=bf16-bf16 \
  MAX_STEPS=20 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- 'grpo.num_prompts_per_step=128' <<<"${qwen35_output}" >/dev/null
grep -F -- 'grpo.num_generations_per_prompt=16' <<<"${qwen35_output}" >/dev/null
grep -F -- 'policy.train_global_batch_size=2048' <<<"${qwen35_output}" >/dev/null

super_output=$(
  ACTION=render \
  CLUSTER=oci \
  MODEL=super \
  MODE=sync \
  ARM=bf16-bf16 \
  PERFORMANCE_RECIPE=1 \
  SUPER_GPU_MEMORY_UTILIZATION=0.6 \
  MAX_STEPS=20 \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- 'policy.generation.vllm_cfg.gpu_memory_utilization=0.6' \
  <<<"${super_output}" >/dev/null

# Qwen3-235B has 1,536 expert rows. FlashInfer TRTLLM BF16 requires each
# TP-local expert dimension to be divisible by 128, so TP8 (192 rows) is
# unsupported. Keep every precision arm on the same supported TP4 topology.
grep -F -- 'tensor_parallel_size: 4' \
  "${SCRIPT_DIR}/qwen235-performance-sync.yaml" >/dev/null
grep -F -- 'gpu_memory_utilization: 0.7' \
  "${SCRIPT_DIR}/qwen235-performance-sync.yaml" >/dev/null

mkdir -p "${TMP_ROOT}/direct-model"
direct_model_output=$(
  PATH="${TMP_ROOT}/bin:${PATH}" \
  ACTION=test-only \
  CLUSTER=oci \
  PARTITION=batch \
  MODEL=qwen30 \
  MODE=sync \
  ARM=bf16-bf16 \
  PERFORMANCE_RECIPE=1 \
  MODEL_SNAPSHOT_OVERRIDE="${TMP_ROOT}/direct-model" \
  SLURM_ACCOUNT=test \
  REPO="${REPO}" \
  CONTAINER="${TMP_ROOT}/container.sqsh" \
  HF_HOME_SOURCE="${TMP_ROOT}/hf" \
  WANDB_HOME="${TMP_ROOT}/home" \
  RESULT_ROOT="${TMP_ROOT}/results" \
  LOCAL_ROOT="${TMP_ROOT}/local" \
  "${SCRIPT_DIR}/submit.sh"
)

grep -F -- "policy.model_name=${TMP_ROOT}/direct-model" \
  <<<"${direct_model_output}" >/dev/null
