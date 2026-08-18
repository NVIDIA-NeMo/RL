#!/bin/bash
# Functional smoke for ModelExpress reshard refit on one 2-GPU node:
# Megatron TP1 trainer (1 GPU) -> vLLM TP1 rollout (1 GPU), two GRPO steps.
#
# Requires the mcore + vllm extras, ModelExpress/NIXL, a reachable MX metadata
# server, and a 2-GPU allocation:
#   MX_SERVER_URL=host:8001 \
#     uv run --extra mcore --extra vllm \
#     bash tests/functional/grpo_mx_reshard_refit.sh

set -eou pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MX_SERVER_URL="${MX_SERVER_URL:-modelexpress-server.kavin.svc.cluster.local:8001}"

rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
uv run coverage run -a \
    --data-file="${PROJECT_ROOT}/tests/.coverage" \
    --source="${PROJECT_ROOT}/nemo_rl" \
    "${PROJECT_ROOT}/examples/run_grpo.py" \
    --config "${PROJECT_ROOT}/examples/configs/grpo_math_1B_megatron.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.dtensor_cfg.enabled=false \
    policy.generation.backend=vllm \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    ++policy.generation.refit_transport=mx_reshard \
    ++policy.generation.refit_cfg.mx_reshard.server_url="${MX_SERVER_URL}" \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"

# A broken refit corrupts generation weights and makes the train/gen
# importance-sampling ratio explode.
uv run tests/check_metrics.py "${JSON_METRICS}" \
    'max(data["train/token_mult_prob_error"]) < 1.05'
