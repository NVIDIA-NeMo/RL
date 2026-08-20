#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/../..")
EXP_NAME=$(basename "$0" .sh)
EXP_DIR="${SCRIPT_DIR}/${EXP_NAME}"
LOG_DIR="${EXP_DIR}/logs"
JSON_METRICS="${EXP_DIR}/metrics.json"
RUN_LOG="${EXP_DIR}/run.log"
CHECKPOINT_DIR="${EXP_DIR}/checkpoints"
DATA_DIR="${EXP_DIR}/data"
rm -rf "${EXP_DIR}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${DATA_DIR}"
trap 'rm -rf "${CHECKPOINT_DIR}"' EXIT

cd "${PROJECT_ROOT}/3rdparty/Gym-workspace/Gym"
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: ${HF_TOKEN}" >> env.yaml
fi

uv run ng_prepare_data \
    "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface

TRAIN_PATH="${DATA_DIR}/workplace_assistant_train.jsonl"
VALIDATION_PATH="${DATA_DIR}/workplace_assistant_validation.jsonl"
# Keep the smoke-test prompts within the model context window.
jq -c '.responses_create_params.tools |= (.[0:1])' \
    data/workplace_assistant/train.jsonl > "${TRAIN_PATH}"
jq -c '.responses_create_params.tools |= (.[0:1])' \
    data/workplace_assistant/validation.jsonl > "${VALIDATION_PATH}"

cd "${PROJECT_ROOT}"
uv run coverage run -a \
    --data-file="${PROJECT_ROOT}/tests/.coverage" \
    --source="${PROJECT_ROOT}/nemo_rl" \
    "${PROJECT_ROOT}/examples/nemo_gym/run_ppo_nemo_gym.py" \
    --config "${PROJECT_ROOT}/examples/nemo_gym/ppo_math_rlvr_nemo_gym.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.dtensor_cfg.enabled=true \
    policy.megatron_cfg.enabled=false \
    value.model_name=Qwen/Qwen3-0.6B \
    value.dtensor_cfg.enabled=true \
    value.megatron_cfg.enabled=false \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    ++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.enable_auto_tools=true \
    ++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser=hermes \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.7 \
    ++policy.generation.vllm_kwargs.compilation_config.backend=eager \
    ++policy.generation.vllm_kwargs.mamba_ssm_cache_dtype=float32 \
    policy.max_total_sequence_length=768 \
    policy.generation.colocated.enabled=true \
    ppo.num_prompts_per_step=4 \
    ppo.num_generations_per_prompt=2 \
    ppo.max_num_steps=10 \
    ppo.val_period=5 \
    ppo.policy_training_start_step=0 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    loss_fn.use_importance_sampling_correction=true \
    logger.tensorboard_enabled=true \
    logger.log_dir="${LOG_DIR}" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.save_period=5 \
    checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
    'env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]' \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.uses_reasoning_parser=false \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.extra_body.chat_template_kwargs.enable_thinking=false \
    env.should_log_nemo_gym_responses=true \
    ++env.should_mask_flagged_samples=true \
    data.shuffle=true \
    data.num_workers=0 \
    data.train.data_path="${TRAIN_PATH}" \
    data.validation.data_path="${VALIDATION_PATH}" \
    "$@" \
    2>&1 | tee "${RUN_LOG}"

grep -q "Running synchronous PPO training with NeMo Gym" "${RUN_LOG}"
uv run tests/json_dump_tb_logs.py "${LOG_DIR}" --output_path "${JSON_METRICS}"
uv run tests/check_metrics.py "${JSON_METRICS}" \
    'len(data["train/reward"]) == 10' \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'data["validation/accuracy"]["10"] > 0.1'
