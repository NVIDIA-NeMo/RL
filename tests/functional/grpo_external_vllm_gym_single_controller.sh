#!/bin/bash
# Two-GPU external vLLM + Megatron SingleController/Gym smoke.
# Run inside an isolated allocation; service GPU and training GPU are disjoint.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
MODEL=${EXTERNAL_VLLM_TEST_MODEL:-Qwen/Qwen3-0.6B}
MAX_MODEL_LEN=${EXTERNAL_VLLM_TEST_MAX_MODEL_LEN:-8192}
MAX_NEW_TOKENS=${EXTERNAL_VLLM_TEST_MAX_NEW_TOKENS:-512}
MAX_AGENT_STEPS=${EXTERNAL_VLLM_TEST_MAX_AGENT_STEPS:-1}
RUN_DIR=${EXTERNAL_VLLM_TEST_RUN_DIR:-$SCRIPT_DIR/grpo_external_vllm_gym_single_controller}
SERVER_PORT=${EXTERNAL_VLLM_TEST_PORT:-19210}
BASE_URL=${EXTERNAL_VLLM_TEST_BASE_URL:-http://127.0.0.1:$SERVER_PORT/v1}
mkdir -p "$RUN_DIR"
EXPORT_DIR="$RUN_DIR/hf_exports"
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if [[ -z "${EXTERNAL_VLLM_TEST_BASE_URL:-}" ]]; then
    VLLM_PYTHON=${EXTERNAL_VLLM_TEST_PYTHON:?Set EXTERNAL_VLLM_TEST_PYTHON to the vLLM environment python}
    CUDA_VISIBLE_DEVICES=${EXTERNAL_VLLM_TEST_SERVICE_GPU:-1} VLLM_SERVER_DEV_MODE=1 \
    "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name policy --port "$SERVER_PORT" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.7 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$RUN_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    TRAIN_GPUS=${EXTERNAL_VLLM_TEST_TRAIN_GPU:-0}
else
    # Preserve the training step's assigned devices when the service is external.
    TRAIN_GPUS=${EXTERNAL_VLLM_TEST_TRAIN_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
fi
ready=0
for ((attempt=0; attempt<300; attempt++)); do
    if curl --fail --silent "${BASE_URL%/v1}/health" >/dev/null; then
        ready=1
        break
    fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
        cat "$RUN_DIR/server.log"
        exit 1
    fi
    sleep 2
done
[[ "$ready" == 1 ]] || { echo "vLLM service did not become ready: $BASE_URL"; exit 1; }
if [[ "${EXTERNAL_VLLM_TEST_REQUIRE_PREFIX:-0}" == 1 ]]; then
    curl --fail --silent "$BASE_URL/nemo-rl/prefix-token-capability" \
        | jq -e '.active == true and .required_prefix_token_ids == true' >/dev/null
fi
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS \
SC_TEST_KEEP_ALL_TOOLS=1 \
SC_TEST_RUN_DIR="$RUN_DIR/train" \
    bash "$SCRIPT_DIR/grpo_async_gym_single_controller.sh" \
    policy.model_name="$MODEL" \
    policy.max_total_sequence_length="$MAX_MODEL_LEN" \
    policy.generation.max_new_tokens="$MAX_NEW_TOKENS" \
    grpo.num_generations_per_prompt=4 \
    ++grpo.adv_estimator.name=reinforce_plus_plus \
    ++grpo.adv_estimator.minus_baseline=false \
    policy.train_global_batch_size=16 \
    policy.generation.backend=remote_vllm \
    ++policy.generation.refit_transport=hf_checkpoint \
    ++policy.generation.refit_cfg.hf_checkpoint.checkpoint_dir="$EXPORT_DIR" \
    ++policy.generation.remote_vllm_cfg.base_url="$BASE_URL" \
    ++policy.generation.remote_vllm_cfg.served_model_name=policy \
    ++policy.generation.remote_vllm_cfg.max_model_len="$MAX_MODEL_LEN" \
    ++policy.generation.remote_vllm_cfg.refit={} \
    cluster.gpus_per_node=1 \
    ++token_capture.enabled=false \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.return_token_id_information=true \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.request_prompt_and_generation_token_ids=true \
    ++env.nemo_gym.policy_model.responses_api_models.vllm_model.supply_prefix_token_ids=false \
    ++env.nemo_gym.workplace_assistant_simple_agent.responses_api_agents.simple_agent.max_steps="$MAX_AGENT_STEPS" \
    "$@"

uv run "$PROJECT_ROOT/tests/check_metrics.py" "$RUN_DIR/train/metrics.json" \
    'len(data["train/gen_kl_error"]) >= 10' \
    'max(data["train/grad_norm"]) > 0'
uv run python "$SCRIPT_DIR/check_hf_refit_exports.py" "$EXPORT_DIR" \
    --minimum-versions 10 --output "$RUN_DIR/refit-validation.json"
