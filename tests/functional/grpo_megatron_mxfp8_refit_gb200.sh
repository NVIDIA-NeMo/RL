# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
git config --global --add safe.directory "$PROJECT_ROOT"

EXP_NAME=$(basename "$0" .sh)
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "$EXP_DIR"
mkdir -p "$LOG_DIR"

assert_grep() {
    local pattern=$1
    local file=$2
    grep -Eq "$pattern" "$file" || {
        echo "[FAIL] expected '$pattern' in $file"
        exit 1
    }
}

cd "$PROJECT_ROOT"
uv run coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    "$PROJECT_ROOT/examples/run_grpo.py" \
    --config "$PROJECT_ROOT/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml" \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=2 \
    env.math.num_workers=2 \
    loss_fn.reference_policy_kl_penalty=0.0 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=1 \
    policy.max_total_sequence_length=512 \
    policy.make_sequence_length_divisible_by=32 \
    ++policy.megatron_cfg.train_iters=2 \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=2 \
    policy.megatron_cfg.sequence_parallel=false \
    policy.megatron_cfg.activation_checkpointing=true \
    policy.megatron_cfg.fp8_cfg.enabled=false \
    policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
    policy.megatron_cfg.optimizer.optimizer_offload_fraction=1.0 \
    policy.generation.backend=megatron \
    policy.generation.max_new_tokens=64 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=2 \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    policy.generation.mcore_generation_config.expert_model_parallel_size=2 \
    policy.generation.mcore_generation_config.sequence_parallel=false \
    policy.generation.mcore_generation_config.inference_grouped_gemm_backend=torch \
    ++policy.generation.mcore_generation_config.inference_moe_token_dispatcher_type=nvls \
    policy.generation.mcore_generation_config.cuda_graph_impl=local \
    policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    policy.generation.mcore_generation_config.num_cuda_graphs=-1 \
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=true \
    policy.generation.mcore_generation_config.enable_chunked_prefill=false \
    policy.generation.mcore_generation_config.buffer_size_gb=2 \
    policy.generation.mcore_generation_config.max_model_len=512 \
    policy.generation.mcore_generation_config.max_tokens=512 \
    policy.generation.mcore_generation_config.logprobs_mode=raw_logprobs \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    ++policy.generation.mcore_generation_config.fp8_cfg.enabled=true \
    ++policy.generation.mcore_generation_config.fp8_cfg.fp8=e4m3 \
    ++policy.generation.mcore_generation_config.fp8_cfg.fp8_recipe=mxfp8 \
    ++policy.generation.mcore_generation_config.fp8_cfg.fp8_param=true \
    cluster.gpus_per_node=4 \
    cluster.num_nodes=1 \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

# The setup weight-sync timing proves that the initial refit ran, while the
# per-step transfer timing proves that the post-update refit ran. Raw rollout
# log-probs are compared against BF16 policy recomputation; the bounds allow the
# established MXFP8 quantization delta while rejecting a bad refit.
uv run tests/check_metrics.py "$JSON_METRICS" \
    'len(data["train/loss"]) == 2' \
    'len(data["timing/setup/weight_sync_time_s"]) == 1' \
    'min(data["timing/setup/weight_sync_time_s"]) > 0' \
    'len(data["timing/train/prepare_for_generation/transfer_and_update_weights"]) == 1' \
    'min(data["timing/train/prepare_for_generation/transfer_and_update_weights"]) > 0' \
    'len(data["train/gen_kl_error"]) == 2' \
    'max(data["train/gen_kl_error"]) < 0.15' \
    'len(data["train/token_mult_prob_error"]) == 2' \
    'max(data["train/token_mult_prob_error"]) < 1.5'

assert_grep 'cuda graph warmup' "$RUN_LOG"

echo "[PASS] GB200 Nano-v3 BF16-to-MXFP8 Megatron refit functional test"
