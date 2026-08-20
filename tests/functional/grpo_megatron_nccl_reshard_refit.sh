#!/bin/bash
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

# End-to-end non-colocated Megatron train -> Megatron generation refit over
# nccl_reshard with CUDA graphs. The default CI environment may exercise the
# exact-transfer Python implementation. Set REQUIRE_REAL_NCCL_M2N=1 in an
# environment with nccl-extensions installed to require nccl.m2n.reshard
# instead. Set REFIT_PRECISION=mxfp8 on Blackwell to cover inference-side
# MXFP8 refit and Torch-layout weights; BF16 is the portable default.

set -eou pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
git config --global --add safe.directory "$PROJECT_ROOT"

REFIT_PRECISION=${REFIT_PRECISION:-bf16}
case "$REFIT_PRECISION" in
    bf16)
        model_name=Qwen/Qwen2.5-0.5B
        max_token_mult_prob_error=1.05
        precision_args=()
        ;;
    mxfp8)
        # Qwen2.5 uses QKV bias, which the inference-optimized spec rejects.
        # Qwen3 is bias-free and is supported by Megatron Bridge.
        model_name=Qwen/Qwen3-0.6B
        max_token_mult_prob_error=1.10
        precision_args=(
            ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized
            ++policy.generation.mcore_generation_config.fp8_cfg.enabled=true
            ++policy.generation.mcore_generation_config.fp8_cfg.fp8=e4m3
            ++policy.generation.mcore_generation_config.fp8_cfg.fp8_recipe=mxfp8
            ++policy.generation.mcore_generation_config.fp8_cfg.fp8_param=true
            ++policy.generation.mcore_generation_config.inference_grouped_gemm_backend=torch
        )
        ;;
    *)
        echo "REFIT_PRECISION must be bf16 or mxfp8, got $REFIT_PRECISION" >&2
        exit 1
        ;;
esac

EXP_NAME="$(basename "$0" .sh)-$REFIT_PRECISION"
EXP_DIR="$SCRIPT_DIR/$EXP_NAME"
LOG_DIR="$EXP_DIR/logs"
JSON_METRICS="$EXP_DIR/metrics.json"
RUN_LOG="$EXP_DIR/run.log"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

rm -rf "$EXP_DIR"
mkdir -p "$LOG_DIR"

# Exercise the default dispatcher rather than forcing either fallback path.
unset NRL_XFERDTENSOR_GOLDEN
unset NRL_XFERDTENSOR_PYTHON

cd "$PROJECT_ROOT"
uv run coverage run -a \
    --data-file="$PROJECT_ROOT/tests/.coverage" \
    --source="$PROJECT_ROOT/nemo_rl" \
    "$PROJECT_ROOT/examples/run_grpo.py" \
    --config "$PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml" \
    policy.model_name="$model_name" \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.backend=megatron \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.refit_transport=nccl_reshard \
    policy.generation.mcore_generation_config.refit_impl=bridge \
    policy.generation.mcore_generation_config.cuda_graph_impl=local \
    policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    policy.generation.mcore_generation_config.num_cuda_graphs=4 \
    policy.generation.mcore_generation_config.use_cuda_graphs_for_non_decode_steps=true \
    "${precision_args[@]}" \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

grep -q "nccl_reshard bulk comm group" "$RUN_LOG"
grep -q "cuda graph warmup" "$RUN_LOG"
if [[ "${REQUIRE_REAL_NCCL_M2N:-0}" == "1" ]]; then
    grep -q "reshard path: real nccl.m2n.reshard" "$RUN_LOG"
else
    grep -Eq \
        "reshard path: (real nccl.m2n.reshard|xferdtensor_python \(exact-transfer\))" \
        "$RUN_LOG"
fi

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    "max(data[\"train/token_mult_prob_error\"]) < $max_token_mult_prob_error"
