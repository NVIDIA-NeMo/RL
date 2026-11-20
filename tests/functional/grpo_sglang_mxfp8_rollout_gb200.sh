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
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
results=$(mktemp -d "${TMPDIR:-/tmp}/sglang-gb200-mxfp8.XXXXXX")
echo "Saving paired metrics under $results"
recipe=examples/configs/grpo_math_1B_sglang.yaml
shared=(
    --config "$recipe"
    policy.model_name=Qwen/Qwen3-0.6B
    policy.automodel_cfg.enabled=false
    policy.megatron_cfg.enabled=true
    cluster.gpus_per_node=2
    data.train.dataset_name=GSM8K
    +data.train.subset=main
    +data.train.split=train
    +data.train.extract_answer=true
    data.train.split_validation_size=0
    '~data.train.seed'
    'policy.tokenizer.chat_template_kwargs={enable_thinking: false}'
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    grpo.max_num_steps=3
    policy.train_global_batch_size=4
    policy.train_micro_batch_size=1
    policy.max_total_sequence_length=256
    policy.generation.sglang_cfg.context_length=256
    policy.generation.max_new_tokens=128
    logger.tensorboard_enabled=true
    logger.wandb_enabled=false
    checkpointing.enabled=false
)
for precision in bf16 mxfp8; do
    quant=("policy.generation.sglang_cfg.quantization.scheme=$precision")
    if [[ "$precision" == mxfp8 ]]; then
        quant+=("+policy.generation.sglang_cfg.quantization.converted_model_path=$results/checkpoint")
        uv run --extra sglang python examples/converters/convert_hf_to_mxfp8.py "${shared[@]}" "${quant[@]}"
    fi
    uv run python examples/run_grpo.py "${shared[@]}" "${quant[@]}" "logger.log_dir=$results/$precision"
    uv run tests/json_dump_tb_logs.py "$results/$precision" --output_path "$results/$precision.json"
    uv run tests/check_metrics.py "$results/$precision.json" \
        'len(data["train/token_mult_prob_error"]) == 3' \
        'all_finite(data["train/token_mult_prob_error"])' \
        'max(data["train/token_mult_prob_error"]) < 2.0'
    jq -c --arg precision "$precision" '{($precision): .["train/token_mult_prob_error"]}' "$results/$precision.json"
done
