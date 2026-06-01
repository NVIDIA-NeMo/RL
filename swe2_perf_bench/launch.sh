#!/usr/bin/env bash
# SWE2 direct multi-turn perf bench launcher.
# Run from inside the NeMo-RL repo: cd /path/to/RL && bash /home/zhiyul/swe2_perf_bench/launch.sh

set -euo pipefail

CONFIG=/home/zhiyul/swe2_perf_bench/grpo_qwen3_30ba3b_thinking_swe2.yaml

: "${MODEL:=Qwen/Qwen3-30B-A3B-Thinking-2507}"
: "${TRAIN_DATA:?set TRAIN_DATA=/path/to/swe2/train-split.jsonl}"
: "${VAL_DATA:?set VAL_DATA=/path/to/swe2/val-split.jsonl}"
: "${SIF_FORMATTER:?set SIF_FORMATTER=/path/to/sif/sweb.eval.x86_64.{instance_id}.sif}"

uv run --frozen ./examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "${CONFIG}" \
  policy.model_name="${MODEL}" \
  data.train.data_path="${TRAIN_DATA}" \
  data.validation.data_path="${VAL_DATA}" \
  "env.nemo_gym.swe_agents_train.responses_api_agents.swe_agents.container_formatter=[${SIF_FORMATTER}]" \
  "env.nemo_gym.swe_agents_val.responses_api_agents.swe_agents.container_formatter=[${SIF_FORMATTER}]" \
  checkpointing.save_period=999999 \
  grpo.val_period=999999 \
  grpo.val_at_start=false \
  grpo.max_num_epochs=1
