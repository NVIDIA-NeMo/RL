#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_eval_random_dataset.py \
    generation.model_name=Qwen/Qwen3-0.6B \
    tokenizer.name=Qwen/Qwen3-0.6B \
    generation.max_new_tokens=4 \
    ++generation.output_len_or_output_len_generator=4 \
    generation.ignore_eos=true \
    generation.num_prompts_per_step=2 \
    generation.vllm_cfg.max_model_len=32 \
    generation.vllm_cfg.gpu_memory_utilization=0.45 \
    data.max_input_seq_length=16 \
    ++data.input_len_or_input_len_generator=8 \
    ++data.num_samples=2 \
    cluster.gpus_per_node=1 \
    $@ \
    2>&1 | tee $RUN_LOG

cat $RUN_LOG | grep "score=" | sed 's/.*score=\([^ ]*\).*/{"score": \1}/' > $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
  'data["score"] == 0.0'
