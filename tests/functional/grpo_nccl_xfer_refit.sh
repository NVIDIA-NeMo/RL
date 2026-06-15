#!/bin/bash
# Functional smoke for the nccl_xfer disaggregated weight-refit path
# (Megatron train -> vLLM gen on disjoint GPUs), forced onto the broadcast-based
# GOLDEN reshard (NRL_XFERDTENSOR_GOLDEN=1) so it does not require the real
# nccl.xfer op in the container.
#
# 1T1G disaggregated run on 2 GPUs (1 node): Megatron TP1 (train, 1 GPU) ->
# vLLM TP1 (gen, 1 GPU), non-colocated.  This is sized for the 2-GPU functional
# CI runner, so the reshard itself is trivial (fully replicated 1->1), but it
# still EXERCISES the whole disaggregated nccl_xfer code path end to end:
# prepare_nccl_xfer_refit_info / build_nccl_xfer_refit_info, the gen-side
# _build_hf_to_gen_backend_mapping (qkv / gate_up merge slices, lm_head tie),
# nccl_xfer_refit + get_dst_dtensor, the misc packed_broadcast, and
# xferdtensor_golden -- which is the coverage this smoke is here to add.
#
# REAL reshards (TP/EP/PP down- and up-shard, e.g. TP4xDP2 -> TP2xDP4) plus
# MoE / PP / FP8 / large-model coverage live in the SLURM script/new_refit/
# matrix (>=16 GPUs); the MoE expert grouping + w13/w2 mapping are covered by
# the unit tests (tests/unit/distributed/test_nccl_xfer_utils.py and
# tests/unit/models/generation/test_nccl_xfer_backend.py).
#
# Requires the mcore + vllm extras and a 2-GPU allocation (the functional CI
# runner), e.g.:
#   uv run --extra mcore --extra vllm bash tests/functional/grpo_nccl_xfer_refit.sh

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

# Force the golden (broadcast) reshard path so the test does not depend on the
# real nccl.xfer XdtensorRedistribute op being present in the container.
export NRL_XFERDTENSOR_GOLDEN=1

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
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
    +policy.nccl_xfer_refit=true \
    cluster.num_nodes=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# A broken refit corrupts the gen weights -> the importance-sampling ratio
# (train vs gen logprobs) explodes; assert it stays sane across the 2 steps.
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05'
