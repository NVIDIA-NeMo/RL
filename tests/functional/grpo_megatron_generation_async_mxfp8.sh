#!/bin/bash

# Async non-colocated Megatron generation on nano-3.5 (30B-A3B mamba-hybrid MoE)
# with MXFP8 training. Runs generation on a dedicated inference node while training
# overlaps on the training node (1-off async, in-flight weight updates), quantizing
# the training GEMMs to MXFP8.
#
# This is a CLUSTER test (GB200 / oci-hsg). It expects to run from /opt/nemo-rl on
# the HEAD node of a 2-node `cog submit --launcher ray` allocation (1 gen + 1 train
# node, 4 GPUs each -> EP=4). Weights come from the public nano-3.5 release on the
# HF hub; override MODEL_NAME to point at a local HF snapshot instead. See
# .claude/skills/run-nano35-megatron-inference-cog/SKILL.md for the wiring.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

# Public nano-3.5 release. The first run downloads it and converts HF -> Megatron;
# the converted checkpoint is cached under $NRL_MEGATRON_CHECKPOINT_DIR (or
# $HF_HOME/nemo_rl), which must be visible to both nodes.
: "${MODEL_NAME:=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16}"

# expandable_segments defrags the tight nano-3.5 training footprint (megatron gen path).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd $PROJECT_ROOT
uv run --no-sync python $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml \
    policy.model_name=$MODEL_NAME \
    policy.tokenizer.name=$MODEL_NAME \
    policy.generation.backend=megatron \
    policy.max_total_sequence_length=2048 \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    policy.make_sequence_length_divisible_by=32 \
    ++loss_fn.reference_policy_kl_penalty=0 \
    ++loss_fn.use_importance_sampling_correction=true \
    grpo.async_grpo.enabled=true \
    ++grpo.async_grpo.max_trajectory_age_steps=1 \
    ++grpo.async_grpo.in_flight_weight_updates=true \
    policy.megatron_cfg.fp8_cfg.enabled=true \
    policy.megatron_cfg.fp8_cfg.fp8=e4m3 \
    policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8 \
    policy.megatron_cfg.fp8_cfg.fp8_param=false \
    policy.generation.colocated.enabled=false \
    ++policy.generation.colocated.resources.gpus_per_node=4 \
    ++policy.generation.colocated.resources.num_nodes=1 \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.refit_backend=nccl \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.expert_model_parallel_size=4 \
    ++policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
    ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.context_parallel_size=1 \
    ++policy.generation.mcore_generation_config.sequence_parallel=false \
    ++policy.generation.mcore_generation_config.buffer_size_gb=20 \
    ++policy.generation.mcore_generation_config.num_cuda_graphs=-1 \
    ++policy.generation.mcore_generation_config.cuda_graph_impl=local \
    ++policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    ++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false \
    policy.megatron_cfg.activation_checkpointing=true \
    grpo.max_num_steps=5 \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.val_at_start=false \
    grpo.val_period=0 \
    cluster.gpus_per_node=4 \
    cluster.num_nodes=2 \
    checkpointing.enabled=false \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Smoke-level threshold. MXFP8 quantizes the TRAINING GEMMs, so the training-side
# recomputed logprobs diverge more from the bf16 generation logprobs than on a
# pure-bf16 run (gen KL error ~1e-2 vs ~1e-3, token_mult_prob_error ~1.3 vs ~1.02
# measured on nano-3.5). This is expected for FP8 training and is why async uses
# importance-sampling correction; the bound only guards against a gross blowup
# (e.g. FP8-param NaNs). Tighten once we have more clean runs on CI.
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.5'
