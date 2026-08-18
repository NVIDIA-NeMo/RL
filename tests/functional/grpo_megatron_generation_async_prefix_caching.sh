#!/bin/bash

# Async non-colocated Megatron generation on nano-3.5 (30B-A3B mamba-hybrid MoE)
# with PREFIX CACHING enabled, in a scenario engineered to force real cache hits:
#   - num_prompts_per_step=1, num_generations_per_prompt=16 -> every rollout in a
#     step is the SAME prompt, so the whole prompt is a shared prefix.
#   - a long fixed instruction preamble (examples/prompts/prefix_caching_cot_prompt.txt,
#     ~430 tokens) is prepended to every prompt. mcore only caches/matches WHOLE
#     KV blocks and the paged-KV block size must be divisible by 256, so the prefix
#     has to exceed one 256-token block to be cacheable. The preamble guarantees the
#     leading 256-block is byte-identical across all rollouts (and across steps).
# The mcore coordinator ("first_prefix_block") then serves the shared block(s) from
# the cache on the 2nd..16th rollout. The test asserts the cache is actually HIT
# (not merely enabled) via the counter surfaced by the worker.
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
    ++loss_fn.reference_policy_kl_penalty=0 \
    ++loss_fn.use_importance_sampling_correction=true \
    grpo.async_grpo.enabled=true \
    ++grpo.async_grpo.max_trajectory_age_steps=1 \
    ++grpo.async_grpo.in_flight_weight_updates=true \
    policy.generation.colocated.enabled=false \
    ++policy.generation.colocated.resources.gpus_per_node=4 \
    ++policy.generation.colocated.resources.num_nodes=1 \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.refit_backend=nccl \
    ++policy.generation.mcore_generation_config.enable_prefix_caching=true \
    ++policy.generation.mcore_generation_config.logging_step_interval=1 \
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
    data.default.prompt_file=$PROJECT_ROOT/examples/prompts/prefix_caching_cot_prompt.txt \
    grpo.max_num_steps=5 \
    grpo.num_prompts_per_step=1 \
    grpo.num_generations_per_prompt=16 \
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

# Correctness: gen<->train logprob agreement must stay tight (prefix caching must
# not corrupt logprobs, cf. the vLLM prefix-caching NaN bug).
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.1'

# Prefix-cache HIT assertion: with num_generations_per_prompt=16 the same prompt
# prefix is served many times, so the mcore prefix cache MUST report hits. mcore's
# own DynamicInferenceEngine counters are only surfaced via its (wandb) metrics
# writer / a logging.info line that NeMo-RL does not configure to emit, so the
# MegatronPolicyWorker mirrors the cumulative counters to captured stdout when
# prefix caching is on (see megatron_worker.py `_sleep`):
#     [Rank <r>] mcore prefix cache (cumul): <N> hits, <M> blocks matched
# Fail loudly if no hit is observed, so an accidentally-disabled cache (or a
# regression that stops matching prefixes) cannot pass silently.
PREFIX_CACHE_HIT_PATTERN='mcore prefix cache \(cumul\): [1-9][0-9]* hits'
echo "Verifying prefix-cache hits in $RUN_LOG ..."
if grep -aE "$PREFIX_CACHE_HIT_PATTERN" "$RUN_LOG" > "$EXP_DIR/prefix_cache_hits.txt"; then
    echo "PASS: prefix-cache hits observed (last line):"
    tail -1 "$EXP_DIR/prefix_cache_hits.txt"
else
    echo "FAIL: no prefix-cache hit signal found in generation log."
    echo "      Expected a line matching: $PREFIX_CACHE_HIT_PATTERN"
    echo "      (prefix caching enabled + num_generations_per_prompt=16 should force hits)"
    exit 1
fi
