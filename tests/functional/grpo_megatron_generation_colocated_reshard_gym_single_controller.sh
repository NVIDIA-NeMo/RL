#!/bin/bash
# SingleController + NeMo-Gym + colocated Megatron generation, reshard mode:
# TE TP2 training shares both GPUs with inference_optimized TP1 generation on
# a dedicated model, resharded into on every post-step wake. The SC pump
# sleeps the engine for each train step (whole-step phases); Gym spinup
# overlaps the trainer + engine init via the held-socket reservation.
#
# Router replay is on under MCore's async scheduler, so the run also covers
# MInf routing-index capture through the canonical stager, the finalizer's
# route assembly, and the trainer replaying those routes. That needs MoE
# routers, which no small pretrained checkpoint provides, so the served model
# is a tiny random-init Qwen3 MoE built below with Qwen3-0.6B's tokenizer and
# chat template. Random weights earn no reward; the gates are engine/trainer
# parity, route coverage, and a positive async scheduling step count, which
# hold regardless of what the model has learned.

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
CHECKPOINT_DIR=$EXP_DIR/checkpoints
DATA_DIR=$EXP_DIR/data
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR $CHECKPOINT_DIR $DATA_DIR

# clean up checkpoint directory on exit
trap "rm -rf $CHECKPOINT_DIR" EXIT

cd $PROJECT_ROOT

# Follow nemo-gym instructions here to get this data:
# https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/nemo-rl-grpo/setup.html#training-nemo-rl-grpo-setup
cd 3rdparty/Gym-workspace/Gym

# We need HF_TOKEN to download the data from huggingface
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[ERROR] HF_TOKEN is not set"
        exit 1
    fi
    echo "hf_token: $HF_TOKEN" >> env.yaml
fi

uv run ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
cd -

# This trimming of the workplace assistant dataset is necessary b/c with all the tools the first prompt is >4000 tokens
# which will cause the generation engine to return nothing on the first prompt and crash RL. Since we want to keep this test short to
# smoke test, we trim all but the first tool
TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > $TRAIN_PATH
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > $VALIDATION_PATH

# Tiny random-init Qwen3 MoE (2 layers, 4 experts, top-2) sharing Qwen3-0.6B's
# tokenizer. Sized so TP2 training and TP1 inference both divide evenly and
# TE grouped GEMM alignment holds. MCore rejects MoE training under TP without
# sequence parallelism, so the run enables it. Its KV blocks are tiny (128 KiB),
# so the recipe's 10 GB inference buffer would default max_requests to ~82k,
# past max_tokens (16384), which MCore asserts against; 1 GB keeps it at ~8k.
MODEL_DIR=$DATA_DIR/tiny_qwen3_moe
uv run python - "$MODEL_DIR" <<'PY'
import sys

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

model_dir = sys.argv[1]
base = "Qwen/Qwen3-0.6B"
base_config = AutoConfig.from_pretrained(base)
tokenizer = AutoTokenizer.from_pretrained(base)
config = Qwen3MoeConfig(
    vocab_size=base_config.vocab_size,
    hidden_size=128,
    intermediate_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_experts=4,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    norm_topk_prob=True,
    max_position_embeddings=base_config.max_position_embeddings,
    tie_word_embeddings=False,
    bos_token_id=base_config.bos_token_id,
    eos_token_id=base_config.eos_token_id,
)
torch.manual_seed(0)
model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"tiny Qwen3 MoE written to {model_dir}")
PY

uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo_single_controller.py \
    --config $PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml \
    policy.model_name=$MODEL_DIR \
    ++policy.router_replay.enabled=true \
    policy.dtensor_cfg.enabled=false \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=2 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=true \
    policy.generation.backend=megatron \
    +policy.generation.refit_transport=mcore \
    policy.generation.mcore_generation_config.expose_http_server=true \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    policy.generation.mcore_generation_config.refit_backend=nccl \
    ++policy.generation.mcore_generation_config.buffer_size_gb=1 \
    ++policy.generation.mcore_generation_config.async_sched_mode=async \
    policy.generation.max_new_tokens=128 \
    policy.max_total_sequence_length=512 \
    policy.generation.colocated.enabled=true \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=10 \
    grpo.val_period=-1 \
    grpo.val_at_start=false \
    grpo.async_grpo=null \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    loss_fn.reference_policy_kl_penalty=0.01 \
    grpo.skip_reference_policy_logprobs_calculation=false \
    loss_fn.use_importance_sampling_correction=true \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=true \
    checkpointing.checkpoint_dir=$CHECKPOINT_DIR \
    checkpointing.save_period=5 \
    checkpointing.metric_name=null \
    ++checkpointing.save_data_plane=true \
    data.train.data_path=$TRAIN_PATH \
    data.validation.data_path=$VALIDATION_PATH \
    ++token_capture.enabled=true \
    ++data_plane.enabled=true \
    ++data_plane.impl=transfer_queue \
    ++data_plane.backend=simple \
    ++data_plane.simple.storage_capacity=1000000 \
    ++data_plane.simple.num_storage_units=2 \
    ++data_plane.claim_meta_poll_interval_s=0.5 \
    ++async_rl.sampler.name=in_order \
    ++async_rl.sampler.max_lookahead_versions=0 \
    ++async_rl.min_groups_for_streaming_train=4 \
    ++async_rl.max_inflight_prompts=4 \
    ++async_rl.max_buffered_rollouts=4 \
    $@ \
    2>&1 | tee $RUN_LOG

if ! grep -q "\[colocated-reshard\] building dedicated inference model" $RUN_LOG; then
    echo "FAIL: dedicated-model build log line not found (reshard path not exercised)"
    exit 1
fi

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Parity gates hold for random weights. Route coverage of exactly 1 means every
# finalized row carried MInf-recorded routes: the capture, staging, and route
# assembly path ran end to end rather than falling back to the trainer's router.
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05' \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'min(data["train/finalize/routed_experts_row_coverage"]) == 1' \
    'max(data["train/finalize/capture_poisoned_rollouts"]) == 0'

# The counter includes both overlapped and non-overlapped async orderings; a
# positive value confirms that routes were recorded under the async scheduler.
ASYNC_SCHED_STEPS=$(grep -o 'mcore async scheduling steps (cumul): [0-9]*' $RUN_LOG | grep -o '[0-9]*$' | sort -n | tail -1 || true)
if [[ -z "${ASYNC_SCHED_STEPS:-}" ]]; then
    echo "FAIL: async scheduling counter not found"
    exit 1
fi
if [[ "$ASYNC_SCHED_STEPS" -eq 0 ]]; then
    echo "FAIL: async scheduler reported 0 scheduling steps"
    exit 1
fi
echo "async scheduling steps: $ASYNC_SCHED_STEPS"
