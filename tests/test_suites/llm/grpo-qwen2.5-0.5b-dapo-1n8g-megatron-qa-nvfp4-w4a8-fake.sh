#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
GPUS_PER_NODE=8
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd $PROJECT_ROOT
uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    policy.tokenizer.name=Qwen/Qwen2.5-0.5B \
    policy.megatron_cfg.converter_type=Qwen2ForCausalLM \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false \
    policy.megatron_cfg.activation_checkpointing=false \
    policy.megatron_cfg.apply_rope_fusion=true \
    policy.max_total_sequence_length=256 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.logprob_batch_size=4 \
    policy.quant_calib_size=4 \
    policy.quant_batch_size=1 \
    policy.quant_sequence_length=128 \
    policy.generation.max_new_tokens=128 \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.max_model_len=256 \
    policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
    policy.generation.vllm_cfg.enforce_eager=true \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=$MAX_STEPS \
    grpo.val_period=1 \
    grpo.max_val_samples=8 \
    grpo.val_batch_size=8 \
    env.math.num_workers=2 \
    cluster.num_nodes=$NUM_NODES \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/gen_kl_error"]["1"] < 0.008' \
        'max(data["train/token_mult_prob_error"]) < 1.06'

    grep -q "FakeQuantWorker" "$RUN_LOG"
    grep -q "VLLM_QUANT_CFG" "$RUN_LOG"
    ! grep -q "Detected ModelOpt NVFP4 checkpoint" "$RUN_LOG"

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
