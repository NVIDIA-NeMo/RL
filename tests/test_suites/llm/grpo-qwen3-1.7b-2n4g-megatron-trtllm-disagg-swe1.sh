#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=2
GPUS_PER_NODE=4
STEPS_PER_RUN=5
MAX_STEPS=5
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=60
USES_SANDBOX=1
USE_GYM_CONTAINER=true
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
VLLM_CACHE_DIR=${HF_HOME}/vllm_compile_cache \
FLASHINFER_CUBIN_CACHE=${HF_HOME}/flashinfer_cubins \
FLASHINFER_WS_BASE=${HF_HOME}/flashinfer_workspace \
uv run examples/nemo_gym/run_grpo_nemo_gym.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    env.nemo_gym.uv_venv_dir=/opt/gym_venvs \
    $@ \
    2>&1 | tee $RUN_LOG

# A disagg replica serves through its frontends only. TrtllmGeneration's
# direct token-in-token-out path raises rather than silently running without
# disaggregation; surface that as an attributable failure instead of letting
# the run die with a bare traceback.
assert_not_grep \
    "PD disaggregation is only wired for the HTTP/NeMo-Gym rollout" \
    "$RUN_LOG" \
    "Rollouts took the direct engine path instead of the disagg frontends"

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # Same gates as the vLLM sibling (grpo-qwen3-1.7b-1n8g-megatron-super-swe1).
    # token_mult_prob_error is the one that matters here: the KV handoff, the
    # base64 token-id relay and frontend-side tokenization all change how the
    # sampled tokens are produced, and any disagreement with the trainer's
    # recomputed logprobs shows up in it first.
    uv run tests/check_metrics.py $JSON_METRICS \
        'median(data["train/token_mult_prob_error"]) < 1.1' \
        "data['train/token_mult_prob_error']['$MAX_STEPS'] < 1.1" \
        'mean(data["train/gen_kl_error"]) < 0.02'

    # Clean up checkpoint directory after successful run to save space.
    rm -rf "$CKPT_DIR"
fi
