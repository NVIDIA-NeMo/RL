#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
MAX_STEPS=100
NUM_MINUTES=20
STUDENT_MODEL=meta-llama/Llama-3.2-1B
TEACHER_MODEL=Qwen/Qwen3-4B
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Build the (student, teacher) runtime projection matrix on the fly from the HF
# tokenizer pair, so this test carries no binary dependency. --enable-exact-match
# constructs a usable matrix from tokenizer surface forms directly, skipping the
# optional (slow) embedding-seed pass that build_projection_matrix.sh runs.
PROJ_DIR=$EXP_DIR/projection
mkdir -p $PROJ_DIR
PROJ_PATH=$PROJ_DIR/xtoken_nightly_special.pt
if [[ ! -f $PROJ_PATH ]]; then
  uv run python -m tools.x_token.minimal_projection_via_multitoken \
      --student-model $STUDENT_MODEL \
      --teacher-model $TEACHER_MODEL \
      --top-k 4 \
      --enable-special-token-mapping \
      --enable-exact-match \
      --disable-reverse-pass \
      --disable-scale-trick \
      --output-filename xtoken_nightly \
      --output-dir $PROJ_DIR
fi

# Run the experiment. Parallelism (student TP4xCP2 <- teacher TP2xCP2), models,
# loss mode (P-KL), and batch/seq sizes live in the recipe YAML.
uv run examples/run_xtoken_off_policy_distillation.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    loss_fn.projection_matrix_path=$PROJ_PATH \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=True \
    logger.wandb.project=nemo-rl \
    logger.wandb.name=$EXP_NAME \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    $@ \
    2>&1 | tee $RUN_LOG

# Convert tensorboard logs to json
uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Only run metrics if the target step is reached
if [[ $(jq 'to_entries | .[] | select(.key == "train/loss") | .value | keys | map(tonumber) | max' $JSON_METRICS) -ge $MAX_STEPS ]]; then
    # train/loss is the total P-KL+CE loss. Over 100 steps the sharded student
    # goes from step1 ~1.88 to mean(last5) ~0.97 and accuracy 0.78 -> ~0.86
    # (per-step loss is noisy at this tiny GBS, so checks keep margin). A
    # student-logit gradient bug under TP/CP would leave the loss flat near
    # step1 and accuracy flat, which the decrease + accuracy checks catch.
    # validation/kl_loss is the held-out distillation KL (val_period=10): it
    # falls ~0.057 -> ~0.029 (to ~0.5x). A CP-non-invariance regression in the
    # sharded loss would leave the held-out KL elevated, which the val checks
    # below catch.
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["1"] < 2.2' \
        'mean(data["train/loss"], -5, -1) < 1.15' \
        'mean(data["train/loss"], -5, -1) < 0.7 * data["train/loss"]["1"]' \
        'mean(data["train/accuracy"], -5, -1) > 0.80' \
        'data["validation/kl_loss"]["100"] < 0.04' \
        'data["validation/kl_loss"]["100"] < 0.65 * data["validation/kl_loss"]["0"]' \
        'max(data["ray/node.0.gpu.0.mem_gb"]) < 30'
fi
