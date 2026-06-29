#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source $SCRIPT_DIR/common.env

# ===== BEGIN CONFIG =====
NUM_NODES=1
MAX_STEPS=100
NUM_MINUTES=30
STUDENT_MODEL=meta-llama/Llama-3.2-1B
TEACHER0_MODEL=Qwen/Qwen3-4B            # cross-tokenizer (P-KL via projection)
TEACHER1_MODEL=meta-llama/Llama-3.2-3B  # same tokenizer/vocab as student
# ===== END CONFIG =====

exit_if_max_steps_reached

cd $PROJECT_ROOT

# Build teacher0's (student, teacher) runtime projection matrix on the fly from
# the HF tokenizer pair, so this test carries no binary dependency.
# --enable-exact-match constructs a usable matrix from tokenizer surface forms
# directly, skipping the optional (slow) embedding-seed pass that
# build_projection_matrix.sh runs. teacher1 (Llama-3.2-3B) shares the student
# tokenizer, so it is a same-vocab teacher and needs no projection.
PROJ_DIR=$EXP_DIR/projection
mkdir -p $PROJ_DIR
PROJ_PATH=$PROJ_DIR/xtoken_nightly_special.pt
if [[ ! -f $PROJ_PATH ]]; then
  uv run python -m tools.x_token.minimal_projection_via_multitoken \
      --student-model $STUDENT_MODEL \
      --teacher-model $TEACHER0_MODEL \
      --top-k 4 \
      --enable-special-token-mapping \
      --enable-exact-match \
      --disable-reverse-pass \
      --disable-scale-trick \
      --output-filename xtoken_nightly \
      --output-dir $PROJ_DIR
fi

# Run the experiment. Parallelism (student TP4xCP2 <- teachers TP2xCP2), the two
# teachers (teacher0 Qwen3-4B cross-tok + teacher1 Llama-3.2-3B same-vocab),
# loss mode (kd_loss_mode=sum), and batch/seq sizes live in the recipe YAML.
# Only teacher0's projection path is injected here; teacher1 keeps the recipe's
# null projection (same-vocab => direct per-position KL, no projection).
uv run examples/run_xtoken_off_policy_distillation.py \
    --config $CONFIG_PATH \
    distillation.max_num_steps=$MAX_STEPS \
    teachers.0.projection_matrix_path=$PROJ_PATH \
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
    # train/loss is the aggregate (sum-mode total KD + CE) loss. Over 100 steps
    # the sharded student goes from step1 ~1.88 to mean(last5) ~1.0 and accuracy
    # 0.78 -> ~0.86 (per-step loss is noisy at this tiny GBS, so checks keep
    # margin). A student-logit gradient bug under TP/CP would leave the loss
    # flat near step1 and accuracy flat, which the decrease + accuracy checks
    # catch.
    # kl_loss_t0 (Qwen3-4B cross-tok P-KL) and kl_loss_t1 (Llama-3.2-3B
    # same-vocab direct KL) are the per-teacher KD terms summed by
    # kd_loss_mode=sum -- the >=2-teacher guard. Both keys only exist if both
    # teachers fire and their per-teacher CP-buffer sharding + aggregation work
    # (a single-teacher regression would drop one term -> KeyError -> fail).
    # kl_loss_t0 falls ~0.51 -> ~0.26 as the cross-tok KD is minimized. The
    # same-vocab term kl_loss_t1 stays small and roughly flat (~0.08-0.21) since
    # the 3B teacher and 1B student already share a vocab, so it is bracketed by
    # an upper AND lower bound: the lower bound catches a broken/zeroed
    # same-vocab path, the upper bound catches a blow-up.
    # validation/kl_loss is the held-out distillation KL (val_period=10): it
    # falls ~0.068 -> ~0.045. A CP-non-invariance regression in the sharded loss
    # would leave it elevated, which the val checks below catch.
    uv run tests/check_metrics.py $JSON_METRICS \
        'data["train/loss"]["1"] < 2.2' \
        'mean(data["train/loss"], -5, -1) < 1.3' \
        'mean(data["train/loss"], -5, -1) < 0.7 * data["train/loss"]["1"]' \
        'mean(data["train/accuracy"], -5, -1) > 0.80' \
        'mean(data["train/kl_loss_t0"], -5, -1) < 0.40' \
        'mean(data["train/kl_loss_t1"], -5, -1) > 0.05' \
        'mean(data["train/kl_loss_t1"], -5, -1) < 0.20' \
        'data["validation/kl_loss"]["100"] < 0.06' \
        'data["validation/kl_loss"]["100"] < 0.80 * data["validation/kl_loss"]["0"]' \
        'max(data["ray/node.0.gpu.0.mem_gb"]) < 40'
fi
