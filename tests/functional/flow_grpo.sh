#!/usr/bin/env bash
# Functional test for flow-GRPO: a 1-GPU sanity run of `flow_grpo_train`.
#
# Layers CLI overrides on the OCR exemplar config to train 5 steps of
# `tiny-random/Qwen-Image` with the `jpeg_compressibility` reward (a real
# optimizable signal), then asserts metric health (ratio window, grad norm)
# via check_metrics. Designed for a 16 GB GPU.
#
# This script is wired into `tests/functional/L1_Functional_Tests_Other_2.sh`;
# the CI-facing nightly recipe lives at
# `tests/test_suites/diffusion/flow_grpo-qwen-image-ocr-1n8g-dp8-lora.sh`.
#
# Usage:
#   bash tests/functional/flow_grpo.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# --extra diffusion: diffusers lives behind the optional extra and is not
# part of the base CI environment. The policy worker itself launches in the
# automodel+diffusion venv via the actor registry
# (PY_EXECUTABLES.AUTOMODEL_DIFFUSION). Other environments can inject an
# interpreter via NRL_PYTHON.
PY=${NRL_PYTHON:-uv run --frozen --extra diffusion python}
export PATH="$HOME/.local/bin:$PATH"

LOG_DIR="${LOG_DIR:-results/flow_grpo/$(date +%Y%m%d_%H%M%S)}"
# Timestamped per run: a reused checkpoint dir would auto-resume at step 5
# and the run would train 0 steps.
CKPT_DIR="${CKPT_DIR:-$LOG_DIR/ckpts}"
JSON_METRICS="$LOG_DIR/metrics.json"
mkdir -p "$LOG_DIR"
echo "Logging to: $LOG_DIR"

# Visually simple, compressible prompts so the jpeg_compressibility reward
# has headroom to rise within 5 steps.
cat > "$LOG_DIR/prompts.txt" <<'EOF'
A red circle on a white background
A blue square on a black background
A green triangle next to an orange rectangle
The word HELLO written in bold letters
A yellow star above a purple crescent moon
Two overlapping circles, one red and one blue
A gradient from dark blue to light blue
A checkerboard pattern of black and white squares
EOF

# Overrides shrink the OCR exemplar to the tiny 1-GPU setup: tiny-random
# model, 128px/8-step pipeline, rank-4 LoRA on qkv, 1x4 rollout groups and
# the CPU jpeg_compressibility reward with a single worker.
$PY examples/run_flow_grpo.py \
  --config examples/configs/flow_grpo_qwen_image_ocr.yaml \
  policy.model_name="tiny-random/Qwen-Image" \
  policy.train_micro_batch_size=4 \
  policy.enable_gradient_checkpointing=false \
  policy.optimizer.lr=1.0e-2 \
  policy.optimizer.weight_decay=0.0 \
  policy.pipeline.height=128 \
  policy.pipeline.width=128 \
  policy.pipeline.num_inference_steps=8 \
  policy.pipeline.true_cfg_scale=1.0 \
  policy.pipeline.max_sequence_length=64 \
  policy.algo.noise_level=0.7 \
  policy.algo.sde_window_size=8 \
  'policy.algo.sde_window_range=[0,8]' \
  policy.lora_cfg.rank=4 \
  policy.lora_cfg.alpha=8 \
  'policy.lora_cfg.target_modules=[*.attn.to_q,*.attn.to_k,*.attn.to_v]' \
  flow_grpo.num_prompts_per_step=1 \
  flow_grpo.num_generations_per_prompt=4 \
  flow_grpo.max_num_steps=5 \
  flow_grpo.ppo_epochs=4 \
  flow_grpo.ppo_mini_batch_size=4 \
  flow_grpo.val_period=0 \
  flow_grpo.val_at_start=false \
  loss_fn.ratio_clip_min=0.2 \
  loss_fn.ratio_clip_max=0.2 \
  'env.image_reward.plugins=[{name:jpeg_compressibility,weight:1.0}]' \
  env.image_reward.num_cpus_per_worker=1 \
  env.image_reward.num_workers_per_plugin=1 \
  cluster.gpus_per_node=1 \
  data.train.prompt_file="$LOG_DIR/prompts.txt" \
  data.val.prompt_file="$LOG_DIR/prompts.txt" \
  logger.log_dir="$LOG_DIR" \
  logger.tensorboard_enabled=True \
  logger.num_val_samples_to_print=0 \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  checkpointing.save_period=5 \
  2>&1 | tee "$LOG_DIR/run.log"

# The overrides train 5 steps with save_period=5, so step_5 must exist. The
# CheckpointManager writes the policy under step_N/policy/, and the Automodel
# Checkpointer lays it out as model/ (LoRA adapter) + optim/ (DCP optimizer
# state). A step_N/ that exists at all is already complete: the manager only
# renames tmp_step_N to step_N once the save returned.
CHECKPOINT_DIR="$CKPT_DIR/step_5/policy"
if [[ ! -e "$CHECKPOINT_DIR/model/adapter_model.safetensors" ]] || \
   [[ ! -e "$CHECKPOINT_DIR/optim/.metadata" ]]; then
  echo "FAILED: $CHECKPOINT_DIR is missing the LoRA adapter or DCP optimizer metadata" >&2
  exit 1
fi

# Convert tensorboard logs to json and assert metric health.
# No --frozen here: check_metrics.py is a PEP 723 inline-metadata script and
# --frozen would demand a per-script lockfile.
uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    "median(data['train/mean_ratio']) > 0.5" \
    "median(data['train/mean_ratio']) < 1.5" \
    "max(data['train/grad_norm']) < 100"

echo "Functional test OK: checkpoint at $CHECKPOINT_DIR, metrics healthy."
