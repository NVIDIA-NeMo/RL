#!/bin/bash
# Verifies that on the DTensor v2 (automodel) path the KL reference stays
# anchored to the base (model_name) weights across a resume: the worker defers
# the NeMo RL checkpoint load, captures the reference from the pristine base
# weights, then loads the checkpoint. Without the deferred load the reference
# would re-anchor to the resumed checkpoint and KL would collapse to ~0 at
# every resume boundary, granting the policy a fresh drift budget per resume.
#
# Two runs share one TRAIN_CMD (same max_num_steps, so the resumed run trains
# exactly the step the baseline trained uninterrupted):
#   Run 1 (baseline): fresh run to step 3, checkpointing step_2 on the way.
#     Its step-3 kl_penalty is the uninterrupted-anchor reference value.
#   Run 2 (resume): resumes from a copy of step_2. The reference must stay on
#     base weights, so step-3 kl_penalty must stay in the baseline's range
#     rather than collapsing to the numeric noise floor (~5e-4 at this scale).
# The learning rate is raised so two steps of drift produce a KL signal well
# above that floor; the assertion is directional (continuity) rather than
# exact, to stay robust to sampling variance.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR $LOG_DIR

CKPT_BASE=$EXP_DIR/ckpts_base
CKPT_RESUME=$EXP_DIR/ckpts_resume

TRAIN_CMD=(
    uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl
    $PROJECT_ROOT/examples/run_grpo.py
    policy.model_name=Qwen/Qwen3-0.6B
    grpo.num_prompts_per_step=2
    grpo.num_generations_per_prompt=4
    policy.train_global_batch_size=4
    policy.train_micro_batch_size=1
    policy.optimizer.kwargs.lr=2e-4
    cluster.gpus_per_node=2
    grpo.max_num_steps=3
    logger.tensorboard_enabled=true
    logger.wandb_enabled=false
    logger.monitor_gpus=false
    checkpointing.enabled=true
    checkpointing.save_period=2
    checkpointing.metric_name=null
)

DEFER_LOG_LINE="Deferring NeMo RL checkpoint load"

cd $PROJECT_ROOT

# --- Run 1 (baseline): fresh run to step 3, saving step_2 on the way. ---
echo "=== Run 1: uninterrupted baseline ==="
"${TRAIN_CMD[@]}" \
    checkpointing.checkpoint_dir=$CKPT_BASE \
    logger.log_dir=$LOG_DIR/run_base \
    $@ \
    2>&1 | tee $EXP_DIR/run_base.log

if [[ ! -e "$CKPT_BASE/step_2" ]]; then
    echo "FAIL: step_2 checkpoint missing after baseline run"
    exit 1
fi
if grep -q "$DEFER_LOG_LINE" $EXP_DIR/run_base.log; then
    echo "FAIL: fresh run must not defer the checkpoint load"
    exit 1
fi
echo "✅ fresh run did not defer the checkpoint load"

# Resume from a copy of step_2 so the resumed run cannot see the baseline's
# later steps.
mkdir -p $CKPT_RESUME
cp -r $CKPT_BASE/step_2 $CKPT_RESUME/step_2

# --- Run 2: resume; the reference must stay anchored to base weights. ---
echo "=== Run 2: resume from step_2 ==="
"${TRAIN_CMD[@]}" \
    checkpointing.checkpoint_dir=$CKPT_RESUME \
    logger.log_dir=$LOG_DIR/run_resume \
    $@ \
    2>&1 | tee $EXP_DIR/run_resume.log

if ! grep -q "$DEFER_LOG_LINE" $EXP_DIR/run_resume.log; then
    echo "FAIL: resume with a KL reference did not defer the checkpoint load"
    exit 1
fi
echo "✅ resume deferred the checkpoint load until after reference capture"

# --- Metric assertions on the resume-boundary step (step 3). ---
uv run tests/json_dump_tb_logs.py $LOG_DIR/run_base --output_path $EXP_DIR/metrics_base.json
uv run tests/json_dump_tb_logs.py $LOG_DIR/run_resume --output_path $EXP_DIR/metrics_resume.json

uv run python - "$EXP_DIR" <<'EOF'
import json
import sys

exp_dir = sys.argv[1]

def kl_at_step_3(name):
    with open(f"{exp_dir}/metrics_{name}.json") as f:
        data = json.load(f)
    if "train/kl_penalty" not in data:
        kl_keys = [k for k in data if "kl" in k.lower()]
        raise AssertionError(
            f"train/kl_penalty missing from metrics_{name}.json; kl-ish keys: {kl_keys}"
        )
    return data["train/kl_penalty"]["3"]

base3 = kl_at_step_3("base")
resume3 = kl_at_step_3("resume")
print(f"step-3 kl_penalty: baseline={base3:.3e} resume={resume3:.3e}")

# The baseline must have drifted measurably off the base weights by step 3,
# otherwise the continuity assertion below is vacuous. The KL metric has a
# numeric noise floor of ~5e-4 at this scale (bf16 differences between the
# reference-logprob pass and the training pass); the raised learning rate
# keeps the baseline signal well above that floor.
assert base3 > 1e-2, f"baseline KL too small to test against ({base3:.3e})"
# The reference stays on base weights across the resume, so KL must be
# continuous with the baseline instead of collapsing to the noise floor
# (a re-anchored reference would put it around ~5e-4 here).
assert resume3 > 0.5 * base3, (
    f"resume KL not continuous with baseline: {resume3:.3e} vs {base3:.3e} — "
    "the KL reference likely re-anchored to the resumed checkpoint"
)
print("✅ KL stays anchored to base weights across the resume boundary")
EOF

echo "✅ grpo_automodel_ref_anchor_resume passed"
