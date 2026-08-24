#!/bin/bash
# Verifies the policy.ref_model_from_base_weights flag on the DTensor v2
# (automodel) path: with the flag on, the KL reference stays anchored to the
# base (model_name) weights across a resume; with it off (default), the
# reference is captured from the restored checkpoint, so KL collapses to ~0 at
# the resume boundary.
#
# Three runs share one TRAIN_CMD (same max_num_steps, so the resumed runs train
# exactly the step the baseline trained uninterrupted):
#   Run 1 (baseline): fresh run to step 3, checkpointing step_2 on the way.
#     Its step-3 kl_penalty is the uninterrupted-anchor reference value.
#   Run 2 (flag off): resumes from a copy of step_2. The reference re-anchors
#     to the checkpoint, so step-3 kl_penalty must collapse to ~0.
#   Run 3 (flag on): resumes from another copy of step_2 with
#     policy.ref_model_from_base_weights=true. The reference stays on base
#     weights, so step-3 kl_penalty must stay in the baseline's range.
# The learning rate is raised so two steps of drift produce a KL signal well
# above numeric noise; assertions are directional (collapse vs continuity)
# rather than exact, to stay robust to sampling variance.

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
CKPT_OFF=$EXP_DIR/ckpts_off
CKPT_ON=$EXP_DIR/ckpts_on

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
    echo "FAIL: baseline run must not defer (flag off, fresh run)"
    exit 1
fi

# Give each resumed run its own copy of the step_2 checkpoint so neither sees
# the other's (or the baseline's) later steps.
mkdir -p $CKPT_OFF $CKPT_ON
cp -r $CKPT_BASE/step_2 $CKPT_OFF/step_2
cp -r $CKPT_BASE/step_2 $CKPT_ON/step_2

# --- Run 2 (flag off, default): resume; reference re-anchors to the ckpt. ---
echo "=== Run 2: resume with ref_model_from_base_weights=false (default) ==="
"${TRAIN_CMD[@]}" \
    checkpointing.checkpoint_dir=$CKPT_OFF \
    logger.log_dir=$LOG_DIR/run_off \
    $@ \
    2>&1 | tee $EXP_DIR/run_off.log

if grep -q "$DEFER_LOG_LINE" $EXP_DIR/run_off.log; then
    echo "FAIL: default resume must not defer the checkpoint load"
    exit 1
fi
echo "✅ default resume did not defer the checkpoint load"

# --- Run 3 (flag on): resume; reference stays anchored to base weights. ---
echo "=== Run 3: resume with ref_model_from_base_weights=true ==="
"${TRAIN_CMD[@]}" \
    checkpointing.checkpoint_dir=$CKPT_ON \
    policy.ref_model_from_base_weights=true \
    logger.log_dir=$LOG_DIR/run_on \
    $@ \
    2>&1 | tee $EXP_DIR/run_on.log

if ! grep -q "$DEFER_LOG_LINE" $EXP_DIR/run_on.log; then
    echo "FAIL: flag-on resume did not defer the checkpoint load"
    exit 1
fi
echo "✅ flag-on resume deferred the checkpoint load"

# --- Metric assertions on the resume-boundary step (step 3). ---
uv run tests/json_dump_tb_logs.py $LOG_DIR/run_base --output_path $EXP_DIR/metrics_base.json
uv run tests/json_dump_tb_logs.py $LOG_DIR/run_off --output_path $EXP_DIR/metrics_off.json
uv run tests/json_dump_tb_logs.py $LOG_DIR/run_on --output_path $EXP_DIR/metrics_on.json

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
off3 = kl_at_step_3("off")
on3 = kl_at_step_3("on")
print(f"step-3 kl_penalty: baseline={base3:.3e} flag_off={off3:.3e} flag_on={on3:.3e}")

# The baseline must have drifted measurably off the base weights by step 3,
# otherwise the collapse/continuity assertions below are vacuous. The KL metric
# has a numeric noise floor of ~5e-4 at this scale (bf16 differences between the
# reference-logprob pass and the training pass), so the thresholds below compare
# against the baseline signal, which the raised learning rate keeps well above
# that floor.
assert base3 > 1e-5, f"baseline KL too small to test against ({base3:.3e})"
# Flag off: reference re-anchored to the resumed checkpoint -> KL collapses to
# the noise floor.
assert off3 < 0.25 * base3, (
    f"default resume KL did not collapse: {off3:.3e} vs baseline {base3:.3e}"
)
# Flag on: reference stays on base weights -> KL continuous with the baseline.
assert on3 > 0.5 * base3, (
    f"flag-on resume KL not continuous with baseline: {on3:.3e} vs {base3:.3e}"
)
assert on3 > 3 * off3, (
    f"flag-on resume KL ({on3:.3e}) not clearly above collapsed KL ({off3:.3e})"
)
print("✅ KL collapses at the resume boundary by default and stays anchored with the flag")
EOF

echo "✅ grpo_automodel_ref_anchor_resume passed"
