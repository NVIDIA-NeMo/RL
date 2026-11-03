#!/bin/bash
# Forest (context compaction) GRPO checkpoint/resume: two steps with checkpoints,
# then a second run resumes from step 2 to step 4.
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
# Outside the runs' own experiment dirs, which each run deletes.
CKPT_DIR=$EXP_DIR/checkpoints
rm -rf $EXP_DIR
mkdir -p $EXP_DIR

CHECKPOINT_ARGS=(
    checkpointing.enabled=true
    checkpointing.checkpoint_dir=$CKPT_DIR
    checkpointing.metric_name=null
    checkpointing.save_period=1
    ++checkpointing.save_data_plane=true
)
for run_spec in "run1 2" "run2 4"; do
    read -r run steps <<< "$run_spec"
    SC_TEST_CONTEXT_COMPACTION=1 \
    SC_TEST_EXP_NAME=${EXP_NAME}_${run} \
    RUN_CONVERGENCE_CHECKS=0 \
        bash $SCRIPT_DIR/grpo_async_gym_single_controller.sh \
        grpo.max_num_steps=$steps "${CHECKPOINT_ARGS[@]}" "$@"
done

test -d $CKPT_DIR/step_2/data_plane
test -f $CKPT_DIR/step_2/replay_buffer_metadata.pt
grep -q "Restoring replay buffer metadata" $SCRIPT_DIR/${EXP_NAME}_run2/run.log
test -d $CKPT_DIR/step_4/policy/weights

cd $SCRIPT_DIR/../..
for run_spec in "run1 2" "run2 2"; do
    read -r run steps <<< "$run_spec"
    metrics=$EXP_DIR/metrics_${run}.json
    uv run tests/json_dump_tb_logs.py $SCRIPT_DIR/${EXP_NAME}_${run}/logs --output_path $metrics
    # Eight logical rollouts per step. More valid rows requires a context
    # boundary; padding rows have zero sample_mask and cannot satisfy it.
    uv run tests/check_metrics.py $metrics \
        'max(data["train/finalize/total_ms"]) > 0' \
        'max(data["train/global_valid_seqs"]) > 8' \
        'max(data["train/token_mult_prob_error"]) < 1.05' \
        "len(data[\"train/token_mult_prob_error\"]) == ${steps}"
done
