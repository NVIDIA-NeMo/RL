#!/bin/bash

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
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT

# cluster.segment_size only engages when Ray nodes carry nvlink_domain_* labels,
# which ray.sub probes from `nvidia-smi -q` ClusterUUID on NVLink-fabric clusters
# (e.g. GB200 NVL72); CI runners have none. Pre-start a Ray head with a synthetic
# domain label so init_ray() attaches to it (externally managed cluster) and the
# topology-aware megatron placement path runs for real.
cleanup() {
    uv run ray stop --force || true
}
trap cleanup EXIT
uv run ray stop --force || true # don't attach to a stale cluster
uv run ray start --head --disable-usage-stats \
    --resources='{"nvlink_domain_ci_synthetic": 1, "topo_rank": 1}'

uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.backend=megatron \
    cluster.gpus_per_node=2 \
    cluster.segment_size=1 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

# Guard against the silent fallback: with no (or unreadable) domain labels the run
# would succeed without ever exercising the topology placement path under test.
grep -q "Topology-aware allocation" $RUN_LOG || {
    echo "ERROR: topology-aware allocation did not engage (no segment selection logged)" >&2
    exit 1
}
# NOTE: `! grep` is exempt from `set -e`, hence the explicit if.
if grep -q "no NVLink domain info" $RUN_LOG; then
    echo "ERROR: segment_size fell back to unordered allocation" >&2
    exit 1
fi

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 1.05'
