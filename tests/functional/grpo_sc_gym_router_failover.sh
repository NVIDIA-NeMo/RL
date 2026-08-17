#!/bin/bash
# SingleController + NeMo-Gym: the router must let Gym survive a dead shard.
#
# The deliberate inverse of grpo_dp_single_controller_chaos.sh. That one runs a
# single-shard fleet and asserts a BOUNDED FAILURE -- there is nowhere to fail over to,
# so the right outcome is a fast, attributable abort. This one runs TWO generation shards
# and asserts SURVIVAL: kill one, and the run finishes because the fleet ledger
# quarantines it and stops pushing its URL to the router.
#
# That property -- "NeMo-Gym never has to fail over, because it holds one NeMo-RL-owned
# URL" -- is the entire reason the router exists, and it had no functional coverage:
# grpo_async_gym_single_controller.sh runs gpus_per_node=1, tensor_parallel_size=1, so
# dp_size=1. With one backend _pick_backend has one choice, set_serving_backends never
# sees a shrinking set, and the no-healthy-backend path never fires. That lane proves
# pass-through and nothing else.
#
# Needs >= 3 GPUs (2 generation + 1 trainer) and self-skips below that rather than
# passing vacuously -- a green tick on a 2-GPU runner would be worse than no test.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
git config --global --add safe.directory "$PROJECT_ROOT"

set -eou pipefail

EXP_NAME=$(basename "$0" .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
DATA_DIR=$EXP_DIR/data
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

# Completed training steps to wait for before killing a shard.
#
# NOT a fixed sleep, which is what this originally did and why job 6251221 proved
# nothing: generation actors exist as soon as the inference cluster is built, but
# NeMo-Gym's _spinup runs afterwards and takes minutes. A timer started at actor
# discovery therefore fired while the run was still in SETUP, the kill took down the
# vLLM server Gym's policy_model process depends on, and Gym's own poll() failed the
# run -- which the harness then reported as "died after losing one of two shards".
# True, and completely misleading: no rollout had ever been served.
#
# "train step N/M" is the marker, because it is only printed once the train pump has
# completed a step, i.e. rollouts are flowing through the router for real.
KILL_AFTER_STEPS=${KILL_AFTER_STEPS:-3}
# Ceiling on that wait. Generous: it covers Gym data prep, engine load and spinup.
STEADY_STATE_WAIT_S=${STEADY_STATE_WAIT_S:-1800}
# How long to wait for generation actors to appear.
ACTOR_WAIT_S=${ACTOR_WAIT_S:-600}
ACTOR_QUERY_TIMEOUT_S=${ACTOR_QUERY_TIMEOUT_S:-120}

free_gpu_count() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '$1 < 1024' | wc -l
}

GPUS=$(free_gpu_count)
if (( GPUS < 3 )); then
    echo "[failover] SKIP: needs >= 3 free GPUs (2 generation + 1 trainer), found $GPUS."
    echo "[failover] Failover cannot be exercised on fewer -- with one generation shard"
    echo "[failover] there is nothing to fail over to, which is what the chaos harness covers."
    exit 0
fi

rm -rf "$EXP_DIR" "$LOG_DIR"
mkdir -p "$EXP_DIR" "$LOG_DIR" "$DATA_DIR"

cd "$PROJECT_ROOT"

# Same Gym data preparation as grpo_async_gym_single_controller.sh.
cd 3rdparty/Gym-workspace/Gym
if [[ ! -f env.yaml ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[failover] FAIL: HF_TOKEN is not set"
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

TRAIN_PATH=$DATA_DIR/workplace_assistant_train.jsonl
VALIDATION_PATH=$DATA_DIR/workplace_assistant_validation.jsonl
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > "$TRAIN_PATH"
jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > "$VALIDATION_PATH"

cleanup() {
    kill -9 $TRAIN_PID 2>/dev/null || true
    # vLLM runs its engine in a child that outlives the actor this test kills on purpose,
    # and it holds tens of GB of device memory. A leak makes the NEXT run fail in
    # placement-group setup, which reads as an unrelated flake.
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    pkill -9 -f "megatron_policy_worker" 2>/dev/null || true
    sleep 5
}
trap cleanup EXIT

# 3 GPUs: 2 generation (dp_size=2, the whole point) + 1 trainer.
uv run "$PROJECT_ROOT/examples/run_grpo_single_controller.py" \
    --config "$PROJECT_ROOT/examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml" \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.dtensor_cfg.enabled=false \
    policy.megatron_cfg.enabled=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false \
    policy.generation.vllm_cfg.tensor_parallel_size=1 \
    policy.generation.vllm_cfg.async_engine=true \
    policy.max_total_sequence_length=512 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.num_nodes=1 \
    policy.generation.colocated.resources.gpus_per_node=2 \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=2 \
    grpo.max_num_steps=150 \
    grpo.val_period=-1 \
    grpo.val_at_start=false \
    grpo.async_grpo=null \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=3 \
    loss_fn.reference_policy_kl_penalty=0.01 \
    grpo.skip_reference_policy_logprobs_calculation=false \
    loss_fn.use_importance_sampling_correction=true \
    logger.tensorboard_enabled=true \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    data.train.data_path="$TRAIN_PATH" \
    data.validation.data_path="$VALIDATION_PATH" \
    ++data_plane.enabled=true \
    ++data_plane.impl=transfer_queue \
    ++data_plane.backend=simple \
    ++data_plane.storage_capacity=1000000 \
    ++data_plane.num_storage_units=2 \
    ++data_plane.claim_meta_poll_interval_s=0.5 \
    ++data_plane.global_segment_size=549755813888 \
    ++data_plane.local_buffer_size=68719476736 \
    ++async_rl.sampler.name=in_order \
    ++async_rl.sampler.max_lookahead_versions=0 \
    ++async_rl.min_groups_for_streaming_train=4 \
    ++async_rl.max_inflight_prompts=4 \
    ++async_rl.max_buffered_rollouts=4 \
    ++async_rl.generation_fleet_health.enabled=true \
    ++async_rl.generation_router.enabled=true \
    ++async_rl.generation_fleet_health.min_healthy_shards=1 \
    "$@" \
    > "$RUN_LOG" 2>&1 &
TRAIN_PID=$!

echo "[failover] waiting for ${KILL_AFTER_STEPS} training steps (up to ${STEADY_STATE_WAIT_S}s)..."
STEPS=0
for _ in $(seq 1 "$STEADY_STATE_WAIT_S"); do
    kill -0 $TRAIN_PID 2>/dev/null || { echo "[failover] FAIL: job died before the kill"; tail -60 "$RUN_LOG"; exit 1; }
    STEPS=$(grep -cE "train step [0-9]+/" "$RUN_LOG" 2>/dev/null || true)
    (( STEPS >= KILL_AFTER_STEPS )) && break
    sleep 1
done

if (( STEPS < KILL_AFTER_STEPS )); then
    # Distinguished from a failover failure on purpose: reaching steady state is a
    # PRECONDITION of this test, not the thing it measures. Reporting it as "the run
    # died after losing a shard" is what made job 6251221's result unreadable.
    echo "[failover] FAIL: only $STEPS training step(s) in ${STEADY_STATE_WAIT_S}s;"
    echo "[failover] the run never reached steady state, so nothing about failover was tested."
    tail -60 "$RUN_LOG"
    exit 1
fi
echo "[failover] $STEPS training steps done -- rollouts are flowing through the router."


# Discovered AFTER the run is confirmed training, not before. Each attempt is a full
# ray.init round trip, and running this first put minutes of latency ahead of the kill --
# enough that job 6251951's entire 40-step run finished before discovery returned. By
# here the actors certainly exist, so this is one query rather than a polling loop.
echo "[failover] discovering generation actors (up to ${ACTOR_WAIT_S}s)..."
ACTORS=()
for _ in $(seq 1 $((ACTOR_WAIT_S / 3))); do
    kill -0 $TRAIN_PID 2>/dev/null || { echo "[failover] FAIL: job died before any actor appeared"; tail -60 "$RUN_LOG"; exit 1; }
    : > "$EXP_DIR/pids.tmp"
    timeout "$ACTOR_QUERY_TIMEOUT_S" uv run --no-sync python \
        "$SCRIPT_DIR/_find_generation_actors.py" >"$EXP_DIR/pids.tmp" 2>>"$EXP_DIR/discover.log" || true
    mapfile -t ACTORS < <(sort -n < "$EXP_DIR/pids.tmp")
    (( ${#ACTORS[@]} >= 2 )) && break
    sleep 3
done

if (( ${#ACTORS[@]} < 2 )); then
    echo "[failover] FAIL: expected >= 2 generation actors for dp_size=2, found ${#ACTORS[@]}"
    tail -40 "$EXP_DIR/discover.log" 2>/dev/null || true
    tail -60 "$RUN_LOG"
    exit 1
fi
echo "[failover] generation actors: ${ACTORS[*]}"

# Guard the other end of the window, re-read HERE rather than at the step gate above,
# because discovery sits between the two and is not free.
#
# Job 6251951 crossed the step threshold with the run ALREADY COMPLETE -- 40/40 -- so the
# kill hit a shutting-down job and proved nothing. Two causes, both addressed: discovery
# ran first and cost more wall-clock than the entire run (its first ray.init hit a stale
# GCS address and spent two minutes timing out), and a 40-step run at ~4s/step is under
# three minutes end to end. Discovery now runs after the step gate, and max_num_steps is
# large enough that the kill lands with real work left.
STEPS=$(grep -cE "train step [0-9]+/" "$RUN_LOG" 2>/dev/null || true)
TOTAL_STEPS=$(grep -oE "train step [0-9]+/[0-9]+" "$RUN_LOG" | tail -1 | sed -E "s|.*/||")
if [[ -n "$TOTAL_STEPS" ]] && (( STEPS > TOTAL_STEPS - 5 )); then
    echo "[failover] FAIL: the run is at step $STEPS of $TOTAL_STEPS at kill time --"
    echo "[failover] too little left for surviving on N-1 shards to mean anything."
    echo "[failover] Raise grpo.max_num_steps or lower KILL_AFTER_STEPS."
    exit 1
fi

VICTIM=${ACTORS[0]}
echo "[failover] killing generation actor $VICTIM"
kill -9 "$VICTIM" 2>/dev/null || true

echo "[failover] waiting for the run to finish on the surviving shard..."
set +e
wait $TRAIN_PID
RUN_RC=$?
set -e

if (( RUN_RC != 0 )); then
    echo "[failover] FAIL: the run died after losing one of two shards (rc=$RUN_RC)."
    echo "[failover] Surviving on N-1 shards is the property under test."
    tail -80 "$RUN_LOG"
    exit 1
fi

# The run completing is necessary but not sufficient: it must have completed *because*
# the shard was quarantined and traffic moved, not because the kill missed.
grep -qE "gen_fleet: shard [0-9]+ (healthy|suspect) -> dead" "$RUN_LOG" || {
    echo "[failover] FAIL: no shard was quarantined -- the kill did not land where it matters."
    grep -iE "gen_fleet|router" "$RUN_LOG" | tail -30
    exit 1
}
echo "[failover] shard quarantined, run completed on the survivor."

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"

# gen_kl_error is the assertion that earns its keep: it compares vLLM's logprobs against
# the trainer's recomputation, so a router that corrupted or truncated a response during
# the failover would blow it up. A run that merely completes would not prove the payload
# survived the hop.
uv run tests/check_metrics.py "$JSON_METRICS" \
    'median(data["train/gen_kl_error"]) < 1.3' \
    'max(data["train/reward"]) > 0'
