#!/bin/bash
# Lightweight e2e for grpo_sync.py — TQ pipeline with the mooncake_cpu
# backend. Same shape as tests/functional/grpo.sh (Qwen3-0.6B, 2 GPUs,
# 2 steps); exercises the real Mooncake transfer engine on the CPU path.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

# mooncake_cpu is RDMA-only, so this needs an mlx5 device libibverbs can open
# (either fabric). Skip rather than fail on hosts that have none. Mirrors
# rdma_device() in nemo_rl/data_plane/adapters/transfer_queue.py.
if [[ -z "${MC_MOONCAKE_DEVICE:-}" ]] &&
   ! { compgen -G "/dev/infiniband/uverbs*" >/dev/null &&
       compgen -G "/sys/class/infiniband/mlx5_*/ports/1/link_layer" >/dev/null; }; then
    echo "[SKIP] no usable mlx5 RDMA device; mooncake_cpu requires RDMA." \
         "Set MC_MOONCAKE_DEVICE=<dev> to override."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# TEMPORARY DIAGNOSTIC — REMOVE BEFORE MERGE.
#
# Why: the gb200 runners fail every transfer with a QP/retry error, and 100% of
# the failing pairs are cross-rail while zero are same-rail. Cross-rail RoCE was
# measured to be unroutable (same-rail 16.25 GB/s vs QP-to-RTR timeout), so the
# open question is only *why mooncake draws a cross-rail pair at all*.
#
# Mooncake builds `preferred_hca` from `hca.numa_node == node_id` of the buffer,
# then picks at random among the eligible rails. So a cross-rail draw needs
# either >1 RoCE rail eligible for one NUMA node, or none (empty preferred_hca
# falls back to every rail). On a 1 RoCE-rail-per-NUMA-node host the draw is
# unambiguous and everything works. That mapping is what this dumps.
# ─────────────────────────────────────────────────────────────────────────────
echo "=== [tmp diag] RDMA rails ==="
for dev in /sys/class/infiniband/*; do
    [[ -e "$dev" ]] || continue
    port=$dev/ports/1
    # gid 3 is the RoCEv2 GID; its low 32 bits are the IPv4 address, so two rails
    # on different subnets here means a cross-rail pair has no route.
    echo "$(basename $dev)" \
         "numa=$(cat $dev/device/numa_node 2>/dev/null || echo '?')" \
         "link=$(cat $port/link_layer 2>/dev/null || echo '?')" \
         "state=$(cat $port/state 2>/dev/null || echo '?')" \
         "lid=$(cat $port/lid 2>/dev/null || echo '?')" \
         "gid3=$(cat $port/gids/3 2>/dev/null || echo '?')"
done

echo "=== [tmp diag] RoCE rails eligible per memory-bearing NUMA node ==="
# This is the decisive table: a host buffer can only live on a node with memory,
# and mooncake will only prefer rails whose numa_node matches that node.
for node in /sys/devices/system/node/node*; do
    n=${node##*/node}
    mem=$(grep -oP 'MemTotal:\s+\K[0-9]+' "$node/meminfo" 2>/dev/null || echo 0)
    [[ "${mem:-0}" -gt 0 ]] || continue
    rails=""
    for dev in /sys/class/infiniband/*; do
        [[ -e "$dev/device/numa_node" ]] || continue
        [[ "$(cat "$dev/ports/1/link_layer" 2>/dev/null)" == "Ethernet" ]] || continue
        [[ "$(cat "$dev/device/numa_node")" == "$n" ]] && rails="$rails $(basename "$dev")"
    done
    count=$(echo $rails | wc -w)
    case $count in
        0) verdict="NO LOCAL RAIL -> preferred_hca empty -> mooncake picks among ALL rails at random" ;;
        1) verdict="unambiguous -> same-rail by construction" ;;
        *) verdict="AMBIGUOUS -> mooncake picks among these at random -> cross-rail draws expected" ;;
    esac
    echo "node$n mem=${mem}kB roce=[${rails# }] n=$count $verdict"
done

echo "=== [tmp diag] device_name the adapter will pass to mooncake ==="
# Mirrors rdma_devices() in nemo_rl/data_plane/adapters/transfer_queue.py: gate on a
# verbs node, prefer every IB rail, fall back to every RoCE rail. Done in bash rather
# than `uv run` so this costs nothing and cannot drag a dependency sync into the test.
# Cross-check against the "RDMA device: <name>" lines mooncake logs during the run —
# if they disagree, mooncake ignored the list.
ib="" roce=""
if compgen -G "/dev/infiniband/uverbs*" >/dev/null; then
    for dev in /sys/class/infiniband/*; do
        case "$(cat "$dev/ports/1/link_layer" 2>/dev/null)" in
            InfiniBand) ib="$ib,$(basename "$dev")" ;;
            Ethernet) roce="$roce,$(basename "$dev")" ;;
        esac
    done
fi
selected=${ib:-$roce}
echo "MC_MOONCAKE_DEVICE=${MC_MOONCAKE_DEVICE:-(unset)} -> device_name=\"${selected#,}\""

echo "=== [tmp diag] MC_* env in effect / memlock ==="
mc_env=$(env | grep '^MC_' | sort || true)
echo "${mc_env:-(no MC_* set)}"
echo "memlock soft=$(ulimit -Sl) hard=$(ulimit -Hl)"
echo "=== [tmp diag] end ==="

# TEMPORARY, pairs with the block above. TransferQueue starts mooncake_master
# (which also hosts the HTTP metadata server on :50050) with its stdout and
# stderr redirected to /tmp/mooncake_master.log, so none of it reaches the CI
# log. When the store fails with rc=-200 / "Couldn't connect to <ip>:50050",
# that file holds the only account of why -- TQ waits a flat 3s and checks the
# process is alive, never that the port accepts connections. Dump it on exit so
# it survives the failing run.
dump_mooncake_master_log() {
    echo "=== [tmp diag] /tmp/mooncake_master.log ==="
    if [[ -f /tmp/mooncake_master.log ]]; then
        tail -60 /tmp/mooncake_master.log
    else
        echo "(absent -- mooncake_master was never started)"
    fi
    echo "=== [tmp diag] listeners on 50050/50051 ==="
    # ss and netstat are both often missing from the image; /proc always works.
    # Ports are hex in /proc/net/tcp: 50050=C382, 50051=C383.
    grep -iE ':(C382|C383) ' /proc/net/tcp 2>/dev/null || echo "(no listener on 50050/50051)"
}
trap dump_mooncake_master_log EXIT

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

cd $PROJECT_ROOT
uv run coverage run -a --data-file=$PROJECT_ROOT/tests/.coverage --source=$PROJECT_ROOT/nemo_rl \
    $PROJECT_ROOT/examples/run_grpo.py \
    policy.model_name=Qwen/Qwen3-0.6B \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.train_micro_batch_size=1 \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=2 \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    data_plane.enabled=true \
    data_plane.impl=transfer_queue \
    data_plane.backend=mooncake_cpu \
    data_plane.mooncake_cpu.global_segment_size=4294967296 \
    data_plane.mooncake_cpu.local_buffer_size=1073741824 \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/gen_kl_error"]) < 0.002' \
    'min(data["train/probs_ratio_clamped_min"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_min"]) < 1.21' \
    'min(data["train/probs_ratio_clamped_max"]) > 0.79' \
    'max(data["train/probs_ratio_clamped_max"]) < 1.21'
