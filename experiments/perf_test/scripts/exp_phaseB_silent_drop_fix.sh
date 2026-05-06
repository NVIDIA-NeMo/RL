#!/bin/bash
# Phase B — re-run v07_07 / v07_08 / v07_09 paired benchmarks after the
# silent-drop fix (commit 1f788697). Prior measurements were baselines under
# different yaml labels because the four knobs (overlap_p2p_comm,
# defer_embedding_wgrad_compute, tp_comm_atomic_ag, tp_comm_atomic_rs) were
# not enumerated in setup.py.
#
# Same-batch pairs control within-batch noise (~±2%). 12 jobs total.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "${PARTITION:-batch}"
export_cluster_config

PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

declare -a JOBS=(
    # Qwen3 32B (TP=2, PP=4): all three Tier 2 knobs apply.
    "perf_test/v07_tier1/qwen3_32b/v07_00_baseline          4 qwen32b-phaseB-base"
    "perf_test/v07_tier1/qwen3_32b/v07_07_overlap_p2p       4 qwen32b-phaseB-p2p"
    "perf_test/v07_tier1/qwen3_32b/v07_08_defer_embed_wgrad 4 qwen32b-phaseB-deferembed"
    "perf_test/v07_tier1/qwen3_32b/v07_09_tp_atomic         4 qwen32b-phaseB-tpatomic"

    # Qwen3 235B (TP=2, PP=4, EP=16): same three knobs.
    "perf_test/v07_tier1/qwen3_235b/v07_00_baseline          16 qwen235b-phaseB-base"
    "perf_test/v07_tier1/qwen3_235b/v07_07_overlap_p2p       16 qwen235b-phaseB-p2p"
    "perf_test/v07_tier1/qwen3_235b/v07_08_defer_embed_wgrad 16 qwen235b-phaseB-deferembed"
    "perf_test/v07_tier1/qwen3_235b/v07_09_tp_atomic         16 qwen235b-phaseB-tpatomic"
)

FILTER="${1:-}"

for job in "${JOBS[@]}"; do
    read -r config num_nodes suffix <<<"$job"
    if [[ -n "$FILTER" ]] && [[ "$config" != *"$FILTER"* ]] && [[ "$suffix" != *"$FILTER"* ]]; then
        continue
    fi
    submit_variant "$PROJECT_ROOT" "$config" "$num_nodes" "nrl-phaseB-${suffix}"
done

echo ""
echo "[MONITOR] squeue -u \$USER -o '%.18i %.30j %.8T %.10M %R'"
