#!/bin/bash
# Launch v0.7 Tier 1 perf_test variants.
#
# Usage:
#   bash exp_v07_tier1.sh                # submit everything
#   bash exp_v07_tier1.sh llama          # only llama_8b cells
#   bash exp_v07_tier1.sh v07_00_baseline   # only baselines across all models
#
# Default cluster: OCI-HSG GB200, account coreai_dlalgo_llm, partition batch.
# Override per-cluster with env vars (see cluster_config.sh): ACCOUNT=, PARTITION=, GRES_FLAG=.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "${PARTITION:-batch}"
export_cluster_config

PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

# (config_rel, num_nodes, job_suffix)
declare -a JOBS=(
    # Node counts match the n4g (GB200, 4 GPUs/node) recipes.
    # Llama 3.1 8B 2n4g (TP=1, PP=1) — only G6 fusion applies.
    "perf_test/v07_tier1/llama_8b/v07_00_baseline           2 llama8b-00-base"
    "perf_test/v07_tier1/llama_8b/v07_01_g6_fusion          2 llama8b-01-g6"
    "perf_test/v07_tier1/llama_8b/v07_06_grad_accum_fusion  2 llama8b-06-gradacc"

    # Qwen3 30B-A3B 4n4g (TP=1, PP=1, EP=16) — G6 + G3 (MoE A2A) apply.
    "perf_test/v07_tier1/qwen3_30ba3b/v07_00_baseline           4 qwen30ba3b-00-base"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_01_g6_fusion          4 qwen30ba3b-01-g6"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_04_g3_moe_a2a         4 qwen30ba3b-04-g3"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_05_stack              4 qwen30ba3b-05-stack"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_06_grad_accum_fusion  4 qwen30ba3b-06-gradacc"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_07_moe_shared_expert  4 qwen30ba3b-07-shexp"

    # Qwen3 32B 4n4g (TP=2, PP=4) — G1, G5, G6, stack apply (no MoE).
    "perf_test/v07_tier1/qwen3_32b/v07_00_baseline           4 qwen32b-00-base"
    "perf_test/v07_tier1/qwen3_32b/v07_01_g6_fusion          4 qwen32b-01-g6"
    "perf_test/v07_tier1/qwen3_32b/v07_02_g1_tpoverlap       4 qwen32b-02-g1"
    "perf_test/v07_tier1/qwen3_32b/v07_03_g5_delaywgrad      4 qwen32b-03-g5"
    "perf_test/v07_tier1/qwen3_32b/v07_05_stack              4 qwen32b-05-stack"
    "perf_test/v07_tier1/qwen3_32b/v07_06_grad_accum_fusion  4 qwen32b-06-gradacc"
    "perf_test/v07_tier1/qwen3_32b/v07_07_overlap_p2p        4 qwen32b-07-p2p"
    "perf_test/v07_tier1/qwen3_32b/v07_08_defer_embed_wgrad  4 qwen32b-08-deferembed"
    "perf_test/v07_tier1/qwen3_32b/v07_09_tp_atomic          4 qwen32b-09-tpatomic"

    # Qwen3 235B 16n4g (TP=2, PP=4, EP=16) — full matrix.
    "perf_test/v07_tier1/qwen3_235b/v07_00_baseline           16 qwen235b-00-base"
    "perf_test/v07_tier1/qwen3_235b/v07_01_g6_fusion          16 qwen235b-01-g6"
    "perf_test/v07_tier1/qwen3_235b/v07_02_g1_tpoverlap       16 qwen235b-02-g1"
    "perf_test/v07_tier1/qwen3_235b/v07_03_g5_delaywgrad      16 qwen235b-03-g5"
    "perf_test/v07_tier1/qwen3_235b/v07_04_g3_moe_a2a         16 qwen235b-04-g3"
    "perf_test/v07_tier1/qwen3_235b/v07_05_stack              16 qwen235b-05-stack"
    "perf_test/v07_tier1/qwen3_235b/v07_06_grad_accum_fusion  16 qwen235b-06-gradacc"
    "perf_test/v07_tier1/qwen3_235b/v07_07_overlap_p2p        16 qwen235b-07-p2p"
    "perf_test/v07_tier1/qwen3_235b/v07_08_defer_embed_wgrad  16 qwen235b-08-deferembed"
    "perf_test/v07_tier1/qwen3_235b/v07_09_tp_atomic          16 qwen235b-09-tpatomic"
    "perf_test/v07_tier1/qwen3_235b/v07_10_moe_shared_expert  16 qwen235b-10-shexp"
)

FILTER="${1:-}"

for job in "${JOBS[@]}"; do
    read -r config num_nodes suffix <<<"$job"
    if [[ -n "$FILTER" ]] && [[ "$config" != *"$FILTER"* ]] && [[ "$suffix" != *"$FILTER"* ]]; then
        continue
    fi
    submit_variant "$PROJECT_ROOT" "$config" "$num_nodes" "nrl-v07-${suffix}"
done

echo ""
echo "[MONITOR] squeue -u \$USER -o '%.18i %.30j %.8T %.10M %R'"
