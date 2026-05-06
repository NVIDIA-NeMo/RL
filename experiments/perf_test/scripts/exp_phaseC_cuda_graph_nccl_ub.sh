#!/bin/bash
# Phase C — newly-wired knobs that NeMo-RL had no interface for prior to
# commit 1f788697. CUDA graph + cross-entropy TE fusion + NCCL UserBuffers.
# Same-batch baselines pair each variant. ~14 jobs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_config.sh"
setup_cluster_config "${PARTITION:-batch}"
export_cluster_config

PROJECT_ROOT="${PROJECT_ROOT:-${BASE}/RL-v07-tier1-bench}"

declare -a JOBS=(
    # Llama 8B 2n4g (TP=1, PP=1) — small + dense, CUDA graph captures cleanly.
    "perf_test/v07_tier1/llama_8b/v07_00_baseline             2 llama8b-phaseC-base"
    "perf_test/v07_tier1/llama_8b/v07_12_external_cuda_graph  2 llama8b-phaseC-extgraph"
    "perf_test/v07_tier1/llama_8b/v07_13_ce_te_fusion         2 llama8b-phaseC-cete"

    # Qwen3 30B-A3B 4n4g (MoE) — CUDA graph w/ MoE routing has known issues; measure anyway.
    "perf_test/v07_tier1/qwen3_30ba3b/v07_00_baseline            4 qwen30ba3b-phaseC-base"
    "perf_test/v07_tier1/qwen3_30ba3b/v07_12_external_cuda_graph 4 qwen30ba3b-phaseC-extgraph"

    # Qwen3 32B 4n4g (TP=2, PP=4) — big test for NCCL UB on multi-node DP.
    "perf_test/v07_tier1/qwen3_32b/v07_00_baseline             4 qwen32b-phaseC-base"
    "perf_test/v07_tier1/qwen3_32b/v07_12_external_cuda_graph  4 qwen32b-phaseC-extgraph"
    "perf_test/v07_tier1/qwen3_32b/v07_13_ce_te_fusion         4 qwen32b-phaseC-cete"
    "perf_test/v07_tier1/qwen3_32b/v07_14_nccl_ub              4 qwen32b-phaseC-ncclub"
)

FILTER="${1:-}"

for job in "${JOBS[@]}"; do
    read -r config num_nodes suffix <<<"$job"
    if [[ -n "$FILTER" ]] && [[ "$config" != *"$FILTER"* ]] && [[ "$suffix" != *"$FILTER"* ]]; then
        continue
    fi
    submit_variant "$PROJECT_ROOT" "$config" "$num_nodes" "nrl-phaseC-${suffix}"
done

echo ""
echo "[MONITOR] squeue -u \$USER -o '%.18i %.30j %.8T %.10M %R'"
