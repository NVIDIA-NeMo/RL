#!/bin/bash
# Launch DeepSeek-V3 HybridEP + CudaGraph Sweep
#
# Jobs:
#   1. Baseline (AllToAll, no HybridEP)
#   2. HybridEP SM=16 (no CG)
#   3. HybridEP SM=16 + CG [attn,moe_router]
#
# All use: 32 nodes, EP=16, TP=1, PP=8, Gen TP=16, 5 steps

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  DeepSeek-V3 HybridEP + CG Sweep"
echo "============================================================"
echo "  Model:   DeepSeek-V3 (671B MoE, BF16)"
echo "  Config:  32 nodes, EP=16, TP=1, PP=8, Gen TP=16"
echo "  Steps:   5"
echo "============================================================"
echo ""

JOB_COUNT=0

echo "--- Job 1: Baseline (AllToAll, no HybridEP) ---"
bash "${SCRIPT_DIR}/exp_deepseekv3_baseline.sh"
JOB_COUNT=$((JOB_COUNT + 1))
echo ""

echo "--- Job 2: HybridEP SM=16 ---"
bash "${SCRIPT_DIR}/exp_deepseekv3_hybridep_sweep.sh" 16
JOB_COUNT=$((JOB_COUNT + 1))
echo ""

echo "--- Job 3: HybridEP SM=16 + CG [attn,moe_router] ---"
bash "${SCRIPT_DIR}/exp_deepseekv3_hybridep_sweep.sh" 16 cg
JOB_COUNT=$((JOB_COUNT + 1))
echo ""

echo "============================================================"
echo "  All ${JOB_COUNT} jobs submitted!"
echo "  W&B project: RL_GB200_deepseekv3_sweep"
echo "  Run 'squeue -u \$USER' to check status."
echo "============================================================"
