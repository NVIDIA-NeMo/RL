#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Compaction arm paired with smoke_off_32turns_256k.sh: identical tasks, batch,
# turn budget, 256k policy window and prompt order; the only difference is that
# OpenCode compacts (compaction.auto=true) and NeMo-RL reconstructs one training
# row per context segment. OpenCode is told the policy's real window
# (AGENTIC_VBENCH_CONTEXT_TOKENS) so it compacts only when the context fills;
# set a smaller value to force compaction inside shorter episodes.
#   AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke-2 \
#     bash scripts/agentic_vbench/smoke_cc_32turns_256k.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CONTEXT_COMPACTION=true
export AGENTIC_VBENCH_COMPACTION_AUTO=true
export AGENTIC_VBENCH_COMPACTION_PRUNE=false
export BASE_NAME=${BASE_NAME:-agentic_vbench_sc_cc_32t_256k_step120_$(date -u +%Y%m%d-%H%M%S)-$$}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-$BASE_NAME}
exec bash "$script_dir/smoke_off_32turns_256k.sh" "$@"
