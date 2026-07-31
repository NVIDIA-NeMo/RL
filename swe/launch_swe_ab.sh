#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
# SWE token-capture A/B arm launcher — see swe/SWE_RUN.md.
#
# Picks the arm, derives its config, names the run, forces the venv posture,
# then delegates to YOUR site wrapper (account/container/secrets live there).
# All launcher knobs (TP, PPS, MAX_NUM_STEPS, DRY_RUN=1, ...) pass through.
#
# Required:
#   ARM=legacy|capture
#   SITE_WRAPPER=/path/to/your grpo_swe_tests.sh-style wrapper
# Optional:
#   BYTES=1        enable per-hop HTTP byte counters (JSON under the
#                  checkpoint dir; aggregate with swe/aggregate_perf.py)
#   BASE_CONFIG    site yaml to derive from (default: the wrapper's default,
#                  discovered via CONFIG_FILE after the wrapper sources it —
#                  set explicitly if your wrapper computes CONFIG_FILE late)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${ARM:?set ARM=legacy|capture}"
SITE_WRAPPER="${SITE_WRAPPER:?set SITE_WRAPPER=/path/to/your site wrapper}"
[[ "${ARM}" == "legacy" || "${ARM}" == "capture" ]] || { echo "ARM must be legacy|capture" >&2; exit 1; }

# ---- Derive the arm's config from the site yaml -----------------------------
# Default BASE_CONFIG: the yaml sitting next to the site wrapper (the
# grpo_swe_tests.sh convention). Override for other layouts.
BASE_CONFIG="${BASE_CONFIG:-$(dirname "${SITE_WRAPPER}")/grpo_qwen3_30b_async_swe.yaml}"
[ -f "${BASE_CONFIG}" ] || { echo "BASE_CONFIG not found: ${BASE_CONFIG}" >&2; exit 1; }
DERIVED_DIR="${DERIVED_DIR:-$(dirname "${BASE_CONFIG}")/derived_configs}"
mkdir -p "${DERIVED_DIR}"
export CONFIG_FILE="${DERIVED_DIR}/grpo_swe_ab_${ARM}.yaml"
python3 "${SCRIPT_DIR}/make_capture_config.py" "${BASE_CONFIG}" "${CONFIG_FILE}" "${ARM}"

# ---- Run naming (W&B separates the arms) ------------------------------------
export EXP_SUFFIX="${EXP_SUFFIX:-swe-ab-${ARM}-$(date +%m%d%H%M)}"

# ---- Venv posture: identical on both arms (see SWE_RUN.md § 3) --------------
# Baked /opt/ray_venvs predate this branch's lock; the capture arm needs the
# unbaked VLLM_GYM worker venv. Forcing rebuild on both arms keeps setup cost
# out of the A/B.
export NRL_FORCE_REBUILD_VENVS=true
# Gym venvs: use the image-baked /opt/gym_venvs (present on EVERY node).
# A node-local GYM_VENV_DIR=/tmp/... does NOT work multi-node: the venv build
# runs only on the NemoGym actor's node while Gym spawns servers cluster-wide
# (learned from job 14542017). The fork's only dep-floor change vs the baked
# venvs is the aiohttp CVE bump (verified: swe_agents/vllm_model requirements
# unchanged), and editable installs serve the fork's *code* either way.
export GYM_VENV_DIR="${GYM_VENV_DIR:-/opt/gym_venvs}"

# ---- Optional per-hop HTTP byte accounting ----------------------------------
if [ "${BYTES:-0}" = "1" ]; then
    BYTES_DIR="${BYTES_DIR:-${CHECKPOINT_ROOT:-$(dirname "${SITE_WRAPPER}")/../..}/http_bytes/${EXP_SUFFIX}}"
    mkdir -p "${BYTES_DIR}"
    export NG_HTTP_BYTES_DIR="${BYTES_DIR}"
    export NRL_HTTP_BYTES_DIR="${BYTES_DIR}"
    echo "[launch_swe_ab] byte counters on -> ${BYTES_DIR}"
fi

echo "[launch_swe_ab] arm=${ARM} config=${CONFIG_FILE} exp=${EXP_SUFFIX}"
exec bash "${SITE_WRAPPER}" "$@"
