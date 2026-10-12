#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit the rootless Podman/Docker-API backend (serve_rootless_backend.sh) on a
# CPU node and print how to wait for its client.env.
#   AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke bash scripts/agentic_vbench/submit_backend.sh
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# serve_rootless_backend.sh keeps its original AVB_BACKEND_DIR name; the agent reads AGENTIC_VBENCH_BACKEND_DIR.
export AVB_BACKEND_DIR=${AGENTIC_VBENCH_BACKEND_DIR:-${AVB_BACKEND_DIR:?Set AGENTIC_VBENCH_BACKEND_DIR to a new shared directory for backend certificates and client.env}}
mkdir -p "$AVB_BACKEND_DIR"
if [[ -s $AVB_BACKEND_DIR/client.env ]]; then
  echo "A backend already published $AVB_BACKEND_DIR/client.env; use a new AVB_BACKEND_DIR for a second backend." >&2
  exit 1
fi
# The idle-GPU reaper also inspects CPU jobs' comments; declare the same exemption
# the training launcher uses so a quiet backend between waves is not reaped.
sbatch_comment="${SBATCH_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"120\",\"reason\":\"data_loading\",\"description\":\"Task-container backend for agentic rollouts; idle between rollout waves\"}}}"
# QoS: cpu-normal (default) allows 2 nodes/user for 24 h; cpu-long 1 node for 7 d;
# cpu-interactive 2 nodes for 24 h at higher priority. Set BACKEND_QOS when the
# default pool is full (QOSMaxNodePerUserLimit).
job=$(sbatch --parsable \
  --comment="${sbatch_comment}" \
  ${BACKEND_QOS:+--qos="$BACKEND_QOS"} \
  --account="${SLURM_ACCOUNT:-nemotron_n4_omni}" --partition="${BACKEND_PARTITION:-cpu}" \
  --nodes=1 --ntasks=1 --cpus-per-task="${BACKEND_CPUS:-120}" --mem="${BACKEND_MEM:-700G}" \
  --time="${BACKEND_TIME:-24:00:00}" --job-name="${BACKEND_JOB_NAME:-avb-cc-backend}" \
  --output="$AVB_BACKEND_DIR/slurm-%j.log" --export=ALL,AVB_BACKEND_DIR="$AVB_BACKEND_DIR" \
  "$script_dir/serve_rootless_backend.sh")
echo "backend_job=$job"
echo "Wait for: test -s $AVB_BACKEND_DIR/client.env   (then run_sc.sh or smoke_cc_step120.sh will read it)"
echo "Stop with: scancel $job"
