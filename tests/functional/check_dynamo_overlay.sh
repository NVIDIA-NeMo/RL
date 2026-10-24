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

# What a native functional test also checks when it runs as the Dynamo overlay
# (++policy.generation.dynamo=true): every engine registered with the managed
# frontend, the frontend routed rollouts to them, the KV router applied the
# engines' KV-cache events (when they publish any: KV routing with prefix
# caching), and shutdown stopped its services. The test's own metric checks hold
# as they do natively.
#
#   bash check_dynamo_overlay.sh <run log>

set -euo pipefail

run_log=$1

grep -E "\[Dynamo\] frontend .* routes to [0-9]+ engine" "${run_log}"
routed=$(grep -c 'route request completed request.outcome="success"' "${run_log}" || true)
echo "Requests the Dynamo frontend routed: ${routed}"
if [[ "${routed}" -eq 0 ]]; then
  echo "No request went through the Dynamo frontend" >&2
  exit 1
fi

# vLLM's KV-event wire format is the one coupling between NeMo RL's vLLM and
# Dynamo's router; a format change shows up here as no applied events.
if grep -q "\[Dynamo\] engine publishes KV events: True" "${run_log}"; then
  grep -E "\[Dynamo\] router applied KV events" "${run_log}"
  applied=$(grep -oE '\[Dynamo\] router applied KV events: .*"ok": [0-9.]+' "${run_log}" \
    | grep -oE '[0-9.]+$' | tail -1)
  if [[ -z "${applied}" || "${applied%.*}" -eq 0 ]]; then
    echo "The KV router applied none of the engines' KV-cache events" >&2
    exit 1
  fi
fi

dynamo_bin_dir=${NEMO_RL_DYNAMO_BIN_DIR:-/opt/dynamo/bin}
if pgrep -f "[d]ynamo.frontend|${dynamo_bin_dir}/[e]tcd|${dynamo_bin_dir}/[n]ats-server"; then
  echo "Managed Dynamo services remain after shutdown" >&2
  exit 1
fi
