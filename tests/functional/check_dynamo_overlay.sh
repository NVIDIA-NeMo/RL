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
# frontend, the frontend routed rollouts to them, the engines published KV-cache
# events and the KV router applied them, and shutdown stopped its services. The
# test's own metric checks hold as they do natively. The KV check needs the KV
# router with vLLM prefix caching (the default on sm>=80); an overlay that turns
# either off cannot use this check as is.
#
#   bash check_dynamo_overlay.sh <run log>

set -euo pipefail

run_log=$1

fail() {
  echo "$1" >&2
  exit 1
}

grep -E "\[Dynamo\] frontend .* routes to [0-9]+ engine" "${run_log}" \
  || fail "No engine registered with the Dynamo frontend"
routed=$(grep -c 'route request completed request.outcome="success"' "${run_log}" || true)
echo "Requests the Dynamo frontend routed: ${routed}"
[[ "${routed}" -gt 0 ]] || fail "No request went through the Dynamo frontend"

# vLLM's KV-event wire format is the one coupling between NeMo RL's vLLM and
# Dynamo's router; a format change shows up here as no applied events.
grep -q "\[Dynamo\] engine publishes KV events: True" "${run_log}" \
  || fail "No engine published KV-cache events for the KV router"
applied=$(grep -oE '\[Dynamo\] router applied KV events: .*"ok": [0-9.]+' "${run_log}" \
  | grep -oE '[0-9.]+$' | tail -1 || true)
echo "KV events the router applied: ${applied:-0}"
[[ "${applied:-0}" =~ ^[0-9]+ && "${applied%%.*}" -gt 0 ]] \
  || fail "The KV router applied none of the engines' KV-cache events"

dynamo_bin_dir=${NEMO_RL_DYNAMO_BIN_DIR:-/opt/dynamo/bin}
if pgrep -f "[d]ynamo.frontend|${dynamo_bin_dir}/[e]tcd|${dynamo_bin_dir}/[n]ats-server"; then
  fail "Managed Dynamo services remain after shutdown"
fi
