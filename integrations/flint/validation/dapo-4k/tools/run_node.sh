#!/usr/bin/env bash
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
set -euo pipefail
set +x
umask 077
cd /flint-work/RL
flint_host=$(hostname -s)
flint_rank=${SLURM_PROCID:?}
flint_attempt="/flint-run/evidence/jobs/${FLINT_ATTEMPT_ID:?}"
flint_runtime="/flint-run/evidence/runtime-$flint_host/$FLINT_ATTEMPT_ID"
mkdir -p "$flint_runtime"
export PYTHONPATH=/flint-work/RL:/flint-work/tools:/flint-work/sglang-plugin/src:/flint-run/tools
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/flint-run/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_DISABLE_GIT=true
export WANDB_DIR=/flint-run/logs/dapo50 WANDB_CACHE_DIR=/flint-run/cache/wandb
export WANDB_CONFIG_DIR=/flint-run/cache/wandb-config WANDB_DATA_DIR=/flint-run/cache/wandb-data
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"
export RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 NCCL_CUMEM_ENABLE=0
export SGLANG_EXTERNAL_MODEL_PACKAGE=flint_sglang FLASHINFER_USE_CUDA_NORM=1
export FLINT_TRITON_CACHE_BASE=/flint-run/cache/triton
export XDG_CACHE_HOME=/flint-run/cache
export TMPDIR="/tmp/flint-dapo50-$FLINT_ATTEMPT_ID"
export FLINT_METRICS_PATH="$flint_attempt/training.metrics.jsonl"
export FLINT_REFIT_AUDIT_DIR="$flint_attempt/refit"
export FLINT_REFIT_AUDIT_HEAD_ROW_START=1
unset NRL_IGNORE_VERSION_MISMATCH FLINT_SYNTHETIC_SMOKE
mkdir -p "$TMPDIR" "$HF_HOME" "$FLINT_REFIT_AUDIT_DIR"
flint_ray_pid=
flint_driver_pid=
flint_monitor_pid=
flint_report_pid=
flint_fault_watch_pid=
flint_cleanup() {
  local flint_status=$?
  trap - EXIT TERM INT HUP
  if [[ "$flint_rank" == 0 ]]; then
    printf '%s\n' "$flint_status" >"$flint_attempt/.ended.tmp"
    mv "$flint_attempt/.ended.tmp" "$flint_attempt/ended"
  fi
  for flint_child in "$flint_fault_watch_pid" "$flint_report_pid" "$flint_monitor_pid" "$flint_driver_pid" "$flint_ray_pid"; do
    if [[ -n "$flint_child" ]] && kill -0 "$flint_child" 2>/dev/null; then
      kill -TERM "$flint_child" 2>/dev/null || true
    fi
  done
  # Ray and actors belong to this Slurm container step. Returning from every
  # node lets Slurm reap this step's cgroup without touching another job's Ray.
  for flint_child in "$flint_fault_watch_pid" "$flint_report_pid" "$flint_monitor_pid" "$flint_driver_pid" "$flint_ray_pid"; do
    if [[ -n "$flint_child" ]]; then wait "$flint_child" 2>/dev/null || true; fi
  done
  printf '%s\n' "$flint_status" >"$flint_runtime/node.exit-code"
  exit "$flint_status"
}
trap flint_cleanup EXIT
trap 'exit 143' TERM HUP
trap 'exit 130' INT

/opt/flint/bin/python-driver /flint-run/tools/apply_runtime_overlays.py \
  --output "$flint_runtime/runtime-overlays.json"
/opt/flint/bin/python-sglang - "$flint_runtime/router-policy-check.json" <<'PY'
import dataclasses, hashlib, inspect, json, sys
from pathlib import Path
from sglang_router.launch_router import RouterArgs
args = RouterArgs()
args.router_policy = "round_robin"
assert args.policy == "round_robin", "Managed-router policy did not reach the actual router field"
assert "router_policy" not in vars(args), "Unexpected router constructor keyword"
assert dataclasses.asdict(args)["policy"] == "round_robin"
source = Path(inspect.getfile(RouterArgs))
assert hashlib.sha256(source.read_bytes()).hexdigest() == "09cd8984a1f693cc2efe40b963befd6d5bc578e972272e994a046f183807f3e8"
Path(sys.argv[1]).write_text(json.dumps({"status": "pass", "requested_policy": args.router_policy,
    "effective_policy": args.policy, "extra_constructor_keyword": False,
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}, indent=2) + "\n")
PY
flint_node_ip=$(/opt/flint/bin/python-driver -c 'import socket; print(socket.gethostbyname(socket.gethostname()))')
flint_head_ip=$(/opt/flint/bin/python-driver -c 'import os,socket; print(socket.gethostbyname(os.environ["FLINT_HEAD_HOST"]))')
/opt/flint/bin/python-driver - "$flint_runtime/network.json" "$flint_node_ip" "$flint_head_ip" <<'PY'
import ipaddress, json, socket, sys
from pathlib import Path
for address in sys.argv[2:]:
    if ipaddress.ip_address(address).is_loopback:
        raise RuntimeError("A cluster hostname resolved to a loopback address")
Path(sys.argv[1]).write_text(json.dumps({"hostname": socket.gethostname(), "node_ip": sys.argv[2], "head_ip": sys.argv[3]}, indent=2) + "\n")
PY
export RAY_ADDRESS="$flint_head_ip:${FLINT_RAY_PORT:?}"
flint_resources=$(/opt/flint/bin/python-driver -c 'import json,os; print(json.dumps({"slurm_managed_ray_cluster":1,"worker_units":int(os.environ["FLINT_GPUS_PER_NODE"]),"topo_rank":int(os.environ["SLURM_PROCID"])+1}))')
flint_ray_args=(start --disable-usage-stats --node-ip-address="$flint_node_ip"
  --num-cpus="${FLINT_NODE_CPUS:?}" --num-gpus="${FLINT_GPUS_PER_NODE:?}"
  --resources="$flint_resources" --object-store-memory=34359738368
  --min-worker-port=2000 --max-worker-port=2999
  --node-manager-port=1301 --object-manager-port=1303
  --runtime-env-agent-port=1305 --dashboard-agent-grpc-port=1307
  --metrics-export-port=1309 --dashboard-agent-listen-port=1311 --block)

if [[ "$flint_rank" == 0 ]]; then
  # Check online logging authentication without creating a separate probe run.
  /opt/flint/bin/python-driver /flint-run/tools/check_wandb.py \
    --output "$flint_attempt/wandb-auth.json"
  # Dataset and recipe checks run once, before any full-model allocation.
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 /opt/flint/bin/python-driver /flint-run/tools/preflight_dapo.py \
    --config /flint-run/config/grpo-flint-dapo50.yaml \
    --output "$flint_attempt/preflight.json" >"$flint_attempt/preflight.log" 2>&1
  /opt/flint/bin/python-driver -m ray.scripts.scripts "${flint_ray_args[@]}" \
    --head --port="$FLINT_RAY_PORT" --ray-client-server-port=1201 \
    --include-dashboard=false --temp-dir="$TMPDIR/ray" >"$flint_runtime/ray.log" 2>&1 &
  flint_ray_pid=$!
  # A successful TCP connection opens worker startup only for this attempt.
  /opt/flint/bin/python-driver - "$flint_head_ip" "$FLINT_RAY_PORT" <<'PY'
import socket, sys, time
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    try:
        with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2):
            break
    except OSError:
        time.sleep(2)
else:
    raise TimeoutError("Ray head did not open its allocation-local GCS port")
PY
  touch "$flint_attempt/ray-head-ready"
  /opt/flint/bin/python-driver /flint-run/tools/wait_for_ray.py \
    --nodes "$FLINT_EXPECTED_NODES" --gpus-per-node "$FLINT_GPUS_PER_NODE" \
    --output "$flint_attempt/ray-ready.json" >"$flint_attempt/ray-readiness.log" 2>&1
  cp /flint-run/config/grpo-flint-dapo50.yaml "$flint_attempt/recipe.yaml"
  flint_start_step=$(/opt/flint/bin/python-driver /flint-run/tools/checkpoint_window.py start \
    --run-root /flint-run --output "$flint_attempt/starting-checkpoint.json")
  flint_resume_args=()
  if (( flint_start_step > 0 )); then
    flint_resume_args+=(grpo.val_at_start=false)
  fi
  if (( flint_start_step < 50 )); then
  # Keep synchronous CUDA fault attribution scoped to the SGLang actor profile;
  # Automodel's training profile retains its normal asynchronous execution.
  /opt/flint/bin/python-driver /flint-run/tools/launch_grpo_diagnostic.py \
    --config /flint-run/config/grpo-flint-dapo50.yaml "${flint_resume_args[@]}" \
    >"$flint_attempt/training.log" 2>&1 &
  flint_driver_pid=$!
  /opt/flint/bin/python-driver /flint-run/tools/watch_sglang_faults.py \
    --pid "$flint_driver_pid" --log "$flint_attempt/training.log" \
    --output "$flint_attempt/serving-fatal-watch.json" \
    >"$flint_attempt/serving-fatal-watch.log" 2>&1 &
  flint_fault_watch_pid=$!
  # Require the guard to start successfully in this image before trusting it.
  # A failed guard stops this driver's allocation instead of running unguarded.
  if ! /opt/flint/bin/python-driver - "$flint_attempt/serving-fatal-watch.json" <<'PY'
import json, sys, time
from pathlib import Path
path = Path(sys.argv[1])
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    if path.is_file():
        receipt = json.loads(path.read_text())
        if receipt.get("status") == "watching" and receipt.get("driver_birth_identity"):
            break
        raise RuntimeError("Serving fatal-error guard did not start: " + receipt.get("status", "unknown"))
    time.sleep(0.1)
else:
    raise TimeoutError("Serving fatal-error guard did not write its startup receipt")
PY
  then
    printf '1\n' >"$flint_attempt/training.exit-code"
    exit 1
  fi
  /opt/flint/bin/python-driver /flint-work/tools/monitor_phase.py \
    --pid "$flint_driver_pid" --output "$flint_runtime/memory.jsonl" &
  flint_monitor_pid=$!
  /opt/flint/bin/python-driver /flint-run/tools/summarize_long_run.py \
    --run-root /flint-run --watch-pid "$flint_driver_pid" >"$flint_attempt/report.log" 2>&1 &
  flint_report_pid=$!
  set +e
  wait "$flint_driver_pid"
  flint_status=$?
  set -e
  flint_driver_pid=
  kill -TERM "$flint_fault_watch_pid" 2>/dev/null || true
  wait "$flint_fault_watch_pid" 2>/dev/null || true
  flint_fault_watch_pid=
  # Stop the periodic writer before the final audit/report so the final HTML
  # has one writer and cannot be replaced by an earlier progress snapshot.
  kill -TERM "$flint_report_pid" 2>/dev/null || true
  wait "$flint_report_pid" 2>/dev/null || true
  flint_report_pid=
  else
    flint_status=0
    printf 'Final step50 checkpoint already exists; training was not repeated.\n' >"$flint_attempt/training.log"
  fi
  printf '%s\n' "$flint_status" >"$flint_attempt/training.exit-code"
  if [[ "$flint_status" == 0 ]]; then
    /opt/flint/bin/python-driver /flint-run/tools/checkpoint_window.py finish \
      --run-root /flint-run --starting "$flint_attempt/starting-checkpoint.json" \
      --log "$flint_attempt/training.log" --output "$flint_attempt/window-result.json" \
      >"$flint_attempt/window-result.log" 2>&1
    flint_disposition=$( /opt/flint/bin/python-driver -c 'import json,sys; print(json.load(open(sys.argv[1]))["disposition"])' "$flint_attempt/window-result.json" )
    if [[ "$flint_disposition" == continue ]]; then
      /opt/flint/bin/python-driver /flint-run/tools/summarize_long_run.py \
        --run-root /flint-run >"$flint_attempt/report-final.log" 2>&1 || true
      exit 0
    fi
    set +e
    CUDA_VISIBLE_DEVICES= /opt/flint/bin/python-training /flint-run/tools/final_audit.py \
      --run-root /flint-run --output "$flint_attempt/final-validation.json" \
      >"$flint_attempt/final-validation.log" 2>&1
    flint_validation_status=$?
    set -e
    printf '%s\n' "$flint_validation_status" >"$flint_attempt/final-validation.exit-code"
    if /opt/flint/bin/python-driver /flint-run/tools/summarize_long_run.py \
      --run-root /flint-run >"$flint_attempt/report-final.log" 2>&1; then
      printf '0\n' >"$flint_attempt/report-final.exit-code"
    else
      printf '%s\n' "$?" >"$flint_attempt/report-final.exit-code"
    fi
    exit "$flint_validation_status"
  fi
  if /opt/flint/bin/python-driver /flint-run/tools/summarize_long_run.py \
    --run-root /flint-run >"$flint_attempt/report-final.log" 2>&1; then
    printf '0\n' >"$flint_attempt/report-final.exit-code"
  else
    printf '%s\n' "$?" >"$flint_attempt/report-final.exit-code"
  fi
  exit "$flint_status"
fi

flint_deadline=$((SECONDS + 1800))
until [[ -f "$flint_attempt/ray-head-ready" ]]; do
  [[ ! -f "$flint_attempt/ended" ]] || exit 1
  (( SECONDS < flint_deadline )) || { echo 'Ray head gate timed out' >&2; exit 1; }
  sleep 2
done
/opt/flint/bin/python-driver -m ray.scripts.scripts "${flint_ray_args[@]}" \
  --address="$RAY_ADDRESS" >"$flint_runtime/ray.log" 2>&1 &
flint_ray_pid=$!
/opt/flint/bin/python-driver /flint-work/tools/monitor_phase.py \
  --pid "$flint_ray_pid" --output "$flint_runtime/memory.jsonl" &
flint_monitor_pid=$!
until [[ -f "$flint_attempt/ended" ]]; do
  kill -0 "$flint_ray_pid" 2>/dev/null || { echo 'Own Ray worker stopped early' >&2; exit 1; }
  sleep 5
done
flint_status=$(cat "$flint_attempt/ended")
[[ "$flint_status" =~ ^[0-9]+$ ]] || exit 1
exit "$flint_status"
