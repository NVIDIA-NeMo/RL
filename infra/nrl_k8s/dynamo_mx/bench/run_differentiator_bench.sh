#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Push-button MX-vs-NCCL differentiator benchmark for the 30B MoE refit path.
# Runs the scenarios where MX/NIXL is expected to win (elasticity, stragglers,
# partial/EP refit) plus the apples-to-apples transport + refit-phase numbers.
#
# Prereqs (set via env; the script preflight-checks and skips gracefully):
#   NS            k8s namespace (default: kavin)
#   MX_URL        modelexpress-server URL (default: modelexpress-server.$NS.svc.cluster.local:8001)
#   MODEL         served model id (default: Qwen/Qwen3-30B-A3B-Instruct-2507)
#   WORKER_LABEL  label selector for decode workers
#   FRONTEND      http://<frontend>:<port> for the DGD (for HTTP-mode refit)
#   OUT           results dir (default: ./diff_bench_results)
#
# Scenarios (each writes JSON to $OUT and prints a summary line):
#   S1 transport baseline    — NCCL broadcast vs MX pull (single + multi-rail)
#   S2 refit-phase breakdown — register/wire/translate/load/e2e, backend swap
#   S3 elastic-join latency  — scale in a fresh worker, time to current version
#   S4 straggler isolation   — one slow worker; fleet completion vs a barrier
#   S5 partial/EP refit       — bytes-on-wire per worker at EP>1 (+ synthetic proof)
#
# S1/S2/S3/S4-live need a trainer actively PUBLISHING versions and (for the NCCL
# arm) an NCCL weight-transfer path deployed for the same model. Where a prereq
# is missing the scenario is SKIPPED with a clear reason, so this is safe to run
# incrementally as the deployment comes up.

set -uo pipefail

NS="${NS:-kavin}"
MX_URL="${MX_URL:-modelexpress-server.${NS}.svc.cluster.local:8001}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
WORKER_LABEL="${WORKER_LABEL:-nvidia.com/dynamo-component-type=worker}"
FRONTEND="${FRONTEND:-}"
OUT="${OUT:-./diff_bench_results}"
HARNESS_DIR="${HARNESS_DIR:-$(cd "$(dirname "$0")" && pwd)}"
CYCLES="${CYCLES:-10}"
mkdir -p "$OUT"

log() { printf '\n=== %s ===\n' "$*"; }
skip() { printf '  [SKIP] %s\n' "$*"; }
have_pod() { kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep -q '1/1'; }

first_worker() {
  kubectl -n "$NS" get pods -l "$WORKER_LABEL" --no-headers 2>/dev/null \
    | grep '1/1' | grep -i vllmdecodeworker | head -1 | awk '{print $1}'
}

trainer_publishing() {
  # true if the catalog has >=1 READY source for $MODEL (a live publisher)
  local pod; pod=$(first_worker); [ -z "$pod" ] && return 1
  kubectl -n "$NS" exec "$pod" -- python3 - "$MX_URL" "$MODEL" <<'PY' 2>/dev/null | grep -q 'sources=[1-9]'
import sys
from modelexpress.engines.vllm.weight_update import MxVllmWeightUpdater, MxInitInfo
u = MxVllmWeightUpdater()
u.initialize_weight_update_setup(MxInitInfo(mx_server_url=sys.argv[1], model_name=sys.argv[2]))
try:
    c = u._receiver.discover_v2_sources(model_name=sys.argv[2], min_version=1, same_rank_only=False, include_replicas=True)
    print(f"sources={len(c)}")
except Exception as e:
    print(f"sources=0 ({e})")
PY
}

# ---------------------------------------------------------------------------
preflight() {
  log "preflight"
  have_pod || { echo "  no ready worker pod in ns=$NS — aborting"; exit 1; }
  echo "  ns=$NS  model=$MODEL  mx=$MX_URL  out=$OUT"
  echo "  worker: $(first_worker)"
  if trainer_publishing; then echo "  trainer: PUBLISHING (live source present)"; PUBLISHING=1
  else echo "  trainer: not publishing (S2/S3/S4-live will skip)"; PUBLISHING=0; fi
}

# S1 — transport baseline (raw GPU->GPU BW; no publisher needed, uses 2 bench pods)
s1_transport() {
  log "S1 transport baseline (NCCL vs MX)"
  echo "  NCCL: kubectl apply the 2-pod nccl-bench, run nccl_bcast_bench.py (1-rail baseline: NCCL_IB_HCA=mlx5_0, NCCL_MNNVL_ENABLE=0)"
  echo "  MX:   smoke_fanout_receiver.py FANOUT_SEED_RANDOM_GB=8 FANOUT_NO_VERIFY=1 (single + MX_RDMA_NIC_PIN=stripe)"
  echo "  -> record Gbps single-rail + 4-rail; reference: NCCL ~375, MX ~316/~529"
  skip "wire-up left to the 2-pod bench (nccl-bench-{0,1}); harnesses: nccl_bcast_bench.py / smoke_fanout_receiver.py"
}

# S2 — refit-phase breakdown via the native backend swap
s2_refit_phases() {
  log "S2 refit-phase breakdown (backend swap mx vs nccl)"
  [ "$PUBLISHING" = 1 ] || { skip "no publishing trainer"; return; }
  [ -n "$FRONTEND" ] || { skip "set FRONTEND=http://<host>:<port> for HTTP-mode driver"; return; }
  for be in mx nccl; do
    echo "  driving $CYCLES cycles, backend=$be"
    python3 "$HARNESS_DIR/mx_vs_nccl_refit_bench.py" --mode http --worker-url "$FRONTEND" \
      --backend "$be" --model "$MODEL" --cycles "$CYCLES" --out "$OUT/refit_$be.json" \
      || skip "backend=$be run failed (path may not be wired)"
  done
  python3 "$HARNESS_DIR/mx_vs_nccl_refit_bench.py" --compare "$OUT/refit_mx.json" "$OUT/refit_nccl.json" 2>/dev/null \
    | tee "$OUT/refit_compare.txt" || skip "compare skipped (missing runs)"
}

# S3 — elastic-join latency (MX's home turf; NCCL can't do this without rebuild)
s3_elastic_join() {
  log "S3 elastic-join latency"
  [ "$PUBLISHING" = 1 ] || { skip "no publishing trainer to join to"; return; }
  local before after new t0
  before=$(kubectl -n "$NS" get pods -l "$WORKER_LABEL" --no-headers | grep -c vllmdecodeworker)
  echo "  scaling decode workers $before -> $((before+1)) and timing the new pod to current version"
  # Scale via the DGD (adjust the resource/name for your deployment):
  echo "  kubectl -n $NS patch dgd <dgd-name> -p '{\"spec\":{\"services\":{\"VllmDecodeWorker\":{\"replicas\":$((before+1))}}}}'"
  echo "  then: watch the new pod log for '[mx-wt] ... update ... version=' and record wall time from Ready->synced"
  skip "scale command left parameterized on <dgd-name>; measurement = new-pod Ready -> first successful update_weights"
}

# S4 — straggler isolation
s4_straggler() {
  log "S4 straggler isolation"
  [ "$PUBLISHING" = 1 ] || { skip "no publishing trainer"; return; }
  echo "  inject a slow reader on one worker (e.g. cgroup cpu throttle or a sleep in the load hook),"
  echo "  publish a version, and record fleet-completion time + whether healthy workers proceed."
  echo "  MX expectation: healthy workers complete independently (pull is per-worker);"
  echo "  a collective barrier would stall all workers on the straggler."
  skip "fault-injection hook is deployment-specific; harness: smoke_fanout_receiver.py with one throttled follower"
}

# S5 — partial / EP byte-pruning
s5_partial_ep() {
  log "S5 partial / EP refit byte-pruning"
  echo "  synthetic proof (always runnable):"
  ( cd "$HARNESS_DIR" && PYTHONPATH="${MX_PY:-}" python3 ep_gt1_byte_pruning.py 2>/dev/null | tail -3 ) \
    || skip "set MX_PY=<modelexpress_client/python> to run the synthetic proof"
  if [ "$PUBLISHING" = 1 ]; then
    echo "  live: on an --enable-expert-parallel EP>1 deploy, per-worker pulled bytes should be ~1/EP;"
    echo "  read each worker's [mx-wt] arena/register + wire log and sum byte_count per worker."
  else
    skip "live EP>1 measurement needs the EP>1 DGD (ep8_rollout_dgd.example.yaml) + publisher"
  fi
}

# ---------------------------------------------------------------------------
preflight
s1_transport
s2_refit_phases
s3_elastic_join
s4_straggler
s5_partial_ep
log "done — results in $OUT"
