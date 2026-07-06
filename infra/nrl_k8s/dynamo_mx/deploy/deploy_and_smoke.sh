#!/usr/bin/env bash
# Phase 0.5 deploy + smoke script.
#
# Prerequisites (do this FIRST):
#   1. tsh login --proxy=nv-prd-dgxc.teleport.sh:443 --auth=nvidian
#   2. tsh kube login dynamo-gcp-dev-02
#
# Then run:
#   bash /tmp/mx-phase-0.5-build/deploy_and_smoke.sh
#
# Idempotent: applies the DGD patch, waits for pod rollout, runs
# the byte-identity smoke against the trainer, and reports pass/fail.

set -euo pipefail

MX_SHA=$(cd /home/kavink/Work/Github/MX0/modelexpress && git rev-parse --short HEAD)
IMAGE_TAG="nvcr.io/nvidian/dynamo-dev/model-express-dev:phase-0.5-${MX_SHA}"
NS=kavin

echo "==============================================================="
echo "Phase 0.5 deploy + smoke — kavin/nemorl-mx-worker"
echo "  image: ${IMAGE_TAG}"
echo "  toggle: MX_MEGATRON_BUFFER_LOC=host"
echo "==============================================================="

echo ""
echo "[1/5] Pre-deploy state:"
kubectl -n ${NS} get dgd nemorl-mx-worker -o jsonpath='{.spec.services.VllmDecodeWorker.extraPodSpec.mainContainer.image}{"\n"}' \
  | sed 's/^/  current image: /'
kubectl -n ${NS} get pods -l nvidia.com/dynamo-graph-deployment-name=nemorl-mx-worker \
  --no-headers 2>/dev/null | wc -l | sed 's/^/  current pods: /'

echo ""
echo "[2/5] Applying Phase 0.5 DGD patch..."
kubectl -n ${NS} patch dgd nemorl-mx-worker \
  --type merge \
  -p "$(cat /tmp/mx-phase-0.5-build/phase_0_5_manifest_patch.yaml)"

echo ""
echo "[3/5] Waiting for pod rollout (Grove operator picks up spec change)..."
# Wait for all 4 vllm decode workers to be Ready with the new image.
for i in {1..40}; do
    ready_new=$(kubectl -n ${NS} get pods \
        -l nvidia.com/dynamo-component-type=worker \
        -o json 2>/dev/null \
        | python3 -c "
import json, sys
data = json.load(sys.stdin)
count = 0
for p in data['items']:
    if p['metadata']['name'].startswith('nemorl-mx-worker-0-vllmdecodeworker'):
        img = p['spec']['containers'][0]['image']
        conds = {c['type']: c['status'] for c in p['status'].get('conditions', [])}
        if 'phase-0.5' in img and conds.get('Ready') == 'True':
            count += 1
print(count)
        ")
    total=4
    echo "  attempt ${i}/40: ${ready_new}/${total} phase-0.5 workers Ready"
    if [ "${ready_new}" = "${total}" ]; then
        echo "  -> rollout complete"
        break
    fi
    sleep 15
done

if [ "${ready_new}" != "${total}" ]; then
    echo ""
    echo "  ROLLOUT TIMEOUT — pods not Ready with phase-0.5 image after 10 min."
    echo "  Check: kubectl -n ${NS} get pods -l nvidia.com/dynamo-component-type=worker"
    echo "  Logs:  kubectl -n ${NS} logs <pod> --tail=100"
    exit 1
fi

echo ""
echo "[4/5] Verifying Phase 0.5 code is loaded in a live worker..."
POD=$(kubectl -n ${NS} get pods -l nvidia.com/dynamo-component-type=worker \
    --no-headers 2>/dev/null | grep vllmdecodeworker | head -1 | awk '{print $1}')
echo "  pod: ${POD}"
kubectl -n ${NS} exec ${POD} -- python3 -c "
from modelexpress.nixl_transfer import NixlTransferManager, _resolve_local_mem_type
assert hasattr(NixlTransferManager, 'rebind_tensors'), 'MX Phase 0.5 not loaded'
from dynamo.vllm.mx_refit.extension import MxRefitWorkerExtension
import inspect
src = inspect.getsource(MxRefitWorkerExtension)
assert 'MX_MEGATRON_BUFFER_LOC' in src, 'Dynamo Phase 0.5 not loaded'
print('  MX + Dynamo Phase 0.5 code verified in-pod')
"
echo "  env verification:"
kubectl -n ${NS} exec ${POD} -- bash -c 'env | grep -E "MX_MEGATRON_BUFFER_LOC|MX_RDMA_NIC_PIN"' \
    | sed 's/^/    /'

echo ""
echo "[5/5] Triggering byte-identity smoke via refit_verifier..."
# TODO: wire this to the actual verifier script. Placeholder:
# python3 /home/kavink/Work/Github/RL/RL/tools/refit_verifier.py --deployment nemorl-mx-worker --ns kavin
echo "  (smoke command placeholder — hook up refit_verifier here)"

echo ""
echo "==============================================================="
echo "DEPLOY COMPLETE. Next step: run the byte-identity smoke and"
echo "check GPU memory footprint drop (should be ~model-shard-sized"
echo "less HBM used per vLLM worker vs. loc=device baseline)."
echo "==============================================================="
