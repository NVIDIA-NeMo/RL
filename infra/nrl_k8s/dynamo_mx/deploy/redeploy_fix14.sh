#!/usr/bin/env bash
# Redeploy the vLLM decode workers onto fix14 (replica_uid fan-out fix).
# Patches the DGD image at the correct components[] index, then forces a
# rollout (Grove does not auto-restart on spec change -> delete pods).
set -euo pipefail
NS=kavin
DGD=nemorl-mx-worker
NEWIMG="nvcr.io/nvidian/dynamo-dev/model-express-dev:phase-0.5-3623f45-fix14"

# Find the VllmDecodeWorker index in spec.components dynamically.
IDX=$(kubectl -n ${NS} get dgd ${DGD} -o json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for i,c in enumerate(d['spec']['components']):
    if c.get('name')=='VllmDecodeWorker': print(i); break
")
echo "VllmDecodeWorker is components[${IDX}]"

echo "current image:"
kubectl -n ${NS} get dgd ${DGD} -o jsonpath="{.spec.components[${IDX}].podTemplate.spec.containers[0].image}"; echo

kubectl -n ${NS} patch dgd ${DGD} --type=json \
  -p "[{\"op\":\"replace\",\"path\":\"/spec/components/${IDX}/podTemplate/spec/containers/0/image\",\"value\":\"${NEWIMG}\"}]"

echo "patched image:"
kubectl -n ${NS} get dgd ${DGD} -o jsonpath="{.spec.components[${IDX}].podTemplate.spec.containers[0].image}"; echo

echo "forcing rollout (delete vllm decode worker pods)..."
kubectl -n ${NS} delete pod -l nvidia.com/dynamo-component-type=worker --field-selector=status.phase=Running 2>/dev/null || true
echo "done — watch: kubectl -n ${NS} get pods -l nvidia.com/dynamo-component-type=worker -w"
