#!/bin/bash

set -euo pipefail

arm=""
result_dir=""
expected_product_head=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) arm=${2:?}; shift 2 ;;
    --result-dir) result_dir=${2:?}; shift 2 ;;
    --expected-product-head) expected_product_head=${2:?}; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${arm}" && -n "${result_dir}" && -n "${expected_product_head}" ]]
[[ "$(pwd -P)" == /home/* ]]
[[ "${result_dir}" == /lustre/* ]]
[[ "${RAY_TMPDIR:-}" == /tmp* ]]
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"

python3 -m research.qwen3_8b_draft_cadence_200step.launch preflight \
  --arm "${arm}" --source-root "$(pwd -P)" \
  --expected-product-head "${expected_product_head}"

identity="${result_dir}/run-identity.json"
if [[ -e "${identity}" ]]; then
  python3 - "${identity}" "${arm}" "${expected_product_head}" "${WANDB_RUN_ID}" <<'PY'
import json
import sys

path, arm, head, wandb_id = sys.argv[1:]
payload = json.load(open(path))
expected = {"arm": arm, "product_head": head, "wandb_run_id": wandb_id}
if any(payload.get(key) != value for key, value in expected.items()):
    raise SystemExit("existing run identity does not match resume request")
PY
  python3 -m research.qwen3_8b_draft_cadence_200step.launch resume-preflight \
    --arm "${arm}" --result-dir "${result_dir}" \
    --expected-product-head "${expected_product_head}"
else
  mkdir "${result_dir}"
  python3 - "${identity}" "${arm}" "${expected_product_head}" "${WANDB_RUN_ID}" <<'PY'
import json
import os
import sys

path, arm, head, wandb_id = sys.argv[1:]
payload = {
    "schema_version": 1,
    "arm": arm,
    "product_head": head,
    "wandb_run_id": wandb_id,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
with open(path, "x") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
fi

mapfile -t overrides < <(
  python3 -m research.qwen3_8b_draft_cadence_200step.launch overrides \
    --arm "${arm}" --result-dir "${result_dir}"
)
config_path="$({ python3 - "${arm}" <<'PY'
import sys
from research.qwen3_8b_draft_cadence_200step.matrix import build_arms

name = sys.argv[1]
print(next(arm.config_path for arm in build_arms() if arm.name == name))
PY
} )"

uv run examples/run_grpo.py --config "${config_path}" "${overrides[@]}"

python3 -m research.qwen3_8b_draft_cadence_200step.launch adapt-native \
  --arm "${arm}" --result-dir "${result_dir}" \
  --expected-product-head "${expected_product_head}"

python3 -m research.qwen3_8b_draft_cadence_200step.launch terminal-preflight \
  --arm "${arm}" --result-dir "${result_dir}"
