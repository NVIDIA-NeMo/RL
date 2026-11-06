#!/bin/bash
# Opt-in end-to-end CUA smoke: one update and one held-out task.
# Requires 16 GB200-class GPUs and the separately provisioned desktop assets.
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJECT_ROOT"
if [[ "${LOGRA_RUN_CUA_TEST:-0}" != 1 ]]; then
  echo "SKIP: set LOGRA_RUN_CUA_TEST=1 after preparing the CUA runtime and 16-GPU Ray allocation."
  exit 0
fi
: "${OSWORLD_ROOT:?Set the external model/data root}"
: "${OSWORLD_CREDENTIAL_FILE:?Set the private sandbox credential file}"
: "${CUA_OUTPUT_DIR:?Use a fresh output directory outside the source tree}"
uv run --locked --project ../.. --extra automodel --extra nemo_gym python \
  run_cua.py --config configs/cua_smoke.yaml "$@"

uv run --locked --project ../.. --extra automodel --extra nemo_gym python - <<'CHECK'
import json
import math
import os
from pathlib import Path

records = [json.loads(line) for line in (Path(os.environ["CUA_OUTPUT_DIR"]) / "metrics.jsonl").read_text().splitlines()]
updates = [row for row in records if "train" in row]
assert [row["step"] for row in updates] == [1], "Expected exactly one completed training update"
for key in ("loss", "grad_norm", "score"):
    assert math.isfinite(updates[0]["train"][key]), key
assert any("validation" in row for row in records), "Missing final held-out evaluation"
print("CUA_FUNCTIONAL_OK: one update and final evaluation")
CHECK
