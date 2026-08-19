#!/bin/bash

set -euo pipefail

: "${PHASE2_REPO:?PHASE2_REPO must be set}"
: "${PHASE2_RUNTIME_ENV:?PHASE2_RUNTIME_ENV must be set}"
: "${PHASE2_NEMO_GYM_ENV:?PHASE2_NEMO_GYM_ENV must be set}"
: "${PHASE2_RAY_VERSION:?PHASE2_RAY_VERSION must be set}"
: "${PHASE2_RAY_RUNTIME_ENV_VERIFICATION:?set an explicit, write-once output path}"

PHASE2_RAY_CONTROL_PYTHON=${PHASE2_RAY_CONTROL_PYTHON:-$PHASE2_RUNTIME_ENV/bin/python}
[[ ! -e $PHASE2_RAY_RUNTIME_ENV_VERIFICATION ]] || {
  echo "Refusing to overwrite Ray validation evidence: $PHASE2_RAY_RUNTIME_ENV_VERIFICATION" >&2
  exit 1
}
for executable in \
  "$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_NEMO_GYM_ENV/bin/python" \
  "$PHASE2_RAY_CONTROL_PYTHON"; do
  [[ -x $executable ]] || {
    echo "Required Ray validation executable is missing: $executable" >&2
    exit 1
  }
done

"$PHASE2_RUNTIME_ENV/bin/python" \
  "$PHASE2_REPO/experiments/nemo_gym_phase2/validate_ray_runtime_env.py" \
  --repo "$PHASE2_REPO" \
  --expected-ray-version "$PHASE2_RAY_VERSION" \
  --expected-control-plane-python "$PHASE2_RAY_CONTROL_PYTHON" \
  --py-executable "$PHASE2_RUNTIME_ENV/bin/python" \
  --py-executable "$PHASE2_NEMO_GYM_ENV/bin/python" \
  --output "$PHASE2_RAY_RUNTIME_ENV_VERIFICATION"
