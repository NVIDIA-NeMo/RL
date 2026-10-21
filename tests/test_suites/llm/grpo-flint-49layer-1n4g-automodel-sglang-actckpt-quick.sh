#!/usr/bin/env bash
# Private full49 integration test for the pinned Flint container. Mount the
# staged task workspace at /flint-work and original checkpoint at /flint-model.
# run_grpo.sh applies the guarded Automodel overlay before creating workers.
# This preview checkpoint is unavailable to the public nightly runner.
set -euo pipefail

test -x /opt/flint/bin/python-driver
test -f /flint-model/config.json
test -f /flint-work/tools/apply_automodel_overlay.py
export FLINT_SYNTHETIC_SMOKE=0
exec bash /flint-work/tools/run_grpo.sh "$@"
