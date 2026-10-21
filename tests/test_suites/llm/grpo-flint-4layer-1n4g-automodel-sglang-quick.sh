#!/usr/bin/env bash
# Private preview checkpoint test. Run inside the supplied Flint container with
# this workspace mounted at /flint-work; the public nightly runner has no access.
set -euo pipefail

test -x /opt/flint/bin/python-driver
test -f /flint-work/checkpoint/config.json
exec bash /flint-work/tools/run_grpo.sh "$@"
