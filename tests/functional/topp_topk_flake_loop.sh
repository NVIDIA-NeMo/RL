#!/bin/bash
# TEMP, DO NOT MERGE: run grpo_megatron_generation_topp_topk.sh N times and count
# how often its metric check fails, to tell a flaky check from a real regression.
#
# Usage: topp_topk_flake_loop.sh <num_runs> <tag>
# Always exits 0 so every job finishes; read the FLAKE_* lines in the log.

set -uo pipefail

NUM_RUNS=${1:?num_runs}
TAG=${2:?tag}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
TEST=grpo_megatron_generation_topp_topk
METRICS=$SCRIPT_DIR/$TEST/metrics.json
OUT_DIR=$SCRIPT_DIR/topp_topk_flake_results/$TAG
mkdir -p $OUT_DIR

fails=0
crashes=0
for i in $(seq 1 $NUM_RUNS); do
    echo "FLAKE_START tag=$TAG run=$i/$NUM_RUNS"
    rm -f $METRICS
    bash $SCRIPT_DIR/$TEST.sh
    rc=$?
    if [[ -f $METRICS ]]; then
        cp $METRICS $OUT_DIR/run_$i.json
        # Per-step values of the checked metric and any other prob-error / KL metric.
        detail=$(python3 - "$METRICS" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
parts = []
for k in sorted(d):
    if "prob_error" in k or "kl" in k.lower():
        vals = [d[k][s] for s in sorted(d[k], key=int)]
        parts.append(f"{k}=" + ",".join(f"{v:.4f}" for v in vals))
print(" ".join(parts))
EOF
)
        verdict=$([[ $rc -eq 0 ]] && echo PASS || echo FAIL)
        [[ $rc -ne 0 ]] && fails=$((fails + 1))
    else
        # No metrics at all: the training run itself died, not the check.
        detail="no metrics.json"
        verdict=CRASH
        crashes=$((crashes + 1))
    fi
    echo "FLAKE_RESULT tag=$TAG run=$i rc=$rc verdict=$verdict $detail"
done

echo "FLAKE_SUMMARY tag=$TAG runs=$NUM_RUNS metric_fails=$fails crashes=$crashes"
exit 0
