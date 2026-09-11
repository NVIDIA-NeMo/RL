#!/usr/bin/env bash
# Drive the Energon SFT pipeclean inside the container.
#   stage 1: loader-only iteration (fast, no Ray / no Megatron)
#   stage 2: full run_sft_v2
# Usage: STAGE=1|2|all bash tools/pipeclean.sh
set -euo pipefail

cd /opt/nemo-rl || cd "$(dirname "$0")/.."
CFG=${CFG:-examples/configs/sft_v2_tests/sft-super67b-generalist.n16soak.yaml}
STAGE=${STAGE:-2}
STEPS=${STEPS:-20}
BATCH=${BATCH:-128}
OVERRIDES=${OVERRIDES:-}
# Per-run dir: concurrent jobs share this mount and would truncate one log.
OUT=${OUT:-/opt/nemo-rl/pipeclean-logs/${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"

if [[ "${SKIP_INIT:-0}" != "1" ]]; then
  echo "===== init deps ====="
  # Same set as code/RL/init.sh, but pinned to the container venv for every
  # package (the original omits --python on the last line).
  uv pip install -q --no-config --python /opt/nemo_rl_venv/bin/python \
      --index-url https://pypi.org/simple \
      "av>=17.1.0" "librosa==0.11.0" "soundfile>=0.13.1" soxr megatron-energon 2>&1
  echo "init deps exit=$?"
fi

echo "===== env ====="
echo "pwd=$(pwd)  host=$(hostname)"
python3 -c 'import sys; print("python:", sys.executable, sys.version.split()[0])' 2>&1
for m in ray megatron.energon av librosa soundfile; do
  python3 -c "import $m; print('  OK  $m', getattr($m,'__version__','?'))" 2>&1 | tail -1 || true
done

if [[ "$STAGE" == "1" || "$STAGE" == "all" ]]; then
  # batch_size mirrors what sft_worker passes: global batch / DP world size.
  # Packing pressure (and the ValueErrors it triggers) only shows at the real size.
  echo "===== stage 1: loader-only ($STEPS steps, batch=$BATCH) ====="
  uv run tools/iterate_energon_sft.py \
      --config "$CFG" --steps "$STEPS" --batch-size "$BATCH" $OVERRIDES \
      > "$OUT/loader.log" 2>&1
  echo "stage 1 exit=$?"
  tail -40 "$OUT/loader.log"
fi

if [[ "$STAGE" == "2" || "$STAGE" == "all" ]]; then
  echo "===== stage 2: full sft ====="
  uv run examples/run_sft_v2.py --config "$CFG" $OVERRIDES \
      > "$OUT/sft.log" 2>&1
  echo "stage 2 exit=$?"
  tail -60 "$OUT/sft.log"
fi
