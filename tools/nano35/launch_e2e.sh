#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Submit one 32-node job only after the candidate image passes GPU preflight.
set -euo pipefail
# Ray's Unix socket paths must stay short and node-local even when submission
# inherits a long login-node TMPDIR (for example, from an interactive agent).
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp RAY_TMPDIR=/tmp
MODE=${1:?Usage: launch_e2e.sh fresh|resume /absolute/run/directory}
export NANO35_RUN_DIR=${2:?Pass a new run directory, or the same directory for resume}
[[ "$MODE" == fresh || "$MODE" == resume ]] || exit 2
[[ "$NANO35_RUN_DIR" == /* ]] || { echo 'Run directory must be absolute' >&2; exit 2; }
case "$NANO35_RUN_DIR" in /lustre/share|/lustre/share/*)
    echo '/lustre/share is read-only on Lyris compute nodes' >&2; exit 2;; esac
for key in NANO35_IMAGE NANO35_PREFLIGHT_REPORT NANO35_DATA_PREFLIGHT_REPORT NANO35_MODEL NANO35_DATA NANO35_SIF_TEMPLATE \
           NANO35_ACCOUNT NANO35_PARTITION; do
    [[ -n "${!key:-}" ]] || { echo "Set $key first" >&2; exit 2; }
    [[ "${!key}" != *$'\n'* && "${!key}" != *,* ]] || { echo "Unsupported path in $key" >&2; exit 2; }
    export "$key"
done
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export NANO35_RECIPE_RELATIVE=examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml
export NANO35_RECIPE_SOURCE="$REPO_ROOT/$NANO35_RECIPE_RELATIVE"
export NANO35_RUN_NAME=$(basename "$NANO35_RUN_DIR")
export NANO35_START_MODE="$MODE"
[[ "$NANO35_RUN_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 2
mkdir -p "$NANO35_RUN_DIR"
exec 9>"$NANO35_RUN_DIR/.submit.lock"
flock -n 9 || { echo 'A submission is already in progress' >&2; exit 1; }
JOB_PREFIX="${NANO35_ACCOUNT}-nano35."
ACTIVE=$(squeue -h -u "$(id -un)" -o '%A|%j' | awk -F '|' -v prefix="$JOB_PREFIX" 'index($2,prefix)==1')
[[ -z "$ACTIVE" ]] || { echo "Wait for the existing Nano job: $ACTIVE" >&2; exit 1; }

if [[ "$MODE" == fresh ]]; then
    [[ ! -e "$NANO35_RUN_DIR/run-manifest.json" && ! -e "$NANO35_RUN_DIR/job-id.txt" ]] || {
        echo 'Fresh mode requires a new experiment directory' >&2; exit 1;
    }
else
    PREVIOUS_JOB=$(cat "$NANO35_RUN_DIR/job-id.txt")
    [[ "$PREVIOUS_JOB" =~ ^[0-9]+$ ]] || exit 1
    PREVIOUS_STATE=$(sacct -X -n -P -j "$PREVIOUS_JOB" --format=JobIDRaw,State | awk -F '|' -v id="$PREVIOUS_JOB" '$1==id {print $2}')
    [[ "$PREVIOUS_STATE" == COMPLETED ]] || {
        echo "Previous job is $PREVIOUS_STATE; inspect it before submitting a continuation" >&2; exit 1;
    }
    export NANO35_PREVIOUS_JOB="$PREVIOUS_JOB"
fi

export NANO35_IMAGE_SHA256=$(sha256sum "$NANO35_IMAGE" | awk '{print $1}')
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

env = os.environ
report = json.loads(Path(env['NANO35_PREFLIGHT_REPORT']).read_text())
required = ('tp4_generation', 'mnnvl_breakable', 'conversation_reuse', 'cache_reset', 'swe_agent_and_evaluation')
if report.get('status') != 'passed' or report.get('image_sha256') != env['NANO35_IMAGE_SHA256']:
    raise SystemExit('GPU preflight did not pass for this exact image')
if not all(report.get('checks', {}).get(key) == 'passed' for key in required):
    raise SystemExit('GPU preflight is missing a required check')
data = Path(env['NANO35_DATA'])
rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
if len(rows) != 403 or any(row['agent_ref']['name'] != 'swe_agents_train' for row in rows):
    raise SystemExit('Expected the 403-row SWE training dataset')
for row in rows:
    sif = Path(env['NANO35_SIF_TEMPLATE'].format(**row))
    if not sif.is_file() or sif.stat().st_size == 0:
        raise SystemExit(f'Missing prebuilt SWE SIF: {sif}')
model = Path(env['NANO35_MODEL'])
index = json.loads((model/'model.safetensors.index.json').read_text())
for name in ('config.json', 'tokenizer_config.json', *set(index['weight_map'].values())):
    if not (model/name).is_file() or (model/name).stat().st_size == 0:
        raise SystemExit(f'Missing model asset: {model/name}')
run = Path(env['NANO35_RUN_DIR'])
recipe = Path(env['NANO35_RECIPE_SOURCE']) if env['NANO35_START_MODE'] == 'fresh' else run/'recipe.yaml'
data_report = json.loads(Path(env['NANO35_DATA_PREFLIGHT_REPORT']).read_text())
expected = {
    'image_sha256': env['NANO35_IMAGE_SHA256'],
    'recipe_sha256': hashlib.sha256(recipe.read_bytes()).hexdigest(),
    'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest(),
}
if data_report.get('status') != 'passed' or any(data_report.get(key) != value for key, value in expected.items()):
    raise SystemExit('Training data preflight did not pass for this exact image, recipe and dataset')
for split in ('train', 'validation'):
    result = data_report.get('splits', {}).get(split, {})
    if result.get('rows') != 403 or result.get('raw_payload_and_order_preserved') is not True:
        raise SystemExit(f'Training data preflight did not verify all 403 {split} rows')
if env['NANO35_START_MODE'] == 'fresh':
    # Pin the externally supplied recipe: old candidate images contain a
    # stale inherited dataset selection. Continuations reuse these exact bytes.
    with (run/'recipe.yaml').open('xb') as frozen:
        frozen.write(recipe.read_bytes())
    (run/'recipe.yaml').chmod(0o444)
    with (run/'data-preflight.json').open('x') as frozen:
        frozen.write(json.dumps(data_report, indent=2) + '\n')
if env['NANO35_START_MODE'] == 'resume':
    previous = env['NANO35_PREVIOUS_JOB']
    attempt = run/'attempts'/previous
    if (attempt/'driver-status.txt').read_text().strip() != 'completed':
        raise SystemExit('Previous training/checkpoint verification did not complete')
    checkpoint_report = json.loads((attempt/'checkpoint-report.json').read_text())
    if checkpoint_report['status'] != 'metadata-passed':
        raise SystemExit('Previous checkpoint metadata verification failed')
    if checkpoint_report['training_complete']:
        raise SystemExit('The configured training target is already complete')
    checkpoint = Path(checkpoint_report['checkpoint'])
    if checkpoint.parent != run/'checkpoints':
        raise SystemExit('Checkpoint belongs to another experiment')
    for name, recorded in checkpoint_report['files'].items():
        stat = (checkpoint/name).stat()
        if {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns} != recorded:
            raise SystemExit(f'Checkpoint changed after verification: {name}')
    manifest = json.loads((run/'run-manifest.json').read_text())
    if manifest['image_sha256'] != env['NANO35_IMAGE_SHA256']:
        raise SystemExit('The continuation image differs from the original image')
    if manifest['data_sha256'] != hashlib.sha256(data.read_bytes()).hexdigest():
        raise SystemExit('Dataset changed since the original run')
print('Input assets, image validation and single-job checks passed')
PY

export CONTAINER="$NANO35_IMAGE"
export MOUNTS="/lustre:/lustre,/dev/fuse:/dev/fuse,$NANO35_RUN_DIR/recipe.yaml:/opt/nemo-rl/$NANO35_RECIPE_RELATIVE:ro"
export GPUS_PER_NODE=4 DEDICATED_RAY_HEAD=0
export BASE_LOG_DIR="$NANO35_RUN_DIR"
export RAY_LOG_SYNC_FREQUENCY=60
export COMMAND='exec bash /opt/nemo-rl/tools/nano35/e2e_driver.sh'
export SETUP_COMMAND='test -x /opt/nemo_rl_venv/bin/ray && test -f /opt/nano35-metadata/runtime-ready'
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_USAGE_STATS_ENABLED=0
export WANDB_MODE=disabled NRL_FORCE_REBUILD_VENVS=false TRTLLM_REQUIRE_CACHED_WHEEL=1
cd "$REPO_ROOT"
JOB_ID=$(sbatch --parsable --nodes=32 --segment=16 --exclusive --time=05:00:00 \
    --account="$NANO35_ACCOUNT" --partition="$NANO35_PARTITION" \
    --job-name="${JOB_PREFIX}e2e" --output="$NANO35_RUN_DIR/slurm-%j.out" ray.sub)
JOB_ID=${JOB_ID%%;*}
printf '%s\n' "$JOB_ID" > "$NANO35_RUN_DIR/job-id.txt"
printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$JOB_ID" "$MODE" >> "$NANO35_RUN_DIR/submissions.tsv"
echo "Submitted $JOB_ID ($MODE). No continuation has been queued."
