#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Maintainer-only local format conversion. This command never uploads an image.
set -euo pipefail
IMAGE=${1:?Usage: export_oci.sh candidate.sqsh /absolute/output.oci.tar tag}
OUTPUT=${2:?Pass an unused absolute output path}
TAG=${3:?Pass the local OCI tag}
[[ "$OUTPUT" == /* && "$TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] || exit 2
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.partial" ]] || exit 1
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run on an allocated compute node' >&2; exit 1; }
command -v enroot >/dev/null
TASK_TMP=$(mktemp -d "/tmp/nano35-oci-${SLURM_JOB_ID}.XXXXXXXX")
export ENROOT_DATA_PATH="$TASK_TMP/enroot-data" ENROOT_CACHE_PATH="$TASK_TMP/enroot-cache"
export ENROOT_TEMP_PATH="$TASK_TMP" ENROOT_MOUNT_HOME=n NVIDIA_VISIBLE_DEVICES=void
NAME="nano35-oci-${SLURM_JOB_ID}"
mkdir -p "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH" "$(dirname "$OUTPUT")"
enroot create --name "$NAME" "$IMAGE"
ROOTFS="$ENROOT_DATA_PATH/$NAME"
[[ -f "$ROOTFS/opt/nano35-metadata/runtime-ready" ]]
[[ -x "$ROOTFS/usr/bin/umoci" && -x "$ROOTFS/usr/bin/skopeo" ]]
cp "$ROOTFS/opt/nano35-metadata/runtime-environment.json" "$TASK_TMP/runtime-environment.json"
sha256sum "$IMAGE" > "$TASK_TMP/source-sqsh.sha256"
# A single complete rootfs layer preserves installed hardlinks and absolute
# interpreter paths. Exclude temporary build state and credential locations.
tar --one-file-system --numeric-owner --owner=0 --group=0 \
    --exclude='./dev/*' --exclude='./proc/*' --exclude='./sys/*' \
    --exclude='./run/*' --exclude='./tmp/*' --exclude='./var/log/*' \
    --exclude='./root/.cache' --exclude='./root/.ssh' --exclude='./root/.aws' \
    --exclude='./root/.config' --exclude='./root/.netrc' \
    -C "$ROOTFS" -cf "$TASK_TMP/rootfs.tar" .
mkdir -p "$ROOTFS/nano35-export"
enroot start --root --mount "$TASK_TMP:/nano35-export:none:rbind,rw" "$NAME" \
    /opt/nemo_rl_venv/bin/python - "$TAG" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

tag = sys.argv[1]
root = Path('/nano35-export')
image = f'{root}/layout:{tag}'
subprocess.run(['umoci', 'init', '--layout', str(root/'layout')], check=True)
subprocess.run(['umoci', 'new', '--image', image], check=True)
subprocess.run(['umoci', 'raw', 'add-layer', '--image', image,
                str(root/'rootfs.tar')], check=True)
command = [
    'umoci', 'config', '--image', image,
    '--architecture', 'arm64', '--os', 'linux', '--config.user', 'root',
    '--config.workingdir', '/opt/nemo-rl',
    '--config.entrypoint', 'bash',
    '--config.entrypoint', '/opt/nemo-rl/docker/nano35/entrypoint.sh',
    '--config.cmd', 'bash',
    '--config.label', 'org.opencontainers.image.title=Nano 3.5 NeMo-RL SWE runtime',
]
for key, value in json.loads((root/'runtime-environment.json').read_text()).items():
    command += ['--config.env', f'{key}={value}']
subprocess.run(command, check=True)
subprocess.run(['umoci', 'gc', '--layout', str(root/'layout')], check=True)
subprocess.run(['skopeo', 'copy', f'oci:{image}',
                f'oci-archive:{root}/image.oci.tar:{tag}'], check=True)
with (root/'oci-config.json').open('w') as output:
    subprocess.run(['skopeo', 'inspect', '--config', f'oci:{image}'],
                   stdout=output, check=True)
with (root/'oci-manifest.json').open('w') as output:
    subprocess.run(['skopeo', 'inspect', '--raw', f'oci:{image}'],
                   stdout=output, check=True)
PY
cp "$TASK_TMP/image.oci.tar" "$OUTPUT.partial"
mv "$OUTPUT.partial" "$OUTPUT"
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
cp "$TASK_TMP/source-sqsh.sha256" "$OUTPUT.source-sqsh.sha256"
cp "$TASK_TMP/oci-config.json" "$OUTPUT.config.json"
cp "$TASK_TMP/oci-manifest.json" "$OUTPUT.manifest.json"
enroot remove -f "$NAME"
rm -rf -- "$TASK_TMP"
echo "Local OCI archive ready: $OUTPUT"
