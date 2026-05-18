#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Sync the local dynamo Python checkout to a Lustre path that DGD worker pods
# mount, so worker pods can pick up Python-only dynamo changes via PYTHONPATH
# without rebuilding the container image.
#
# Layout (assumes the standard rl-workspace conventions):
#   SRC: /mnt/rl-workspace/<user>/dynamo/components/src/dynamo
#   DST: /mnt/rl-workspace/<user>/dynamo-dev/dynamo
#
# The DGD manifests in this directory set:
#   PYTHONPATH=/mnt/rl-workspace/<user>/dynamo-dev:$PYTHONPATH
#
# After running this script:
#   * Edit Python files under SRC on your dev pod.
#   * Re-run this script.
#   * `kubectl delete pod -n default -l <dgd-worker-selector>` to recycle
#     workers (the operator reconciles them back; ~30s).

set -euo pipefail

USER_NAME="${USER:-$(whoami)}"
DEFAULT_SRC="/mnt/rl-workspace/${USER_NAME}/dynamo/components/src/dynamo"
DEFAULT_DST_PARENT="/mnt/rl-workspace/${USER_NAME}/dynamo-dev"

SRC="${DYNAMO_DEV_SRC:-$DEFAULT_SRC}"
DST_PARENT="${DYNAMO_DEV_DST:-$DEFAULT_DST_PARENT}"
DST="${DST_PARENT}/dynamo"

if [ ! -d "$SRC" ]; then
  echo "ERROR: source dynamo Python tree not found: $SRC" >&2
  echo "  Set DYNAMO_DEV_SRC to override, or check out ai-dynamo/dynamo at" >&2
  echo "  /mnt/rl-workspace/${USER_NAME}/dynamo." >&2
  exit 1
fi

mkdir -p "$DST_PARENT"

echo "[dev_sync] SRC=$SRC"
echo "[dev_sync] DST=$DST"

# --delete so a removed file disappears in the dev tree too. Exclude caches
# and *.pyc so the worker pod doesn't load stale bytecode from a previous
# image. --update-only would keep stale files; we want to mirror.
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '*.pyo' \
  --exclude '.pytest_cache' \
  --exclude '*.egg-info' \
  "$SRC/" "$DST/"

echo "[dev_sync] done"
echo
echo "To pick up the change, recycle the affected worker pods:"
echo "  kubectl delete pod -n default -l app.kubernetes.io/managed-by=dynamo-operator"
echo "  # or, more targeted: -l nvidia.com/dynamo-graph-deployment-name=<your-dgd>"
