#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal bootstrap for the MX path on the DGD worker side.
#
# After the nemo-rl-mx image rebuild (which bakes nixl-cu12 + modelexpress +
# wandb + protobuf-6 into every venv) and the receiver-side polling switch
# (which dropped the trainer→worker RPC and therefore the ai-dynamo-runtime
# trainer dep), nothing actually needs installing at container start.
#
# The only remaining job is PYTHONPATH override for hot-reload of pure-Python
# edits to `dynamo.vllm.*` from Lustre — useful on the DGD worker side so
# changes to `dynamo.vllm.mx_refit.*`, `handlers.py`, `worker_factory.py`,
# or `main.py` take effect after a pod restart without rebuilding the dynamo
# image.
#
# SOURCE me, don't exec — we mutate the calling shell's PYTHONPATH.
#
#   source /mnt/rl-workspace/$USER/nemo-rl/infra/nrl_k8s/dynamo_mx/bootstrap_mx.sh
#   exec python3 -m dynamo.vllm "$@"
#
# Environment:
#   MX_DEV_USER  — owner namespace for the Lustre staging paths.
#                  Defaults to $USER; override when nrl-k8s renders the
#                  manifest for a different user.

_mx_user="${MX_DEV_USER:-${USER:-$(whoami)}}"
_mx_lustre_root="/mnt/rl-workspace/${_mx_user}"
LUSTRE_DYNAMO_DEV="${_mx_lustre_root}/dynamo-dev"
LUSTRE_MX="${_mx_lustre_root}/modelexpress"

echo "[mx-bootstrap] user=${_mx_user}" >&2

# Lustre-staged dynamo Python tree (`dev_sync.sh` syncs from the dev pod's
# local checkout). Prepending it to PYTHONPATH shadows `dynamo.vllm.*`
# from our local code while keeping `dynamo._core` loading from the
# installed Rust binding (dynamo is a PEP 420 namespace package).
if [ -d "${LUSTRE_DYNAMO_DEV}/dynamo" ]; then
  export PYTHONPATH="${LUSTRE_DYNAMO_DEV}:${PYTHONPATH:-}"
  echo "[mx-bootstrap] PYTHONPATH+= ${LUSTRE_DYNAMO_DEV} (dynamo dev override)" >&2
fi

# Lustre-staged modelexpress checkout. The dynamo image doesn't bake
# modelexpress in (verified 2026-05-13 — without this override the worker
# crashes with `ModuleNotFoundError: No module named 'modelexpress'`).
if [ -d "${LUSTRE_MX}/modelexpress_client/python" ]; then
  export PYTHONPATH="${LUSTRE_MX}/modelexpress_client/python:${PYTHONPATH:-}"
  echo "[mx-bootstrap] PYTHONPATH+= ${LUSTRE_MX}/modelexpress_client/python (modelexpress dev override)" >&2
fi

# Workaround for the dynamo operator hardcoding
# NIXL_PLUGIN_DIR=$VENV/site-packages/nixl_cu12.libs/nixl (auditwheel-renamed
# layout). In the current ucx-1.21.x image build, `nixl_cu12==1.1.0` got
# installed without auditwheel running for cu12 — so that path doesn't exist.
# Plugins ended up at `.nixl_cu12.mesonpy.libs/plugins/` instead. We probe
# both locations and pick whichever exists; export at shell level so this
# beats the container env injection at exec time.
_nixl_site="/opt/dynamo/venv/lib/python3.12/site-packages"
for _cand in \
  "${_nixl_site}/nixl_cu12.libs/nixl" \
  "${_nixl_site}/.nixl_cu12.mesonpy.libs/plugins" \
  "${_nixl_site}/nixl_cu13.libs/nixl" \
  "${_nixl_site}/.nixl_cu13.mesonpy.libs/plugins"; do
  if [ -f "${_cand}/libplugin_UCX.so" ]; then
    export NIXL_PLUGIN_DIR="${_cand}"
    echo "[mx-bootstrap] NIXL_PLUGIN_DIR=${_cand} (override; operator default may point elsewhere)" >&2
    break
  fi
done

