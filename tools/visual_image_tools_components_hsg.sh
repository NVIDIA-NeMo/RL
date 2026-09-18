#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
CONTAINER=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yifuw/images/striped/rl-gym.67009223-gym_ln_fix.sqsh
export NEMO_RL_SOURCE_OVERRIDE_ROOT=/opt/nemo-rl
export NEMO_GYM_EXTRA_ROOTS="/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym:$RUN_DIR/assets"
export PYTHONPATH="$OVERLAY_DIR/packages:/opt/nemo-rl/tools/runtime_source_override:/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export MPLCONFIGDIR="$RUN_DIR/matplotlib" HF_HOME="$RUN_DIR/huggingface" XDG_CACHE_HOME="$RUN_DIR/cache"
export IMAGE_TOOLS_OUTPUT_DIR="$RUN_DIR/image-crops"
srun --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=16 \
  --container-image="$CONTAINER" --no-container-mount-home \
  --container-mounts="$PROJECT_ROOT:/opt/nemo-rl:ro,$RUN_DIR:$RUN_DIR,$OVERLAY_DIR:$OVERLAY_DIR:ro,$GAMES_VENV_ROOT:$GAMES_VENV_ROOT:ro,$GAMES_TRAIN_MANIFEST:$GAMES_TRAIN_MANIFEST:ro,$GAMES_EVAL_MANIFEST:$GAMES_EVAL_MANIFEST:ro,$IMAGE_TRAIN_MANIFEST:$IMAGE_TRAIN_MANIFEST:ro,$VISGYM_ASSET_ARCHIVE:$VISGYM_ASSET_ARCHIVE:ro" \
  --container-workdir=/opt/nemo-rl/3rdparty/Gym-workspace/Gym \
  /bin/bash -c '
set -euo pipefail
mkdir -p "$RUN_DIR/assets/resources_servers/visgym/data" "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME"
/opt/nemo_rl_venv/bin/python - <<"PY"
import hashlib, os, tarfile
from pathlib import Path
archive = Path(os.environ["VISGYM_ASSET_ARCHIVE"])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == "468d4f7b54e1cc6ba5aad76287b15319c83bccad569b8ca5ec14ebab95855a21"
with tarfile.open(archive) as stream:
    stream.extractall(Path(os.environ["RUN_DIR"]) / "assets/resources_servers/visgym/data", filter="data")
PY
(
cd /opt/nemo-rl
/opt/nemo_rl_venv/bin/python -m pytest /opt/nemo-rl/tests/unit/experience/test_rollouts.py \
  -k postprocess_nemo_gym_group_reports_per_agent_live_metrics \
  --confcutdir=/opt/nemo-rl/tests/unit/experience -q -o addopts= \
  -o cache_dir="$RUN_DIR/pytest-rl"
) > "$RUN_DIR/rl-metric-test.log" 2>&1 &
metric_pid=$!
"$GAMES_VENV_ROOT/responses_api_agents/gymv_agent/.venv/bin/python" -m pytest \
  responses_api_models/vllm_model/tests/test_context_rejection.py \
  responses_api_agents/visgym_agent/tests responses_api_agents/gymv_agent/tests \
  --confcutdir=responses_api_models --import-mode=importlib -q -o addopts= \
  -o cache_dir="$RUN_DIR/pytest-agents" > "$RUN_DIR/agent-guard-tests.log" 2>&1 &
agent_pid=$!
/opt/nemo_rl_venv/bin/python -m tools.check_visual_image_tools_components &
components_pid=$!
result=0
for pid in "$metric_pid" "$agent_pid" "$components_pid"; do
  wait "$pid" || result=1
done
exit "$result"
'
