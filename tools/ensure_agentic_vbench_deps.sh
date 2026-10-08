#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run inside each training container before Ray starts: verify the gym-native
# Agentic-VBench inputs (pinned benchmark checkout, Harbor runtime, Docker client,
# task backend) and the Gym/reassembler worker environments.
set -euo pipefail
venvs=${NEMO_RL_VENV_DIR:-/opt/ray_venvs}
gym_python="$venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python"
reassembler_python="$venvs/nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor/bin/python"
test -x "$gym_python" || { echo "Missing NemoGym worker venv: $gym_python" >&2; exit 1; }
"$gym_python" -c 'import aiohttp; import nemo_rl.environments.nemo_gym, nemo_rl.environments.gym_selection; import nemo_gym.token_id_capture.completion, nemo_gym.chat_streaming'
if [[ -x "$reassembler_python" ]]; then
  "$reassembler_python" -c 'import nemo_rl.experience.rollout_reassembler'
fi
: "${AGENTIC_VBENCH_ROOT:?Set AGENTIC_VBENCH_ROOT to the pinned benchmark checkout}"
: "${AGENTIC_VBENCH_HARBOR_PYTHON:?Set AGENTIC_VBENCH_HARBOR_PYTHON}"
test -d "$AGENTIC_VBENCH_ROOT/tasks/agentic_vbench_repair" || { echo "Pinned benchmark tasks missing under $AGENTIC_VBENCH_ROOT" >&2; exit 1; }
test -x "$AGENTIC_VBENCH_HARBOR_PYTHON" || { echo "Harbor Python not executable inside the container: $AGENTIC_VBENCH_HARBOR_PYTHON" >&2; exit 1; }
"$AGENTIC_VBENCH_HARBOR_PYTHON" -c 'import importlib.metadata as m, dotenv, harbor; assert m.version("harbor") == "0.6.6"'
if [[ ${AGENTIC_VBENCH_BACKEND:-remote} == remote ]]; then
  : "${AGENTIC_VBENCH_DOCKER_CLI_DIR:?Set AGENTIC_VBENCH_DOCKER_CLI_DIR}"
  : "${AGENTIC_VBENCH_BACKEND_DIR:?Set AGENTIC_VBENCH_BACKEND_DIR}"
  test -x "$AGENTIC_VBENCH_DOCKER_CLI_DIR/docker" || { echo 'Docker CLI missing' >&2; exit 1; }
  test -s "$AGENTIC_VBENCH_BACKEND_DIR/client.env" || { echo "Task backend not published: $AGENTIC_VBENCH_BACKEND_DIR/client.env" >&2; exit 1; }
  set -a; source "$AGENTIC_VBENCH_BACKEND_DIR/client.env"; set +a
  export DOCKER_CERT_PATH=$AGENTIC_VBENCH_BACKEND_DIR PATH=$AGENTIC_VBENCH_DOCKER_CLI_DIR:$PATH DOCKER_CONFIG=$AGENTIC_VBENCH_DOCKER_CLI_DIR
  docker info --format '{{.ServerVersion}}' >/dev/null
  docker compose version >/dev/null
fi
