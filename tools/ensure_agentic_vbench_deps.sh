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
for runtime_python in /opt/nemo_rl_venv/bin/python \
  "$gym_python" "$reassembler_python" \
  "$venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python" \
  "$venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"; do
"$runtime_python" - <<'PY'
from importlib.metadata import distribution
import json
import orjson

# The controller validates cleanup receipts locally on failed finalization;
# checking only the Gym/reassembler actor misses this driver-runtime import.
from nemo_gym.token_id_capture.staging.records import RolloutReceipt

dist = distribution("TransferQueue")
source = json.loads(dist.read_text("direct_url.json") or "{}")
expected = "c51614308b68c8d7a87c9b3ef62d59e14c69bde2"
if source.get("vcs_info", {}).get("commit_id") != expected:
    raise RuntimeError(
        f"TransferQueue must use the repository pin {expected}; "
        f"found {dist.version} at {dist.locate_file('')}"
    )
print(f"TransferQueue pin verified: {dist.version} {expected}")
PY
done
"$gym_python" -c 'import aiohttp; import nemo_rl.environments.nemo_gym, nemo_rl.environments.gym_selection; import nemo_gym.token_id_capture.completion, nemo_gym.chat_streaming'
"$gym_python" - <<'PY'
from importlib.metadata import distribution, version

from nemo_gym.global_config import (
    _nemo_gym_openai_requirement,
    _openai_version_matches_nemo_gym_constraint,
)

constraint = _nemo_gym_openai_requirement()
parent_version = version("openai")
if constraint is None or not _openai_version_matches_nemo_gym_constraint(parent_version):
    raise RuntimeError(
        f"Gym metadata under {distribution('nemo-gym').locate_file('')} declares {constraint!r}, "
        f"but its worker runs openai=={parent_version}. Install the pinned Gym "
        "distribution alongside its source before launching training."
    )
print(f"Gym dependency metadata verified: {constraint}; openai=={parent_version}")
PY
test -x "$reassembler_python" || { echo "Missing rollout-reassembler worker venv: $reassembler_python" >&2; exit 1; }
"$reassembler_python" -c 'import nemo_rl.experience.rollout_reassembler'
: "${AGENTIC_VBENCH_ROOT:?Set AGENTIC_VBENCH_ROOT to the pinned benchmark checkout}"
: "${AGENTIC_VBENCH_HARBOR_PYTHON:?Set AGENTIC_VBENCH_HARBOR_PYTHON}"
test -d "$AGENTIC_VBENCH_ROOT/tasks/agentic_vbench_repair" || { echo "Pinned benchmark tasks missing under $AGENTIC_VBENCH_ROOT" >&2; exit 1; }
test -x "$AGENTIC_VBENCH_HARBOR_PYTHON" || { echo "Harbor Python not executable inside the container: $AGENTIC_VBENCH_HARBOR_PYTHON" >&2; exit 1; }
"$AGENTIC_VBENCH_HARBOR_PYTHON" -c 'import importlib.metadata as m, dotenv, harbor; assert m.version("harbor") == "0.6.6"'
"$AGENTIC_VBENCH_HARBOR_PYTHON" "${NRL_REPO_ROOT:-/opt/nemo-rl}/tools/patch_harbor_opencode_setup.py"
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
