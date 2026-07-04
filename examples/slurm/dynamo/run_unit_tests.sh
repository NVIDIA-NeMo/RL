#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new}
REPO=${REPO:-${ROOT}/RL}
TEST_TOOLS=${TEST_TOOLS:-${ROOT}/cache/test-tools}
UV=${UV:-}
PYTHON=${PYTHON:-/opt/nemo_rl_venv/bin/python}

if [[ -z "${UV}" ]]; then
  UV=$(find /usr/local/bin /usr/bin /root/.local/bin -maxdepth 1 -type f -name uv -print -quit)
fi
test -x "${UV}"

"${UV}" pip install --target "${TEST_TOOLS}" pytest ruff
export PYTHONPATH="${REPO}:${TEST_TOOLS}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"

/bin/bash -n \
  docker/build-dynamo-slurm.sh \
  docker/install-dynamo-source.sh \
  examples/slurm/dynamo/assemble_container.sh \
  examples/slurm/dynamo/build_sqsh.sub \
  examples/slurm/dynamo/build_sqsh_hsg.sub \
  examples/slurm/dynamo/launch.sh \
  examples/slurm/dynamo/run_unit_tests.sh \
  examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_e2e.sh

"${PYTHON}" -m pytest -q \
  tests/unit/models/generation/test_dynamo_arguments.py \
  tests/unit/models/generation/test_dynamo_managed_runtime.py \
  tests/unit/models/generation/test_dynamo_generation.py \
  tests/unit/algorithms/test_grpo.py

"${PYTHON}" -m ruff check \
  examples/run_grpo.py \
  nemo_rl/algorithms/grpo.py \
  nemo_rl/distributed/ray_actor_environment_registry.py \
  nemo_rl/distributed/worker_groups.py \
  nemo_rl/models/generation/dynamo \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/models/generation/test_dynamo_arguments.py \
  tests/unit/models/generation/test_dynamo_generation.py \
  tests/unit/models/generation/test_dynamo_managed_runtime.py \
  tools/refit_verifier.py

"${PYTHON}" -m ruff format --check \
  nemo_rl/models/generation/dynamo/arguments.py \
  nemo_rl/models/generation/dynamo/config.py \
  nemo_rl/models/generation/dynamo/dynamo_worker.py \
  nemo_rl/models/generation/dynamo/managed_runtime.py \
  nemo_rl/models/generation/dynamo/worker_pool.py \
  tests/unit/models/generation/test_dynamo_arguments.py \
  tests/unit/models/generation/test_dynamo_managed_runtime.py \
  tools/refit_verifier.py
