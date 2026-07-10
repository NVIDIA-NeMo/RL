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
  examples/slurm/dynamo/validate_hsg_image.sub \
  examples/slurm/dynamo/validate_hsg_image.sh \
  examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_e2e.sh \
  examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_r2_wandb.sh
test -s docker/patches/vllm-0.23.0-layerwise-reload-composed-loader.patch

DRY_RUN_SECRET=wandb-secret-must-not-leak
DRY_RUN_OUTPUT=$(\
  DRY_RUN=1 \
  WANDB_API_KEY="${DRY_RUN_SECRET}" \
  EXP_NAME=nano-v3-5-swe-dynamo-r2-wandb-dry-run \
  bash examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_r2_wandb.sh
)
for expected in \
  '--nodes=6' \
  '--account=coreai_tritoninference_triton3' \
  '--partition=batch' \
  'cluster.num_nodes=6' \
  'grpo.num_prompts_per_step=2' \
  'policy.train_global_batch_size=4' \
  'policy.generation.colocated.resources.num_nodes=2' \
  'logger.wandb_enabled=True' \
  'grpo.max_num_steps=4'; do
  grep -F -- "${expected}" <<<"${DRY_RUN_OUTPUT}"
done
if grep -F -- "${DRY_RUN_SECRET}" <<<"${DRY_RUN_OUTPUT}"; then
  echo 'R2 launcher exposed WANDB_API_KEY in dry-run output.' >&2
  exit 1
fi

"${PYTHON}" -m pytest -q \
  tests/unit/models/generation/test_dynamo_arguments.py \
  tests/unit/models/generation/test_dynamo_managed_runtime.py \
  tests/unit/models/generation/test_dynamo_generation.py \
  tests/unit/models/generation/test_dynamo_token_wrapper.py \
  tests/unit/models/generation/test_swe_dynamo_r2_config.py \
  tests/unit/utils/test_prefix_reuse.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/algorithms/test_async_utils.py::TestAsyncTrajectoryCollector::test_dynamo_prepare_for_refit_drains_pending_generations \
  tests/unit/tools/test_refit_verifier.py

"${PYTHON}" -m ruff check \
  examples/run_grpo.py \
  nemo_rl/algorithms/grpo.py \
  nemo_rl/distributed/ray_actor_environment_registry.py \
  nemo_rl/distributed/worker_groups.py \
  nemo_rl/models/generation/dynamo \
  nemo_rl/utils/prefix_reuse.py \
  tests/unit/algorithms/test_grpo.py \
  tests/unit/algorithms/test_async_utils.py \
  tests/unit/models/generation/test_dynamo_arguments.py \
  tests/unit/models/generation/test_dynamo_generation.py \
  tests/unit/models/generation/test_dynamo_managed_runtime.py \
  tests/unit/models/generation/test_dynamo_token_wrapper.py \
  tests/unit/models/generation/test_swe_dynamo_r2_config.py \
  tests/unit/tools/test_refit_verifier.py \
  tests/unit/utils/test_prefix_reuse.py \
  tools/refit_verifier.py

"${PYTHON}" -m ruff format --check \
  nemo_rl/models/generation/dynamo/arguments.py \
  nemo_rl/models/generation/dynamo/config.py \
  nemo_rl/models/generation/dynamo/dynamo_generation.py \
  nemo_rl/models/generation/dynamo/dynamo_worker.py \
  nemo_rl/models/generation/dynamo/managed_runtime.py \
  nemo_rl/models/generation/dynamo/token_wrapper.py \
  nemo_rl/models/generation/dynamo/worker_pool.py \
  nemo_rl/utils/prefix_reuse.py \
  tests/unit/models/generation/test_dynamo_arguments.py \
  tests/unit/models/generation/test_dynamo_managed_runtime.py \
  tests/unit/models/generation/test_dynamo_generation.py \
  tests/unit/models/generation/test_dynamo_token_wrapper.py \
  tests/unit/models/generation/test_swe_dynamo_r2_config.py \
  tests/unit/tools/test_refit_verifier.py \
  tests/unit/utils/test_prefix_reuse.py \
  tests/unit/algorithms/test_async_utils.py \
  tools/refit_verifier.py
