#!/bin/bash

set -euo pipefail

mode=${1:-submit}
case "${mode}" in
  submit) submit_mode=(--parsable) ;;
  test-only) submit_mode=(--test-only) ;;
  *) printf 'Usage: %s [submit|test-only]\n' "$0" >&2; exit 2 ;;
esac

work_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna
experiment_root=${work_root}/experiments/hybridep-performance-recipes-20260808
repo=${VALIDATION_REPO_OVERRIDE:?VALIDATION_REPO_OVERRIDE is required}
container=${CONTAINER_OVERRIDE:-${work_root}/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh}
validation_head=${VALIDATION_HEAD_OVERRIDE:?VALIDATION_HEAD_OVERRIDE is required}
run_root=${experiment_root}/runs/recipe-tests-${validation_head:0:12}
job_reaper_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"15","reason":"other","description":"Focused NeMo-RL HybridEP recipe configuration tests"}}'

test "$(git -C "${repo}" rev-parse HEAD)" = "${validation_head}"
test -z "$(git -C "${repo}" status --porcelain --untracked-files=no)"
test -r "${container}"
mkdir -p "${run_root}/ray"

COMMAND="PYTHONPATH=${repo}:\${PYTHONPATH:-} PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/nemo_rl_venv/bin/python -m pytest -q ${repo}/tests/unit/tools/test_hybridep_default_8g_recipes.py"
export COMMAND
export CONTAINER="${container}"
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${run_root}/ray"
export GPUS_PER_NODE=8

cd "${repo}"
sbatch "${submit_mode[@]}" \
  --export=ALL \
  --nodes=1 \
  --gpus-per-node=8 \
  --exclusive \
  --account=coreai_chef_posttrain \
  --partition=batch \
  --time=00:15:00 \
  --job-name=coreai_chef_posttrain.hybridep-recipe-test \
  --output="${run_root}/slurm-%j.out" \
  --comment="${job_reaper_comment}" \
  ray.sub
