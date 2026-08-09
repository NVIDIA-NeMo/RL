#!/bin/bash

set -euo pipefail

mode=${1:-submit}
case "${mode}" in
  submit) submit_mode=(--parsable) ;;
  test-only) submit_mode=(--test-only) ;;
  *) printf 'Usage: %s [submit|test-only]\n' "$0" >&2; exit 2 ;;
esac

work_root=/lustre/fsw/portfolios/coreai/projects/coreai_chef_posttrain/users/sna
experiment_root=${work_root}/experiments/pr2964-20step-20260807
repo=${VALIDATION_REPO_OVERRIDE:-${experiment_root}/RL}
bridge_source=${BRIDGE_SOURCE_OVERRIDE:-${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}
mcore_source=${MCORE_SOURCE_OVERRIDE:-${bridge_source}/3rdparty/Megatron-LM}
container=${CONTAINER_OVERRIDE:-${work_root}/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh}
test_gpus_per_node=${TEST_GPUS_PER_NODE:-8}
test -n "${VALIDATION_HEAD_OVERRIDE:-}"
[[ "${test_gpus_per_node}" =~ ^[1-8]$ ]]
run_root=${experiment_root}/runs/hybridep-prepadding-${VALIDATION_HEAD_OVERRIDE:0:12}
job_reaper_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"30","reason":"other","description":"Focused NeMo-RL HybridEP pre-padding tests"}}'

test "$(git -C "${repo}" rev-parse HEAD)" = "${VALIDATION_HEAD_OVERRIDE}"
test -d "${bridge_source}/src/megatron/bridge"
test -d "${mcore_source}/megatron/core"
test -r "${container}"
mkdir -p "${run_root}/ray"

COMMAND="PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/nemo_rl_venv/bin/python -m pytest --mcore-only -q \
  ${repo}/tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_input_prepadding_wins_after_bridge_validation \
  ${repo}/tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_dispatch_padding_stays_enabled_without_input_prepadding \
  ${repo}/tests/unit/models/megatron/test_megatron_data.py::test_hybridep_prepads_packed_inputs_before_model_forward \
  ${repo}/tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_sequence_packing_explicitly_uses_input_prepadding"
export COMMAND
export CONTAINER="${container}"
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${run_root}/ray"
export GPUS_PER_NODE="${test_gpus_per_node}"
export PYTHONPATH="${repo}:${bridge_source}/src:${mcore_source}"

cd "${repo}"
slurm_args=(
  "${submit_mode[@]}"
  --export=ALL
  --nodes=1
  --gpus-per-node="${test_gpus_per_node}"
  --account=coreai_chef_posttrain
  --partition=batch
  --time=00:20:00
  --job-name=coreai_chef_posttrain.hybridep-prepadding-test
  --output="${run_root}/slurm-%j.out"
  --comment="${job_reaper_comment}"
)
if [[ "${test_gpus_per_node}" -eq 8 ]]; then
  slurm_args+=(--exclusive)
  sbatch "${slurm_args[@]}" ray.sub
else
  printf -v wrapped_command '%q ' \
    srun \
    --nodes=1 \
    --ntasks=1 \
    --no-container-mount-home \
    --container-mounts="${MOUNTS}" \
    --container-image="${CONTAINER}" \
    --container-workdir="${repo}" \
    bash -lc "${COMMAND}"
  sbatch "${slurm_args[@]}" --wrap="${wrapped_command}"
fi
