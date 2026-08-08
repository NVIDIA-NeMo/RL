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
container=${CONTAINER_OVERRIDE:-${work_root}/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh}
test -n "${VALIDATION_HEAD_OVERRIDE:-}"
run_root=${experiment_root}/runs/hybridep-prepadding-${VALIDATION_HEAD_OVERRIDE:0:12}
job_reaper_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"30","reason":"other","description":"Focused NeMo-RL HybridEP pre-padding tests"}}'
policy_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python

test "$(git -C "${repo}" rev-parse HEAD)" = "${VALIDATION_HEAD_OVERRIDE}"
test -r "${container}"
mkdir -p "${run_root}/ray"

COMMAND="PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ${policy_python} -m pytest --mcore-only -q \
  ${repo}/tests/unit/models/megatron/test_megatron_data.py::test_hybridep_prepads_packed_inputs_before_model_forward \
  ${repo}/tests/unit/models/megatron/test_megatron_data.py::test_hybridep_padding_mask_preserves_existing_cp_local_layout \
  ${repo}/tests/unit/models/megatron/test_megatron_data.py::test_hybridep_padding_mask_rejects_model_owned_cp_slicing \
  ${repo}/tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_sequence_packing_without_opt_in_keeps_dispatch_padding \
  ${repo}/tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_sequence_packing_explicitly_uses_input_prepadding \
  ${repo}/tests/unit/models/megatron/test_megatron_setup.py::TestApplyMoeConfig::test_hybridep_input_prepadding_rejects_unsupported_layouts"
export COMMAND
export CONTAINER="${container}"
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${run_root}/ray"
export GPUS_PER_NODE=8
export PYTHONPATH="${repo}:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"

cd "${repo}"
sbatch "${submit_mode[@]}" \
  --export=ALL \
  --nodes=1 \
  --gpus-per-node=8 \
  --exclusive \
  --account=coreai_chef_posttrain \
  --partition=batch \
  --time=00:20:00 \
  --job-name=coreai_chef_posttrain.hybridep-prepadding-test \
  --output="${run_root}/slurm-%j.out" \
  --comment="${job_reaper_comment}" \
  ray.sub
