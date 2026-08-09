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
repo=${experiment_root}/RL
mcore=${MCORE_SOURCE_OVERRIDE:-${experiment_root}/Megatron-LM-padding-mask-test}
container=${CONTAINER_OVERRIDE:-${work_root}/containers/nemo-rl-nightly-cw-fallback-20260808/nemo_rl_nightly_20260805_15171871.sqsh}
test -n "${MCORE_EXPECTED_COMMIT:-}"
run_root=${experiment_root}/runs/mcore-padding-mask-${MCORE_EXPECTED_COMMIT:0:12}
job_reaper_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"30","reason":"other","description":"Focused Megatron-LM eight-GPU unit test"}}'

test "$(git -C "${mcore}" rev-parse HEAD)" = "${MCORE_EXPECTED_COMMIT}"
test -r "${container}"
mkdir -p "${run_root}/ray"

COMMAND="/opt/nemo_rl_venv/bin/torchrun --standalone --nproc-per-node=8 -m pytest ${mcore}/tests/unit_tests/transformer/moe/test_routers.py -q -k 'test_expert_bias_ignores_padding_tokens or test_expert_bias_token_counts_with_padding_mask or test_router_with_padding_mask or test_padding_mask_preserves_routes_outside_dropless_hybridep'"
export COMMAND
export CONTAINER="${container}"
export MOUNTS=/lustre:/lustre
export BASE_LOG_DIR="${run_root}/ray"
export GPUS_PER_NODE=8
export PYTHONPATH="${mcore}"

cd "${repo}"
sbatch "${submit_mode[@]}" \
  --export=ALL \
  --nodes=1 \
  --gpus-per-node=8 \
  --exclusive \
  --account=coreai_chef_posttrain \
  --partition=batch \
  --time=00:20:00 \
  --job-name=coreai_chef_posttrain.mcore-padding-mask-test \
  --output="${run_root}/slurm-%j.out" \
  --comment="${job_reaper_comment}" \
  ray.sub
