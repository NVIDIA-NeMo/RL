#!/bin/bash

set -euo pipefail

experiment_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${experiment_dir}/performance_case.sh"

assert_contains() {
  local haystack=$1
  local needle=$2
  [[ "${haystack}" == *"${needle}"* ]] || {
    printf 'expected %q in %q\n' "${needle}" "${haystack}" >&2
    exit 1
  }
}

assert_not_contains() {
  local haystack=$1
  local needle=$2
  [[ "${haystack}" != *"${needle}"* ]] || {
    printf 'did not expect %q in %q\n' "${needle}" "${haystack}" >&2
    exit 1
  }
}

render_case qwen3-30ba3b baseline /tmp/q30-baseline
q30_baseline=$(printf '%s\n' "${driver_args[@]}")
[[ "${num_nodes}" == 4 ]]
[[ "${segment_size}" == 4 ]]
assert_contains "${q30_baseline}" 'grpo-qwen3-30ba3b-4n8g.yaml'
assert_contains "${q30_baseline}" 'policy.megatron_cfg.moe_token_dispatcher_type=alltoall'
assert_contains "${q30_baseline}" 'logger.tensorboard_enabled=true'
assert_not_contains "${q30_baseline}" 'moe_flex_dispatcher_backend=hybridep'
assert_not_contains "${q30_baseline}" 'moe_hybridep_prepad_packed_inputs=true'

render_case qwen3-30ba3b hybridep /tmp/q30-hybridep
q30_hybridep=$(printf '%s\n' "${driver_args[@]}")
assert_contains "${q30_hybridep}" '++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true'

render_case qwen3-235b hybridep /tmp/q235-hybridep
q235_hybridep=$(printf '%s\n' "${driver_args[@]}")
[[ "${num_nodes}" == 16 ]]
[[ -z "${segment_size}" ]]
assert_contains "${q235_hybridep}" 'grpo-qwen3-235b-16n8g.yaml'
assert_contains "${q235_hybridep}" 'policy.megatron_cfg.moe_token_dispatcher_type=flex'
assert_contains "${q235_hybridep}" '++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep'
assert_contains "${q235_hybridep}" "++policy.megatron_cfg.env_vars.NVLINK_DOMAIN_SIZE='8'"
assert_contains "${q235_hybridep}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${q235_hybridep}" '++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false'
assert_not_contains "${q235_hybridep}" 'logger.tensorboard_enabled=true'
assert_not_contains "${q235_hybridep}" 'moe_hybridep_prepad_packed_inputs=true'

render_case qwen3-235b baseline /tmp/q235-baseline
q235_baseline=$(printf '%s\n' "${driver_args[@]}")
assert_contains "${q235_baseline}" 'policy.megatron_cfg.moe_token_dispatcher_type=alltoall'
assert_contains "${q235_baseline}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${q235_baseline}" '++policy.generation.vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms=false'

render_case nemotron3-super baseline /tmp/super-baseline
super_baseline=$(printf '%s\n' "${driver_args[@]}")
[[ "${num_nodes}" == 32 ]]
assert_contains "${super_baseline}" 'grpo-nemotron3-super-120BA12B-32n8g.yaml'
assert_contains "${super_baseline}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${super_baseline}" '++policy.generation.vllm_kwargs.moe_backend=triton'
assert_contains "${super_baseline}" 'policy.generation.vllm_cfg.enforce_eager=true'
assert_contains "${super_baseline}" 'logger.tensorboard_enabled=true'
assert_not_contains "${super_baseline}" 'moe_flex_dispatcher_backend=hybridep'

render_case nemotron3-super hybridep /tmp/super-hybridep
super_hybridep=$(printf '%s\n' "${driver_args[@]}")
assert_contains "${super_hybridep}" '++policy.generation.vllm_kwargs.disable_custom_all_reduce=true'
assert_contains "${super_hybridep}" '++policy.generation.vllm_kwargs.moe_backend=triton'
assert_contains "${super_hybridep}" 'policy.generation.vllm_cfg.enforce_eager=true'
assert_contains "${super_hybridep}" '++policy.megatron_cfg.moe_flex_dispatcher_backend=hybridep'
assert_contains "${super_hybridep}" '++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true'

submit_script=$(<"${experiment_dir}/submit_performance_20step.sh")
assert_contains "${submit_script}" '--comment=${job_reaper_comment}'
assert_contains "${submit_script}" '"exemptIdleTimeMins":"90"'
assert_contains "${submit_script}" 'model initialization and colocated vLLM startup'
assert_contains "${submit_script}" 'nemotron3-super) time_limit=08:00:00'
assert_contains "${submit_script}" 'repo=${VALIDATION_REPO_OVERRIDE:-${work_root}/experiments/pr2964-20step-20260807/RL}'
assert_contains "${submit_script}" 'validation_head=${VALIDATION_HEAD_OVERRIDE:-a028b33bcde0ef8aeb9fcc626a2e0c57fb568d2f}'
assert_contains "${submit_script}" 'test "$(git -C "${repo}" rev-parse HEAD)" = "${validation_head}"'
assert_contains "${submit_script}" 'mcore_source=${MCORE_SOURCE_OVERRIDE:-${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM}'
assert_contains "${submit_script}" 'mcore_commit=${MCORE_EXPECTED_COMMIT_OVERRIDE:-$(git -C "${mcore_source}" rev-parse HEAD)}'
assert_contains "${submit_script}" 'hybridep_dependency_ancestor=${HYBRIDEP_DEPENDENCY_ANCESTOR_OVERRIDE:-a9aaa395c37963a9fd8a7320d61a516c7b714e57}'
assert_contains "${submit_script}" 'test "$(git -C "${mcore_source}" rev-parse HEAD)" = "${mcore_commit}"'
assert_contains "${submit_script}" 'git -C "${repo}" merge-base --is-ancestor "${hybridep_dependency_ancestor}" HEAD'
assert_contains "${submit_script}" 'PYTHONPATH="${overlay}:${repo}:${repo}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${mcore_source}"'
assert_contains "${submit_script}" "import megatron.core; print(megatron.core.__file__)"
assert_contains "${submit_script}" 'mcore_source=%s\nmcore_commit=%s\n'

printf 'performance-case-tests-pass\n'
