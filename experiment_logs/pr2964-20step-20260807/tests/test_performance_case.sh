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

submit_script=$(<"${experiment_dir}/submit_performance_20step.sh")
assert_contains "${submit_script}" '--comment=${job_reaper_comment}'
assert_contains "${submit_script}" '"exemptIdleTimeMins":"90"'
assert_contains "${submit_script}" 'model initialization and colocated vLLM startup'

printf 'performance-case-tests-pass\n'
