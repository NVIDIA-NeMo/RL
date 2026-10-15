#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sourced by super35_launch.sh after tools/external_gym_vllm/pool_config.sh.
register_judge() {
  local role="$1" checkpoint="$2" replicas="$3" tp="$4" port="$5" config_key="$6"
  local served_model="${7:-model}"
  local default_python="${JUDGE_VLLM_PYTHON:-/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python}"
  local container_var="${role}_CONTAINER" python_var="${role}_VLLM_PYTHON"
  local container="${!container_var:-${JUDGE_CONTAINER:-${CONTAINER}}}"
  register_external_vllm_pool "${role}_SERVICE" \
    --display-name "${role}" --model "${checkpoint}" \
    --container "${container}" --python "${!python_var:-${default_python}}" \
    --replicas "${replicas}" --tensor-parallel-size "${tp}" \
    --served-model-name "${served_model}" --lb-port "${port}" \
    --startup-timeout 3600 --url-placeholder "__${role}_BASE_URL__"
  export "${role}_BASE_URL=__${role}_BASE_URL__" "${role}_MODEL=${served_model}" "${role}_API_KEY=EMPTY"
  JUDGE_OVERRIDES+=("++${config_key}=__${role}_BASE_URL__")
}

if [[ "${STAGE}" == student_rlvr || "${STAGE}" == rlhf_teacher ]]; then
  GENRM_CHECKPOINT="${GENRM_CHECKPOINT:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-GenRM}"
  GENRM_ARGS=(--trust-remote-code --dtype bfloat16 --kv-cache-dtype fp8
    --gpu-memory-utilization 0.95 --enable-prefix-caching --enable-expert-parallel
    --reasoning-parser nemotron_v3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":96}')
  if [[ "${STAGE}" == student_rlvr ]]; then
    GENRM_REPLICAS="${GENRM_REPLICAS:-8}"
    GENRM_LB_PORT=9213
    GENRM_ARGS+=(--max-model-len 131072 --max-num-seqs 384)
  else
    GENRM_REPLICAS="${GENRM_REPLICAS:-16}"
    GENRM_LB_PORT=9216
    GENRM_ARGS+=(--max-num-seqs 256 --speculative-config '{"method":"mtp","num_speculative_tokens":3}')
  fi
  register_judge GENRM "${GENRM_CHECKPOINT}" "${GENRM_REPLICAS}" 8 "${GENRM_LB_PORT}" \
    env.nemo_gym.genrm_model.responses_api_models.genrm_model.base_url
  external_vllm_pool_env GENRM_SERVICE FLASHINFER_WORKSPACE_BASE=/tmp \
    VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm VLLM_ALLREDUCE_USE_SYMM_MEM=0
  external_vllm_pool_args GENRM_SERVICE "${GENRM_ARGS[@]}"
fi

if [[ "${STAGE}" == student_rlvr || "${STAGE}" == rlhf_teacher ]]; then
  NL2BASH_CHECKPOINT="${NL2BASH_CHECKPOINT:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}"
  if [[ "${STAGE}" == student_rlvr ]]; then
    NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-8}"
  else
    # RLHF uses Qwen to judge the abstention rows.
    NL2BASH_REPLICAS="${NL2BASH_REPLICAS:-4}"
  fi
  register_judge NL2BASH "${NL2BASH_CHECKPOINT}" "${NL2BASH_REPLICAS}" 4 9214 \
    env.nemo_gym.nl2bash_judge_model.responses_api_models.local_vllm_model.base_url
  external_vllm_pool_env NL2BASH_SERVICE FLASHINFER_WORKSPACE_BASE=/tmp \
    VLLM_USE_FLASHINFER_MOE_FP16=0 VLLM_USE_FLASHINFER_MOE_FP8=0 \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 NCCL_MNNVL_ENABLE=1
  external_vllm_pool_args NL2BASH_SERVICE \
    --dtype bfloat16 --pipeline-parallel-size 1 --max-model-len 131072 \
    --max-num-seqs 256 --gpu-memory-utilization 0.85 \
    --enable-prefix-caching --enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --attention-backend TRITON_ATTN --enable-expert-parallel \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256]}' \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":112}'
fi

if [[ "${STAGE}" == student_rlvr ]]; then
  SAFETY_CHECKPOINT="${SAFETY_CHECKPOINT:-nvidia/Nemotron-Content-Safety-Reasoning-4B}"
  register_judge SAFETY "${SAFETY_CHECKPOINT}" "${SAFETY_REPLICAS:-2}" 4 9215 \
    env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.base_url
  external_vllm_pool_env SAFETY_SERVICE VLLM_RAY_DP_PACK_STRATEGY=strict VLLM_USE_FASTOKENS=0
  external_vllm_pool_args SAFETY_SERVICE \
    --trust-remote-code --dtype bfloat16 --max-model-len 96000 --max-num-seqs 256 \
    --gpu-memory-utilization 0.85 --attention-backend TRITON_ATTN \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16]}' \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}'
fi

if [[ "${STAGE}" == reasoning_teacher ]]; then
  REASONING_JUDGE_CHECKPOINT="${REASONING_JUDGE_CHECKPOINT:-deepseek-ai/DeepSeek-V4-Flash}"
  register_judge REASONING_JUDGE "${REASONING_JUDGE_CHECKPOINT}" \
    "${REASONING_JUDGE_REPLICAS:-4}" 4 9216 \
    env.nemo_gym.deepseek_v4_flash_judge_model.responses_api_models.inference_provider.base_url \
    deepseek-v4-flash
  # Match the bundled DSV4 topk wrapper to the compiled kernel's padding argument.
  # Each TP4 judge replica occupies one node; policy inference is unaffected.
  external_vllm_pool_env REASONING_JUDGE_SERVICE \
    "NRL_DSV4_TOPK_PADDING_COMPAT=${REASONING_JUDGE_TOPK_PADDING_COMPAT:-1}"
  external_vllm_pool_args REASONING_JUDGE_SERVICE \
    --trust-remote-code --dtype auto --tokenizer-mode deepseek_v4 \
    --kv-cache-dtype fp8 --block-size 256 --max-num-seqs 64 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 --enable-prefix-caching \
    --reasoning-parser deepseek_v4 --enable-expert-parallel
fi
