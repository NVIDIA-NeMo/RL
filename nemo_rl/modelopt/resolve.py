# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

POLICY_WORKER_OVERRIDES = {
    "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker.MegatronQuantPolicyWorker",
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker.DTensorQuantPolicyWorker",
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": "nemo_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker_v2.DTensorQuantPolicyWorkerV2",
}

GENERATION_WORKER_OVERRIDES = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker",
}


def resolve_policy_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized policy worker class if quant_cfg is set, otherwise the default.

    Safe to call even when ModelOpt is not installed — returns *default_cls* unchanged.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return POLICY_WORKER_OVERRIDES.get(default_cls, default_cls)


def resolve_generation_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized generation worker class if quant_cfg is set, otherwise the default.

    Safe to call even when ModelOpt is not installed — returns *default_cls* unchanged.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return GENERATION_WORKER_OVERRIDES.get(default_cls, default_cls)
