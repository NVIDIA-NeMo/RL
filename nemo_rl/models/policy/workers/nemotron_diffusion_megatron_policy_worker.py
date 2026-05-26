# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from contextlib import contextmanager

import ray

from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl


class NemotronDiffusionMegatronPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    @contextmanager
    def use_nemotron_causal_attention_forward(self):
        """Run NemotronLabsDiffusionAttention through its causal inference path."""
        attention_modules = [
            module
            for module in self.model.modules()
            if hasattr(module, "set_inference_mode")
            and hasattr(module, "set_inference_params")
        ]
        if not attention_modules:
            yield
            return

        saved_states = []
        for module in attention_modules:
            saved_states.append(
                (
                    module,
                    getattr(module, "_inference_mode", False),
                    getattr(module, "_inference_causal", True),
                    getattr(module, "_cache_enabled", False),
                )
            )
            if hasattr(module, "clear_kv_cache"):
                module.clear_kv_cache()
            module.set_inference_params(causal=True, cache_enabled=False)
            module.set_inference_mode(True)

        try:
            yield
        finally:
            for module, inference_mode, inference_causal, cache_enabled in saved_states:
                module.set_inference_params(
                    causal=inference_causal, cache_enabled=cache_enabled
                )
                module.set_inference_mode(inference_mode)
                if hasattr(module, "clear_kv_cache"):
                    module.clear_kv_cache()

    def train(self, *args, **kwargs):
        with self.use_nemotron_causal_attention_forward():
            return super().train(*args, **kwargs)

    def get_logprobs(self, *args, **kwargs):
        with self.use_nemotron_causal_attention_forward():
            return super().get_logprobs(*args, **kwargs)

    def get_topk_logits(self, *args, **kwargs):
        with self.use_nemotron_causal_attention_forward():
            return super().get_topk_logits(*args, **kwargs)


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker(
        "nemotron_diffusion_megatron_policy_worker"
    )
)
class NemotronDiffusionMegatronPolicyWorker(
    NemotronDiffusionMegatronPolicyWorkerImpl
):
    pass
