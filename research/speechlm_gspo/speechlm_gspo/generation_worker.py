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

import ray

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)  # pragma: no cover
class SpeechLMVllmGenerationWorker(VllmGenerationWorkerImpl):
    """Generic vLLM worker in an environment that ships the Speech vLLM plugin.

    The recipe enables the plugin with ``VLLM_PLUGINS=nemo_speechlm``.
    """

    def __init__(self, *args, **kwargs):
        from nemo.collections.speechlm2.vllm.salm.config import NeMoSpeechLMConfig
        from transformers import AutoConfig

        # Ray deserializes the actor in a different process from its initializer.
        # Register before the generic worker probes AutoConfig or loads vLLM.
        AutoConfig.register(
            NeMoSpeechLMConfig.model_type, NeMoSpeechLMConfig, exist_ok=True
        )
        super().__init__(*args, **kwargs)
