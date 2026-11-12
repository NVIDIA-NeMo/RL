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

from nemo_rl.models.generation.megatron.config import (
    MCoreGenerationConfig,
    MCoreGenerationSpecificArgs,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.megatron.dynamo_config import dynamo_enabled
from nemo_rl.models.generation.megatron.megatron_generation import (
    MegatronGeneration,
)


def megatron_generation_cls(generation_config: dict) -> type[GenerationInterface]:
    """``MegatronGeneration``, or the class that serves prefill/decode engines through Dynamo.

    ``policy.generation.dynamo=true`` picks ``MegatronDynamoGeneration``, which
    owns one ``MegatronGeneration`` per prefill or decode engine instead of one
    dedicated inference policy.
    """
    if dynamo_enabled(generation_config):
        # Deferred: the plain backend never needs the Dynamo engines.
        from nemo_rl.models.generation.megatron.dynamo_generation import (
            MegatronDynamoGeneration,
        )

        return MegatronDynamoGeneration
    return MegatronGeneration


__all__ = [
    "MCoreGenerationConfig",
    "MCoreGenerationSpecificArgs",
    "MegatronGeneration",
    "megatron_generation_cls",
]
