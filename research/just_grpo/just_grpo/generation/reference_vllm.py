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
"""Reference diffusion sampling on NeMo-RL's native vLLM worker."""

from collections.abc import AsyncIterator
from typing import Any

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


class _ReferenceDiffusionSampling:
    """Keep native lifecycle/refit; specialize the reference fork's diffusion API."""

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.config.diffusion import DiffusionConfig

        if "leftmost" not in str(
            DiffusionConfig.__annotations__.get("selection_policy")
        ):
            raise RuntimeError(
                "Reference vLLM must support leftmost diffusion sampling"
            )
        super()._create_engine(llm_kwargs)

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        stop_strings: list[str] | None,
        max_new_tokens: int | None = None,
    ) -> Any:
        params = super()._build_sampling_params(
            greedy=greedy, stop_strings=stop_strings, max_new_tokens=max_new_tokens
        )
        # The reference fork uses 1 as a sentinel for engine-level temperature.
        # logprobs=0 returns sampled-token scores at commitment; top-k scores
        # instead describe the final canvas and cannot be used for GRPO.
        params.temperature = 0.0 if greedy else 1.0
        params.logprobs = 0
        params.top_k = -1
        return params

    def _validate_context(self, data: BatchedDataDict[GenerationDatumSpec]) -> None:
        diffusion = self.cfg["vllm_kwargs"]["diffusion_config"]
        if len(data["input_lengths"]) and (
            int(data["input_lengths"].max())
            + self.cfg["max_new_tokens"]
            + diffusion["canvas_length"]
            > self.cfg["vllm_cfg"]["max_model_len"]
        ):
            raise ValueError(
                "Prompt plus response canvas exceeds generation context length"
            )


class ReferenceVllmWorkerImpl(_ReferenceDiffusionSampling, VllmGenerationWorkerImpl):
    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        self._validate_context(data)
        return super().generate(data, greedy=greedy)


class ReferenceVllmAsyncWorkerImpl(
    _ReferenceDiffusionSampling, VllmAsyncGenerationWorkerImpl
):
    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncIterator[tuple[int, BatchedDataDict[GenerationOutputSpec]]]:
        self._validate_context(data)
        async for result in super().generate_async(data, greedy=greedy):
            yield result


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class ReferenceVllmWorker(ReferenceVllmWorkerImpl):
    pass


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")}
)
class ReferenceVllmAsyncWorker(ReferenceVllmAsyncWorkerImpl):
    pass
