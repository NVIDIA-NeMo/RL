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

"""Test-only fixed-token scoring on the existing async rollout engine."""

import asyncio
from contextlib import aclosing

import ray

from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


@ray.remote
class ScoringVllmWorker(VllmAsyncGenerationWorkerImpl):
    async def score_fixed_suffix(
        self, token_ids: list[int], *, suffix_start: int, request_id: str
    ) -> tuple[str, list[float]]:
        """Score supplied tokens through prefill, ignoring the one decoded token."""
        if self.llm is None or self.cfg["vllm_cfg"]["enable_prefix_caching"]:
            raise RuntimeError("Scoring requires a live engine with prefix caching off")
        params = self.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        result = None
        async with (
            asyncio.timeout(120),
            aclosing(
                self.llm.generate(
                    {"prompt_token_ids": token_ids}, params, request_id=request_id
                )
            ) as outputs,
        ):
            async for result in outputs:
                pass
        if (
            result is None
            or not result.finished
            or result.prompt_token_ids != token_ids
            or result.prompt_logprobs is None
            or len(result.prompt_logprobs) != len(token_ids)
            or result.num_cached_tokens not in (None, 0)
        ):
            raise RuntimeError(
                f"Incomplete, mismatched, or cached score for {request_id}"
            )
        scores = []
        for index in range(suffix_start, len(token_ids)):
            row = result.prompt_logprobs[index]
            if row is None or token_ids[index] not in row:
                raise RuntimeError(f"Missing prompt logprob at token index {index}")
            scores.append(row[token_ids[index]].logprob)
        return ray.get_runtime_context().get_actor_id(), scores
