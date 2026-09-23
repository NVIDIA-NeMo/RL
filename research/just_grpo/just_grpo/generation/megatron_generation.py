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
"""In-process diffusion generation using the live Megatron policy model."""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from itertools import count
from typing import TYPE_CHECKING, Any

import ray
import torch

from just_grpo.config import DiffusionSamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration

if TYPE_CHECKING:
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.models.policy.lm_policy import Policy


class MegatronDiffusionGeneration(MegatronGeneration):
    """Reuse native Policy dispatch and colocated Megatron weight synchronization."""

    uses_native_refit = True

    def __init__(self, *, policy: "Policy", config: "MasterConfig") -> None:
        # Preserve native Megatron identity/lifecycle for upstream async GRPO.
        # HTTP serving is disabled, so this does not initialize an AR engine.
        super().__init__(config=config.policy, tokenizer=None, policy=policy)
        self.policy = policy
        self.batch_size = config.policy["logprob_batch_size"]
        # Collector threads share this counter; unlike an asyncio lock it is
        # independent of their event loops and can be serialized into Ray.
        self._next_replica = count()

    def generate(
        self, data: BatchedDataDict[Any], greedy: bool = False
    ) -> BatchedDataDict[Any]:
        return self.policy.generate(data, greedy=greedy)

    async def generate_async(
        self, data: BatchedDataDict[Any], greedy: bool = False
    ) -> AsyncIterator[tuple[int, BatchedDataDict[Any]]]:
        """Stream indexed rows from TP=1 DP workers without blocking the event loop.

        Dispatch inference microbatches directly: Policy.generate requires
        complete DP shards, whereas async rollouts can request a single row.
        Drain submitted work on failure or cancellation before collection can
        report completion and allow the shared policy to enter training.
        """
        group = self.policy.worker_group

        async def generate_chunk(start: int) -> tuple[int, BatchedDataDict[Any]]:
            replica = next(self._next_replica) % self.policy.data_parallel_size
            result = await group.run_single_worker_single_data(
                "generate",
                worker_idx=group.get_dp_leader_worker_idx(replica),
                data=data.slice(start, min(start + self.batch_size, data.size)),
                greedy=greedy,
            )
            return start, result

        tasks = [
            asyncio.create_task(generate_chunk(start))
            for start in range(0, data.size, self.batch_size)
        ]
        try:
            for completed in asyncio.as_completed(tasks):
                start, result = await completed
                for row in range(result.size):
                    yield start + row, result.slice(row, row + 1)
        finally:
            # as_completed does not cancel the underlying Ray work. Waiting
            # here makes the collector's pending-rollout drain authoritative.
            await asyncio.gather(*tasks, return_exceptions=True)

    def blocks_training(self) -> bool:
        return True

    def wake_carries_weight_updates(self) -> bool:
        return True

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        self.policy.prepare_for_lp_inference()
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        # No engine, cache, or separate weights to release. The policy owns
        # the next transition to reference scoring or training.
        return True

    def shutdown(self) -> bool:
        return True

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        raise NotImplementedError("Diffusion generation shares the colocated policy")


def sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return samples and logprobs from the same temperature-scaled distribution."""
    logprobs = (
        logits.float() / temperature if temperature > 0 else logits.float()
    ).log_softmax(-1)
    tokens = (
        torch.multinomial(
            logprobs.exp().reshape(-1, logits.shape[-1]), 1, generator=generator
        ).reshape(logits.shape[:-1])
        if temperature > 0
        else logits.argmax(-1)
    )
    return tokens, logprobs.gather(-1, tokens[..., None]).squeeze(-1)


@torch.no_grad()
def sample_batch(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prompts: torch.Tensor,
    *,
    sampling: DiffusionSamplingParams,
    max_new_tokens: int,
    max_sequence_length: int,
    mask_token_id: int,
    stop_token_ids: list[int],
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    """Decode aligned prompts using the training model's doubled block mask.

    Each forward receives fixed-width [noisy | clean] inputs and repeated
    positions. The current block is identical in both halves; noisy queries
    can only read their own noisy block and earlier clean blocks. Unknown
    future blocks remain MASK. Returned scores are measured at commitment.
    """
    block = sampling.block_size
    if prompts.ndim != 2 or prompts.shape[1] == 0 or prompts.shape[1] % block:
        raise ValueError("Require nonempty block-aligned prompts")
    if max_new_tokens <= 0 or max_new_tokens % block:
        raise ValueError("Generation length must be positive and block-aligned")
    if (
        max_sequence_length % block
        or prompts.shape[1] + max_new_tokens > max_sequence_length
    ):
        raise ValueError("Prompt plus response canvas exceeds aligned context length")
    if sampling.selection_policy != "leftmost" or block % sampling.max_steps:
        raise ValueError("Megatron generation requires a divisible leftmost schedule")
    batch, prefix = prompts.shape
    tokens = torch.full(
        (batch, max_sequence_length),
        mask_token_id,
        device=prompts.device,
        dtype=torch.long,
    )
    tokens[:, :prefix] = prompts
    positions = torch.arange(max_sequence_length, device=prompts.device)[
        None
    ].expand_as(tokens)
    positions = positions.repeat(1, 2)
    scores = torch.zeros(
        (batch, max_new_tokens), device=prompts.device, dtype=torch.float32
    )
    finished = torch.zeros(batch, device=prompts.device, dtype=torch.bool)
    lengths = torch.full(
        (batch,), max_new_tokens, device=prompts.device, dtype=torch.long
    )
    stops = torch.tensor(stop_token_ids, device=prompts.device, dtype=torch.long)
    reveal_width = block // sampling.max_steps
    for offset in range(0, max_new_tokens, reveal_width):
        start, end = prefix + offset, prefix + offset + reveal_width
        # The attention implementation owns the asymmetric block mask. Keep
        # its fixed width even when the current prefix is short.
        logits = forward(torch.cat([tokens, tokens], dim=1), positions)
        candidates, logprobs = sample_tokens(
            logits[:, start:end], temperature=sampling.temperature, generator=generator
        )
        del logits
        active = ~finished
        tokens[:, start:end] = torch.where(
            active[:, None], candidates, tokens[:, start:end]
        )
        scores[:, offset : offset + reveal_width] = torch.where(
            active[:, None], logprobs, 0.0
        )
        hits = torch.isin(candidates, stops) & active[:, None]
        ended = hits.any(-1)
        first_stop = hits.long().argmax(-1)
        lengths = torch.where(ended, offset + first_stop + 1, lengths)
        finished |= ended
        if finished.all():
            break
    return [
        {
            "token_ids": row[prefix : prefix + length].tolist(),
            "logprobs": row_scores[:length].tolist(),
            "finish_reason": "stop" if done else "length",
        }
        for row, row_scores, length, done in zip(
            tokens, scores, lengths.tolist(), finished.tolist()
        )
    ]


@torch.no_grad()
def generate_responses(
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prompts: list[list[int]],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    sampling: DiffusionSamplingParams,
    max_new_tokens: int,
    max_sequence_length: int,
    mask_token_id: int,
    stop_token_ids: list[int],
) -> list[dict[str, Any]]:
    """Batch equal-length prompts and restore upstream request order."""
    if not prompts or batch_size < 1:
        raise ValueError("Require prompts and a positive inference batch size")
    groups = defaultdict(list)
    for index, prompt in enumerate(prompts):
        groups[len(prompt)].append(index)
    responses = [{} for _ in prompts]
    generator = torch.Generator(device=device).manual_seed(seed)
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = torch.tensor(
                [prompts[i] for i in selected], device=device, dtype=torch.long
            )
            outputs = sample_batch(
                forward,
                batch,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                max_sequence_length=max_sequence_length,
                mask_token_id=mask_token_id,
                stop_token_ids=stop_token_ids,
                generator=generator,
            )
            for index, output in zip(selected, outputs):
                responses[index] = output
    return responses


def pack_responses(
    prompts: list[list[int]],
    responses: list[dict[str, Any]],
    *,
    pad_token_id: int,
    max_new_tokens: int,
    stop_token_ids: list[int],
) -> BatchedDataDict[Any]:
    """Return the right-padded GenerationOutputSpec used by upstream rollouts."""
    lengths = [len(p) + len(r["token_ids"]) for p, r in zip(prompts, responses)]
    width = max(lengths)
    ids = torch.full((len(prompts), width), pad_token_id, dtype=torch.long)
    logprobs = torch.zeros_like(ids, dtype=torch.float32)
    generated = []
    truncated = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        tokens, scores = response["token_ids"], response["logprobs"]
        if not tokens or len(tokens) != len(scores):
            raise ValueError("Empty response or missing sampled-token logprobs")
        ids[i, : lengths[i]] = torch.tensor(prompt + tokens)
        logprobs[i, len(prompt) : lengths[i]] = torch.tensor(scores)
        generated.append(len(tokens))
        truncated.append(
            len(tokens) >= max_new_tokens and tokens[-1] not in stop_token_ids
        )
    return BatchedDataDict(
        output_ids=ids,
        logprobs=logprobs,
        generation_lengths=torch.tensor(generated),
        unpadded_sequence_lengths=torch.tensor(lengths),
        truncated=torch.tensor(truncated),
    )
