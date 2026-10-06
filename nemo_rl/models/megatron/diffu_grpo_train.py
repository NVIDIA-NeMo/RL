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

"""Same-position logprob gathers for the DiffuGRPO ``[noisy | clean]`` layout.

Megatron's stock :class:`~nemo_rl.models.megatron.train.LogprobsPostProcessor`
gathers *next-token* logprobs: position ``i`` is scored against ``input_ids``
at ``i + 1``. The diffusion layout needs a *same-position* gather instead --
the noisy half predicts the token sitting at its own index, and the clean half
gets its autoregressive shift baked into the target tensor by the batch builder
(see :func:`nemo_rl.algorithms.hybrid_ar_diffusion.build_hybrid_ar_diffusion_batch`)
so that one gather serves both halves.

This module holds that gather. It is shared by the hybrid AR + diffusion
post-processors in :mod:`nemo_rl.models.megatron.hybrid_ar_diffusion_train`.
"""

from typing import Optional

import torch
from megatron.core.parallel_state import (
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedLogprob,
    ChunkedDistributedLogprobWithSampling,
    DistributedLogprob,
    DistributedLogprobWithSampling,
)
from nemo_rl.models.policy import PolicyConfig

__all__ = [
    "_cp_sharded_same_position_logprobs",
    "_same_position_logprobs",
]


def _same_position_logprobs(
    output_tensor: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    cfg: PolicyConfig,
    sampling_params: Optional[TrainingSamplingParams],
    inference_only: bool,
    exclude_token_id: Optional[int],
) -> torch.Tensor:
    """Gather ``log p(target_ids[:, i])`` from the logits *at* position ``i``.

    Args:
        output_tensor: ``[B, S, V/tp]`` vocab-parallel logits.
        target_ids: ``[B, S]`` token ids to score, already position-aligned.
        cfg: Policy(-like) config; only ``logprob_chunk_size`` is read.
        sampling_params: Optional top-k / top-p filtering applied before the
            log-softmax, matching the generation-time distribution.
        inference_only: Skip saving backward state.
        exclude_token_id: Drop this vocabulary column (set it to ``-inf``)
            before the log-softmax, renormalizing over the remaining tokens.
            Used to remove the MASK id, which is never a legitimate target on
            either half of the layout.

    Returns:
        ``[B, S]`` per-position log-probabilities.

    Raises:
        ValueError: If ``output_tensor`` is not 3-D or ``target_ids`` does not
            match its ``[B, S]`` prefix.
        NotImplementedError: If ``exclude_token_id`` is combined with
            ``logprob_chunk_size``.
    """
    if output_tensor.ndim != 3:
        raise ValueError(f"output_tensor must be [B, S, V], got {output_tensor.shape}")
    if target_ids.shape != output_tensor.shape[:2]:
        raise ValueError(
            f"target_ids shape {target_ids.shape} must match logits prefix "
            f"{output_tensor.shape[:2]}"
        )

    tp_group = get_tensor_model_parallel_group()
    tp_rank = get_tensor_model_parallel_rank()
    vocab_start_index = tp_rank * output_tensor.shape[-1]
    vocab_end_index = (tp_rank + 1) * output_tensor.shape[-1]
    logits = output_tensor
    local_exclude_index = None
    if (
        exclude_token_id is not None
        and vocab_start_index <= exclude_token_id < vocab_end_index
    ):
        local_exclude_index = exclude_token_id - vocab_start_index

    logprob_chunk_size = cfg.get("logprob_chunk_size", None)
    # TODO(perf): the mask-exclusion logprob path below is ~2x slower than the
    # non-exclude path (measured policy_training 165s vs 85s at DeepScaleR /
    # 16-node). It clones the full-vocab logits and calls the plain
    # DistributedLogprob once per sequence chunk, which fragments the op into
    # many small autograd nodes and retains an extra full-vocab clone through
    # the backward -- instead of the memory-efficient ChunkedDistributedLogprob
    # (which recomputes softmax per chunk in backward and never materializes a
    # whole fp32 logits tensor). Proper fix: add exclude_token_id support to
    # ChunkedDistributedLogprob (mask the per-chunk fp32 upcast in its forward
    # and backward, on the rank that owns the id) and route exclusion through it,
    # mirroring the non-exclude dispatch -- that recovers both the speed and the
    # memory bound. Until then, chunked logprobs + exclusion is unsupported:
    if exclude_token_id is not None and logprob_chunk_size is not None:
        raise NotImplementedError(
            "logprob_chunk_size (ChunkedDistributedLogprob) is not supported on "
            "the mask-exclusion logprob path. Unset logprob_chunk_size, or add "
            "exclude_token_id support to ChunkedDistributedLogprob."
        )
    target_ids = target_ids.to(device=logits.device, dtype=torch.long)
    # Branch on the GLOBAL exclude_token_id, not the per-rank local_exclude_index:
    # under TP>1 the exclude token lives in only one rank's vocab shard, so keying the
    # chunked path (one tp_group all_reduce per chunk) off local_exclude_index made TP
    # ranks issue different numbers of TENSOR_MODEL_PARALLEL_GROUP collectives and hang.
    if exclude_token_id is not None:
        seq_len = int(logits.shape[1])
        chunk_size = int(logprob_chunk_size or min(seq_len, 64))
        chunks = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            logits_chunk = logits[:, chunk_start:chunk_end, :].clone()
            if local_exclude_index is not None:
                logits_chunk[..., local_exclude_index] = -torch.inf
            target_chunk = target_ids[:, chunk_start:chunk_end]
            if need_top_k_or_top_p_filtering(sampling_params):
                chunk_logprobs = DistributedLogprobWithSampling.apply(  # type: ignore
                    logits_chunk,
                    target_chunk,
                    tp_group,
                    sampling_params.top_k,
                    sampling_params.top_p,
                    inference_only,
                )
            else:
                chunk_logprobs = DistributedLogprob.apply(  # type: ignore
                    logits_chunk,
                    target_chunk,
                    vocab_start_index,
                    vocab_end_index,
                    tp_group,
                    inference_only,
                )
            chunks.append(chunk_logprobs)
        return torch.cat(chunks, dim=1).contiguous()

    if need_top_k_or_top_p_filtering(sampling_params):
        if logprob_chunk_size is not None:
            token_logprobs: torch.Tensor = ChunkedDistributedLogprobWithSampling.apply(  # type: ignore
                logits,
                target_ids,
                tp_group,
                sampling_params.top_k,
                sampling_params.top_p,
                logprob_chunk_size,
                inference_only,
            )
        else:
            token_logprobs = DistributedLogprobWithSampling.apply(  # type: ignore
                logits,
                target_ids,
                tp_group,
                sampling_params.top_k,
                sampling_params.top_p,
                inference_only,
            )
    elif logprob_chunk_size is not None:
        token_logprobs = ChunkedDistributedLogprob.apply(  # type: ignore
            logits,
            target_ids,
            vocab_start_index,
            vocab_end_index,
            logprob_chunk_size,
            tp_group,
            inference_only,
        )
    else:
        token_logprobs = DistributedLogprob.apply(  # type: ignore
            logits,
            target_ids,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            inference_only,
        )

    return token_logprobs.contiguous()


def _cp_sharded_same_position_logprobs(
    output_tensor: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    cfg: PolicyConfig,
    sampling_params: Optional[TrainingSamplingParams],
    inference_only: bool,
    exclude_token_id: Optional[int],
) -> torch.Tensor:
    """Context-parallel-aware wrapper around :func:`_same_position_logprobs`.

    Today this is a pass-through plus a guard. The diffusion layout runs with
    sequence packing disabled, and Megatron's *unpacked* microbatch path does
    not context-parallel-shard its inputs at all (see
    ``nemo_rl/models/megatron/data.py``, where the unpacked branch sets
    ``input_ids_cp_sharded = input_ids``). Every CP rank therefore sees the full
    sequence and emits full-width logits, so slicing ``target_ids`` to ``1/cp``
    and all-gathering the result would silently score the wrong tokens -- the
    dense pad factor is already inflated by ``lcm(..., cp_size * 2)``, so the
    shapes line up and nothing would raise.

    The guard lives here as well as in the config validators
    (``_validate_hybrid_ar_diffusion_setup`` and ``_validate_diffusion_support``)
    because this function is also reachable from a hand-built post-processor in
    tests.

    Raises:
        NotImplementedError: If the context-parallel world size exceeds 1.
    """
    cp_size = get_context_parallel_world_size()
    if cp_size > 1:
        raise NotImplementedError(
            "The hybrid AR + diffusion logprob path requires "
            "policy.megatron_cfg.context_parallel_size=1 (got "
            f"{cp_size}): Megatron's unpacked microbatch path does not "
            "CP-shard its inputs, so re-gathering CP-sharded logprobs would "
            "read the wrong tokens."
        )
    return _same_position_logprobs(
        output_tensor,
        target_ids,
        cfg=cfg,
        sampling_params=sampling_params,
        inference_only=inference_only,
        exclude_token_id=exclude_token_id,
    )
