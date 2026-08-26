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
"""Block-wise iterative denoising for masked diffusion language models.

Every function here is pure tensor math parameterized by a ``logits_fn``
callable, so the sampler can be exercised on CPU with a stub model. The Ray
worker that supplies the real forward pass lives in ``dllm_worker.py``.

The schedule follows LLaDA (https://github.com/ML-GSAI/LLaDA) as used by GDPO
(https://arxiv.org/abs/2510.08554): the generation region is split into blocks
that are denoised left to right, and within a block each step unmasks a fixed
budget of the highest-confidence positions.
"""

from typing import Callable, Optional

import torch

from nemo_rl.algorithms.logits_sampling_utils import apply_top_k_top_p

# Callable returning ``[B, L, V]`` logits for ``(input_ids, attention_mask)``.
LogitsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Guards ``log(-log(u))`` against ``u`` landing on exactly 0 or 1.
_UNIFORM_EPS = 1e-20


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Splits each row's masked positions into a per-step unmasking budget.

    The budgets sum to the row's masked count, so a block is fully denoised
    after ``steps`` steps. Remainders go to the earliest steps.

    Args:
        mask_index: Boolean ``[B, L]`` tensor marking still-masked positions.
        steps: Number of denoising steps to spread the positions over.

    Returns:
        An int64 ``[B, steps]`` tensor of per-step unmasking counts.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = base.expand(-1, steps).clone()
    indices = torch.arange(steps, device=mask_index.device)
    num_transfer_tokens += (indices.unsqueeze(0) < remainder).to(num_transfer_tokens)

    return num_transfer_tokens.to(torch.int64)


def build_canvas(
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    gen_length: int,
    mask_id: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds the left-padded denoising canvas from a right-padded prompt batch.

    Diffusion decoding writes into a fixed-width region that must start at the
    same offset for every row, so the right-padded prompts of
    :class:`~nemo_rl.models.generation.interfaces.GenerationDatumSpec` are
    re-aligned against the right edge of the prompt region. The padding that
    moves to the left is excluded by the returned attention mask rather than
    being attended as ordinary text.

    Args:
        input_ids: Right-padded ``[B, P]`` prompt token ids.
        input_lengths: ``[B]`` unpadded prompt lengths.
        gen_length: Width of the region to denoise.
        mask_id: Token id of ``[MASK]``.
        pad_id: Token id to write into the inert left padding.

    Returns:
        A tuple of the ``[B, P + gen_length]`` canvas and its matching
        ``[B, P + gen_length]`` attention mask.
    """
    batch_size, prompt_width = input_ids.shape
    device = input_ids.device

    offsets = prompt_width - input_lengths.to(device).view(batch_size, 1)
    positions = torch.arange(prompt_width, device=device).view(1, prompt_width)
    # Row b holds its prompt at [offset_b, prompt_width); gather from the
    # right-padded source with the offset removed.
    source = (positions - offsets).clamp_min(0)
    prompt = torch.gather(input_ids, 1, source)
    is_prompt = positions >= offsets
    prompt = torch.where(is_prompt, prompt, torch.full_like(prompt, pad_id))

    generation = torch.full(
        (batch_size, gen_length), mask_id, dtype=input_ids.dtype, device=device
    )
    canvas = torch.cat([prompt, generation], dim=1)

    attention_mask = torch.cat(
        [is_prompt, torch.ones_like(generation, dtype=torch.bool)], dim=1
    )
    return canvas, attention_mask


def _sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Draws one token per position by Gumbel-max, or argmax at temperature 0.

    LLaDA samples ``argmax(exp(logits) / (-log u) ** temperature)`` in float64.
    This is the algebraically identical log-space form,
    ``argmax(logits + temperature * Gumbel)``, which needs no widening because
    it never exponentiates the logits.

    Args:
        logits: ``[B, L, V]`` logits, already top-k/top-p filtered.
        temperature: Sampling temperature. 0.0 selects greedy decoding.
        generator: Optional RNG for reproducible draws.

    Returns:
        An int64 ``[B, L]`` tensor of sampled token ids.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    uniform = torch.rand(
        logits.shape, dtype=torch.float32, device=logits.device, generator=generator
    ).clamp_(_UNIFORM_EPS, 1.0 - _UNIFORM_EPS)
    gumbel = -torch.log(-torch.log(uniform))
    return (logits.float() + temperature * gumbel).argmax(dim=-1)


def _select_by_rank(
    confidence: torch.Tensor, num_transfer: torch.Tensor
) -> torch.Tensor:
    """Marks each row's ``num_transfer`` highest-confidence positions.

    Equivalent to a per-row ``topk`` with a row-dependent ``k``, but without a
    Python loop over the batch.

    Uses the same sort/rank/scatter shape as
    ``nemo_automodel.components.datasets.dllm.corruption._batched_gumbel_topk``.
    That helper is not reused because it always adds Gumbel noise to the
    scores: remasking has to pick the genuinely most confident positions, and
    perturbing them would change which tokens get committed.

    Args:
        confidence: ``[B, L]`` scores, ``-inf`` where a position is ineligible.
        num_transfer: ``[B]`` per-row counts of positions to select.

    Returns:
        A boolean ``[B, L]`` selection mask.
    """
    order = confidence.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        1, order, torch.arange(order.shape[1], device=order.device).expand_as(order)
    )
    return ranks < num_transfer.view(-1, 1)


@torch.no_grad()
def block_denoise(
    logits_fn: LogitsFn,
    canvas: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    gen_start: int,
    mask_id: int,
    steps: int,
    block_length: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    cfg_scale: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Denoises the masked region of a canvas block by block, left to right.

    Args:
        logits_fn: Callable mapping ``(input_ids, attention_mask)`` to
            ``[B, L, V]`` logits.
        canvas: ``[B, gen_start + gen_length]`` canvas from :func:`build_canvas`.
        attention_mask: Matching ``[B, L]`` attention mask.
        gen_start: Index where the region to denoise begins.
        mask_id: Token id of ``[MASK]``.
        steps: Total denoising steps, split evenly across blocks.
        block_length: Width of each block. Must divide the generation width.
        temperature: Sampling temperature. 0.0 selects greedy decoding.
        top_k: Optional top-k filter applied before sampling.
        top_p: Optional top-p filter applied before sampling.
        cfg_scale: Unsupervised classifier-free guidance scale. Values above 0
            double the number of forward passes.
        generator: Optional RNG for reproducible draws.

    Returns:
        The fully denoised ``[B, L]`` canvas. The input is not modified.

    Raises:
        ValueError: If ``block_length`` does not divide the generation width.
    """
    gen_length = canvas.shape[1] - gen_start
    if gen_length % block_length != 0:
        raise ValueError(
            f"block_length={block_length} must divide the generation width "
            f"{gen_length}."
        )

    canvas = canvas.clone()
    num_blocks = gen_length // block_length
    steps_per_block = max(1, steps // num_blocks)
    is_prompt = (
        torch.arange(canvas.shape[1], device=canvas.device) < gen_start
    ).expand_as(canvas)

    for block in range(num_blocks):
        start_idx = gen_start + block * block_length
        end_idx = start_idx + block_length

        block_mask = canvas[:, start_idx:end_idx] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            mask_index = canvas == mask_id

            if cfg_scale > 0.0:
                unconditional = torch.where(
                    is_prompt, torch.full_like(canvas, mask_id), canvas
                )
                logits = logits_fn(
                    torch.cat([canvas, unconditional], dim=0),
                    torch.cat([attention_mask, attention_mask], dim=0),
                )
                conditional, unconditional_logits = torch.chunk(logits, 2, dim=0)
                logits = unconditional_logits + (cfg_scale + 1) * (
                    conditional - unconditional_logits
                )
            else:
                logits = logits_fn(canvas, attention_mask)

            logits, _ = apply_top_k_top_p(logits, top_k, top_p)

            sampled = _sample_tokens(
                logits, temperature=temperature, generator=generator
            )
            # Confidence reuses the filtered logits so a token the filter
            # excluded can never win a transfer slot.
            probs = torch.softmax(logits.float(), dim=-1)
            confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            # Positions past the current block stay masked until their block.
            confidence[:, end_idx:] = -torch.inf
            confidence = torch.where(mask_index, confidence, -torch.inf)

            transfer = _select_by_rank(confidence, num_transfer_tokens[:, step])
            transfer &= mask_index
            canvas = torch.where(transfer, sampled.to(canvas.dtype), canvas)

    return canvas


def unpack_generations(
    canvas: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    gen_start: int,
    eos_token_ids: list[int],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Repacks a denoised canvas into right-padded generation outputs.

    Diffusion fills the whole generation region regardless of where the answer
    ends, so the response length is recovered after the fact by scanning for
    the first stop token. Tokens past it are dropped.

    Args:
        canvas: Denoised ``[B, gen_start + gen_length]`` canvas.
        input_lengths: ``[B]`` unpadded prompt lengths.
        gen_start: Index where the generated region begins.
        eos_token_ids: Token ids that terminate a response. The stop token
            itself is kept in the output.
        pad_id: Token id to right-pad the outputs with.

    Returns:
        A dict with ``output_ids``, ``generation_lengths``,
        ``unpadded_sequence_lengths`` and ``truncated``, matching the fields of
        :class:`~nemo_rl.models.generation.interfaces.GenerationOutputSpec`.
    """
    batch_size, width = canvas.shape
    device = canvas.device
    gen_length = width - gen_start
    input_lengths = input_lengths.to(device)

    generated = canvas[:, gen_start:]
    positions = torch.arange(gen_length, device=device).view(1, gen_length)

    is_stop = torch.zeros_like(generated, dtype=torch.bool)
    for token_id in eos_token_ids:
        is_stop |= generated == token_id

    # First stop index per row, or gen_length when the row has none.
    first_stop = torch.where(
        is_stop, positions.expand_as(generated), torch.full_like(generated, gen_length)
    ).min(dim=1)[0]
    truncated = ~is_stop.any(dim=1)
    generation_lengths = torch.where(truncated, first_stop, first_stop + 1)

    unpadded_sequence_lengths = input_lengths + generation_lengths

    # Row b becomes prompt[:len_b] + generated[:gen_len_b], right padded.
    out_positions = torch.arange(width, device=device).view(1, width)
    prompt_offsets = gen_start - input_lengths.view(batch_size, 1)
    # Read prompts from their left-aligned canvas slots and generations from
    # the start of the generated region, both shifted to a common origin.
    source = torch.where(
        out_positions < input_lengths.view(batch_size, 1),
        out_positions + prompt_offsets,
        out_positions - input_lengths.view(batch_size, 1) + gen_start,
    ).clamp(0, width - 1)
    output_ids = torch.gather(canvas, 1, source)
    output_ids = torch.where(
        out_positions < unpadded_sequence_lengths.view(batch_size, 1),
        output_ids,
        torch.full_like(output_ids, pad_id),
    )

    return {
        "output_ids": output_ids,
        "generation_lengths": generation_lengths,
        "unpadded_sequence_lengths": unpadded_sequence_lengths,
        "truncated": truncated,
    }
