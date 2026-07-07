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

"""Megatron loss post-processor for block-aware ESPO."""

from typing import Any, Dict, Optional, Tuple

import torch

from nemo_rl.algorithms.espo_logprobs import (
    compute_coupled_block_aware_elbo,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.diffu_grpo_train import DiffuGRPOLossPostProcessor


class ESPOLossPostProcessor(DiffuGRPOLossPostProcessor):
    """Megatron loss post-processor for block-aware ESPO.

    Overrides the aligned-loss reduction with the ESPO block-aware ELBO: each
    per-token ``[N, noisy_len]`` logprob row is reduced to a per-sequence
    block-aware ELBO scalar (each block reweighted by its realized masking ratio,
    then divided by the total response length L) and the ``[N_seq]`` scalars are
    fed to the loss (Route B): the GSPO sequence-ratio path gives the ESPO ratio
    and the k2 KL a genuine 0.5*(Lnorm_theta - Lnorm_ref)^2 per sequence. The
    mask(s) (shared seed) are identical for curr/prev/ref, so the ratio is valid.
    (k2 is on the NORMALIZED ELBO Lnorm; the paper uses the raw L_hat -- differs
    by a 1/L^2 constant, absorbed into beta.)

    ESPO is the antithetic coupled pair (num_mc_samples == 2). Training is
    SAMPLE-MAJOR: this microbatch holds K whole sequences with their two level
    rows grouped (level 0 = mask M, level 1 = complement Mbar). We reshape to
    ``[K, M]``, reduce each level with its own harvest mask, and average the two
    ELBOs (scheme (b)) BEFORE forming the per-sequence ratio. The objective is
    separable across sequences, so the gradient accumulates over the remaining
    microbatches (standard NeMo-RL path).
    """

    def _aligned_loss(
        self,
        *,
        token_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        sample_mask: torch.Tensor,
        advantages: torch.Tensor,
        prev_logprobs: torch.Tensor,
        generation_logprobs: torch.Tensor,
        reference_policy_logprobs: Optional[torch.Tensor],
        curr_logprobs_unfiltered: Optional[torch.Tensor],
        gen_kl_logprobs: Optional[torch.Tensor],
        noisy_length: int,
        data_dict: BatchedDataDict[Any],
        global_valid_seqs: Optional[torch.Tensor],
        global_valid_toks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logprob_estimation_cfg = self.cfg.get("logprob_estimation", {})
        response_lengths = data_dict["diffu_grpo_response_lengths"]
        noisy_response_offset = int(
            data_dict["diffu_grpo_noisy_response_offsets"][0].item()
        )
        # block_size from the estimator config (defaults to the model's
        # block_size used to build the noisy layout; 32 if unset).
        block_size = int(logprob_estimation_cfg.get("block_size", 32) or 32)
        num_masks = int(logprob_estimation_cfg.get("num_mc_samples", 2) or 2)

        # SAMPLE-MAJOR rows: this microbatch holds K whole sequences, each
        # with its num_masks level rows contiguous ([s0L0, s0L1, s1L0, ...]).
        # Recover the layout with a reshape [K*M, ...] -> [K, M, ...] (K is
        # the OUTER axis, level inner); level j is x[:, j]. K = rows / M.
        k = loss_mask.shape[0] // num_masks
        loss_mask_r = loss_mask.reshape(k, num_masks, -1)
        response_lengths_r = response_lengths.reshape(k, num_masks)
        advantages_r = advantages.reshape(k, num_masks, -1)
        sample_mask_r = sample_mask.reshape(k, num_masks)

        # Validate (read-only) that the row order is sample-major: each
        # K-group's levels are [0, 1, ..., M-1], and a group's M rows share
        # one block_reveal_sample_index (constant across the LEVEL axis).
        # Cheap guard -- catches any future schedule / microbatch-order
        # change without paying for an argsort.
        levels_r = data_dict["coupled_grpo_level"].reshape(k, num_masks)
        sample_index_r = data_dict["block_reveal_sample_index"].reshape(
            k, num_masks
        )
        expected_levels = torch.arange(
            num_masks, device=levels_r.device, dtype=levels_r.dtype
        ).unsqueeze(0)
        if not torch.equal(levels_r, expected_levels.expand_as(levels_r)):
            raise ValueError(
                "ESPO expects sample-major rows (each group's levels "
                f"[0..M-1]); got coupled_grpo_level {levels_r.tolist()}"
            )
        if num_masks > 1 and not bool(
            (sample_index_r == sample_index_r[:, :1]).all()
        ):
            raise ValueError(
                "ESPO expects the M level rows of each sequence to share one "
                f"block_reveal_sample_index; got {sample_index_r.tolist()}"
            )

        def _espo_lnorm(per_row_logprobs):
            x = per_row_logprobs.reshape(k, num_masks, -1)
            level_lps = [x[:, j] for j in range(num_masks)]
            level_harvests = [loss_mask_r[:, j] for j in range(num_masks)]
            _, lnorm = compute_coupled_block_aware_elbo(
                level_lps,
                level_harvests,
                response_lengths_r[:, 0],
                noisy_response_offset,
                block_size,
            )
            return lnorm

        curr_lnorm = _espo_lnorm(token_logprobs)
        prev_lnorm = _espo_lnorm(prev_logprobs)
        gen_lnorm = _espo_lnorm(generation_logprobs)
        ref_lnorm = (
            _espo_lnorm(reference_policy_logprobs)
            if reference_policy_logprobs is not None
            else None
        )
        # Per-sequence advantage / sample_mask: advantages are GRPO-broadcast
        # over the response, so the value at the first response position of a
        # sequence's level-0 row is A_n; sample_mask is per row, dedup to one
        # per sequence (each group's level-0 row).
        advantages_per_seq = advantages_r[:, 0, noisy_response_offset]
        sample_mask_per_seq = sample_mask_r[:, 0]
        espo_token_mask = torch.ones_like(curr_lnorm)
        loss, metrics = self.loss_fn.compute_from_aligned_tensors(
            curr_logprobs=curr_lnorm,
            token_mask=espo_token_mask,
            sample_mask=sample_mask_per_seq,
            advantages=advantages_per_seq,
            prev_logprobs=prev_lnorm,
            generation_logprobs=gen_lnorm,
            reference_policy_logprobs=ref_lnorm,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )
        return loss * self.num_microbatches, metrics
