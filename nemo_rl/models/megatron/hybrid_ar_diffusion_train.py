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

"""Megatron loss / logprob post-processors for hybrid AR + diffusion training."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams

from nemo_rl.algorithms.hybrid_ar_diffusion import get_hybrid_ar_diffusion_cfg
from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.diffu_grpo_train import (
    _cp_sharded_same_position_logprobs,
)
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor
from nemo_rl.models.policy import PolicyConfig

__all__ = [
    "HybridARDiffusionLogprobsPostProcessor",
    "HybridARDiffusionLossPostProcessor",
]


def _exclude_token_id(cfg: PolicyConfig) -> Optional[int]:
    """Return the vocabulary column to drop before the log-softmax, if any.

    The MASK id is excluded from the scored logits on *both* halves. The noisy
    half's target is never MASK (it is the true response token), and the clean
    (AR) half can never legitimately target it either, so applying the exclusion
    uniformly keeps the training logprobs consistent with ``prev_logprobs``,
    which take this same path.

    ``get_hybrid_ar_diffusion_cfg`` is used rather than a raw ``cfg[...]`` read
    because ``policy.logprob_estimation`` arrives as a plain dict on some paths
    (hand-built configs, unit tests) and as an already-validated
    ``HybridARDiffusionLogprobEstimationConfig`` on others (pydantic coerces it
    when ``MasterConfig`` validates). ``model_validate`` accepts both.
    """
    estimation_cfg = get_hybrid_ar_diffusion_cfg(cfg)
    if not estimation_cfg.exclude_mask_token_from_logits:
        return None
    return estimation_cfg.mask_token_id


class HybridARDiffusionLossPostProcessor(LossPostProcessor):
    """Score both halves of the ``[noisy | clean]`` layout with one gather.

    ``hybrid_target_ids`` already encodes each half's alignment -- the noisy half
    stores the token at its own position, the clean half stores the *next* clean
    token -- so a single same-position gather produces the ``[N, T]`` logprob
    vector both loss terms read. Unlike Megatron's stock ``LossPostProcessor``
    the result is *not* a next-token gather against ``input_ids``, and it is not
    truncated to either half: the clean half carries the policy-gradient term
    and the noisy half the cross-entropy term.
    """

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int = 1,
        cp_normalize: bool = True,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        super().__init__(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            cp_normalize=cp_normalize,
            sampling_params=sampling_params,
            draft_model=None,
        )
        # Validate (and cache) once per microbatch iterator rather than once per
        # forward: model_validate is not free and the closure below runs on every
        # microbatch.
        self._exclude_token_id = _exclude_token_id(cfg)

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]:
        """Build the per-microbatch loss closure.

        Args:
            data_dict: The hybrid training microbatch.
            packed_seq_params: Must be ``None`` -- sequence packing would break
                the fixed ``[noisy | clean]`` split the attention metadata and
                both loss masks depend on.
            global_valid_seqs: Global valid sequence count.
            global_valid_toks: Global valid (response) token count; normalizes
                both loss terms.

        Returns:
            A callable mapping the model output tensor to ``(loss, metrics)``.

        Raises:
            NotImplementedError: If sequence packing is enabled.
        """
        if self.cfg["sequence_packing"]["enabled"] or packed_seq_params is not None:
            raise NotImplementedError(
                "Hybrid AR+diffusion Megatron training requires "
                "sequence_packing.enabled=false"
            )

        def loss_fn_inner(
            output_tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            token_logprobs = _cp_sharded_same_position_logprobs(
                output_tensor,
                data_dict["hybrid_target_ids"],
                cfg=self.cfg,
                sampling_params=self.sampling_params,
                inference_only=False,
                exclude_token_id=self._exclude_token_id,
            )
            loss, metrics = self.loss_fn(
                token_logprobs,
                data_dict,
                global_valid_seqs,
                global_valid_toks,
            )
            # Megatron's forward-backward divides each microbatch loss by
            # num_microbatches. The loss is already globally normalized (both
            # terms divide by global_valid_toks), so that extra division has to
            # be cancelled or gradients come out num_microbatches times too
            # small. Every sibling post-processor does the same.
            return loss * self.num_microbatches, metrics

        return loss_fn_inner


class HybridARDiffusionLogprobsPostProcessor(LogprobsPostProcessor):
    """No-grad post-processor producing full-sequence hybrid logprobs.

    Differs from Megatron's stock ``LogprobsPostProcessor`` in that it gathers
    against ``hybrid_target_ids`` at the same position (which carries the clean
    half's next-token shift) instead of against ``input_ids`` shifted by one,
    and it emits the full ``[N, T]`` layout width so the clean-half
    autoregressive logprobs survive for the caller to unscatter.
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        sampling_params: Optional[TrainingSamplingParams] = None,
        use_fused_linear_logprobs: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            sampling_params=sampling_params,
            use_fused_linear_logprobs=use_fused_linear_logprobs,
        )
        if use_fused_linear_logprobs:
            raise NotImplementedError(
                "Hybrid AR+diffusion logprobs are incompatible with "
                "megatron_cfg.use_fused_linear_logprobs: the fused forward "
                "gathers at the next input token, not at hybrid_target_ids."
            )
        self._exclude_token_id = _exclude_token_id(cfg)

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        original_seq_length: int,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Build the per-microbatch logprob closure.

        Args:
            data_dict: The hybrid microbatch.
            input_ids: Unused; kept for interface compatibility. The stock
                processor scores against this; the hybrid scores against
                ``hybrid_target_ids`` instead.
            cu_seqlens_padded: Unused; sequence packing is not supported.
            original_seq_length: Unused; kept for interface compatibility
                because ``forward_with_post_processing_fn`` passes it by
                keyword. The ``[noisy | clean]`` layout defines its own width,
                and the caller (``_finalize_logprobs_from_outputs``) maps it
                back to the rollout width using the layout's own
                ``diffu_grpo_clean_lengths``, not this value.

        Returns:
            A callable mapping the model output tensor to
            ``(zero_loss, {"logprobs": [N, T]})``.
        """
        del input_ids, cu_seqlens_padded, original_seq_length

        def processor_fn_inner(output_tensor):
            token_logprobs = _cp_sharded_same_position_logprobs(
                output_tensor,
                data_dict["hybrid_target_ids"],
                cfg=self.cfg,
                sampling_params=self.sampling_params,
                inference_only=True,
                exclude_token_id=self._exclude_token_id,
            )
            return torch.tensor(0.0, device=token_logprobs.device), {
                "logprobs": token_logprobs
            }

        return processor_fn_inner
