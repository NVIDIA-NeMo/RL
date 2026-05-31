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

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    from_parallel_logits_to_same_position_logprobs,
)
from nemo_rl.models.megatron.config import MegatronModule
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor
from nemo_rl.models.policy import PolicyConfig


class JustGRPOLossPostProcessor(LossPostProcessor):
    """Megatron loss post-processor for JustGRPO leftmost-reveal logprobs."""

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int = 1,
        cp_normalize: bool = True,
        sampling_params: Optional[TrainingSamplingParams] = None,
        draft_model: Optional[MegatronModule] = None,
    ):
        super().__init__(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            cp_normalize=cp_normalize,
            sampling_params=sampling_params,
            draft_model=draft_model,
        )

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]:
        if self.cfg["sequence_packing"]["enabled"] or packed_seq_params is not None:
            raise NotImplementedError(
                "JustGRPO Megatron training currently requires sequence_packing.enabled=false"
            )
        if get_context_parallel_world_size() > 1:
            raise NotImplementedError(
                "JustGRPO Megatron training currently requires context_parallel_size=1"
            )

        def just_grpo_loss_fn(
            output_tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            tp_grp = get_tensor_model_parallel_group()
            tp_rank = get_tensor_model_parallel_rank()
            logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
            logprob_estimation_cfg = self.cfg.get("logprob_estimation", {})
            position_shift = int(logprob_estimation_cfg.get("logits_position_shift", 0))
            mask_token_id = logprob_estimation_cfg.get("mask_token_id", None)

            token_logprobs = from_parallel_logits_to_same_position_logprobs(
                output_tensor,
                target_positions=data_dict["just_grpo_target_positions"],
                target_tokens=data_dict["just_grpo_target_tokens"],
                vocab_start_index=tp_rank * output_tensor.shape[-1],
                vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                tp_group=tp_grp,
                inference_only=False,
                chunk_size=logprob_chunk_size,
                sampling_params=self.sampling_params,
                position_shift=position_shift,
                exclude_token_id=mask_token_id,
            )

            loss_mask = data_dict["just_grpo_loss_mask"]
            if need_top_k_or_top_p_filtering(self.sampling_params):
                token_logprobs = mask_out_neg_inf_logprobs(
                    token_logprobs, loss_mask, "curr_logprobs"
                )

            if need_top_k_or_top_p_filtering(self.sampling_params) and (
                hasattr(self.loss_fn, "reference_policy_kl_penalty")
                and self.loss_fn.reference_policy_kl_penalty != 0
            ):
                token_logprobs_unfiltered = (
                    from_parallel_logits_to_same_position_logprobs(
                        output_tensor,
                        target_positions=data_dict["just_grpo_target_positions"],
                        target_tokens=data_dict["just_grpo_target_tokens"],
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=False,
                        chunk_size=logprob_chunk_size,
                        sampling_params=None,
                        position_shift=position_shift,
                        exclude_token_id=mask_token_id,
                    )
                )
                data_dict["curr_logprobs_unfiltered"] = token_logprobs_unfiltered

            token_mask = torch.ones_like(loss_mask)
            if hasattr(self.loss_fn, "compute_from_aligned_tensors"):
                loss, metrics = self.loss_fn.compute_from_aligned_tensors(
                    curr_logprobs=token_logprobs,
                    token_mask=token_mask,
                    sample_mask=loss_mask,
                    advantages=data_dict["advantages"],
                    prev_logprobs=data_dict["prev_logprobs"],
                    generation_logprobs=data_dict["generation_logprobs"],
                    reference_policy_logprobs=data_dict.get(
                        "reference_policy_logprobs", None
                    ),
                    curr_logprobs_unfiltered=data_dict.get(
                        "curr_logprobs_unfiltered", None
                    ),
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                )
            else:
                loss_data = BatchedDataDict[Any](
                    {
                        "input_ids": torch.zeros(
                            token_logprobs.shape[0],
                            2,
                            device=token_logprobs.device,
                            dtype=torch.long,
                        ),
                        "token_mask": torch.stack(
                            [torch.zeros_like(loss_mask), token_mask], dim=1
                        ),
                        "sample_mask": loss_mask,
                    }
                )
                for key in (
                    "advantages",
                    "prev_logprobs",
                    "generation_logprobs",
                    "reference_policy_logprobs",
                ):
                    if key in data_dict:
                        values = data_dict[key]
                        loss_data[key] = torch.stack(
                            [torch.zeros_like(values), values], dim=1
                        )
                if "curr_logprobs_unfiltered" in data_dict:
                    loss_data["curr_logprobs_unfiltered"] = data_dict[
                        "curr_logprobs_unfiltered"
                    ].unsqueeze(1)
                loss, metrics = self.loss_fn(
                    next_token_logprobs=token_logprobs.unsqueeze(1),
                    data=loss_data,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                )
            return loss * self.num_microbatches, metrics

        return just_grpo_loss_fn


class JustGRPOLogprobsPostProcessor(LogprobsPostProcessor):
    """Megatron no-grad post-processor for flat JustGRPO reveal rows."""

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        def processor_fn_inner(output_tensor):
            tp_grp = get_tensor_model_parallel_group()
            tp_rank = get_tensor_model_parallel_rank()
            logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
            logprob_estimation_cfg = self.cfg.get("logprob_estimation", {})
            position_shift = int(logprob_estimation_cfg.get("logits_position_shift", 0))
            mask_token_id = logprob_estimation_cfg.get("mask_token_id", None)
            token_logprobs = from_parallel_logits_to_same_position_logprobs(
                output_tensor,
                target_positions=data_dict["just_grpo_target_positions"],
                target_tokens=data_dict["just_grpo_target_tokens"],
                vocab_start_index=tp_rank * output_tensor.shape[-1],
                vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                tp_group=tp_grp,
                inference_only=True,
                chunk_size=logprob_chunk_size,
                sampling_params=self.sampling_params,
                position_shift=position_shift,
                exclude_token_id=mask_token_id,
            )
            if need_top_k_or_top_p_filtering(self.sampling_params):
                token_logprobs = mask_out_neg_inf_logprobs(
                    token_logprobs, torch.ones_like(token_logprobs), "prev_logprobs"
                )
            return torch.tensor(0.0, device=token_logprobs.device), {
                "logprobs": token_logprobs
            }

        return processor_fn_inner
