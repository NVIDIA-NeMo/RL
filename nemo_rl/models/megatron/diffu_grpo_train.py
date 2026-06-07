# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import os
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
from nemo_rl.algorithms.utils import calculate_kl
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    ChunkedDistributedLogprob,
    ChunkedDistributedLogprobWithSampling,
    DistributedLogprob,
    DistributedLogprobWithSampling,
)
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor
from nemo_rl.models.policy import PolicyConfig


def _debug_print_diffu_grpo_logprob_alignment(
    *,
    curr_logprobs: torch.Tensor,
    generation_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_mask: torch.Tensor,
    response_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    target_ids: torch.Tensor,
    block_size: int,
    kl_type: str,
) -> None:
    if os.environ.get("DIFFUGRPO_DEBUG_LOGPROB_ALIGNMENT") != "1":
        return
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
    if rank != 0:
        return

    with torch.no_grad():
        mask = (loss_mask * sample_mask.unsqueeze(-1)).bool()
        lp_delta = curr_logprobs - generation_logprobs
        abs_delta = lp_delta.abs()
        gen_kl = calculate_kl(
            logprobs=generation_logprobs,
            logprobs_reference=curr_logprobs,
            kl_type=kl_type,
            input_clamp_value=None,
            output_clamp_value=None,
        )
        valid_abs = abs_delta[mask]
        valid_kl = gen_kl[mask]
        payload: dict[str, Any] = {
            "tag": "DIFFUGRPO_LOGPROB_ALIGNMENT",
            "batch_size": int(curr_logprobs.shape[0]),
            "seq_len": int(curr_logprobs.shape[1]),
            "valid_tokens": int(mask.sum().item()),
            "response_lengths": [
                int(x) for x in response_lengths.detach().cpu().tolist()
            ],
            "prompt_lengths": [int(x) for x in prompt_lengths.detach().cpu().tolist()],
            "mean_abs_delta": float(valid_abs.mean().item())
            if valid_abs.numel()
            else 0.0,
            "max_abs_delta": float(valid_abs.max().item())
            if valid_abs.numel()
            else 0.0,
            "mean_gen_kl": float(valid_kl.mean().item()) if valid_kl.numel() else 0.0,
            "blocks": [],
        }
        for block_start in range(0, curr_logprobs.shape[1], block_size):
            block_end = min(block_start + block_size, curr_logprobs.shape[1])
            block_mask = mask[:, block_start:block_end]
            if not block_mask.any():
                continue
            block_abs = abs_delta[:, block_start:block_end][block_mask]
            block_kl = gen_kl[:, block_start:block_end][block_mask]
            payload["blocks"].append(
                {
                    "start": block_start,
                    "end": block_end,
                    "tokens": int(block_mask.sum().item()),
                    "mean_abs_delta": float(block_abs.mean().item()),
                    "max_abs_delta": float(block_abs.max().item()),
                    "mean_gen_kl": float(block_kl.mean().item()),
                }
            )
        if os.environ.get("DIFFUGRPO_DEBUG_LOGPROB_ALIGNMENT_DETAILS") == "1":
            flat_abs = abs_delta.masked_fill(~mask, -1).reshape(-1)
            k = min(10, int(mask.sum().item()))
            if k > 0:
                _, flat_idx = torch.topk(flat_abs, k=k)
                worst = []
                seq_len = curr_logprobs.shape[1]
                for idx in flat_idx.detach().cpu().tolist():
                    b = idx // seq_len
                    pos = idx % seq_len
                    block_start = (pos // block_size) * block_size
                    worst.append(
                        {
                            "batch": int(b),
                            "pos": int(pos),
                            "response_offset": int(pos),
                            "block_start": int(block_start),
                            "token_id": int(target_ids[b, pos].detach().cpu().item()),
                            "curr_logprob": float(
                                curr_logprobs[b, pos].detach().cpu().item()
                            ),
                            "generation_logprob": float(
                                generation_logprobs[b, pos].detach().cpu().item()
                            ),
                            "abs_delta": float(abs_delta[b, pos].detach().cpu().item()),
                            "gen_kl": float(gen_kl[b, pos].detach().cpu().item()),
                        }
                    )
                payload["worst_tokens"] = worst
        if os.environ.get("DIFFUGRPO_DEBUG_LOGPROB_TRACE") == "1":
            trace_limit = int(os.environ.get("DIFFUGRPO_DEBUG_LOGPROB_TRACE_LIMIT", "120"))
            sample_idx = int(os.environ.get("DIFFUGRPO_DEBUG_LOGPROB_TRACE_SAMPLE", "0"))
            if 0 <= sample_idx < curr_logprobs.shape[0]:
                valid_positions = torch.nonzero(
                    mask[sample_idx], as_tuple=False
                ).flatten()[:trace_limit]
                payload["token_trace_sample"] = sample_idx
                payload["token_trace"] = [
                    {
                        "pos": int(pos.item()),
                        "token_id": int(target_ids[sample_idx, pos].detach().cpu().item()),
                        "curr_logprob": float(
                            curr_logprobs[sample_idx, pos].detach().cpu().item()
                        ),
                        "generation_logprob": float(
                            generation_logprobs[sample_idx, pos].detach().cpu().item()
                        ),
                        "delta": float(abs_delta[sample_idx, pos].detach().cpu().item()),
                    }
                    for pos in valid_positions
                ]
        print(json.dumps(payload, sort_keys=True), flush=True)


def _same_position_logprobs(
    output_tensor: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    cfg: PolicyConfig,
    sampling_params: Optional[TrainingSamplingParams],
    inference_only: bool,
    exclude_token_id: Optional[int],
) -> torch.Tensor:
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
    target_ids = target_ids.to(device=logits.device, dtype=torch.long)
    if local_exclude_index is not None:
        seq_len = int(logits.shape[1])
        chunk_size = int(logprob_chunk_size or min(seq_len, 64))
        chunks = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(seq_len, chunk_start + chunk_size)
            logits_chunk = logits[:, chunk_start:chunk_end, :].clone()
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
            token_logprobs: torch.Tensor = (
                ChunkedDistributedLogprobWithSampling.apply(  # type: ignore
                    logits,
                    target_ids,
                    tp_group,
                    sampling_params.top_k,
                    sampling_params.top_p,
                    logprob_chunk_size,
                    inference_only,
                )
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


class DiffuGRPOLossPostProcessor(LossPostProcessor):
    """Megatron loss post-processor for one-pass fully-masked DiffuGRPO."""

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

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]:
        if self.cfg["sequence_packing"]["enabled"] or packed_seq_params is not None:
            raise NotImplementedError(
                "DiffuGRPO Megatron training requires sequence_packing.enabled=false"
            )
        if get_context_parallel_world_size() > 1:
            raise NotImplementedError(
                "DiffuGRPO Megatron training currently requires context_parallel_size=1"
            )

        def loss_fn_inner(
            output_tensor: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
            logprob_estimation_cfg = self.cfg.get("logprob_estimation", {})
            mask_token_id = logprob_estimation_cfg.get("mask_token_id", None)
            exclude_mask = logprob_estimation_cfg.get(
                "exclude_mask_token_from_logits", True
            )
            token_logprobs = _same_position_logprobs(
                output_tensor,
                data_dict["diffu_grpo_target_ids"],
                cfg=self.cfg,
                sampling_params=self.sampling_params,
                inference_only=False,
                exclude_token_id=mask_token_id if exclude_mask else None,
            )

            noisy_length = int(data_dict["diffu_grpo_noisy_lengths"][0].item())
            token_logprobs = token_logprobs[:, :noisy_length]
            loss_mask = data_dict["diffu_grpo_loss_mask"][:, :noisy_length]
            sample_mask = data_dict["sample_mask"]
            advantages = data_dict["advantages"][:, :noisy_length]
            prev_logprobs = data_dict["prev_logprobs"][:, :noisy_length]
            generation_logprobs = data_dict["generation_logprobs"][:, :noisy_length]
            reference_policy_logprobs = data_dict.get("reference_policy_logprobs", None)
            if reference_policy_logprobs is not None:
                reference_policy_logprobs = reference_policy_logprobs[:, :noisy_length]
            combined_mask = loss_mask * sample_mask.unsqueeze(-1)
            if need_top_k_or_top_p_filtering(self.sampling_params):
                token_logprobs = mask_out_neg_inf_logprobs(
                    token_logprobs, combined_mask, "curr_logprobs"
                )

            curr_logprobs_unfiltered = None
            if need_top_k_or_top_p_filtering(self.sampling_params) and (
                hasattr(self.loss_fn, "reference_policy_kl_penalty")
                and self.loss_fn.reference_policy_kl_penalty != 0
            ):
                curr_logprobs_unfiltered = _same_position_logprobs(
                    output_tensor,
                    data_dict["diffu_grpo_target_ids"],
                    cfg=self.cfg,
                    sampling_params=None,
                    inference_only=False,
                    exclude_token_id=mask_token_id if exclude_mask else None,
                )[:, :noisy_length]

            _debug_print_diffu_grpo_logprob_alignment(
                curr_logprobs=token_logprobs,
                generation_logprobs=generation_logprobs,
                loss_mask=loss_mask,
                sample_mask=sample_mask,
                response_lengths=data_dict["diffu_grpo_response_lengths"],
                prompt_lengths=data_dict["diffu_grpo_completion_starts"],
                target_ids=data_dict["diffu_grpo_target_ids"][:, :noisy_length],
                block_size=int(getattr(self.cfg, "get", lambda *_: None)("block_size", 32) or 32),
                kl_type=getattr(self.loss_fn, "reference_policy_kl_type", "k3"),
            )

            if hasattr(self.loss_fn, "compute_from_aligned_tensors"):
                loss, metrics = self.loss_fn.compute_from_aligned_tensors(
                    curr_logprobs=token_logprobs,
                    token_mask=loss_mask,
                    sample_mask=sample_mask,
                    advantages=advantages,
                    prev_logprobs=prev_logprobs,
                    generation_logprobs=generation_logprobs,
                    reference_policy_logprobs=reference_policy_logprobs,
                    curr_logprobs_unfiltered=curr_logprobs_unfiltered,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                )
            else:
                leading_zeros = torch.zeros_like(token_logprobs[:, :1])
                loss_data = BatchedDataDict[Any](
                    {
                        "input_ids": torch.zeros(
                            token_logprobs.shape[0],
                            token_logprobs.shape[1] + 1,
                            device=token_logprobs.device,
                            dtype=torch.long,
                        ),
                        "token_mask": torch.cat([leading_zeros, loss_mask], dim=1),
                        "sample_mask": sample_mask,
                        "advantages": torch.cat([leading_zeros, advantages], dim=1),
                        "prev_logprobs": torch.cat([leading_zeros, prev_logprobs], dim=1),
                        "generation_logprobs": torch.cat(
                            [leading_zeros, generation_logprobs], dim=1
                        ),
                    }
                )
                if reference_policy_logprobs is not None:
                    loss_data["reference_policy_logprobs"] = torch.cat(
                        [leading_zeros, reference_policy_logprobs], dim=1
                    )
                if curr_logprobs_unfiltered is not None:
                    loss_data["curr_logprobs_unfiltered"] = curr_logprobs_unfiltered
                loss, metrics = self.loss_fn(
                    next_token_logprobs=token_logprobs,
                    data=loss_data,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                )
            return loss * self.num_microbatches, metrics

        return loss_fn_inner


class DiffuGRPOLogprobsPostProcessor(LogprobsPostProcessor):
    """No-grad post-processor for fully-masked DiffuGRPO logprobs."""

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        def processor_fn_inner(output_tensor):
            logprob_estimation_cfg = self.cfg.get("logprob_estimation", {})
            mask_token_id = logprob_estimation_cfg.get("mask_token_id", None)
            exclude_mask = logprob_estimation_cfg.get(
                "exclude_mask_token_from_logits", True
            )
            token_logprobs = _same_position_logprobs(
                output_tensor,
                data_dict["diffu_grpo_target_ids"],
                cfg=self.cfg,
                sampling_params=self.sampling_params,
                inference_only=True,
                exclude_token_id=mask_token_id if exclude_mask else None,
            )
            noisy_length = int(data_dict["diffu_grpo_noisy_lengths"][0].item())
            token_logprobs = token_logprobs[:, :noisy_length]
            if need_top_k_or_top_p_filtering(self.sampling_params):
                loss_mask = data_dict["diffu_grpo_score_mask"][:, :noisy_length]
                loss_mask = loss_mask * data_dict["sample_mask"].unsqueeze(-1)
                token_logprobs = mask_out_neg_inf_logprobs(
                    token_logprobs, loss_mask, "prev_logprobs"
                )
            return torch.tensor(0.0, device=token_logprobs.device), {
                "logprobs": token_logprobs
            }

        return processor_fn_inner
