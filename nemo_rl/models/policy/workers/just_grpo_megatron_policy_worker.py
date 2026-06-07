# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from contextlib import AbstractContextManager
from typing import Any, Iterator, Optional

import ray
import torch

from nemo_rl.algorithms.just_grpo_logprobs import (
    build_leftmost_reveal_batch,
    build_leftmost_reveal_loss_batch,
    get_leftmost_reveal_logprob_estimation_cfg,
    pad_reveal_batch_to_multiple,
    scatter_leftmost_reveal_logprobs,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.just_grpo_train import (
    JustGRPOLogprobsPostProcessor,
    JustGRPOLossPostProcessor,
)
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
    DiffusionMegatronPolicyWorkerImpl,
)


class JustGRPOMegatronPolicyWorkerImpl(DiffusionMegatronPolicyWorkerImpl):
    """Megatron worker with JustGRPO leftmost-reveal logprob semantics."""

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("JustGRPO")
        get_leftmost_reveal_logprob_estimation_cfg(self.cfg)

    def _just_grpo_cfg(self):
        return get_leftmost_reveal_logprob_estimation_cfg(self.cfg)

    def _attention_mode(self) -> str:
        return self._just_grpo_cfg()["megatron_attention_mode"]

    def _training_attention_context(self) -> AbstractContextManager[Any]:
        return self._megatron_attention_context(self._attention_mode())

    def _logprob_attention_context(self) -> AbstractContextManager[Any]:
        return self._megatron_attention_context(self._attention_mode())

    def _drop_non_sequence_reveal_metadata(
        self,
        data: BatchedDataDict[Any],
    ) -> BatchedDataDict[Any]:
        """Drop reveal metadata that does not follow Megatron's [B, S] convention."""
        return BatchedDataDict[Any](
            {
                key: value
                for key, value in data.items()
                if key != "just_grpo_output_shape"
            }
        )

    def _reveal_schedule_args(self) -> tuple[str, Optional[int]]:
        cfg = self._just_grpo_cfg()
        reveal_schedule = cfg["reveal_schedule"]
        max_reveal_positions = (
            cfg["max_reveal_positions"]
            if reveal_schedule == "fixed_response_window"
            else None
        )
        return reveal_schedule, max_reveal_positions

    def _set_block_bidirectional_mask(
        self,
        attention_modules: list[Any],
        data: BatchedDataDict[Any],
        seq_len: int,
        block_size: int,
    ) -> None:
        target_positions = data["just_grpo_target_positions"]
        response_starts = data["just_grpo_response_starts"]
        response_ends = data["just_grpo_response_ends"]
        device = target_positions.device
        relative_positions = (target_positions - response_starts).clamp_min(0)
        block_starts = response_starts + (relative_positions // block_size) * block_size
        response_ends = response_ends.clamp(max=seq_len)
        block_ends = torch.minimum(block_starts + block_size, response_ends)
        block_starts = block_starts.to(device=device, dtype=torch.long)
        block_ends = block_ends.to(device=device, dtype=torch.long)

        for module in attention_modules:
            if not hasattr(module, "set_block_bidirectional_mask"):
                raise RuntimeError(
                    "Megatron diffusion attention does not support "
                    "set_block_bidirectional_mask. Ensure the Megatron-Bridge "
                    "block-bidirectional attention patch is on PYTHONPATH."
                )
            module.set_block_bidirectional_mask(block_starts, block_ends)

    def _with_block_bidirectional_attention(
        self,
        data_iterator: Iterator[Any],
        *,
        source: str,
    ) -> Iterator[Any]:
        attention_modules = self._diffusion_attention_modules(self.model)
        block_size = self._diffusion_block_size()
        self._maybe_print_diffusion_block_size(source, block_size)
        for processed_mb in data_iterator:
            self._set_block_bidirectional_mask(
                attention_modules=attention_modules,
                data=processed_mb.data_dict,
                seq_len=processed_mb.input_ids.shape[1],
                block_size=block_size,
            )
            yield processed_mb

    def _wrap_training_microbatch_iterator(
        self,
        data_iterator: Iterator[Any],
        cfg: PolicyConfig,
    ) -> Iterator[Any]:
        if self._attention_mode() != "inference_block_bidirectional":
            return data_iterator
        return self._with_block_bidirectional_attention(
            data_iterator,
            source="train",
        )

    def _wrap_logprob_microbatch_iterator(
        self,
        data_iterator: Iterator[Any],
        cfg: PolicyConfig,
    ) -> Iterator[Any]:
        if self._attention_mode() != "inference_block_bidirectional":
            return data_iterator
        return self._with_block_bidirectional_attention(
            data_iterator,
            source="get_logprobs",
        )

    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._just_grpo_cfg()
        reveal_schedule, max_reveal_positions = self._reveal_schedule_args()
        batch = build_leftmost_reveal_loss_batch(
            data,
            mask_token_id=cfg["mask_token_id"],
            reveal_schedule=reveal_schedule,
            max_reveal_positions=max_reveal_positions,
        )
        train_reveal_batch_size = cfg["train_reveal_batch_size"]
        reveal_count = pad_reveal_batch_to_multiple(batch, train_reveal_batch_size)
        return (
            self._drop_non_sequence_reveal_metadata(batch),
            self._cfg_without_sequence_packing(),
            train_reveal_batch_size,
            {"reveal_count": reveal_count},
        )

    def _build_logprob_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int],
    ) -> tuple[
        BatchedDataDict[Any] | None,
        PolicyConfig,
        int,
        dict[str, Any],
    ]:
        cfg = self._just_grpo_cfg()
        reveal_schedule, max_reveal_positions = self._reveal_schedule_args()
        reveal_data = build_leftmost_reveal_batch(
            input_ids=data["input_ids"],
            input_lengths=data["input_lengths"],
            token_mask=data["token_mask"],
            sample_mask=data.get("sample_mask", None),
            mask_token_id=cfg["mask_token_id"],
            reveal_schedule=reveal_schedule,
            max_reveal_positions=max_reveal_positions,
        )
        if reveal_data["input_ids"].shape[0] == 0:
            return (
                None,
                self._cfg_without_sequence_packing(),
                1,
                {
                    "empty_logprobs": torch.zeros_like(
                        data["input_ids"],
                        dtype=torch.float32,
                    )
                },
            )

        reveal_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else cfg["reveal_batch_size"]
        )
        reveal_count = pad_reveal_batch_to_multiple(reveal_data, reveal_batch_size)
        return (
            self._drop_non_sequence_reveal_metadata(reveal_data),
            self._cfg_without_sequence_packing(),
            reveal_batch_size,
            {
                "reveal_data": reveal_data,
                "reveal_count": reveal_count,
            },
        )

    def _make_loss_post_processor(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int,
    ) -> LossPostProcessor:
        return JustGRPOLossPostProcessor(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            sampling_params=self.sampling_params,
            draft_model=None,
        )

    def _make_logprobs_post_processor(
        self,
        cfg: PolicyConfig,
    ) -> LogprobsPostProcessor:
        return JustGRPOLogprobsPostProcessor(
            cfg=cfg,
            sampling_params=self.sampling_params,
            use_linear_ce_fusion=False,
        )

    def _finalize_logprobs_from_outputs(
        self,
        list_of_logprobs: list[dict[str, torch.Tensor]],
        *,
        original_data: BatchedDataDict[Any],
        transformed_data: BatchedDataDict[Any],
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        reveal_data = metadata["reveal_data"]
        reveal_count = metadata["reveal_count"]
        flat_logprobs = torch.cat([l["logprobs"] for l in list_of_logprobs], dim=0)
        flat_logprobs = flat_logprobs[:reveal_count]
        return scatter_leftmost_reveal_logprobs(
            flat_logprobs=flat_logprobs,
            batch_indices=reveal_data["just_grpo_batch_indices"][:reveal_count],
            target_positions=reveal_data["just_grpo_target_positions"][:reveal_count],
            output_shape=reveal_data["just_grpo_output_shape"][:reveal_count],
            row_mask=reveal_data["just_grpo_row_mask"][:reveal_count],
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class JustGRPOMegatronPolicyWorker(JustGRPOMegatronPolicyWorkerImpl):
    pass
