# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from contextlib import AbstractContextManager
from typing import Any, Iterator, Optional

import ray
import torch

from nemo_rl.models.megatron.data import ProcessedMicrobatch
from nemo_rl.algorithms.diffu_grpo_logprobs import (
    build_fully_masked_completion_batch,
    build_fully_masked_completion_loss_batch,
    get_diffu_grpo_logprob_estimation_cfg,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.diffu_grpo_train import (
    DiffuGRPOLogprobsPostProcessor,
    DiffuGRPOLossPostProcessor,
)
from nemo_rl.models.megatron.train import LogprobsPostProcessor, LossPostProcessor
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
    DiffusionMegatronPolicyWorkerImpl,
)


class DiffuGRPOMegatronPolicyWorkerImpl(DiffusionMegatronPolicyWorkerImpl):
    """Megatron worker for DiffuGRPO fully-masked completion logprobs."""

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("DiffuGRPO")
        get_diffu_grpo_logprob_estimation_cfg(self.cfg)

    def _diffu_grpo_cfg(self):
        return get_diffu_grpo_logprob_estimation_cfg(self.cfg)

    def _cfg_for_diffu_grpo_sequence(self, sequence_length: int) -> PolicyConfig:
        cfg = self._cfg_without_sequence_packing()
        cfg["max_total_sequence_length"] = int(sequence_length)
        return cfg

    def _diffu_grpo_sequence_length_round(self) -> int:
        return int(self.cfg.get("sequence_packing", {}).get("sequence_length_round", 64))

    def _training_attention_context(self) -> AbstractContextManager[Any]:
        # Keep NemotronLabsDiffusionAttention on its training forward. DiffuGRPO
        # supplies asymmetric [masked_response | clean_prompt_response] metadata
        # per microbatch, so no fixed half-sequence mask length is needed.
        return self._megatron_attention_context("training")

    def _logprob_attention_context(self) -> AbstractContextManager[Any]:
        # Same as training: diffuGRPO scores one asymmetric masked-response pass.
        return self._megatron_attention_context("training")

    def _set_asymmetric_ar_metadata(
        self,
        microbatch: ProcessedMicrobatch,
    ) -> None:
        data_dict = microbatch.data_dict
        if "diffu_grpo_noisy_lengths" not in data_dict:
            return

        noisy_lengths = data_dict["diffu_grpo_noisy_lengths"]
        noisy_valid_lengths = data_dict["diffu_grpo_noisy_valid_lengths"]
        clean_padded_lengths = data_dict["diffu_grpo_clean_padded_lengths"]
        noisy_response_offsets = data_dict["diffu_grpo_noisy_response_offsets"]
        if not torch.all(noisy_lengths == noisy_lengths[0]):
            raise ValueError("diffuGRPO noisy length must be constant within a microbatch")
        if not torch.all(clean_padded_lengths == clean_padded_lengths[0]):
            raise ValueError("diffuGRPO clean padded length must be constant within a microbatch")
        if not torch.all(noisy_response_offsets == noisy_response_offsets[0]):
            raise ValueError("diffuGRPO noisy response offset must be constant within a microbatch")

        for module in self._diffusion_attention_modules(self.model):
            if not hasattr(module, "set_asymmetric_ar_metadata"):
                raise RuntimeError(
                    "DiffuGRPO completion-only replay requires "
                    "NemotronLabsDiffusionAttention.set_asymmetric_ar_metadata"
                )
            module.set_asymmetric_ar_metadata(
                noisy_length=int(noisy_lengths[0].item()),
                clean_length=int(clean_padded_lengths[0].item()),
                noisy_response_offset=int(noisy_response_offsets[0].item()),
                prompt_lengths=data_dict["diffu_grpo_completion_starts"],
                response_lengths=data_dict["diffu_grpo_response_lengths"],
                noisy_valid_lengths=noisy_valid_lengths,
                clean_lengths=data_dict["diffu_grpo_clean_lengths"],
            )

    def _clear_asymmetric_ar_metadata(self) -> None:
        for module in self._diffusion_attention_modules(self.model):
            if hasattr(module, "clear_asymmetric_ar_metadata"):
                module.clear_asymmetric_ar_metadata()

    def _wrap_iterator_with_diffu_grpo_metadata(
        self,
        data_iterator: Iterator[ProcessedMicrobatch],
    ) -> Iterator[ProcessedMicrobatch]:
        try:
            for microbatch in data_iterator:
                self._set_asymmetric_ar_metadata(microbatch)
                yield microbatch
        finally:
            self._clear_asymmetric_ar_metadata()

    def _wrap_training_microbatch_iterator(
        self,
        data_iterator: Iterator[ProcessedMicrobatch],
        cfg: PolicyConfig,
    ) -> Iterator[ProcessedMicrobatch]:
        return self._wrap_iterator_with_diffu_grpo_metadata(data_iterator)

    def _wrap_logprob_microbatch_iterator(
        self,
        data_iterator: Iterator[ProcessedMicrobatch],
        cfg: PolicyConfig,
    ) -> Iterator[ProcessedMicrobatch]:
        return self._wrap_iterator_with_diffu_grpo_metadata(data_iterator)

    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        block_size = self._diffusion_block_size()
        self._maybe_print_diffusion_block_size("diffu_grpo_train", block_size)
        batch = build_fully_masked_completion_loss_batch(
            data,
            mask_token_id=self._diffu_grpo_cfg()["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            block_size=block_size,
        )
        return batch, self._cfg_for_diffu_grpo_sequence(batch["input_ids"].shape[1]), mbs, {}

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
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        block_size = self._diffusion_block_size()
        self._maybe_print_diffusion_block_size("diffu_grpo_logprob", block_size)
        transformed_data = build_fully_masked_completion_batch(
            data,
            mask_token_id=self._diffu_grpo_cfg()["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            block_size=block_size,
        )
        return (
            transformed_data,
            self._cfg_for_diffu_grpo_sequence(transformed_data["input_ids"].shape[1]),
            logprob_batch_size,
            {"original_seq_len": data["input_ids"].shape[1]},
        )

    def _make_loss_post_processor(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int,
    ) -> LossPostProcessor:
        return DiffuGRPOLossPostProcessor(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            sampling_params=self.sampling_params,
        )

    def _make_logprobs_post_processor(
        self,
        cfg: PolicyConfig,
    ) -> LogprobsPostProcessor:
        return DiffuGRPOLogprobsPostProcessor(
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
        original_seq_len = metadata["original_seq_len"]
        all_logprobs = torch.cat(
            [lp_dict["logprobs"] for lp_dict in list_of_logprobs], dim=0
        )
        output = torch.zeros(
            (all_logprobs.shape[0], original_seq_len),
            dtype=all_logprobs.dtype,
            device=all_logprobs.device,
        )
        starts = transformed_data["diffu_grpo_completion_starts"].to(
            device=all_logprobs.device
        )
        lengths = transformed_data["diffu_grpo_response_lengths"].to(
            device=all_logprobs.device
        )
        offsets = transformed_data["diffu_grpo_noisy_response_offsets"].to(
            device=all_logprobs.device
        )
        for batch_idx in range(all_logprobs.shape[0]):
            start = int(starts[batch_idx].item())
            length = int(lengths[batch_idx].item())
            offset = int(offsets[batch_idx].item())
            if length > 0:
                output[batch_idx, start : start + length] = all_logprobs[
                    batch_idx, offset : offset + length
                ]
        return output


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("diffu_grpo_megatron_policy_worker")
)  # pragma: no cover
class DiffuGRPOMegatronPolicyWorker(DiffuGRPOMegatronPolicyWorkerImpl):
    pass
