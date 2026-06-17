# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any, Optional

import ray
import torch

from nemo_rl.algorithms.block_just_grpo_logprobs import (
    BlockJustGRPORevealSchedule,
    build_block_reveal_base,
    get_block_reveal_logprob_estimation_cfg,
    make_reveal_level_view,
    scatter_block_reveal_logprobs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffu_grpo_megatron_policy_worker import (
    DiffuGRPOMegatronPolicyWorkerImpl,
)


class BlockJustGRPOMegatronPolicyWorkerImpl(DiffuGRPOMegatronPolicyWorkerImpl):
    """JustGRPO leftmost-reveal logprobs computed in ``block_size`` passes.

    Reuses DiffuGRPO's asymmetric ``[noisy | clean]`` block-diffusion layout,
    attention metadata, and same-position post-processors. Rather than
    materializing all ``num_levels x N`` reveal rows at once, the worker iterates
    reveal levels explicitly:

    * logprobs: an outer Python for-loop runs one forward per reveal level and
      sums the scattered per-level logprobs.
    * training: a ``BlockJustGRPORevealSchedule`` yields one reveal level per
      forward, so the single ``forward_backward`` accumulates gradients across
      all levels before one optimizer step.

    Only one reveal level (N rows) is ever resident, which is what makes long
    context tractable.
    """

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("BlockJustGRPO")
        get_block_reveal_logprob_estimation_cfg(self.cfg)

    def _block_reveal_cfg(self):
        return get_block_reveal_logprob_estimation_cfg(self.cfg)

    # DiffuGRPO's inherited methods read mask_token_id via ``_diffu_grpo_cfg``; the
    # block-reveal estimator carries the same key, so route them to our config.
    def _diffu_grpo_cfg(self):
        return self._block_reveal_cfg()

    def _block_reveal_block_size(self) -> int:
        cfg = self._block_reveal_cfg()
        return int(cfg.get("block_size") or self._diffusion_block_size())

    def _block_reveal_tokens_per_level(self) -> int:
        cfg = self._block_reveal_cfg()
        return max(1, int(cfg.get("reveal_tokens_per_level") or 1))

    # ---- training: lazy reveal-level iterator (one optimizer step) ----------
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._block_reveal_cfg()
        block_size = self._block_reveal_block_size()
        reveal_k = self._block_reveal_tokens_per_level()
        self._maybe_print_diffusion_block_size("block_reveal_train", block_size)
        base, _num_samples, num_levels = build_block_reveal_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=block_size,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=True,
            max_reveal_levels=cfg.get("max_reveal_levels"),
            reveal_tokens_per_level=reveal_k,
        )
        # One forward per reveal level; samples within a level are microbatched the
        # standard way by train_micro_batch_size (passed in as ``mbs``). A single
        # forward_backward over the whole schedule accumulates gradients across all
        # reveal levels before one optimizer step.
        schedule = BlockJustGRPORevealSchedule(base).configure(
            num_levels=num_levels,
            block_size=block_size,
            harvest_keys=("diffu_grpo_score_mask", "diffu_grpo_loss_mask"),
            reveal_tokens_per_level=reveal_k,
        )
        return (
            schedule,
            self._cfg_for_diffu_grpo_sequence(base["input_ids"].shape[1]),
            mbs,
            {},
        )

    # ---- logprobs: explicit for-loop over reveal levels ---------------------
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        self._validate_diffusion_algorithm_support()
        cfg = self._block_reveal_cfg()
        block_size = self._block_reveal_block_size()
        reveal_k = self._block_reveal_tokens_per_level()
        reveal_mbs = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        base, num_samples, num_levels = build_block_reveal_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=block_size,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=False,
            max_reveal_levels=cfg.get("max_reveal_levels"),
            reveal_tokens_per_level=reveal_k,
        )
        original_seq_len = int(data["input_ids"].shape[1])
        if num_levels == 0:
            empty = torch.zeros_like(data["input_ids"], dtype=torch.float32)
            return BatchedDataDict[LogprobOutputSpec](logprobs=empty).to("cpu")

        # For each reveal level, reveal one more token per block and run one
        # forward; its scattered logprobs land at that level's harvested positions
        # (zero elsewhere). Summing across levels yields the full [N, S] logprobs.
        self._br_base = base
        self._br_block_size_cur = block_size
        self._br_reveal_k = reveal_k
        self._br_reveal_mbs = reveal_mbs
        self._br_num_samples = num_samples
        self._br_original_seq_len = original_seq_len
        accumulated: Optional[torch.Tensor] = None
        try:
            for level in range(num_levels):
                self._br_level = level
                level_out = super().get_logprobs(
                    data=data, micro_batch_size=reveal_mbs
                )["logprobs"]
                accumulated = (
                    level_out if accumulated is None else accumulated + level_out
                )
        finally:
            self._br_base = None
        return BatchedDataDict[LogprobOutputSpec](logprobs=accumulated).to("cpu")

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
        # Called by super().get_logprobs once per reveal level; builds the view for
        # the level currently selected by the outer loop.
        base = self._br_base
        view = make_reveal_level_view(
            base,
            self._br_level,
            self._br_block_size_cur,
            ("diffu_grpo_score_mask",),
            self._br_reveal_k,
        )
        level_mbs = (
            micro_batch_size if micro_batch_size is not None else self._br_reveal_mbs
        )
        noisy_response_offset = int(
            view["diffu_grpo_noisy_response_offsets"][0].item()
        )
        return (
            view,
            self._cfg_for_diffu_grpo_sequence(view["input_ids"].shape[1]),
            level_mbs,
            {
                "num_samples": self._br_num_samples,
                "original_seq_len": self._br_original_seq_len,
                "noisy_response_offset": noisy_response_offset,
            },
        )

    def _finalize_logprobs_from_outputs(
        self,
        list_of_logprobs: list[dict[str, torch.Tensor]],
        *,
        original_data: BatchedDataDict[Any],
        transformed_data: BatchedDataDict[Any],
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        flat_logprobs = torch.cat(
            [lp["logprobs"] for lp in list_of_logprobs], dim=0
        )
        return scatter_block_reveal_logprobs(
            flat_logprobs=flat_logprobs,
            harvest_mask=transformed_data["block_reveal_harvest_mask"],
            sample_index=transformed_data["block_reveal_sample_index"],
            completion_starts=transformed_data["diffu_grpo_completion_starts"],
            noisy_response_offset=metadata["noisy_response_offset"],
            original_seq_len=metadata["original_seq_len"],
            num_samples=metadata["num_samples"],
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("block_just_grpo_megatron_policy_worker")
)  # pragma: no cover
class BlockJustGRPOMegatronPolicyWorker(BlockJustGRPOMegatronPolicyWorkerImpl):
    pass
