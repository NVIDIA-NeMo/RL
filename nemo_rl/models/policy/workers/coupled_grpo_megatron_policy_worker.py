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

from typing import Any, Optional

import ray
import torch

from nemo_rl.algorithms.block_just_grpo_logprobs import scatter_block_reveal_logprobs
from nemo_rl.algorithms.coupled_grpo_logprobs import (
    CoupledGRPORevealSchedule,
    build_coupled_base,
    get_coupled_grpo_logprob_estimation_cfg,
    make_coupled_level_view,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffu_grpo_megatron_policy_worker import (
    DiffuGRPOMegatronPolicyWorkerImpl,
)


class CoupledGRPOMegatronPolicyWorkerImpl(DiffuGRPOMegatronPolicyWorkerImpl):
    """CoupledGRPO logprobs from two complementary masked forward passes.

    Reuses DiffuGRPO's asymmetric ``[noisy | clean]`` layout, attention metadata,
    and same-position post-processors. Level 0 masks a per-sample random subset
    ``M`` of the response (ratio ``t ~ U(0, 1)``); level 1 masks the exact
    complement. Each valid response token is masked in exactly one level, so:

    * logprobs: an outer Python for-loop runs one forward per level and sums the
      scattered per-level logprobs into the full ``[N, S]`` vector.
    * training: a ``CoupledGRPORevealSchedule`` yields one level per forward, so a
      single ``forward_backward`` accumulates gradients across both levels before
      one optimizer step.

    The same ``M`` (seeded from ``data["coupled_grpo_seed"]``) is regenerated in
    every forward, so prev / reference / training logprobs are mutually consistent
    -- which is what makes the GRPO importance ratio valid. The pass count is the
    constant 2 on every rank, so there is no DP-uniform deadlock concern.
    """

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("CoupledGRPO")
        get_coupled_grpo_logprob_estimation_cfg(self.cfg)

    def _coupled_cfg(self):
        return get_coupled_grpo_logprob_estimation_cfg(self.cfg)

    # DiffuGRPO's inherited helpers read mask_token_id via ``_diffu_grpo_cfg``; the
    # coupled estimator carries the same key, so route them to our config.
    def _diffu_grpo_cfg(self):
        return self._coupled_cfg()

    # ---- training: lazy two-level schedule (one optimizer step) -------------
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._coupled_cfg()
        block_size = self._diffusion_block_size()
        self._maybe_print_diffusion_block_size("coupled_grpo_train", block_size)
        base, _num_samples, num_levels = build_coupled_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            noisy_block_size=block_size,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=True,
        )
        # One forward per level; samples within a level are microbatched the
        # standard way by train_micro_batch_size (passed in as ``mbs``). A single
        # forward_backward over the whole schedule accumulates gradients across
        # both complementary levels before one optimizer step.
        schedule = CoupledGRPORevealSchedule(base).configure(
            num_levels=num_levels,
            harvest_keys=("diffu_grpo_score_mask", "diffu_grpo_loss_mask"),
        )
        return (
            schedule,
            self._cfg_for_diffu_grpo_sequence(base["input_ids"].shape[1]),
            mbs,
            {},
        )

    # ---- logprobs: explicit for-loop over the two levels --------------------
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        # Sum both complementary levels: each masks its half of the response and
        # runs one forward; the scattered logprobs land at that level's harvested
        # positions (zero elsewhere), so the two levels reconstruct the [N, S]
        # vector.
        return self._coupled_logprobs(data=data, micro_batch_size=micro_batch_size)

    def get_logprobs_with_provided_mask(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Score logprobs under a caller-provided level-0 mask with ONE forward.

        ``data`` must carry ``coupled_grpo_level0_mask_override`` (``[N, S]``, 1 at
        the masked positions, original sequence layout). Level 0 masks exactly those
        positions and reveals the rest of the response as clean context, so the
        harvested logprobs reproduce the conditioning recorded by SGLang's
        FastDiffuser ``final_step`` logprob mode -- letting an external check compare
        the two. Returns ``[N, S]`` logprobs, zero off the mask. Runs a single
        forward on every rank (DP-uniform, like get_logprobs).
        """
        if "coupled_grpo_level0_mask_override" not in data:
            raise ValueError(
                "get_logprobs_with_provided_mask requires a "
                "'coupled_grpo_level0_mask_override' tensor in the data batch."
            )
        return self._coupled_logprobs(
            data=data, micro_batch_size=micro_batch_size, only_level0=True
        )

    def _coupled_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int],
        only_level0: bool = False,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Build the coupled base and sum the scattered per-level ``[N, S]`` logprobs.

        ``only_level0`` runs just level 0 (the provided-mask path); both choices are
        rank-independent, so the pass count stays DP-uniform.
        """
        self._validate_diffusion_algorithm_support()
        cfg = self._coupled_cfg()
        block_size = self._diffusion_block_size()
        coupled_mbs = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        base, num_samples, num_levels = build_coupled_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            noisy_block_size=block_size,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=False,
        )
        if num_levels == 0:
            empty = torch.zeros_like(data["input_ids"], dtype=torch.float32)
            return BatchedDataDict[LogprobOutputSpec](logprobs=empty).to("cpu")
        out = self._run_coupled_levels(
            data=data,
            base=base,
            num_samples=num_samples,
            original_seq_len=int(data["input_ids"].shape[1]),
            coupled_mbs=coupled_mbs,
            levels=(0,) if only_level0 else range(num_levels),
        )
        return BatchedDataDict[LogprobOutputSpec](logprobs=out).to("cpu")

    def _run_coupled_levels(
        self,
        *,
        data: BatchedDataDict[Any],
        base: BatchedDataDict[Any],
        num_samples: int,
        original_seq_len: int,
        coupled_mbs: Optional[int],
        levels,
    ) -> torch.Tensor:
        """Run one masked forward per level in ``levels`` and sum the scattered
        per-level ``[N, S]`` logprobs.

        Each forward selects its level view through the ``self._cp_*`` instance
        state that ``_build_logprob_megatron_batch`` reads back. The pass count is
        ``len(levels)`` on every rank, so callers must pass a rank-independent
        ``levels`` (e.g. ``range(2)`` or ``(0,)``) to stay DP-uniform.
        """
        self._cp_base = base
        self._cp_coupled_mbs = coupled_mbs
        self._cp_num_samples = num_samples
        self._cp_original_seq_len = original_seq_len
        accumulated: Optional[torch.Tensor] = None
        try:
            for level in levels:
                self._cp_level = level
                level_out = super().get_logprobs(
                    data=data, micro_batch_size=coupled_mbs
                )["logprobs"]
                accumulated = (
                    level_out if accumulated is None else accumulated + level_out
                )
        finally:
            self._cp_base = None
        return accumulated

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
        # Called by super().get_logprobs once per level; builds the view for the
        # level currently selected by the outer loop.
        base = self._cp_base
        view = make_coupled_level_view(
            base, self._cp_level, ("diffu_grpo_score_mask",)
        )
        level_mbs = (
            micro_batch_size
            if micro_batch_size is not None
            else self._cp_coupled_mbs
        )
        noisy_response_offset = int(
            view["diffu_grpo_noisy_response_offsets"][0].item()
        )
        return (
            view,
            self._cfg_for_diffu_grpo_sequence(view["input_ids"].shape[1]),
            level_mbs,
            {
                "num_samples": self._cp_num_samples,
                "original_seq_len": self._cp_original_seq_len,
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
    runtime_env=get_runtime_env_for_policy_worker("coupled_grpo_megatron_policy_worker")
)  # pragma: no cover
class CoupledGRPOMegatronPolicyWorker(CoupledGRPOMegatronPolicyWorkerImpl):
    pass
