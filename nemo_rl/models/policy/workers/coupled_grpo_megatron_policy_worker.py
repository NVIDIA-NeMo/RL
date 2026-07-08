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
    COUPLED_NUM_LEVELS,
    CoupledGRPORevealSchedule,
    build_coupled_base,
    get_coupled_grpo_logprob_estimation_cfg,
    make_coupled_level_view,
)
from nemo_rl.algorithms.diffu_grpo_logprobs import _scatter_original_response_values
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

    def _noisy_tail_mode(self) -> str:
        """How the final block-padding tail of the noisy side is built.

        ``mask`` (default): tail filled with the mask token (matches generation);
        ``eos``: tail filled with the EOS token (matches SFT padding); ``none``: no
        block-padding, so no tail. See ``_build_completion_only_tensors``."""
        mode = str(self._coupled_cfg().get("noisy_tail_mode", "mask"))
        if mode not in ("mask", "eos", "none"):
            raise ValueError(
                "noisy_tail_mode must be one of mask, eos, none; got "
                f"{mode!r}."
            )
        return mode

    # DiffuGRPO's inherited helpers read mask_token_id via ``_diffu_grpo_cfg``; the
    # coupled estimator carries the same key, so route them to our config.
    def _diffu_grpo_cfg(self):
        return self._coupled_cfg()

    def _coupled_num_levels(self) -> int:
        """Total mask levels per sequence (``num_mc_samples``, defaults to 2).

        Each Monte-Carlo sample is an antithetic coupled pair, so this is
        ``COUPLED_NUM_LEVELS * num_pairs`` and must be even and >= 2.
        ``num_mc_samples == 2`` is the single-pair path (byte-for-byte CoupledGRPO);
        4 -> 2 pairs, 6 -> 3 pairs, ..."""
        num = int(self._coupled_cfg().get("num_mc_samples", COUPLED_NUM_LEVELS))
        if num < COUPLED_NUM_LEVELS or num % COUPLED_NUM_LEVELS != 0:
            raise ValueError(
                "num_mc_samples must be even and >= 2 (each Monte-Carlo sample is an "
                f"antithetic coupled pair, so COUPLED_NUM_LEVELS * num_pairs); got {num}."
            )
        return num

    def _coupled_pair_count_scale(self) -> int:
        # Level-major training accumulates each token/sequence once per pair across the
        # 2K levels, so the loss normalizer scales by the pair count (1 at a single
        # pair -> byte-for-byte with the original CoupledGRPO).
        return self._coupled_num_pairs()

    def _coupled_num_pairs(self) -> int:
        """Independent coupled pairs (K = num_mc_samples // COUPLED_NUM_LEVELS)."""
        return self._coupled_num_levels() // COUPLED_NUM_LEVELS

    # ---- training: lazy two-level schedule (one optimizer step) -------------
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._coupled_cfg()
        num_pairs = self._coupled_num_pairs()
        block_size = self._diffusion_block_size()
        self._maybe_print_diffusion_block_size("coupled_grpo_train", block_size)
        base, num_samples, num_levels = build_coupled_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            noisy_block_size=block_size,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=True,
            num_pairs=num_pairs,
            noisy_tail_mode=self._noisy_tail_mode(),
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Scatter pairs 1..K-1 prev / reference logprobs into the noisy layout so
        # ``make_coupled_level_view`` can route pair = level // 2's summed [N, S] onto
        # its two level rows. Pair 0 is the standard prev_logprobs /
        # reference_policy_logprobs base field (already scattered by include_loss).
        # No-op at num_pairs == 1 (byte-for-byte CoupledGRPO).
        if num_samples:
            self._scatter_pair_logprobs(base, data, num_pairs)
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

    # ---- logprobs: explicit for-loop over the 2K levels ---------------------
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        # Run all 2K levels and sum WITHIN each pair (a pair's two complementary
        # levels partition the response tokens, so its per-pair sum holds each token's
        # logprob once). Pair 0's sum is the standard ["logprobs"] [N, S]; pairs
        # 1..K-1 ride as extra keys logprobs_pair{p}. At one pair (num_mc_samples=2)
        # this is the single-tensor CoupledGRPO output.
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

    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[Any]:
        # Mirror base_policy_worker.get_reference_policy_logprobs but also carry the
        # per-pair extra keys (renamed logprobs_pair{p} -> reference_policy_logprobs
        # _pair{p}) so pairs 1..K-1 reference logprobs reach the training batch. At
        # one pair no extra keys exist -> {"reference_logprobs": ...} only.
        with self.use_reference_model():
            reference = self.get_logprobs(
                data=data, micro_batch_size=micro_batch_size
            )
        out = BatchedDataDict[Any]()
        out["reference_logprobs"] = reference["logprobs"].cpu()
        for pair in range(1, self._coupled_num_pairs()):
            key = f"logprobs_pair{pair}"
            if key in reference:
                out[f"reference_logprobs_pair{pair}"] = reference[key].cpu()
        return out

    def _scatter_pair_logprobs(
        self,
        base: BatchedDataDict[Any],
        data: BatchedDataDict[Any],
        num_pairs: int,
    ) -> None:
        """Scatter pairs 1..K-1 prev / reference logprobs into the noisy layout.

        Pair p's per-token ``[N, S]`` prev / reference logprobs arrive as extra keys
        ``prev_logprobs_pair{p}`` / ``reference_policy_logprobs_pair{p}`` (the summed
        logprobs of pair p's two complementary levels; pair 0 uses the standard
        ``prev_logprobs`` / ``reference_policy_logprobs``, already scattered by
        ``build_fully_masked_completion_loss_batch``). Map them into the noisy layout
        with the SAME scatter as those standard keys and attach as per-pair base
        fields so ``make_coupled_level_view`` can route pair = level // 2's tensor per
        row. No-op at num_pairs == 1 (the loop is empty).
        """
        total_length = base["input_ids"].shape[1]
        completion_starts = base["diffu_grpo_completion_starts"]
        response_lengths = base["diffu_grpo_response_lengths"]
        for pair in range(1, num_pairs):
            for prefix in ("prev_logprobs", "reference_policy_logprobs"):
                key = f"{prefix}_pair{pair}"
                if key not in data:
                    continue
                base[key] = _scatter_original_response_values(
                    values=data[key],
                    total_length=total_length,
                    completion_starts=completion_starts,
                    response_lengths=response_lengths,
                )

    def _coupled_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int],
        only_level0: bool = False,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Build the coupled base and return K per-pair summed ``[N, S]`` logprobs.

        Pair 0's sum is ["logprobs"] (the single-tensor contract); pairs 1..K-1 are
        ``logprobs_pair{p}``. ``only_level0`` (the provided-mask / generation-KL
        verification path) pins ONE mask and runs a single level -- inherently
        single-pair, so it collapses to num_pairs=1 and returns just ["logprobs"]. At
        num_pairs == 1 the output is ``{"logprobs": <single [N, S]>}`` (byte-for-byte
        CoupledGRPO). Both choices are rank-independent, so the pass count stays
        DP-uniform.
        """
        self._validate_diffusion_algorithm_support()
        cfg = self._coupled_cfg()
        num_pairs = 1 if only_level0 else self._coupled_num_pairs()
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
            num_pairs=num_pairs,
            noisy_tail_mode=self._noisy_tail_mode(),
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if num_levels == 0:
            empty = torch.zeros_like(data["input_ids"], dtype=torch.float32)
            return BatchedDataDict[LogprobOutputSpec](logprobs=empty).to("cpu")
        pair_logprobs = self._run_coupled_levels(
            data=data,
            base=base,
            num_samples=num_samples,
            original_seq_len=int(data["input_ids"].shape[1]),
            coupled_mbs=coupled_mbs,
            levels=(0,) if only_level0 else range(num_levels),
        )
        out = BatchedDataDict[LogprobOutputSpec](logprobs=pair_logprobs[0])
        if not only_level0:
            for pair in range(1, num_pairs):
                out[f"logprobs_pair{pair}"] = pair_logprobs[pair]
        return out.to("cpu")

    def _run_coupled_levels(
        self,
        *,
        data: BatchedDataDict[Any],
        base: BatchedDataDict[Any],
        num_samples: int,
        original_seq_len: int,
        coupled_mbs: Optional[int],
        levels,
    ) -> list[torch.Tensor]:
        """Run one masked forward per level and sum the scattered ``[N, S]`` logprobs
        WITHIN each pair (levels ``2p`` / ``2p+1`` -> ``pair_out[p]``).

        Approach A: a pair's two complementary levels partition the response tokens,
        so summing those two levels alone is lossless per pair (a GLOBAL sum over all
        2K levels would over-count each token K times when K > 1). Returns the K
        per-pair ``[N, S]`` tensors ordered by pair (at one pair, a single-element
        list -- byte-for-byte CoupledGRPO). Each forward selects its level view
        through the ``self._cp_*`` instance state ``_build_logprob_megatron_batch``
        reads back; the pass count is ``len(levels)`` on every rank (rank-independent,
        DP-uniform).
        """
        self._cp_base = base
        self._cp_coupled_mbs = coupled_mbs
        self._cp_num_samples = num_samples
        self._cp_original_seq_len = original_seq_len
        # Derive the pair count from the levels actually run (not the config) so the
        # single-level provided-mask / generation-KL path (levels=(0,)) collapses to
        # one pair; the full path (range(2K)) yields K.
        levels_list = [int(level) for level in levels]
        num_pairs = max(l // COUPLED_NUM_LEVELS for l in levels_list) + 1
        pair_out: list[Optional[torch.Tensor]] = [None] * num_pairs
        try:
            for level in levels_list:
                self._cp_level = level
                level_out = super().get_logprobs(
                    data=data, micro_batch_size=coupled_mbs
                )["logprobs"]
                pair = int(level) // COUPLED_NUM_LEVELS
                pair_out[pair] = (
                    level_out
                    if pair_out[pair] is None
                    else pair_out[pair] + level_out
                )
        finally:
            self._cp_base = None
        return pair_out

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
