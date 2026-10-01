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

"""Megatron worker layer for DiffuGRPO's asymmetric ``[noisy | clean]`` layout.

Everything here is common to any objective scored over the layout built by
:func:`nemo_rl.algorithms.diffu_grpo_logprobs.build_fully_masked_completion_batch`:
the per-microbatch sequence-length override (the layout is wider than the
rollout, and it is built *after* the policy config is frozen), the attention
mode both passes run under, and the per-microbatch asymmetric-AR metadata the
diffusion attention module needs to place the noisy/clean boundary.

It stays abstract: the concrete batch builders, post-processors and logprob
unscatter live in the leaf worker. The only leaf shipped today is
:class:`~nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker.HybridARDiffusionMegatronPolicyWorker`.
"""

from contextlib import AbstractContextManager
from math import lcm
from typing import Any, Iterator

import torch

from nemo_rl.models.megatron.data import (
    ProcessedMicrobatch,
    _get_non_packed_sequence_pad_factor,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.workers.diffusion_megatron_policy_worker import (
    DiffusionMegatronPolicyWorkerImpl,
)


class DiffuGRPOMegatronPolicyWorkerImpl(DiffusionMegatronPolicyWorkerImpl):
    """Megatron worker layer for the DiffuGRPO ``[noisy | clean]`` layout."""

    def _cfg_for_diffu_grpo_sequence(self, sequence_length: int) -> PolicyConfig:
        """Clone the policy config with the layout's own sequence length.

        The ``[noisy | clean]`` layout is roughly twice the rollout width and is
        only known once the batch has been built, so the microbatch iterator
        needs a config that advertises it.
        """
        cfg = self._cfg_without_sequence_packing()
        cfg["max_total_sequence_length"] = int(sequence_length)
        return cfg

    def _diffu_grpo_sequence_length_round(self) -> int:
        """Multiple the ``[noisy | clean]`` layout's total width is padded up to.

        Folds in Megatron's own dense pad factor (user alignment, FP8, sequence
        parallelism) so that ``_pad_sequence_aligned_tensors`` -- which right-pads
        every ``[B, S, ...]`` tensor of a dense microbatch -- is a no-op here.
        Without that, the microbatch would be widened *after* the layout was
        built, leaving the asymmetric-AR attention metadata
        (``noisy_length + clean_length``) describing a narrower sequence than the
        tensor the model actually sees.
        """
        requested = int(
            self.cfg.get("sequence_packing", {}).get("sequence_length_round", 64)
        )
        return lcm(requested, _get_non_packed_sequence_pad_factor(self.cfg))

    def _training_attention_context(self) -> AbstractContextManager[Any]:
        # Keep the diffusion attention module on its training forward. The
        # asymmetric [noisy | clean] metadata is supplied per microbatch, so no
        # fixed half-sequence mask length is needed.
        return self._megatron_attention_context("training")

    def _logprob_attention_context(self) -> AbstractContextManager[Any]:
        # Same as training: one asymmetric forward scores the whole layout.
        return self._megatron_attention_context("training")

    def _set_asymmetric_ar_metadata(
        self,
        microbatch: ProcessedMicrobatch,
    ) -> None:
        """Tell the attention module where the noisy/clean boundary sits.

        Raises:
            ValueError: If the noisy length, clean padded length or noisy
                response offset varies within the microbatch -- the attention
                mask is built once per microbatch, so they must be constant.
            RuntimeError: If the model's attention modules do not implement
                ``set_asymmetric_ar_metadata``.
        """
        data_dict = microbatch.data_dict
        if "diffu_grpo_noisy_lengths" not in data_dict:
            return

        noisy_lengths = data_dict["diffu_grpo_noisy_lengths"]
        noisy_valid_lengths = data_dict["diffu_grpo_noisy_valid_lengths"]
        clean_padded_lengths = data_dict["diffu_grpo_clean_padded_lengths"]
        noisy_response_offsets = data_dict["diffu_grpo_noisy_response_offsets"]
        if not torch.all(noisy_lengths == noisy_lengths[0]):
            raise ValueError(
                "diffuGRPO noisy length must be constant within a microbatch"
            )
        if not torch.all(clean_padded_lengths == clean_padded_lengths[0]):
            raise ValueError(
                "diffuGRPO clean padded length must be constant within a microbatch"
            )
        if not torch.all(noisy_response_offsets == noisy_response_offsets[0]):
            raise ValueError(
                "diffuGRPO noisy response offset must be constant within a microbatch"
            )

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
