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

"""Hybrid AR + diffusion Megatron policy worker.

Trains both generation modes from one forward pass over the asymmetric
``[noisy | clean]`` layout: the clean half carries the GRPO policy-gradient term
(ordinary causal attention, so it is a teacher-forced AR forward), the noisy half
carries a masked cross-entropy term. See
``nemo_rl.algorithms.hybrid_ar_diffusion``.

Select it with::

    policy:
      worker_extension_cls_fqn: nemo_rl.models.policy.workers.hybrid_ar_diffusion_megatron_policy_worker.HybridARDiffusionMegatronPolicyWorker
      logprob_estimation:
        type: hybrid_ar_diffusion
        mask_token_id: <the model's MASK id>
        ce_loss_weight: 0.1
"""

from typing import Any, Optional

import ray
import torch

from nemo_rl.algorithms.hybrid_ar_diffusion import (
    build_hybrid_ar_diffusion_batch,
    get_hybrid_ar_diffusion_cfg,
    unscatter_clean_aligned,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
    HybridARDiffusionLogprobsPostProcessor,
    HybridARDiffusionLossPostProcessor,
)
from nemo_rl.models.policy import (
    HybridARDiffusionLogprobEstimationConfig,
    PolicyConfig,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffu_grpo_megatron_policy_worker import (
    DiffuGRPOMegatronPolicyWorkerImpl,
)


class HybridARDiffusionMegatronPolicyWorkerImpl(DiffuGRPOMegatronPolicyWorkerImpl):
    """Policy worker for joint AR policy-gradient + diffusion cross-entropy.

    Both ``get_logprobs`` and ``train`` route through the *same* asymmetric
    forward path. That is deliberate: the clean-half logprobs are mathematically
    a plain causal forward, but computing ``prev_logprobs`` through a different
    kernel (the pure-causal path) than the training forward (flex attention)
    leaves a small numerical gap. That gap makes the importance ratio differ from
    1 on the first inner step and injects gradient that is not real signal, so
    both passes must use one path.
    """

    def _validate_diffusion_algorithm_support(self) -> None:
        # Keep the inherited context-parallel and draft-model guards; only the
        # estimator check is specific to this worker.
        self._validate_diffusion_support("HybridARDiffusion")
        get_hybrid_ar_diffusion_cfg(self.cfg)

    def _hybrid_cfg(self) -> HybridARDiffusionLogprobEstimationConfig:
        return get_hybrid_ar_diffusion_cfg(self.cfg)

    def _hybrid_block_size(self) -> int:
        return self._diffusion_block_size()

    def _build_hybrid_batch(
        self,
        data: BatchedDataDict[Any],
    ) -> BatchedDataDict[Any]:
        """Build the ``[noisy | clean]`` batch with a random noisy-half mask."""
        cfg = self._hybrid_cfg()
        block_size = self._hybrid_block_size()
        if "hybrid_mask_seed" not in data:
            # The clean half -- the only half get_logprobs reads -- does not
            # attend to the noisy half, so its logprobs are independent of the
            # mask. A constant seed is therefore sufficient on paths where the
            # GRPO loop has not attached one.
            data["hybrid_mask_seed"] = torch.zeros(
                data["input_ids"].shape[0], dtype=torch.long
            )
        return build_hybrid_ar_diffusion_batch(
            data,
            mask_token_id=cfg.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_ratio_min=cfg.mask_ratio_min,
            mask_ratio_max=cfg.mask_ratio_max,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            block_size=block_size,
            noisy_tail_mode=cfg.noisy_tail_mode,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        self._maybe_print_diffusion_block_size(
            "hybrid_ar_diffusion_train", self._hybrid_block_size()
        )
        batch = self._build_hybrid_batch(data)
        return (
            batch,
            self._cfg_for_diffu_grpo_sequence(batch["input_ids"].shape[1]),
            mbs,
            {"original_seq_len": data["input_ids"].shape[1]},
        )

    def _build_logprob_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int],
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        # get_logprobs is called with micro_batch_size=None on the production
        # path; mirror the parent's fallback or None reaches
        # get_microbatch_iterator and policy.logprob_batch_size is never read.
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        batch = self._build_hybrid_batch(data)
        return (
            batch,
            self._cfg_for_diffu_grpo_sequence(batch["input_ids"].shape[1]),
            logprob_batch_size,
            {"original_seq_len": data["input_ids"].shape[1]},
        )

    def _make_loss_post_processor(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int,
    ) -> HybridARDiffusionLossPostProcessor:
        return HybridARDiffusionLossPostProcessor(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            sampling_params=self.sampling_params,
        )

    def _make_logprobs_post_processor(
        self,
        cfg: PolicyConfig,
    ) -> HybridARDiffusionLogprobsPostProcessor:
        # The stock LogprobsPostProcessor gathers next-token logprobs against
        # input_ids; the hybrid needs a same-position gather against
        # hybrid_target_ids over the full layout so the clean (AR) half
        # survives.
        return HybridARDiffusionLogprobsPostProcessor(
            cfg=cfg,
            sampling_params=self.sampling_params,
            use_fused_linear_logprobs=False,
        )

    def _finalize_logprobs_from_outputs(
        self,
        list_of_logprobs: list[dict[str, torch.Tensor]],
        *,
        original_data: BatchedDataDict[Any],
        transformed_data: BatchedDataDict[Any],
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Map clean-half next-token logprobs back to the original ``[N, S]`` layout.

        Delegates to :func:`unscatter_clean_aligned`, which is the exact inverse
        of the builder's clean-side alignment and lives beside it so the
        round-trip can be unit-tested without importing Megatron.
        """
        all_logprobs = torch.cat(
            [lp_dict["logprobs"] for lp_dict in list_of_logprobs], dim=0
        )
        return unscatter_clean_aligned(
            all_logprobs, transformed_data, metadata["original_seq_len"]
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker(
        "hybrid_ar_diffusion_megatron_policy_worker"
    )
)  # pragma: no cover
class HybridARDiffusionMegatronPolicyWorker(HybridARDiffusionMegatronPolicyWorkerImpl):
    pass
