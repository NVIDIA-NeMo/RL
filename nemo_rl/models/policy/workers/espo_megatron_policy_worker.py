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

from typing import Any, Iterator, Optional

import ray
import torch

from nemo_rl.algorithms.coupled_grpo_logprobs import (
    CoupledGRPORevealSchedule,
    build_coupled_base,
    make_coupled_level_view,
)
from nemo_rl.algorithms.espo_logprobs import get_espo_logprob_estimation_cfg
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.coupled_grpo_megatron_policy_worker import (
    CoupledGRPOMegatronPolicyWorkerImpl,
)

# ESPO is the antithetic coupled pair: level 0 masks M, level 1 the complement.
ESPO_COUPLED_NUM_LEVELS = 2


class ESPOMegatronPolicyWorkerImpl(CoupledGRPOMegatronPolicyWorkerImpl):
    """Block-aware ESPO logprobs from the antithetic coupled mask pair.

    Subclasses the CoupledGRPO worker. Level 0 masks a per-sample random subset
    ``M`` of the response (ratio ``t ~ U(0.2, 0.8)``), level 1 the exact complement
    ``Mbar`` (every response token is masked in exactly one level), scored over the
    ``[noisy | clean]`` asymmetric-AR layout. ``get_logprobs`` runs BOTH levels and
    returns the summed raw ``[N, S]`` logprobs (each response token's logprob
    appears once); the loss reduces curr / prev / ref per level with each level's
    own harvest mask, then averages the two ELBOs (scheme (b)) -- so prev / ref
    need NO scalar plumbing and stay ``[N, S]``. The ESPO objective is separable
    ACROSS sequences (sum over n) and only coupled WITHIN a sequence (its two levels
    must combine to form one ratio), so training uses SAMPLE-MAJOR microbatching:
    each microbatch holds ``num_samples_per_micro_batch`` (K) whole sequences with
    their ``num_mc_samples`` level rows grouped (``[s0L0, s0L1, s1L0, ...]``), and
    the gradient accumulates over ``per_rank_sequences / K`` microbatches (the
    standard NeMo-RL path). The loss sees both levels of each of its K sequences,
    so it computes the genuine per-sequence loss self-contained -- no cross-
    microbatch aggregation. Memory is ``K * num_mc_samples`` rows per forward
    (K = 1 -> 2 rows, like CoupledGRPO). Requires
    ``train_micro_batch_size == num_samples_per_micro_batch * num_mc_samples``. The
    mask is seeded per row from ``data["coupled_grpo_seed"]`` so the SAME
    realization feeds curr / prev / ref (a valid ESPO sequence ratio). Two forwards
    per level on every rank -> DP-uniform.
    """

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("ESPO")
        get_espo_logprob_estimation_cfg(self.cfg)

    def _coupled_cfg(self):
        return get_espo_logprob_estimation_cfg(self.cfg)

    def _espo_num_masks(self) -> int:
        """Monte-Carlo masks per sequence. ESPO is the antithetic coupled pair, so
        only ``num_mc_samples == 2`` is supported (defaults to 2). MC > 2 is
        deferred; values other than 2 are rejected."""
        num = int(self._coupled_cfg().get("num_mc_samples", ESPO_COUPLED_NUM_LEVELS))
        if num != ESPO_COUPLED_NUM_LEVELS:
            raise ValueError(
                "ESPO requires num_mc_samples == 2 (antithetic coupled pair); "
                f"got {num}. MC > 2 is not yet supported."
            )
        return num

    def _espo_samples_per_micro_batch(self) -> int:
        """Whole sequences per training microbatch (K); defaults to 1. Each carries
        its ``num_mc_samples`` level rows, so a microbatch is ``K * num_mc_samples``
        rows and the gradient accumulates over ``per_rank_sequences / K``
        microbatches."""
        k = int(self._coupled_cfg().get("num_samples_per_micro_batch", 1))
        if k < 1:
            raise ValueError(
                f"num_samples_per_micro_batch must be >= 1, got {k}"
            )
        return k

    # ---- training: SAMPLE-MAJOR microbatches + gradient accumulation -------
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._coupled_cfg()
        num_levels = self._espo_num_masks()  # validate num_mc_samples == 2
        k = self._espo_samples_per_micro_batch()
        # The ESPO objective is separable across sequences and coupled only within
        # a sequence's num_levels rows, so each microbatch holds K WHOLE sequences
        # (their level rows grouped) and the gradient accumulates over the rest.
        # train_micro_batch_size must therefore be exactly K * num_levels rows.
        if mbs != k * num_levels:
            raise ValueError(
                "ESPO requires train_micro_batch_size == num_samples_per_micro_batch"
                f" * num_mc_samples = {k} * {num_levels} = {k * num_levels}, got "
                f"train_micro_batch_size={mbs}."
            )
        block_size = self._diffusion_block_size()
        self._maybe_print_diffusion_block_size("espo_block_aware_train", block_size)
        base, num_samples, num_built = build_coupled_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            noisy_block_size=block_size,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=True,
        )
        # K must divide the per-rank sequence count so num_microbatches =
        # num_samples / K is an integer (DP-uniform microbatch count).
        if num_samples and num_samples % k != 0:
            raise ValueError(
                "ESPO num_samples_per_micro_batch must divide the per-rank sequence "
                f"count; got K={k} and {num_samples} sequences on this rank."
            )
        harvest_keys = ("diffu_grpo_score_mask", "diffu_grpo_loss_mask")
        # SAMPLE-MAJOR schedule: lazily yields one microbatch per K-sample group,
        # interleaving the K samples' num_levels coupled level views into rows
        # [s0L0, s0L1, s1L0, s1L1, ...]. Mirrors CoupledGRPORevealSchedule (lazy,
        # one optimizer step over the whole schedule) but groups by sample (K) so a
        # sequence's levels stay together; gradient accumulates over num_samples / K
        # microbatches. The loss reshapes each [K*num_levels] microbatch to [K, M].
        num_built = min(num_built, ESPO_COUPLED_NUM_LEVELS) if num_built else 0
        schedule = ESPORevealSchedule(base).configure(
            num_levels=num_built,
            harvest_keys=harvest_keys,
            num_samples_per_micro_batch=k,
        )
        return (
            schedule,
            self._cfg_for_diffu_grpo_sequence(base["input_ids"].shape[1]),
            mbs,
            {},
        )

    # ---- logprobs: both complementary levels, summed -----------------------
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        # Run BOTH complementary levels and sum -- every response token is harvested
        # in exactly one level, so the summed [N, S] holds each token's logprob
        # once. The loss reduces it per level with each level's harvest mask, so
        # prev / ref stay [N, S] (no scalar plumbing). One forward per level on
        # every rank keeps the pass count DP-uniform.
        self._espo_num_masks()  # validate num_mc_samples == 2
        return self._coupled_logprobs(
            data=data, micro_batch_size=micro_batch_size, only_level0=False
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("espo_megatron_policy_worker")
)  # pragma: no cover
class ESPOMegatronPolicyWorker(ESPOMegatronPolicyWorkerImpl):
    pass


def _interleave_sample_major(
    views: list[BatchedDataDict[Any]],
) -> BatchedDataDict[Any]:
    """Interleave ``M`` equal-length level views into one SAMPLE-MAJOR batch.

    View ``j`` holds ``N`` rows (sequence-aligned); the result holds ``M * N``
    rows ordered ``[s0L0..s0L{M-1}, s1L0.., ...]`` so each sequence's ``M``
    mask-variant rows are contiguous. ``ESPORevealSchedule`` calls this per
    K-sample group to form one ``K * M``-row microbatch. Per key, tensor values
    are stacked on a new level axis and flattened; list values are interleaved
    element-wise.
    """
    out = BatchedDataDict[Any]()
    for key in views[0].keys():
        values = [view[key] for view in views]
        first = values[0]
        if torch.is_tensor(first):
            stacked = torch.stack(values, dim=1)  # [N, M, ...]
            out[key] = stacked.reshape((-1,) + tuple(stacked.shape[2:]))
        else:
            out[key] = [item for group in zip(*values) for item in group]
    return out


class ESPORevealSchedule(CoupledGRPORevealSchedule):
    """Lazy SAMPLE-MAJOR coupled schedule for ESPO training.

    Mirrors ``CoupledGRPORevealSchedule`` (a ``RevealLevelSchedule`` holding the
    coupled ``base`` of N samples, presented to Megatron as the training batch for
    one optimizer step) but yields microbatches SAMPLE-major, K samples at a time,
    instead of level-major. Each microbatch is the ``num_samples_per_micro_batch``
    (K) samples' ``num_levels`` coupled level views interleaved into
    ``[s0L0, s0L1, s1L0, s1L1, ...]`` (``K * num_levels`` rows), so a sequence's
    levels stay together and the loss can reduce them per microbatch. The gradient
    accumulates over ``num_samples / K`` microbatches.
    """

    def configure(
        self,
        *,
        num_levels: int,
        harvest_keys: tuple[str, ...],
        num_samples_per_micro_batch: int,
    ) -> "ESPORevealSchedule":
        self._configure_levels(num_levels=num_levels, harvest_keys=harvest_keys)
        self._rl_samples_per_micro_batch = int(num_samples_per_micro_batch)
        return self

    @property
    def size(self) -> int:
        # One full sample batch per level; get_microbatch_iterator divides this by
        # microbatch_size (= K * num_levels) to get num_microbatches = N / K.
        return self._rl_num_levels * self._sample_count()

    def make_microbatch_iterator(
        self, microbatch_size: int
    ) -> Iterator[BatchedDataDict[Any]]:
        k = self._rl_samples_per_micro_batch
        if microbatch_size != k * self._rl_num_levels:
            raise ValueError(
                "ESPORevealSchedule expects microbatch_size == "
                "num_samples_per_micro_batch * num_levels = "
                f"{k} * {self._rl_num_levels} = {k * self._rl_num_levels}, got "
                f"{microbatch_size}."
            )
        num_samples = self._sample_count()
        for start in range(0, num_samples, k):
            base_slice = self.slice(start, start + k)
            views = [
                make_coupled_level_view(base_slice, level, self._rl_harvest_keys)
                for level in range(self._rl_num_levels)
            ]
            yield _interleave_sample_major(views)
