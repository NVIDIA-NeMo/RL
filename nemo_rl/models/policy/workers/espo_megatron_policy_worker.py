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
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.espo_train import ESPOLossPostProcessor
from nemo_rl.models.megatron.train import LossPostProcessor
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.coupled_grpo_megatron_policy_worker import (
    CoupledGRPOMegatronPolicyWorkerImpl,
)


class ESPOMegatronPolicyWorkerImpl(CoupledGRPOMegatronPolicyWorkerImpl):
    """Block-aware ESPO logprobs from K antithetic coupled mask pairs.

    Subclasses the CoupledGRPO worker. Each of K = ``num_mc_samples`` // 2 pairs draws
    a per-sample random subset ``M`` of the response (ratio ``t ~ U(0.2, 0.8)``) for
    its even level and the exact complement ``Mbar`` for its odd level (every response
    token is masked in exactly one of a pair's two levels), scored over the
    ``[noisy | clean]`` asymmetric-AR layout. ``get_logprobs`` runs all 2K levels and
    returns each pair's within-pair summed raw ``[N, S]`` logprobs (each response
    token's logprob appears once per pair): pair 0's sum is ``["logprobs"]``, pairs
    1..K-1 are ``logprobs_pair{p}``. The loss routes pair = level // 2's tensor onto
    each level row and reduces curr / prev / ref per level with each level's own
    harvest mask, then averages all 2K ELBOs (scheme (b)) -- so prev / ref need NO
    scalar plumbing and stay ``[N, S]`` per pair. The ESPO objective is separable
    ACROSS sequences (sum over n) and only coupled WITHIN a sequence (its two levels
    must combine to form one ratio), so training uses SAMPLE-MAJOR microbatching:
    each microbatch holds ``num_samples_per_micro_batch`` (K) whole sequences with
    their ``num_mc_samples`` level rows grouped (``[s0L0, s0L1, s1L0, ...]``), and
    the gradient accumulates over ``per_rank_sequences / K`` microbatches (the
    standard NeMo-RL path). The loss sees both levels of each of its K sequences,
    so it computes the genuine per-sequence loss self-contained -- no cross-
    microbatch aggregation. Memory is ``K_seq * num_mc_samples`` rows per training
    forward (K_seq = num_samples_per_micro_batch; = 2 rows at K_seq = 1 and one pair,
    like CoupledGRPO). Requires ``train_micro_batch_size == num_samples_per_micro_batch
    * num_mc_samples``. Each pair's masks are seeded per row from a sub-seed of
    ``data["coupled_grpo_seed"]`` so the SAME realization feeds curr / prev / ref (a
    valid ESPO sequence ratio); at one pair the seed stream is identical to
    CoupledGRPO. prev / ref issue 2K forwards each on every rank -> DP-uniform.
    """

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("ESPO")
        get_espo_logprob_estimation_cfg(self.cfg)

    def _coupled_cfg(self):
        return get_espo_logprob_estimation_cfg(self.cfg)

    def _coupled_pair_count_scale(self) -> int:
        # ESPO's loss averages the 2K per-level ELBOs into ONE per-sequence value
        # (compute_coupled_block_aware_elbo divides by num_masks), so each sequence
        # contributes exactly once to global_valid_seqs -- no pair scaling (overrides
        # the CoupledGRPO base, which scales for its level-major token accumulation).
        return 1

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

    def _make_loss_post_processor(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int,
    ) -> LossPostProcessor:
        return ESPOLossPostProcessor(
            loss_fn=loss_fn,
            cfg=cfg,
            num_microbatches=num_microbatches,
            sampling_params=self.sampling_params,
        )

    # ---- training: SAMPLE-MAJOR microbatches + gradient accumulation -------
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._coupled_cfg()
        num_levels = self._coupled_num_levels()  # validate num_mc_samples even, >= 2
        num_pairs = self._coupled_num_pairs()
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
        # Build the base with K coupled pairs (2 * num_pairs levels). include_loss
        # scatters pair 0's prev / reference logprobs (the standard [N, S] keys) into
        # the noisy layout; the pairs 1..K-1 tensors ride as extra keys and are
        # scattered below onto per-pair base fields.
        base, num_samples, num_built = build_coupled_base(
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
        # K must divide the per-rank sequence count so num_microbatches =
        # num_samples / K is an integer (DP-uniform microbatch count).
        if num_samples and num_samples % k != 0:
            raise ValueError(
                "ESPO num_samples_per_micro_batch must divide the per-rank sequence "
                f"count; got K={k} and {num_samples} sequences on this rank."
            )
        # Scatter pairs 1..K-1 prev / reference logprobs into the noisy layout (pair
        # 0 is the standard prev_logprobs / reference_policy_logprobs base field). The
        # ESPO level view routes pair = level // 2's tensor onto each level row so the
        # loss reshape [K_seq, num_masks, S] has row 2p / 2p+1 carry pair p's summed
        # logprob (Approach A: per-pair sum is lossless -- a pair's two complementary
        # levels partition the tokens).
        if num_samples:
            self._scatter_pair_logprobs(base, data, num_pairs)
        harvest_keys = ("diffu_grpo_score_mask", "diffu_grpo_loss_mask")
        # SAMPLE-MAJOR schedule: lazily yields one microbatch per K-sample group,
        # interleaving the K samples' num_levels coupled level views into rows
        # [s0L0, s0L1, s1L0, s1L1, ...]. Mirrors CoupledGRPORevealSchedule (lazy,
        # one optimizer step over the whole schedule) but groups by sample (K) so a
        # sequence's levels stay together; gradient accumulates over num_samples / K
        # microbatches. The loss reshapes each [K*num_levels] microbatch to [K, M].
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
            # make_coupled_level_view routes pair = level // 2's prev / reference
            # logprobs onto each level row from the base's per-pair fields (scattered
            # by _scatter_pair_logprobs), so no per-view routing is needed here.
            views = [
                make_coupled_level_view(base_slice, level, self._rl_harvest_keys)
                for level in range(self._rl_num_levels)
            ]
            yield _interleave_sample_major(views)
