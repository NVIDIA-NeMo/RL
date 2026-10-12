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
"""The advantage stage, separated from the controller that used to run it.

This is the compute half of ``SingleControllerActor._advantage_stage``. It was
lifted out for two reasons, both of which showed up at Ultra scale:

* It is the controller's largest transient allocation. It fetches every
  advantage input column for a whole cohort, and the controller's RSS burst
  tracked it exactly.
* It is ~200 lines of synchronous torch between two awaits, so while it ran
  nothing else on the controller's event loop could make progress -- including
  the Ray liveness ping, which is why the controller looked hung.

Neither is fixed by making the code faster; both are fixed by running it
somewhere else. Keeping the body here, rather than in the actor, lets the
controller drive it in-process when no actor pool is configured, so the two
paths cannot drift.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    _clip_grpo_advantages,
    compute_and_apply_seq_logprob_error_masking,
)
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.algorithms.single_controller_utils.utils import (
    AdvantagePartial,
    RewardPartial,
    apply_message_level_advantage_penalties,
    fields_for_put,
    squeeze_trailing_unit_dim,
    tensor_field,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.async_utils import call_data_plane
from nemo_rl.data_plane.grouping import group_index_column, row_group_ids
from nemo_rl.data_plane.schema import INPUT_IDS, INPUT_LENGTHS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.train_data_dump import TrainDataDump

if TYPE_CHECKING:
    # Annotation-only: importing ppo at runtime here would close a cycle
    # back through the controller.
    from nemo_rl.algorithms.ppo import PPOConfig


# Estimators whose advantages for a row depend only on that row and the rest
# of its prompt group, so a whole-group shard produces the same numbers as the
# whole batch. That holds because the baseline keys on GROUP_ID_TAG: while it
# keyed on prompt tokens, two groups sharing prompt text merged into one
# baseline when they landed in the same call and separated when a shard split
# them, which made grpo's advantages a function of num_advantage_workers.
# Everything else reachable from _build_advantage_estimator reduces over the
# entire batch -- gdpo and reinforce_plus_plus always, gae and raw_reward
# whenever normalize_advantages is set -- and a shard is not that batch.
# Membership is opt-in for exactly that reason: a new estimator counts as
# unshardable until someone checks it.
SHARD_INVARIANT_ESTIMATORS = frozenset({"grpo", "opd"})


@dataclass(frozen=True)
class AdvantageStageConfig:
    """Everything the stage reads off the controller, fixed at setup time."""

    advantage: AdvantageConfig
    algo: GRPOConfig | PPOConfig
    is_ppo: bool
    policy_logprobs_required: bool
    reference_logprobs_required: bool
    teacher_logprobs_required: bool
    message_level_advantage_penalties_enabled: bool
    shardable: bool
    # Where the per-step training dump writes, or None when it is disabled.
    # The stage owns the dump because it is the only place that still holds
    # the untruncated tensors: the controller stopped fetching them when the
    # computation moved here.
    train_data_dump_dir: Optional[str]

    @classmethod
    def from_master_config(cls, master_config: Any) -> AdvantageStageConfig:
        """Derive the stage's settings once, for the controller and the pool.

        The pool is built driver-side in ``setup_single_controller`` while the
        controller builds its in-process fallback in ``__init__``. Deriving the
        gates here is what keeps a remote call and a local call from disagreeing
        about which columns to fetch or which masks to apply.
        """
        # Deferred: this module is imported from the controller, and opd pulls
        # ray in transitively.
        from nemo_rl.algorithms import opd as opd_module
        from nemo_rl.algorithms.single_controller_utils.config import (
            algo_config,
            is_ppo_run,
        )

        is_ppo = is_ppo_run(master_config)
        algo_cfg = algo_config(master_config)
        loss_cfg = master_config.loss_fn
        return cls(
            advantage=AdvantageConfig(),
            algo=algo_cfg,
            is_ppo=is_ppo,
            policy_logprobs_required=not (
                loss_cfg.force_on_policy_ratio
                and algo_cfg.seq_logprob_error_threshold is None
            ),
            # _build_trainer initializes the reference model only for a positive
            # KL penalty, so this must use the same gate before requesting it.
            reference_logprobs_required=bool(
                loss_cfg.reference_policy_kl_penalty > 0
                and not algo_cfg.skip_reference_policy_logprobs_calculation
            ),
            teacher_logprobs_required=opd_module.is_opd_enabled(master_config),
            message_level_advantage_penalties_enabled=(
                algo_cfg.invalid_tool_call_advantage is not None
                or algo_cfg.malformed_thinking_advantage is not None
            ),
            shardable=algo_cfg.adv_estimator.name in SHARD_INVARIANT_ESTIMATORS,
            # logger is still a TypedDict, so this is the same value the
            # Logger exposes as base_log_dir.
            train_data_dump_dir=(
                master_config.logger["log_dir"]
                if master_config.async_rl.log_full_train_data
                else None
            ),
        )

    def input_fields(self) -> list[str]:
        """Return the advantage input columns to fetch, in a stable order."""
        adv_cfg = self.advantage
        fields = [
            adv_cfg.reward_field,
            adv_cfg.token_mask_field,
            adv_cfg.sample_mask_field,
            *adv_cfg.repeated_batch_fields,
            adv_cfg.mask_sample_field,
            adv_cfg.truncated_field,
        ]
        if self.message_level_advantage_penalties_enabled:
            fields.extend(
                [
                    adv_cfg.invalid_tool_call_mask_field,
                    adv_cfg.malformed_thinking_mask_field,
                ]
            )
        if self.policy_logprobs_required:
            fields.append(adv_cfg.policy_logprobs_field)
            fields.append(adv_cfg.generation_logprobs_field)
        if self.reference_logprobs_required:
            fields.append(adv_cfg.reference_logprobs_field)
        if self.teacher_logprobs_required:
            fields.append(adv_cfg.teacher_logprobs_field)
        if self.is_ppo:
            fields.append(adv_cfg.values_field)
        if self.train_data_dump_dir is not None:
            # The dump is the only remaining reader of the raw prompt tokens:
            # the baseline keys on the group-id tag now, so prompt_ids is not
            # otherwise fetched.
            fields.extend(
                [
                    INPUT_IDS,
                    INPUT_LENGTHS,
                    adv_cfg.prompt_ids_field,
                    adv_cfg.generation_logprobs_field,
                ]
            )
        return list(dict.fromkeys(fields))


@dataclass(frozen=True)
class AdvantageRequest:
    """One advantage-stage call, described without any payload.

    ``meta`` already names the rows; the tensors are fetched from DataPlane by
    whichever process runs the stage.
    """

    meta: KVBatchMeta
    # Optimizer step these rows belong to, carried because the training dump
    # names its file after it and the pool has no view of the controller's
    # counter. Unset when the dump is off, which is the default.
    train_step: Optional[int] = None


def split_meta_by_prompt_group(
    meta: KVBatchMeta, num_shards: int
) -> Optional[list[KVBatchMeta]]:
    """Split ``meta`` into at most ``num_shards`` metas of whole prompt groups.

    The group-relative estimators are only valid on complete prompt groups, so
    a split that cut one would corrupt the baseline silently instead of
    raising. This returns ``None`` -- meaning "do not shard" -- whenever the
    group layout cannot be established, and the caller falls back to one
    whole-batch call.

    Boundaries come from where ``GROUP_ID_TAG`` changes, so groups of unequal
    size are cut correctly. Deriving them from ``num_generations_per_prompt``
    instead assumed every group had exactly that many rows, which silently
    mis-cut any feature that produces variable-size groups. An even earlier
    version read ``DATASET_SOURCE_TAG``, which ``TQReplayBuffer.commit`` only
    stamps when the dataset row carries a ``dataset`` key; on a dataset without
    one the tag was absent, every split declined, and the fallback was silent
    enough that a whole 256-node run measured nothing.

    Chunks are contiguous, so concatenating the shards reproduces the original
    row order and the caller never has to rebuild a permutation. A layout that
    interleaves groups has no contiguous whole-group cut, so it declines rather
    than reordering rows behind the caller's back.
    """
    if num_shards <= 1:
        return None
    group_ids = row_group_ids(meta)
    starts = [0] + [
        i for i in range(1, len(group_ids)) if group_ids[i] != group_ids[i - 1]
    ]
    num_groups = len(starts)
    if num_groups < 2:
        return None
    if num_groups != len(set(group_ids)):
        return None
    # Balanced by group count rather than by row count: groups are equal-sized
    # in every current producer, and balancing by rows would buy a second-order
    # gain on the variable-size case only.
    groups_per_shard = -(-num_groups // num_shards)
    bounds = starts + [len(group_ids)]
    shards = [
        meta.slice(bounds[begin], bounds[min(begin + groups_per_shard, num_groups)])
        for begin in range(0, num_groups, groups_per_shard)
    ]
    return shards if len(shards) > 1 else None


@dataclass(frozen=True)
class AdvantageOutcome:
    """One call's results, every one of them already reduced to metadata.

    The stage used to hand the controller whole tensors to accumulate. It now
    hands back only what the step-close reduction actually reads, which is what
    keeps this small enough to cross a Ray RPC boundary.
    """

    meta: KVBatchMeta
    has_valid_training_tokens: bool
    num_mask_sample_filtered: int
    reward_partial: RewardPartial
    advantage_partial: AdvantagePartial
    seq_logprob_error_metrics: Optional[dict[str, float]] = None
    # OPD's running moments, as this call's contribution rather than a total.
    opd_stat_sum: float = 0.0
    opd_stat_sumsq: float = 0.0
    opd_stat_count: int = 0
    opd_gap_sum: float = 0.0
    # Seconds this call spent serializing the training dump. Reported back
    # because the writing moved off the controller with the rest of the
    # stage, and the controller still owns the timer the metric lands in.
    train_data_dump_s: float = 0.0
    # Rows this call wrote to its shard's part file, so the controller's merge
    # can tell a missing part from an empty one.
    train_data_dump_rows: int = 0


class AdvantageComputer:
    """Fetch advantage inputs, compute advantages, and write them back.

    The selected ``KVBatchMeta`` still contains complete prompt groups before
    trainer DP sharding, which is what makes the group-relative estimators
    valid here. Tensor payloads only ever move through DataPlane: this fetches
    the configured advantage input columns and writes the computed
    ``advantages`` column back under the same ``sample_ids``.
    """

    def __init__(
        self,
        dp_client: Any,
        *,
        config: AdvantageStageConfig,
        advantage_estimator: Any,
        shard_id: str = "0",
    ) -> None:
        self._dp_client = dp_client
        self._config = config
        self._advantage_estimator = advantage_estimator
        # One writer per shard: the pool runs these concurrently, so they must
        # not share a file. The controller merges them on publish.
        self._train_data_dump = (
            TrainDataDump(config.train_data_dump_dir, shard_id=shard_id)
            if config.train_data_dump_dir is not None
            else None
        )

    async def run(self, request: AdvantageRequest) -> AdvantageOutcome:
        """Run one call's advantage stage and return its reduced results."""
        meta = request.meta
        cfg = self._config
        adv_cfg = cfg.advantage

        data = await call_data_plane(
            self._dp_client,
            "get_samples",
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=cfg.input_fields(),
        )

        # Group by the group a row was generated in, not by its prompt tokens.
        # Two groups can carry identical prompt text -- DAPO-Math-17k stores
        # each prompt 100 times -- and the token key gave them one shared
        # baseline whenever they landed in the same call, so the result moved
        # with chunk boundaries and with how the shards fell.
        prompt_ids = group_index_column(row_group_ids(meta))
        rewards = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.reward_field)
        ).float()
        token_mask = tensor_field(data, adv_cfg.token_mask_field).float()
        sample_mask = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.sample_mask_field)
        ).float()
        mask_sample = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.mask_sample_field)
        ).bool()
        truncated = squeeze_trailing_unit_dim(
            tensor_field(data, adv_cfg.truncated_field)
        ).bool()

        num_mask_sample_filtered = int(mask_sample.sum().item())
        final_sample_mask = sample_mask * (~mask_sample).to(sample_mask.dtype)
        if cfg.algo.overlong_filtering:
            final_sample_mask = final_sample_mask * (~truncated).to(sample_mask.dtype)

        pre_seq_error_sample_mask = final_sample_mask.clone()

        seq_error_metrics: Optional[dict[str, float]] = None
        # Match the legacy path: whenever real policy logprobs are available,
        # report sequence-level generation/training mismatch. A threshold adds
        # masking; leaving it unset keeps this metrics-only.
        if cfg.policy_logprobs_required:
            masking_data = BatchedDataDict(
                {
                    "token_mask": token_mask,
                    "sample_mask": final_sample_mask,
                    "prev_logprobs": tensor_field(
                        data,
                        adv_cfg.policy_logprobs_field,
                    ),
                    "generation_logprobs": tensor_field(
                        data,
                        adv_cfg.generation_logprobs_field,
                    ),
                }
            )
            num_valid_seqs_before = float(
                ((token_mask[:, 1:] * final_sample_mask.unsqueeze(-1)).sum(dim=-1) > 0)
                .sum()
                .item()
            )
            seq_error_metrics = compute_and_apply_seq_logprob_error_masking(
                train_data=masking_data,
                rewards=rewards,
                seq_logprob_error_threshold=cfg.algo.seq_logprob_error_threshold,
            )
            final_sample_mask = masking_data["sample_mask"]
            num_valid_seqs_after = float(
                ((token_mask[:, 1:] * final_sample_mask.unsqueeze(-1)).sum(dim=-1) > 0)
                .sum()
                .item()
            )
            seq_error_metrics["num_masked_seqs_by_logprob_error"] = (
                seq_error_metrics.pop("num_masked_seqs")
            )
            seq_error_metrics["_num_valid_seqs_before"] = num_valid_seqs_before
            seq_error_metrics["_num_valid_seqs_after"] = num_valid_seqs_after

        mask = token_mask * final_sample_mask.unsqueeze(-1)

        repeated_batch: dict[str, torch.Tensor] = {
            "total_reward": rewards,
        }
        for field_name in adv_cfg.repeated_batch_fields:
            repeated_batch[field_name] = squeeze_trailing_unit_dim(
                tensor_field(data, field_name)
            )

        kwargs: dict[str, torch.Tensor] = {}
        if cfg.policy_logprobs_required:
            policy_logprobs = tensor_field(data, adv_cfg.policy_logprobs_field)
            if cfg.teacher_logprobs_required:
                kwargs["prev_logprobs"] = policy_logprobs
            else:
                kwargs["logprobs_policy"] = policy_logprobs
        if cfg.reference_logprobs_required:
            kwargs["logprobs_reference"] = tensor_field(
                data,
                adv_cfg.reference_logprobs_field,
            )
        if cfg.teacher_logprobs_required:
            kwargs["teacher_logprobs"] = tensor_field(
                data,
                adv_cfg.teacher_logprobs_field,
            )
        if cfg.is_ppo:
            kwargs["values"] = tensor_field(data, adv_cfg.values_field)

        # Training predicts token t from position t - 1, so token_mask[:, 1:]
        # is the exact mask used when global_valid_toks and the loss are built.
        has_valid_training_tokens = bool(mask[:, 1:].bool().any().item())
        # Value-model estimators (GAE) hand back the regression target alongside
        # the advantages; the group-relative ones return a bare tensor.
        returns: Optional[torch.Tensor] = None
        if has_valid_training_tokens:
            result = self._advantage_estimator.compute_advantage(
                prompt_ids=prompt_ids,
                rewards=rewards,
                mask=mask,
                repeated_batch=repeated_batch,
                # Real validity (token-capture placeholders carry sample_mask 0,
                # and mask_sample/overlong/seq-logprob-error rows are folded in
                # via final_sample_mask) instead of the hardwired all-ones.
                valid_mask=final_sample_mask,
                **kwargs,
            )
            if cfg.is_ppo:
                advantages, returns = result
            else:
                advantages = result
        else:
            advantages = torch.zeros_like(mask)
            if cfg.is_ppo:
                returns = torch.zeros_like(mask)

        if cfg.message_level_advantage_penalties_enabled:
            # Sequence-error filtering and the pre-existing sample mask remain
            # authoritative: a message penalty must not make a filtered token
            # trainable again.
            valid_tokens = mask.bool()
            advantages = apply_message_level_advantage_penalties(
                advantages,
                invalid_tool_call_mask=(
                    tensor_field(data, adv_cfg.invalid_tool_call_mask_field).bool()
                    & valid_tokens
                ),
                malformed_thinking_mask=(
                    tensor_field(data, adv_cfg.malformed_thinking_mask_field).bool()
                    & valid_tokens
                ),
                invalid_tool_call_advantage=cfg.algo.invalid_tool_call_advantage,
                malformed_thinking_advantage=cfg.algo.malformed_thinking_advantage,
            )

        response_advantages = torch.masked_select(advantages, mask.bool())
        reward_partial = RewardPartial.from_rows(rewards, final_sample_mask)
        opd_stat_sum = 0.0
        opd_stat_sumsq = 0.0
        opd_stat_count = 0
        opd_gap_sum = 0.0
        if cfg.teacher_logprobs_required:
            valid = response_advantages.detach().double()
            opd_stat_sum = float(valid.sum())
            opd_stat_sumsq = float((valid * valid).sum())
            opd_stat_count = int(valid.numel())
            # Pooled over the same tokens as the advantage; the gap metric must
            # not change when TROPD or the global baseline reshapes the advantage.
            raw_gap = torch.masked_select(
                kwargs["teacher_logprobs"] - kwargs["prev_logprobs"], mask.bool()
            )
            opd_gap_sum = float(raw_gap.detach().double().sum())

        # OPD accumulates its statistics from the estimator output above. The
        # ordinary advantage metrics and policy training use the clipped values,
        # matching the legacy paths.
        if not cfg.is_ppo:
            assert isinstance(cfg.algo, GRPOConfig)
            advantages = _clip_grpo_advantages(advantages, cfg.algo)
            response_advantages = torch.masked_select(advantages, mask.bool())
        advantage_partial = AdvantagePartial.from_values(response_advantages)

        train_data_dump_s = 0.0
        train_data_dump_rows = 0
        if self._train_data_dump is not None:
            assert request.train_step is not None
            dump_started = time.perf_counter()
            sequences = {
                "token_ids": tensor_field(data, INPUT_IDS),
                "token_loss_mask": token_mask,
                "advantages": advantages,
                "generation_logprobs": tensor_field(
                    data, adv_cfg.generation_logprobs_field
                ),
            }
            if cfg.policy_logprobs_required:
                sequences["prev_logprobs"] = tensor_field(
                    data, adv_cfg.policy_logprobs_field
                )
            if cfg.teacher_logprobs_required:
                sequences["teacher_logprobs"] = kwargs["teacher_logprobs"]
            await asyncio.to_thread(
                self._train_data_dump.add_chunk,
                step=request.train_step,
                sample_ids=list(meta.sample_ids),
                tags=meta.tags,
                input_lengths=tensor_field(data, INPUT_LENGTHS),
                sequences=sequences,
                scalars={
                    "sample_loss_mask": final_sample_mask,
                    "pre_seq_error_sample_loss_mask": pre_seq_error_sample_mask,
                    "rewards": rewards,
                    # Raw column: jagged when the chunk mixes prompt
                    # lengths, so rows carry no zero padding. Not the
                    # group-index key the estimator reduces over.
                    "prompt_ids": data[adv_cfg.prompt_ids_field],
                },
            )
            train_data_dump_s = time.perf_counter() - dump_started
            train_data_dump_rows = len(meta.sample_ids)

        fields_to_put = {adv_cfg.output_field: advantages}
        if not torch.equal(final_sample_mask, sample_mask):
            fields_to_put[adv_cfg.sample_mask_field] = final_sample_mask
        new_fields = [adv_cfg.output_field]
        if returns is not None:
            fields_to_put[adv_cfg.returns_field] = returns
            new_fields.append(adv_cfg.returns_field)

        await call_data_plane(
            self._dp_client,
            "put_samples",
            offload_sync=True,
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            fields=fields_for_put(meta, fields_to_put),
        )
        return AdvantageOutcome(
            meta=meta.with_fields(new_fields),
            has_valid_training_tokens=has_valid_training_tokens,
            num_mask_sample_filtered=num_mask_sample_filtered,
            reward_partial=reward_partial,
            advantage_partial=advantage_partial,
            seq_logprob_error_metrics=seq_error_metrics,
            opd_stat_sum=opd_stat_sum,
            opd_stat_sumsq=opd_stat_sumsq,
            opd_stat_count=opd_stat_count,
            opd_gap_sum=opd_gap_sum,
            train_data_dump_s=train_data_dump_s,
            train_data_dump_rows=train_data_dump_rows,
        )
