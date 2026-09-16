# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ray payload construction and output ownership for dynamic context parallelism."""

import logging
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.dynamic_context_parallel import (
    CPRankGroup,
    CPRankPlan,
    CPRankStep,
    CPSyncGroup,
    DynamicContextParallelConfig,
    assignments_for_lane,
    plan_cp_phases,
)
from nemo_rl.distributed.named_sharding import NamedSharding

logger = logging.getLogger(__name__)


def dynamic_cp_config(cfg: dict[str, Any]) -> DynamicContextParallelConfig | None:
    """Read optional config without introducing defaults at worker call sites."""
    megatron = cfg.get("megatron_cfg")
    raw = megatron.get("dynamic_context_parallel") if megatron is not None else None
    if raw is None:
        return None
    parsed = DynamicContextParallelConfig.model_validate(raw)
    return parsed if parsed.enabled else None


def validate_dynamic_cp(cfg: dict[str, Any], *, lanes: int) -> None:
    """Reject incompatible execution paths before launching model collectives."""
    dynamic = dynamic_cp_config(cfg)
    if dynamic is None:
        return
    mc = cfg["megatron_cfg"]
    if not mc["enabled"] or mc["pipeline_model_parallel_size"] != 1:
        raise ValueError("Dynamic CP requires Megatron with PP=1")
    if not cfg["sequence_packing"]["enabled"] or cfg["dynamic_batching"]["enabled"]:
        raise ValueError(
            "Dynamic CP requires sequence_packing and disables dynamic_batching"
        )
    if mc.get("use_fused_linear_logprobs") or cfg.get("draft", {}).get("enabled"):
        raise ValueError(
            "Dynamic CP does not support fused linear logprobs or draft training"
        )
    if mc.get("cuda_graph_impl") not in (None, "none"):
        raise ValueError("Dynamic CP does not support CUDA graph capture")
    if mc.get("mtp_num_layers") or mc.get("moe_hybridep_prepad_packed_inputs"):
        raise ValueError("Dynamic CP does not support MTP or HybridEP input prepadding")
    if cfg["sequence_packing"].get("pair_grouping_key"):
        raise ValueError("Dynamic CP does not yet schedule atomic preference pairs")
    # Probe even empty plans, so invalid domains/bounds fail at initialization.
    tp = mc["tensor_model_parallel_size"]
    minimum = dynamic.min_size
    while minimum * tp < mc["expert_model_parallel_size"]:
        minimum *= 2
    plan_cp_phases(
        [],
        lanes=lanes,
        min_size=minimum,
        max_size=dynamic.max_size or lanes,
        tokens_per_rank=dynamic.tokens_per_rank,
        sequence_parallel_size=tp if mc["sequence_parallel"] else 1,
        user_pad_multiple=cfg["make_sequence_length_divisible_by"],
    )


@dataclass
class CPDispatch:
    """Nested DP/CP payloads plus unique result row selection per lane."""

    data: list[list[BatchedDataDict]]
    plans: list[list[CPRankPlan]]
    output_rows: list[tuple[list[int], list[int]]]
    schedule: "CPBatchSchedule"


@dataclass(frozen=True)
class CPBatchSchedule:
    """Immutable sample grouping shared by score and train dispatches."""

    input_lengths: tuple[int, ...]
    batch_size: int
    lanes: int
    min_size: int
    max_size: int
    tokens_per_rank: int
    sequence_parallel_size: int
    user_pad_multiple: int
    token_alignment: int
    groups_by_batch: tuple[tuple[CPSyncGroup, ...], ...]


def _schedule_parameters(
    cfg: dict[str, Any], sharding: NamedSharding
) -> tuple[int, int, int, int, int, int, int]:
    dynamic = dynamic_cp_config(cfg)
    if dynamic is None:
        raise ValueError("Dynamic CP scheduling requires enabled configuration")
    mc = cfg["megatron_cfg"]
    cp = sharding.shape["context_parallel"]
    lanes = sharding.shape["data_parallel"] * cp
    tp = mc["tensor_model_parallel_size"]
    minimum = dynamic.min_size
    while minimum * tp < mc["expert_model_parallel_size"]:
        minimum *= 2
    fp8 = mc.get("fp8_cfg") or {}
    alignment = 1
    if fp8.get("enabled"):
        alignment = {"blockwise": 128, "mxfp8": 32}.get(fp8["fp8_recipe"], 16)
    if (
        mc.get("moe_token_dispatcher_type") == "flex"
        and mc.get("moe_flex_dispatcher_backend") == "hybridep"
    ):
        alignment = max(alignment, 128)
    return (
        lanes,
        minimum,
        dynamic.max_size or lanes,
        dynamic.tokens_per_rank,
        tp if mc["sequence_parallel"] else 1,
        cfg["make_sequence_length_divisible_by"],
        alignment,
    )


def _schedule_batch_size(data: BatchedDataDict, batch_size: int | None) -> int:
    gbs = batch_size if batch_size is not None else data.size
    if gbs < 1 or data.size % gbs:
        raise ValueError("Dynamic CP data must contain complete schedule batches")
    return gbs


def _input_lengths(data: BatchedDataDict) -> tuple[int, ...]:
    values = data["input_lengths"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(int(value) for value in values)


def build_cp_schedule(
    data: BatchedDataDict,
    cfg: dict[str, Any],
    sharding: NamedSharding,
    *,
    batch_size: int | None,
) -> CPBatchSchedule:
    """Plan sample groups once, using the training-safe token budget."""
    gbs = _schedule_batch_size(data, batch_size)
    lengths = _input_lengths(data)
    (
        lanes,
        minimum,
        maximum,
        tokens_per_rank,
        sequence_parallel_size,
        user_pad_multiple,
        alignment,
    ) = _schedule_parameters(cfg, sharding)
    groups_by_batch = tuple(
        plan_cp_phases(
            list(lengths[start : start + gbs]),
            lanes=lanes,
            min_size=minimum,
            max_size=maximum,
            tokens_per_rank=tokens_per_rank,
            sequence_parallel_size=sequence_parallel_size,
            user_pad_multiple=user_pad_multiple,
            token_alignment=alignment,
        )
        for start in range(0, len(lengths), gbs)
    )
    return CPBatchSchedule(
        lengths,
        gbs,
        lanes,
        minimum,
        maximum,
        tokens_per_rank,
        sequence_parallel_size,
        user_pad_multiple,
        alignment,
        groups_by_batch,
    )


def cp_schedule_matches(
    schedule: CPBatchSchedule,
    data: BatchedDataDict,
    cfg: dict[str, Any],
    sharding: NamedSharding,
    *,
    batch_size: int | None,
) -> bool:
    """Return whether a cached schedule is valid for this ordered batch."""
    try:
        gbs = _schedule_batch_size(data, batch_size)
        parameters = _schedule_parameters(cfg, sharding)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        schedule.input_lengths == _input_lengths(data)
        and schedule.batch_size == gbs
        and (
            schedule.lanes,
            schedule.min_size,
            schedule.max_size,
            schedule.tokens_per_rank,
            schedule.sequence_parallel_size,
            schedule.user_pad_multiple,
            schedule.token_alignment,
        )
        == parameters
    )


def build_cp_dispatch(
    data: BatchedDataDict,
    cfg: dict[str, Any],
    sharding: NamedSharding,
    *,
    batch_size: int | None,
    training: bool,
    schedule: CPBatchSchedule | None = None,
) -> CPDispatch:
    """Plan before replication; count each original training target once."""
    dynamic = dynamic_cp_config(cfg)
    if dynamic is None:
        raise ValueError("Dynamic CP dispatch requires enabled configuration")
    if data.size == 0:
        raise ValueError("Cannot schedule an empty dynamic CP batch")
    cp = sharding.shape["context_parallel"]
    dp = sharding.shape["data_parallel"]
    lanes = cp * dp
    group_ranks = tuple(
        sharding.get_ranks_by_coord(
            pipeline_parallel=0,
            data_parallel=i // cp,
            context_parallel=i % cp,
            tensor_parallel=0,
        )[0]
        for i in range(lanes)
    )
    gbs = _schedule_batch_size(data, batch_size)
    reused_schedule = schedule is not None
    if schedule is None:
        schedule = build_cp_schedule(data, cfg, sharding, batch_size=batch_size)
    elif not cp_schedule_matches(schedule, data, cfg, sharding, batch_size=batch_size):
        raise ValueError("Cached dynamic CP schedule does not match this ordered batch")
    rank_steps: list[list[CPRankStep]] = [[] for _ in range(lanes)]
    for batch_index, start in enumerate(range(0, data.size, gbs)):
        batch = data.select_indices(list(range(start, start + gbs)))
        groups = schedule.groups_by_batch[batch_index]
        if training:
            if "sample_mask" not in batch or "token_mask" not in batch:
                raise ValueError(
                    "Dynamic CP training requires sample_mask and token_mask"
                )
            valid_sequences = float(batch["sample_mask"].sum().item())
            valid_tokens = float(
                (batch["token_mask"][:, 1:] * batch["sample_mask"].unsqueeze(-1))
                .sum()
                .item()
            )
        else:
            valid_sequences = valid_tokens = 0.0
        samples_by_cp = Counter()
        for group in groups:
            for task in group.assignments:
                samples_by_cp[task.cp_size] += len(task.sample_indices)
        group_task_ranges = [
            (
                min(len(assignments_for_lane(group, lane)) for lane in range(lanes)),
                max(len(assignments_for_lane(group, lane)) for lane in range(lanes)),
            )
            for group in groups
        ]
        logger.info(
            "Dynamic CP %s: samples=%d groups=%d uneven_groups=%d "
            "local_tasks=[%d,%d] "
            "samples_by_cp=%s "
            "valid_sequences=%s valid_tokens=%s schedule=%s",
            "train" if training else "score",
            gbs,
            len(groups),
            sum(low < high for low, high in group_task_ranges),
            min(
                sum(len(assignments_for_lane(group, lane)) for group in groups)
                for lane in range(lanes)
            ),
            max(
                sum(len(assignments_for_lane(group, lane)) for group in groups)
                for lane in range(lanes)
            ),
            dict(sorted(samples_by_cp.items())),
            valid_sequences,
            valid_tokens,
            "reused" if reused_schedule else "new",
        )
        for lane in range(lanes):
            rank_groups = tuple(
                CPRankGroup(
                    tuple(
                        replace(
                            assignment,
                            sample_indices=tuple(
                                i + start for i in assignment.sample_indices
                            ),
                        )
                        for assignment in assignments_for_lane(group, lane)
                    )
                )
                for group in groups
            )
            rank_steps[lane].append(
                CPRankStep(rank_groups, valid_sequences, valid_tokens)
            )

    payloads, plans, output_rows = [], [], []
    for lane, steps in enumerate(rank_steps):
        indices = sorted(
            {
                i
                for step in steps
                for task in step.assignments
                for i in task.sample_indices
            }
        )
        # Padding-only lanes still need one prototype row to execute attention.
        if not indices:
            indices = [0]
        local_indices = {index: local for local, index in enumerate(indices)}
        remapped_steps = tuple(
            replace(
                step,
                groups=tuple(
                    CPRankGroup(
                        tuple(
                            replace(
                                task,
                                sample_indices=tuple(
                                    local_indices[i] for i in task.sample_indices
                                ),
                            )
                            for task in group.assignments
                        )
                    )
                    for group in step.groups
                ),
            )
            for step in steps
        )
        rows, originals, offset = [], [], 0
        for step in steps:
            for task in step.assignments:
                if lane == task.lane_start:
                    rows.extend(range(offset, offset + len(task.sample_indices)))
                    originals.extend(task.sample_indices)
                offset += max(1, len(task.sample_indices))
        output_rows.append((rows, originals))
        payloads.append(data.select_indices(indices))
        plans.append(CPRankPlan(lane, group_ranks, remapped_steps))
    return CPDispatch(
        data=[payloads[i : i + cp] for i in range(0, lanes, cp)],
        plans=[plans[i : i + cp] for i in range(0, lanes, cp)],
        output_rows=output_rows,
        schedule=schedule,
    )


def collect_cp_outputs(
    results: list[BatchedDataDict], dispatch: CPDispatch, size: int
) -> BatchedDataDict:
    """Keep only task owners and restore original rollout sample order."""
    if len(results) != len(dispatch.output_rows):
        raise ValueError("Dynamic CP must return results from every DP*CP lane")
    selected, original_indices = [], []
    for result, (rows, indices) in zip(results, dispatch.output_rows):
        if rows:
            selected.append(result.select_indices(rows))
            original_indices.extend(indices)
    if sorted(original_indices) != list(range(size)):
        raise ValueError("Dynamic CP output has missing or duplicated sample IDs")
    merged = BatchedDataDict.from_batches(selected)
    # reorder_data takes each current row's original position, and sorts it
    # internally; passing argsort here would apply the inverse permutation.
    merged.reorder_data(original_indices)
    return merged
