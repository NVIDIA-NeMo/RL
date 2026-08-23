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

"""Materialize model-free TraceBatchPlan rows into pre-scoring tensors."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence, TypedDict

import torch

from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    message_log_to_flat_messages,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_traces import validate_trace_batch_plan


class TraceBatchMaterialization(TypedDict):
    """Worker-shaped tensors plus non-worker ownership metadata."""

    plan_id: str
    train_data: BatchedDataDict[Any]
    materialized_message_logs: list[list[dict[str, Any]]]
    parent_indices: torch.Tensor
    row_rewards: torch.Tensor
    row_rollout_ids: list[str | None]
    row_trace_ids: list[str | None]


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _tensor_matches_list(tensor: torch.Tensor, values: Sequence[Any]) -> bool:
    expected = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    return tensor.shape == expected.shape and torch.equal(tensor, expected)


def _vision_cache_key_tensor(media_ids: Sequence[str]) -> torch.Tensor | None:
    """Return stable 128-bit cache keys without exposing strings to model.forward."""
    if not media_ids:
        return None
    keys = []
    for media_id in media_ids:
        digest = hashlib.blake2b(
            media_id.encode("utf-8"),
            digest_size=16,
            person=b"nrl-radio-v1",
        ).digest()
        keys.append(
            [
                int.from_bytes(digest[:8], byteorder="little", signed=True),
                int.from_bytes(digest[8:], byteorder="little", signed=True),
            ]
        )
    return torch.tensor(keys, dtype=torch.int64)


def _copy_trace_message_log(
    message_log: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    segments = trace.get("segments")
    if not isinstance(segments, list) or len(message_log) != len(segments):
        raise ValueError(
            f"Physical trace {trace.get('trace_id')!r} message/segment count mismatch"
        )

    result: list[dict[str, Any]] = []
    for message_index, (message, segment) in enumerate(zip(message_log, segments)):
        if not isinstance(message, Mapping) or not isinstance(segment, Mapping):
            raise TypeError(
                f"Physical trace {trace.get('trace_id')!r} has malformed "
                f"message/segment {message_index}"
            )
        kind = segment.get("kind")
        expected_role = "user" if kind == "prompt" else "assistant"
        if kind not in {"prompt", "completion"} or message.get("role") != expected_role:
            raise ValueError(
                f"Physical trace {trace.get('trace_id')!r} message {message_index} "
                "role disagrees with its exact segment"
            )
        token_ids = message.get("token_ids")
        if (
            not isinstance(token_ids, torch.Tensor)
            or token_ids.ndim != 1
            or not _tensor_matches_list(token_ids, segment.get("token_ids", []))
        ):
            raise ValueError(
                f"Physical trace {trace.get('trace_id')!r} message {message_index} "
                "tokens disagree with its exact segment"
            )

        copied = dict(message)
        copied["token_loss_mask"] = torch.tensor(
            segment.get("loss_mask", []),
            dtype=torch.int64,
            device=token_ids.device,
        )
        if kind == "prompt":
            existing_logprobs = message.get("generation_logprobs")
            if existing_logprobs is not None and (
                not isinstance(existing_logprobs, torch.Tensor)
                or existing_logprobs.shape != token_ids.shape
                or torch.count_nonzero(existing_logprobs).item() != 0
            ):
                raise ValueError(
                    f"Physical trace {trace.get('trace_id')!r} prompt message "
                    f"{message_index} has nonzero generation logprobs"
                )
            copied["generation_logprobs"] = torch.zeros(
                token_ids.shape,
                dtype=torch.float32,
                device=token_ids.device,
            )
        else:
            generation_logprobs = message.get("generation_logprobs")
            if (
                not isinstance(generation_logprobs, torch.Tensor)
                or generation_logprobs.ndim != 1
                or not _tensor_matches_list(
                    generation_logprobs,
                    segment.get("generation_logprobs", []),
                )
            ):
                raise ValueError(
                    f"Physical trace {trace.get('trace_id')!r} completion "
                    f"message {message_index} logprobs disagree with exact evidence"
                )
            copied["generation_logprobs"] = generation_logprobs

        routed_experts = message.get("routed_experts")
        if routed_experts is not None and (
            not isinstance(routed_experts, torch.Tensor)
            or routed_experts.ndim != 3
            or routed_experts.shape[0] != token_ids.shape[0]
        ):
            raise ValueError(
                f"Physical trace {trace.get('trace_id')!r} message {message_index} "
                "has misaligned routed_experts"
            )
        result.append(copied)
    return result


def _packed_tensor_specs(
    message_logs: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[int, bool]]:
    specs: dict[str, tuple[int, bool]] = {}
    for message_log in message_logs:
        for message in message_log:
            for key, value in message.items():
                if not isinstance(value, PackedTensor):
                    continue
                spec = (value.dim_to_pack, value.pad_to_max_shape)
                previous = specs.setdefault(key, spec)
                if previous != spec:
                    raise ValueError(
                        f"Multimodal key {key!r} has inconsistent packing: "
                        f"{previous!r} versus {spec!r}"
                    )
    return specs


def _normalize_packed_tensor_rows(
    message_logs: list[list[dict[str, Any]]],
) -> dict[str, tuple[int, bool]]:
    specs = _packed_tensor_specs(message_logs)
    for row_index, message_log in enumerate(message_logs):
        if not message_log:
            raise ValueError(f"Materialized row {row_index} has no messages")
        for key, (dim_to_pack, pad_to_max_shape) in specs.items():
            values = [message[key] for message in message_log if key in message]
            if any(not isinstance(value, PackedTensor) for value in values):
                raise TypeError(
                    f"Materialized row {row_index} key {key!r} mixes packed "
                    "and non-packed values"
                )
            if not values:
                message_log[0][key] = PackedTensor(
                    [None],
                    dim_to_pack=dim_to_pack,
                    pad_to_max_shape=pad_to_max_shape,
                )
    return specs


def _normalize_routed_expert_rows(
    message_logs: list[list[dict[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    physical_messages = [
        message
        for row, message_log in zip(rows, message_logs)
        if row["row_kind"] == "physical_trace"
        for message in message_log
    ]
    routed_values = [message.get("routed_experts") for message in physical_messages]
    if not any(value is not None for value in routed_values):
        return
    if any(value is None for value in routed_values):
        raise ValueError(
            "routed_experts must cover every physical token-bearing message"
        )
    routed_tensors = [
        value for value in routed_values if isinstance(value, torch.Tensor)
    ]
    if len(routed_tensors) != len(routed_values):
        raise TypeError("routed_experts values must be tensors")
    first = routed_tensors[0]
    trailing_shape = first.shape[1:]
    if any(
        value.shape[1:] != trailing_shape
        or value.dtype != first.dtype
        or value.device != first.device
        for value in routed_tensors
    ):
        raise ValueError("routed_experts tensors have inconsistent worker shapes")
    for row, message_log in zip(rows, message_logs):
        if row["row_kind"] == "padding":
            topk = trailing_shape[-1]
            message_log[0]["routed_experts"] = (
                torch.arange(
                    topk,
                    dtype=first.dtype,
                    device=first.device,
                )
                .view(1, 1, topk)
                .expand(1, trailing_shape[0], topk)
                .clone()
            )


def materialize_trace_batch_plan(
    plan: Mapping[str, Any],
    *,
    bundles: Sequence[Mapping[str, Any]],
    physical_message_logs_by_rollout: Mapping[
        str, Sequence[Sequence[Mapping[str, Any]]]
    ],
    pad_token_id: int,
    make_sequence_length_divisible_by: int = 1,
) -> TraceBatchMaterialization:
    """Materialize exact physical rows without policy/reference scoring."""
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise ValueError("pad_token_id must be an integer")
    make_sequence_length_divisible_by = _require_positive_int(
        make_sequence_length_divisible_by,
        field="make_sequence_length_divisible_by",
    )
    validate_trace_batch_plan(plan, bundles=bundles)

    rollout_ids = plan["rollout_ids"]
    if set(physical_message_logs_by_rollout) != set(rollout_ids):
        missing = sorted(set(rollout_ids) - set(physical_message_logs_by_rollout))
        extra = sorted(set(physical_message_logs_by_rollout) - set(rollout_ids))
        raise ValueError(
            "Physical message logs must match planned rollouts exactly: "
            f"missing={missing!r}, extra={extra!r}"
        )

    bundle_by_rollout_id = {str(bundle["rollout_id"]): bundle for bundle in bundles}
    message_logs: list[list[dict[str, Any]]] = []
    for row in plan["rows"]:
        if row["row_kind"] == "padding":
            message_logs.append(
                [
                    {
                        "role": "user",
                        "content": "",
                        "token_ids": torch.tensor([pad_token_id], dtype=torch.int64),
                        "token_loss_mask": torch.zeros(1, dtype=torch.int64),
                        "generation_logprobs": torch.zeros(1, dtype=torch.float32),
                    }
                ]
            )
            continue

        rollout_id = row["rollout_id"]
        assert isinstance(rollout_id, str)
        bundle = bundle_by_rollout_id[rollout_id]
        trace_index = row["trace_index"]
        rollout_message_logs = physical_message_logs_by_rollout[rollout_id]
        if len(rollout_message_logs) != len(bundle["physical_traces"]):
            raise ValueError(
                f"Rollout {rollout_id!r} physical message-log count disagrees "
                "with its trace bundle"
            )
        trace = bundle["physical_traces"][trace_index]
        if trace["trace_id"] != row["trace_id"]:
            raise ValueError(
                f"Planned trace {row['trace_id']!r} disagrees with bundle ordering"
            )
        message_logs.append(
            _copy_trace_message_log(
                rollout_message_logs[trace_index],
                trace,
            )
        )

    packed_specs = _normalize_packed_tensor_rows(message_logs)
    _normalize_routed_expert_rows(message_logs, plan["rows"])
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        pad_value_dict={"token_ids": pad_token_id},
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    required_flat_keys = {
        "token_ids",
        "token_loss_mask",
        "generation_logprobs",
    }
    missing_flat_keys = required_flat_keys - set(flat)
    if missing_flat_keys:
        raise ValueError(
            f"Materialized trace rows are missing fields {sorted(missing_flat_keys)!r}"
        )

    batch_size, sequence_length = flat["token_ids"].shape
    if batch_size != plan["total_row_count"]:
        raise ValueError("Materialized trace-row count disagrees with TraceBatchPlan")
    row_advantages = torch.tensor(
        [row["advantage"] for row in plan["rows"]],
        dtype=torch.float32,
        device=flat["token_ids"].device,
    )
    parent_indices = torch.tensor(
        plan["parent_indices"],
        dtype=torch.int64,
        device=flat["token_ids"].device,
    )
    train_data = BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "token_mask": flat["token_loss_mask"],
            "sample_mask": torch.tensor(
                [row["sample_mask"] for row in plan["rows"]],
                dtype=torch.float32,
                device=flat["token_ids"].device,
            ),
            "advantages": row_advantages.unsqueeze(-1)
            .expand(batch_size, sequence_length)
            .clone(),
            # Stable ownership across context-compaction traces. Padding rows
            # use -1 and are excluded by sample_mask.
            "logical_rollout_ids": parent_indices,
            # Keep the human-auditable media identity row-aligned through
            # scoring. The model receives the corresponding fixed-width digest
            # below because pipeline forwards cannot carry Python strings.
            "ordered_media_ids": [
                [str(media_id) for media_id in row["ordered_media_ids"]]
                for row in plan["rows"]
            ],
        }
    )
    train_data.update(flat.get_multimodal_dict(as_tensors=False))
    train_data["image_cache_keys"] = PackedTensor(
        [
            _vision_cache_key_tensor(row["ordered_media_ids"])
            for row in plan["rows"]
        ],
        dim_to_pack=0,
    )
    if "routed_experts" in flat:
        train_data["routed_experts"] = flat["routed_experts"]

    materialization: TraceBatchMaterialization = {
        "plan_id": str(plan["plan_id"]),
        "train_data": train_data,
        "materialized_message_logs": message_logs,
        "parent_indices": parent_indices,
        "row_rewards": torch.tensor(
            [row["reward"] for row in plan["rows"]],
            dtype=torch.float32,
            device=flat["token_ids"].device,
        ),
        "row_rollout_ids": [row["rollout_id"] for row in plan["rows"]],
        "row_trace_ids": [row["trace_id"] for row in plan["rows"]],
    }
    validate_trace_batch_materialization(
        materialization,
        plan=plan,
        bundles=bundles,
        pad_token_id=pad_token_id,
        packed_specs=packed_specs,
    )
    return materialization


def validate_trace_batch_materialization(
    materialization: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    bundles: Sequence[Mapping[str, Any]],
    pad_token_id: int,
    packed_specs: Mapping[str, tuple[int, bool]] | None = None,
) -> dict[str, Any]:
    """Independently validate pre-scoring tensors against the plan and bundles."""
    validate_trace_batch_plan(plan, bundles=bundles)
    if materialization.get("plan_id") != plan.get("plan_id"):
        raise ValueError("Materialized batch has the wrong TraceBatchPlan identity")
    train_data = materialization.get("train_data")
    if not isinstance(train_data, BatchedDataDict):
        raise TypeError("Materialized train_data must be a BatchedDataDict")

    required_keys = {
        "input_ids",
        "input_lengths",
        "generation_logprobs",
        "token_mask",
        "sample_mask",
        "advantages",
        "ordered_media_ids",
        "image_cache_keys",
    }
    missing_keys = required_keys - set(train_data)
    if missing_keys:
        raise ValueError(f"Materialized train_data is missing {sorted(missing_keys)!r}")
    input_ids = train_data["input_ids"]
    input_lengths = train_data["input_lengths"]
    generation_logprobs = train_data["generation_logprobs"]
    token_mask = train_data["token_mask"]
    sample_mask = train_data["sample_mask"]
    advantages = train_data["advantages"]
    ordered_media_ids = train_data["ordered_media_ids"]
    image_cache_keys = train_data["image_cache_keys"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            input_ids,
            input_lengths,
            generation_logprobs,
            token_mask,
            sample_mask,
            advantages,
        )
    ):
        raise TypeError("Materialized core worker fields must be tensors")
    batch_size = plan["total_row_count"]
    if ordered_media_ids != [
        [str(media_id) for media_id in row["ordered_media_ids"]]
        for row in plan["rows"]
    ]:
        raise ValueError("Materialized ordered media IDs are corrupted")
    if not isinstance(image_cache_keys, PackedTensor) or len(image_cache_keys) != batch_size:
        raise ValueError("Materialized image cache keys lost row ownership")
    for row_index, row in enumerate(plan["rows"]):
        expected_cache_keys = _vision_cache_key_tensor(row["ordered_media_ids"])
        observed_cache_keys = image_cache_keys.tensors[row_index]
        if expected_cache_keys is None or observed_cache_keys is None:
            if expected_cache_keys is not None or observed_cache_keys is not None:
                raise ValueError(
                    f"Materialized image cache keys row {row_index} changed empty ownership"
                )
        elif not torch.equal(observed_cache_keys, expected_cache_keys):
            raise ValueError(
                f"Materialized image cache keys row {row_index} are corrupted"
            )
    if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
        raise ValueError("Materialized input_ids has the wrong batch shape")
    if (
        generation_logprobs.shape != input_ids.shape
        or token_mask.shape != input_ids.shape
        or advantages.shape != input_ids.shape
        or input_lengths.shape != (batch_size,)
        or sample_mask.shape != (batch_size,)
    ):
        raise ValueError("Materialized worker fields have inconsistent shapes")

    bundle_by_rollout_id = {str(bundle["rollout_id"]): bundle for bundle in bundles}
    for row_index, row in enumerate(plan["rows"]):
        length = int(input_lengths[row_index].item())
        if row["row_kind"] == "padding":
            if (
                length != 1
                or torch.any(input_ids[row_index] != pad_token_id)
                or torch.count_nonzero(token_mask[row_index]).item() != 0
                or torch.count_nonzero(generation_logprobs[row_index]).item() != 0
                or torch.count_nonzero(advantages[row_index]).item() != 0
                or sample_mask[row_index].item() != 0.0
            ):
                raise ValueError(f"Materialized padding row {row_index} is not masked")
            continue

        rollout_id = row["rollout_id"]
        assert isinstance(rollout_id, str)
        trace = bundle_by_rollout_id[rollout_id]["physical_traces"][row["trace_index"]]
        expected_tokens: list[int] = []
        expected_mask: list[int] = []
        expected_logprobs: list[float] = []
        for segment in trace["segments"]:
            segment_tokens = list(segment["token_ids"])
            expected_tokens.extend(segment_tokens)
            expected_mask.extend(segment["loss_mask"])
            if segment["kind"] == "completion":
                expected_logprobs.extend(segment["generation_logprobs"])
            else:
                expected_logprobs.extend([0.0] * len(segment_tokens))
        if length != len(expected_tokens):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} has the wrong length"
            )
        if not _tensor_matches_list(input_ids[row_index, :length], expected_tokens):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} tokens are corrupted"
            )
        if not _tensor_matches_list(token_mask[row_index, :length], expected_mask):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} token mask is corrupted"
            )
        if not _tensor_matches_list(
            generation_logprobs[row_index, :length],
            expected_logprobs,
        ):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} logprobs are corrupted"
            )
        if not torch.all(
            advantages[row_index]
            == torch.tensor(
                row["advantage"],
                dtype=advantages.dtype,
                device=advantages.device,
            )
        ):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} advantage is corrupted"
            )
        if sample_mask[row_index].item() != 1.0:
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} is sample-masked"
            )
        if torch.any(input_ids[row_index, length:] != pad_token_id):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} token padding is corrupted"
            )
        if (
            torch.count_nonzero(token_mask[row_index, length:]).item() != 0
            or torch.count_nonzero(generation_logprobs[row_index, length:]).item() != 0
        ):
            raise ValueError(
                f"Materialized trace {trace['trace_id']!r} mask/logprob padding "
                "is corrupted"
            )

    effective_token_count = int((token_mask * sample_mask.unsqueeze(-1)).sum().item())
    if effective_token_count != plan["eligible_action_token_count"]:
        raise ValueError(
            "Materialized batch changed the global eligible-action-token count"
        )

    parent_indices = materialization.get("parent_indices")
    row_rewards = materialization.get("row_rewards")
    if (
        not isinstance(parent_indices, torch.Tensor)
        or parent_indices.tolist() != plan["parent_indices"]
    ):
        raise ValueError("Materialized parent indices are corrupted")
    if not isinstance(row_rewards, torch.Tensor) or not _tensor_matches_list(
        row_rewards,
        [row["reward"] for row in plan["rows"]],
    ):
        raise ValueError("Materialized row rewards are corrupted")
    if materialization.get("row_rollout_ids") != [
        row["rollout_id"] for row in plan["rows"]
    ]:
        raise ValueError("Materialized rollout ownership is corrupted")
    if materialization.get("row_trace_ids") != [
        row["trace_id"] for row in plan["rows"]
    ]:
        raise ValueError("Materialized trace ownership is corrupted")

    message_logs = materialization.get("materialized_message_logs")
    if not isinstance(message_logs, list) or len(message_logs) != batch_size:
        raise ValueError("Materialized message-log rows are incomplete")
    specs = dict(packed_specs or _packed_tensor_specs(message_logs))
    for key, (dim_to_pack, pad_to_max_shape) in specs.items():
        packed = train_data.get(key)
        if (
            not isinstance(packed, PackedTensor)
            or len(packed) != batch_size
            or packed.dim_to_pack != dim_to_pack
            or packed.pad_to_max_shape != pad_to_max_shape
        ):
            raise ValueError(
                f"Materialized multimodal key {key!r} lost its row packing"
            )
        expected_tensors = []
        for message_log in message_logs:
            row_flat = message_log_to_flat_messages(message_log)
            row_packed = row_flat.get(key)
            if not isinstance(row_packed, PackedTensor):
                raise ValueError(f"Materialized multimodal row is missing key {key!r}")
            expected_tensors.append(row_packed.as_tensor())
        for row_index, (observed, expected) in enumerate(
            zip(packed.tensors, expected_tensors)
        ):
            if observed is None or expected is None:
                if observed is not None or expected is not None:
                    raise ValueError(
                        f"Materialized multimodal key {key!r} row {row_index} "
                        "changed empty-row ownership"
                    )
            elif not torch.equal(observed, expected):
                raise ValueError(
                    f"Materialized multimodal key {key!r} row {row_index} is corrupted"
                )

    routed_experts = train_data.get("routed_experts")
    if routed_experts is not None and (
        not isinstance(routed_experts, torch.Tensor)
        or routed_experts.shape[:2] != input_ids.shape
    ):
        raise ValueError("Materialized routed_experts is misaligned")
    if isinstance(routed_experts, torch.Tensor):
        for row_index, (row, message_log) in enumerate(zip(plan["rows"], message_logs)):
            if row["row_kind"] == "padding":
                expected_padding_routes = message_log[0]["routed_experts"]
                if (
                    not torch.equal(
                        routed_experts[row_index, :1],
                        expected_padding_routes,
                    )
                    or torch.count_nonzero(routed_experts[row_index, 1:]).item() != 0
                ):
                    raise ValueError(
                        f"Materialized padding row {row_index} has invalid routed "
                        "experts"
                    )
                continue
            expected_routes = torch.cat(
                [message["routed_experts"] for message in message_log],
                dim=0,
            )
            length = int(input_lengths[row_index].item())
            if (
                not torch.equal(
                    routed_experts[row_index, :length],
                    expected_routes,
                )
                or torch.count_nonzero(routed_experts[row_index, length:]).item() != 0
            ):
                raise ValueError(
                    f"Materialized trace row {row_index} routed experts are corrupted"
                )

    return {
        "row_count": batch_size,
        "physical_trace_count": plan["physical_trace_count"],
        "padding_row_count": plan["padding_row_count"],
        "eligible_action_token_count": effective_token_count,
        "multimodal_key_count": len(specs),
    }
