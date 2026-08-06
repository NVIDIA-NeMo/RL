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

"""Capture, tensorize, and validate native environment turn rewards."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Sequence

import torch

if TYPE_CHECKING:
    from nemo_rl.environments.interfaces import EnvironmentReturn

TURN_REWARD_KEY = "_turn_credit_environment_reward"
TURN_REWARD_COMPONENTS_KEY = "_turn_credit_reward_components"
TURN_TERMINATED_KEY = "_turn_credit_terminated"

TURN_REWARDS_FIELD = "turn_rewards"
TURN_MASK_FIELD = "turn_mask"
TURN_TRAINABLE_MASK_FIELD = "turn_trainable_mask"
ASSISTANT_TURN_SPANS_FIELD = "assistant_turn_spans"
TURN_TERMINATEDS_FIELD = "turn_terminateds"


@dataclass(frozen=True)
class TurnRecord:
    """One environment transition aligned with one generated assistant message."""

    generated_message_index: int
    environment_reward: float
    reward_components: dict[str, float]
    terminated: bool


@dataclass(frozen=True)
class TurnBatch:
    """Compact padded turn representation for one rollout batch."""

    rewards: torch.Tensor
    mask: torch.Tensor
    trainable_mask: torch.Tensor
    assistant_spans: torch.Tensor
    terminateds: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Return the number of trajectories."""
        return self.rewards.shape[0]

    @property
    def max_turns(self) -> int:
        """Return the padded turn dimension."""
        return self.rewards.shape[1]


def _as_scalar_reward_rows(
    rewards: torch.Tensor | dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Convert scalar or component rewards into one scalar per batch row."""
    if isinstance(rewards, dict):
        if not rewards:
            raise ValueError("Environment returned an empty reward-component mapping")
        component_names = sorted(rewards)
        for name in component_names:
            component = rewards[name]
            if component.shape != (batch_size,):
                raise ValueError(
                    f"Reward component {name!r} must have shape ({batch_size},), "
                    f"got {tuple(component.shape)}"
                )
            if not torch.isfinite(component).all():
                raise ValueError(
                    f"Reward component {name!r} contains non-finite values"
                )
        scalar_rewards = torch.stack(
            [rewards[name] for name in component_names],
            dim=0,
        ).sum(dim=0)
        component_rows = [
            {name: float(rewards[name][row].item()) for name in component_names}
            for row in range(batch_size)
        ]
    else:
        if rewards.shape != (batch_size,):
            raise ValueError(
                f"Environment rewards must have shape ({batch_size},), "
                f"got {tuple(rewards.shape)}"
            )
        scalar_rewards = rewards
        component_rows = [{} for _ in range(batch_size)]

    if not torch.isfinite(scalar_rewards).all():
        raise ValueError("Environment rewards contain non-finite values")
    return scalar_rewards, component_rows


def record_environment_turn(
    message_logs: Sequence[list[dict[str, Any]]],
    environment_return: "EnvironmentReturn",
) -> None:
    """Attach each environment result to its generated assistant message.

    The annotations are temporary. ``tensorize_turn_traces`` converts them to
    compact batch tensors and ``remove_turn_annotations`` deletes them before
    normal NeMo-RL message flattening.

    Args:
        message_logs: Active rollout message logs passed to the environment.
        environment_return: Batched result for the current environment step.

    Raises:
        ValueError: If reward shapes are invalid or a generated message cannot
            be identified unambiguously.
    """
    batch_size = len(message_logs)
    if environment_return.terminateds.shape != (batch_size,):
        raise ValueError(
            "Environment terminated flags must have shape "
            f"({batch_size},), got {tuple(environment_return.terminateds.shape)}"
        )

    scalar_rewards, component_rows = _as_scalar_reward_rows(
        environment_return.rewards,
        batch_size=batch_size,
    )

    for row, message_log in enumerate(message_logs):
        generated_indices = [
            index
            for index, message in enumerate(message_log)
            if message.get("role") == "assistant" and "generation_logprobs" in message
        ]
        if not generated_indices:
            raise ValueError(
                f"Rollout row {row} has no policy-generated assistant message"
            )
        message = message_log[generated_indices[-1]]
        if TURN_REWARD_KEY in message:
            raise ValueError(
                f"Generated assistant message in row {row} was recorded twice"
            )
        message[TURN_REWARD_KEY] = float(scalar_rewards[row].item())
        message[TURN_REWARD_COMPONENTS_KEY] = component_rows[row]
        message[TURN_TERMINATED_KEY] = bool(environment_return.terminateds[row].item())


def extract_turn_records(message_log: Sequence[Mapping[str, Any]]) -> list[TurnRecord]:
    """Extract ordered turn records from one annotated message log."""
    records = []
    for message_index, message in enumerate(message_log):
        if TURN_REWARD_KEY not in message:
            continue
        if message.get("role") != "assistant":
            raise ValueError(
                f"Turn annotation at message {message_index} is not on an assistant message"
            )
        if "generation_logprobs" not in message:
            raise ValueError(
                f"Turn annotation at message {message_index} is not policy-generated"
            )
        reward = float(message[TURN_REWARD_KEY])
        if not torch.isfinite(torch.tensor(reward)):
            raise ValueError(
                f"Turn annotation at message {message_index} has non-finite reward"
            )
        if TURN_TERMINATED_KEY not in message:
            raise ValueError(
                f"Turn annotation at message {message_index} lacks a terminal flag"
            )
        components = {
            str(name): float(value)
            for name, value in dict(message.get(TURN_REWARD_COMPONENTS_KEY, {})).items()
        }
        records.append(
            TurnRecord(
                generated_message_index=message_index,
                environment_reward=reward,
                reward_components=components,
                terminated=bool(message[TURN_TERMINATED_KEY]),
            )
        )
    return records


def _message_token_length(message: Mapping[str, Any], *, message_index: int) -> int:
    token_ids = message.get("token_ids")
    if not isinstance(token_ids, torch.Tensor):
        raise ValueError(
            f"Message {message_index} must contain tensor token_ids before tensorization"
        )
    if token_ids.ndim != 1:
        raise ValueError(
            f"Message {message_index} token_ids must be one-dimensional, "
            f"got shape {tuple(token_ids.shape)}"
        )
    return token_ids.numel()


def tensorize_turn_traces(
    message_logs: Sequence[Sequence[Mapping[str, Any]]],
) -> TurnBatch:
    """Convert annotated message logs to compact padded turn tensors."""
    records_per_row = [
        extract_turn_records(message_log) for message_log in message_logs
    ]
    max_turns = max((len(records) for records in records_per_row), default=0)
    batch_size = len(message_logs)

    rewards = torch.zeros((batch_size, max_turns), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_turns), dtype=torch.bool)
    trainable_mask = torch.zeros((batch_size, max_turns), dtype=torch.bool)
    assistant_spans = torch.zeros((batch_size, max_turns, 2), dtype=torch.int64)
    terminateds = torch.zeros((batch_size, max_turns), dtype=torch.bool)

    for row, (message_log, records) in enumerate(
        zip(message_logs, records_per_row, strict=True)
    ):
        offsets = [0]
        for message_index, message in enumerate(message_log):
            offsets.append(
                offsets[-1]
                + _message_token_length(message, message_index=message_index)
            )

        previous_end = 0
        for turn_index, record in enumerate(records):
            start = offsets[record.generated_message_index]
            end = offsets[record.generated_message_index + 1]
            if start < previous_end:
                raise ValueError(
                    f"Turn spans overlap or are not monotonic in row {row}"
                )
            previous_end = end
            rewards[row, turn_index] = record.environment_reward
            mask[row, turn_index] = True
            trainable_mask[row, turn_index] = end > start
            assistant_spans[row, turn_index] = torch.tensor([start, end])
            terminateds[row, turn_index] = record.terminated

    return TurnBatch(
        rewards=rewards,
        mask=mask,
        trainable_mask=trainable_mask,
        assistant_spans=assistant_spans,
        terminateds=terminateds,
    )


def attach_turn_batch(
    batch: MutableMapping[str, Any],
    turn_batch: TurnBatch,
) -> None:
    """Attach compact turn tensors to a rollout batch."""
    batch[TURN_REWARDS_FIELD] = turn_batch.rewards
    batch[TURN_MASK_FIELD] = turn_batch.mask
    batch[TURN_TRAINABLE_MASK_FIELD] = turn_batch.trainable_mask
    batch[ASSISTANT_TURN_SPANS_FIELD] = turn_batch.assistant_spans
    batch[TURN_TERMINATEDS_FIELD] = turn_batch.terminateds


def turn_batch_from_mapping(batch: Mapping[str, Any]) -> TurnBatch:
    """Read compact turn tensors from a mapping and validate their shapes."""
    required_fields = (
        TURN_REWARDS_FIELD,
        TURN_MASK_FIELD,
        TURN_TRAINABLE_MASK_FIELD,
        ASSISTANT_TURN_SPANS_FIELD,
        TURN_TERMINATEDS_FIELD,
    )
    missing = [field for field in required_fields if field not in batch]
    if missing:
        raise ValueError(f"Turn-credit batch is missing fields: {missing}")
    non_tensors = [
        field for field in required_fields if not isinstance(batch[field], torch.Tensor)
    ]
    if non_tensors:
        raise TypeError(f"Turn-credit batch fields must be tensors: {non_tensors}")

    turn_batch = TurnBatch(
        rewards=batch[TURN_REWARDS_FIELD],
        mask=batch[TURN_MASK_FIELD],
        trainable_mask=batch[TURN_TRAINABLE_MASK_FIELD],
        assistant_spans=batch[ASSISTANT_TURN_SPANS_FIELD],
        terminateds=batch[TURN_TERMINATEDS_FIELD],
    )
    if not turn_batch.rewards.is_floating_point():
        raise ValueError("turn_rewards must use a floating-point dtype")
    for field_name, value in (
        (TURN_MASK_FIELD, turn_batch.mask),
        (TURN_TRAINABLE_MASK_FIELD, turn_batch.trainable_mask),
        (TURN_TERMINATEDS_FIELD, turn_batch.terminateds),
    ):
        if value.dtype != torch.bool:
            raise ValueError(f"{field_name} must use torch.bool")
    if turn_batch.assistant_spans.dtype != torch.int64:
        raise ValueError("assistant_turn_spans must use torch.int64")
    devices = {
        value.device
        for value in (
            turn_batch.rewards,
            turn_batch.mask,
            turn_batch.trainable_mask,
            turn_batch.assistant_spans,
            turn_batch.terminateds,
        )
    }
    if len(devices) != 1:
        raise ValueError("All turn-credit tensors must be on the same device")

    expected = turn_batch.rewards.shape
    if turn_batch.rewards.ndim != 2:
        raise ValueError(f"turn_rewards must have shape [B, T], got {tuple(expected)}")
    if turn_batch.mask.shape != expected:
        raise ValueError("turn_mask must match turn_rewards")
    if turn_batch.trainable_mask.shape != expected:
        raise ValueError("turn_trainable_mask must match turn_rewards")
    if turn_batch.terminateds.shape != expected:
        raise ValueError("turn_terminateds must match turn_rewards")
    if turn_batch.assistant_spans.shape != (*expected, 2):
        raise ValueError(
            "assistant_turn_spans must have shape [B, T, 2], got "
            f"{tuple(turn_batch.assistant_spans.shape)}"
        )
    if not torch.isfinite(turn_batch.rewards[turn_batch.mask]).all():
        raise ValueError("Observed turn rewards contain non-finite values")
    return turn_batch


def validate_raw_reward_sums(
    turn_batch: TurnBatch,
    raw_total_reward: torch.Tensor,
    *,
    atol: float,
) -> None:
    """Require raw turn rewards to sum to the pre-transform trajectory reward."""
    expected_shape = (turn_batch.batch_size,)
    if raw_total_reward.shape != expected_shape:
        raise ValueError(
            f"total_reward must have shape {expected_shape}, "
            f"got {tuple(raw_total_reward.shape)}"
        )
    turn_sums = (turn_batch.rewards * turn_batch.mask).sum(dim=1)
    if not torch.allclose(turn_sums, raw_total_reward.float(), rtol=0.0, atol=atol):
        raise ValueError(
            "Raw turn rewards do not sum to total_reward: "
            f"turn_sums={turn_sums.tolist()}, "
            f"total_reward={raw_total_reward.tolist()}, atol={atol}"
        )


def validate_turn_spans(turn_batch: TurnBatch, token_mask: torch.Tensor) -> None:
    """Validate turn spans against the actual trainable-token mask."""
    if token_mask.ndim != 2:
        raise ValueError(
            f"token_mask must have shape [B, S], got {tuple(token_mask.shape)}"
        )
    if token_mask.shape[0] != turn_batch.batch_size:
        raise ValueError("token_mask batch dimension does not match turn tensors")

    sequence_length = token_mask.shape[1]
    for row in range(turn_batch.batch_size):
        row_is_filtered = not bool(token_mask[row].bool().any().item())
        previous_end = 0
        for turn_index in range(turn_batch.max_turns):
            observed = bool(turn_batch.mask[row, turn_index].item())
            trainable = bool(turn_batch.trainable_mask[row, turn_index].item())
            start, end = (
                int(value)
                for value in turn_batch.assistant_spans[row, turn_index].tolist()
            )
            if not observed:
                if trainable or start != 0 or end != 0:
                    raise ValueError(
                        f"Padded turn {turn_index} in row {row} contains data"
                    )
                continue
            if start < previous_end or end < start or end > sequence_length:
                raise ValueError(
                    f"Invalid turn span [{start}, {end}) at row {row}, "
                    f"turn {turn_index}, sequence length {sequence_length}"
                )
            previous_end = end
            if trainable != (end > start):
                raise ValueError(
                    f"Turn trainable mask disagrees with span [{start}, {end}) "
                    f"at row {row}, turn {turn_index}"
                )
            if (
                trainable
                and not row_is_filtered
                and not bool(token_mask[row, start:end].bool().all().item())
            ):
                raise ValueError(
                    f"Turn span [{start}, {end}) at row {row}, turn {turn_index} "
                    "touches a non-trainable token"
                )


def remove_turn_annotations(message_logs: Sequence[list[dict[str, Any]]]) -> None:
    """Remove temporary per-message annotations after tensorization."""
    for message_log in message_logs:
        for message in message_log:
            message.pop(TURN_REWARD_KEY, None)
            message.pop(TURN_REWARD_COMPONENTS_KEY, None)
            message.pop(TURN_TERMINATED_KEY, None)


def compute_environment_credit(
    turn_batch: TurnBatch,
    *,
    mode: str,
    discount: float,
) -> torch.Tensor:
    """Compute immediate or discounted return-to-go native credit."""
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must be in [0, 1]")
    if mode == "immediate":
        return turn_batch.rewards * turn_batch.mask
    if mode != "return_to_go":
        raise ValueError(f"Unsupported environment credit mode: {mode!r}")

    credit = torch.zeros_like(turn_batch.rewards)
    for row in range(turn_batch.batch_size):
        running_return = torch.zeros(
            (),
            dtype=turn_batch.rewards.dtype,
            device=turn_batch.rewards.device,
        )
        for turn_index in range(turn_batch.max_turns - 1, -1, -1):
            if not bool(turn_batch.mask[row, turn_index].item()):
                continue
            running_return = (
                turn_batch.rewards[row, turn_index] + discount * running_return
            )
            credit[row, turn_index] = running_return
    return credit


def scatter_turn_credit(
    credit: torch.Tensor,
    turn_batch: TurnBatch,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter one credit value over each generated assistant-token span."""
    if credit.shape != turn_batch.rewards.shape:
        raise ValueError("Turn credit must match turn_rewards shape")
    if not credit.is_floating_point():
        raise ValueError("Turn credit must use a floating-point dtype")
    if credit.device != turn_batch.rewards.device:
        raise ValueError("Turn credit and turn rewards must be on the same device")
    if token_mask.dtype != torch.bool:
        raise ValueError("Turn-credit token eligibility mask must use torch.bool")
    if token_mask.device != turn_batch.rewards.device:
        raise ValueError(
            "Turn-credit token eligibility mask and turn rewards must be "
            "on the same device"
        )
    if not torch.isfinite(credit[turn_batch.mask]).all():
        raise ValueError("Observed turn credit contains non-finite values")
    validate_turn_spans(turn_batch, token_mask)

    scattered = torch.zeros_like(token_mask, dtype=credit.dtype)
    for row in range(turn_batch.batch_size):
        for turn_index in range(turn_batch.max_turns):
            if not bool(turn_batch.trainable_mask[row, turn_index].item()):
                continue
            start, end = (
                int(value)
                for value in turn_batch.assistant_spans[row, turn_index].tolist()
            )
            scattered[row, start:end] = credit[row, turn_index]
    return scattered * token_mask.to(dtype=scattered.dtype)
