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

"""Prompt-group staleness policies over a TQReplayBuffer.

A *staleness policy* owns the whole off-policyness contract for the SC async
loop in one object, injected into both pumps:

  - ``admit``  (rollout-pump side): block until the next prompt batch may
    dispatch, then return the ``target_step`` stamp for that batch (``None``
    when the policy doesn't stamp).  Owning admission here is what lets the
    rollout pump follow whichever sampling algorithm is selected without a
    second, hand-kept copy of the gating logic.
  - ``select`` / ``evict`` (train-pump side): pick / drop prompt groups.
  - ``is_on_policy`` / ``required_buffer_capacity`` (derived facts): what the
    weight-sync and capacity-validation paths need *without* re-reading raw
    knobs, so those consumers can't drift out of sync with the sampler.

``PromptGroupSampler`` is the interface; ``WindowedSampler`` /
``WeightFifoSampler`` / ``InOrderSampler`` are the built-in policies, one per
behavior, each owning only the args that apply to it.  ``create_sampler`` builds
one from a discriminated-union config (or a ``module:ClassName`` FQN for a
policy defined outside this repo) — the config's ``name`` is the single source
of truth for which behavior runs, so there are no cross-field knob combinations
to validate.
"""

from __future__ import annotations

import abc
import asyncio
import importlib
from typing import (
    Annotated,
    Callable,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data_plane import KVBatchMeta

# Poll interval for the rollout-pump admission gate.
_GATE_POLL_SECONDS = 0.005


@runtime_checkable
class PromptGroupSampler(Protocol):
    """Staleness policy shared by the SC rollout and train pumps.

    Implement this (or subclass ``BaseSampler``) to add a custom sampling
    algorithm; point ``async_rl.sampler`` at ``module:ClassName`` to load it.
    """

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        """Block until the next prompt batch may dispatch.

        Args:
            trainer_version_fn: Zero-arg accessor for the live trainer version
                (polled, so a blocking policy sees updates while it waits).

        Returns:
            The ``target_step`` to stamp on this batch's slots, or ``None`` when
            the policy does not stamp target steps.
        """
        ...

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        """Pick up to ``max_prompt_groups`` eligible groups; drop them locally."""
        ...

    async def evict(self, *, current_train_weight: int) -> int:
        """Drop groups that can no longer be selected; clear their DP rows."""
        ...

    @property
    def is_on_policy(self) -> bool:
        """True when the policy admits zero staleness (sync mode)."""
        ...

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        """Buffer-capacity the policy needs, or ``None`` if unconstrained."""
        ...


class BaseSampler(abc.ABC):
    """Shared machinery for the built-in policies.

    Owns the monotonic dispatch counter (the batch index formerly tracked as
    ``SingleControllerActor._max_rollout_version``) and the common
    select-finalize / weight-window-evict helpers.
    """

    def __init__(self, buffer: TQReplayBuffer) -> None:
        self._buffer = buffer
        # Pre-incremented before each admitted batch, so -1 lets the first
        # batch through a zero-staleness gate.
        self._dispatch_index: int = -1

    # ── rollout-pump side ────────────────────────────────────────────────
    @abc.abstractmethod
    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        ...

    # ── train-pump side ──────────────────────────────────────────────────
    @abc.abstractmethod
    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        ...

    async def evict(self, *, current_train_weight: int) -> int:
        """Default: drop *ready* groups below the weight window.

        Skips unready (reserved-but-uncommitted) slots so eviction can't race a
        concurrent ``commit`` that re-looks-up the slot after its ``await``.
        Policies whose ``select`` key isn't the start weight (e.g.
        ``InOrderSampler``) override this so evict and select agree.
        """
        min_valid_version = max(0, current_train_weight - self._eviction_window())
        stale_idxs = [
            i
            for i, weight in enumerate(self._buffer.start_weight_list)
            if weight < min_valid_version and self._buffer.ready_list[i]
        ]
        if not stale_idxs:
            return 0
        return await self._buffer.remove(stale_idxs, remove_in_dp=True)

    # ── derived facts ────────────────────────────────────────────────────
    @property
    def is_on_policy(self) -> bool:
        return self._eviction_window() == 0

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        return None

    # ── shared helpers ───────────────────────────────────────────────────
    def _eviction_window(self) -> int:
        """Weight-version span kept selectable; drives the default ``evict``."""
        return 0

    @staticmethod
    def _validate_group_bounds(min_prompt_groups: int, max_prompt_groups: int) -> None:
        if min_prompt_groups < 1:
            raise ValueError(f"min_prompt_groups must be >= 1, got {min_prompt_groups}")
        if max_prompt_groups < min_prompt_groups:
            raise ValueError(
                f"max_prompt_groups ({max_prompt_groups}) must be >= "
                f"min_prompt_groups ({min_prompt_groups})"
            )

    async def _finalize_selection(
        self,
        valid_idxs: list[int],
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        """Cap, drop from the buffer, and concat the chosen groups.

        Greedy without waiting: returns all currently-eligible groups up to
        ``max_prompt_groups`` (never fewer on purpose, never waits to fill it),
        or ``(None, 0)`` below ``min_prompt_groups``.
        """
        if len(valid_idxs) < min_prompt_groups:
            return None, 0
        requested_groups = min(len(valid_idxs), max_prompt_groups)
        selected_idxs = valid_idxs[:requested_groups]
        selected_metas = [self._buffer.meta_list[i] for i in selected_idxs]
        await self._buffer.remove(selected_idxs, remove_in_dp=False)
        return (
            selected_metas[0].concat(*selected_metas[1:]),  # type: ignore
            len(selected_idxs),
        )


class WindowedSampler(BaseSampler):
    """Over-sampled windowed selection.

    Rollout never gates on the trainer version — the pump keeps producing and
    samples aged past the window are evicted. ``select`` takes any ready group
    within ``[train_weight - max_staleness_versions, train_weight]``, optionally
    freshest-first.
    """

    def __init__(
        self,
        buffer: TQReplayBuffer,
        *,
        max_staleness_versions: int,
        sample_freshest_first: bool = False,
    ) -> None:
        super().__init__(buffer)
        if max_staleness_versions < 0:
            raise ValueError(
                f"max_staleness_versions must be non-negative, got "
                f"{max_staleness_versions}"
            )
        self.max_staleness_versions = max_staleness_versions
        self.sample_freshest_first = sample_freshest_first

    def _eviction_window(self) -> int:
        return self.max_staleness_versions

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        # Over-sampled: dispatch is bounded by buffer capacity, not by version.
        return None

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        self._validate_group_bounds(min_prompt_groups, max_prompt_groups)
        min_valid_version = max(0, current_train_weight - self.max_staleness_versions)
        valid_idxs = [
            i
            for i, weight in enumerate(self._buffer.start_weight_list)
            if min_valid_version <= weight <= current_train_weight
            and self._buffer.ready_list[i]
        ]
        if self.sample_freshest_first:
            valid_idxs.sort(
                key=lambda i: (
                    current_train_weight - self._buffer.start_weight_list[i],
                    i,
                )
            )
        return await self._finalize_selection(
            valid_idxs, min_prompt_groups, max_prompt_groups
        )


class _GatedSampler(BaseSampler):
    """Base for policies that admit exactly one dispatch batch per trainer step.

    The gate bounds how far generation may run ahead of the trainer
    (``gate_window`` versions of lookahead).
    """

    def __init__(self, buffer: TQReplayBuffer, *, gate_window: int) -> None:
        super().__init__(buffer)
        if gate_window < 0:
            raise ValueError(f"gate_window must be non-negative, got {gate_window}")
        self._gate_window = gate_window

    def _eviction_window(self) -> int:
        return self._gate_window

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        # One batch of lookahead per version in the window, plus the live batch.
        return groups_per_step * (self._gate_window + 1)

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        while self._dispatch_index >= trainer_version_fn() + self._gate_window:
            await asyncio.sleep(_GATE_POLL_SECONDS)
        self._dispatch_index += 1
        return self._stamp()

    def _stamp(self) -> Optional[int]:
        return None


class WeightFifoSampler(_GatedSampler):
    """Gated, strict weight-version FIFO.

    ``select`` drains the oldest in-window ``start_weight`` first and waits for
    that weight's batch to fill. Evict uses the weight window (default).
    """

    def __init__(self, buffer: TQReplayBuffer, *, max_staleness_versions: int) -> None:
        super().__init__(buffer, gate_window=max_staleness_versions)
        self.max_staleness_versions = max_staleness_versions

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        self._validate_group_bounds(min_prompt_groups, max_prompt_groups)
        min_valid_version = max(0, current_train_weight - self.max_staleness_versions)
        in_window = [
            weight
            for weight in self._buffer.start_weight_list
            if min_valid_version <= weight <= current_train_weight
        ]
        if not in_window:
            return None, 0
        target_version = min(in_window)
        valid_idxs = [
            i
            for i, weight in enumerate(self._buffer.start_weight_list)
            if weight == target_version and self._buffer.ready_list[i]
        ]
        return await self._finalize_selection(
            valid_idxs, min_prompt_groups, max_prompt_groups
        )


class InOrderSampler(_GatedSampler):
    """Gated, exact batch->step matching.

    Each dispatched batch is stamped with its dispatch index as ``target_step``;
    ``select`` takes the batch whose ``target_step`` equals the trainer version
    (the staleness window is not used for selection). ``evict`` is keyed on
    ``target_step`` — not the start weight — so a slot whose target step is still
    upcoming is never dropped early, and evict/select can't disagree.
    """

    def __init__(self, buffer: TQReplayBuffer, *, max_lookahead_versions: int) -> None:
        super().__init__(buffer, gate_window=max_lookahead_versions)
        self.max_lookahead_versions = max_lookahead_versions

    def _stamp(self) -> Optional[int]:
        return self._dispatch_index

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        self._validate_group_bounds(min_prompt_groups, max_prompt_groups)
        valid_idxs = [
            i
            for i, target in enumerate(self._buffer.target_step_list)
            if target == current_train_weight and self._buffer.ready_list[i]
        ]
        return await self._finalize_selection(
            valid_idxs, min_prompt_groups, max_prompt_groups
        )

    async def evict(self, *, current_train_weight: int) -> int:
        # Keyed on target_step (matches `select`): a ready group whose target
        # step has already passed can never be selected, so it is stale. Unready
        # slots are skipped to avoid racing a concurrent commit.
        stale_idxs = [
            i
            for i, target in enumerate(self._buffer.target_step_list)
            if target is not None
            and target < current_train_weight
            and self._buffer.ready_list[i]
        ]
        if not stale_idxs:
            return 0
        return await self._buffer.remove(stale_idxs, remove_in_dp=True)


# ── config + factory ────────────────────────────────────────────────────────


class WindowedSamplerConfig(BaseModel, extra="allow"):
    name: Literal["windowed"] = "windowed"
    # Max weight-version gap a selected group may have from the trainer.
    max_staleness_versions: int = 1
    # Prefer smallest lag when picking from the in-window set.
    sample_freshest_first: bool = False


class WeightFifoSamplerConfig(BaseModel, extra="allow"):
    name: Literal["weight_fifo"] = "weight_fifo"
    # Lookahead + selectable weight window, in trainer versions.
    max_staleness_versions: int = 1


class InOrderSamplerConfig(BaseModel, extra="allow"):
    name: Literal["in_order"] = "in_order"
    # How far generation may run ahead of the trainer, in dispatch batches.
    max_lookahead_versions: int = 1


class CustomSamplerConfig(BaseModel, extra="allow"):
    name: Literal["custom"] = "custom"
    # "module:ClassName" of a PromptGroupSampler defined outside this repo.
    # Extra keys are forwarded to the constructor (after ``buffer``).
    target: str


# Discriminated on ``name`` so each variant carries only its own typed args and
# pydantic narrows the type at construction — invalid arg combinations are
# unrepresentable rather than caught by a runtime assert.
SamplerConfig = Annotated[
    Union[
        WindowedSamplerConfig,
        WeightFifoSamplerConfig,
        InOrderSamplerConfig,
        CustomSamplerConfig,
    ],
    Field(discriminator="name"),
]


def create_sampler(
    buffer: TQReplayBuffer,
    cfg: SamplerConfig,
) -> PromptGroupSampler:
    """Build a sampler from its config (or import one by FQN)."""
    if isinstance(cfg, WindowedSamplerConfig):
        return WindowedSampler(
            buffer,
            max_staleness_versions=cfg.max_staleness_versions,
            sample_freshest_first=cfg.sample_freshest_first,
        )
    if isinstance(cfg, WeightFifoSamplerConfig):
        return WeightFifoSampler(
            buffer, max_staleness_versions=cfg.max_staleness_versions
        )
    if isinstance(cfg, InOrderSamplerConfig):
        return InOrderSampler(
            buffer, max_lookahead_versions=cfg.max_lookahead_versions
        )
    if isinstance(cfg, CustomSamplerConfig):
        module_name, sep, class_name = cfg.target.partition(":")
        if not sep:
            raise ValueError(
                f"custom sampler target must be 'module:ClassName', got "
                f"{cfg.target!r}"
            )
        sampler_cls = getattr(importlib.import_module(module_name), class_name)
        sampler = sampler_cls(buffer, **(cfg.model_extra or {}))
        if not isinstance(sampler, PromptGroupSampler):
            raise TypeError(
                f"{cfg.target} does not implement the PromptGroupSampler "
                f"interface (needs admit/select/evict)"
            )
        return sampler
    raise ValueError(f"unknown sampler config {type(cfg).__name__}")
