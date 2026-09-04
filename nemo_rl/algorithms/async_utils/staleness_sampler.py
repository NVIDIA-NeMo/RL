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
``ReadyFirstSampler`` / ``WeightFifoSampler`` / ``InOrderSampler`` are the
built-in policies, one per behavior, each owning only the args that apply to it.
``create_sampler`` builds one from a discriminated-union config (or a
``module:ClassName`` FQN for a policy defined outside this repo) — the config's
``name`` is the single source of truth for which behavior runs, so there are no
cross-field knob combinations to validate.
"""

from __future__ import annotations

import abc
import asyncio
import importlib
from typing import (
    Annotated,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field, NonNegativeInt, model_validator

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneMutationCut,
    TQReplayBuffer,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROLLOUT_METRICS

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

    def should_abort_inflight(
        self,
        *,
        start_weight_version: int,
        current_train_weight: int,
    ) -> bool:
        """Return whether an unfinished rollout can no longer be selected."""
        ...

    @property
    def is_on_policy(self) -> bool:
        """True when the policy admits zero staleness (sync mode)."""
        ...

    supports_buffer_checkpoint: ClassVar[bool]
    """Whether completed buffered groups can be restored safely."""

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        """Buffer-capacity the policy needs, or ``None`` if unconstrained."""
        ...

    @property
    def dispatch_index(self) -> int:
        """Last admitted dispatch batch index."""
        ...

    def set_dispatch_index(self, resume_from_trainer_version: int) -> None:
        """Seed the cursor for checkpoints that predate exact sampler state."""
        ...

    def restore_dispatch_index(self, dispatch_index: int) -> None:
        """Restore the exact dispatch cursor from controller state."""
        ...


@runtime_checkable
class TransactionalAdmissionSampler(Protocol):
    """Sampler whose blocking wait is separate from its cursor mutation."""

    async def wait_until_admissible(
        self, *, trainer_version_fn: Callable[[], int]
    ) -> None:
        """Wait until one admission can commit without mutating sampler state."""
        ...

    def commit_admission(self, cut: DataPlaneMutationCut) -> Optional[int]:
        """Advance the cursor under a live data-plane mutation cut."""
        ...


class BaseSampler(abc.ABC):
    """Shared machinery for the built-in policies.

    Owns the monotonic dispatch counter (the batch index formerly tracked as
    ``SingleControllerActor._max_rollout_version``) and the common
    select-finalize / weight-window-evict helpers.
    """

    supports_buffer_checkpoint: ClassVar[bool] = False

    def __init__(self, buffer: TQReplayBuffer) -> None:
        self._buffer = buffer
        # Pre-incremented before each admitted batch, so -1 lets the first
        # batch through a zero-staleness gate.
        self._dispatch_index: int = -1

    @property
    def dispatch_index(self) -> int:
        """Return the last admitted dispatch batch index."""
        return self._dispatch_index

    def set_dispatch_index(self, resume_from_trainer_version: int) -> None:
        """Seed the cursor for checkpoints that predate exact sampler state.

        Args:
            resume_from_trainer_version: Trainer version from which the run
                resumes. The next admitted batch receives that version.
        """
        if resume_from_trainer_version < 0:
            raise ValueError(
                "resume_from_trainer_version must be non-negative, got "
                f"{resume_from_trainer_version}"
            )
        self._dispatch_index = resume_from_trainer_version - 1

    def restore_dispatch_index(self, dispatch_index: int) -> None:
        """Restore the exact dispatch cursor.

        Args:
            dispatch_index: Last admitted batch index, or ``-1`` when no batch
                has been admitted. Call before the first ``admit``.
        """
        if dispatch_index < -1:
            raise ValueError(
                f"dispatch_index must be at least -1, got {dispatch_index}"
            )
        self._dispatch_index = dispatch_index

    # ── rollout-pump side ────────────────────────────────────────────────
    @abc.abstractmethod
    async def admit(
        self, *, trainer_version_fn: Callable[[], int]
    ) -> Optional[int]: ...

    # ── train-pump side ──────────────────────────────────────────────────
    @abc.abstractmethod
    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]: ...

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
    def should_abort_inflight(
        self,
        *,
        start_weight_version: int,
        current_train_weight: int,
    ) -> bool:
        return False

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
        selected_rollout_metrics = [
            metrics
            for meta in selected_metas
            for metrics in meta.extra_info.get(ROLLOUT_METRICS, [])  # type: ignore[union-attr]
        ]
        selected_meta = selected_metas[0].concat(*selected_metas[1:])  # type: ignore[union-attr]
        selected_meta.extra_info[ROLLOUT_METRICS] = selected_rollout_metrics
        await self._buffer.remove(selected_idxs, remove_in_dp=False)
        return selected_meta, len(selected_idxs)


class WindowedSampler(BaseSampler):
    """Over-sampled windowed selection.

    Rollout never gates on the trainer version — the pump keeps producing and
    samples aged past the window are evicted. ``select`` takes any ready group
    within ``[train_weight - max_staleness_versions, train_weight]``, optionally
    freshest-first.
    """

    # Ungated restored groups are ordinary in-window candidates.
    supports_buffer_checkpoint: ClassVar[bool] = True

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

    def should_abort_inflight(
        self,
        *,
        start_weight_version: int,
        current_train_weight: int,
    ) -> bool:
        min_valid_version = max(
            0,
            current_train_weight - self.max_staleness_versions,
        )
        return start_weight_version < min_valid_version

    async def wait_until_admissible(
        self, *, trainer_version_fn: Callable[[], int]
    ) -> None:
        """Return immediately because buffer capacity is this policy's gate."""
        del trainer_version_fn

    def commit_admission(self, cut: DataPlaneMutationCut) -> Optional[int]:
        """Return the unstamped admission result without changing a cursor."""
        cut.require_live()
        return None

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        await self.wait_until_admissible(trainer_version_fn=trainer_version_fn)
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


def _gated_required_buffer_capacity(
    groups_per_step: int,
    *,
    gate_window: int,
) -> int:
    """Return capacity for one live batch plus each lookahead batch."""
    return groups_per_step * (gate_window + 1)


class _GatedSampler(BaseSampler):
    """Base for policies that admit exactly one dispatch batch per trainer step.

    The gate bounds how far generation may run ahead of the trainer
    (``gate_window`` versions of lookahead).
    """

    def __init__(self, buffer: TQReplayBuffer, *, gate_window: int) -> None:
        super().__init__(buffer)
        self._gate_window = 0
        self.set_gate_window(gate_window)

    def set_gate_window(self, gate_window: int) -> None:
        if gate_window < 0:
            raise ValueError(f"gate_window must be non-negative, got {gate_window}")
        self._gate_window = gate_window

    def _eviction_window(self) -> int:
        return self._gate_window

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        # One batch of lookahead per version in the window, plus the live batch.
        return _gated_required_buffer_capacity(
            groups_per_step,
            gate_window=self._gate_window,
        )

    async def wait_until_admissible(
        self, *, trainer_version_fn: Callable[[], int]
    ) -> None:
        """Wait for the gate without advancing the durable dispatch cursor."""
        while self._dispatch_index >= trainer_version_fn() + self._gate_window:
            await asyncio.sleep(_GATE_POLL_SECONDS)

    def commit_admission(self, cut: DataPlaneMutationCut) -> Optional[int]:
        """Advance the cursor after the controller enters its mutation cut."""
        cut.require_live()
        return self._commit_admission()

    def _commit_admission(self) -> Optional[int]:
        """Advance admission for the legacy monolithic API."""
        self._dispatch_index += 1
        return self._stamp()

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        await self.wait_until_admissible(trainer_version_fn=trainer_version_fn)
        return self._commit_admission()

    def _stamp(self) -> Optional[int]:
        return None


class ReadyFirstSampler(_GatedSampler):
    """Gated admission with ready-first, mixed-version selection.

    Admission limits generation to ``max_staleness_versions`` dispatch batches
    ahead of the current trainer version. Selection remains ready-first across
    weight versions instead of draining one version at a time: every ready
    group generated by a policy version no newer than the trainer stays
    selectable, including late stragglers outside the admission window, so no
    rollout is ever discarded.
    """

    # Committed groups retain start_weight, which is sufficient for selection.
    supports_buffer_checkpoint: ClassVar[bool] = True

    def __init__(
        self,
        buffer: TQReplayBuffer,
        *,
        max_staleness_versions: int,
    ) -> None:
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
        valid_idxs = [
            i
            for i, weight in enumerate(self._buffer.start_weight_list)
            if weight <= current_train_weight and self._buffer.ready_list[i]
        ]
        return await self._finalize_selection(
            valid_idxs, min_prompt_groups, max_prompt_groups
        )

    async def evict(self, *, current_train_weight: int) -> int:
        # Every already-admitted group remains selectable; nothing is stale.
        return 0


class WeightFifoSampler(_GatedSampler):
    """Gated, strict weight-version FIFO.

    ``select`` drains the oldest in-window ``start_weight`` first and waits for
    that weight's batch to fill. Evict uses the weight window (default).
    """

    # Committed groups retain start_weight, which is sufficient for selection.
    supports_buffer_checkpoint: ClassVar[bool] = True

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

    warmup_lookahead_versions widens the gate while the PPO policy is frozen, so more
    batches stay in flight during critic warmup. The driver retunes the window every
    step and shrinks it back to max_lookahead_versions once the policy starts training,
    so the widened lookahead does not turn into permanent extra staleness. Buffer
    capacity is sized for the peak of the two.
    """

    # Committed groups retain target_step, which is sufficient for selection.
    # The controller checkpoints the exact dispatch cursor separately.
    supports_buffer_checkpoint: ClassVar[bool] = True

    def __init__(
        self,
        buffer: TQReplayBuffer,
        *,
        max_lookahead_versions: int,
        warmup_lookahead_versions: Optional[int] = None,
    ) -> None:
        super().__init__(buffer, gate_window=max_lookahead_versions)
        self.max_lookahead_versions = max_lookahead_versions
        self.warmup_lookahead_versions = warmup_lookahead_versions

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        # Sized for the peak window: otherwise the buffer, not the gate, bounds
        # the lookahead and widening it during warmup does nothing.
        if self.warmup_lookahead_versions is None:
            peak = self.max_lookahead_versions
        else:
            peak = self.warmup_lookahead_versions
        return _gated_required_buffer_capacity(groups_per_step, gate_window=peak)

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
    max_staleness_versions: NonNegativeInt = 1
    # Prefer smallest lag when picking from the in-window set.
    sample_freshest_first: bool = False


class ReadyFirstSamplerConfig(BaseModel, extra="allow"):
    name: Literal["ready_first"] = "ready_first"
    # How far generation may run ahead of the trainer, in dispatch batches.
    max_staleness_versions: NonNegativeInt = 1


class WeightFifoSamplerConfig(BaseModel, extra="allow"):
    name: Literal["weight_fifo"] = "weight_fifo"
    # Lookahead + selectable weight window, in trainer versions.
    max_staleness_versions: NonNegativeInt = 1


class InOrderSamplerConfig(BaseModel, extra="allow"):
    name: Literal["in_order"] = "in_order"
    # How far generation may run ahead of the trainer, in dispatch batches.
    max_lookahead_versions: NonNegativeInt = 1
    # Widened lookahead while the PPO policy is frozen; None keeps the steady value.
    warmup_lookahead_versions: Optional[NonNegativeInt] = None

    @model_validator(mode="after")
    def validate_warmup_lookahead(self) -> "InOrderSamplerConfig":
        if (
            self.warmup_lookahead_versions is not None
            and self.warmup_lookahead_versions < self.max_lookahead_versions
        ):
            raise ValueError(
                "warmup_lookahead_versions must be greater than or equal "
                "to max_lookahead_versions"
            )
        return self

    @property
    def peak_lookahead_versions(self) -> int:
        """Widest window the run can reach; what buffer capacity must cover."""
        if self.warmup_lookahead_versions is None:
            return self.max_lookahead_versions
        return self.warmup_lookahead_versions


class CustomSamplerConfig(BaseModel, extra="allow"):
    name: Literal["custom"] = "custom"
    # "module:ClassName" of a PromptGroupSampler defined outside this repo.
    # Extra keys are forwarded to the constructor (after ``buffer``). The
    # target class must declare a boolean ``supports_buffer_checkpoint`` class
    # attribute so setup can validate recovery requirements before allocating
    # cluster resources.
    target: str


# Discriminated on ``name`` so each variant carries only its own typed args and
# pydantic narrows the type at construction — invalid arg combinations are
# unrepresentable rather than caught by a runtime assert.
SamplerConfig = Annotated[
    Union[
        WindowedSamplerConfig,
        ReadyFirstSamplerConfig,
        WeightFifoSamplerConfig,
        InOrderSamplerConfig,
        CustomSamplerConfig,
    ],
    Field(discriminator="name"),
]


def required_buffer_capacity_for_config(
    cfg: SamplerConfig,
    groups_per_step: int,
) -> Optional[int]:
    """Return a built-in sampler's required capacity without constructing it."""
    if isinstance(cfg, ReadyFirstSamplerConfig):
        return _gated_required_buffer_capacity(
            groups_per_step,
            gate_window=cfg.max_staleness_versions,
        )
    if isinstance(cfg, WeightFifoSamplerConfig):
        return _gated_required_buffer_capacity(
            groups_per_step,
            gate_window=cfg.max_staleness_versions,
        )
    if isinstance(cfg, InOrderSamplerConfig):
        return _gated_required_buffer_capacity(
            groups_per_step,
            gate_window=cfg.peak_lookahead_versions,
        )
    return None


def _custom_sampler_class(cfg: CustomSamplerConfig) -> type:
    """Import and return a custom sampler class without constructing it."""
    module_name, sep, class_name = cfg.target.partition(":")
    if not sep:
        raise ValueError(
            f"custom sampler target must be 'module:ClassName', got {cfg.target!r}"
        )
    sampler_cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(sampler_cls, type):
        raise TypeError(f"custom sampler target is not a class: {cfg.target!r}")
    return sampler_cls


def _sampler_class_for_config(cfg: SamplerConfig) -> type:
    """Return the sampler class selected by a built-in or custom config."""
    if isinstance(cfg, CustomSamplerConfig):
        return _custom_sampler_class(cfg)
    try:
        return {
            WindowedSamplerConfig: WindowedSampler,
            ReadyFirstSamplerConfig: ReadyFirstSampler,
            WeightFifoSamplerConfig: WeightFifoSampler,
            InOrderSamplerConfig: InOrderSampler,
        }[type(cfg)]
    except KeyError:
        raise ValueError(f"unknown sampler config {type(cfg).__name__}") from None


def sampler_supports_buffer_checkpoint(cfg: SamplerConfig) -> bool:
    """Return a sampler class's static replay-checkpoint capability.

    Custom classes are imported but not instantiated, allowing setup to fail
    before allocating cluster resources or triggering constructor side effects.
    """
    sampler_cls = _sampler_class_for_config(cfg)

    if isinstance(cfg, CustomSamplerConfig):
        # A custom subclass must opt in explicitly instead of inheriting a
        # built-in sampler's capability declaration accidentally.
        capability = sampler_cls.__dict__.get("supports_buffer_checkpoint", False)
    else:
        capability = getattr(sampler_cls, "supports_buffer_checkpoint", None)
    if not isinstance(capability, bool):
        raise TypeError(
            f"{sampler_cls.__name__}.supports_buffer_checkpoint must be a "
            f"boolean class attribute, got {capability!r}"
        )
    return capability


def create_sampler(
    buffer: TQReplayBuffer,
    cfg: SamplerConfig,
) -> PromptGroupSampler:
    """Build a sampler from its config (or import one by FQN).

    Args:
        buffer: Shared TQReplayBuffer holding the candidate slots.
        cfg: Discriminated sampler config selecting the policy.
    """
    sampler_cls = _sampler_class_for_config(cfg)
    sampler: PromptGroupSampler
    if isinstance(cfg, WindowedSamplerConfig):
        sampler = sampler_cls(
            buffer,
            max_staleness_versions=cfg.max_staleness_versions,
            sample_freshest_first=cfg.sample_freshest_first,
        )
    elif isinstance(cfg, ReadyFirstSamplerConfig):
        sampler = sampler_cls(
            buffer,
            max_staleness_versions=cfg.max_staleness_versions,
        )
    elif isinstance(cfg, WeightFifoSamplerConfig):
        sampler = sampler_cls(
            buffer,
            max_staleness_versions=cfg.max_staleness_versions,
        )
    elif isinstance(cfg, InOrderSamplerConfig):
        sampler = sampler_cls(
            buffer,
            max_lookahead_versions=cfg.max_lookahead_versions,
            warmup_lookahead_versions=cfg.warmup_lookahead_versions,
        )
    elif isinstance(cfg, CustomSamplerConfig):
        sampler_supports_buffer_checkpoint(cfg)
        sampler = sampler_cls(buffer, **(cfg.model_extra or {}))
        if not isinstance(sampler, PromptGroupSampler):
            raise TypeError(
                f"{cfg.target} does not implement the PromptGroupSampler "
                f"interface (needs admit/select/evict/should_abort_inflight, "
                f"dispatch_index, set_dispatch_index, restore_dispatch_index, "
                f"is_on_policy, supports_buffer_checkpoint, "
                f"required_buffer_capacity)"
            )
    else:
        raise ValueError(f"unknown sampler config {type(cfg).__name__}")
    return sampler
