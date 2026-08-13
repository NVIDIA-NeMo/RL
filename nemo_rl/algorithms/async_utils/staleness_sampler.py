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
from dataclasses import dataclass
from typing import (
    Annotated,
    Callable,
    Iterable,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, Field, NonNegativeInt

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.data_plane import KVBatchMeta

# Poll interval for the rollout-pump admission gate.
_GATE_POLL_SECONDS = 0.005


@dataclass(frozen=True)
class GroupLengthStats:
    """Sequence-length totals over a set of prompt groups.

    ``groups`` counts every group looked at; ``measured_groups`` counts only
    those carrying usable ``KVBatchMeta.sequence_lengths``. The two differ when
    a slot is unready or its metadata is malformed, so a reader can tell a
    genuine zero from an unmeasured pass.
    """

    groups: int
    measured_groups: int
    samples: int
    tokens: int

    @property
    def mean_tokens_per_group(self) -> float:
        """Mean tokens per measured group; ``0.0`` when nothing was measured."""
        if self.measured_groups == 0:
            return 0.0
        return self.tokens / self.measured_groups


def summarize_group_lengths(
    metas: Iterable[Optional[KVBatchMeta]],
) -> GroupLengthStats:
    """Total the sequence lengths carried by ``metas``.

    Instrumentation helper: a group whose metadata is absent (an unready slot
    holds ``None``) or unparseable is counted in ``groups`` but skipped for the
    token totals rather than raising, because the callers run inside the train
    pump where an exception would end the run.

    Args:
        metas: Per-group metadata, possibly containing ``None`` entries.

    Returns:
        Totals over the groups that had usable ``sequence_lengths``.
    """
    groups = 0
    measured_groups = 0
    samples = 0
    tokens = 0
    for meta in metas:
        groups += 1
        if meta is None or not meta.sequence_lengths:
            continue
        try:
            group_tokens = sum(int(length) for length in meta.sequence_lengths)
        except (TypeError, ValueError):
            continue
        measured_groups += 1
        samples += len(meta.sequence_lengths)
        tokens += group_tokens
    return GroupLengthStats(
        groups=groups,
        measured_groups=measured_groups,
        samples=samples,
        tokens=tokens,
    )


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

    def set_dispatch_index(self, resume_from_step: int) -> None:
        """Seed the dispatch cursor when resuming from a checkpoint."""
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
        # Run-cumulative selection-side totals. Kept so an eviction line can
        # state, within one run, whether the groups being dropped are longer
        # than the ones actually trained on.
        self._selected_groups: int = 0
        self._selected_tokens: int = 0

    def set_dispatch_index(self, resume_from_step: int) -> None:
        """Seed the dispatch cursor when resuming from a checkpoint.

        Args:
            resume_from_step: Trainer step this run starts from — 0 for a
                fresh run, the restored ``current_step`` when resuming. Sets
                the cursor to ``resume_from_step - 1`` so gated ``admit`` and
                ``InOrderSampler``'s target_step stamps line up with the
                restored trainer version exactly as at step 0 of a fresh run.
                Call before the first ``admit``.
        """
        if resume_from_step < 0:
            raise ValueError(
                f"resume_from_step must be non-negative, got {resume_from_step}"
            )
        self._dispatch_index = resume_from_step - 1

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
        return await self._evict_idxs(
            stale_idxs, current_train_weight=current_train_weight
        )

    # ── derived facts ────────────────────────────────────────────────────
    @property
    def is_on_policy(self) -> bool:
        return self._eviction_window() == 0

    @property
    def selected_mean_tokens_per_group(self) -> float:
        """Mean tokens per selected group so far; ``0.0`` before any selection."""
        if self._selected_groups == 0:
            return 0.0
        return self._selected_tokens / self._selected_groups

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        return None

    # ── shared helpers ───────────────────────────────────────────────────
    def _eviction_window(self) -> int:
        """Weight-version span kept selectable; drives the default ``evict``."""
        return 0

    async def _evict_idxs(
        self, stale_idxs: list[int], *, current_train_weight: int
    ) -> int:
        """Report then drop ``stale_idxs``; shared by every policy's ``evict``.

        Args:
            stale_idxs: Buffer indices to drop, from a key the policy's
                ``select`` agrees with.
            current_train_weight: Trainer version driving this evict pass.

        Returns:
            Number of groups removed from the buffer.
        """
        if not stale_idxs:
            return 0
        self._report_eviction(stale_idxs, current_train_weight=current_train_weight)
        return await self._buffer.remove(stale_idxs, remove_in_dp=True)

    def _report_eviction(
        self, stale_idxs: list[int], *, current_train_weight: int
    ) -> None:
        """Print evicted-vs-selected sequence lengths for one evict pass.

        Answers, from the log alone, whether the groups a policy discards are
        the long ones: a windowed policy keys ``select`` on the *dispatch*-time
        weight, so a slow rollout can commit too late to ever be selected and
        is dropped here instead. Runs before ``remove`` while ``meta_list`` is
        still populated, and is synchronous so no ``commit`` can interleave.
        """
        stats = summarize_group_lengths(self._buffer.meta_list[i] for i in stale_idxs)
        evicted_mean = stats.mean_tokens_per_group
        selected_mean = self.selected_mean_tokens_per_group
        ratio = (
            f"{evicted_mean / selected_mean:.2f}x"
            if evicted_mean > 0 and selected_mean > 0
            else "n/a"
        )
        print(
            f"♻️  eviction lengths (train_weight={current_train_weight}): "
            f"evicted_groups={stats.groups} "
            f"measured_groups={stats.measured_groups} "
            f"evicted_samples={stats.samples} "
            f"evicted_tokens={stats.tokens} "
            f"evicted_mean_tokens_per_group={evicted_mean:.1f} "
            f"selected_groups={self._selected_groups} "
            f"selected_tokens={self._selected_tokens} "
            f"selected_mean_tokens_per_group={selected_mean:.1f} "
            f"evicted_over_selected={ratio}",
            flush=True,
        )

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
        # Same statistic the eviction line reports, over the same field, so the
        # two are comparable within a run rather than across runs.
        selected_stats = summarize_group_lengths(selected_metas)
        self._selected_groups += selected_stats.measured_groups
        self._selected_tokens += selected_stats.tokens
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

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        # One full batch, not the gated samplers' batch-per-version: windowed
        # needs no lookahead residency, it just has to be able to hold enough
        # for `select` to reach min_prompt_groups. Below that the train pump
        # waits on a batch the buffer is too small to ever offer, and without a
        # floor here validate_sampler_buffer_capacity skips this sampler
        # entirely and the misconfiguration presents as a silent hang.
        return groups_per_step

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

    async def admit(self, *, trainer_version_fn: Callable[[], int]) -> Optional[int]:
        while self._dispatch_index >= trainer_version_fn() + self._gate_window:
            await asyncio.sleep(_GATE_POLL_SECONDS)
        self._dispatch_index += 1
        return self._stamp()

    def _stamp(self) -> Optional[int]:
        return None


class ReadyFirstSampler(_GatedSampler):
    """Gated admission with ready-first, mixed-version selection.

    Admission limits generation to ``max_staleness_versions`` dispatch batches
    ahead of the current trainer version. Selection remains ready-first across
    weight versions instead of draining one version at a time.

    When ``evict_stale_samples`` is false, every ready group generated by a
    policy version no newer than the trainer remains selectable, including late
    stragglers outside the admission window. When it is true, selection and
    eviction share the hard weight window
    ``[trainer_version - max_staleness_versions, trainer_version]``.

    ``evict_stale_samples=false`` is the intended setting, and not only as a
    default. A backstop belongs where the work is not yet done: dropping a
    rollout that has produced nothing costs nothing, while evicting a committed
    group discards finished work, which is the thing this policy exists to
    stop. So ``max_staleness_versions`` is an admission window here — it bounds
    how far generation may run ahead, not how stale a *trained* group may be.
    Realized staleness is bounded instead by a group's own latency, roughly
    ``ceil(rollout_latency / step_time) + 1``.

    ``evict_stale_samples=true`` re-arms a wedge and should stay off. Eviction
    removes groups without crediting the admission counter, so with ``P``
    groups per step and ``eta = max_staleness_versions``, more than ``P * eta``
    cumulative evictions leave the gate closed against a step that can no
    longer fill, and the run hangs rather than erroring. A retention-side cap
    is only safe if every evicted group advances the admission counter by its
    share.
    """

    def __init__(
        self,
        buffer: TQReplayBuffer,
        *,
        max_staleness_versions: int,
        evict_stale_samples: bool,
    ) -> None:
        super().__init__(buffer, gate_window=max_staleness_versions)
        self.max_staleness_versions = max_staleness_versions
        self.evict_stale_samples = evict_stale_samples

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
            if weight <= current_train_weight
            and (not self.evict_stale_samples or weight >= min_valid_version)
            and self._buffer.ready_list[i]
        ]
        return await self._finalize_selection(
            valid_idxs, min_prompt_groups, max_prompt_groups
        )

    async def evict(self, *, current_train_weight: int) -> int:
        if not self.evict_stale_samples:
            return 0
        return await super().evict(current_train_weight=current_train_weight)


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
        return await self._evict_idxs(
            stale_idxs, current_train_weight=current_train_weight
        )


# ── config + factory ────────────────────────────────────────────────────────


class WindowedSamplerConfig(BaseModel, extra="allow"):
    name: Literal["windowed"] = "windowed"
    # Max weight-version gap a selected group may have from the trainer.
    max_staleness_versions: NonNegativeInt = 1
    # Prefer smallest lag when picking from the in-window set.
    sample_freshest_first: bool = False


class ReadyFirstSamplerConfig(BaseModel, extra="allow"):
    name: Literal["ready_first"] = "ready_first"
    # Dispatch lookahead and, when eviction is enabled, selectable weight window.
    max_staleness_versions: NonNegativeInt = 1
    # Keep late stragglers trainable by default; enable for a hard staleness cap.
    evict_stale_samples: bool = False


class WeightFifoSamplerConfig(BaseModel, extra="allow"):
    name: Literal["weight_fifo"] = "weight_fifo"
    # Lookahead + selectable weight window, in trainer versions.
    max_staleness_versions: NonNegativeInt = 1


class InOrderSamplerConfig(BaseModel, extra="allow"):
    name: Literal["in_order"] = "in_order"
    # How far generation may run ahead of the trainer, in dispatch batches.
    max_lookahead_versions: NonNegativeInt = 1


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
            gate_window=cfg.max_lookahead_versions,
        )
    if isinstance(cfg, WindowedSamplerConfig):
        # Keep in step with WindowedSampler.required_buffer_capacity: one batch,
        # since windowed carries no lookahead residency requirement.
        return groups_per_step
    return None


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
    if isinstance(cfg, ReadyFirstSamplerConfig):
        return ReadyFirstSampler(
            buffer,
            max_staleness_versions=cfg.max_staleness_versions,
            evict_stale_samples=cfg.evict_stale_samples,
        )
    if isinstance(cfg, WeightFifoSamplerConfig):
        return WeightFifoSampler(
            buffer, max_staleness_versions=cfg.max_staleness_versions
        )
    if isinstance(cfg, InOrderSamplerConfig):
        return InOrderSampler(buffer, max_lookahead_versions=cfg.max_lookahead_versions)
    if isinstance(cfg, CustomSamplerConfig):
        module_name, sep, class_name = cfg.target.partition(":")
        if not sep:
            raise ValueError(
                f"custom sampler target must be 'module:ClassName', got {cfg.target!r}"
            )
        sampler_cls = getattr(importlib.import_module(module_name), class_name)
        sampler = sampler_cls(buffer, **(cfg.model_extra or {}))
        if not isinstance(sampler, PromptGroupSampler):
            raise TypeError(
                f"{cfg.target} does not implement the PromptGroupSampler "
                f"interface (needs admit/select/evict, set_dispatch_index, "
                f"is_on_policy, required_buffer_capacity)"
            )
        return sampler
    raise ValueError(f"unknown sampler config {type(cfg).__name__}")
