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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

import torch

from nemo_rl.weight_sync.interfaces import WeightSyncSelection

SpeculatorType = Literal["eagle3", "dflash", "dspark"]


class SpeculatorRuntimeError(RuntimeError):
    """The runtime cannot safely apply a speculative-model refit."""


class RunnerFamily(str, Enum):
    """Capability used to access the live speculative model."""

    ACCESSOR = "get_draft_model"
    DRAFTER = "drafter.model"
    SPECULATOR = "speculator.model"


@dataclass(frozen=True, slots=True)
class WeightComponentManifest:
    """Ordered transport and loading contract for one model component."""

    component: Literal["target", "draft"]
    ordered_names: tuple[str, ...]
    byte_count: int
    owner_ranks: tuple[int, ...]
    loader: str
    finalizer: str


@dataclass(frozen=True, slots=True)
class ModelUpdateManifest:
    """Transport-neutral target and optional draft update description."""

    target: WeightComponentManifest
    draft: WeightComponentManifest | None

    @classmethod
    def from_state_dict_info(
        cls,
        state_dict_info: Mapping[str, tuple[Sequence[int], torch.dtype]],
        *,
        target_owner_ranks: tuple[int, ...],
        draft_owner_ranks: tuple[int, ...],
    ) -> ModelUpdateManifest:
        target_names = tuple(
            name for name in state_dict_info if not name.startswith("draft.")
        )
        draft_names = tuple(
            name for name in state_dict_info if name.startswith("draft.")
        )

        def byte_count(names: tuple[str, ...]) -> int:
            total = 0
            for name in names:
                shape, dtype = state_dict_info[name]
                total += int(torch.Size(shape).numel()) * dtype.itemsize
            return total

        target = WeightComponentManifest(
            component="target",
            ordered_names=target_names,
            byte_count=byte_count(target_names),
            owner_ranks=target_owner_ranks,
            loader="target.load_weights",
            finalizer="process_weights_after_loading",
        )
        draft = None
        if draft_names:
            draft = WeightComponentManifest(
                component="draft",
                ordered_names=draft_names,
                byte_count=byte_count(draft_names),
                owner_ranks=draft_owner_ranks,
                loader="draft.load_weights",
                finalizer="process_weights_after_loading",
            )
        return cls(target=target, draft=draft)

    def for_selection(self, selection: WeightSyncSelection) -> ModelUpdateManifest:
        """Return an immutable transfer contract for ``selection``.

        The source and receiver must derive this exact manifest from their
        common selection, so a target-only sync cannot carry, load, or finalize
        draft state.  The original manifest remains reusable for later full
        synchronizations.
        """
        if selection.draft:
            return self
        return ModelUpdateManifest(target=self.target, draft=None)

    @property
    def ordered_names(self) -> tuple[str, ...]:
        names = self.target.ordered_names
        if self.draft is not None:
            names += self.draft.ordered_names
        return names


class ModelUpdateCoverage:
    """Validate exact input coverage while allowing explicit non-owner skips."""

    def __init__(
        self,
        manifest: ModelUpdateManifest,
        *,
        rank: int,
        draft_selected: bool,
    ) -> None:
        self._manifest = manifest
        self._rank = rank
        self._draft_selected = draft_selected
        self._expected = set(manifest.target.ordered_names)
        if manifest.draft is not None:
            self._expected.update(manifest.draft.ordered_names)
        self._covered: set[str] = set()

    @property
    def has_draft(self) -> bool:
        return self._manifest.draft is not None

    @property
    def draft_selected(self) -> bool:
        return self._draft_selected

    @property
    def expected_names(self) -> tuple[str, ...]:
        return self._manifest.ordered_names

    def _record(self, names: Sequence[str]) -> None:
        incoming: set[str] = set()
        duplicate: set[str] = set()
        for name in names:
            if name in incoming:
                duplicate.add(name)
            incoming.add(name)
        duplicate.update(incoming & self._covered)
        unexpected = incoming - self._expected
        if duplicate:
            raise SpeculatorRuntimeError(
                f"duplicate keys ({len(duplicate)}): {sorted(duplicate)[:8]}"
            )
        if unexpected:
            raise SpeculatorRuntimeError(
                f"unexpected keys ({len(unexpected)}): {sorted(unexpected)[:8]}"
            )
        self._covered.update(incoming)

    def record_loaded(self, names: Sequence[str]) -> None:
        self._record(names)

    def record_owner_skip(
        self, names: Sequence[str], *, component: Literal["target", "draft"]
    ) -> None:
        manifest = self._manifest.target
        if component == "draft":
            manifest = self._manifest.draft
        if manifest is None or self._rank in manifest.owner_ranks:
            raise SpeculatorRuntimeError(
                f"rank {self._rank} cannot skip owned {component} weights"
            )
        allowed = set(manifest.ordered_names)
        invalid = set(names) - allowed
        if invalid:
            raise SpeculatorRuntimeError(
                f"unexpected {component} owner skips: {sorted(invalid)[:8]}"
            )
        self._record(names)

    def require_complete(self) -> None:
        missing = self._expected - self._covered
        if missing:
            raise SpeculatorRuntimeError(
                f"missing keys ({len(missing)}): {sorted(missing)[:8]}"
            )


@dataclass(slots=True)
class DraftRuntimeAdapter:
    """Capability-driven access to a live vLLM speculative model."""

    speculator_type: SpeculatorType
    vllm_version: str
    runner_family: RunnerFamily
    model: Any | None
    pp_rank: int
    pp_size: int
    is_owner: bool

    @classmethod
    def resolve(
        cls,
        model_runner: Any,
        *,
        speculator_type: str,
        vllm_version: str,
        pp_rank: int,
        pp_size: int,
    ) -> DraftRuntimeAdapter:
        if speculator_type not in ("eagle3", "dflash", "dspark"):
            raise SpeculatorRuntimeError(
                f"unsupported speculator_type={speculator_type!r} with "
                f"vLLM={vllm_version}"
            )
        if pp_size <= 0 or not 0 <= pp_rank < pp_size:
            raise SpeculatorRuntimeError(
                f"invalid pipeline rank {pp_rank} for PP={pp_size}"
            )

        accessor = getattr(model_runner, "get_draft_model", None)
        if callable(accessor):
            family = RunnerFamily.ACCESSOR
            model = accessor()
        elif hasattr(model_runner, "drafter"):
            family = RunnerFamily.DRAFTER
            model = getattr(getattr(model_runner, "drafter"), "model", None)
        elif hasattr(model_runner, "speculator"):
            family = RunnerFamily.SPECULATOR
            model = getattr(getattr(model_runner, "speculator"), "model", None)
        else:
            raise SpeculatorRuntimeError(
                f"speculator_type={speculator_type}, vLLM={vllm_version}: no "
                "get_draft_model, drafter.model, or speculator.model capability"
            )

        if speculator_type in ("dflash", "dspark") and pp_size != 1:
            raise SpeculatorRuntimeError(
                f"speculator_type={speculator_type}, vLLM={vllm_version}, "
                f"runner_family={family.value}: PP={pp_size} is not supported"
            )

        is_owner = pp_rank == pp_size - 1
        if is_owner and model is None:
            raise SpeculatorRuntimeError(
                f"speculator_type={speculator_type}, vLLM={vllm_version}, "
                f"runner_family={family.value}: owner draft model is unavailable"
            )
        return cls(
            speculator_type=speculator_type,
            vllm_version=vllm_version,
            runner_family=family,
            model=model,
            pp_rank=pp_rank,
            pp_size=pp_size,
            is_owner=is_owner,
        )

    @property
    def owner_ranks(self) -> tuple[int, ...]:
        return (self.pp_size - 1,)
