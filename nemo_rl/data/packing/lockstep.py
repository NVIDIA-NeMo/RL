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

"""Deterministic controller-side packing shared by multiple model sides.

The planner in this module owns logical bin membership. Model-specific code may
materialize the geometry stored in :class:`SidePackingPlan`, but it must not
change ``LockstepPackingPlan.bins`` or independently repack a side.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral


PhysicalSizeFn = Callable[[tuple[int, ...]], int]


def _normalize_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {normalized}.")
    return normalized


def _normalize_int_tuple(
    values: Sequence[int], *, name: str, minimum: int = 0
) -> tuple[int, ...]:
    return tuple(
        _normalize_int(value, name=f"{name}[{i}]", minimum=minimum)
        for i, value in enumerate(values)
    )


def _cumulative_lengths(lengths: Sequence[int]) -> tuple[int, ...]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)


@dataclass(frozen=True)
class LockstepPackingItem:
    """One occurrence in the canonical logical global batch.

    ``sample_id`` identifies the durable source row and may legitimately repeat,
    for example after validation padding. ``batch_item_id`` identifies this one
    occurrence and must be unique within the plan.
    """

    sample_id: Hashable
    batch_item_id: int

    def __post_init__(self) -> None:
        if self.sample_id is None:
            raise ValueError("sample_id must not be None.")
        try:
            hash(self.sample_id)
        except TypeError as error:
            raise ValueError(
                f"sample_id must be hashable, got {self.sample_id!r}."
            ) from error
        object.__setattr__(
            self,
            "batch_item_id",
            _normalize_int(self.batch_item_id, name="batch_item_id"),
        )


@dataclass(frozen=True)
class SidePackingSpec:
    """Controller input describing one model side's exact token geometry.

    Length tuples align one-to-one with the canonical ``items`` passed to
    :func:`build_lockstep_packing_plan`. ``raw_lengths`` are exact tokenizer
    lengths before backend padding. ``effective_lengths`` include every
    per-sample padding rule. ``physical_size_fn``, when supplied, receives the
    ordered effective lengths in a candidate bin and returns its complete
    physical size, including packed-tail padding. It must be deterministic and
    may not return a value smaller than the sum of the effective lengths.
    """

    side_id: str
    capacity: int
    raw_lengths: tuple[int, ...]
    effective_lengths: tuple[int, ...]
    physical_size_fn: PhysicalSizeFn | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.side_id, str) or not self.side_id:
            raise ValueError(
                f"side_id must be a non-empty string, got {self.side_id!r}."
            )
        object.__setattr__(
            self, "capacity", _normalize_int(self.capacity, name="capacity", minimum=1)
        )
        raw_lengths = _normalize_int_tuple(
            self.raw_lengths, name=f"{self.side_id}.raw_lengths", minimum=1
        )
        effective_lengths = _normalize_int_tuple(
            self.effective_lengths,
            name=f"{self.side_id}.effective_lengths",
            minimum=1,
        )
        if len(raw_lengths) != len(effective_lengths):
            raise ValueError(
                f"side {self.side_id!r} has {len(raw_lengths)} raw lengths but "
                f"{len(effective_lengths)} effective lengths."
            )
        for ordinal, (raw, effective) in enumerate(
            zip(raw_lengths, effective_lengths, strict=True)
        ):
            if effective < raw:
                raise ValueError(
                    f"side {self.side_id!r} item {ordinal} has effective length "
                    f"{effective} smaller than raw length {raw}."
                )
        if self.physical_size_fn is not None and not callable(self.physical_size_fn):
            raise ValueError(
                f"side {self.side_id!r} physical_size_fn must be callable or None."
            )
        object.__setattr__(self, "raw_lengths", raw_lengths)
        object.__setattr__(self, "effective_lengths", effective_lengths)

    def physical_size(self, item_ordinals: Sequence[int]) -> int:
        """Return the complete physical size of an ordered candidate bin."""
        effective_lengths = tuple(self.effective_lengths[i] for i in item_ordinals)
        if not effective_lengths:
            raise ValueError(f"side {self.side_id!r} cannot size an empty bin.")
        minimum_size = sum(effective_lengths)
        if self.physical_size_fn is None:
            return minimum_size
        try:
            physical_size = self.physical_size_fn(effective_lengths)
        except Exception as error:
            raise ValueError(
                f"side {self.side_id!r} physical_size_fn failed for effective "
                f"lengths {effective_lengths}."
            ) from error
        physical_size = _normalize_int(
            physical_size,
            name=f"side {self.side_id!r} physical size",
            minimum=1,
        )
        if physical_size < minimum_size:
            raise ValueError(
                f"side {self.side_id!r} physical_size_fn returned {physical_size} "
                f"for effective lengths {effective_lengths}, smaller than their "
                f"sum {minimum_size}."
            )
        return physical_size


@dataclass(frozen=True)
class SidePackingPlan:
    """Immutable model-specific geometry for shared logical bins."""

    side_id: str
    capacity: int
    raw_lengths: tuple[int, ...]
    effective_lengths: tuple[int, ...]
    raw_cu_seqlens_by_bin: tuple[tuple[int, ...], ...]
    padded_cu_seqlens_by_bin: tuple[tuple[int, ...], ...]
    physical_tokens_by_bin: tuple[int, ...]
    rank_bin_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.side_id, str) or not self.side_id:
            raise ValueError(
                f"side_id must be a non-empty string, got {self.side_id!r}."
            )
        object.__setattr__(
            self, "capacity", _normalize_int(self.capacity, name="capacity", minimum=1)
        )
        object.__setattr__(
            self,
            "raw_lengths",
            _normalize_int_tuple(
                self.raw_lengths, name=f"{self.side_id}.raw_lengths", minimum=1
            ),
        )
        object.__setattr__(
            self,
            "effective_lengths",
            _normalize_int_tuple(
                self.effective_lengths,
                name=f"{self.side_id}.effective_lengths",
                minimum=1,
            ),
        )
        if len(self.raw_lengths) != len(self.effective_lengths):
            raise ValueError(
                f"side {self.side_id!r} raw/effective length counts do not match."
            )
        for ordinal, (raw, effective) in enumerate(
            zip(self.raw_lengths, self.effective_lengths, strict=True)
        ):
            if effective < raw:
                raise ValueError(
                    f"side {self.side_id!r} item {ordinal} has effective length "
                    f"{effective} smaller than raw length {raw}."
                )

        raw_cu = tuple(
            _normalize_int_tuple(
                values,
                name=f"{self.side_id}.raw_cu_seqlens_by_bin[{bin_idx}]",
            )
            for bin_idx, values in enumerate(self.raw_cu_seqlens_by_bin)
        )
        padded_cu = tuple(
            _normalize_int_tuple(
                values,
                name=f"{self.side_id}.padded_cu_seqlens_by_bin[{bin_idx}]",
            )
            for bin_idx, values in enumerate(self.padded_cu_seqlens_by_bin)
        )
        physical_tokens = _normalize_int_tuple(
            self.physical_tokens_by_bin,
            name=f"{self.side_id}.physical_tokens_by_bin",
            minimum=1,
        )
        rank_bin_indices = tuple(
            _normalize_int_tuple(
                values, name=f"{self.side_id}.rank_bin_indices[{rank_idx}]"
            )
            for rank_idx, values in enumerate(self.rank_bin_indices)
        )
        object.__setattr__(self, "raw_cu_seqlens_by_bin", raw_cu)
        object.__setattr__(self, "padded_cu_seqlens_by_bin", padded_cu)
        object.__setattr__(self, "physical_tokens_by_bin", physical_tokens)
        object.__setattr__(self, "rank_bin_indices", rank_bin_indices)

        bin_count = len(physical_tokens)
        if bin_count == 0:
            raise ValueError(f"side {self.side_id!r} must contain at least one bin.")
        if len(raw_cu) != bin_count or len(padded_cu) != bin_count:
            raise ValueError(
                f"side {self.side_id!r} bin metadata counts differ: raw_cu="
                f"{len(raw_cu)}, padded_cu={len(padded_cu)}, physical={bin_count}."
            )
        for bin_idx, (raw, padded, physical) in enumerate(
            zip(raw_cu, padded_cu, physical_tokens, strict=True)
        ):
            if len(raw) < 2 or raw[0] != 0:
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} raw cu_seqlens must "
                    "start at zero and describe at least one item."
                )
            if len(padded) != len(raw) or padded[0] != 0:
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} raw and padded "
                    "cu_seqlens must have the same shape and start at zero."
                )
            if any(left >= right for left, right in zip(raw, raw[1:])):
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} raw cu_seqlens must "
                    "be strictly increasing."
                )
            if any(left >= right for left, right in zip(padded, padded[1:])):
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} padded cu_seqlens "
                    "must be strictly increasing."
                )
            if padded[-1] != physical:
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} padded cu_seqlens end "
                    f"at {padded[-1]}, not physical size {physical}."
                )
            if raw[-1] > physical:
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} raw size {raw[-1]} "
                    f"exceeds physical size {physical}."
                )
            if physical > self.capacity:
                raise ValueError(
                    f"side {self.side_id!r} bin {bin_idx} uses {physical} "
                    f"physical tokens, exceeding capacity {self.capacity}."
                )

        if not rank_bin_indices or any(not indices for indices in rank_bin_indices):
            raise ValueError(
                f"side {self.side_id!r} must assign at least one bin to every DP rank."
            )
        flattened_rank_bins = tuple(
            bin_idx for indices in rank_bin_indices for bin_idx in indices
        )
        if flattened_rank_bins != tuple(range(bin_count)):
            raise ValueError(
                f"side {self.side_id!r} rank_bin_indices must cover bins once in "
                f"contiguous rank order; got {rank_bin_indices}."
            )
        rank_bin_counts = {len(indices) for indices in rank_bin_indices}
        if len(rank_bin_counts) != 1:
            raise ValueError(
                f"side {self.side_id!r} has unequal DP bin counts "
                f"{tuple(len(indices) for indices in rank_bin_indices)}."
            )


class _FrozenSidePlans(Mapping[str, SidePackingPlan]):
    """Small immutable and pickle-friendly mapping used by a lockstep plan."""

    __slots__ = ("__items",)

    def __init__(self, values: Mapping[str, SidePackingPlan]):
        object.__setattr__(self, "_FrozenSidePlans__items", tuple(values.items()))

    def __getitem__(self, key: str) -> SidePackingPlan:
        for candidate, value in self.__items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.__items)

    def __len__(self) -> int:
        return len(self.__items)

    def __hash__(self) -> int:
        return hash(frozenset(self.__items))

    def __repr__(self) -> str:
        return repr(dict(self.__items))

    def __reduce__(self) -> tuple[object, tuple[dict[str, SidePackingPlan]]]:
        return type(self), (dict(self.__items),)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable.")


@dataclass(frozen=True)
class LockstepPackingPlan:
    """Immutable shared membership and side-local layout for one global batch."""

    batch_uid: int
    canonical_batch_item_ids: tuple[int, ...]
    bins: tuple[tuple[int, ...], ...]
    sides: Mapping[str, SidePackingPlan]
    equalization_splits: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "batch_uid", _normalize_int(self.batch_uid, name="batch_uid")
        )
        object.__setattr__(
            self,
            "equalization_splits",
            _normalize_int(self.equalization_splits, name="equalization_splits"),
        )
        canonical_ids = _normalize_int_tuple(
            self.canonical_batch_item_ids, name="canonical_batch_item_ids"
        )
        bins = tuple(
            _normalize_int_tuple(values, name=f"bins[{bin_idx}]")
            for bin_idx, values in enumerate(self.bins)
        )
        object.__setattr__(self, "canonical_batch_item_ids", canonical_ids)
        object.__setattr__(self, "bins", bins)

        if not canonical_ids:
            raise ValueError("LockstepPackingPlan requires at least one batch item.")
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError(
                "canonical_batch_item_ids must be occurrence-unique; got "
                f"{canonical_ids}."
            )
        if not bins or any(not values for values in bins):
            raise ValueError("LockstepPackingPlan bins must all be non-empty.")
        flattened_ids = tuple(
            batch_item_id for values in bins for batch_item_id in values
        )
        if len(flattened_ids) != len(canonical_ids) or set(flattened_ids) != set(
            canonical_ids
        ):
            raise ValueError(
                "LockstepPackingPlan bins must contain every canonical "
                "batch_item_id exactly once."
            )

        if not isinstance(self.sides, Mapping) or not self.sides:
            raise ValueError("LockstepPackingPlan requires at least one side.")
        frozen_sides = _FrozenSidePlans(dict(self.sides))
        object.__setattr__(self, "sides", frozen_sides)

        id_to_ordinal = {
            batch_item_id: ordinal
            for ordinal, batch_item_id in enumerate(canonical_ids)
        }
        reference_routing: tuple[tuple[int, ...], ...] | None = None
        for side_key, side in frozen_sides.items():
            if not isinstance(side, SidePackingPlan):
                raise ValueError(
                    f"side {side_key!r} must be a SidePackingPlan, got "
                    f"{type(side).__name__}."
                )
            if side_key != side.side_id:
                raise ValueError(
                    f"side mapping key {side_key!r} does not match plan side_id "
                    f"{side.side_id!r}."
                )
            if len(side.raw_lengths) != len(canonical_ids):
                raise ValueError(
                    f"side {side.side_id!r} has {len(side.raw_lengths)} lengths "
                    f"for {len(canonical_ids)} canonical batch items."
                )
            if len(side.physical_tokens_by_bin) != len(bins):
                raise ValueError(
                    f"side {side.side_id!r} has geometry for "
                    f"{len(side.physical_tokens_by_bin)} bins, expected {len(bins)}."
                )
            if reference_routing is None:
                reference_routing = side.rank_bin_indices
            elif side.rank_bin_indices != reference_routing:
                raise ValueError(
                    f"side {side.side_id!r} rank routing differs from the shared "
                    "lockstep routing."
                )

            for bin_idx, batch_item_ids in enumerate(bins):
                ordinals = tuple(id_to_ordinal[item_id] for item_id in batch_item_ids)
                raw_lengths = tuple(side.raw_lengths[i] for i in ordinals)
                effective_lengths = tuple(side.effective_lengths[i] for i in ordinals)
                expected_raw_cu = _cumulative_lengths(raw_lengths)
                if side.raw_cu_seqlens_by_bin[bin_idx] != expected_raw_cu:
                    raise ValueError(
                        f"side {side.side_id!r} bin {bin_idx} raw cu_seqlens do "
                        "not match shared bin membership."
                    )
                expected_effective_cu = _cumulative_lengths(effective_lengths)
                padded_cu = side.padded_cu_seqlens_by_bin[bin_idx]
                if padded_cu[:-1] != expected_effective_cu[:-1]:
                    raise ValueError(
                        f"side {side.side_id!r} bin {bin_idx} padded cu_seqlens "
                        "do not preserve per-item effective boundaries."
                    )
                if padded_cu[-1] < expected_effective_cu[-1]:
                    raise ValueError(
                        f"side {side.side_id!r} bin {bin_idx} padded size "
                        f"{padded_cu[-1]} is smaller than effective size "
                        f"{expected_effective_cu[-1]}."
                    )

        assert reference_routing is not None
        dp_size = len(reference_routing)
        if len(canonical_ids) % dp_size != 0:
            raise ValueError(
                f"{len(canonical_ids)} canonical items cannot be divided into "
                f"{dp_size} equal contiguous DP shards."
            )
        items_per_rank = len(canonical_ids) // dp_size
        for rank, rank_bins in enumerate(reference_routing):
            rank_ordinals = {
                id_to_ordinal[item_id]
                for bin_idx in rank_bins
                for item_id in bins[bin_idx]
            }
            expected_ordinals = set(
                range(rank * items_per_rank, (rank + 1) * items_per_rank)
            )
            if rank_ordinals != expected_ordinals:
                raise ValueError(
                    f"DP rank {rank} bins do not contain exactly its canonical "
                    "contiguous logical shard."
                )


def _pack_dp_shard(
    item_ordinals: Sequence[int], side_specs: Sequence[SidePackingSpec]
) -> list[list[int]]:
    pressures = {
        item_ordinal: max(
            Fraction(side.effective_lengths[item_ordinal], side.capacity)
            for side in side_specs
        )
        for item_ordinal in item_ordinals
    }
    decreasing = sorted(
        item_ordinals, key=lambda item_ordinal: (-pressures[item_ordinal], item_ordinal)
    )

    bins: list[list[int]] = []
    for item_ordinal in decreasing:
        # First-fit is lockstep across model sides: reuse the first existing bin
        # only when the candidate's complete side-local geometry fits every side.
        for bin_items in bins:
            candidate = (*bin_items, item_ordinal)
            if all(
                side.physical_size(candidate) <= side.capacity for side in side_specs
            ):
                bin_items.append(item_ordinal)
                break
        else:
            # Every item was prevalidated to fit an empty bin. If no existing
            # bin has room on all sides, retain the item as the new bin's first.
            bins.append([item_ordinal])
    return bins


def _equalize_rank_bin_counts(
    rank_bins: list[list[list[int]]], *, canonical_item_count: int
) -> int:
    target_count = max(len(bins) for bins in rank_bins)
    split_count = 0
    for rank, bins in enumerate(rank_bins):
        while len(bins) < target_count:
            splittable = [
                bin_idx for bin_idx, bin_items in enumerate(bins) if len(bin_items) > 1
            ]
            if not splittable:
                raise ValueError(
                    f"DP rank {rank} has {len(bins)} bins and cannot reach the "
                    f"required equalized count {target_count} by splitting its "
                    f"logical samples (global batch has {canonical_item_count} items)."
                )
            source_bin_idx = max(splittable, key=lambda i: (len(bins[i]), -i))
            moved_item = bins[source_bin_idx].pop()
            bins.append([moved_item])
            split_count += 1
    return split_count


def build_lockstep_packing_plan(
    *,
    batch_uid: int,
    items: Sequence[LockstepPackingItem],
    sides: Sequence[SidePackingSpec],
    data_parallel_size: int,
) -> LockstepPackingPlan:
    """Build one deterministic multi-side FFD plan over contiguous DP shards.

    The input order is the canonical global-batch order. It is divided evenly
    into contiguous DP shards before packing. Each shard is packed independently
    with stable multi-dimensional first-fit-decreasing. Ranks with fewer bins are
    equalized only by deterministic subset splitting; logical ownership never
    changes.
    """
    batch_uid = _normalize_int(batch_uid, name="batch_uid")
    data_parallel_size = _normalize_int(
        data_parallel_size, name="data_parallel_size", minimum=1
    )
    canonical_items = tuple(items)
    side_specs = tuple(sides)
    if not canonical_items:
        raise ValueError("Cannot build a lockstep packing plan for an empty batch.")
    if not side_specs:
        raise ValueError("Cannot build a lockstep packing plan without model sides.")
    if any(not isinstance(item, LockstepPackingItem) for item in canonical_items):
        raise ValueError("items must contain only LockstepPackingItem records.")
    if any(not isinstance(side, SidePackingSpec) for side in side_specs):
        raise ValueError("sides must contain only SidePackingSpec records.")

    batch_item_ids = tuple(item.batch_item_id for item in canonical_items)
    if len(set(batch_item_ids)) != len(batch_item_ids):
        duplicates = sorted(
            item_id
            for item_id in set(batch_item_ids)
            if batch_item_ids.count(item_id) > 1
        )
        raise ValueError(
            "batch_item_id must be unique for every logical occurrence; duplicate "
            f"IDs: {duplicates}."
        )
    side_ids = tuple(side.side_id for side in side_specs)
    if len(set(side_ids)) != len(side_ids):
        raise ValueError(f"side_id values must be unique, got {side_ids}.")
    if len(canonical_items) % data_parallel_size != 0:
        raise ValueError(
            f"global batch has {len(canonical_items)} logical items, which is not "
            f"divisible by data_parallel_size={data_parallel_size}. Pad validation "
            "batches before assigning batch_item_id and planning."
        )
    for side in side_specs:
        if len(side.raw_lengths) != len(canonical_items):
            raise ValueError(
                f"side {side.side_id!r} has {len(side.raw_lengths)} lengths for "
                f"{len(canonical_items)} logical items."
            )

    for item_ordinal, item in enumerate(canonical_items):
        for side in side_specs:
            physical_tokens = side.physical_size((item_ordinal,))
            if physical_tokens > side.capacity:
                raise ValueError(
                    f"sample_id={item.sample_id!r} (batch_item_id={item.batch_item_id}) "
                    f"does not fit side {side.side_id!r}: raw_length="
                    f"{side.raw_lengths[item_ordinal]}, effective_length="
                    f"{side.effective_lengths[item_ordinal]}, physical_tokens="
                    f"{physical_tokens}, capacity={side.capacity}."
                )

    items_per_rank = len(canonical_items) // data_parallel_size
    rank_bins = [
        _pack_dp_shard(
            range(rank * items_per_rank, (rank + 1) * items_per_rank), side_specs
        )
        for rank in range(data_parallel_size)
    ]
    equalization_splits = _equalize_rank_bin_counts(
        rank_bins, canonical_item_count=len(canonical_items)
    )

    flattened_ordinal_bins: list[list[int]] = []
    rank_bin_indices: list[tuple[int, ...]] = []
    for bins in rank_bins:
        first_bin_idx = len(flattened_ordinal_bins)
        flattened_ordinal_bins.extend(bins)
        rank_bin_indices.append(tuple(range(first_bin_idx, first_bin_idx + len(bins))))
    shared_bins = tuple(
        tuple(batch_item_ids[item_ordinal] for item_ordinal in bin_items)
        for bin_items in flattened_ordinal_bins
    )
    shared_rank_bin_indices = tuple(rank_bin_indices)

    side_plans: dict[str, SidePackingPlan] = {}
    for side in side_specs:
        raw_cu_by_bin: list[tuple[int, ...]] = []
        padded_cu_by_bin: list[tuple[int, ...]] = []
        physical_tokens_by_bin: list[int] = []
        for bin_items in flattened_ordinal_bins:
            raw_lengths = tuple(side.raw_lengths[i] for i in bin_items)
            effective_lengths = tuple(side.effective_lengths[i] for i in bin_items)
            physical_tokens = side.physical_size(bin_items)
            if physical_tokens > side.capacity:
                raise ValueError(
                    f"internal lockstep packing error: side {side.side_id!r} bin "
                    f"{tuple(batch_item_ids[i] for i in bin_items)} uses "
                    f"{physical_tokens} physical tokens, exceeding capacity "
                    f"{side.capacity}."
                )
            raw_cu = _cumulative_lengths(raw_lengths)
            padded_cu = list(_cumulative_lengths(effective_lengths))
            padded_cu[-1] = physical_tokens
            raw_cu_by_bin.append(raw_cu)
            padded_cu_by_bin.append(tuple(padded_cu))
            physical_tokens_by_bin.append(physical_tokens)

        side_plans[side.side_id] = SidePackingPlan(
            side_id=side.side_id,
            capacity=side.capacity,
            raw_lengths=side.raw_lengths,
            effective_lengths=side.effective_lengths,
            raw_cu_seqlens_by_bin=tuple(raw_cu_by_bin),
            padded_cu_seqlens_by_bin=tuple(padded_cu_by_bin),
            physical_tokens_by_bin=tuple(physical_tokens_by_bin),
            rank_bin_indices=shared_rank_bin_indices,
        )

    return LockstepPackingPlan(
        batch_uid=batch_uid,
        canonical_batch_item_ids=batch_item_ids,
        bins=shared_bins,
        sides=side_plans,
        equalization_splits=equalization_splits,
    )
