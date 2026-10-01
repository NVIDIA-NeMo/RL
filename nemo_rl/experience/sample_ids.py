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
"""Canonical TQ sample-id scheme shared by the finalizer, controller, and buffer.

Pure stdlib on purpose: the controller's id parsing, the replay buffer's
checkpoint validator, and the finalizer's row minting must all agree on one
grammar, and that grammar has to be testable without torch/tensordict.

Grammar::

    {group_id}_g{i}          canonical row of rollout ``i`` (trace 0)
    {group_id}_g{i}_t{j}     extra segment row ``j >= 1`` of rollout ``i``

``group_id`` is free-form (it may itself contain ``_g`` -- the parser binds
the *last* ``_g<digits>`` group, matching the historical ``rsplit("_g", 1)``
behaviour). The canonical id of rollout ``i`` is exactly the recovery
ledger's ``logical_rollout_id`` (``rollout_recovery.py``), so every existing
id, ledger entry, and checkpoint stays valid; only ``_t{j}`` is new.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_SAMPLE_ID_RE = re.compile(r"^(?P<g>.+)_g(?P<i>\d+)(?:_t(?P<j>\d+))?$")


def make_sample_id(group_id: str, i: int, j: int = 0) -> str:
    """Mint the sample id of trace ``j`` of rollout ``i`` in ``group_id``.

    ``j == 0`` yields the canonical ``{group_id}_g{i}`` id; ``j >= 1`` appends
    ``_t{j}``. Indices must be non-negative ints; ``group_id`` must be a
    non-empty string.
    """
    if not isinstance(group_id, str) or not group_id:
        raise ValueError(f"group_id must be a non-empty string, got {group_id!r}")
    if isinstance(i, bool) or not isinstance(i, int) or i < 0:
        raise ValueError(f"rollout index must be a non-negative int, got {i!r}")
    if isinstance(j, bool) or not isinstance(j, int) or j < 0:
        raise ValueError(f"trace index must be a non-negative int, got {j!r}")
    if j == 0:
        return f"{group_id}_g{i}"
    return f"{group_id}_g{i}_t{j}"


def try_parse_sample_id(sample_id: str) -> tuple[str, int, int] | None:
    """Parse ``sample_id`` into ``(group_id, i, j)`` or return None if malformed.

    ``j`` defaults to 0 for canonical ids. Never raises for a string input;
    non-string input is a programming error and raises ``TypeError``.
    """
    if not isinstance(sample_id, str):
        raise TypeError(f"sample_id must be a str, got {type(sample_id).__name__}")
    match = _SAMPLE_ID_RE.match(sample_id)
    if match is None:
        return None
    j_text = match.group("j")
    return match.group("g"), int(match.group("i")), (int(j_text) if j_text else 0)


def parse_sample_id(sample_id: str) -> tuple[str, int, int]:
    """Parse ``sample_id`` into ``(group_id, rollout_local_idx, trace_in_rollout_idx)``.

    Strict: raises ``ValueError`` when the id does not match the grammar.
    """
    parsed = try_parse_sample_id(sample_id)
    if parsed is None:
        raise ValueError(
            f"malformed sample id {sample_id!r}: expected "
            "'{group_id}_g{i}' or '{group_id}_g{i}_t{j}'"
        )
    return parsed


def is_canonical_sample_id(sample_id: str) -> bool:
    """True for a well-formed trace-0 id (``{group}_g{i}`` with no ``_t``)."""
    parsed = try_parse_sample_id(sample_id)
    return parsed is not None and parsed[2] == 0


def group_ids_in_order(sample_ids: Iterable[str]) -> list[str]:
    """Distinct group ids in first-seen order.

    A malformed id (one that does not match the grammar) is treated as its
    own group id, matching the controller's historical lenient behaviour for
    ids minted outside the ``_g{i}`` scheme.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for sample_id in sample_ids:
        parsed = try_parse_sample_id(sample_id)
        group_id = parsed[0] if parsed is not None else sample_id
        if group_id not in seen:
            ordered.append(group_id)
            seen.add(group_id)
    return ordered


def extra_sample_ids_for(
    canonical_sample_ids: Sequence[str], max_per_rollout: int
) -> list[str]:
    """Deterministic extra-row ids a rollout set *may* have published.

    For every canonical id ``c`` this yields ``f"{c}_t{j}"`` for
    ``j in 1..max_per_rollout-1`` (identical to
    ``make_sample_id(group, i, j)`` when ``c`` is canonical). Cleanup paths use
    this to clear rows that were published under the cap even when the
    publisher's own id list is unavailable.
    """
    if max_per_rollout < 1:
        raise ValueError(f"max_per_rollout must be >= 1, got {max_per_rollout}")
    extras: list[str] = []
    for canonical in canonical_sample_ids:
        for j in range(1, max_per_rollout):
            extras.append(f"{canonical}_t{j}")
    return extras


def validate_group_sample_ids(
    sample_ids: Sequence[str], *, expected_group_size: int
) -> None:
    """Check that ``sample_ids`` is one finalized group of ``expected_group_size`` rollouts.

    Accepts the canonical N-row shape and the segment-row shape (N canonical
    rows plus ``_t{j}`` extras). Raises ``ValueError`` when any id is
    malformed, the rollout indices do not cover exactly
    ``range(expected_group_size)``, any ``(i, j)`` pair repeats, an extra row
    has no canonical row, or the ids name more than one group.
    """
    if expected_group_size < 1:
        raise ValueError(
            f"expected_group_size must be >= 1, got {expected_group_size}"
        )
    parsed: list[tuple[str, int, int]] = []
    for sample_id in sample_ids:
        parsed.append(parse_sample_id(sample_id))
    groups = {group for group, _, _ in parsed}
    if len(groups) != 1:
        raise ValueError(
            f"sample ids span {len(groups)} group ids, expected exactly one: "
            f"{sorted(groups)!r}"
        )
    pairs = [(i, j) for _, i, j in parsed]
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"duplicate (rollout, trace) pairs in {list(sample_ids)!r}")
    canonical_indices = {i for i, j in pairs if j == 0}
    if canonical_indices != set(range(expected_group_size)):
        raise ValueError(
            "canonical rollout indices "
            f"{sorted(canonical_indices)!r} do not cover "
            f"range({expected_group_size})"
        )
    for i, j in pairs:
        if j != 0 and i not in canonical_indices:
            raise ValueError(
                f"extra row (rollout {i}, trace {j}) has no canonical row"
            )
