# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy discovery and configuration work without optional Gym imports."""

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nemo_rl.algorithms.single_controller_utils.config import TokenCaptureConfig
from nemo_rl.environments.call_selection import (
    SelectionPolicy,
    cap_selected_rows,
    read_selection_priorities,
    resolve_selection_policy,
    select_all_captured,
    select_captured_calls,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("returned_history", select_captured_calls),
        ("all_captured", select_all_captured),
    ],
)
def test_builtin_selection_policy(name: str, expected: SelectionPolicy) -> None:
    assert resolve_selection_policy(name) is expected


def test_external_selection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "custom_selection.py").write_text(
        "def choose(records, result):\n"
        "    return records[:1], result['terminal_response_id']\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    name = "custom_selection:choose"
    config = TokenCaptureConfig(forest_training=True, selection_policy=name)
    select = resolve_selection_policy(config.selection_policy)
    assert select([], {"terminal_response_id": "served-id"}) == ([], "served-id")
    with pytest.raises(ValueError, match="requires forest_training"):
        TokenCaptureConfig(selection_policy=name)


@pytest.mark.parametrize(
    "name,error",
    [
        ("guess", "Unknown selection_policy"),
        ("", "Unknown selection_policy"),
        (":choose", "Unknown selection_policy"),
        ("json:", "Unknown selection_policy"),
        ("json:loads:extra", "Unknown selection_policy"),
        ("missing_selection_package:choose", "Cannot load selection_policy"),
        ("json:missing_selector", "Cannot load selection_policy"),
        ("json:__name__", "not callable"),
    ],
)
def test_bad_selection_policy_fails_in_config(name: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        TokenCaptureConfig(forest_training=True, selection_policy=name)


@dataclass(frozen=True)
class Call:
    """The CallRecord fields that row capping reads."""

    model_call_id: str
    parent_call_id: str | None
    delta_len: int

    @property
    def response_id(self) -> str:
        return f"resp-{self.model_call_id}"


def forest(edges: dict[str, tuple[str | None, int]], *, salt: str = "") -> list[Any]:
    """Build calls from ``name: (parent, delta_len)``; ``salt`` varies the IDs."""
    return [
        Call(salt + name, None if parent is None else salt + parent, delta)
        for name, (parent, delta) in edges.items()
    ]


def cap(
    calls: list[Any],
    terminal: str,
    max_rows: int,
    priorities: dict[str, int] | None = None,
    *,
    salt: str = "",
) -> list[str]:
    kept = cap_selected_rows(
        calls,
        f"resp-{salt}{terminal}",
        max_rows=max_rows,
        priorities={f"resp-{salt}{k}": v for k, v in (priorities or {}).items()},
    )
    return [call.model_call_id.removeprefix(salt) for call in kept]


def leaves(names: list[str], calls: list[Any]) -> set[str]:
    parents = {c.parent_call_id for c in calls if c.model_call_id in names}
    return {name for name in names if name not in parents}


# Main A -> B -> scored C, compaction root D -> E, and subagents F -> G / H.
SUBAGENTS = {
    "A": (None, 10),
    "B": ("A", 5),
    "C": ("B", 5),
    "D": (None, 8),
    "E": ("D", 4),
    "F": (None, 6),
    "G": ("F", 3),
    "H": ("F", 9),
}


def test_selection_within_budget_is_unchanged() -> None:
    calls = forest(SUBAGENTS)
    assert cap_selected_rows(calls, "resp-C", max_rows=4, priorities={}) is calls


@pytest.mark.parametrize("max_rows", [1, 2, 3])
@pytest.mark.parametrize("terminal", ["C", "E", "H"])
def test_cap_keeps_the_scored_path_within_budget(max_rows: int, terminal: str) -> None:
    for trial in range(50):
        salt = f"t{trial}-"
        calls = forest(SUBAGENTS, salt=salt)
        kept = cap(calls, terminal, max_rows, salt=salt)
        assert len(leaves(kept, forest(SUBAGENTS))) == max_rows
        scored = [terminal]
        while SUBAGENTS[scored[-1]][0] is not None:
            scored.append(SUBAGENTS[scored[-1]][0])
        assert set(scored) <= set(kept)
        # Closed: every kept call's parent is kept and precedes it.
        for index, name in enumerate(kept):
            parent = SUBAGENTS[name][0]
            assert parent is None or parent in kept[:index]


def test_budget_of_one_keeps_only_the_scored_path() -> None:
    assert cap(forest(SUBAGENTS), "C", 1) == ["A", "B", "C"]


def test_scored_inner_call_becomes_a_leaf() -> None:
    # The scored call B has a selected child C; one row ends at B itself.
    edges = {
        "A": (None, 1),
        "B": ("A", 1),
        "C": ("B", 1),
        "D": (None, 1),
        "E": (None, 1),
    }
    assert cap(forest(edges), "B", 1) == ["A", "B"]
    # Drawing C extends the scored row instead of adding one, so a draw follows.
    for trial in range(50):
        salt = f"t{trial}-"
        kept = cap(forest(edges, salt=salt), "B", 2, salt=salt)
        assert {"A", "B"} <= set(kept)
        assert len(leaves(kept, forest(edges))) == 2


def draw_frequencies(
    edges: dict[str, tuple[str | None, int]], terminal: str, trials: int = 4000
) -> Counter:
    counts: Counter = Counter()
    for trial in range(trials):
        salt = f"t{trial}-"
        kept = set(cap(forest(edges, salt=salt), terminal, 2, salt=salt))
        counts.update(name for name in edges if name in kept)
    return counts


def test_draw_is_proportional_to_unkept_tokens() -> None:
    # Scored T and leaves X (1 token) and Y (3 tokens).
    counts = draw_frequencies({"T": (None, 5), "X": (None, 1), "Y": (None, 3)}, "T")
    assert counts["X"] + counts["Y"] == 4000
    assert counts["Y"] / 4000 == pytest.approx(0.75, abs=0.03)


def test_kept_ancestors_do_not_weight_a_draw() -> None:
    # X shares the scored path's 100-token root, so it adds one token; Z adds two.
    counts = draw_frequencies(
        {"A": (None, 100), "T": ("A", 1), "X": ("A", 1), "Y": (None, 1), "Z": ("Y", 1)},
        "T",
    )
    assert counts["Z"] / 4000 == pytest.approx(2 / 3, abs=0.03)


def test_later_draws_reweight_by_the_tokens_still_unkept() -> None:
    # After Y is kept, X adds only its own token, not the shared root's 9.
    second: Counter = Counter()
    for trial in range(4000):
        salt = f"t{trial}-"
        calls = forest(
            {
                "T": (None, 1),
                "R": (None, 9),
                "X": ("R", 1),
                "Y": ("R", 30),
                "Z": (None, 9),
            },
            salt=salt,
        )
        kept = set(cap(calls, "T", 3, {"Y": 0}, salt=salt))
        assert {"R", "Y"} <= kept
        second.update(name for name in ("X", "Z") if name in kept)
    assert second["X"] / 4000 == pytest.approx(0.1, abs=0.02)


def test_higher_priority_classes_fill_the_budget_first() -> None:
    priorities = {"E": 0, "G": 1, "H": 1}
    for trial in range(50):
        salt = f"t{trial}-"
        calls = forest(SUBAGENTS, salt=salt)
        kept = cap(calls, "C", 2, priorities, salt=salt)
        assert leaves(kept, forest(SUBAGENTS)) == {"C", "E"}
        kept = cap(calls, "C", 3, priorities, salt=salt)
        assert leaves(kept, forest(SUBAGENTS)) & {"G", "H"}
        # Listed classes, even the lowest, rank before unlisted leaves.
        kept = cap(calls, "C", 2, {"G": 7}, salt=salt)
        assert leaves(kept, forest(SUBAGENTS)) == {"C", "G"}


def test_cap_is_deterministic_and_order_independent() -> None:
    calls = forest(SUBAGENTS)
    expected = set(cap(calls, "C", 3))
    shuffled = list(calls)
    for seed in range(20):
        random.Random(seed).shuffle(shuffled)
        assert set(cap(shuffled, "C", 3)) == expected
    # Different attempts (call IDs) do not all keep the same rows.
    assert (
        len(
            {
                frozenset(cap(forest(SUBAGENTS, salt=f"{i}-"), "C", 3, salt=f"{i}-"))
                for i in range(50)
            }
        )
        > 1
    )


def test_cap_keeps_selection_order_with_parents_first() -> None:
    # A custom policy may list a child before its root.
    calls = forest({"T": (None, 1), "X": ("R", 1), "R": (None, 1), "Y": (None, 1)})
    kept = cap(calls, "T", 2, {"X": 0})
    assert kept == ["T", "R", "X"]


def test_selection_priorities_are_optional() -> None:
    assert read_selection_priorities({}, forest(SUBAGENTS)) == {}
    assert read_selection_priorities(
        {"selection_priorities": {"resp-E": 0, "resp-H": 2}}, forest(SUBAGENTS)
    ) == {"resp-E": 0, "resp-H": 2}


@pytest.mark.parametrize(
    "priorities,error",
    [
        (["resp-E"], "nonnegative integers"),
        ({"resp-E": -1}, "nonnegative integers"),
        ({"resp-E": True}, "nonnegative integers"),
        ({"resp-E": 1.0}, "nonnegative integers"),
        ({"resp-other": 0}, "outside this capture"),
    ],
)
def test_malformed_selection_priorities_fail(priorities: Any, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        read_selection_priorities(
            {"selection_priorities": priorities}, forest(SUBAGENTS)
        )


def test_row_cap_requires_forest_training() -> None:
    TokenCaptureConfig(forest_training=True, max_rows_per_rollout=1)
    with pytest.raises(ValueError, match="requires forest_training"):
        TokenCaptureConfig(max_rows_per_rollout=1)
