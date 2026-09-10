# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guards against declarations drifting from the call sites that use them.

The other suites in this directory compare one declaration to another, which
catches a constant updated in one place and not the other. These tests close the
remaining direction -- a *call site* naming something no declaration knows about
-- which is the direction that fails silently.

Source is parsed rather than imported: the algorithm modules pull in torch, and
none of this needs it (or a GPU, or nemo-lens).
"""

import ast
import re
from pathlib import Path

from tests.unit.telemetry.conftest import algorithms_utils_categories

_REPO = Path(__file__).resolve().parents[3]
_ALGORITHMS = _REPO / "nemo_rl" / "algorithms"

# Efficiency timers are driver-side only, but spans are not: rl.vllm.* is
# emitted from the generation worker and belongs in the doc tables too.
_SPAN_EMITTING_DIRS = (_ALGORITHMS, _REPO / "nemo_rl" / "models" / "generation")

# Timer methods that take a category label as their first argument.
_TIMER_METHODS = frozenset({"time", "reduce", "record", "start", "stop"})

# Prefixes owned by the efficiency accounting in nemo_rl/algorithms/utils.py.
_EFFICIENCY_PREFIXES = ("idle/", "wasted/")


def _python_sources(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _string_arg(node: ast.Call, index: int = 0) -> str | None:
    """The *index*-th positional argument of *node*, if it is a string literal."""
    if len(node.args) <= index:
        return None
    arg = node.args[index]
    return (
        arg.value
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        else None
    )


def test_every_efficiency_timer_at_a_call_site_is_declared():
    """An undeclared idle/wasted timer is counted as *productive*, silently.

    ``print_efficiency_summary`` derives waste by iterating the declared lists,
    not by reading what the Timer recorded, and productive time is
    ``step_wall_time - waste``. So a category nothing declares does not go
    missing from the report -- it inverts, and efficiency reads higher than
    reality. Nothing warns, and ``bucket_for_efficiency_category`` returns None
    for it, leaving any matching span unbucketed too.
    """
    declared: set[str] = set().union(
        *algorithms_utils_categories(
            "WALL_CLOCK_EFFICIENCY_CATEGORIES",
            "THREAD_ACCUMULATED_EFFICIENCY_CATEGORIES",
        ).values()
    )

    used: dict[str, str] = {}
    for source in _python_sources(_ALGORITHMS):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.Call):
                continue
            called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if called not in _TIMER_METHODS | {"efficiency_span"}:
                continue
            label = _string_arg(node)
            if label and label.startswith(_EFFICIENCY_PREFIXES):
                used[label] = source.relative_to(_REPO).as_posix()

    undeclared = {
        label: where for label, where in used.items() if label not in declared
    }
    assert not undeclared, (
        "efficiency categories measured but not declared in "
        f"nemo_rl/algorithms/utils.py, so their time is charged to productive: "
        f"{undeclared}"
    )
    # Sanity: the walk found something, so a refactor that moves these calls
    # cannot quietly turn this test into a tautology.
    assert used, "found no idle/* or wasted/* timers -- has the matcher gone stale?"


def test_every_emitted_span_name_is_documented():
    """Direction is ``emitted <= documented``.

    The docs may list a span that is not wired yet (``forward_backward``,
    ``optimizer``), but a name that is emitted and undocumented leaves someone
    filtering a Tempo/Jaeger query on a name that does not exist -- zero
    results, nothing to explain it. A typo at an emit site fails the same way,
    and produces a real span under the wrong name, since the goodput rollup keys
    on ``rl.bucket`` rather than the name.
    """
    documented = set(
        _span_names_in(
            (_REPO / "docs" / "observability" / "span-groups.md").read_text()
        )
    )

    emitted: dict[str, str] = {}
    for directory in _SPAN_EMITTING_DIRS:
        for source in _python_sources(directory):
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "attr", None) or getattr(
                    node.func, "id", None
                )
                if called not in ("managed_span", "trace_fn"):
                    continue
                # Span name is the second positional arg, after the span group.
                name = _string_arg(node, index=1)
                if name:
                    emitted[name] = source.relative_to(_REPO).as_posix()

    undocumented = {
        name: where for name, where in emitted.items() if name not in documented
    }
    assert not undocumented, (
        "spans emitted but absent from docs/observability/span-groups.md: "
        f"{undocumented}"
    )
    assert emitted, "found no span names -- has the matcher gone stale?"


def _span_names_in(markdown: str) -> set[str]:
    """Every ``rl.*`` name in backticks, which is how the doc tables list them."""
    return set(re.findall(r"`(rl\.[a-z0-9_.]+)`", markdown))
