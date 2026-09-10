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
# emitted from the generation worker, rl.policy.* / rl.value.* from the training
# workers, rl.setup.ray_init from the cluster bootstrap, and rl.startup from the
# launchers -- all belong in the doc tables.
_SPAN_EMITTING_DIRS = (
    _ALGORITHMS,
    _REPO / "nemo_rl" / "models" / "generation",
    _REPO / "nemo_rl" / "models" / "policy",
    _REPO / "nemo_rl" / "models" / "value",
    _REPO / "nemo_rl" / "distributed",
    _REPO / "nemo_rl" / "data_plane",
    _REPO / "nemo_rl" / "environments",
    _REPO / "examples",
)

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


# Span helpers whose first argument is a span group and whose second is the
# span name. The umbrella pair emits the same span as the leaf pair, minus the
# rl.bucket, so every guard here has to look at all four or it silently stops
# covering whichever spans were most recently moved between them.
_GROUP_TAKING_HELPERS = frozenset(
    {"managed_span", "trace_fn", "umbrella_span", "umbrella_trace_fn"}
)

# Which helper goes with which kind of group.
_UMBRELLA_HELPERS = frozenset({"umbrella_span", "umbrella_trace_fn"})
_LEAF_HELPERS = frozenset({"managed_span", "trace_fn"})


# Driver-side spans opened once per concurrently-dispatched unit of rollout
# work, so many are open at the same time. Each must sit in an umbrella group:
# tagged with a bucket they would sum to a multiple of the wall clock they
# happened in. Kept as an explicit list because the property that makes them
# unsummable -- concurrency at the dispatch site -- is not visible in the span
# name, so nothing else would catch a group swapped back.
_CONCURRENT_ROLLOUT_SPANS = {
    # One asyncio task per prompt group, bounded by max_inflight_prompts.
    "rl.sc.generate_and_push": _ALGORITHMS / "single_controller.py",
    # One batch-worker thread per rollout batch.
    "rl.grpo.generation": _ALGORITHMS / "grpo.py",
}


def _span_groups_by_name(source: Path) -> dict[str, set[str]]:
    """``{span name: {group attribute names it is opened with}}`` for one file."""
    groups: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if not isinstance(node, ast.Call):
            continue
        called = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if called not in _GROUP_TAKING_HELPERS:
            continue
        name = _string_arg(node, index=1)
        if name is None or not node.args:
            continue
        group = node.args[0]
        if isinstance(group, ast.Attribute):
            groups.setdefault(name, set()).add(group.attr)
    return groups


def test_concurrently_dispatched_rollout_spans_stay_unbucketed():
    """A bucket on these overcounts productive time by the concurrency factor.

    ``rl.sc.generate_and_push`` shipped in ``GENERATION`` (bucket
    ``productive``) while up to ``max_inflight_prompts`` of them were open at
    once, so a rollup summing by ``rl.bucket`` could report a large multiple of
    the run's wall clock as productive. Productive generation is attributed by
    the worker-side ``rl.vllm.generate`` spans instead.
    """
    from nemo_rl.telemetry.instrumentation import UMBRELLA_GROUPS
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    umbrella_attrs = {
        attr
        for attr in dir(RLSpanGroup)
        if not attr.startswith("_") and getattr(RLSpanGroup, attr) in UMBRELLA_GROUPS
    }

    for name, source in _CONCURRENT_ROLLOUT_SPANS.items():
        found = _span_groups_by_name(source).get(name)
        assert found, f"{name} is no longer emitted from {source.name}"
        bucketed = found - umbrella_attrs
        assert not bucketed, (
            f"{name} is opened with non-umbrella group(s) {sorted(bucketed)}; "
            "these spans overlap, so a bucket on them sums past wall time"
        )


def test_umbrella_spans_are_spelled_as_umbrellas_at_the_call_site():
    """Whether a span carries ``rl.bucket`` should be readable where it is opened.

    ``managed_span(RLSpanGroup.ROLLOUT, ...)`` behaves correctly -- the group is
    an umbrella either way -- so nothing downstream can tell it apart from the
    intended spelling. The point of the convention is the neighbouring case it
    makes visible: ``GENERATION`` and ``U_ROLLOUT`` are both plausible for a
    generation span, and only one of them enters a goodput rollup. That choice
    is invisible when every group is spelled the same way, which is how
    ``rl.sc.generate_and_push`` shipped as ``productive`` while up to
    ``max_inflight_prompts`` of them ran at once.

    Enforced here rather than by raising, because the misspelling produces
    correct telemetry: a runtime error would take down a training run over
    something that only matters to a reader.
    """
    from nemo_rl.telemetry.instrumentation import UMBRELLA_GROUPS
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    wrong_helper: dict[str, str] = {}
    wrong_alias: dict[str, str] = {}
    seen = 0
    for directory in _SPAN_EMITTING_DIRS:
        for source in _python_sources(directory):
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                called = getattr(node.func, "attr", None) or getattr(
                    node.func, "id", None
                )
                group = node.args[0]
                if called not in _GROUP_TAKING_HELPERS or not isinstance(
                    group, ast.Attribute
                ):
                    continue
                value = getattr(RLSpanGroup, group.attr, None)
                if value is None:
                    continue
                seen += 1
                where = f"{source.relative_to(_REPO).as_posix()}:{node.lineno}"
                site = f"{called}(...{group.attr})"
                if value in UMBRELLA_GROUPS:
                    if called in _LEAF_HELPERS:
                        wrong_helper[where] = f"{site} -> use umbrella_*"
                    elif not group.attr.startswith("U_"):
                        wrong_alias[where] = f"{site} -> use U_{group.attr}"
                elif called in _UMBRELLA_HELPERS:
                    # Only warns at runtime, so this is the gate that stops it.
                    wrong_helper[where] = f"{site} is a leaf -> use managed_span"

    assert not wrong_helper, (
        f"span helper does not match the group kind: {wrong_helper}"
    )
    assert not wrong_alias, (
        f"umbrella group spelled without its U_ alias: {wrong_alias}"
    )
    assert seen, "found no span call sites -- has the matcher gone stale?"


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


def _teed_logger_keys() -> dict[str, int]:
    """``{logger key: line}`` for every ``_TeedMetric`` declared in metrics.py.

    Parsed rather than imported for the reason the module header gives, and
    because the key is the first positional argument either way -- importing
    would buy nothing but a dependency on lens being installed.
    """
    source = _REPO / "nemo_rl" / "telemetry" / "metrics.py"
    keys: dict[str, int] = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) != (
            "_TeedMetric"
        ):
            continue
        key = _string_arg(node)
        if key:
            keys[key] = node.lineno
    return keys


def _metric_dict_keys(root: Path) -> dict[str, str]:
    """String literals used as metrics-dict keys under *root*.

    Two shapes, which is how every teed scalar is actually produced: a literal
    key in a dict display, and a subscript assignment onto an accumulating dict.
    Deliberately not "every string constant in the file" -- that would also
    match a key named only in a docstring, and a rename that updated the prose
    but not the code would still pass.
    """
    found: dict[str, str] = {}
    for source in _python_sources(root):
        where = source.relative_to(_REPO).as_posix()
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        found.setdefault(key.value, where)
            elif isinstance(node, ast.Assign | ast.AugAssign):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        found.setdefault(target.slice.value, where)
    return found


def test_every_teed_logger_key_is_emitted():
    """Direction is ``declared <= emitted``.

    ``_TeedMetric`` is both the declaration of an OTel series and the mapping
    from the logger key that feeds it, which is the whole point of keeping them
    in one row: an earlier design kept a separate key-to-field map, three of its
    entries pointed at keys nothing emitted, and those gauges reported a flat
    line rather than an error. Nothing at runtime can catch that -- the tee
    reads the key out of the metrics dict with ``.get`` and a miss is
    indistinguishable from a step that did not report the value -- so a rename
    on one side only has to fail here or not at all.
    """
    declared = _teed_logger_keys()
    emitted: dict[str, str] = {}
    for directory in (
        _ALGORITHMS,
        _REPO / "nemo_rl" / "models",
        _REPO / "nemo_rl" / "experience",
    ):
        emitted.update(_metric_dict_keys(directory))

    unemitted = {
        key: f"metrics.py:{line}"
        for key, line in declared.items()
        if key not in emitted
    }
    assert not unemitted, (
        "logger keys declared in _TEED_SCALARS that no call site emits, so each "
        f"is a gauge that will report nothing: {unemitted}"
    )
    # Sanity on both halves, so neither a moved declaration nor a stale matcher
    # can turn this into a tautology.
    assert declared, "found no _TeedMetric declarations -- has the matcher gone stale?"
    assert emitted, "found no metrics-dict keys -- has the matcher gone stale?"


def _emitted_span_name(called: str, node: ast.Call) -> str | None:
    """The span name a call emits, or None if the call does not emit one.

    Three of the helpers build the name rather than taking it, so matching only
    on a literal argument would miss them entirely -- which is how ``rl.idle.*``
    stayed outside this guard until ``rl.setup.*`` arrived the same way.
    """
    if called in _GROUP_TAKING_HELPERS:
        # Span name is the second positional arg, after the span group.
        return _string_arg(node, index=1)
    if called == "startup_span":
        return "rl.startup"
    if called == "setup_span":
        phase = _string_arg(node)
        return f"rl.setup.{phase}" if phase else None
    if called == "efficiency_span":
        category = _string_arg(node)
        return f"rl.{category.replace('/', '.')}" if category else None
    if called == "traced_worker_init":
        # Takes the name directly: the group is fixed at U_MODEL_INIT.
        return _string_arg(node)
    return None


def test_every_emitted_span_name_is_documented():
    """Direction is ``emitted <= documented``.

    The docs may describe a span in more places than one, but a name that is
    emitted and undocumented leaves someone filtering a Tempo/Jaeger query on a
    name that does not exist -- zero results, nothing to explain it. A typo at
    an emit site fails the same way, and produces a real span under the wrong
    name, since the goodput rollup keys on ``rl.bucket`` rather than the name.
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
                name = _emitted_span_name(called, node) if called else None
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


def test_every_registered_group_has_an_emitter():
    """Direction is ``registered <= emitted`` -- the reverse of the docs guard.

    A group in ``ALL_GROUPS`` with no call site is worse than a missing one: it
    is listed in the preset table and accepted by ``span_groups``, so a user who
    selects it is told they asked for something and gets silence, with no way to
    tell that apart from a phase that simply did not run. Four groups sat in
    exactly that state -- ``reference_policy`` (in ``per_step``, so it was
    advertised) plus ``load_checkpoint``, ``forward_backward`` and
    ``optimizer``. Register the group in the same change that emits it.
    """
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    # Helpers that fix their span group instead of taking it as an argument.
    fixed_group_helpers = {
        "startup_span": RLSpanGroup.SETUP,
        "setup_span": RLSpanGroup.SETUP,
        "efficiency_span": RLSpanGroup.EFFICIENCY,
        "traced_worker_init": RLSpanGroup.MODEL_INIT,
    }

    emitted: set[str] = set()
    for directory in _SPAN_EMITTING_DIRS:
        for source in _python_sources(directory):
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "attr", None) or getattr(
                    node.func, "id", None
                )
                if called in fixed_group_helpers:
                    emitted.add(fixed_group_helpers[called])
                elif called in _GROUP_TAKING_HELPERS and node.args:
                    group = node.args[0]
                    if isinstance(group, ast.Attribute):
                        emitted.add(getattr(RLSpanGroup, group.attr, group.attr))

    assert emitted, "found no span groups -- has the matcher gone stale?"
    unemitted = RLSpanGroup.ALL_GROUPS - emitted
    assert not unemitted, (
        "registered as span groups but no call site emits them, so selecting "
        f"them yields nothing: {sorted(unemitted)}"
    )


def test_every_context_accepting_method_is_dispatched_with_a_carrier():
    """Both halves of the propagation pair have to be wired, and only one is visible.

    ``@accepts_trace_context`` on a worker method is inert on its own: the
    caller has to send the carrier too. A decorated method whose dispatch site
    forgets it looks fully wired at the definition -- which is where anyone
    checks -- and still emits root spans, exactly the bug this mechanism exists
    to fix. Three of the six presharded entrypoints were in that state when the
    guard was written.

    Both dispatch spellings count as sending: ``dispatch_with_trace_context``
    for a direct ``remote()`` call, and ``trace_context_kwargs()`` spread into
    a ``RayWorkerGroup.run_all_workers_*`` call, which names its method by
    string. A bare ``.remote()`` on a decorated method counts as *not* sending.
    """
    decorated = _context_accepting_methods()
    assert decorated, "found no decorated methods -- has the matcher gone stale?"
    indirect = _indirect_dispatchers()

    dispatched: set[str] = set()
    unwired: dict[str, str] = {}
    for source in _python_sources(_REPO / "nemo_rl"):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.Call):
                continue
            targets = _dispatch_targets(node, decorated, indirect)
            if not targets:
                continue
            where = f"{source.relative_to(_REPO).as_posix()}:{node.lineno}"
            for target, sends_carrier in targets:
                if sends_carrier:
                    dispatched.add(target)
                else:
                    unwired[target] = where

    assert not unwired, (
        "methods decorated with @accepts_trace_context whose dispatch site does "
        f"not send one, so their spans still re-root: {unwired}"
    )
    assert decorated <= dispatched, (
        "methods decorated with @accepts_trace_context that no dispatch site "
        f"targets, so the decorator is dead weight: {sorted(decorated - dispatched)}"
    )


def test_every_presharded_entrypoint_is_wired_for_context():
    """The guard above cannot see an entrypoint that is wired on *neither* side.

    It starts from the set of decorated methods, so a presharded entrypoint that
    is neither decorated nor dispatched with a carrier is not a finding -- it is
    invisible. That is the state the three split-API lifecycle calls
    (``begin`` / ``finish`` / ``abort``) shipped in while the microbatch call
    between them was wired, which split one logical train step across two
    traces: mcore's ``megatron.microbatch.*`` spans nested under the driver's
    step, while the ``megatron.grad_sync.*`` spans that ``finish`` reaches
    through ``finalize_model_grads`` re-rooted into a detached trace per rank
    per step.

    The presharded family is the right unit to require this of because every
    member is a worker entrypoint reached only by driver dispatch, so there is
    no case where one legitimately wants its own trace.
    """
    presharded = _presharded_entrypoints()
    assert presharded, "found no presharded entrypoints -- has the matcher gone stale?"
    indirect = _indirect_dispatchers()
    decorated = _context_accepting_methods()

    dispatched: dict[str, str] = {}
    uncarried: dict[str, str] = {}
    for source in _python_sources(_REPO / "nemo_rl"):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.Call):
                continue
            targets = _dispatch_targets(node, presharded, indirect)
            where = f"{source.relative_to(_REPO).as_posix()}:{node.lineno}"
            for target, sends_carrier in targets:
                dispatched[target] = where
                if not sends_carrier:
                    uncarried[target] = where

    assert not uncarried, (
        "presharded entrypoints dispatched without a trace carrier, so their "
        f"worker spans start a new trace: {uncarried}"
    )
    undecorated = {
        name: where for name, where in dispatched.items() if name not in decorated
    }
    assert not undecorated, (
        "presharded entrypoints missing @accepts_trace_context, so the carrier "
        f"their dispatch site sends is discarded: {undecorated}"
    )
    assert dispatched, "found no presharded dispatches -- has the matcher gone stale?"


def _presharded_entrypoints() -> set[str]:
    """Every ``*_presharded`` worker entrypoint defined under ``nemo_rl/``."""
    names: set[str] = set()
    for source in _python_sources(_REPO / "nemo_rl"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)
            ) and node.name.endswith("_presharded"):
                names.add(node.name)
    return names


def _context_accepting_methods() -> set[str]:
    """Every method under ``nemo_rl/`` carrying ``@accepts_trace_context``."""
    decorated: set[str] = set()
    for source in _python_sources(_REPO / "nemo_rl"):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {
                getattr(decorator, "id", None) or getattr(decorator, "attr", None)
                for decorator in node.decorator_list
            }
            if "accepts_trace_context" in names:
                decorated.add(node.name)
    return decorated


def _indirect_dispatchers() -> dict[str, bool]:
    """Helpers that dispatch a method named by one of their own arguments.

    ``TQPolicy._logprob_dispatch`` is the case: it takes ``worker_method`` and
    forwards it, so the ``run_all_workers_*`` call itself names nothing a static
    read can attribute. Recording whether the helper sends a carrier lets the
    guard resolve one level rather than exempting the methods routed through it
    -- an exemption would hide the helper dropping the carrier later, which is
    the failure this whole test is about.

    Maps helper name -> whether its dispatch sends a carrier.
    """
    dispatchers: dict[str, bool] = {}
    for source in _python_sources(_REPO / "nemo_rl"):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for inner in ast.walk(node):
                if not isinstance(inner, ast.Call):
                    continue
                called = getattr(inner.func, "attr", None)
                if not called or not called.startswith("run_all_workers_"):
                    continue
                if _string_arg(inner) or _keyword_string(inner, "method_name"):
                    continue
                dispatchers[node.name] = _sends_carrier(inner)
    return dispatchers


def _dispatch_targets(
    node: ast.Call, decorated: set[str], indirect: dict[str, bool]
) -> list[tuple[str, bool]]:
    """The decorated methods *node* dispatches, each with whether it carries.

    Empty when *node* is not a dispatch at all.
    """
    called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    if called == "dispatch_with_trace_context":
        target = _decorated_attribute(node.args[0], decorated) if node.args else None
        return [(target, True)] if target else []
    if called == "remote":
        target = _decorated_attribute(node.func, decorated)
        return [(target, False)] if target else []
    if called and called.startswith("run_all_workers_"):
        name = _string_arg(node) or _keyword_string(node, "method_name")
        return [(name, _sends_carrier(node))] if name in decorated else []
    if called in indirect:
        # A call into a helper that forwards the method name it is handed.
        named = {
            keyword.value.value
            for keyword in node.keywords
            if isinstance(keyword.value, ast.Constant)
            and keyword.value.value in decorated
        }
        return [(name, indirect[called]) for name in sorted(named)]
    return []


def _decorated_attribute(node: ast.expr, decorated: set[str]) -> str | None:
    """The decorated method named anywhere in an attribute chain.

    Both ``actor.run_rollouts`` and ``actor.run_rollouts.options(...)`` resolve
    to ``run_rollouts``, so the guard does not care whether the call site
    reshapes the handle before dispatching it.
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.Attribute) and inner.attr in decorated:
            return inner.attr
    return None


def _sends_carrier(node: ast.Call) -> bool:
    """Whether *node* hands the callee a trace carrier."""
    called = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
    if called == "dispatch_with_trace_context":
        return True
    return any(
        getattr(inner.func, "id", None) == "trace_context_kwargs"
        for inner in ast.walk(node)
        if isinstance(inner, ast.Call)
    )


def _keyword_string(node: ast.Call, name: str) -> str | None:
    """The string literal passed as keyword *name*, if any."""
    for keyword in node.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    return None


def _span_names_in(markdown: str) -> set[str]:
    """Every ``rl.*`` name in backticks, which is how the doc tables list them."""
    return set(re.findall(r"`(rl\.[a-z0-9_.]+)`", markdown))
