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

"""Every policy worker must accept the kwargs ``Policy`` forwards to it.

``Policy`` fans a single call out to every worker over Ray, so a kwarg it sends is sent
to *all* of them. A worker whose signature lacks that kwarg does not fail at import or at
type-check time -- it fails at the Ray boundary, at refit, as
``TypeError: got an unexpected keyword argument``, several frames from the cause.

That is not hypothetical. ``refit_timeout_s`` was added to ``Policy`` and to the Megatron
worker; both DTensor workers were missed, which broke every non-colocated DTensor refit
until a GPU test caught it. Nothing in between could: there is no shared declaration to
diverge from, since ``broadcast_weights_for_collective`` is defined independently on each
worker.

Read from the AST rather than by importing, deliberately: ``dtensor_policy_worker_v2``
imports ``nemo_automodel``, which lives in a per-worker venv and is absent from the base
one, so an import-based check would skip on precisely the worker that regressed.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
POLICY = REPO_ROOT / "nemo_rl" / "models" / "policy"

# (method, files that must accept whatever Policy forwards)
FANOUT_METHODS = [
    (
        "broadcast_weights_for_collective",
        [
            "workers/dtensor_policy_worker.py",
            "workers/dtensor_policy_worker_v2.py",
            "workers/megatron_policy_worker.py",
            "interfaces.py",
        ],
    ),
    (
        "nccl_reshard_refit",
        [
            "workers/base_policy_worker.py",
            "workers/megatron_policy_worker.py",
            "interfaces.py",
        ],
    ),
]

# The same contract on the generation side, and worse there: VllmGeneration picks the
# method NAME from config --
#
#     method_name = "..._async" if cfg["vllm_cfg"]["async_engine"] else "..."
#     getattr(worker, method_name).remote(refit_timeout_s=refit_timeout_s)
#
# -- so the two branches are separate functions in separate files that drift
# independently, and only the configuration decides which one a run reaches. Adding the
# kwarg to the async worker and not the sync one type-checks, imports, and passes every
# async test. It cost two hangs in job 6321283: the generation actor raised TypeError at
# the Ray boundary, never joined the NCCL broadcast, and the training side blocked in
# ray.get forever. Both branches, always, or one config value is a hang.
GEN = REPO_ROOT / "nemo_rl" / "models" / "generation" / "vllm"
GEN_FANOUT = [
    (
        "update_weights_from_collective",
        "vllm_worker.py",
        "update_weights_from_collective",
    ),
    (
        "update_weights_from_collective",
        "vllm_worker_async.py",
        "update_weights_from_collective_async",
    ),
    ("nccl_reshard_refit", "vllm_worker.py", "nccl_reshard_refit"),
    ("nccl_reshard_refit", "vllm_worker_async.py", "nccl_reshard_refit_async"),
]

# One level up again: CollectiveWeightSynchronizer holds a GenerationInterface and calls
# update_weights_from_collective on it, so EVERY backend has to take what it sends --
# not just the one the author happened to test. Missing this cost a red
# L1_Functional_Tests_Dynamo: the Dynamo backend arrived from upstream while this branch
# was adding refit_timeout_s to the call, and nothing connected the two until a GPU job
# failed. SGLang and Megatron cannot reach that synchronizer today (SGLang is rejected
# for non-colocated, Megatron routes to its own), but they implement the same interface
# method and are listed so the contract stays uniform rather than "uniform where it
# currently matters".
GENERATION = REPO_ROOT / "nemo_rl" / "models" / "generation"
GENERATION_BACKENDS = [
    "interfaces.py",
    "vllm/vllm_generation.py",
    "dynamo/dynamo_generation.py",
    "trtllm/trtllm_generation.py",
    "megatron/megatron_generation.py",
    "sglang/sglang_generation.py",
]


def _kwargs_of(path: Path, method: str) -> set[str]:
    """Keyword names accepted by the outermost definition of ``method`` in ``path``."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == method
        ):
            a = node.args
            return {
                arg.arg
                for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)
                if arg.arg != "self"
            }
    raise AssertionError(f"{path.name} does not define {method}()")


def _forwarded_by_policy(method: str) -> set[str]:
    """Keywords ``Policy.<method>`` passes through to the worker group."""
    tree = ast.parse((POLICY / "lm_policy.py").read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (
            isinstance(fn, ast.Attribute)
            and fn.attr.startswith("run_all_workers")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == method
        ):
            continue
        return {kw.arg for kw in node.keywords if kw.arg is not None}
    raise AssertionError(f"Policy never fans {method}() out to the workers")


@pytest.mark.parametrize(
    ("method", "impl"),
    [(m, impl) for m, impls in FANOUT_METHODS for impl in impls],
)
def test_every_worker_accepts_what_policy_forwards(method, impl):
    forwarded = _forwarded_by_policy(method)
    accepted = _kwargs_of(POLICY / impl, method)
    missing = forwarded - accepted
    assert not missing, (
        f"Policy.{method}() forwards {sorted(missing)} to every worker, but "
        f"{impl} does not accept {'it' if len(missing) == 1 else 'them'}. "
        "Ray rejects the call at the actor boundary, so this surfaces as a TypeError "
        "at refit rather than anywhere near this signature."
    )


def _forwarded_by_vllm_generation(method: str) -> set[str]:
    """Keywords ``VllmGeneration.<method>`` passes to whichever worker method it picked.

    The call is ``getattr(worker, method_name).remote(...)``, so the callee is not named
    at the call site -- it is picked from config. Anchor on the enclosing def instead.
    """
    tree = ast.parse((GEN / "vllm_generation.py").read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method
        ):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "remote"
            ):
                return {kw.arg for kw in call.keywords if kw.arg is not None}
    raise AssertionError(f"VllmGeneration.{method}() has no .remote() fan-out")


@pytest.mark.parametrize(
    ("method", "worker_file", "worker_method"),
    [(m, f, wm) for m, f, wm in GEN_FANOUT],
    ids=[f"{m}->{f}::{wm}" for m, f, wm in GEN_FANOUT],
)
def test_both_engine_branches_accept_what_generation_forwards(
    method, worker_file, worker_method
):
    forwarded = _forwarded_by_vllm_generation(method)
    accepted = _kwargs_of(GEN / worker_file, worker_method)
    missing = forwarded - accepted
    assert not missing, (
        f"VllmGeneration.{method}() forwards {sorted(missing)}, but "
        f"{worker_file}::{worker_method}() does not accept {'it' if len(missing) == 1 else 'them'}. "
        "Which branch a run takes is decided by vllm_cfg.async_engine, so this is a "
        "config-dependent hang: the generation actor raises TypeError, never joins the "
        "collective, and the training side blocks in ray.get with no error anywhere."
    )


def _forwarded_by_collective_synchronizer() -> set[str]:
    """Keywords the synchronizer sends to ``generation.update_weights_from_collective``."""
    path = REPO_ROOT / "nemo_rl" / "weight_sync" / "collective_weight_synchronizer.py"
    tree = ast.parse(path.read_text())
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "update_weights_from_collective"
        ):
            return {kw.arg for kw in call.keywords if kw.arg is not None}
    raise AssertionError(
        "CollectiveWeightSynchronizer no longer calls update_weights_from_collective"
    )


@pytest.mark.parametrize("backend", GENERATION_BACKENDS)
def test_every_generation_backend_accepts_what_the_synchronizer_sends(backend):
    forwarded = _forwarded_by_collective_synchronizer()
    accepted = _kwargs_of(GENERATION / backend, "update_weights_from_collective")
    missing = forwarded - accepted
    assert not missing, (
        f"CollectiveWeightSynchronizer sends {sorted(missing)} to "
        f"generation.update_weights_from_collective(), but {backend} does not accept "
        f"{'it' if len(missing) == 1 else 'them'}. The synchronizer holds a "
        "GenerationInterface, so whichever backend is configured receives this call; "
        "a backend missing the kwarg fails at the Ray boundary during the first refit."
    )


def test_the_refit_deadline_is_one_of_those_kwargs():
    """Guards the guard: if the deadline stops being forwarded, the test above goes vacuous."""
    assert "refit_timeout_s" in _forwarded_by_policy("broadcast_weights_for_collective")


# Refit entrypoints on the Ray-actor side of a vLLM collective_rpc. Everything these call
# runs in the EngineCore process, whose RPC preserves the exception message and discards
# the type -- so a RefitAborted raised inside the engine arrives here as a plain Exception.
_VLLM_RPC_REFIT_ENTRYPOINTS = [
    ("vllm/vllm_worker.py", "update_weights_from_collective"),
    ("vllm/vllm_worker.py", "nccl_reshard_refit"),
    ("vllm/vllm_worker_async.py", "update_weights_from_collective_async"),
    ("vllm/vllm_worker_async.py", "nccl_reshard_refit_async"),
]


@pytest.mark.parametrize(("module", "method"), _VLLM_RPC_REFIT_ENTRYPOINTS)
def test_the_abort_is_detected_by_message_not_by_type(module, method):
    """A bare `except RefitAborted` here is dead code, and it silently wedges the run.

    vLLM's EngineCore stringifies the worker exception into failure_message and re-raises
    it client-side as Exception(...), so the type never crosses. All four of these once
    had `except RefitAborted: raise` and none of them ever fired. Job 6484412: the
    deadline fired, the abort was named in the log, the handler did not match, and the run
    sat at step 4/24 until the harness killed it.
    """
    tree = ast.parse((GENERATION / module).read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == method
        ):
            called = {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            assert "is_refit_abort" in called, (
                f"{module}::{method} does not call is_refit_abort(). Its broad "
                "`except Exception` will fold a deliberate abort into `return False`, so "
                "the controller never rebuilds and the run wedges instead of recovering."
            )
            return
    raise AssertionError(f"{module}::{method} not found")


def test_the_reshard_refit_does_not_occupy_the_actors_event_loop():
    """A sync method here starves every other call to the same actor.

    Ray runs a sync actor method directly in the event loop (sync_to_async wraps it as
    `async def wrapper: return func(...)`, no executor), so a refit blocked in NCCL leaves
    nothing able to service the recovery's init_collective. max_concurrency cannot fix
    that -- it interleaves coroutines, and a coroutine blocked in C never yields.

    Job 6509685: the controller gave up on the stuck refit and called init_collective to
    rebuild, that call queued behind the refit still holding the loop, rank 0 never created
    the rendezvous store, and the surviving generation worker timed out dialling it for
    300s twice before the run ended.
    """
    worker = REPO_ROOT / "nemo_rl" / "models" / "policy" / "workers"
    tree = ast.parse((worker / "megatron_policy_worker.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == "nccl_reshard_refit"
        ):
            assert isinstance(node, ast.AsyncFunctionDef), (
                "nccl_reshard_refit must be async and hand the blocking transfer to a "
                "thread; as a sync method it holds the actor's event loop for the whole "
                "refit and the rebuild can never be serviced."
            )
            called = {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            assert "await_off_loop" in called, (
                "async alone is not enough -- the blocking body must leave the loop."
            )
            return
    raise AssertionError("megatron_policy_worker.nccl_reshard_refit not found")


def test_the_reshard_group_does_not_double_count_the_shard_offset():
    """Both rank computations in vllm_backend must go through the same helper.

    Under vLLM's external data parallelism each engine's torch world spans the whole
    rollout, so get_rank() is ALREADY global and adding rank_prefix counts the shard offset
    twice -- the higher shards then get NCCL ranks past the end of the group. init_collective
    resolves this through resolve_rollout_rank; init_nccl_reshard_comm_group open-coded
    `rank_prefix + get_rank()` and did not. Without external DP the two agree, which is why
    it survived.
    """
    tree = ast.parse((GENERATION / "vllm" / "vllm_backend.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "init_nccl_reshard_comm_group"
        ):
            called = {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            assert "resolve_rollout_rank" in called, (
                "init_nccl_reshard_comm_group must resolve its rank through "
                "resolve_rollout_rank, like init_collective; adding rank_prefix to a "
                "global get_rank() puts the higher shards past the end of the group."
            )
            return
    raise AssertionError("init_nccl_reshard_comm_group not found")


def test_the_reshard_preconditions_are_checked_on_the_single_controller_path():
    """The guard had one caller, in grpo.setup, which SC does not go through.

    run_grpo_single_controller calls setup_single_controller directly, so every
    nccl_reshard precondition -- colocated.enabled, enable_eplb and the rest -- went
    unenforced there and a bad config reached the first refit before anything noticed.
    """
    setup = (
        REPO_ROOT / "nemo_rl" / "algorithms" / "single_controller_utils" / "setup.py"
    )
    tree = ast.parse(setup.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "setup_single_controller"
        ):
            called = {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            assert "check_nccl_reshard_refit_support" in called, (
                "setup_single_controller must run the nccl_reshard precondition guard; "
                "its only other caller is grpo.setup, which this path never reaches."
            )
            return
    raise AssertionError("setup_single_controller not found")


def _sent_by_reshard_synchronizer(method: str, *, receiver: str) -> set[str]:
    """Keywords NcclReshardWeightSynchronizer sends to ONE side of the refit.

    Scoped by receiver on purpose. Both sides are called `nccl_reshard_refit`, and they do
    not take the same arguments: kv_scales is read off the trainer and rides the misc
    broadcast, so the generation side never sees it. Matching on the method name alone
    compares the two sides against each other and reports a difference that is correct.
    """
    path = REPO_ROOT / "nemo_rl" / "weight_sync" / "nccl_reshard_weight_synchronizer.py"
    tree = ast.parse(path.read_text())
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == method
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == receiver
        ):
            return {kw.arg for kw in call.keywords if kw.arg is not None}
    raise AssertionError(
        f"the reshard synchronizer no longer calls {receiver}.{method}"
    )


def test_the_generation_reshard_hook_accepts_what_the_synchronizer_sends():
    """The second of the two generation refit hooks, and the one that was missed.

    The rule was already written down on `update_weights_from_collective`: the synchronizer
    calls these polymorphically, so a signature that omits the parameter does not fail at
    import or type-check time -- it fails at the Ray boundary during the first refit. The
    rule was stated, applied to the policy interface and to one generation hook, and not to
    this one.
    """
    sent = _sent_by_reshard_synchronizer("nccl_reshard_refit", receiver="_generation")
    accepted = _kwargs_of(
        REPO_ROOT / "nemo_rl" / "models" / "generation" / "interfaces.py",
        "nccl_reshard_refit",
    )
    missing = sent - accepted
    assert not missing, (
        f"NcclReshardWeightSynchronizer sends {sorted(missing)} to "
        "generation.nccl_reshard_refit(), but GenerationInterface does not accept "
        f"{'it' if len(missing) == 1 else 'them'}."
    )


@pytest.mark.parametrize(
    "synchronizer",
    ["collective_weight_synchronizer.py", "nccl_reshard_weight_synchronizer.py"],
)
def test_the_rebuild_bootstraps_with_the_same_peer_protocol_as_the_first_build(
    synchronizer,
):
    """A rebuilt communicator must not silently fall back to the "nemo" default.

    The receiver's bootstrap is not negotiable: "nemo" publishes a raw unique ID and warms
    up with a rank-0 broadcast, "vllm" adds a pickled ID key and warms up with an
    all-reduce. Mismatched warmups on one communicator HANG rather than error -- the exact
    failure the rebuild exists to remove, reappearing inside the recovery.
    """
    path = REPO_ROOT / "nemo_rl" / "weight_sync" / synchronizer
    tree = ast.parse(path.read_text())
    calls = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "init_collective"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "_policy"
    ]
    assert calls, "no policy.init_collective call found"
    for call in calls:
        assert "nccl_peer" in {kw.arg for kw in call.keywords}, (
            "every policy.init_collective must pass nccl_peer; the rebuild omitted it and "
            "silently bootstrapped with the wrong protocol."
        )


@pytest.mark.parametrize(
    "synchronizer",
    ["collective_weight_synchronizer.py", "nccl_reshard_weight_synchronizer.py"],
)
def test_a_failed_refit_waits_for_stragglers_before_it_propagates(synchronizer):
    """The caller rebuilds on this failure, and a rebuild needs every rank out first.

    ray.get raises on the first future that fails and leaves the rest running. Job 6512153:
    the rebuild began two log lines before the second trainer's watchdog had even fired,
    and the rendezvous it built was one nobody could join.
    """
    path = REPO_ROOT / "nemo_rl" / "weight_sync" / synchronizer
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "sync_weights"
        ):
            called = {
                c.func.id
                for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
            }
            assert "_settle_before_propagating" in called, (
                f"{synchronizer}::sync_weights must let every train rank unwind before "
                "the failure reaches the caller; otherwise the rebuild races them."
            )
            return
    raise AssertionError(f"{synchronizer}::sync_weights not found")


@pytest.mark.parametrize(
    "synchronizer",
    ["nccl_reshard_weight_synchronizer.py", "collective_weight_synchronizer.py"],
)
def test_both_sides_unwind_before_a_refit_failure_propagates(synchronizer):
    """Settling only the train half leaves the generation ranks racing the rebuild.

    The rebuild dispatches init_collective to BOTH sides, so both have to be out of the
    old refit first. ray.get(futures_train) raising leaves futures_inference running, and
    the original fix settled only the half that happened to be named in the traceback.
    """
    path = REPO_ROOT / "nemo_rl" / "weight_sync" / synchronizer
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "sync_weights"
        ):
            settled = {
                c.args[0].id
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Name)
                and c.func.id == "_settle_before_propagating"
                and c.args
                and isinstance(c.args[0], ast.Name)
            }
            assert settled == {"futures_train", "futures_inference"}, (
                f"{synchronizer}::sync_weights settles {sorted(settled)}; both sides must "
                "unwind before the failure propagates, or the rebuild races the half that "
                "was not settled."
            )
            return
    raise AssertionError(f"{synchronizer}::sync_weights not found")


@pytest.mark.parametrize(
    "worker",
    [
        ("models/policy/workers/base_policy_worker.py", "init_collective"),
        ("models/generation/vllm/vllm_backend.py", "init_collective"),
        # The bulk groups are step 2 of the same rebuild and had the same bare abort().
        # Listed because this stack has now shipped the same fix to one of a pair four
        # separate times -- see design_vllm_fault_tolerance.md section 8.5.5.
        (
            "models/policy/workers/base_policy_worker.py",
            "init_nccl_reshard_comm_group",
        ),
        (
            "models/generation/vllm/vllm_backend.py",
            "init_nccl_reshard_comm_group",
        ),
    ],
    ids=["train-collective", "gen-collective", "train-bulk", "gen-bulk"],
)
def test_the_rendezvous_is_built_before_the_old_group_is_released(worker):
    """Job 6518381: releasing first cost the run, because the release never returned.

    rank 0 is the rendezvous store's master, so every other rank is already counting down
    a 300s connect timeout against the port it has not bound yet. On the reshard path the
    release has to abort the split children, and ncclCommAbort joins a proxy thread that a
    SIGSTOPped peer never lets return -- so the store was never created and the survivor
    spent 600s failing to reach it.
    """
    module, func_name = worker
    tree = ast.parse((REPO_ROOT / "nemo_rl" / module).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            body = list(ast.walk(node))
            builds = [
                n.lineno
                for n in body
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "StatelessProcessGroup"
            ]
            releases = [
                n.lineno
                for n in body
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "release_within"
            ]
            assert builds, f"{module}::{func_name} builds no StatelessProcessGroup"
            assert releases, (
                f"{module}::{func_name} must release the previous group through "
                "release_within; a bare abort() can block forever on a frozen peer."
            )
            assert min(builds) < min(releases), (
                f"{module}::{func_name} releases the old group at line {min(releases)} "
                f"before binding the new one at line {min(builds)}; the rendezvous must "
                "come up first, because the release may never return."
            )
            return
    raise AssertionError(f"{module}::{func_name} not found")


def test_the_release_is_abandoned_rather_than_waited_on():
    """A daemon thread, and never joined -- otherwise the wedge moves to shutdown.

    asyncio.to_thread and a bare ThreadPoolExecutor both use non-daemon threads that the
    interpreter joins at exit, which is the same trap await_off_loop had to avoid.
    """
    path = REPO_ROOT / "nemo_rl" / "distributed" / "refit_watchdog.py"
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "release_within":
            threads = [
                c
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "Thread"
            ]
            assert threads, (
                "release_within must run the release off the caller's thread"
            )
            daemon = {
                k.value.value for t in threads for k in t.keywords if k.arg == "daemon"
            }
            assert daemon == {True}, (
                "release_within's thread must be a daemon; a non-daemon thread is joined "
                "at interpreter exit and moves the hang to shutdown."
            )
            joins = [
                c
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and c.func.attr == "join"
            ]
            assert not joins, (
                "release_within must not join the release thread; the bounded wait is the "
                "whole point, and a join reintroduces the unbounded one."
            )
            return
    raise AssertionError("release_within not found")
