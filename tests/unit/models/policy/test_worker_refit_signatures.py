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
