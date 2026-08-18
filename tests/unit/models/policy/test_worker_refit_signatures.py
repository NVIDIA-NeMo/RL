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


def test_the_refit_deadline_is_one_of_those_kwargs():
    """Guards the guard: if the deadline stops being forwarded, the test above goes vacuous."""
    assert "refit_timeout_s" in _forwarded_by_policy("broadcast_weights_for_collective")
