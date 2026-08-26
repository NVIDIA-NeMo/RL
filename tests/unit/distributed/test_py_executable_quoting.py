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

"""PY_EXECUTABLES must survive a repo path containing a space.

``create_local_venv`` re-parses these command strings with ``shlex.split``
(`nemo_rl/utils/venvs.py`), so a space in the interpolated repo root becomes an
argument boundary and ``uv run --directory`` receives only the first half.
``/mnt/c/Users/First Last/...`` and ``/Users/First Last/...`` are the default
shapes on WSL and macOS, so this is ordinary rather than exotic.
"""

import contextlib
import importlib
import os
import shlex
from unittest.mock import patch

import pytest

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES, git_root

_UV_EXECUTABLES = [
    "BASE",
    "VLLM",
    "FSDP",
    "AUTOMODEL",
    "MCORE",
    "NEMO_GYM",
    "SGLANG",
    "TRTLLM",
]


def _directory_arg(command: str) -> str:
    """The value `uv run` actually receives for --directory."""
    parts = shlex.split(command)
    return parts[parts.index("--directory") + 1]


@pytest.mark.parametrize("attr", _UV_EXECUTABLES)
def test_directory_survives_the_shlex_split_in_venvs(attr):
    """Vacuous on a space-free checkout, and the whole point on one with a space."""
    assert _directory_arg(getattr(PY_EXECUTABLES, attr)) == git_root


@pytest.mark.parametrize("attr", _UV_EXECUTABLES)
def test_the_repo_root_is_quoted_rather_than_interpolated_raw(attr):
    """Holds regardless of where the checkout happens to live.

    ``shlex.quote`` is a no-op on a path that needs no quoting, so asserting the
    quoted form is present is what makes this independent of the CI path.
    """
    assert shlex.quote(git_root) in getattr(PY_EXECUTABLES, attr)


@pytest.mark.parametrize(
    "root",
    [
        "/mnt/c/Users/First Last/src/NeMo-RL",  # WSL, Windows account with a space
        "/Users/First Last/src/NeMo-RL",  # macOS default home
        "/home/u/NeMo RL",  # space in the repo directory itself
    ],
)
def test_raw_interpolation_truncates_where_quoting_does_not(root):
    """Pins the mechanism, so a future rewrite cannot silently reintroduce it."""
    raw = f"uv run --locked --extra vllm --directory {root}"
    assert _directory_arg(raw) != root, "expected the unquoted form to truncate"

    quoted = f"uv run --locked --extra vllm --directory {shlex.quote(root)}"
    assert _directory_arg(quoted) == root


# ============================================================================
# The tests above are vacuous on a space-free checkout, which is every CI
# runner: `shlex.quote` is the identity on a path that needs no quoting, so
# both assertions hold with or without it. These rebuild the constants against
# a repo root that does contain a space, which is what actually fails on a
# revert.
# ============================================================================

_SPACEY_SUFFIX = " with space"

# The three modelopt constants are built the same way but live in their own
# module, and nothing in this file reached them before.
_MODELOPT_EXECUTABLES = [
    "MODELOPT_VLLM_EXECUTABLE",
    "MODELOPT_AUTOMODEL_EXECUTABLE",
    "MODELOPT_MCORE_EXECUTABLE",
]


@contextlib.contextmanager
def _repo_root_with_a_space():
    """Re-import the executable modules as if the checkout path had a space.

    ``git_root`` is computed at module scope from ``os.path.abspath``, and the
    executable strings are f-strings evaluated at the same time, so the only
    way to see the un-quoted behaviour is to re-execute both with a different
    root. Both modules are reloaded again on the way out.
    """
    import nemo_rl.distributed.virtual_cluster as vc
    import nemo_rl.modelopt.registry as registry

    real_abspath = os.path.abspath
    try:
        with patch("os.path.abspath", lambda p: real_abspath(p) + _SPACEY_SUFFIX):
            importlib.reload(vc)
            importlib.reload(registry)
            yield vc, registry
    finally:
        importlib.reload(vc)
        importlib.reload(registry)


@pytest.mark.parametrize("attr", _UV_EXECUTABLES)
def test_a_repo_root_with_a_space_survives_uv_argument_parsing(attr):
    with _repo_root_with_a_space() as (vc, _):
        assert _SPACEY_SUFFIX in vc.git_root, "the fixture did not take effect"
        assert _directory_arg(getattr(vc.PY_EXECUTABLES, attr)) == vc.git_root


@pytest.mark.parametrize("name", _MODELOPT_EXECUTABLES)
def test_the_modelopt_executables_quote_the_repo_root_too(name):
    """Three of the eleven interpolation sites live in ``nemo_rl/modelopt``.
    Reverting that file alone left the rest of this suite green."""
    with _repo_root_with_a_space() as (_, registry):
        assert _directory_arg(getattr(registry, name)) == registry.git_root


def test_the_fixture_itself_would_catch_an_unquoted_root():
    """Guards the guard: if reloading ever stopped changing the root, every
    assertion above would go quiet again rather than fail."""
    with _repo_root_with_a_space() as (vc, _):
        unquoted = f"uv run --locked --directory {vc.git_root}"
        assert _directory_arg(unquoted) != vc.git_root
