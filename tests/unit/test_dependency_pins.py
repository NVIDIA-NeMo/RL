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

"""Guards on pins we declare directly even though a submodule already declares them.

`megatron-energon` reaches us transitively as
nemo-rl[mcore] -> megatron-bridge[te,ssm] -> megatron-core[dev,mlm] -> megatron-energon,
and the `mcore` extra floors it higher than Megatron-LM does. Bumping the
Megatron-Bridge submodule moves the Megatron-LM pointer underneath us, so these tests
fail the moment upstream's own pin changes and the floor needs a second look.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).parents[2]
MEGATRON_LM_PYPROJECT = (
    REPO_ROOT
    / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/pyproject.toml"
)

# What Megatron-LM's `dev` extra pins today. If a Megatron-Bridge bump moves this,
# re-check the megatron-energon floor in our `mcore` extra and update this constant in
# the same commit.
MEGATRON_LM_ENERGON_SPEC = "~=7.0"


def _requirement(pyproject: Path, extra: str, name: str) -> Requirement:
    """The lone requirement for `name` in `pyproject`'s `extra` optional-dependency list."""
    project = tomllib.loads(pyproject.read_text())["project"]
    deps = project["optional-dependencies"][extra]
    matches = [
        req
        for req in map(Requirement, deps)
        if canonicalize_name(req.name) == canonicalize_name(name)
    ]
    assert len(matches) == 1, (
        f"expected exactly one {name} requirement in [{extra}] of {pyproject}, got {matches}"
    )
    return matches[0]


def _bounds(spec: SpecifierSet) -> tuple[Version, Version]:
    """Inclusive lower and exclusive upper bound of a lone `~=X.Y[.Z]` specifier."""
    specifiers = list(spec)
    assert len(specifiers) == 1 and specifiers[0].operator == "~=", (
        f"expected a single compatible-release pin, got '{spec}'"
    )
    release = Version(specifiers[0].version).release
    prefix = release[:-1]  # `~=X.Y` implies `<X+1`; `~=X.Y.Z` implies `<X.Y+1`
    upper = prefix[:-1] + (prefix[-1] + 1,)
    return Version(specifiers[0].version), Version(
        ".".join(str(part) for part in upper)
    )


@pytest.fixture(scope="module")
def megatron_lm_energon() -> Requirement:
    if not MEGATRON_LM_PYPROJECT.exists():
        pytest.skip(
            f"{MEGATRON_LM_PYPROJECT} missing; run `git submodule update --init --recursive`"
        )
    return _requirement(MEGATRON_LM_PYPROJECT, "dev", "megatron-energon")


def test_megatron_lm_energon_pin_is_unchanged(megatron_lm_energon: Requirement) -> None:
    assert str(megatron_lm_energon.specifier) == MEGATRON_LM_ENERGON_SPEC, (
        f"Megatron-LM now pins megatron-energon '{megatron_lm_energon.specifier}', not "
        f"'{MEGATRON_LM_ENERGON_SPEC}'. Re-check the megatron-energon floor in the `mcore` "
        f"extra of pyproject.toml against the new range, then update "
        f"MEGATRON_LM_ENERGON_SPEC here in the same commit."
    )


def test_mcore_energon_floor_is_within_megatron_lm_range(
    megatron_lm_energon: Requirement,
) -> None:
    ours = _requirement(REPO_ROOT / "pyproject.toml", "mcore", "megatron-energon")
    our_low, our_high = _bounds(ours.specifier)
    their_low, their_high = _bounds(megatron_lm_energon.specifier)
    assert our_low >= their_low and our_high <= their_high, (
        f"the `mcore` extra pins megatron-energon '{ours.specifier}', which is not a subset "
        f"of Megatron-LM's '{megatron_lm_energon.specifier}'. Our pin only exists to raise "
        f"the floor; widening it past upstream either makes `uv lock` unsatisfiable or "
        f"silently has no effect."
    )
