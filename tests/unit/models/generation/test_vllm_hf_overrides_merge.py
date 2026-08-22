# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Regression tests for propagating hf_config_overrides to the generation engine.

The algorithm entry points push ``policy.hf_config_overrides`` into
``policy.generation.vllm_kwargs.hf_overrides`` so the generation engine sees the same
HF config as the training policy. Assigning it wholesale discarded any user-supplied
``hf_overrides``, which is the same overwrite-vs-merge bug the worker hit in
#1413/#2904 -- where it was fixed, silently reverted (#2188), and fixed again. The
source checks below pin the call sites so this one cannot be reverted the same way.
"""

import ast
import logging
import pathlib
import warnings

import pytest

from nemo_rl.models.generation.vllm.config import merge_hf_overrides

REPO = pathlib.Path(__file__).resolve().parents[4]

ENTRY_POINTS = (
    "nemo_rl/algorithms/grpo.py",
    "nemo_rl/algorithms/ppo.py",
    "nemo_rl/algorithms/distillation.py",
    "nemo_rl/algorithms/single_controller_utils/setup.py",
)

# The merge helper itself, and the worker's own fp8 merge and non-dict guard.
ALLOWED_TO_SET_HF_OVERRIDES = frozenset(
    {
        "nemo_rl/models/generation/vllm/config.py",
        "nemo_rl/models/generation/vllm/vllm_worker.py",
    }
)


def _lines_setting_hf_overrides(tree: ast.Module) -> list[int]:
    """Lines that replace ``hf_overrides`` wholesale, by assignment or ``update()``."""
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value == "hf_overrides"
    ]
    lines += [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        for arg in [*node.args, *(kw.value for kw in node.keywords)]
        if isinstance(arg, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == "hf_overrides"
            for key in arg.keys
        )
    ]
    lines += [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update"
        and any(kw.arg == "hf_overrides" for kw in node.keywords)
    ]
    return sorted(lines)


def test_generation_only_hf_overrides_survive():
    """A generation-side override is kept when the policy declares none."""
    vllm_kwargs = {"hf_overrides": {"rope_scaling": {"rope_type": "yarn"}}}

    merge_hf_overrides(vllm_kwargs, hf_config_overrides={})

    assert vllm_kwargs["hf_overrides"] == {"rope_scaling": {"rope_type": "yarn"}}


def test_policy_hf_config_overrides_are_propagated():
    """The training policy's overrides still reach the generation engine."""
    vllm_kwargs = {}

    merge_hf_overrides(
        vllm_kwargs, hf_config_overrides={"max_position_embeddings": 8192}
    )

    assert vllm_kwargs["hf_overrides"] == {"max_position_embeddings": 8192}


def test_both_sides_coexist():
    """Non-conflicting keys from both sides are merged."""
    vllm_kwargs = {"hf_overrides": {"rope_scaling": {"rope_type": "yarn"}}}

    merge_hf_overrides(
        vllm_kwargs, hf_config_overrides={"max_position_embeddings": 8192}
    )

    assert vllm_kwargs["hf_overrides"] == {
        "rope_scaling": {"rope_type": "yarn"},
        "max_position_embeddings": 8192,
    }


def test_policy_wins_on_conflict_and_warns():
    """The policy wins on key collision, loudly, so the configs cannot disagree."""
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 4096}}

    with pytest.warns(UserWarning, match="max_position_embeddings"):
        merge_hf_overrides(
            vllm_kwargs, hf_config_overrides={"max_position_embeddings": 8192}
        )

    assert vllm_kwargs["hf_overrides"] == {"max_position_embeddings": 8192}


def test_shallow_merge_does_not_blend_nested_overrides():
    """A policy-declared key replaces its generation-side counterpart wholesale."""
    vllm_kwargs = {"hf_overrides": {"rope_scaling": {"rope_type": "yarn", "factor": 4}}}

    with pytest.warns(UserWarning, match="rope_scaling"):
        merge_hf_overrides(
            vllm_kwargs, hf_config_overrides={"rope_scaling": {"rope_type": "linear"}}
        )

    assert vllm_kwargs["hf_overrides"] == {"rope_scaling": {"rope_type": "linear"}}


def test_none_treated_as_empty():
    """``None`` on either side (e.g. from config defaults) is handled as empty."""
    vllm_kwargs = {"hf_overrides": None}

    merge_hf_overrides(vllm_kwargs, hf_config_overrides=None)

    assert vllm_kwargs["hf_overrides"] == {}


def test_callable_hf_overrides_dropped_with_a_warning():
    """vLLM's callable form cannot be merged; drop it as the worker already does."""
    vllm_kwargs = {"hf_overrides": lambda config: config}

    with pytest.warns(UserWarning, match="cannot be merged"):
        merge_hf_overrides(
            vllm_kwargs, hf_config_overrides={"max_position_embeddings": 8192}
        )

    assert vllm_kwargs["hf_overrides"] == {"max_position_embeddings": 8192}


def test_identical_values_do_not_warn():
    """A key both sides set to the same value overrides nothing, so stay quiet."""
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merge_hf_overrides(
            vllm_kwargs, hf_config_overrides={"max_position_embeddings": 8192}
        )

    assert vllm_kwargs["hf_overrides"] == {"max_position_embeddings": 8192}


def test_resolved_overrides_are_logged(caplog):
    """The value the engine will actually see is reported at launch."""
    vllm_kwargs = {"hf_overrides": {"rope_scaling": {"rope_type": "yarn"}}}

    with caplog.at_level(logging.INFO, logger=merge_hf_overrides.__module__):
        merge_hf_overrides(vllm_kwargs, hf_config_overrides=None)

    assert "rope_scaling" in caplog.text


def test_only_the_merge_helper_sets_hf_overrides():
    """Nothing outside the helper may replace hf_overrides, including future call sites.

    The dynamo and single-controller generation paths were added after the original
    grpo/ppo/distillation ones and inherited this bug, so listing today's call sites
    would not catch the next one. Scan the package instead and allowlist the two files
    that legitimately set hf_overrides.
    """
    offenders = {}
    for path in sorted((REPO / "nemo_rl").rglob("*.py")):
        rel_path = path.relative_to(REPO).as_posix()
        if rel_path in ALLOWED_TO_SET_HF_OVERRIDES:
            continue
        lines = _lines_setting_hf_overrides(ast.parse(path.read_text()))
        if lines:
            offenders[rel_path] = lines

    assert not offenders, (
        f"{offenders} set hf_overrides directly; call merge_hf_overrides() instead so "
        "generation-side overrides are not discarded."
    )


@pytest.mark.parametrize("rel_path", ENTRY_POINTS)
def test_entry_point_calls_merge_hf_overrides(rel_path: str):
    """Every entry point that wires up generation routes through the merge helper."""
    tree = ast.parse((REPO / rel_path).read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "merge_hf_overrides"
    ]

    assert calls, f"{rel_path} does not call merge_hf_overrides()"
