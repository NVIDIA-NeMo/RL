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

"""Exercise the real setup handoff without starting Ray or importing GPU backends."""

import ast
import copy
from pathlib import Path

import pytest


def _setup_handoff(policy_config):
    source = Path(__file__).parents[3] / "nemo_rl/algorithms/grpo.py"
    tree = ast.parse(source.read_text())
    setup = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "setup"
    )
    # Execute the actual two setup statements immediately before actor creation.
    nodes = list(ast.walk(setup))
    start = next(
        node
        for node in nodes
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value) == "generation_config.setdefault('vllm_kwargs', {})"
    )
    end = next(
        node
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(
            ast.unparse(target) == "vllm_kwargs['hf_overrides']"
            for target in node.targets
        )
    )
    namespace = {
        "policy_config": policy_config,
        "generation_config": policy_config["generation"],
    }
    exec(
        compile(ast.Module(body=[start, end], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["generation_config"]["vllm_kwargs"]


@pytest.mark.parametrize(
    "policy_overrides", [{}, {"rope_theta": 10000, "shared": "policy"}]
)
def test_setup_retains_generation_only_audit_and_policy_precedence(policy_overrides):
    audit = {"output_dir": "/job/refit", "samples": {"perception.proj.weight": None}}
    policy = {
        "hf_config_overrides": copy.deepcopy(policy_overrides),
        "generation": {
            "vllm_kwargs": {
                "skip_mm_profiling": True,
                "hf_overrides": {
                    "speechlm_refit_verification": audit,
                    "shared": "generation",
                },
            }
        },
    }
    original_policy_overrides = copy.deepcopy(policy["hf_config_overrides"])
    kwargs = _setup_handoff(policy)
    assert kwargs["hf_overrides"]["speechlm_refit_verification"] == audit
    assert kwargs["hf_overrides"]["shared"] == policy_overrides.get(
        "shared", "generation"
    )
    if policy_overrides:
        assert kwargs["hf_overrides"]["rope_theta"] == 10000
    assert kwargs["skip_mm_profiling"] is True
    assert policy["hf_config_overrides"] == original_policy_overrides
    assert "speechlm_refit_verification" not in policy["hf_config_overrides"]


def test_setup_without_generation_overrides_keeps_legacy_policy_config():
    policy = {"hf_config_overrides": {"rope_theta": 10000}, "generation": {}}
    assert _setup_handoff(policy) == {"hf_overrides": {"rope_theta": 10000}}


@pytest.mark.parametrize(
    "policy_overrides,generation_overrides,expected",
    [
        (
            None,
            {"audit": {"output_dir": "/job/refit"}},
            {"audit": {"output_dir": "/job/refit"}},
        ),
        ({"rope_theta": 10000}, None, {"rope_theta": 10000}),
        (None, None, {}),
    ],
)
def test_setup_none_overrides_are_empty_without_losing_other_side(
    policy_overrides, generation_overrides, expected
):
    policy = {
        "hf_config_overrides": policy_overrides,
        "generation": {"vllm_kwargs": {"hf_overrides": generation_overrides}},
    }
    assert _setup_handoff(policy)["hf_overrides"] == expected
