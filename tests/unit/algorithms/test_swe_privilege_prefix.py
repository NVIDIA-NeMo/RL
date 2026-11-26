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
import json

import pytest

from nemo_rl.algorithms.ppo_utils import swe_privilege_prefix as spp
from nemo_rl.algorithms.ppo_utils.privilege_prefix_utils import TRUNCATION_MARKER
from nemo_rl.models.value.config import PrivilegedValueConfig
from tests.unit.algorithms.test_privileged_value import _CharTokenizer

_CAPS = spp.SwePrivilegeCaps(
    golden_patch=24576, test_patch=16384, pass_to_pass=4096, fail_to_pass=4096
)


def _env_info(instance_id="repo__1", patch="diff --git a/x b/x\n+fix"):
    metadata = {"instance_id": instance_id, "FAIL_TO_PASS": ["t::a"]}
    if patch:
        metadata["patch"] = patch
    return {"responses_create_params": {"metadata": metadata}}


def _config(max_total_tokens=10_000):
    return PrivilegedValueConfig(
        enabled=True,
        max_total_tokens=max_total_tokens,
        prefix_builder_fqn=f"{spp.__name__}.build_privilege_prefix",
        prefix_builder_kwargs={"caps": _CAPS.model_dump()},
    )


def test_builder_emits_the_reference_block():
    block, stats = spp.build_privilege_prefix(_env_info(), _CharTokenizer(), _config())

    assert block.startswith("<reference>")
    assert "<golden_patch>\ndiff --git a/x b/x\n+fix\n</golden_patch>" in block
    assert "<fail_to_pass>\nt::a\n</fail_to_pass>" in block
    assert stats["truncated_fields"] == []
    assert stats["wanted_tokens"] == stats["kept_tokens"]


def test_builder_fails_loudly_without_a_golden_patch():
    with pytest.raises(ValueError, match="no golden patch"):
        spp.build_privilege_prefix(_env_info(patch=""), _CharTokenizer(), _config())


def test_builder_spends_the_total_budget_from_the_config():
    # pass_to_pass (6 tokens) gets 3 tokens of the 17-token budget, so it is cut.
    row = {
        "patch": "g" * 10,
        "FAIL_TO_PASS": ["t::b"],
        "PASS_TO_PASS": ["t::abc"],
    }

    _, stats = spp.build_privilege_prefix(
        row, _CharTokenizer(), _config(max_total_tokens=17)
    )

    assert stats["truncated_fields"] == ["pass_to_pass"]


def _sandbox_row(instance_id="repo__2", patch="diff --git a/y b/y\n+fix2"):
    """A Gym sandboxed-agent row (e.g. swe_rebench): instance fields at top level."""
    return {
        "instance_id": instance_id,
        "patch": patch,
        "test_patch": "diff --git a/t b/t\n+test",
        "FAIL_TO_PASS": ["t::b"],
        "PASS_TO_PASS": ["t::c"],
        "responses_create_params": {"input": [{"role": "user", "content": "fix"}]},
        "agent_ref": {"type": "responses_api_agents", "name": "agent"},
    }


def test_sandboxed_agent_rows_resolve_from_the_row_top_level():
    fields = spp.resolve_privilege_fields(_sandbox_row())
    assert fields["instance_id"] == "repo__2"
    assert fields["golden_patch"] == "diff --git a/y b/y\n+fix2"
    assert fields["test_patch"] == "diff --git a/t b/t\n+test"
    assert fields["fail_to_pass"] == "t::b"
    assert fields["pass_to_pass"] == "t::c"


def test_metadata_still_wins_over_the_row_top_level():
    row = {**_env_info(), "instance_id": "other", "patch": "top-level patch"}
    fields = spp.resolve_privilege_fields(row)
    assert fields["instance_id"] == "repo__1"
    assert fields["golden_patch"] == "diff --git a/x b/x\n+fix"


def test_r2e_gym_commit_is_reassembled_into_a_patch():
    commit = {
        "file_diffs": [
            {
                "header": {"file": {"path": "pkg/mod.py"}},
                "hunks": [
                    {
                        "descriptor": {
                            "old_range": {"start": 1, "length": 1},
                            "new_range": {"start": 1, "length": 1},
                        },
                        "line_group": {
                            "all_lines": [
                                {"type": "deleted", "content": "a = 1"},
                                {"type": "added", "content": "a = 2"},
                            ]
                        },
                    }
                ],
            }
        ]
    }
    info = {
        "responses_create_params": {
            "metadata": {
                "instance_id": "r2e__1",
                "parsed_commit_content": json.dumps(commit),
            }
        }
    }
    fields = spp.resolve_privilege_fields(info)
    assert "-a = 1\n+a = 2" in fields["golden_patch"]


def test_caps_bound_a_field_within_the_total_budget():
    caps = _CAPS.model_copy(update={"golden_patch": 40})
    fields = {"golden_patch": "g" * 100, "fail_to_pass": "t::b"}

    block, stats = spp.build_reference_block(fields, _CharTokenizer(), 1000, caps)

    assert stats["truncated_fields"] == ["golden_patch"]
    golden = block.split("<golden_patch>\n", 1)[1].split("\n</golden_patch>", 1)[0]
    assert len(golden) == 40 and golden.endswith(TRUNCATION_MARKER)


def test_a_field_cut_down_to_the_marker_still_counts_as_truncated():
    # pass_to_pass (6 tokens) gets 3 tokens of budget: it is replaced by the
    # 16-token marker, which outweighs the original.
    fields = {
        "fail_to_pass": "t::b",
        "golden_patch": "g" * 10,
        "pass_to_pass": "t::abc",
    }

    block, stats = spp.build_reference_block(fields, _CharTokenizer(), 17, _CAPS)

    assert stats["truncated_fields"] == ["pass_to_pass"]
    assert "[truncated: pass_to_pass]" in block
