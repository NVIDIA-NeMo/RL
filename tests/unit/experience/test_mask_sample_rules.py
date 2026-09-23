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
from __future__ import annotations

import pytest

from nemo_rl.experience.mask_sample_rules import (
    MaskSampleRule,
    apply_mask_sample_rules,
    mask_rule_metrics,
    mask_rule_step_metrics,
    matching_rules,
    parse_mask_sample_rules,
)

RULES_CFG = [
    {"name": "eval_incomplete", "field": "evaluation_completed", "equals": False},
    {"name": "harness_unfinished", "field": "opencode_finished", "equals": False},
    {"field": "verify.status", "equals": "timeout"},  # name defaults to the path
]


def test_parse_absent_none_or_empty_means_no_rules():
    assert parse_mask_sample_rules(None) == ()
    assert parse_mask_sample_rules({}) == ()
    assert parse_mask_sample_rules({"mask_sample_rules": None}) == ()
    assert parse_mask_sample_rules({"mask_sample_rules": []}) == ()


def test_parse_reads_rules_and_defaults_the_name():
    rules = parse_mask_sample_rules({"mask_sample_rules": RULES_CFG})
    assert rules == (
        MaskSampleRule("eval_incomplete", "evaluation_completed", False),
        MaskSampleRule("harness_unfinished", "opencode_finished", False),
        MaskSampleRule("verify_status", "verify.status", "timeout"),
    )
    assert rules[0].metric_key == "mask_rules/eval_incomplete_rate"


@pytest.mark.parametrize(
    "bad",
    [
        "evaluation_completed",  # not a list
        [{"equals": False}],  # missing field
        [{"field": "", "equals": False}],  # empty field
        [{"field": "x"}],  # missing equals
        [{"field": "x", "equals": [1]}],  # non-scalar equals
        [{"field": "x", "equals": 1, "op": "lt"}],  # unknown key
        [
            {"name": "a", "field": "x", "equals": 1},
            {"name": "a", "field": "y", "equals": 2},
        ],  # dup name
    ],
)
def test_parse_rejects_malformed_rules(bad):
    with pytest.raises(ValueError):
        parse_mask_sample_rules({"mask_sample_rules": bad})


def test_matching_is_type_strict_and_missing_fields_never_match():
    rules = parse_mask_sample_rules({"mask_sample_rules": RULES_CFG})
    assert matching_rules(
        {"evaluation_completed": False, "opencode_finished": True}, rules
    ) == ["eval_incomplete"]
    # `equals: false` must not match 0, "" or None.
    assert matching_rules({"evaluation_completed": 0}, rules) == []
    assert matching_rules({"evaluation_completed": ""}, rules) == []
    assert matching_rules({"evaluation_completed": None}, rules) == []
    # Absent fields are not a match.
    assert matching_rules({"reward": 1.0}, rules) == []
    # Dotted paths descend into nested mappings.
    assert matching_rules({"verify": {"status": "timeout"}}, rules) == ["verify_status"]
    assert matching_rules({"verify": "timeout"}, rules) == []
    # Several rules can match one response.
    assert matching_rules(
        {"evaluation_completed": False, "opencode_finished": False}, rules
    ) == [
        "eval_incomplete",
        "harness_unfinished",
    ]


def test_apply_sets_the_env_mask_flag_only_on_match():
    rules = parse_mask_sample_rules({"mask_sample_rules": RULES_CFG})
    clean = {"evaluation_completed": True, "opencode_finished": True, "reward": 1.0}
    assert apply_mask_sample_rules(clean, rules) == []
    assert "instance_config" not in clean  # untouched

    flagged = {"evaluation_completed": False, "reward": 0.0}
    assert apply_mask_sample_rules(flagged, rules) == ["eval_incomplete"]
    assert flagged["instance_config"] == {"mask_sample": True}

    # An existing instance_config is extended, not replaced; a non-dict one is replaced.
    existing = {"opencode_finished": False, "instance_config": {"keep": 1}}
    apply_mask_sample_rules(existing, rules)
    assert existing["instance_config"] == {"keep": 1, "mask_sample": True}
    odd = {"opencode_finished": False, "instance_config": None}
    apply_mask_sample_rules(odd, rules)
    assert odd["instance_config"] == {"mask_sample": True}

    # No rules -> nothing happens, even for responses that would match.
    untouched = {"evaluation_completed": False}
    assert apply_mask_sample_rules(untouched, ()) == []
    assert untouched == {"evaluation_completed": False}


def test_metrics_report_a_rate_for_every_configured_rule():
    rules = parse_mask_sample_rules({"mask_sample_rules": RULES_CFG})
    assert mask_rule_metrics({}, (), 16) == {}
    assert mask_rule_metrics({"eval_incomplete": 4}, rules, 0) == {}
    assert mask_rule_metrics(
        {"eval_incomplete": 4, "harness_unfinished": 1},
        rules,
        16,
        reward_sums={"eval_incomplete": 1.0, "harness_unfinished": 0.0},
        any_count=4,
    ) == {
        "mask_rules/eval_incomplete_rate": 0.25,
        "mask_rules/eval_incomplete_reward_mean": 0.25,
        "mask_rules/harness_unfinished_rate": 1 / 16,
        "mask_rules/harness_unfinished_reward_mean": 0.0,
        "mask_rules/verify_status_rate": 0.0,
        "mask_rules/any_rate": 0.25,
    }
    # Without reward sums only the rates (and any_rate) are reported.
    assert mask_rule_metrics({"eval_incomplete": 2}, rules, 8) == {
        "mask_rules/eval_incomplete_rate": 0.25,
        "mask_rules/harness_unfinished_rate": 0.0,
        "mask_rules/verify_status_rate": 0.0,
        "mask_rules/any_rate": 0.0,
    }


def test_step_metrics_report_counts_fracs_and_reward_means():
    rules = parse_mask_sample_rules({"mask_sample_rules": RULES_CFG})
    assert (
        mask_rule_step_metrics({}, (), reward_sums={}, any_count=0, rollouts_seen=512)
        == {}
    )
    assert (
        mask_rule_step_metrics({}, rules, reward_sums={}, any_count=0, rollouts_seen=0)
        == {}
    )
    out = mask_rule_step_metrics(
        {"eval_incomplete": 6, "harness_unfinished": 4},
        rules,
        reward_sums={"eval_incomplete": 0.0, "harness_unfinished": 1.0},
        any_count=9,  # one rollout matched both rules
        rollouts_seen=512,
    )
    assert out == {
        "mask_rules/rollouts_seen": 512.0,
        "mask_rules/eval_incomplete_count": 6.0,
        "mask_rules/eval_incomplete_frac": 6 / 512,
        "mask_rules/eval_incomplete_reward_mean": 0.0,
        "mask_rules/harness_unfinished_count": 4.0,
        "mask_rules/harness_unfinished_frac": 4 / 512,
        "mask_rules/harness_unfinished_reward_mean": 0.25,
        "mask_rules/verify_status_count": 0.0,
        "mask_rules/verify_status_frac": 0.0,
        "mask_rules/any_count": 9.0,
        "mask_rules/any_frac": 9 / 512,
    }
