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


# --- numeric comparators (gt / ge / lt / le) ---------------------------------

COMPARATOR_CFG = [
    {"name": "too_many_compactions", "field": "opencode_num_compactions", "gt": 3},
    {"name": "at_least_one_compaction", "field": "opencode_num_compactions", "ge": 1},
    {"name": "few_calls", "field": "opencode_num_model_calls", "lt": 5},
    {"name": "short_run", "field": "stats.duration_s", "le": 2.5},
]


def test_parse_reads_comparator_rules_with_operator_and_operand():
    rules = parse_mask_sample_rules({"mask_sample_rules": COMPARATOR_CFG})
    assert rules == (
        MaskSampleRule("too_many_compactions", "opencode_num_compactions", 3, "gt"),
        MaskSampleRule("at_least_one_compaction", "opencode_num_compactions", 1, "ge"),
        MaskSampleRule("few_calls", "opencode_num_model_calls", 5, "lt"),
        MaskSampleRule("short_run", "stats.duration_s", 2.5, "le"),
    )
    assert [rule.operator for rule in rules] == ["gt", "ge", "lt", "le"]
    assert rules[0].operand == 3
    # Metric names do not depend on the operator.
    assert rules[0].metric_key == "mask_rules/too_many_compactions_rate"


def test_operator_defaults_to_equals_so_existing_constructors_keep_working():
    rule = MaskSampleRule("eval_incomplete", "evaluation_completed", False)
    assert rule.operator == "equals"
    parsed = parse_mask_sample_rules(
        {"mask_sample_rules": [{"field": "evaluation_completed", "equals": False}]}
    )
    assert parsed == (rule.__class__("evaluation_completed", "evaluation_completed", False),)
    assert parsed[0].operator == "equals"


@pytest.mark.parametrize(
    ("op", "operand", "hits", "misses"),
    [
        ("gt", 3, [4, 100, 3.5], [3, 2, 0, -1]),
        ("ge", 3, [3, 4, 3.0], [2, 2.999, 0]),
        ("lt", 3, [2, 0, -1, 2.5], [3, 4, 3.0]),
        ("le", 3, [3, 2, 3.0, -7], [4, 3.01]),
        ("gt", 2.5, [3, 2.51], [2.5, 2, 0]),
        ("le", 0.5, [0.5, 0, -1], [1, 0.51]),
    ],
)
def test_each_comparator_matches_numbers_on_the_expected_side(op, operand, hits, misses):
    rules = parse_mask_sample_rules(
        {"mask_sample_rules": [{"name": "r", "field": "n", op: operand}]}
    )
    for value in hits:
        assert matching_rules({"n": value}, rules) == ["r"], (op, operand, value)
    for value in misses:
        assert matching_rules({"n": value}, rules) == [], (op, operand, value)


def test_comparators_never_match_bools_or_non_numeric_values():
    rules = parse_mask_sample_rules(
        {
            "mask_sample_rules": [
                {"name": "gt0", "field": "n", "gt": 0},
                {"name": "ge0", "field": "n", "ge": 0},
                {"name": "lt1", "field": "n", "lt": 1},
                {"name": "le1", "field": "n", "le": 1},
            ]
        }
    )
    # bool is an int subclass in Python, but a flag is not a count: True must
    # not satisfy `gt: 0` and False must not satisfy `lt: 1` / `le: 1` / `ge: 0`.
    assert matching_rules({"n": True}, rules) == []
    assert matching_rules({"n": False}, rules) == []
    # Strings, None and containers are not comparable -> no match, no exception.
    assert matching_rules({"n": "3"}, rules) == []
    assert matching_rules({"n": None}, rules) == []
    assert matching_rules({"n": [3]}, rules) == []
    assert matching_rules({"n": {"value": 3}}, rules) == []
    # NaN compares false on every side.
    assert matching_rules({"n": float("nan")}, rules) == []
    # Sanity: a real number still matches.
    assert matching_rules({"n": 0.5}, rules) == ["gt0", "ge0", "lt1", "le1"]


def test_comparators_do_not_match_missing_fields():
    rules = parse_mask_sample_rules({"mask_sample_rules": COMPARATOR_CFG})
    assert matching_rules({}, rules) == []
    assert matching_rules({"reward": 1.0}, rules) == []
    # Partial dotted path (stats present but not stats.duration_s) is still missing.
    assert matching_rules({"stats": {}}, rules) == []
    assert matching_rules({"stats": 2.0}, rules) == []
    # And nested numeric fields are found when present.
    assert matching_rules({"stats": {"duration_s": 1.0}}, rules) == ["short_run"]


def test_comparator_rules_set_the_mask_flag_like_equals_rules():
    rules = parse_mask_sample_rules({"mask_sample_rules": COMPARATOR_CFG[:1]})
    clean = {"opencode_num_compactions": 3, "reward": 1.0}
    assert apply_mask_sample_rules(clean, rules) == []
    assert "instance_config" not in clean
    capped = {"opencode_num_compactions": 4, "reward": 1.0}
    assert apply_mask_sample_rules(capped, rules) == ["too_many_compactions"]
    assert capped["instance_config"] == {"mask_sample": True}


def test_equals_semantics_are_unchanged_by_the_comparator_support():
    rules = parse_mask_sample_rules(
        {
            "mask_sample_rules": [
                {"name": "is_zero", "field": "n", "equals": 0},
                {"name": "is_false", "field": "b", "equals": False},
                {"name": "is_timeout", "field": "s", "equals": "timeout"},
            ]
        }
    )
    assert matching_rules({"n": 0}, rules) == ["is_zero"]
    assert matching_rules({"n": 0.0}, rules) == ["is_zero"]  # plain == for numbers
    assert matching_rules({"n": False}, rules) == []  # bool never equals a number
    assert matching_rules({"b": 0}, rules) == []  # number never equals a bool
    assert matching_rules({"b": False}, rules) == ["is_false"]
    assert matching_rules({"s": "timeout"}, rules) == ["is_timeout"]
    # `equals` still accepts any scalar operand, including strings and null.
    parse_mask_sample_rules({"mask_sample_rules": [{"field": "x", "equals": None}]})
    parse_mask_sample_rules({"mask_sample_rules": [{"field": "x", "equals": "s"}]})


@pytest.mark.parametrize(
    "bad",
    [
        [{"field": "x", "equals": 1, "gt": 0}],  # equals + comparator
        [{"field": "x", "gt": 0, "lt": 5}],  # two comparators (no ranges)
        [{"field": "x", "ge": 0, "le": 5}],
        [{"field": "x", "gt": 0, "ge": 0, "lt": 5, "le": 5, "equals": 1}],  # all five
        [{"field": "x"}],  # none at all
        [{"name": "r", "field": "x"}],  # none at all, even with a name
    ],
)
def test_parse_requires_exactly_one_operator(bad):
    with pytest.raises(ValueError, match="exactly one of"):
        parse_mask_sample_rules({"mask_sample_rules": bad})


@pytest.mark.parametrize(
    "bad",
    [
        [{"field": "x", "gt": True}],  # bool operand
        [{"field": "x", "le": False}],
        [{"field": "x", "lt": "3"}],  # string operand
        [{"field": "x", "ge": None}],  # null operand
        [{"field": "x", "gt": [3]}],  # container operand
    ],
)
def test_parse_rejects_non_numeric_comparator_operands(bad):
    with pytest.raises(ValueError, match="must be a number"):
        parse_mask_sample_rules({"mask_sample_rules": bad})


@pytest.mark.parametrize(
    "bad",
    [
        [{"field": "x", "gt": 3, "op": "lt"}],  # unknown key alongside a comparator
        [{"field": "x", "greater_than": 3}],  # misspelt operator is an unknown key
        [{"field": "x", "gte": 3}],
        [{"field": "x", "gt": 3, "threshold": 3}],
    ],
)
def test_parse_still_rejects_unknown_keys_with_comparators(bad):
    with pytest.raises(ValueError, match="unknown keys"):
        parse_mask_sample_rules({"mask_sample_rules": bad})


def test_direct_constructor_validates_operator_and_operand():
    with pytest.raises(ValueError, match="unknown operator"):
        MaskSampleRule("r", "x", 3, "between")
    with pytest.raises(ValueError, match="numeric operand"):
        MaskSampleRule("r", "x", True, "gt")
    with pytest.raises(ValueError, match="numeric operand"):
        MaskSampleRule("r", "x", "3", "lt")
    # Non-numeric operands remain fine for equals.
    assert MaskSampleRule("r", "x", "timeout").operator == "equals"


def test_metrics_are_operator_agnostic():
    rules = parse_mask_sample_rules({"mask_sample_rules": COMPARATOR_CFG[:2]})
    assert mask_rule_metrics({"too_many_compactions": 2}, rules, 8, any_count=2) == {
        "mask_rules/too_many_compactions_rate": 0.25,
        "mask_rules/at_least_one_compaction_rate": 0.0,
        "mask_rules/any_rate": 0.25,
    }
    out = mask_rule_step_metrics(
        {"too_many_compactions": 3},
        rules,
        reward_sums={"too_many_compactions": 1.5},
        any_count=3,
        rollouts_seen=12,
    )
    assert out["mask_rules/too_many_compactions_count"] == 3.0
    assert out["mask_rules/too_many_compactions_frac"] == 0.25
    assert out["mask_rules/too_many_compactions_reward_mean"] == 0.5
    assert out["mask_rules/at_least_one_compaction_count"] == 0.0
