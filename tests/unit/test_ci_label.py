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

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "ci_label", Path(__file__).resolve().parents[2] / ".github/scripts/ci_label.py"
)
ci_label = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_label)


@pytest.mark.parametrize(
    ("files", "label"),
    [
        (["docs/guides/grpo.md", "README.md"], "CI:docs"),
        (["skills/cicd/SKILL.md", ".github/autobot/review-config.yml"], "CI:docs"),
        ([".github/PULL_REQUEST_TEMPLATE.md"], "CI:docs"),
        (["nemo_rl/algorithms/grpo.py", "docs/guides/grpo.md"], "CI:Lfast"),
        (["examples/configs/grpo_math_1B.yaml"], "CI:Lfast"),
        ([".github/workflows/cherry-pick-release-commit.yml"], "CI:docs"),
        ([".github/workflows/pr-description-check.yml"], "CI:docs"),
        ([".github/scripts/ci_label.py"], "CI:docs"),
        ([".github/workflows/cicd-main.yml"], "CI:Lfast"),
        ([".github/workflows/_build_container.yml"], "CI:Lfast"),
        ([".github/actions/test-template/action.yml"], "CI:Lfast"),
        ([".github/scripts/plan-tests.sh"], "CI:Lfast"),
        (["uv.lock", "docs/x.md"], "CI:L1"),
        (["3rdparty/Gym-workspace/Gym"], "CI:L1"),
        (["docker/Dockerfile"], "CI:L1"),
        (["docker/install_mooncake.sh"], "CI:L1"),
        (["docker/prune_uv_cache.py"], "CI:L1"),
        (["docker/dynamo/install.sh"], "CI:L1"),
        (["docker/README.md"], "CI:docs"),
        ([], "CI:Lfast"),
    ],
)
def test_pick_label(files: list[str], label: str) -> None:
    assert ci_label.pick_label(files)[0] == label


def test_bot_sets_pick_when_no_label() -> None:
    plan = ci_label.decide(pick="CI:Lfast", pick_changed=False, labels={})
    assert (plan.add, plan.remove, plan.reverted) == (["CI:Lfast"], [], [])


def test_non_maintainer_label_is_replaced_and_reported() -> None:
    plan = ci_label.decide(
        pick="CI:Lfast", pick_changed=False, labels={"CI:docs": ("someone", False)}
    )
    assert (plan.add, plan.remove, plan.reverted) == (
        ["CI:Lfast"],
        ["CI:docs"],
        [("CI:docs", "someone")],
    )


def test_maintainer_label_kept_while_pick_unchanged() -> None:
    plan = ci_label.decide(
        pick="CI:Lfast", pick_changed=False, labels={"CI:L1": ("terrykong", True)}
    )
    assert (plan.add, plan.remove, plan.reverted) == ([], [], [])


def test_maintainer_label_dropped_when_push_changes_pick() -> None:
    plan = ci_label.decide(
        pick="CI:L1", pick_changed=True, labels={"CI:docs": ("terrykong", True)}
    )
    assert (plan.add, plan.remove, plan.reverted) == (["CI:L1"], ["CI:docs"], [])


def test_bot_label_updates_silently() -> None:
    plan = ci_label.decide(
        pick="CI:L1", pick_changed=True, labels={"CI:Lfast": ("bot", False)}
    )
    assert (plan.add, plan.remove, plan.reverted) == (["CI:L1"], ["CI:Lfast"], [])


def test_existing_pick_label_is_left_alone() -> None:
    plan = ci_label.decide(
        pick="CI:Lfast", pick_changed=False, labels={"CI:Lfast": ("someone", False)}
    )
    assert (plan.add, plan.remove, plan.reverted) == ([], [], [])


def test_extra_labels_reduced_to_one() -> None:
    labels = {"CI:Lfast": ("bot", False), "CI:L1": ("someone", False)}
    plan = ci_label.decide(pick="CI:Lfast", pick_changed=False, labels=labels)
    assert (plan.add, plan.remove, plan.reverted) == (
        [],
        ["CI:L1"],
        [("CI:L1", "someone")],
    )


@pytest.mark.parametrize(
    ("maintainer", "pick_changed", "removed", "reported"),
    [
        (False, False, True, True),
        (False, True, True, True),
        (True, False, False, False),
        (True, True, True, False),
    ],
)
def test_skip_cicd_only_for_maintainers_until_pick_changes(
    maintainer: bool, pick_changed: bool, removed: bool, reported: bool
) -> None:
    labels = {"CI:Lfast": ("bot", False), "Skip CICD": ("someone", maintainer)}
    plan = ci_label.decide(pick="CI:Lfast", pick_changed=pick_changed, labels=labels)
    assert ("Skip CICD" in plan.remove) is removed
    assert (("Skip CICD", "someone") in plan.reverted) is reported


def test_newest_maintainer_label_wins() -> None:
    labels = {"CI:L1": ("terrykong", True), "CI:L0": ("terrykong", True)}
    plan = ci_label.decide(pick="CI:Lfast", pick_changed=False, labels=labels)
    assert (plan.kept, plan.add, plan.remove, plan.reverted) == (
        "CI:L0",
        [],
        ["CI:L1"],
        [],
    )


def test_newer_non_maintainer_label_does_not_displace_override() -> None:
    labels = {"CI:L1": ("terrykong", True), "CI:L0": ("someone", False)}
    plan = ci_label.decide(pick="CI:Lfast", pick_changed=False, labels=labels)
    assert (plan.kept, plan.remove, plan.reverted) == (
        "CI:L1",
        ["CI:L0"],
        [("CI:L0", "someone")],
    )


def test_label_set_after_push_survives_pick_change() -> None:
    labels = {"CI:L0": ("terrykong", True), "Skip CICD": ("terrykong", True)}
    plan = ci_label.decide(
        pick="CI:L1",
        pick_changed=True,
        labels=labels,
        set_after_push=frozenset({"CI:L0", "Skip CICD"}),
    )
    assert (plan.kept, plan.add, plan.remove) == ("CI:L0", [], [])


def test_label_set_before_push_expires_on_pick_change() -> None:
    labels = {"CI:docs": ("terrykong", True), "CI:L0": ("terrykong", True)}
    plan = ci_label.decide(
        pick="CI:L1",
        pick_changed=True,
        labels=labels,
        set_after_push=frozenset({"CI:L0"}),
    )
    assert (plan.kept, plan.add, plan.remove) == ("CI:L0", [], ["CI:docs"])


def test_comment_names_removed_label_without_mention() -> None:
    body = ci_label.comment_body(
        label="CI:Lfast",
        pick="CI:Lfast",
        reason="default",
        reverted=[("CI:docs", "someone")],
    )
    assert body.startswith(ci_label.COMMENT_MARKER)
    assert "`CI:docs` added by `someone`" in body
    assert "@" not in body


def test_comment_shows_kept_override_not_pick() -> None:
    labels = {"CI:L1": ("terrykong", True), "Skip CICD": ("someone", False)}
    plan = ci_label.decide(pick="CI:Lfast", pick_changed=False, labels=labels)
    body = ci_label.comment_body(
        label=plan.kept, pick="CI:Lfast", reason="default", reverted=plan.reverted
    )
    assert "**CI label:** `CI:L1`, set by a maintainer" in body
    assert "set automatically" not in body
