# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from tests.mock_stack.components import CopyRefit, Generation, Policy
from tests.mock_stack.config import ComponentSpec, Scenario


class FailingRefit(CopyRefit):
    def sync_weights(self, **kwargs):
        raise RuntimeError("injected refit failure")


def test_scenario_can_replace_refit_without_changing_construction():
    policy = ComponentSpec(factory="tests.mock_stack.components:Policy").build()
    generation = ComponentSpec(factory="tests.mock_stack.components:Generation").build()
    refit = ComponentSpec(factory="tests.mock_stack.test_config:FailingRefit").build(
        policy=policy, generation=generation
    )
    assert isinstance(policy, Policy)
    assert isinstance(generation, Generation)
    with pytest.raises(RuntimeError, match="injected refit failure"):
        refit.sync_weights()


def test_component_rejects_unknown_config_before_construction():
    with pytest.raises(ValidationError):
        ComponentSpec(
            factory="tests.mock_stack.components:Policy", config={"train_secods": 1}
        ).build()


def test_component_rejects_negative_training_time():
    with pytest.raises(ValidationError):
        ComponentSpec(
            factory="tests.mock_stack.components:Policy", config={"train_seconds": -1}
        ).build()


def test_checkpoint_scenario_keeps_agreed_workload():
    from pathlib import Path

    scenario = Scenario.load(Path(__file__).with_name("checkpoint.yaml"))
    assert scenario.siblings_per_prompt == 3
    assert sum(p.turns * scenario.siblings_per_prompt for p in scenario.prompts) == 51
    p3 = scenario.prompts[2]
    assert p3.id == "P3"
    assert p3.turn_seconds == [2, 7, 7]


def test_scenario_rejects_wrong_sibling_timing_count():
    with pytest.raises(ValidationError, match="one delay per sibling"):
        Scenario.model_validate(
            {
                "siblings_per_prompt": 3,
                "prompts": [{"id": "P1", "turns": 2, "turn_seconds": [1, 2]}],
                "policy": {"factory": "tests.mock_stack.components:Policy"},
                "generation": {"factory": "tests.mock_stack.components:Generation"},
                "refit": {"factory": "tests.mock_stack.components:CopyRefit"},
            }
        )
