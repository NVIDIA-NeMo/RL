# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.failures import GymTransportError, RolloutDataFailure


@pytest.mark.parametrize(
    "failure,exception",
    [("judge_failed", GymTransportError), ("unknown", RolloutDataFailure)],
)
def test_tagged_failures_never_reach_token_or_reward_processing(failure, exception):
    # No response/tokenizer/self state: rejection must precede all processing.
    with pytest.raises(exception, match="reward is not valid"):
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            None, {}, {"_ng_failure_class": failure, "reward": 0.0}, None
        )


def test_judge_failure_preserves_bounded_cause_and_row_identity(capsys):
    row = {
        "agent_ref": {"name": "equivalence_llm_judge_simple_agent"},
        "metadata": {"uuid": "example-17"},
        "responses_create_params": {"input": "PRIVATE_PROMPT"},
    }
    result = {
        "_ng_failure_class": "judge_failed",
        "_ng_failure_judge_error": "Judge response contains no valid verdict "
        + "x" * 2000,
        "response": "LARGE_POLICY_RESPONSE",
        "reward": 0.0,
    }
    with pytest.raises(GymTransportError) as caught:
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            None, row, result, None
        )
    message = str(caught.value)
    assert "no valid verdict" in message
    assert "example-17" in message
    assert "equivalence_llm_judge_simple_agent" in message
    assert "truncated" in message
    assert len(message) < 800
    trace = capsys.readouterr().out
    assert (
        json.loads(trace.split("[nemo_gym_trace] ", 1)[1])["event"]
        == "actor_tagged_rollout_failure"
    )
    for private in ("PRIVATE_PROMPT", "LARGE_POLICY_RESPONSE"):
        assert private not in message + trace
