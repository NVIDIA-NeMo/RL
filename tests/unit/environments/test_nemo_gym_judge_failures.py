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
