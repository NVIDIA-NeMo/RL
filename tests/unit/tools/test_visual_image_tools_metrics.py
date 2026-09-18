from nemo_rl.experience.nemo_gym_metrics import calculate_per_game_mean_reward


def test_per_game_metrics_keep_existing_names_and_only_emit_means():
    names = ["Games/Wordle-v0", "maze_2d/easy", "Games/Wordle-v0", "image_tools/vstar"]
    assert calculate_per_game_mean_reward(
        [{"env_id": name} for name in names],
        [{"full_result": {"reward": reward}} for reward in [0.0, 0.3, 1.0, 1.02]],
    ) == {
        "game/Games/Wordle-v0/reward/mean": 0.5,
        "game/maze_2d/easy/reward/mean": 0.3,
        "game/image_tools/vstar/reward/mean": 1.02,
    }


def test_missing_identity_or_reward_does_not_invent_a_game_metric():
    assert (
        calculate_per_game_mean_reward(
            [{}, {"env_id": ""}, {"env_id": "game"}, {"env_id": "game"}],
            [
                {"full_result": {"reward": 1}},
                {"full_result": {"reward": 1}},
                {},
                {"full_result": {"reward": "not-numeric"}},
            ],
        )
        == {}
    )
