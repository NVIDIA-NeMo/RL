# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Compact per-environment reward metrics shared by Gym rollout paths."""

from collections import defaultdict


def calculate_per_game_mean_reward(
    nemo_gym_rows: list[dict], results: list[dict]
) -> dict[str, float]:
    """Return only each environment's mean verifier reward, as in Games22/34."""
    game_to_rewards: dict[str, list[float]] = defaultdict(list)
    for row, result in zip(nemo_gym_rows, results):
        game_name = row.get("env_id")
        reward = result.get("full_result", {}).get("reward")
        if (
            isinstance(game_name, str)
            and game_name
            and isinstance(reward, (bool, int, float))
        ):
            game_to_rewards[game_name].append(float(reward))
    return {
        f"game/{game_name}/reward/mean": sum(rewards) / len(rewards)
        for game_name, rewards in sorted(game_to_rewards.items())
    }
