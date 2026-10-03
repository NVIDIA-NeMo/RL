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
"""How ``_build_gym_actor_config`` splits ``env.nemo_gym`` between the actor and Gym.

The keys the actor itself reads (the port range it writes back into Gym's
global configuration, the driver's rollout settings) must become top-level
fields of the actor's configuration; everything else is Gym's global
configuration. A key left in the wrong place is silently ignored, so the split
is pinned here.
"""

import nemo_rl.environments.nemo_gym as nemo_gym_mod


def test_actor_fields_leave_the_gym_global_configuration(monkeypatch):
    monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_uv_cache_dir", lambda: None)
    monkeypatch.setattr(nemo_gym_mod, "get_nemo_gym_venv_dir", lambda: None)

    nemo_gym_dict = {
        "config_paths": ["a.yaml"],
        "port_range_high": 5499,
        "health_check_interval_seconds": 60,
        "max_infra_attempts_per_rollout": 3,
        "rollout_metrics_hook": "package.module.function",
        "thinking_tags": ["<think>", "</think>"],
    }
    cfg = nemo_gym_mod._build_gym_actor_config(
        nemo_gym_dict,
        base_urls=["http://engine.example/v1"],
        model_name="model",
        enable_router_replay=False,
        use_fastokens=False,
    )
    # The bound the launcher injects reaches the field the actor reads at spin-up.
    assert cfg["port_range_high"] == 5499
    assert "port_range_low" not in cfg
    assert cfg["health_check_interval_seconds"] == 60
    assert cfg["max_infra_attempts_per_rollout"] == 3
    assert cfg["rollout_metrics_hook"] == "package.module.function"
    assert cfg["thinking_tags"] == ["<think>", "</think>"]
    gym_global = cfg["initial_global_config_dict"]
    assert gym_global["config_paths"] == ["a.yaml"]
    for key in (
        "port_range_high",
        "health_check_interval_seconds",
        "max_infra_attempts_per_rollout",
        "rollout_metrics_hook",
        "thinking_tags",
    ):
        assert key not in gym_global
    # The caller's mapping is left untouched.
    assert nemo_gym_dict["port_range_high"] == 5499


def test_rollout_config_reports_every_driver_key(monkeypatch):
    """``rollout_config`` hands the driver's keys back; an absent key reads as None."""
    rollout_config = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class.rollout_config

    class _Actor:
        cfg = {
            "health_check_interval_seconds": 60,
            "max_infra_attempts_per_rollout": 2,
            "rollout_metrics_hook": "package.module.function",
        }

    assert rollout_config(_Actor()) == {
        "health_check_interval_seconds": 60,
        "max_infra_attempts_per_rollout": 2,
        "rollout_metrics_hook": "package.module.function",
    }
    _Actor.cfg = {}
    assert rollout_config(_Actor()) == {
        "health_check_interval_seconds": None,
        "max_infra_attempts_per_rollout": None,
        "rollout_metrics_hook": None,
    }
