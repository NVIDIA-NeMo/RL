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

"""VLM entrypoint passes the Gym returned by setup to async rollouts."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("gym_enabled", [True, False])
@pytest.mark.parametrize("validation_enabled", [True, False])
def test_setup_gym_reaches_async_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gym_enabled: bool,
    validation_enabled: bool,
) -> None:
    source = Path(__file__).parents[2] / "examples/run_vlm_grpo.py"
    spec = importlib.util.spec_from_file_location("vlm_gym_launcher", source)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    gym = object() if gym_enabled else None
    other_env = object()
    train_envs = {"math": other_env}
    validation_envs = {"math": other_env} if validation_enabled else None
    cfg = SimpleNamespace(
        logger={"log_dir": str(tmp_path)},
        checkpointing={"enabled": False},
        data_plane=None,
        policy={
            "tokenizer": {},
            "generation": {"vllm_cfg": {"skip_tokenizer_init": False}},
        },
        data={"use_multiple_dataloader": False},
        env={},
        grpo=SimpleNamespace(
            async_grpo=SimpleNamespace(enabled=True, max_trajectory_age_steps=1),
            use_dynamic_sampling=False,
            reward_scaling=SimpleNamespace(enabled=False),
            reward_shaping=SimpleNamespace(enabled=False),
        ),
    )
    processor = SimpleNamespace(tokenizer=object())
    monkeypatch.setattr(
        launcher, "parse_args", lambda: (argparse.Namespace(config="test.yaml"), [])
    )
    monkeypatch.setattr(launcher, "load_config", lambda path: OmegaConf.create({}))
    monkeypatch.setattr(launcher, "MasterConfig", lambda **kwargs: cfg)
    monkeypatch.setattr(launcher, "log_container_init_timing", lambda: None)
    monkeypatch.setattr(launcher, "get_next_experiment_dir", lambda path: path)
    monkeypatch.setattr(launcher, "init_ray", lambda: None)
    monkeypatch.setattr(launcher, "get_tokenizer", lambda *a, **kw: processor)
    monkeypatch.setattr(launcher, "configure_generation_config", lambda cfg, tok: cfg)
    monkeypatch.setattr(
        launcher,
        "setup_response_data",
        lambda *a, **kw: (object(), None, train_envs, validation_envs),
    )
    monkeypatch.setattr(
        launcher,
        "setup",
        lambda *a, **kw: (
            object(),
            object(),
            gym,
            object(),
            object(),
            None,
            object(),
            object(),
            object(),
            object(),
            cfg,
            {},
            {},
        ),
    )
    calls = []
    monkeypatch.setattr(launcher, "async_grpo_train", lambda **kw: calls.append(kw))
    launcher.main()
    assert len(calls) == 1
    call = calls[0]
    assert call["task_to_env"] is train_envs
    assert call["val_task_to_env"] is validation_envs
    assert call["processor"] is processor
    assert call["task_to_env"]["math"] is other_env
    if gym_enabled:
        assert call["task_to_env"]["nemo_gym"] is gym
        if validation_enabled:
            assert call["val_task_to_env"]["nemo_gym"] is gym
    else:
        assert "nemo_gym" not in call["task_to_env"]
        if validation_enabled:
            assert "nemo_gym" not in call["val_task_to_env"]
