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

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

from examples import run_grpo_single_controller


def test_main_configures_generation_for_trained_mtp(monkeypatch) -> None:
    generation_config = {"backend": "vllm"}
    config = SimpleNamespace(
        policy={
            "tokenizer": {},
            "generation": generation_config,
            "draft": {"enabled": False},
            "megatron_cfg": {"mtp_num_layers": 2},
        },
        data_plane={"enabled": True},
        logger={"log_dir": "/tmp/logs"},
        checkpointing={"enabled": False},
    )
    configured_generation = {"backend": "vllm", "_mtp_weights_from_refit": True}
    configure_generation = MagicMock(return_value=configured_generation)
    actor = SimpleNamespace(run=SimpleNamespace(remote=MagicMock(return_value="run")))
    actor_args = SimpleNamespace(env_handles={})

    monkeypatch.setattr(
        run_grpo_single_controller,
        "parse_args",
        lambda: (Namespace(config="config.yaml"), []),
    )
    monkeypatch.setattr(run_grpo_single_controller, "load_config", lambda _: {})
    monkeypatch.setattr(
        run_grpo_single_controller.OmegaConf,
        "to_container",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(run_grpo_single_controller, "MasterConfig", lambda **_: config)
    monkeypatch.setattr(run_grpo_single_controller, "init_ray", lambda: None)
    monkeypatch.setattr(
        run_grpo_single_controller,
        "get_tokenizer",
        lambda _: "tokenizer",
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "get_next_experiment_dir",
        lambda _: "/tmp/logs/0",
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "configure_generation_config",
        configure_generation,
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "setup_single_controller",
        lambda *_args: actor_args,
    )
    monkeypatch.setattr(
        run_grpo_single_controller.SingleControllerActor,
        "remote",
        MagicMock(return_value=actor),
    )
    monkeypatch.setattr(run_grpo_single_controller.ray, "get", lambda _: {})

    run_grpo_single_controller.main()

    configure_generation.assert_called_once_with(
        generation_config,
        "tokenizer",
        has_refit_draft_weights=False,
        trains_mtp=True,
    )
    assert config.policy["generation"] is configured_generation
