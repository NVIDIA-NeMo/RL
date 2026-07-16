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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from examples import run_eval
from nemo_rl.utils.config import load_config


def test_main_wires_nemo_gym_rollout_only_eval(monkeypatch) -> None:
    repo_root = Path(__file__).parents[3]
    config_path = (
        repo_root
        / "examples/nemo_gym/eval_workplace_assistant_nemotron_nano_v2_9b.yaml"
    )
    loaded_config = load_config(str(config_path))
    monkeypatch.setattr(
        run_eval,
        "parse_args",
        lambda: (SimpleNamespace(config=str(config_path)), OmegaConf.create({})),
    )
    monkeypatch.setattr(run_eval, "load_config", lambda _: loaded_config)
    monkeypatch.setattr(run_eval, "init_ray", MagicMock())

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    monkeypatch.setattr(run_eval, "get_tokenizer", MagicMock(return_value=tokenizer))

    dataset = object()
    monkeypatch.setattr(
        run_eval,
        "setup_data",
        MagicMock(return_value=(dataset, None, tokenizer)),
    )
    generation = object()
    dataloader = object()
    captured = {}

    def fake_setup(config, configured_tokenizer, configured_dataset):
        captured["config"] = config
        assert configured_tokenizer is tokenizer
        assert configured_dataset is dataset
        return generation, dataloader, config

    monkeypatch.setattr(run_eval, "setup", fake_setup)
    nemo_gym = object()
    setup_nemo_gym = MagicMock(return_value=nemo_gym)
    monkeypatch.setattr(run_eval, "setup_nemo_gym_environment", setup_nemo_gym)

    logger = MagicMock()
    logger_class = MagicMock(return_value=logger)
    monkeypatch.setattr(run_eval, "Logger", logger_class)
    run_env_eval = MagicMock()
    monkeypatch.setattr(run_eval, "run_env_eval", run_env_eval)

    run_eval.main()

    config = captured["config"]
    assert config.generation["vllm_cfg"]["async_engine"] is True
    assert config.generation["vllm_cfg"]["expose_http_server"] is True
    assert config.generation["vllm_cfg"]["enable_vllm_metrics_logger"] is True
    assert config.generation["stop_strings"] is None
    assert config.generation["stop_token_ids"] is None
    setup_nemo_gym.assert_called_once_with(generation, config)
    logger_class.assert_called_once_with(config.logger)
    logger.log_hyperparams.assert_called_once_with(config.model_dump())
    run_env_eval.assert_called_once_with(
        generation,
        dataloader,
        nemo_gym,
        config,
        tokenizer=tokenizer,
        logger=logger,
    )
    logger.close.assert_called_once_with()
