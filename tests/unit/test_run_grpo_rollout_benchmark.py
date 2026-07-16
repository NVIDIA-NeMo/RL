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

import pytest

from examples.nemo_gym import run_grpo_rollout_benchmark
from nemo_rl.utils.config import register_omegaconf_resolvers


def _load_workplace_config():
    register_omegaconf_resolvers()
    repo_root = Path(__file__).parents[2]
    config_path = (
        repo_root
        / "examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml"
    )
    return run_grpo_rollout_benchmark.load_grpo_config(str(config_path), [])


def test_convert_grpo_to_eval_config_inherits_rollout_settings() -> None:
    grpo_config = _load_workplace_config()
    grpo_config.grpo["num_generations_per_prompt"] = 4
    grpo_config.grpo["num_prompts_per_step"] = 4
    grpo_config.grpo["seed"] = 123
    grpo_config.policy["generation"]["temperature"] = 0.7
    grpo_config.policy["generation"]["top_p"] = 0.9

    eval_config = run_grpo_rollout_benchmark.convert_grpo_to_eval_config(grpo_config)

    assert eval_config.eval == {
        "metric": "mean_reward",
        "num_tests_per_prompt": 4,
        "seed": 123,
        "k_value": 1,
        "save_path": None,
    }
    assert (
        eval_config.generation["backend"] == grpo_config.policy["generation"]["backend"]
    )
    assert eval_config.generation["model_name"] == grpo_config.policy["model_name"]
    assert eval_config.generation["num_prompts_per_step"] == 4
    assert eval_config.generation["temperature"] == 0.7
    assert eval_config.generation["top_p"] == 0.9
    assert eval_config.generation["top_k"] is None
    for key, value in grpo_config.policy["generation"]["vllm_cfg"].items():
        assert eval_config.generation["vllm_cfg"][key] == value
    assert eval_config.generation["vllm_cfg"]["enable_vllm_metrics_logger"] is True
    assert eval_config.generation["vllm_cfg"]["vllm_metrics_logger_interval"] == 0.5
    assert eval_config.data.dataset_name == "NemoGymDataset"
    assert eval_config.data.data_path == grpo_config.data["validation"]["data_path"]
    assert eval_config.env["should_use_nemo_gym"] is True
    converted_gym_config = eval_config.env["nemo_gym"].model_dump()
    assert "is_trajectory_collection" not in converted_gym_config
    assert "trajectory_collection_batch_size" not in converted_gym_config
    assert eval_config.logger["wandb"] == grpo_config.logger["wandb"]
    assert eval_config.cluster == grpo_config.cluster
    assert eval_config.policy is not None
    assert eval_config.policy["generation"] == eval_config.generation


def test_convert_grpo_to_eval_config_requires_one_validation_dataset() -> None:
    grpo_config = _load_workplace_config()
    grpo_config.data["validation"] = None
    with pytest.raises(ValueError, match="requires data.validation"):
        run_grpo_rollout_benchmark.convert_grpo_to_eval_config(grpo_config)

    grpo_config = _load_workplace_config()
    grpo_config.data["validation"] = [{"data_path": "a"}, {"data_path": "b"}]
    with pytest.raises(ValueError, match="exactly one"):
        run_grpo_rollout_benchmark.convert_grpo_to_eval_config(grpo_config)


def test_convert_grpo_to_eval_config_preserves_megatron_backend() -> None:
    grpo_config = _load_workplace_config()
    grpo_config.policy["generation"]["backend"] = "megatron"
    grpo_config.policy["generation"]["mcore_generation_config"] = {
        "async_engine": True,
        "expose_http_server": True,
    }

    eval_config = run_grpo_rollout_benchmark.convert_grpo_to_eval_config(grpo_config)

    assert eval_config.generation["backend"] == "megatron"
    assert eval_config.generation["mcore_generation_config"] == {
        "async_engine": True,
        "expose_http_server": True,
    }
    assert eval_config.policy is not None
    assert eval_config.policy["generation"] == eval_config.generation


def test_main_converts_then_runs_standard_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    grpo_config = object()
    eval_config = object()
    load_grpo_config = MagicMock(return_value=grpo_config)
    convert_config = MagicMock(return_value=eval_config)
    run_eval = MagicMock()
    monkeypatch.setattr(
        run_grpo_rollout_benchmark,
        "parse_args",
        lambda: (SimpleNamespace(config="source.yaml"), ["grpo.seed=7"]),
    )
    monkeypatch.setattr(
        run_grpo_rollout_benchmark, "load_grpo_config", load_grpo_config
    )
    monkeypatch.setattr(
        run_grpo_rollout_benchmark, "convert_grpo_to_eval_config", convert_config
    )
    monkeypatch.setattr(run_grpo_rollout_benchmark, "run_eval", run_eval)

    run_grpo_rollout_benchmark.main()

    load_grpo_config.assert_called_once_with("source.yaml", ["grpo.seed=7"])
    convert_config.assert_called_once_with(grpo_config)
    run_eval.assert_called_once_with(eval_config)
