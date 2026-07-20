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

import argparse
import os
import sys
from copy import deepcopy
from typing import Any, cast

# Allow running this file directly (e.g. `uv run examples/nemo_gym/run_grpo_rollout_benchmark.py`):
# put the repo root on sys.path so the `from examples...` / `from nemo_rl...` imports below resolve
# even when Python placed this file's own directory on sys.path[0] instead of the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf

from examples.run_eval import run_eval
from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.evals.eval import MasterConfig as EvalMasterConfig
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

ROLLOUT_BENCHMARK_METRIC = "mean_reward"
ROLLOUT_BENCHMARK_K_VALUE = 1


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse an RL recipe path and overrides applied before conversion."""
    parser = argparse.ArgumentParser(
        description="Run an RL recipe as a rollout benchmark through eval"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the source GRPO YAML"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def load_grpo_config(config_path: str, overrides: list[str]) -> GRPOMasterConfig:
    """Load and validate the source GRPO recipe before converting it."""
    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    resolved_config = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved_config, dict):
        raise TypeError("The GRPO recipe must resolve to a mapping")
    return GRPOMasterConfig.model_validate(resolved_config)


def _convert_validation_data(grpo_config: GRPOMasterConfig) -> dict[str, Any]:
    """Merge RL validation data with its defaults into eval's flat data block."""
    validation_config = grpo_config.data.get("validation")
    if validation_config is None:
        raise ValueError("Rollout benchmark requires data.validation")
    if isinstance(validation_config, list):
        raise ValueError(
            "Rollout benchmark requires exactly one data.validation dataset"
        )

    default_config = grpo_config.data.get("default")
    if default_config is None:
        raise ValueError("Rollout benchmark requires data.default")

    data_config = cast(dict[str, Any], deepcopy(default_config))
    data_config.update(cast(dict[str, Any], deepcopy(validation_config)))
    data_config["max_input_seq_length"] = grpo_config.data["max_input_seq_length"]
    if data_config.get("dataset_name") != "NemoGymDataset":
        raise ValueError(
            "NeMo Gym rollout benchmark requires "
            "data.default.dataset_name=NemoGymDataset"
        )
    return data_config


def convert_grpo_to_eval_config(
    grpo_config: GRPOMasterConfig,
) -> EvalMasterConfig:
    """Convert an RL recipe into the standard eval configuration schema."""
    if not grpo_config.env.get("should_use_nemo_gym"):
        raise ValueError(
            "NeMo Gym rollout benchmark requires env.should_use_nemo_gym=true"
        )
    nemo_gym_config = grpo_config.env.get("nemo_gym")
    if not isinstance(nemo_gym_config, dict):
        raise ValueError("NeMo Gym rollout benchmark requires env.nemo_gym")

    source_generation_config = grpo_config.policy.get("generation")
    if source_generation_config is None:
        raise ValueError("Rollout benchmark requires policy.generation")
    generation_config = cast(dict[str, Any], deepcopy(source_generation_config))
    generation_config["model_name"] = grpo_config.policy["model_name"]
    generation_config["num_prompts_per_step"] = grpo_config.grpo["num_prompts_per_step"]
    if generation_config["backend"] == "vllm":
        vllm_config = generation_config["vllm_cfg"]
        vllm_config.setdefault("enable_vllm_metrics_logger", True)
        vllm_config.setdefault("vllm_metrics_logger_interval", 0.5)

    converted_nemo_gym_config = deepcopy(nemo_gym_config)
    # These select the legacy collect_trajectories path in the source entrypoint;
    # benchmark batching is derived exclusively from the GRPO step configuration.
    converted_nemo_gym_config.pop("is_trajectory_collection", None)
    converted_nemo_gym_config.pop("trajectory_collection_batch_size", None)

    policy_config = deepcopy(grpo_config.policy)
    policy_config["generation"] = cast(GenerationConfig, generation_config)
    converted_config: dict[str, Any] = {
        "eval": {
            "metric": ROLLOUT_BENCHMARK_METRIC,
            "num_tests_per_prompt": grpo_config.grpo["num_generations_per_prompt"],
            "seed": grpo_config.grpo["seed"],
            "k_value": ROLLOUT_BENCHMARK_K_VALUE,
            "save_path": None,
        },
        "generation": generation_config,
        "tokenizer": deepcopy(grpo_config.policy["tokenizer"]),
        "data": _convert_validation_data(grpo_config),
        "env": {
            "should_use_nemo_gym": True,
            "nemo_gym": converted_nemo_gym_config,
        },
        "logger": deepcopy(grpo_config.logger),
        "cluster": deepcopy(grpo_config.cluster),
        "policy": policy_config,
    }
    return EvalMasterConfig.model_validate(converted_config)


def main() -> None:
    """Convert an RL recipe and run it through the standard eval entrypoint."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if args.config is None:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "grpo_workplace_assistant_nemotron_nano_v2_9b.yaml",
        )

    grpo_config = load_grpo_config(args.config, overrides)
    eval_config = convert_grpo_to_eval_config(grpo_config)
    print(f"Loaded GRPO configuration from: {args.config}")
    print("Converted GRPO configuration to the standard eval schema")
    run_eval(eval_config)


if __name__ == "__main__":
    main()
