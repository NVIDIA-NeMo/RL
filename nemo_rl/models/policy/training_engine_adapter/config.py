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

"""Configuration of the Megatron-Bridge TrainingEngine policy path (``policy.training_engine``)."""

from typing import Any, TypedDict

import ray

from nemo_rl.models.policy.training_engine_adapter.worker_environment import (
    TRAINING_ENGINE_WORKER_FQN,
    register_worker_environment,
)
from nemo_rl.utils.venvs import make_actor_runtime_env

# megatron_cfg fields that describe the parallel layout. With policy.training_engine the
# recipe decides them.
_LAYOUT_FIELDS = (
    "tensor_model_parallel_size",
    "context_parallel_size",
    "pipeline_model_parallel_size",
    "sequence_parallel",
)


class TrainingEngineConfig(TypedDict):
    # Recipe function in megatron.bridge.engine.recipes and its keyword arguments.
    recipe: str
    recipe_kwargs: dict[str, Any]
    # Megatron or HF checkpoint that initializes the policy and the reference model.
    checkpoint_path: str
    # Packed tokens per microbatch.
    seq_length: int
    # Recipe fields to change, by dotted path (for example "optimizer.clip_grad").
    overrides: dict[str, Any]


@ray.remote(num_cpus=1)
def _recipe_layout(engine_config: TrainingEngineConfig) -> tuple[dict[str, Any], int]:
    """Build the recipe and return its parallel layout.

    Runs as a task in the engine worker's environment: the driver's cannot import the
    engine.

    Returns:
        The recipe's value for each of the megatron_cfg layout fields, and the multiple
        every sample's length must be padded to for that layout.
    """
    from megatron.bridge.engine.batching import sequence_pad_multiple

    from nemo_rl.models.policy.training_engine_adapter.engine import build_recipe

    model = build_recipe(engine_config).model
    layout = {
        "tensor_model_parallel_size": model.tensor_model_parallel_size,
        "context_parallel_size": model.context_parallel_size or 1,
        "pipeline_model_parallel_size": model.pipeline_model_parallel_size,
        "sequence_parallel": bool(model.sequence_parallel),
    }
    sample_pad_multiple = sequence_pad_multiple(
        cp_size=layout["context_parallel_size"],
        tp_size=layout["tensor_model_parallel_size"],
        sequence_parallel=layout["sequence_parallel"],
    )
    return layout, sample_pad_multiple


def align_policy_config_with_recipe(policy_config: dict[str, Any]) -> None:
    """Make the policy config describe the engine recipe, in place, then check the rest.

    The one place where the policy config and the recipe are brought together. NeMo RL's
    driver lays the workers out and plans the packed microbatches from the policy
    config; the engine runs the recipe. Everything the recipe decides is written into
    the policy config here, before the driver reads it, so the two cannot disagree.
    What is left is checked: settings that are the user's choice and can be
    incompatible with the recipe, and features this path does not implement.
    """
    register_worker_environment()
    layout, sample_pad_multiple = ray.get(
        _recipe_layout.options(
            runtime_env=make_actor_runtime_env(TRAINING_ENGINE_WORKER_FQN)
        ).remote(policy_config["training_engine"])
    )
    apply_recipe_layout(policy_config, layout, sample_pad_multiple)


def apply_recipe_layout(
    policy_config: dict[str, Any], layout: dict[str, Any], sample_pad_multiple: int
) -> None:
    """Write the recipe's layout into the policy config and check what it cannot decide.

    Args:
        policy_config: The policy config, changed in place.
        layout: The recipe's value for each megatron_cfg layout field.
        sample_pad_multiple: The multiple the layout needs every sample padded to.
    """
    engine_config = policy_config["training_engine"]
    megatron_cfg = policy_config["megatron_cfg"]
    packing = policy_config["sequence_packing"]
    seq_length = engine_config["seq_length"]

    # Decided by the recipe: the parallel layout, and the token budget of a microbatch.
    _override(
        megatron_cfg, "policy.megatron_cfg", {k: layout[k] for k in _LAYOUT_FIELDS}
    )
    _override(
        packing,
        "policy.sequence_packing",
        {"train_mb_tokens": seq_length, "logprob_mb_tokens": seq_length},
    )

    # Read by the algorithm setup before the policy is built, so they cannot be set here.
    if not megatron_cfg["enabled"]:
        raise ValueError(
            "policy.training_engine requires policy.megatron_cfg.enabled=true: NeMo RL "
            "treats the engine as its Megatron backend."
        )
    if not packing["enabled"]:
        raise ValueError(
            "policy.training_engine requires policy.sequence_packing.enabled=true."
        )
    # Not implemented.
    if layout["pipeline_model_parallel_size"] != 1:
        raise NotImplementedError(
            "policy.training_engine does not support a recipe with pipeline parallelism."
        )
    # The user's choices.
    if "seq_length" in engine_config["recipe_kwargs"]:
        raise ValueError(
            "Set policy.training_engine.seq_length, not "
            "policy.training_engine.recipe_kwargs.seq_length."
        )
    plan_multiple = policy_config["make_sequence_length_divisible_by"]
    if plan_multiple % sample_pad_multiple != 0:
        raise ValueError(
            f"policy.make_sequence_length_divisible_by={plan_multiple} must be a "
            f"multiple of {sample_pad_multiple}, which the recipe's layout needs every "
            "sample padded to."
        )
    if policy_config["max_total_sequence_length"] > seq_length:
        raise ValueError(
            "policy.max_total_sequence_length="
            f"{policy_config['max_total_sequence_length']} exceeds "
            f"policy.training_engine.seq_length={seq_length}: one sample would not "
            "fit in a microbatch."
        )


def _override(section: dict[str, Any], name: str, values: dict[str, Any]) -> None:
    """Set the recipe's values in one config section and report every value that changed."""
    for key, value in values.items():
        if key in section and section[key] != value:
            print(
                f"policy.training_engine: {name}.{key}={section[key]!r} replaced by "
                f"the recipe's {value!r}."
            )
        section[key] = value
