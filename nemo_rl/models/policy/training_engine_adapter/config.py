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


def validate_training_engine_config(policy_config: dict[str, Any]) -> None:
    """Reject policy settings the TrainingEngine path does not honor.

    The engine packs and shards the global batch itself, so the native settings that
    describe packing must agree with the engine's packed-token budget.
    """
    engine_config = policy_config["training_engine"]
    megatron_cfg = policy_config["megatron_cfg"]
    if not megatron_cfg["enabled"]:
        raise ValueError(
            "policy.training_engine requires policy.megatron_cfg.enabled=true; its "
            "tensor, context and pipeline parallel sizes describe the engine's topology."
        )
    if megatron_cfg["pipeline_model_parallel_size"] != 1:
        raise ValueError(
            "policy.training_engine does not support pipeline parallelism."
        )
    if "seq_length" in engine_config["recipe_kwargs"]:
        raise ValueError(
            "Set policy.training_engine.seq_length, not "
            "policy.training_engine.recipe_kwargs.seq_length."
        )
    seq_length = engine_config["seq_length"]
    packing = policy_config["sequence_packing"]
    if not packing["enabled"]:
        raise ValueError(
            "policy.training_engine requires policy.sequence_packing.enabled=true."
        )
    for key in ("train_mb_tokens", "logprob_mb_tokens"):
        if packing[key] != seq_length:
            raise ValueError(
                f"policy.sequence_packing.{key}={packing[key]} must equal "
                f"policy.training_engine.seq_length={seq_length}."
            )
    if policy_config["max_total_sequence_length"] > seq_length:
        raise ValueError(
            "policy.max_total_sequence_length="
            f"{policy_config['max_total_sequence_length']} exceeds "
            f"policy.training_engine.seq_length={seq_length}: one sample would not "
            "fit in a microbatch."
        )
