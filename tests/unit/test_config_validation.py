# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import os
from pathlib import Path
from typing import Any, Dict, Type

import pytest
from omegaconf import OmegaConf
from pydantic import TypeAdapter, ValidationError

from nemo_rl.algorithms.distillation import MasterConfig as DistillationMasterConfig
from nemo_rl.algorithms.dpo import MasterConfig as DPOMasterConfig
from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.algorithms.rm import MasterConfig as RMMasterConfig
from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig
from nemo_rl.evals.eval import MasterConfig as EvalMasterConfig
from nemo_rl.utils.config import (
    load_config_with_inheritance,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

# All tests in this module should run first
pytestmark = pytest.mark.run_first

register_omegaconf_resolvers()


def validate_config_section(
    section_config: Dict[str, Any],
    config_class: Type,
    config_file: str,
) -> None:
    """Validate a config section against its TypedDict class using Pydantic.

    Raises AssertionError with formatted error messages if validation fails.
    """
    if not isinstance(section_config, dict):
        raise TypeError("Config must be a dictionary")

    # Use Pydantic's TypeAdapter to validate the TypedDict
    adapter = TypeAdapter(config_class)
    try:
        adapter.validate_python(section_config)
    except ValidationError as e:
        # Format errors nicely with actual values
        error_messages = []
        for error in e.errors():
            path_parts = []
            if error["loc"]:
                path_parts.extend(str(loc) for loc in error["loc"])
            path = ".".join(path_parts) if path_parts else "root"

            # Only include the actual input value for non-missing fields
            # For missing fields, the 'input' is the parent dict which is confusing
            input_info = ""
            if "input" in error and error["type"] != "missing":
                input_value = error.get("input")
                # Truncate very long values for readability
                input_str = str(input_value)
                if len(input_str) > 100:
                    input_str = input_str[:97] + "..."
                input_info = f" (got: {input_str})"

            error_messages.append(
                f"  {path}: {error['msg']} (type={error['type']}){input_info}"
            )

        config_info = f"\n\nConfig file: {config_file}" if config_file else ""
        raise AssertionError(
            f"Config validation failed:{config_info}\n" + "\n".join(error_messages)
        ) from e


absolute_path = os.path.abspath(__file__)
configs_dir = Path(
    os.path.join(os.path.dirname(absolute_path), "../../examples/configs")
).resolve()
config_files = glob.glob(str(configs_dir / "**/*.yaml"), recursive=True)
assert len(config_files) > 0, "No config files found"


def test_distillation_entropy_loss_overrides_are_declared():
    """Entropy-aware distillation knobs should work with normal dot overrides."""
    config = load_config_with_inheritance(str(configs_dir / "distillation_math.yaml"))

    config = parse_hydra_overrides(
        config,
        [
            "loss_fn.sft_weight=0.0",
            "loss_fn.entropy_diagnostics.enabled=True",
            "loss_fn.entropy_diagnostics.teacher_entropy_low=0.25",
            "loss_fn.entropy_diagnostics.teacher_entropy_high=0.75",
            "loss_fn.entropy_diagnostics.log_position_buckets=False",
            "loss_fn.entropy_aware.enabled=True",
            "loss_fn.entropy_aware.teacher_entropy_threshold=0.8",
            "loss_fn.entropy_aware.weighting_mode=linear_ramp",
            "loss_fn.entropy_aware.ramp_start=0.3",
            "loss_fn.entropy_aware.ramp_width=0.6",
            "loss_fn.entropy_aware.forward_kl_weight=1.0",
            "loss_fn.entropy_aware.high_entropy_reverse_kl_weight=0.5",
            "loss_fn.entropy_aware.high_entropy_reverse_kl_threshold=0.7",
            "loss_fn.entropy_aware.low_entropy_sharpening_enabled=True",
            "loss_fn.entropy_aware.low_entropy_sharpening_weight=0.03",
            "loss_fn.entropy_aware.low_entropy_sharpening_temperature=0.9",
            "loss_fn.entropy_aware.low_entropy_sharpening_support_ratio_threshold=1.0",
            "loss_fn.entropy_aware.low_entropy_sharpening_requires_support_ratio_above_threshold=True",
            "loss_fn.entropy_aware.require_sft_weight_zero=True",
            "loss_fn.entropy_aware.require_zero_outside_topk_false=True",
        ],
    )

    assert config.loss_fn.entropy_diagnostics.enabled is True
    assert config.loss_fn.entropy_diagnostics.log_position_buckets is False
    assert config.loss_fn.entropy_aware.enabled is True
    assert config.loss_fn.entropy_aware.weighting_mode == "linear_ramp"
    assert config.loss_fn.entropy_aware.high_entropy_reverse_kl_weight == 0.5
    assert config.loss_fn.entropy_aware.low_entropy_sharpening_enabled is True


@pytest.mark.parametrize("config_file", config_files)
def test_all_config_files_have_required_keys(config_file):
    """Test that all config files in examples/configs have all required keys for their respective sections."""

    print(f"\nValidating config file: {config_file}")

    # Load the config file with inheritance
    config = load_config_with_inheritance(config_file)
    config_dict = OmegaConf.to_container(config, resolve=True)

    if config_dict is None:
        raise AssertionError(f"Config file {config_file} is empty or invalid")

    # Determine which MasterConfig to use based on the config contents
    master_config_class = None
    config_type = None

    if "/evals/" in config_file:
        master_config_class = EvalMasterConfig
        config_type = "eval"
    elif "distillation" in config_dict:
        master_config_class = DistillationMasterConfig
        config_type = "distillation"
    elif "dpo" in config_dict:
        master_config_class = DPOMasterConfig
        config_type = "dpo"
    elif "sft" in config_dict:
        master_config_class = SFTMasterConfig
        config_type = "sft"
    elif "grpo" in config_dict:
        master_config_class = GRPOMasterConfig
        config_type = "grpo"
    elif "rm" in config_dict:
        master_config_class = RMMasterConfig
        config_type = "rm"
    else:
        raise AssertionError(
            f"Could not determine algorithm type for config {config_file}."
        )

    # Validate the entire config using the appropriate MasterConfig
    validate_config_section(config_dict, master_config_class, config_file)


@pytest.mark.parametrize("config_file", config_files)
def test_all_config_no_tp_size_accuracy_issues(config_file):
    """Test that all config files in examples/configs have no TP size >= 4 accuracy issues.

    There is a known batch-variant accuracy issue with TP>=4 for both DTensor and Megatron backend.
    Related document: https://docs.nvidia.com/nemo/rl/latest/guides/dtensor-tp-accuracy.html#root-cause.
    """

    skip_config_files = [
        "grpo-qwen3-30ba3b-4n8g-40K.yaml",
        "grpo-qwen3-30ba3b-8n8g-megatron.yaml",
        "grpo-qwen3-32b-4n8g.yaml",
        "grpo-qwen3-32b-8n8g-async-1off.yaml",
        "grpo-gemma3-27b-it-8n8g-fsdp2tp8-actckpt-long.yaml",
        "grpo-gemma3-27b-it-8n4g-fsdp2tp4-actckpt-long.yaml",
    ]
    if os.path.basename(config_file) in skip_config_files:
        pytest.skip(
            f"Skipping config file {config_file} because it sets NRL_IGNORE_TP_ACCURACY_CHECK=1"
        )

    print(f"\nValidating config file: {config_file}")

    # Load the config file with inheritance
    config = load_config_with_inheritance(config_file)
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Skip if config does not have policy or logprob_batch_size
    if "policy" not in config_dict or "logprob_batch_size" not in config_dict["policy"]:
        return

    # Skip if config set force_on_policy_ratio to True
    if "loss_fn" in config_dict and config_dict["loss_fn"].get(
        "force_on_policy_ratio", False
    ):
        return

    # Check if TP size >= 4 and train_micro_batch_size != logprob_batch_size
    if config_dict["policy"]["megatron_cfg"]["enabled"]:
        tp_size = config_dict["policy"]["megatron_cfg"]["tensor_model_parallel_size"]
    else:
        tp_size = config_dict["policy"]["dtensor_cfg"]["tensor_parallel_size"]

    train_micro_bs = config_dict["policy"]["train_micro_batch_size"]
    logprob_bs = config_dict["policy"]["logprob_batch_size"]

    if tp_size >= 4 and train_micro_bs != logprob_bs:
        raise AssertionError(
            f"Config file {config_file} has TP size >= 4 accuracy issues. "
            "Please set policy.train_micro_batch_size and policy.logprob_batch_size to be the same value."
        )
