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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, get_type_hints

import pytest
from omegaconf import OmegaConf
from typing_extensions import NotRequired

from nemo_rl.algorithms.dpo import DPOConfig
from nemo_rl.algorithms.grpo import GRPOConfig, GRPOLoggerConfig
from nemo_rl.algorithms.sft import SFTConfig
from nemo_rl.data import DataConfig
from nemo_rl.distributed.virtual_cluster import ClusterConfig
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.config import load_config_with_inheritance
from nemo_rl.utils.logger import LoggerConfig


def get_keys_from_typeddict(typed_dict_class: dict) -> Set[str]:
    """Extract required keys from a TypedDict class, excluding NotRequired fields."""
    type_hints = get_type_hints(typed_dict_class, include_extras=True)
    required_keys = set()
    optional_keys = set()

    for key, annotation in type_hints.items():
        # Check if the field is marked as NotRequired
        if hasattr(annotation, "__origin__") and (annotation.__origin__ is NotRequired):
            optional_keys.add(key)

        ## check for Optional fields
        elif (
            hasattr(annotation, "__origin__")
            and annotation.__origin__ is Union
            and type(None) in annotation.__args__
        ):
            raise ValueError(
                f"Please use the NotRequired annotation instead of Optional for key {key}"
            )
        else:
            required_keys.add(key)

    return required_keys, optional_keys


def validate_nested_config_section(
    config_dict: Dict[str, Any], config_class: Type, section_path: str
) -> List[str]:
    """Recursively validate a config section and its nested TypedDict fields."""
    errors = []
    type_hints = get_type_hints(config_class, include_extras=True)

    for key, annotation in type_hints.items():
        current_path = f"{section_path}.{key}" if section_path else key

        # Check if the field is marked as NotRequired
        is_optional = hasattr(annotation, "__origin__") and (
            annotation.__origin__ is NotRequired
        )

        # If the key is not in the config and it's required, add an error
        if key not in config_dict:
            if not is_optional:
                errors.append(f"Missing required key in {section_path}: {key}")
            continue

        # Get the value from the config
        value = config_dict[key]

        # If the annotation is a TypedDict (nested config), validate it recursively
        if hasattr(annotation, "__annotations__") and isinstance(value, dict):
            # This is a nested TypedDict, validate it recursively
            nested_errors = validate_nested_config_section(
                value, annotation, current_path
            )
            errors.extend(nested_errors)
        elif hasattr(annotation, "__origin__") and annotation.__origin__ is Optional:
            # Handle Optional[TypedDict] case
            if (
                value is not None
                and hasattr(annotation.__args__[0], "__annotations__")
                and isinstance(value, dict)
            ):
                nested_errors = validate_nested_config_section(
                    value, annotation.__args__[0], current_path
                )
                errors.extend(nested_errors)

    # Check for extra keys (keys in config that are not in the TypedDict)
    required_keys, optional_keys = get_keys_from_typeddict(config_class)
    all_valid_keys = required_keys | optional_keys

    for key in config_dict.keys():
        if key not in all_valid_keys:
            errors.append(f"Extra key in {section_path}: {key}")

    return errors


def validate_config_section(
    config_dict: Dict[str, Any], config_class: dict, section_name: str
) -> List[str]:
    """Validate a specific section of a config against its TypedDict class."""
    errors = []
    required_keys, optional_keys = get_keys_from_typeddict(config_class)

    if section_name not in config_dict:
        errors.append(f"Missing required section: {section_name}")
        return errors

    section_config = config_dict[section_name]
    if not isinstance(section_config, dict):
        errors.append(f"Section {section_name} must be a dictionary")
        return errors

    # Use the new recursive validation function
    nested_errors = validate_nested_config_section(
        section_config, config_class, section_name
    )
    errors.extend(nested_errors)

    return errors


def test_all_config_files_have_required_keys():
    """Test that all config files in examples/configs have all required keys for their respective sections."""
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

    absolute_path = os.path.abspath(__file__)
    configs_dir = Path(
        os.path.join(os.path.dirname(absolute_path), "../../examples/configs")
    )

    # Get all YAML config files
    config_files = glob.glob(str(configs_dir / "**/*.yaml"), recursive=True)

    assert len(config_files) > 0, "No config files found"

    all_errors = []

    for config_file in config_files:
        print(f"\nValidating config file: {config_file}")

        try:
            # Load the config file with inheritance
            config = load_config_with_inheritance(config_file)
            config_dict = OmegaConf.to_container(config, resolve=True)

            if config_dict is None:
                all_errors.append(f"Config file {config_file} is empty or invalid")
                continue

            # Validate each section against its corresponding config class
            section_validations = [
                ("policy", PolicyConfig),
                ("data", DataConfig),
                ("cluster", ClusterConfig),
                ("checkpointing", CheckpointingConfig),
            ]

            # Add algorithm-specific validation
            if "dpo" in config_dict:
                section_validations.extend(
                    [("dpo", DPOConfig), ("logger", LoggerConfig)]
                )
            elif "sft" in config_dict:
                section_validations.extend(
                    [("sft", SFTConfig), ("logger", LoggerConfig)]
                )
            elif "grpo" in config_dict:
                section_validations.extend(
                    [("grpo", GRPOConfig), ("logger", GRPOLoggerConfig)]
                )
                # GRPO also has a loss_fn section
                if "loss_fn" in config_dict:
                    from nemo_rl.algorithms.loss_functions import ClippedPGLossConfig

                    section_validations.append(("loss_fn", ClippedPGLossConfig))
            else:
                warnings.warn(
                    f"Could not determine algorithm type for config {config_file}. Continuing..."
                )
                continue

            # Validate each section
            for section_name, config_class in section_validations:
                errors = validate_config_section(
                    config_dict, config_class, section_name
                )
                for error in errors:
                    all_errors.append(f"{config_file}: {error}")

            # Additional validation for GRPO configs that have an 'env' section
            if "grpo" in config_dict and "env" in config_dict:
                if not isinstance(config_dict["env"], dict):
                    all_errors.append(
                        f"{config_file}: env section must be a dictionary"
                    )

        except Exception as e:
            all_errors.append(f"Error processing {config_file}: {str(e)}")

    # If there are any errors, fail the test with detailed error messages
    if all_errors:
        error_message = "\n".join(all_errors)
        pytest.fail(f"Config validation failed:\n{error_message}")

    print(f"\n✅ Successfully validated {len(config_files)} config files")
