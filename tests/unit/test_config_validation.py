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
from nemo_rl.utils.config import load_config_with_inheritance

# All tests in this module should run first
pytestmark = pytest.mark.run_first

if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

if not OmegaConf.has_resolver("max"):
    OmegaConf.register_new_resolver("max", lambda a, b: max(a, b))


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
