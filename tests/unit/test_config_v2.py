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

"""Test config v1(typing.TypedDict) -> v2(pydantic.BaseModel) changes.

This test verifies that the config v1(typing.TypedDict) -> v2(pydantic.BaseModel) changes won't introduce any new default values.
This test file and `tests/unit/reference_configs` will be removed once we completely migrate from v1 to v2.
"""

import glob
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.distillation import MasterConfig as DistillationMasterConfig
from nemo_rl.algorithms.dpo import MasterConfig as DPOMasterConfig
from nemo_rl.algorithms.grpo import MasterConfig as GRPOMasterConfig
from nemo_rl.algorithms.rm import MasterConfig as RMMasterConfig
from nemo_rl.algorithms.sft import MasterConfig as SFTMasterConfig
from nemo_rl.evals.eval import MasterConfig as EvalMasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

# # All tests in this module should run first
# pytestmark = pytest.mark.run_first

register_omegaconf_resolvers()


absolute_dir = os.path.dirname(os.path.abspath(__file__))
real_configs_dir = Path(os.path.join(absolute_dir, "../../examples/configs"))
reference_configs_dir = Path(os.path.join(absolute_dir, "reference_configs"))
reference_config_files = glob.glob(
    str(reference_configs_dir / "*.yaml"), recursive=True
)


def _collect_mismatched_keys(real: dict, reference: dict, path: str = ""):
    """Return keys present in real but absent in reference, and keys with differing values.

    Returns:
        missing: keys in real but not in reference.
        different: keys in both but with different values (leaf nodes only).
    """
    missing = []
    different = []
    for key, value in real.items():
        full_key = f"{path}.{key}" if path else key
        if key not in reference:
            missing.append(full_key)
        elif isinstance(value, dict) and isinstance(reference[key], dict):
            sub_missing, sub_different = _collect_mismatched_keys(
                value, reference[key], full_key
            )
            missing.extend(sub_missing)
            different.extend(sub_different)
        elif value != reference[key]:
            different.append(
                f"{full_key}: real={value!r}, reference={reference[key]!r}"
            )
    return missing, different


@pytest.mark.parametrize("reference_config_file", reference_config_files)
def test_reference_configs_up_to_date(reference_config_file):
    """Test that the reference config is up to date."""

    print(f"\nValidating config file: {reference_config_file}")

    if "/eval.yaml" in reference_config_file:
        real_config_file = (
            real_configs_dir / "evals" / os.path.basename(reference_config_file)
        )
    else:
        real_config_file = real_configs_dir / os.path.basename(reference_config_file)

    reference_config = load_config(reference_config_file)
    reference_config_dict = OmegaConf.to_container(reference_config, resolve=True)
    real_config = load_config(real_config_file)
    real_config_dict = OmegaConf.to_container(real_config, resolve=True)

    missing, different = _collect_mismatched_keys(
        real_config_dict, reference_config_dict
    )
    assert len(missing) == 0, (
        f"\nKeys present in real config {real_config_file} but missing from reference config {reference_config_file}:\n"
        + "\n".join(f"  - {k}" for k in missing)
        + f"\nPlease add the missing keys to {reference_config_file}.",
    )
    assert len(different) == 0, (
        f"\nKeys present in both real config {real_config_file} and reference config {reference_config_file} but with different values:\n"
        + "\n".join(f"  - {k}" for k in different)
        + f"\nPlease update the values in {reference_config_file} to match the values in {real_config_file}.",
    )


@pytest.mark.parametrize("config_file", reference_config_files)
def test_config_v2_same_as_v1(config_file):
    """Test that the config v2's behavior is the same as the config v1."""

    print(f"\nValidating config file: {config_file}")

    config = load_config(config_file)
    config_v1 = OmegaConf.to_container(config, resolve=True)

    if "eval" in config_v1:
        master_config_class = EvalMasterConfig
    elif "distillation" in config_v1:
        master_config_class = DistillationMasterConfig
    elif "dpo" in config_v1:
        master_config_class = DPOMasterConfig
    elif "sft" in config_v1:
        master_config_class = SFTMasterConfig
    elif "grpo" in config_v1:
        master_config_class = GRPOMasterConfig
    elif "rm" in config_v1:
        master_config_class = RMMasterConfig

    config_v2 = master_config_class(**config_v1)
    config_v2 = config_v2.model_dump()

    # Check v1 keys missing from v2, and differing values
    missing_in_v2, different = _collect_mismatched_keys(config_v1, config_v2)
    # Check v2 keys missing from v1 (new defaults introduced by Pydantic model)
    missing_in_v1, _ = _collect_mismatched_keys(config_v2, config_v1)

    # Check new defaults introduced by Pydantic model
    assert len(missing_in_v1) == 0, (
        f"\nKeys present in v2 config but missing from v1 ({config_file}) — Pydantic introduced new defaults:\n"
        + "\n".join(f"  - {k}" for k in missing_in_v1)
        + f"\nPlease make sure the code behavior is not changed and add the missing keys to {config_file} to pass the test.",
    )
    # Theoretically, the two things below shouldn't happen, but let's check them anyway.
    assert len(missing_in_v2) == 0, (
        f"\nKeys present in v1 config but missing from v2 ({config_file}):\n"
        + "\n".join(f"  - {k}" for k in missing_in_v2)
    )
    assert len(different) == 0, (
        f"\nKeys present in both v1 and v2 but with different values ({config_file}):\n"
        + "\n".join(f"  - {k}" for k in different)
    )
