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

from pydantic import ValidationError
import pytest

from nemo_rl.data.energon.config import EnergonCookerConfig, EnergonTaskEncoderConfig
from nemo_rl.data.energon.multimodal.registry import (
    LazyRegistry,
    selected_registry_identity,
)


@pytest.mark.parametrize("config_type", [EnergonCookerConfig, EnergonTaskEncoderConfig])
@pytest.mark.parametrize("key", ["object_name", "object"])
def test_file_backed_component_accepts_object_name_and_legacy_alias(config_type, key):
    config = config_type.model_validate(
        {"python_file": "/workspace/plugins.py", key: "custom_component"}
    )
    assert config.name is None
    assert config.object_name == "custom_component"
    assert "object" not in config_type.model_fields
    assert config.model_dump(mode="json", by_alias=True)["object"] == "custom_component"
    assert (
        config_type.model_validate(config.model_dump()).object_name
        == "custom_component"
    )


@pytest.mark.parametrize("config_type", [EnergonCookerConfig, EnergonTaskEncoderConfig])
def test_file_backed_component_rejects_duplicate_object_names(config_type):
    with pytest.raises(ValidationError, match="Use only one"):
        config_type.model_validate(
            {
                "python_file": "/workspace/plugins.py",
                "object_name": "custom_component",
                "object": "another_component",
            }
        )


@pytest.mark.parametrize("key", ["object_name", "object"])
def test_file_backed_cooker_loads_from_config_and_preserves_identity(tmp_path, key):
    path = tmp_path / "cooker.py"
    path.write_text("def custom_cooker(sample):\n    return sample\n")
    config = EnergonCookerConfig.model_validate(
        {"python_file": str(path), key: "custom_cooker"}
    )
    registry = LazyRegistry("cooker")
    cooker = registry.resolve_configured(
        name=config.name, python_file=config.python_file, object_name=config.object_name
    )
    assert cooker("sample") == "sample"
    identity = selected_registry_identity(
        task_encoder=EnergonTaskEncoderConfig(), cookers=[config]
    )
    assert identity["cookers"][0]["object"] == "custom_cooker"
