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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nemo_rl.data.energon.config import (
    EnergonCookerConfig,
    EnergonLoaderConfig,
    EnergonTaskEncoderConfig,
)
from nemo_rl.data.energon.multimodal.registry import (
    LazyRegistry,
    selected_registry_identity,
)


@pytest.mark.parametrize("config_type", [EnergonCookerConfig, EnergonTaskEncoderConfig])
def test_file_backed_component_uses_object_name(
    config_type: type[EnergonCookerConfig | EnergonTaskEncoderConfig],
) -> None:
    config = config_type.model_validate(
        {"python_file": "/workspace/plugins.py", "object_name": "custom_component"}
    )
    assert config.name is None
    assert config.object_name == "custom_component"
    assert "object" not in config.model_dump()
    assert config.model_dump(mode="json")["object_name"] == "custom_component"
    assert (
        config_type.model_validate(config.model_dump()).object_name
        == "custom_component"
    )


@pytest.mark.parametrize("config_type", [EnergonCookerConfig, EnergonTaskEncoderConfig])
@pytest.mark.parametrize("include_object_name", [False, True])
def test_file_backed_component_rejects_object_key(
    config_type: type[EnergonCookerConfig | EnergonTaskEncoderConfig],
    include_object_name: bool,
) -> None:
    value = {"python_file": "/workspace/plugins.py", "object": "custom_component"}
    if include_object_name:
        value["object_name"] = "custom_component"
    with pytest.raises(
        ValidationError, match="object is not supported; use object_name"
    ):
        config_type.model_validate(value)


def test_file_backed_cooker_loads_from_config_and_preserves_identity(tmp_path) -> None:
    path = tmp_path / "cooker.py"
    path.write_text("def custom_cooker(sample):\n    return sample\n")
    config = EnergonCookerConfig.model_validate(
        {"python_file": str(path), "object_name": "custom_cooker"}
    )
    cooker = LazyRegistry("cooker").resolve_configured(config)
    assert cooker("sample") == "sample"
    identity = selected_registry_identity(
        task_encoder=EnergonTaskEncoderConfig(), cookers=[config]
    )
    assert identity["cookers"][0]["object_name"] == "custom_cooker"


@pytest.mark.parametrize(
    "options",
    [
        {"gc_collect_every_n_steps": -1},
        {"cache_pool_num_workers": 0},
        {"cache_pool_num_workers": -1},
    ],
)
def test_loader_rejects_invalid_gc_and_cache_overrides(options: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        EnergonLoaderConfig(model_family="qwen", **options)


@pytest.mark.mcore
@pytest.mark.parametrize("split_role", ["train", "validation"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"gc_collect_every_n_steps": None, "cache_pool_num_workers": None},
        {"gc_collect_every_n_steps": 0, "cache_pool_num_workers": 1},
        {"gc_collect_every_n_steps": 100000, "cache_pool_num_workers": 3},
    ],
)
def test_loader_preserves_energon_defaults_unless_overridden(
    monkeypatch: pytest.MonkeyPatch, split_role: str, options: dict[str, int | None]
) -> None:
    # The loader imports the optional Energon runtime.
    from nemo_rl.data.energon import sft_dataloader as module

    config = EnergonLoaderConfig(model_family="qwen", **options)
    monkeypatch.setattr(module, "_loader_config", lambda _: config)
    monkeypatch.setattr(
        module,
        "build_processor_adapter",
        Mock(return_value=SimpleNamespace(fingerprint="adapter")),
    )
    monkeypatch.setattr(
        module,
        "_task_encoder",
        Mock(return_value=SimpleNamespace(cookers=[SimpleNamespace(need_cache=True)])),
    )
    monkeypatch.setattr(module, "_worker_config", Mock())
    monkeypatch.setattr(module, "get_train_dataset", Mock())
    monkeypatch.setattr(module, "get_val_dataset", Mock())
    cache_pool, savable_loader = Mock(), Mock()
    monkeypatch.setattr(module, "FileStoreCachePool", cache_pool)
    monkeypatch.setattr(module, "get_savable_loader", savable_loader)

    module.build_energon_sft_loader(
        data_config={"energon": config, "shuffle": True},
        source={
            "path": "/data/prepared",
            "split": split_role,
            "virtual_epoch_length": 8,
        },
        processor=SimpleNamespace(tokenizer=object()),
        batch_size=1,
        max_sequence_length=128,
        split_role=split_role,
        logical_rank=0,
        logical_world_size=1,
        placement_fingerprint="placement",
        only_unmask_final=False,
        sequence_length_pad_multiple=1,
    )

    expected_pool = {"method": "raw"}
    if config.cache_pool_num_workers is not None:
        expected_pool["num_workers"] = config.cache_pool_num_workers
    cache_pool.assert_called_once_with(**expected_pool)
    assert savable_loader.call_count == 1
    kwargs = savable_loader.call_args.kwargs
    assert kwargs["cache_pool"] is cache_pool.return_value
    if config.gc_collect_every_n_steps is None:
        assert "gc_collect_every_n_steps" not in kwargs
    else:
        assert kwargs["gc_collect_every_n_steps"] == config.gc_collect_every_n_steps


@pytest.mark.parametrize(
    ("config_type", "default_name", "specific_field"),
    [
        (EnergonCookerConfig, "generic_conversation", "has_subflavors"),
        (EnergonTaskEncoderConfig, "generic_sft", "packing"),
    ],
)
def test_component_defaults_shorthand_and_options_are_preserved(
    config_type: type[EnergonCookerConfig | EnergonTaskEncoderConfig],
    default_name: str,
    specific_field: str,
) -> None:
    config = config_type()
    expected = {
        "name": default_name,
        "python_file": None,
        "object_name": None,
        "options": {},
        specific_field: None,
    }

    assert config.model_dump(mode="json") == expected
    assert config_type.model_validate(default_name).model_dump(mode="json") == expected
    config.options["custom"] = True
    assert config_type().options == {}


@pytest.mark.parametrize(
    ("config_type", "label"),
    [(EnergonCookerConfig, "Cooker"), (EnergonTaskEncoderConfig, "Task encoder")],
)
@pytest.mark.parametrize("case", ["mixed", "missing_file", "missing_object"])
def test_component_selection_preserves_error_messages(
    config_type: type[EnergonCookerConfig | EnergonTaskEncoderConfig],
    label: str,
    case: str,
) -> None:
    if case == "mixed":
        selection = {
            "name": "registered",
            "python_file": "/workspace/plugins.py",
            "object_name": "custom_component",
        }
        expected = f"{label} must use either name or python_file and object_name."
    else:
        selection = {"name": None}
        if case == "missing_file":
            selection["object_name"] = "custom_component"
        else:
            selection["python_file"] = "/workspace/plugins.py"
        expected = f"File-backed {label.lower()}s require python_file and object_name."

    with pytest.raises(ValidationError) as error:
        config_type.model_validate(selection)

    assert error.value.errors()[0]["msg"] == f"Value error, {expected}"
