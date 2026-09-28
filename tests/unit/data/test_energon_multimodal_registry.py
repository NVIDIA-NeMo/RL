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

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nemo_rl.data.energon.multimodal.registry import (
    COOKER_REGISTRY,
    TASK_ENCODER_REGISTRY,
    LazyRegistry,
    selected_registry_identity,
)


def test_builtin_registries_resolve_lazily_with_stable_versions():
    assert COOKER_REGISTRY.identity("generic_conversation") == {
        "key": "generic_conversation",
        "version": "1",
    }
    assert COOKER_REGISTRY.identity("nemotron_conversation") == {
        "key": "nemotron_conversation",
        "version": "1",
    }
    assert TASK_ENCODER_REGISTRY.identity("generic_sft") == {
        "key": "generic_sft",
        "version": "1",
    }
    assert selected_registry_identity(
        task_encoder=SimpleNamespace(name="generic_sft", python_file=None, object=None),
        cookers=[
            SimpleNamespace(name="generic_conversation", python_file=None, object=None)
        ],
    ) == {
        "task_encoder": {"key": "generic_sft", "version": "1"},
        "cookers": [{"key": "generic_conversation", "version": "1"}],
    }


def test_registry_rejects_duplicate_unknown_and_malformed_keys():
    registry = LazyRegistry("cooker")
    registry.register("one", import_path="module:function", version="1")

    with pytest.raises(ValueError, match="Duplicate cooker"):
        registry.register("one", import_path="other:function", version="2")
    with pytest.raises(ValueError, match="Unknown cooker"):
        registry.resolve("missing")
    with pytest.raises(ValueError, match="expected module:name"):
        registry.register("bad", import_path="module.function", version="1")


# Validating a task encoder imports task_encoders.base, which imports
# megatron.energon at module scope, and that ships only in the `mcore` extra.
# Everything else in this file stays importable without it -- the property
# LazyRegistry exists to provide -- so mark this test alone rather than the
# module.
@pytest.mark.mcore
def test_registry_rejects_invalid_resolved_type(monkeypatch):
    module = ModuleType("_test_energon_registry_module")
    module.invalid = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LazyRegistry("task_encoder")
    registry.register(
        "invalid",
        import_path=f"{module.__name__}:invalid",
        version="1",
    )

    with pytest.raises(TypeError, match="BaseSFTTaskEncoder subclass"):
        registry.resolve("invalid")


def test_registry_does_not_import_component_during_registration(tmp_path, monkeypatch):
    # sys.modules is the right instrument: importlib.import_module never routes
    # the requested module through builtins.__import__, so an eager register()
    # using the same idiom as resolve() would slip past an __import__ sentinel.
    # delitem keeps the precondition true if anything else imports the probe.
    monkeypatch.delitem(sys.modules, "lazy_probe_mod", raising=False)
    monkeypatch.syspath_prepend(tmp_path)
    (tmp_path / "lazy_probe_mod.py").write_text("def probe():\n    return 1\n")

    registry = LazyRegistry("cooker")
    registry.register("probe", import_path="lazy_probe_mod:probe", version="1")

    assert "lazy_probe_mod" not in sys.modules
    assert registry.resolve("probe")() == 1
    assert "lazy_probe_mod" in sys.modules
    imported: list[str] = []
    original = __import__

    def record_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", record_import)
    registry = LazyRegistry("cooker")
    registry.register("json_loads", import_path="json:loads", version="1")

    assert "json" not in imported
    assert registry.resolve("json_loads") is not None


def test_registry_resolves_and_fingerprints_file_backed_component(tmp_path):
    component_file = tmp_path / "custom_cooker.py"
    component_file.write_text(
        "from nemo_rl.data.energon.multimodal.model_families import "
        "supports_model_families\n\n"
        "@supports_model_families('qwen')\n"
        "def custom_cooker(sample, *, prefix=''):\n"
        "    return prefix + sample\n"
    )
    registry = LazyRegistry("cooker")

    resolved = registry.resolve_configured_for_model_family(
        name=None,
        python_file=str(component_file),
        object_name="custom_cooker",
        model_family="qwen",
    )
    identity = registry.configured_identity(
        name=None,
        python_file=str(component_file),
        object_name="custom_cooker",
    )

    assert resolved("sample", prefix="custom-") == "custom-sample"
    assert identity["python_file"] == str(component_file)
    assert identity["object"] == "custom_cooker"
    assert len(identity["sha256"]) == 64

    component_file.write_text(component_file.read_text() + "\n# changed\n")
    changed_identity = registry.configured_identity(
        name=None,
        python_file=str(component_file),
        object_name="custom_cooker",
    )
    assert changed_identity["sha256"] != identity["sha256"]


def test_registry_resolves_and_fingerprints_file_backed_package(tmp_path):
    package = tmp_path / "custom_components"
    package.mkdir()
    init_file = package / "__init__.py"
    init_file.write_text("from .cookers import custom_cooker\n")
    cooker_file = package / "cookers.py"
    cooker_file.write_text(
        "from nemo_rl.data.energon.multimodal.model_families import "
        "supports_model_families\n\n"
        "@supports_model_families('qwen')\n"
        "def custom_cooker(sample):\n"
        "    return sample\n"
    )
    registry = LazyRegistry("cooker")

    resolved = registry.resolve_configured_for_model_family(
        name=None,
        python_file=str(init_file),
        object_name="custom_cooker",
        model_family="qwen",
    )
    identity = registry.configured_identity(
        name=None,
        python_file=str(init_file),
        object_name="custom_cooker",
    )

    assert resolved("sample") == "sample"
    cooker_file.write_text(cooker_file.read_text() + "\n# changed\n")
    changed_identity = registry.configured_identity(
        name=None,
        python_file=str(init_file),
        object_name="custom_cooker",
    )
    assert changed_identity["sha256"] != identity["sha256"]


def test_registry_rejects_invalid_file_backed_component_references(tmp_path):
    registry = LazyRegistry("cooker")
    component_file = tmp_path / "custom_cooker.py"
    component_file.write_text("def custom_cooker(sample):\n    return sample\n")

    with pytest.raises(ValueError, match="must be absolute"):
        registry.resolve_configured(
            name=None,
            python_file="custom_cooker.py",
            object_name="custom_cooker",
        )
    with pytest.raises(TypeError, match="does not resolve"):
        registry.resolve_configured(
            name=None,
            python_file=str(component_file),
            object_name="missing",
        )


@pytest.mark.mcore
def test_registry_resolves_file_backed_task_encoder(tmp_path):
    component_file = tmp_path / "custom_task_encoder.py"
    component_file.write_text(
        "from nemo_rl.data.energon.multimodal.model_families import "
        "supports_model_families\n"
        "from nemo_rl.data.energon.multimodal.task_encoders.base import "
        "BaseSFTTaskEncoder\n\n"
        "@supports_model_families('qwen')\n"
        "class CustomTaskEncoder(BaseSFTTaskEncoder):\n"
        "    def preencode_sample(self, sample):\n"
        "        return sample\n\n"
        "    def postencode_sample(self, sample):\n"
        "        return sample\n\n"
        "    def batch(self, samples):\n"
        "        return samples\n\n"
        "    def encode_batch(self, batch):\n"
        "        return batch\n"
    )

    resolved = LazyRegistry("task_encoder").resolve_configured_for_model_family(
        name=None,
        python_file=str(component_file),
        object_name="CustomTaskEncoder",
        model_family="qwen",
    )

    assert resolved.__name__ == "CustomTaskEncoder"
