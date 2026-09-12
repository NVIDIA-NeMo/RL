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
from types import ModuleType

import pytest

# generic_sft and sft_dataloader import megatron.energon at module scope, and it
# ships only in the `mcore` extra. importorskip must run before those imports:
# the mcore mark is applied in pytest_collection_modifyitems, too late to
# prevent a collection error.
pytest.importorskip("megatron.energon")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.multimodal.model_families import (  # noqa: E402
    ALL_MODEL_FAMILIES,
    get_supported_model_families,
    supports_model_families,
    supports_model_family,
)
from nemo_rl.data.energon.multimodal.registry import (  # noqa: E402
    COOKER_REGISTRY,
    LazyRegistry,
)
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (  # noqa: E402
    GenericSFTTaskEncoder,
)
from nemo_rl.data.energon.sft_dataloader import _loader_config  # noqa: E402


def test_decorator_stores_immutable_model_family_metadata() -> None:
    @supports_model_families("qwen", "nemotron")
    def component() -> None:
        pass

    supported = get_supported_model_families(component)

    assert supported == frozenset({"qwen", "nemotron"})
    assert supports_model_family(component, "qwen")
    assert supports_model_family(component, "nemotron")


def test_all_model_families_marker_supports_each_known_family() -> None:
    @supports_model_families(ALL_MODEL_FAMILIES)
    class Component:
        pass

    assert supports_model_family(Component, "qwen")
    assert supports_model_family(Component, "nemotron")


def test_builtin_generic_components_support_all_model_families() -> None:
    cooker = COOKER_REGISTRY.resolve_for_model_family(
        "generic_conversation",
        model_family="qwen",
    )

    assert get_supported_model_families(cooker) == frozenset({ALL_MODEL_FAMILIES})
    assert get_supported_model_families(GenericSFTTaskEncoder) == frozenset(
        {ALL_MODEL_FAMILIES}
    )


def test_model_family_metadata_rejects_missing_and_invalid_declarations() -> None:
    with pytest.raises(TypeError, match="no supported model-family declaration"):
        get_supported_model_families(lambda: None)
    with pytest.raises(ValueError, match="At least one"):
        supports_model_families()
    with pytest.raises(ValueError, match="Unknown model families"):
        supports_model_families("unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be combined"):
        supports_model_families(ALL_MODEL_FAMILIES, "qwen")


def test_registry_rejects_unsupported_and_undeclared_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("_test_energon_model_family_module")

    @supports_model_families("qwen")
    def qwen_only() -> None:
        pass

    module.qwen_only = qwen_only
    module.undeclared = lambda: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LazyRegistry("cooker")
    registry.register(
        "qwen_only",
        import_path=f"{module.__name__}:qwen_only",
        version="1",
    )
    registry.register(
        "undeclared",
        import_path=f"{module.__name__}:undeclared",
        version="1",
    )

    assert (
        registry.resolve_for_model_family("qwen_only", model_family="qwen") is qwen_only
    )
    with pytest.raises(
        ValueError,
        match="Cooker registry key 'qwen_only'.*model family 'nemotron'.*qwen",
    ):
        registry.resolve_for_model_family("qwen_only", model_family="nemotron")
    with pytest.raises(TypeError, match="registry key 'undeclared'.*must declare"):
        registry.resolve_for_model_family("undeclared", model_family="qwen")


def test_loader_setup_rejects_an_unsupported_cooker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("_test_energon_loader_model_family_module")

    @supports_model_families("qwen")
    def qwen_only() -> None:
        pass

    module.qwen_only = qwen_only
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = LazyRegistry("cooker")
    registry.register(
        "qwen_only",
        import_path=f"{module.__name__}:qwen_only",
        version="1",
    )
    monkeypatch.setattr(
        COOKER_REGISTRY,
        "resolve_for_model_family",
        registry.resolve_for_model_family,
    )

    with pytest.raises(
        ValueError,
        match="Cooker registry key 'qwen_only'.*model family 'nemotron'.*qwen",
    ):
        _loader_config(
            {
                "model_family": "nemotron",
                "cookers": ["qwen_only"],
            }
        )


def test_undecorated_subclass_does_not_inherit_model_family_metadata() -> None:
    @supports_model_families("qwen")
    class QwenComponent:
        pass

    class UndeclaredSubclass(QwenComponent):
        pass

    with pytest.raises(TypeError, match="no supported model-family declaration"):
        get_supported_model_families(UndeclaredSubclass)
