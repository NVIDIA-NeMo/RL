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

import importlib
import sys

import pytest

from nemo_rl.models.generation import registry as registry_mod
from nemo_rl.models.generation.constants import SGLANG_BACKEND, VLLM_BACKEND


def test_builtin_registration_does_not_import_sglang():
    """Register built-in factories without importing every backend module."""
    sys.modules.pop("nemo_rl.models.generation.sglang", None)
    registry = importlib.reload(registry_mod)

    assert set(registry.get_registered_backends()) == {VLLM_BACKEND, SGLANG_BACKEND}
    assert "nemo_rl.models.generation.sglang" not in sys.modules


def test_custom_generation_backend_factory_is_used():
    registry = importlib.reload(registry_mod)
    created = object()

    def custom_factory(cluster, config):
        assert cluster == "cluster"
        assert config == {"name": "custom"}
        return created

    registry.register_generation_backend("custom", custom_factory)

    assert (
        registry.create_generation(
            "custom",
            cluster="cluster",
            config={"name": "custom"},
        )
        is created
    )


def test_unknown_generation_backend_lists_registered_backends():
    registry = importlib.reload(registry_mod)

    with pytest.raises(ValueError, match="Registered backends"):
        registry.create_generation("missing", cluster=None, config={})
