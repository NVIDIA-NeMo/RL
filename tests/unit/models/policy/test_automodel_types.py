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

from typing import get_type_hints

import pytest

# Check if nemo_automodel is available for tests that need it
try:
    from nemo_automodel.components.moe.utils import BackendConfig  # noqa: F401

    NEMO_AUTOMODEL_AVAILABLE = True
except ImportError:
    NEMO_AUTOMODEL_AVAILABLE = False

from nemo_rl.models.policy import AutomodelBackendConfig


def get_typeddict_keys(typed_dict_class):
    """Get all keys from a TypedDict class."""
    return set(get_type_hints(typed_dict_class).keys())


@pytest.mark.automodel
class TestAutoModelBackendConfig:
    def test_backend_config_keys_are_defined(self):
        keys = get_typeddict_keys(AutomodelBackendConfig)
        assert len(keys) > 0, "AutoModelBackendConfig should have keys defined"
        assert "_target_" in keys, "_target_ should be a key in AutoModelBackendConfig"

    @pytest.mark.skipif(
        not NEMO_AUTOMODEL_AVAILABLE, reason="nemo_automodel not available"
    )
    def test_instantiate_backend_config_from_nemo_automodel(self):
        keys = get_typeddict_keys(AutomodelBackendConfig)
        backend_keys = {k for k in keys if k != "_target_"}

        config_dict: AutomodelBackendConfig = {
            "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
            "attn": "te",
            "linear": "te",
            "rms_norm": "te",
            "enable_deepep": True,
            "fake_balanced_gate": False,
            "enable_hf_state_dict_adapter": True,
            "enable_fsdp_optimizations": False,
            "gate_precision": "float64",
        }

        # Remove _target_ as it's a Hydra convention, not a BackendConfig param
        backend_kwargs = {k: v for k, v in config_dict.items() if k != "_target_"}

        # Instantiate the actual BackendConfig
        backend = BackendConfig(**backend_kwargs)

        # Verify each key from TypedDict is accessible on the backend
        for key in backend_keys:
            assert hasattr(backend, key), (
                f"BackendConfig missing attribute '{key}' defined in TypedDict"
            )
